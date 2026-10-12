// SPDX-License-Identifier: Apache-2.0

//! Memtable state of one index: its postings and its corpus stats.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, RandomState};

use hashbrown::HashTable;
use nodedb_types::Surrogate;

use crate::block::CompactPosting;
use crate::codec::smallfloat;

/// One term of an index and its postings. A freed slot holds an empty text
/// and no postings until a new term reuses it.
#[derive(Default)]
struct Term {
    text: Box<str>,
    postings: Vec<CompactPosting>,
}

/// Postings, stats, and fieldnorms of one `(database, tenant, index)`.
///
/// Terms are stored once, in `terms`, and found through `lookup`. Each
/// document keeps the ids of the terms it has postings under, so removing a
/// document touches its own terms instead of every term of the index.
#[derive(Default)]
pub(crate) struct ScopeState {
    /// Term id → term text and postings.
    terms: Vec<Term>,
    /// Ids of freed `terms` slots.
    free: Vec<u32>,
    /// Term ids, hashed by the text their slot holds.
    lookup: HashTable<u32>,
    hasher: RandomState,
    /// Document → ids of the terms holding a posting of it.
    doc_terms: HashMap<Surrogate, Vec<u32>>,
    /// Posting entries across every term of this index.
    pub(super) total_postings: usize,
    /// Incremental stats: (doc_count, total_token_sum).
    pub(super) stats: (u32, u64),
    /// Fieldnorm array: surrogate.0 → SmallFloat-encoded length (used by BM25).
    pub(super) fieldnorms: Vec<u8>,
    /// Exact-length sidecar: surrogate.0 → original u32 token count. Used to
    /// decrement `stats.total_token_sum` symmetrically with the exact
    /// increment in `record_doc`. Storing the original length here keeps
    /// the smallfloat array lossy (space-efficient for ranking) while
    /// making corpus-stat accounting exact.
    pub(super) fieldnorms_exact: Vec<u32>,
    /// Bytes of resident postings, term keys and document term lists, as
    /// [`posting_bytes`], [`term_bytes`] and [`doc_bytes`] count them.
    pub(super) bytes: usize,
}

/// Postings and terms a mutation removed from one index.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Removed {
    pub(crate) postings: usize,
    pub(crate) terms: usize,
    /// Resident bytes the removed postings and terms held.
    pub(crate) bytes: usize,
}

/// Bytes one posting costs. Positions are the only heap allocation per
/// posting, so they are counted. The struct itself is counted by size, and
/// so is the term id its document's term list holds for it.
// ponytail: HashMap slack and allocator overhead are not counted. Count them if
// the governor still reads low against measured heap.
pub(crate) fn posting_bytes(posting: &CompactPosting) -> usize {
    std::mem::size_of::<CompactPosting>()
        + posting.positions.len() * std::mem::size_of::<u32>()
        + std::mem::size_of::<u32>()
}

/// Bytes one term key costs: the key text, its slot and its lookup entry.
// ponytail: a freed slot keeps its struct bytes until a new term reuses it.
pub(crate) fn term_bytes(term: &str) -> usize {
    term.len() + std::mem::size_of::<Term>() + std::mem::size_of::<u32>()
}

/// Bytes one document's term-list entry costs, apart from its term ids.
fn doc_bytes() -> usize {
    std::mem::size_of::<(Surrogate, Vec<u32>)>()
}

impl ScopeState {
    fn find(&self, term: &str) -> Option<u32> {
        let hash = self.hasher.hash_one(term);
        self.lookup
            .find(hash, |&id| &*self.terms[id as usize].text == term)
            .copied()
    }

    /// Insert a posting for a term. Returns whether the term is new here.
    pub(super) fn insert(&mut self, term: &str, posting: CompactPosting) -> bool {
        self.total_postings += 1;
        self.bytes += posting_bytes(&posting);
        let doc_id = posting.doc_id;
        let (id, new_term) = match self.find(term) {
            Some(id) => (id, false),
            None => {
                self.bytes += term_bytes(term);
                let slot = Term {
                    text: term.into(),
                    postings: Vec::new(),
                };
                let id = match self.free.pop() {
                    Some(id) => {
                        self.terms[id as usize] = slot;
                        id
                    }
                    None => {
                        self.terms.push(slot);
                        u32::try_from(self.terms.len() - 1).expect("term ids fit in u32")
                    }
                };
                let (terms, hasher) = (&self.terms, &self.hasher);
                self.lookup.insert_unique(hasher.hash_one(term), id, |&id| {
                    hasher.hash_one(&*terms[id as usize].text)
                });
                (id, true)
            }
        };
        self.terms[id as usize].postings.push(posting);
        match self.doc_terms.entry(doc_id) {
            Entry::Occupied(entry) => entry.into_mut().push(id),
            Entry::Vacant(entry) => {
                self.bytes += doc_bytes();
                entry.insert(vec![id]);
            }
        }
        new_term
    }

    /// Record a document's stats (call once per indexed document).
    pub(super) fn record_doc(&mut self, doc_id: Surrogate, doc_len: u32) {
        self.stats.0 += 1;
        self.stats.1 += doc_len as u64;

        let idx = doc_id.0 as usize;
        if idx >= self.fieldnorms.len() {
            self.fieldnorms.resize(idx + 1, 0);
        }
        self.fieldnorms[idx] = smallfloat::encode(doc_len);
        if idx >= self.fieldnorms_exact.len() {
            self.fieldnorms_exact.resize(idx + 1, 0);
        }
        self.fieldnorms_exact[idx] = doc_len;
    }

    /// Remove a document's postings from the terms it is indexed under.
    #[cfg(test)]
    pub(super) fn remove_doc(&mut self, doc_id: Surrogate) -> Removed {
        self.remove_docs(&[doc_id])
    }

    /// Remove several documents' postings, visiting each touched term once.
    ///
    /// A term shared by most documents holds a posting list as long as the
    /// index, so removing documents one at a time scans that list once per
    /// document.
    pub(super) fn remove_docs(&mut self, docs: &[Surrogate]) -> Removed {
        match docs {
            [doc] => self.remove_matching(docs, |d| d == *doc),
            _ => {
                let set: HashSet<Surrogate> = docs.iter().copied().collect();
                self.remove_matching(docs, |d| set.contains(&d))
            }
        }
    }

    /// Remove the postings of `docs`. `removing` answers whether a posting's
    /// document is one of them.
    fn remove_matching(
        &mut self,
        docs: &[Surrogate],
        removing: impl Fn(Surrogate) -> bool,
    ) -> Removed {
        let mut removed = Removed::default();
        let mut touched = Vec::new();
        for doc_id in docs {
            if let Some(ids) = self.doc_terms.remove(doc_id) {
                removed.bytes += doc_bytes();
                touched.extend(ids);
            }
        }
        touched.sort_unstable();
        touched.dedup();
        for id in touched {
            let slot = &mut self.terms[id as usize];
            let before = slot.postings.len();
            slot.postings.retain(|p| {
                if removing(p.doc_id) {
                    removed.bytes += posting_bytes(p);
                    false
                } else {
                    true
                }
            });
            removed.postings += before - slot.postings.len();
            if slot.postings.is_empty() {
                removed.terms += 1;
                removed.bytes += term_bytes(&slot.text);
                let freed = std::mem::take(slot);
                let hash = self.hasher.hash_one(&*freed.text);
                if let Ok(entry) = self.lookup.find_entry(hash, |&x| x == id) {
                    entry.remove();
                }
                self.free.push(id);
            } else if slot.postings.len() < before
                && slot.postings.capacity() > slot.postings.len().saturating_mul(2).max(4)
            {
                slot.postings.shrink_to(slot.postings.len());
            }
        }
        self.total_postings -= removed.postings;
        self.bytes = self.bytes.saturating_sub(removed.bytes);

        // Decrement stats using the exact-length sidecar so the subtraction
        // matches the exact-length increment in record_doc. Using the
        // smallfloat-decoded length here would drift upward on churn.
        for doc_id in docs {
            if let Some(slot) = self.fieldnorms_exact.get_mut(doc_id.0 as usize)
                && *slot > 0
            {
                let doc_len = *slot;
                *slot = 0;
                self.stats.0 = self.stats.0.saturating_sub(1);
                self.stats.1 = self.stats.1.saturating_sub(doc_len as u64);
            }
        }
        removed
    }

    /// The postings of one term, empty when the term is not held.
    pub(super) fn postings(&self, term: &str) -> &[CompactPosting] {
        self.find(term)
            .map_or(&[], |id| self.terms[id as usize].postings.as_slice())
    }

    /// Every held term with its postings, in no particular order.
    pub(super) fn iter(&self) -> impl Iterator<Item = (&str, &[CompactPosting])> {
        self.terms
            .iter()
            .filter(|t| !t.postings.is_empty())
            .map(|t| (&*t.text, t.postings.as_slice()))
    }

    /// `(len, capacity)` of every held term's posting list.
    #[cfg(test)]
    pub(super) fn posting_capacities(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.terms
            .iter()
            .filter(|t| !t.postings.is_empty())
            .map(|t| (t.postings.len(), t.postings.capacity()))
    }

    /// The term → postings map, consuming the state.
    pub(super) fn into_postings(self) -> HashMap<String, Vec<CompactPosting>> {
        self.terms
            .into_iter()
            .filter(|t| !t.postings.is_empty())
            .map(|t| (t.text.into_string(), t.postings))
            .collect()
    }

    /// Whether the index holds neither postings nor counted documents.
    pub(super) fn is_vacant(&self) -> bool {
        self.lookup.is_empty() && self.stats.0 == 0
    }

    /// What dropping this whole state removes.
    pub(super) fn footprint(&self) -> Removed {
        Removed {
            postings: self.total_postings,
            terms: self.lookup.len(),
            bytes: self.bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn posting(doc: u32) -> CompactPosting {
        CompactPosting {
            doc_id: Surrogate(doc),
            term_freq: 1,
            fieldnorm: 0,
            positions: vec![0],
        }
    }

    fn index(state: &mut ScopeState, doc: u32, terms: &[&str]) {
        for term in terms {
            state.insert(term, posting(doc));
        }
        state.record_doc(Surrogate(doc), terms.len() as u32);
    }

    #[test]
    fn removing_a_doc_leaves_the_state_of_never_having_indexed_it() {
        let mut state = ScopeState::default();
        index(&mut state, 1, &["alpha", "beta", "gamma"]);
        index(&mut state, 2, &["beta", "delta"]);
        index(&mut state, 3, &["gamma", "delta", "omega"]);

        let removed = state.remove_doc(Surrogate(2));

        let mut fresh = ScopeState::default();
        index(&mut fresh, 1, &["alpha", "beta", "gamma"]);
        index(&mut fresh, 3, &["gamma", "delta", "omega"]);
        assert_eq!(removed.postings, 2);
        assert_eq!(
            removed.terms, 0,
            "beta and delta are still held by docs 1 and 3"
        );
        assert_eq!(state.total_postings, fresh.total_postings);
        assert_eq!(state.stats, fresh.stats);
        assert_eq!(state.bytes, fresh.bytes);
        assert_eq!(state.footprint(), fresh.footprint());
        for term in ["alpha", "beta", "gamma", "delta", "omega"] {
            let docs: Vec<u32> = state.postings(term).iter().map(|p| p.doc_id.0).collect();
            let want: Vec<u32> = fresh.postings(term).iter().map(|p| p.doc_id.0).collect();
            assert_eq!(docs, want, "{term}");
        }

        state.remove_doc(Surrogate(1));
        state.remove_doc(Surrogate(3));
        assert!(state.is_vacant());
        assert_eq!(state.bytes, 0);
        assert_eq!(state.footprint(), Removed::default());

        // A freed term comes back as a new term.
        assert!(state.insert("beta", posting(4)));
        assert_eq!(state.postings("beta").len(), 1);
    }

    #[test]
    fn removing_a_small_doc_does_not_scan_a_large_vocabulary() {
        let mut state = ScopeState::default();
        let vocab: Vec<String> = (0..100_000).map(|i| format!("t{i}")).collect();
        for term in &vocab {
            state.insert(term, posting(1));
        }
        let started = std::time::Instant::now();
        for _ in 0..1_000 {
            index(&mut state, 2, &["t1", "t2", "fresh"]);
            state.remove_doc(Surrogate(2));
        }
        // A full scan per removal visits 10^8 postings here. Touching only the
        // document's own terms takes milliseconds.
        let elapsed = started.elapsed();
        assert!(elapsed.as_secs() < 2, "1,000 removals took {elapsed:?}");
        assert_eq!(state.total_postings, 100_000);
    }

    #[test]
    fn removing_docs_together_matches_removing_them_one_at_a_time() {
        let build = || {
            let mut state = ScopeState::default();
            index(&mut state, 1, &["common", "alpha"]);
            index(&mut state, 2, &["common", "beta"]);
            index(&mut state, 3, &["common", "gamma"]);
            index(&mut state, 4, &["common", "alpha", "delta"]);
            state
        };
        let mut one_by_one = build();
        one_by_one.remove_doc(Surrogate(1));
        one_by_one.remove_doc(Surrogate(4));
        let mut together = build();
        // Listed twice and listed but never indexed: both change nothing more.
        let removed =
            together.remove_docs(&[Surrogate(1), Surrogate(4), Surrogate(4), Surrogate(9)]);

        assert_eq!(removed.postings, 5);
        assert_eq!(removed.terms, 2, "alpha and delta are held by no one else");
        assert_eq!(together.total_postings, one_by_one.total_postings);
        assert_eq!(together.stats, one_by_one.stats);
        assert_eq!(together.bytes, one_by_one.bytes);
        assert_eq!(together.footprint(), one_by_one.footprint());
        for term in ["common", "alpha", "beta", "gamma", "delta"] {
            let docs: Vec<u32> = together.postings(term).iter().map(|p| p.doc_id.0).collect();
            let want: Vec<u32> = one_by_one
                .postings(term)
                .iter()
                .map(|p| p.doc_id.0)
                .collect();
            assert_eq!(docs, want, "{term}");
        }
    }

    #[test]
    fn removing_many_docs_scans_a_shared_term_once() {
        let mut state = ScopeState::default();
        for doc in 1..=50_000 {
            state.insert("shared", posting(doc));
        }
        let docs: Vec<Surrogate> = (1..=25_000).map(Surrogate).collect();
        let started = std::time::Instant::now();
        state.remove_docs(&docs);
        // One document at a time scans the shared list 25,000 times.
        let elapsed = started.elapsed();
        assert!(
            elapsed.as_secs() < 2,
            "removing 25,000 docs took {elapsed:?}"
        );
        assert_eq!(state.postings("shared").len(), 25_000);
    }
}
