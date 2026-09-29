// SPDX-License-Identifier: Apache-2.0

//! The document handle and every value derived from it.

use std::cell::RefCell;
use std::ops::Deref;

use loro::LoroDoc;

/// One real snapshot export: the operation count and the encoded size at it.
#[derive(Clone, Copy)]
struct Measurement {
    ops: usize,
    bytes: usize,
}

/// An affine size model fitted to real snapshot exports.
///
/// `len_ops` counts only the operations above the shallow root. For a
/// document that holds its full history that is every operation, and encoded
/// size is close to proportional to it. For a shallow document — compacted
/// in place, or reopened from a compacted snapshot — it is not: the count
/// starts near zero while the encoded size already carries the whole state.
/// A single bytes-per-operation ratio then charges each new operation the
/// average cost of the entire document, which put the estimate near twice the
/// real size after a handful of small writes.
///
/// So the model is `last.bytes + (ops - last.ops) * marginal`, where the
/// marginal cost per operation is the growth from `base` to `last`. For a
/// full-history document `base` is the empty document, which reduces this to
/// the proportional ratio. For a shallow document `base` is its first
/// measurement, so the state it started with is a fixed cost and only the
/// operations written since are priced per operation.
struct Calibration {
    base: Measurement,
    last: Measurement,
    /// Whether the document was shallow when `base` was chosen. An import can
    /// turn an empty document into a shallow one, which moves the base.
    shallow: bool,
}

impl Calibration {
    fn new(measured: Measurement, shallow: bool) -> Self {
        let base = if shallow {
            measured
        } else {
            Measurement { ops: 0, bytes: 0 }
        };
        Self {
            base,
            last: measured,
            shallow,
        }
    }

    /// Operations added since the base, or `None` below it.
    fn grown(&self, ops: usize) -> Option<usize> {
        ops.checked_sub(self.base.ops)
    }

    /// Encoded size implied by `ops`, extrapolating from the last measurement
    /// at the marginal cost per operation measured since the base. Computed in
    /// `i128` because a large document's `bytes * ops` overflows `usize` on
    /// 32-bit targets long before either factor does.
    fn estimate(&self, ops: usize) -> usize {
        let span = (self.last.ops - self.base.ops) as i128;
        // Encoded size can shrink as operations arrive, since deletes and
        // overwrites drop state. A negative marginal is not a cost to project.
        let growth = self.last.bytes.saturating_sub(self.base.bytes) as i128;
        let delta = ops as i128 - self.last.ops as i128;
        let estimate = self.last.bytes as i128 + delta.saturating_mul(growth) / span;
        usize::try_from(estimate.max(0)).unwrap_or(usize::MAX)
    }

    /// Whether `ops` is close enough to the last measurement to extrapolate
    /// from it. Encoded density is stable within a document — what changes it
    /// is a different *kind* of content, which arrives gradually — so the
    /// marginal is re-measured once the operations added since the base have
    /// halved or doubled.
    ///
    /// Bounding recalibration to a doubling makes the export cost amortise to
    /// O(1) per write, rather than being paid on every write. Counting from
    /// the base rather than from zero keeps that true for a shallow document,
    /// whose count is small whatever its size.
    fn covers(&self, ops: usize) -> bool {
        let span = self.last.ops - self.base.ops;
        let Some(grown) = self.grown(ops) else {
            return false;
        };
        span > 0 && grown > 0 && grown <= span.saturating_mul(2) && grown.saturating_mul(2) >= span
    }
}

/// A `LoroDoc` together with the values cached from it.
///
/// Compaction does not mutate a document, it replaces one: `compact_history`
/// and `compact_at_version` build a fresh doc from a shallow snapshot and swap
/// it in. Anything derived from the old doc and stored beside it survives that
/// swap and goes on describing a document that no longer exists.
///
/// The size estimate is the case that bites. A shallow snapshot preserves the
/// version vector — peers have to keep delta-syncing across a compaction — so
/// a cache keyed on the version alone still looks current after the bytes it
/// measured are gone. A caller polling the estimate to decide when to compact
/// would then never observe its own compaction landing, and would compact
/// again, and again.
///
/// Keeping the cache *inside* the cell removes the possibility: `replace` is
/// the only way to swap the document, and it drops the derived state with it.
/// Reads go through `Deref`, so every `self.doc.…` call site is untouched.
pub(in crate::state) struct DocumentCell {
    doc: LoroDoc,
    /// Size model fitted to real measurements of encoded size. `None` until
    /// the estimate is first asked for.
    calibration: RefCell<Option<Calibration>>,
    /// Real snapshot exports performed to answer `estimated_bytes`. The point
    /// of the calibration is that this grows logarithmically with the number
    /// of writes, not linearly, which is a property worth asserting.
    #[cfg(test)]
    exports: std::cell::Cell<usize>,
}

impl DocumentCell {
    /// Wrap a document with an empty derived-value cache.
    pub(in crate::state) fn new(doc: LoroDoc) -> Self {
        Self {
            doc,
            calibration: RefCell::new(None),
            #[cfg(test)]
            exports: std::cell::Cell::new(0),
        }
    }

    /// Swap in a different document, discarding everything cached from the
    /// previous one. The only way to reassign the document.
    pub(in crate::state) fn replace(&mut self, doc: LoroDoc) {
        *self = Self::new(doc);
    }

    /// Estimated encoded size in bytes, as a proxy for memory footprint.
    ///
    /// Loro exposes no direct memory metric, and a snapshot export — the
    /// honest proxy — costs O(document). Callers put this on the write path
    /// (a memory governor updated after every operation), so paying a full
    /// re-encode per call means every write re-serialises the whole document:
    /// ~100 ms per write on a 4 MB document, and proportionally worse above
    /// that.
    ///
    /// So the export is used to *calibrate* rather than to answer. `len_ops`
    /// is an inlined oplog counter, and it counts operations that are still in
    /// an open transaction, so it tracks writes the moment they happen. The
    /// answer extrapolates from the last measurement by that counter at the
    /// measured cost per operation (see [`Calibration`]), and a real export
    /// runs only when the count leaves the calibrated range.
    ///
    /// Exact whenever the document has not changed since it was measured;
    /// an interpolation otherwise, which is what a pressure signal needs.
    pub(in crate::state) fn estimated_bytes(&self) -> usize {
        let ops = self.doc.len_ops();
        if let Some(calibration) = self.calibration.borrow().as_ref() {
            if ops == calibration.last.ops {
                return calibration.last.bytes;
            }
            if calibration.covers(ops) {
                return calibration.estimate(ops);
            }
        }

        #[cfg(test)]
        self.exports.set(self.exports.get() + 1);
        let Ok(snapshot) = self.doc.export(loro::ExportMode::Snapshot) else {
            // A failed export is not a measurement. Caching the zero would pin
            // the document at "empty" until enough writes moved it out of
            // range again.
            return 0;
        };
        let measured = Measurement {
            ops,
            bytes: snapshot.len(),
        };
        let shallow = self.doc.is_shallow();
        let mut calibration = self.calibration.borrow_mut();
        match calibration.as_mut() {
            Some(current) if current.shallow == shallow && current.grown(ops).is_some() => {
                current.last = measured;
            }
            _ => *calibration = Some(Calibration::new(measured, shallow)),
        }
        measured.bytes
    }

    /// How many real snapshot exports `estimated_bytes` has performed.
    #[cfg(test)]
    pub(in crate::state) fn export_count(&self) -> usize {
        self.exports.get()
    }
}

impl Deref for DocumentCell {
    type Target = LoroDoc;

    fn deref(&self) -> &Self::Target {
        &self.doc
    }
}
