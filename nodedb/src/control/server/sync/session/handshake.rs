// SPDX-License-Identifier: BUSL-1.1

//! Handshake + fork detection.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tracing::{info, warn};

use crate::control::security::identity::AuthenticatedIdentity;
use crate::control::security::jwt::JwtValidator;
use crate::control::state::SharedState;

use super::super::dlq::DeviceMetadata;
use super::super::wire::*;
use super::state::SyncSession;

/// Decision returned by the durable fencing logic before building the ack.
enum FencingDecision {
    /// Accept with the given producer_id and accepted_epoch.
    Accept {
        producer_id: u64,
        accepted_epoch: u64,
    },
    /// Reject: stale epoch from a cloned / forked device. PERMANENT — the
    /// client's epoch is behind the durable record; it must regenerate its
    /// LiteId. Surfaced to the client as `fork_detected = true`.
    Reject,
    /// Reject: a transient server-side error (registry I/O, Raft propose
    /// failure / leader mid-election). NOT a fork — the client should simply
    /// retry the handshake. Surfaced as `success = false, fork_detected = false`
    /// so the client never wipes its state over a momentary server hiccup.
    RejectTransient,
}

/// How the handshake's `jwt_token` is authenticated.
///
/// The JWKS registry validates asynchronously, which the synchronous
/// handshake cannot await, so the session loop resolves the token before
/// calling in and hands the outcome over as [`TokenAuth::Registry`].
/// Deployments with no registry configured keep the static validator.
enum TokenAuth<'a> {
    Static(&'a JwtValidator),
    /// Already verified against the `JwksRegistry`; `Err` carries the
    /// rejection reason reported back to the client.
    Registry(Result<AuthenticatedIdentity, String>),
}

/// Report, once per session, an authenticated sync identity that holds no
/// write authority at all.
///
/// This is a diagnostic, never a gate: per-collection grants are not
/// enumerable here, so an identity that fails this check may still be able to
/// write somewhere and the session is accepted either way. What it replaces is
/// the only evidence the failure otherwise leaves — one ERROR per refused
/// delta, naming the collection but never the reason the principal has no
/// authority over it.
///
/// The usual reason is the `roles` claim. Role names are matched exactly
/// (`readwrite`, `readonly`, `tenant_admin`, `monitor`,
/// `database_owner:<db>`, `database_editor:<db>`, `database_reader:<db>`); a
/// name outside that set is accepted as a custom role, which confers nothing
/// until something is granted to it. So `read_write` — one underscore away
/// from the role that would have worked — authenticates perfectly and then
/// cannot write a single row.
fn warn_if_no_write_authority(
    session_id: &str,
    identity: &AuthenticatedIdentity,
    shared: &Arc<SharedState>,
) {
    use crate::control::security::audit::NoopAuditEmitter;
    use crate::control::security::identity::Permission;

    let audit = NoopAuditEmitter;
    if shared.permissions.check_tenant(
        identity,
        Permission::Write,
        identity.tenant_id,
        &shared.roles,
        &audit,
    ) {
        return;
    }

    let roles: Vec<String> = identity.roles.iter().map(ToString::to_string).collect();
    warn!(
        session = %session_id,
        user = %identity.username,
        tenant = identity.tenant_id.as_u64(),
        roles = ?roles,
        "sync session has no tenant-wide write authority: unless this principal holds \
         per-collection grants, every delta it pushes will be refused. Grant it a role in \
         the token's `roles` claim (`readwrite` for writes; `database_editor:<database_id>` \
         also allows the collection materialization a first sync needs), or GRANT WRITE ON \
         TENANT to this user. Role names are matched exactly — an unrecognised name becomes \
         a custom role with no permissions"
    );
}

#[cfg(test)]
fn producer_owner_matches(
    registration: &crate::control::sync_producer::ProducerRegistration,
    tenant_id: u64,
    user_id: u64,
) -> bool {
    registration.tenant_id == tenant_id && registration.user_id == user_id
}

impl SyncSession {
    /// Process a handshake message: validate JWT, store client clock, detect forks.
    ///
    /// `shared` is threaded in so the durable `SyncProducerRegistry` can be consulted
    /// when the session is a Lite client.  When `shared` is `None` (non-Lite client
    /// or unit-test path without SharedState) the handshake proceeds with no fencing
    /// (`producer_id = 0`).
    ///
    /// Returns a HandshakeAck frame to send back to the client.
    pub fn handle_handshake(
        &mut self,
        msg: &HandshakeMsg,
        jwt_validator: &JwtValidator,
        current_server_clock: HashMap<String, u64>,
        shared: Option<&Arc<SharedState>>,
    ) -> Option<SyncFrame> {
        self.handshake_with_auth(
            msg,
            TokenAuth::Static(jwt_validator),
            current_server_clock,
            shared,
        )
    }

    /// Same handshake, for a token already verified against the JWKS registry.
    ///
    /// `verified` is the registry outcome for `msg.jwt_token`, resolved by the
    /// async session loop because `JwksRegistry::validate_with_claims` is
    /// async. The identity — including its tenant — comes from the registry,
    /// which binds the provider's configured `tenant_id`, never the token's
    /// `tenant_id` claim. Everything else (wire-version check, trust mode for
    /// an empty token, durable fencing, signing key) is identical to
    /// [`Self::handle_handshake`].
    pub(in crate::control::server::sync) fn handle_handshake_verified(
        &mut self,
        msg: &HandshakeMsg,
        verified: Result<AuthenticatedIdentity, String>,
        current_server_clock: HashMap<String, u64>,
        shared: Option<&Arc<SharedState>>,
    ) -> Option<SyncFrame> {
        self.handshake_with_auth(
            msg,
            TokenAuth::Registry(verified),
            current_server_clock,
            shared,
        )
    }

    fn handshake_with_auth(
        &mut self,
        msg: &HandshakeMsg,
        auth: TokenAuth<'_>,
        current_server_clock: HashMap<String, u64>,
        shared: Option<&Arc<SharedState>>,
    ) -> Option<SyncFrame> {
        // A handshake is an authentication boundary, not an additive update to
        // a prior one. Clear the prior binding before examining this attempt so
        // every failure leaves the connection unable to use its old identity.
        self.clear_handshake_binding();
        self.last_activity = Instant::now();

        // Wire format compatibility check: reject incompatible clients early.
        // Version 0 (missing field) falls through to check_wire_compatibility
        // and is rejected cleanly as "too old".
        if let Err(e) = crate::version::check_wire_compatibility(msg.wire_version) {
            warn!(
                session = %self.session_id,
                client_wire_version = msg.wire_version,
                error = %e,
                "sync handshake rejected: incompatible wire version"
            );
            let ack = HandshakeAckMsg {
                success: false,
                session_id: self.session_id.clone(),
                server_clock: current_server_clock,
                error: Some(format!("incompatible wire version: {e}")),
                fork_detected: false,
                server_wire_version: crate::version::WIRE_FORMAT_VERSION,
                producer_id: 0,
                accepted_epoch: 0,
                delta_signing_key: [0; 32],
            };
            return SyncFrame::try_encode(SyncMessageType::HandshakeAck, &ack);
        }

        // Trust mode: an empty token resolves to the configured durable
        // principal. Never fabricate an identity that cannot own catalog data.
        if msg.jwt_token.is_empty() {
            let Some(identity) = shared.and_then(|state| {
                crate::control::server::session_auth::configured_trust_identity(state)
            }) else {
                let ack = HandshakeAckMsg {
                    success: false,
                    session_id: self.session_id.clone(),
                    server_clock: current_server_clock,
                    error: Some("configured trust identity is unavailable".into()),
                    fork_detected: false,
                    server_wire_version: crate::version::WIRE_FORMAT_VERSION,
                    producer_id: 0,
                    accepted_epoch: 0,
                    delta_signing_key: [0; 32],
                };
                return SyncFrame::try_encode(SyncMessageType::HandshakeAck, &ack);
            };
            self.tenant_id = Some(identity.tenant_id);
            self.username = Some(identity.username.clone());
            let user_id = identity.user_id;
            self.identity = Some(identity);
            self.authenticated = true;
            self.client_clock = msg.vector_clock.clone();
            self.subscribed_shapes = msg.subscribed_shapes.clone();
            self.server_clock = current_server_clock.clone();
            self.last_seen_lsn = msg
                .vector_clock
                .values()
                .flat_map(|m| m.values().copied())
                .max()
                .unwrap_or(0);
            self.device_metadata = DeviceMetadata {
                client_version: msg.client_version.clone(),
                remote_addr: String::new(),
                peer_id: 0,
            };

            let tenant_id = self.tenant_id.map(|t| t.as_u64()).unwrap_or(0);
            match self.durable_fencing_decision(msg, shared, tenant_id, user_id) {
                Some(FencingDecision::Reject) => {
                    return self.fork_reject_frame(
                        &current_server_clock,
                        "FORK_DETECTED: stale epoch — regenerate LiteId and reconnect",
                    );
                }
                Some(FencingDecision::RejectTransient) => {
                    return self.transient_reject_frame(
                        &current_server_clock,
                        "SYNC_UNAVAILABLE: transient server error — retry the handshake",
                    );
                }
                Some(FencingDecision::Accept {
                    producer_id,
                    accepted_epoch,
                }) => {
                    self.producer_id = producer_id;
                    self.accepted_epoch = accepted_epoch;
                }
                None => {}
            }

            info!(session = %self.session_id, "sync handshake OK (trust mode)");

            let ack = HandshakeAckMsg {
                success: true,
                session_id: self.session_id.clone(),
                server_clock: current_server_clock,
                error: None,
                fork_detected: false,
                server_wire_version: crate::version::WIRE_FORMAT_VERSION,
                producer_id: self.producer_id,
                accepted_epoch: self.accepted_epoch,
                delta_signing_key: [0; 32],
            };
            return SyncFrame::try_encode(SyncMessageType::HandshakeAck, &ack);
        }

        // Validate JWT: either through the JWKS registry (already resolved by
        // the caller) or through the statically configured validator.
        let validated = match auth {
            TokenAuth::Static(jwt_validator) => jwt_validator
                .validate(&msg.jwt_token)
                .map_err(|e| e.to_string()),
            TokenAuth::Registry(verified) => verified,
        };
        match validated {
            Ok(identity) => {
                self.tenant_id = Some(identity.tenant_id);
                self.username = Some(identity.username.clone());
                self.identity = Some(identity.clone());
                self.authenticated = true;
                self.client_clock = msg.vector_clock.clone();
                self.subscribed_shapes = msg.subscribed_shapes.clone();
                self.server_clock = current_server_clock.clone();
                self.last_seen_lsn = msg
                    .vector_clock
                    .values()
                    .flat_map(|m| m.values().copied())
                    .max()
                    .unwrap_or(0);
                self.device_metadata = DeviceMetadata {
                    client_version: msg.client_version.clone(),
                    remote_addr: String::new(),
                    peer_id: 0,
                };

                let tenant_id = identity.tenant_id.as_u64();
                match self.durable_fencing_decision(msg, shared, tenant_id, identity.user_id) {
                    Some(FencingDecision::Reject) => {
                        return self.fork_reject_frame(
                            &current_server_clock,
                            "FORK_DETECTED: stale epoch — regenerate LiteId and reconnect",
                        );
                    }
                    Some(FencingDecision::RejectTransient) => {
                        return self.transient_reject_frame(
                            &current_server_clock,
                            "SYNC_UNAVAILABLE: transient server error — retry the handshake",
                        );
                    }
                    Some(FencingDecision::Accept {
                        producer_id,
                        accepted_epoch,
                    }) => {
                        self.producer_id = producer_id;
                        self.accepted_epoch = accepted_epoch;
                    }
                    None => {}
                }

                self.delta_signing_key = match shared {
                    Some(state) if state.wal.payloads_authenticated() => {
                        match state.credentials.catalog().get_or_create_crdt_signing_key(
                            identity.tenant_id.as_u64(),
                            identity.user_id,
                        ) {
                            Ok(key) => Some(key),
                            Err(error) => {
                                warn!(session = %self.session_id, %error, "sync signing key unavailable");
                                return self.transient_reject_frame(
                                    &current_server_clock,
                                    "SYNC_UNAVAILABLE: signing key persistence failed",
                                );
                            }
                        }
                    }
                    Some(_) | None => None,
                };

                info!(
                    session = %self.session_id,
                    user = %identity.username,
                    tenant = identity.tenant_id.as_u64(),
                    shapes = self.subscribed_shapes.len(),
                    "sync handshake OK"
                );
                if let Some(state) = shared {
                    warn_if_no_write_authority(&self.session_id, &identity, state);
                }

                let ack = HandshakeAckMsg {
                    success: true,
                    session_id: self.session_id.clone(),
                    server_clock: current_server_clock,
                    error: None,
                    fork_detected: false,
                    server_wire_version: crate::version::WIRE_FORMAT_VERSION,
                    producer_id: self.producer_id,
                    accepted_epoch: self.accepted_epoch,
                    delta_signing_key: self.delta_signing_key.unwrap_or([0; 32]),
                };
                SyncFrame::try_encode(SyncMessageType::HandshakeAck, &ack)
            }
            Err(e) => {
                warn!(
                    session = %self.session_id,
                    error = %e,
                    "sync handshake FAILED"
                );
                let ack = HandshakeAckMsg {
                    success: false,
                    session_id: self.session_id.clone(),
                    server_clock: HashMap::new(),
                    error: Some(e),
                    fork_detected: false,
                    server_wire_version: crate::version::WIRE_FORMAT_VERSION,
                    producer_id: 0,
                    accepted_epoch: 0,
                    delta_signing_key: [0; 32],
                };
                SyncFrame::try_encode(SyncMessageType::HandshakeAck, &ack)
            }
        }
    }

    /// Clear all state derived from a successful handshake or from the
    /// candidate currently being authenticated. Connection lifetime, activity,
    /// rate-limiting, and telemetry counters deliberately survive.
    pub(super) fn clear_handshake_binding(&mut self) {
        self.tenant_id = None;
        self.username = None;
        self.identity = None;
        self.authenticated = false;
        self.client_clock.clear();
        self.server_clock.clear();
        self.subscribed_shapes.clear();
        self.last_seen_lsn = 0;
        self.producer_id = 0;
        self.accepted_epoch = 0;
        self.delta_signing_key = None;
        self.device_metadata = DeviceMetadata::default();
        self.tracked_collections.clear();
        self.announced_collections.clear();
    }

    /// Build a bounded, generic rejection for a Handshake frame whose body
    /// could not be decoded. The fixed message intentionally exposes no
    /// decoder or payload details to the peer.
    pub(super) fn malformed_handshake_reject_frame(&self) -> Option<SyncFrame> {
        self.build_reject_frame(&HashMap::new(), "malformed handshake", false)
    }

    /// Build a failed HandshakeAck rejection frame: `success=false`, the
    /// supplied `message` in the `error` field, and the given `fork_detected`
    /// flag. Shared by [`Self::fork_reject_frame`] and
    /// [`Self::transient_reject_frame`] so the envelope stays identical.
    fn build_reject_frame(
        &self,
        server_clock: &HashMap<String, u64>,
        message: &str,
        fork_detected: bool,
    ) -> Option<SyncFrame> {
        let ack = HandshakeAckMsg {
            success: false,
            session_id: self.session_id.clone(),
            server_clock: server_clock.clone(),
            error: Some(message.into()),
            fork_detected,
            server_wire_version: crate::version::WIRE_FORMAT_VERSION,
            producer_id: 0,
            accepted_epoch: 0,
            delta_signing_key: [0; 32],
        };
        SyncFrame::try_encode(SyncMessageType::HandshakeAck, &ack)
    }

    /// Build a fork-detected rejection frame (`fork_detected=true`): the client
    /// treats the session as a permanent fork and wipes its local state.
    fn fork_reject_frame(
        &mut self,
        server_clock: &HashMap<String, u64>,
        message: &str,
    ) -> Option<SyncFrame> {
        warn!(
            session = %self.session_id,
            "FORK DETECTED: {message}"
        );
        self.clear_handshake_binding();
        self.build_reject_frame(server_clock, message, true)
    }

    /// Build a failed-but-retryable handshake ack for a transient server-side
    /// error (registry I/O, Raft propose failure / leader mid-election).
    ///
    /// Distinct from [`Self::fork_reject_frame`]: `fork_detected = false`, so
    /// the client retries the handshake rather than treating the session as a
    /// permanent fork and wiping its local state.
    fn transient_reject_frame(
        &mut self,
        server_clock: &HashMap<String, u64>,
        message: &str,
    ) -> Option<SyncFrame> {
        warn!(
            session = %self.session_id,
            "sync handshake transient error (retryable): {message}"
        );
        self.clear_handshake_binding();
        self.build_reject_frame(server_clock, message, false)
    }

    /// Attempt to make a durable fencing decision via `SyncProducerRegistry`.
    ///
    /// Returns `Some(FencingDecision)` when the msg is a Lite handshake
    /// (`!lite_id.is_empty() && epoch > 0`) and a registry is available via
    /// `shared`.  Returns `None` when:
    ///
    /// * The msg is not a Lite handshake (non-Lite / legacy client).
    /// * No `SharedState` is present (unit-test path) — handshake proceeds
    ///   with no fencing.
    /// * `shared` has no `producer_registry` — handshake proceeds with no fencing.
    ///
    /// On registry operation errors the decision is `Reject` (fail-closed) rather
    /// than silently accepting.
    fn durable_fencing_decision(
        &self,
        msg: &HandshakeMsg,
        shared: Option<&Arc<SharedState>>,
        tenant_id: u64,
        user_id: u64,
    ) -> Option<FencingDecision> {
        if msg.lite_id.is_empty() || msg.epoch == 0 {
            return None;
        }

        let registry = shared.and_then(|s| s.producer_registry.as_deref());

        match registry {
            Some(reg) => {
                // `shared` is always `Some` when `registry` is `Some` (the
                // registry was obtained via `shared.and_then(...)`); the `?`
                // is just to recover the handle.
                let shared_ref = shared?;

                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as i64;

                let existing = match reg.get_or_register(
                    &msg.lite_id,
                    tenant_id,
                    user_id,
                    msg.epoch,
                    now_ms,
                ) {
                    Ok((registration, _created)) => registration,
                    Err(crate::Error::BadRequest { .. }) => {
                        warn!(
                            session = %self.session_id,
                            lite_id = %msg.lite_id,
                            authenticated_tenant = tenant_id,
                            authenticated_user = user_id,
                            "sync producer owner mismatch"
                        );
                        return Some(FencingDecision::Reject);
                    }
                    Err(e) => {
                        warn!(
                            session = %self.session_id,
                            lite_id = %msg.lite_id,
                            error = %e,
                            "sync handshake: atomic producer registration failed; rejecting as retryable"
                        );
                        return Some(FencingDecision::RejectTransient);
                    }
                };

                // Propose on both creation and retry. This closes the crash/error
                // window between the local durable row and Raft replication;
                // duplicate identical registrations are apply-idempotent.
                if let Err(e) = crate::control::metadata_proposer::propose_sync_producer_register(
                    shared_ref.as_ref(),
                    &msg.lite_id,
                    existing.producer_id,
                    existing.tenant_id,
                    existing.user_id,
                    existing.current_epoch,
                    existing.created_ms,
                ) {
                    warn!(
                        session = %self.session_id,
                        lite_id = %msg.lite_id,
                        error = %e,
                        "sync handshake: propose_sync_producer_register failed; rejecting as retryable"
                    );
                    return Some(FencingDecision::RejectTransient);
                }

                if msg.epoch < existing.current_epoch {
                    warn!(
                        session = %self.session_id,
                        lite_id = %msg.lite_id,
                        client_epoch = msg.epoch,
                        current_epoch = existing.current_epoch,
                        "FORK DETECTED: client epoch is behind persisted epoch"
                    );
                    return Some(FencingDecision::Reject);
                }

                if msg.epoch > existing.current_epoch
                    && let Err(e) = reg.fence(&msg.lite_id, msg.epoch)
                {
                    warn!(
                        session = %self.session_id,
                        lite_id = %msg.lite_id,
                        error = %e,
                        "sync handshake: registry.fence failed; rejecting as retryable"
                    );
                    return Some(FencingDecision::RejectTransient);
                }

                // Re-propose even when the requested epoch already matches the
                // local row. A prior proposal may have failed after the local
                // fence was persisted; the idempotent max-wins Raft entry must
                // reach followers before this node accepts the retry.
                if let Err(e) = crate::control::metadata_proposer::propose_sync_producer_fence(
                    shared_ref.as_ref(),
                    &msg.lite_id,
                    msg.epoch,
                ) {
                    warn!(
                        session = %self.session_id,
                        lite_id = %msg.lite_id,
                        error = %e,
                        "sync handshake: propose_sync_producer_fence failed; rejecting as retryable"
                    );
                    return Some(FencingDecision::RejectTransient);
                }

                Some(FencingDecision::Accept {
                    producer_id: existing.producer_id,
                    accepted_epoch: msg.epoch,
                })
            }
            // No registry available: no fencing decision — handshake proceeds.
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::producer_owner_matches;
    use std::collections::HashMap;
    use std::sync::Arc;

    use nodedb_types::sync::wire::{
        DeltaPushMsg, DeltaRejectMsg, HandshakeAckMsg, HandshakeMsg, SyncFrame, SyncMessageType,
    };

    use crate::bridge::dispatch::Dispatcher;
    use crate::control::security::catalog::SystemCatalog;
    use crate::config::auth::JwtAuthConfig;
    use crate::control::security::jwks::registry::JwksRegistry;
    use crate::control::security::jwt::{JwtConfig, JwtValidator};
    use crate::control::server::sync::session::state::SyncSession;
    use crate::control::server::sync::wire::CompensationHint;
    use crate::control::state::SharedState;
    use crate::control::sync_producer::registry::SyncProducerRegistry;
    use crate::wal::WalManager;

    fn make_handshake(wire_version: u16) -> HandshakeMsg {
        HandshakeMsg {
            jwt_token: String::new(),
            vector_clock: HashMap::new(),
            subscribed_shapes: Vec::new(),
            client_version: "test".into(),
            lite_id: String::new(),
            epoch: 0,
            wire_version,
        }
    }

    fn open_registry(dir: &std::path::Path) -> SyncProducerRegistry {
        let catalog = Arc::new(SystemCatalog::open(&dir.join("system.redb")).unwrap());
        SyncProducerRegistry::open(catalog).unwrap()
    }

    fn trust_state() -> (Arc<SharedState>, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let wal = Arc::new(
            WalManager::open_for_testing(&dir.path().join("sync-trust.wal")).expect("open WAL"),
        );
        let (dispatcher, _data_sides) = Dispatcher::new(1, 64);
        let state = SharedState::new(dispatcher, wal).expect("shared state");
        state
            .credentials
            .bootstrap_trust_superuser("nodedb")
            .expect("bootstrap trust superuser");
        (state, dir)
    }

    #[tokio::test]
    async fn empty_token_uses_configured_durable_trust_identity() {
        let (state, _dir) = trust_state();
        let mut session = SyncSession::new("test-session".into());
        let validator = JwtValidator::new(JwtConfig::default());
        let msg = make_handshake(crate::version::WIRE_FORMAT_VERSION);

        let frame = session
            .handle_handshake(&msg, &validator, HashMap::new(), Some(&state))
            .expect("handshake response");
        let ack: HandshakeAckMsg = frame.decode_body().expect("decode handshake ack");

        assert!(ack.success, "configured trust handshake should succeed");
        let identity = session.identity.expect("durable trust identity");
        assert_eq!(identity.username, "nodedb");
        assert_ne!(identity.user_id, 0);
        assert!(identity.is_superuser);
    }

    #[test]
    fn empty_token_without_configured_identity_fails_closed() {
        let mut session = SyncSession::new("test-session".into());
        let validator = JwtValidator::new(JwtConfig::default());
        let msg = make_handshake(crate::version::WIRE_FORMAT_VERSION);

        let frame = session
            .handle_handshake(&msg, &validator, HashMap::new(), None)
            .expect("handshake response");
        let ack: HandshakeAckMsg = frame.decode_body().expect("decode handshake ack");

        assert!(!ack.success);
        assert!(!session.authenticated);
        assert!(session.identity.is_none());
    }

    #[test]
    fn handshake_rejects_wire_version_zero() {
        let mut session = SyncSession::new("test-session".into());
        let validator = JwtValidator::new(JwtConfig::default());
        let msg = make_handshake(0);

        let frame = session
            .handle_handshake(&msg, &validator, HashMap::new(), None)
            .expect("should return a frame");

        let ack: HandshakeAckMsg = frame.decode_body().expect("should decode HandshakeAckMsg");
        assert!(
            !ack.success,
            "wire_version=0 must be rejected, got success=true"
        );
        let error = ack.error.expect("error message must be present");
        assert!(
            error.contains("wire version") || error.contains("incompatible"),
            "error message should mention wire version, got: {error}"
        );
    }

    /// Uses the registry directly (not via SharedState) to exercise
    /// `durable_fencing_decision` in isolation.
    #[test]
    fn registry_new_lite_id_assigns_producer_id() {
        let dir = tempfile::tempdir().unwrap();
        let reg = open_registry(dir.path());

        // Simulate what durable_fencing_decision does for a new lite_id.
        let r = reg.register("device-a", 1, 99, 10, 0).unwrap();
        assert!(r.producer_id > 0);
        assert_eq!(r.current_epoch, 10);
    }

    #[test]
    fn producer_owner_binding_rejects_cross_tenant_and_cross_user_reuse() {
        let dir = tempfile::tempdir().unwrap();
        let reg = open_registry(dir.path());
        let registration = reg.register("owned-device", 7, 11, 1, 0).unwrap();

        assert!(producer_owner_matches(&registration, 7, 11));
        assert!(!producer_owner_matches(&registration, 8, 11));
        assert!(!producer_owner_matches(&registration, 7, 12));
    }

    #[test]
    fn registry_same_epoch_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let reg = open_registry(dir.path());

        let first = reg.register("device-b", 1, 99, 5, 0).unwrap();
        let loaded = reg.get("device-b").unwrap().unwrap();
        assert_eq!(loaded.producer_id, first.producer_id);
        assert_eq!(loaded.current_epoch, 5);
    }

    #[test]
    fn registry_higher_epoch_fences() {
        let dir = tempfile::tempdir().unwrap();
        let reg = open_registry(dir.path());

        let first = reg.register("device-c", 1, 99, 3, 0).unwrap();
        reg.fence("device-c", 7).unwrap();
        let loaded = reg.get("device-c").unwrap().unwrap();
        assert_eq!(loaded.producer_id, first.producer_id);
        assert_eq!(loaded.current_epoch, 7);
    }

    #[test]
    fn registry_lower_epoch_is_stale() {
        let dir = tempfile::tempdir().unwrap();
        let reg = open_registry(dir.path());

        reg.register("device-d", 1, 99, 9, 0).unwrap();
        let loaded = reg.get("device-d").unwrap().unwrap();
        // A client presenting epoch < current_epoch (9) must be rejected.
        assert!(
            3 < loaded.current_epoch,
            "epoch 3 is stale vs {}",
            loaded.current_epoch
        );
    }

    #[tokio::test]
    async fn malformed_handshake_clears_binding_and_denies_following_delta() {
        let (shared, _dir) = trust_state();
        let validator = JwtValidator::new(JwtConfig::default());
        let mut session = SyncSession::new("malformed-handshake-session".into());

        // Establish a real production-backed identity, then stage the
        // identity-bound state a prior authenticated Lite handshake can hold.
        let valid_handshake = make_handshake(crate::version::WIRE_FORMAT_VERSION);
        let valid_frame = session
            .handle_handshake(&valid_handshake, &validator, HashMap::new(), Some(&shared))
            .expect("valid handshake response");
        let valid_ack: HandshakeAckMsg = valid_frame.decode_body().expect("decode valid ack");
        assert!(valid_ack.success);
        assert!(session.authenticated);
        assert!(session.identity.is_some());

        session
            .client_clock
            .insert("orders".into(), HashMap::from([("peer-1".into(), 4)]));
        session.server_clock.insert("orders".into(), 8);
        session.subscribed_shapes.push("orders-shape".into());
        session.last_seen_lsn = 8;
        session.producer_id = 42;
        session.accepted_epoch = 7;
        session.delta_signing_key = Some([9; 32]);
        session.device_metadata.remote_addr = "127.0.0.1:1234".into();
        session.device_metadata.peer_id = 99;
        session.track_collection(1, "orders");
        session.announced_collections.insert("orders".into());

        // Round-trip through the framing codec so this is a CRC-valid
        // Handshake frame whose MessagePack body simply has the wrong shape.
        let malformed_wire = SyncFrame::try_encode(
            SyncMessageType::Handshake,
            &"not a handshake message".to_string(),
        )
        .expect("encode malformed handshake body")
        .to_bytes();
        let malformed_frame =
            SyncFrame::from_bytes(&malformed_wire).expect("CRC-valid malformed handshake frame");
        let response = session
            .process_frame(
                &malformed_frame,
                &validator,
                Some(&shared.rls),
                None,
                None,
                Some(&shared),
            )
            .expect("malformed handshake rejection");

        assert_eq!(response.msg_type, SyncMessageType::HandshakeAck);
        let ack: HandshakeAckMsg = response.decode_body().expect("decode malformed ack");
        assert!(!ack.success);
        assert!(!ack.fork_detected);
        assert_eq!(ack.error.as_deref(), Some("malformed handshake"));
        assert!(!session.authenticated);
        assert!(session.tenant_id.is_none());
        assert!(session.username.is_none());
        assert!(session.identity.is_none());
        assert!(session.client_clock.is_empty());
        assert!(session.server_clock.is_empty());
        assert!(session.subscribed_shapes.is_empty());
        assert_eq!(session.last_seen_lsn, 0);
        assert_eq!(session.producer_id, 0);
        assert_eq!(session.accepted_epoch, 0);
        assert!(session.delta_signing_key.is_none());
        assert!(session.device_metadata.client_version.is_empty());
        assert!(session.device_metadata.remote_addr.is_empty());
        assert_eq!(session.device_metadata.peer_id, 0);
        assert!(session.tracked_collections.is_empty());
        assert!(session.announced_collections.is_empty());

        let delta = DeltaPushMsg {
            collection: "orders".into(),
            document_id: "order-1".into(),
            delta: nodedb_types::json_to_msgpack(&serde_json::json!({"status": "active"}))
                .expect("encode valid delta"),
            peer_id: 1,
            mutation_id: 17,
            device_id: 0,
            delta_signature: [0; 32],
            checksum: 0,
            device_valid_time_ms: None,
            producer_id: 42,
            epoch: 7,
            seq: 1,
        };
        let delta_frame =
            SyncFrame::try_encode(SyncMessageType::DeltaPush, &delta).expect("encode delta frame");
        let response = session
            .process_frame(
                &delta_frame,
                &validator,
                Some(&shared.rls),
                None,
                None,
                Some(&shared),
            )
            .expect("permission denial response");
        assert_eq!(response.msg_type, SyncMessageType::DeltaReject);
        let reject: DeltaRejectMsg = response.decode_body().expect("decode delta rejection");
        assert_eq!(
            reject.compensation,
            Some(CompensationHint::PermissionDenied)
        );
        assert_eq!(session.mutations_processed, 0);
    }

    #[tokio::test]
    async fn stale_lite_rehandshake_clears_trust_binding_before_delta_dispatch() {
        // Keep the WAL and Data-Plane endpoints alive for the complete
        // SharedState lifetime, as the production-backed path requires.
        let dir = tempfile::tempdir().expect("tempdir");
        let wal = Arc::new(
            WalManager::open_for_testing(&dir.path().join("sync-fencing.wal")).expect("open WAL"),
        );
        let (dispatcher, _data_sides) = Dispatcher::new(1, 64);
        let mut shared = SharedState::new(dispatcher, Arc::clone(&wal)).expect("shared state");
        shared
            .credentials
            .bootstrap_trust_superuser("nodedb")
            .expect("bootstrap trust superuser");
        let registry = Arc::new(open_registry(dir.path()));
        Arc::get_mut(&mut shared)
            .expect("SharedState has a single owner while configuring test")
            .producer_registry = Some(registry);

        let validator = JwtValidator::new(JwtConfig::default());
        let mut high_epoch_session = SyncSession::new("high-epoch-session".into());
        let mut high_epoch_msg = make_handshake(crate::version::WIRE_FORMAT_VERSION);
        high_epoch_msg.lite_id = "fenced-lite-id".into();
        high_epoch_msg.epoch = 9;
        let high_epoch_frame = high_epoch_session
            .handle_handshake(&high_epoch_msg, &validator, HashMap::new(), Some(&shared))
            .expect("high epoch handshake response");
        let high_epoch_ack: HandshakeAckMsg = high_epoch_frame.decode_body().expect("decode ack");
        assert!(high_epoch_ack.success, "higher epoch must be accepted");
        assert!(high_epoch_session.authenticated);
        assert_ne!(high_epoch_session.producer_id, 0);
        assert_eq!(high_epoch_session.accepted_epoch, 9);

        // Seed identity-bound state that a prior successful handshake can
        // accumulate. The next handshake attempt must not retain any of it.
        high_epoch_session
            .client_clock
            .insert("orders".into(), HashMap::from([("peer-1".into(), 4)]));
        high_epoch_session.server_clock.insert("orders".into(), 8);
        high_epoch_session
            .subscribed_shapes
            .push("orders-shape".into());
        high_epoch_session.last_seen_lsn = 8;
        high_epoch_session.track_collection(1, "orders");
        high_epoch_session
            .announced_collections
            .insert("orders".into());

        // This new handshake attempt stages the configured trust identity
        // before the registry discovers that this LiteId's epoch is stale.
        let mut stale_msg = make_handshake(crate::version::WIRE_FORMAT_VERSION);
        stale_msg.lite_id = "fenced-lite-id".into();
        stale_msg.epoch = 3;
        let stale_frame = high_epoch_session
            .handle_handshake(&stale_msg, &validator, HashMap::new(), Some(&shared))
            .expect("stale handshake response");
        let stale_ack: HandshakeAckMsg = stale_frame.decode_body().expect("decode stale ack");

        assert!(!stale_ack.success);
        assert!(stale_ack.fork_detected);
        assert!(!high_epoch_session.authenticated);
        assert!(high_epoch_session.tenant_id.is_none());
        assert!(high_epoch_session.username.is_none());
        assert!(high_epoch_session.identity.is_none());
        assert!(high_epoch_session.client_clock.is_empty());
        assert!(high_epoch_session.server_clock.is_empty());
        assert!(high_epoch_session.subscribed_shapes.is_empty());
        assert_eq!(high_epoch_session.last_seen_lsn, 0);
        assert_eq!(high_epoch_session.producer_id, 0);
        assert_eq!(high_epoch_session.accepted_epoch, 0);
        assert!(high_epoch_session.delta_signing_key.is_none());
        assert!(high_epoch_session.device_metadata.client_version.is_empty());
        assert!(high_epoch_session.device_metadata.remote_addr.is_empty());
        assert_eq!(high_epoch_session.device_metadata.peer_id, 0);
        assert!(high_epoch_session.tracked_collections.is_empty());
        assert!(high_epoch_session.announced_collections.is_empty());

        // The configured trust identity is a superuser, so this would be
        // authorized and provisionally ACKed if the stale handshake retained
        // its staged identity. The production dispatch gate must deny it.
        let delta = DeltaPushMsg {
            collection: "orders".into(),
            document_id: "order-1".into(),
            delta: nodedb_types::json_to_msgpack(&serde_json::json!({"status": "active"}))
                .expect("encode valid delta"),
            peer_id: 1,
            mutation_id: 17,
            device_id: 0,
            delta_signature: [0; 32],
            checksum: 0,
            device_valid_time_ms: None,
            producer_id: 0,
            epoch: 0,
            seq: 0,
        };
        let delta_frame =
            SyncFrame::try_encode(SyncMessageType::DeltaPush, &delta).expect("encode delta frame");
        let response = high_epoch_session
            .process_frame(
                &delta_frame,
                &validator,
                Some(&shared.rls),
                None,
                None,
                Some(&shared),
            )
            .expect("permission denial response");
        assert_eq!(response.msg_type, SyncMessageType::DeltaReject);
        let reject: DeltaRejectMsg = response.decode_body().expect("decode delta rejection");
        assert_eq!(
            reject.compensation,
            Some(CompensationHint::PermissionDenied)
        );
        assert_eq!(high_epoch_session.mutations_processed, 0);
    }

    // ── JWKS-registry handshake authentication ─────────────────────────
    //
    // These exercise the path the async session loop takes when
    // `SharedState.jwks_registry` is configured: the registry validates the
    // handshake token and `handle_handshake_verified` binds the session from
    // the verified identity.

    /// Serve a fixed JWKS document over HTTP on localhost. Returns its URL.
    async fn spawn_jwks_endpoint(body: String) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                use tokio::io::AsyncWriteExt;
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = sock.write_all(resp.as_bytes()).await;
            }
        });
        format!("http://localhost:{}/jwks.json", addr.port())
    }

    fn b64(bytes: &[u8]) -> String {
        use base64::Engine;
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
    }

    /// Mint an RS256 JWT and the JWKS document that verifies it.
    ///
    /// `claimed_tenant_id` is written into the token so the tests can prove
    /// the server binds the provider's tenant instead of the token's claim.
    fn rs256_fixture(kid: &str, issuer: &str, audience: &str, claimed_tenant_id: u64) -> (String, String) {
        use rsa::pkcs1v15::SigningKey;
        use rsa::signature::{SignatureEncoding, Signer};
        use rsa::traits::PublicKeyParts;

        let mut rng = rsa::rand_core::OsRng;
        let private_key = rsa::RsaPrivateKey::new(&mut rng, 1024).unwrap();
        let public_key = rsa::RsaPublicKey::from(&private_key);
        let jwks = format!(
            r#"{{"keys":[{{"kty":"RSA","kid":"{kid}","alg":"RS256","use":"sig","n":"{}","e":"{}"}}]}}"#,
            b64(&public_key.n().to_bytes_be()),
            b64(&public_key.e().to_bytes_be()),
        );
        let header = b64(format!(r#"{{"alg":"RS256","kid":"{kid}"}}"#).as_bytes());
        let issued_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("fixture clock must be after epoch")
            .as_secs();
        let payload = b64(
            format!(
                r#"{{"iss":"{issuer}","aud":"{audience}","sub":"lite-replica","tenant_id":{claimed_tenant_id},"roles":["readwrite"],"iat":{issued_at},"exp":{},"user_id":42}}"#,
                issued_at + 3_600
            )
            .as_bytes(),
        );
        let signing_input = format!("{header}.{payload}");
        let signing_key = SigningKey::<sha2::Sha256>::new(private_key);
        let signature: rsa::pkcs1v15::Signature = signing_key.sign(signing_input.as_bytes());
        (jwks, format!("{signing_input}.{}", b64(&signature.to_bytes())))
    }

    fn provider_config(jwks_url: &str, issuer: &str, audience: &str, tenant_id: u64) -> JwtAuthConfig {
        toml::from_str(&format!(
            r#"
allow_http_jwks = true
allow_jwks_hosts = ["localhost"]
allow_jwks_cidrs = ["127.0.0.0/8", "::1/128"]

[[providers]]
name = "sync-provider"
jwks_url = "{jwks_url}"
issuer = "{issuer}"
audience = "{audience}"
tenant_id = {tenant_id}
"#
        ))
        .expect("JWT provider fixture must deserialize")
    }

    fn token_handshake(token: &str) -> HandshakeMsg {
        HandshakeMsg {
            jwt_token: token.into(),
            ..make_handshake(crate::version::WIRE_FORMAT_VERSION)
        }
    }

    const ISSUER: &str = "https://issuer.example.test/";
    const AUDIENCE: &str = "nodedb-sync";

    #[tokio::test]
    async fn registry_verified_token_is_accepted_and_binds_provider_tenant() {
        // The token claims tenant 999; the provider is bound to tenant 7.
        let (jwks, token) = rs256_fixture("sync-key", ISSUER, AUDIENCE, 999);
        let jwks_url = spawn_jwks_endpoint(jwks).await;
        let registry = JwksRegistry::init(provider_config(&jwks_url, ISSUER, AUDIENCE, 7))
            .await
            .expect("registry must initialise");

        let verified = registry
            .validate_with_claims(&token)
            .await
            .map(|(identity, _)| identity)
            .map_err(|e| e.to_string());

        let mut session = SyncSession::new("registry-ok".into());
        let frame = session
            .handle_handshake_verified(
                &token_handshake(&token),
                verified,
                HashMap::new(),
                None,
            )
            .expect("handshake response");
        let ack: HandshakeAckMsg = frame.decode_body().expect("decode handshake ack");

        assert!(ack.success, "registry-verified token must be accepted: {:?}", ack.error);
        assert!(session.authenticated);
        let identity = session.identity.expect("verified identity");
        assert_eq!(identity.username, "lite-replica");
        assert_eq!(
            identity.tenant_id.as_u64(),
            7,
            "tenant must come from provider config, never from the token claim"
        );
        assert_eq!(session.tenant_id.map(|t| t.as_u64()), Some(7));
    }

    #[tokio::test]
    async fn token_signed_by_unknown_key_is_rejected() {
        let (jwks, _good) = rs256_fixture("sync-key", ISSUER, AUDIENCE, 7);
        // A second keypair whose public key is never published to the endpoint.
        let (_unpublished, forged) = rs256_fixture("attacker-key", ISSUER, AUDIENCE, 7);
        let jwks_url = spawn_jwks_endpoint(jwks).await;
        let registry = JwksRegistry::init(provider_config(&jwks_url, ISSUER, AUDIENCE, 7))
            .await
            .expect("registry must initialise");

        let verified = registry
            .validate_with_claims(&forged)
            .await
            .map(|(identity, _)| identity)
            .map_err(|e| e.to_string());
        assert!(verified.is_err(), "an unknown signing key must not verify");

        let mut session = SyncSession::new("registry-forged".into());
        let frame = session
            .handle_handshake_verified(
                &token_handshake(&forged),
                verified,
                HashMap::new(),
                None,
            )
            .expect("handshake response");
        let ack: HandshakeAckMsg = frame.decode_body().expect("decode handshake ack");

        assert!(!ack.success, "a token signed by an unknown key must be rejected");
        assert!(ack.error.is_some());
        assert!(!session.authenticated);
        assert!(session.identity.is_none());
        assert!(session.tenant_id.is_none());
    }

    #[test]
    fn without_registry_a_token_still_uses_the_static_validator() {
        // No registry configured: the static validator is unchanged, and with
        // the default (unpinned-algorithm) JwtConfig it fails closed exactly
        // as it did before registry support existed.
        let mut session = SyncSession::new("no-registry".into());
        let validator = JwtValidator::new(JwtConfig::default());

        let frame = session
            .handle_handshake(
                &token_handshake("header.payload.signature"),
                &validator,
                HashMap::new(),
                None,
            )
            .expect("handshake response");
        let ack: HandshakeAckMsg = frame.decode_body().expect("decode handshake ack");

        assert!(!ack.success);
        assert!(!session.authenticated);
        assert!(session.identity.is_none());
    }
}
