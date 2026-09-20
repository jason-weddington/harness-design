//! The cluster/decline ledger seam and the decline filter (the
//! expensive-if-missed part of the pipeline).
//!
//! The ledger table exists on the KB side (`map_cluster_ledger`, both
//! backends), and its HTTP endpoints are LIVE: `POST
//! /api/kb/cluster-ledger/match` and `POST /api/kb/cluster-ledger/decline`
//! (merged on the KB side at `personal_kb` `d39cbc1`, live on all three KB
//! hosts). This module therefore carries the trait designed to that
//! batch-match shape, the pure [`filter_pockets`], and TWO implementations
//! of the trait: [`HttpClusterLedger`], the real reqwest transport, and the
//! in-memory [`InMemoryLedger`] test stand-in the unit tests run against.
//!
//! Why the ledger exists at all: a declined cluster is a real cluster whose
//! members have nothing chunky to point at, and paying Sonnet to re-propose
//! it every night is both waste and noise. A decline is permanent with one
//! escape — reopen when the member set doubles.
//!
//! Layering (pinned): this module references only [`crate::loop_input`]
//! types.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::loop_input::{LoopInputPocket, TokenError, read_token_from};

/// One candidate cluster for the ledger's batch-match call: the member-id
/// set, and NOTHING beyond it — matching is keyed on member-set overlap, not
/// label, because labels drift between runs. `Serialize` is additive so the
/// HTTP transport can embed candidates in the request body verbatim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LedgerCandidate {
    /// The candidate's member entry ids.
    pub member_entry_ids: Vec<String>,
}

impl LedgerCandidate {
    /// The candidate's member-id set, for set arithmetic.
    #[must_use]
    pub fn member_set(&self) -> HashSet<String> {
        self.member_entry_ids.iter().cloned().collect()
    }
}

/// One verdict from the ledger's batch-match call, carrying the requested
/// server batch-match shape verbatim and nothing beyond it. `index` is the
/// verdict's own claim about which candidate it answers, checked against the
/// verdict's position by [`filter_pockets`]. `Deserialize` is additive so
/// the HTTP transport can parse the wire envelope's `results` array.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CandidateVerdict {
    /// The position this verdict claims to answer.
    pub index: usize,
    /// Whether a stored decline matches the candidate.
    pub matched: bool,
    /// The matched decline's ledger row id, when matched.
    pub ledger_id: Option<String>,
    /// The matched decline's status, when matched.
    pub status: Option<String>,
    /// The Jaccard overlap the server computed, when it computed one.
    pub jaccard: Option<f64>,
    /// Whether the candidate's member set has grown enough to reopen the
    /// decline (the stored set doubled or more). This maps the server's
    /// `reopened` PREDICATE — matched AND declined AND `len(members) >= 2 *
    /// declined_member_count` — not a past-tense record, so the name stays
    /// as the contract spells it and the client recomputes nothing.
    pub reopen_eligible: bool,
}

/// Why a ledger interaction failed. [`LedgerError::VerdictMisalignment`] and
/// [`LedgerError::VerdictCountMismatch`] are constructed ONLY by
/// [`filter_pockets`] — a malformed verdict vector must never panic, it must
/// fail closed. [`LedgerError::Source`] is constructed by
/// [`HttpClusterLedger`], the real HTTP transport, whose `post` helper maps
/// every transport, status, timeout, and body-parse failure to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LedgerError {
    /// A verdict at position `got` claims to answer index `expected`.
    VerdictMisalignment {
        /// The index the verdict claims.
        expected: usize,
        /// The position the verdict actually occupies.
        got: usize,
    },
    /// The ledger answered `verdicts` verdicts for `pockets` pockets.
    VerdictCountMismatch {
        /// How many pockets were submitted.
        pockets: usize,
        /// How many verdicts came back.
        verdicts: usize,
    },
    /// The transport behind the ledger failed.
    Source {
        /// Why it failed.
        reason: String,
    },
}

impl std::fmt::Display for LedgerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VerdictMisalignment { expected, got } => write!(
                f,
                "somnus: ledger verdict at position {got} claims index {expected}"
            ),
            Self::VerdictCountMismatch { pockets, verdicts } => write!(
                f,
                "somnus: ledger returned {verdicts} verdicts for {pockets} pockets"
            ),
            Self::Source { reason } => {
                write!(f, "somnus: ledger source error: {reason}")
            }
        }
    }
}

impl std::error::Error for LedgerError {}

/// The ledger seam: batch-match a night's candidates against the stored
/// declines, and record a new decline. The production transport is
/// [`HttpClusterLedger`] over the KB's cluster-ledger endpoints;
/// [`InMemoryLedger`] is the test stand-in.
#[async_trait]
pub trait ClusterLedger: std::fmt::Debug + Send + Sync {
    /// Match `candidates` against the stored declines for `project_ref`,
    /// answering one verdict per candidate, in order.
    ///
    /// # Errors
    /// A transport failure, or a verdict vector too malformed for
    /// [`filter_pockets`] to consume.
    async fn match_candidates(
        &self,
        project_ref: &str,
        candidates: &[LedgerCandidate],
    ) -> Result<Vec<CandidateVerdict>, LedgerError>;

    /// Record a decline: the cluster with these member ids is real, but
    /// nothing chunky exists to point at.
    ///
    /// # Errors
    /// A transport failure.
    async fn record_decline(
        &self,
        project_ref: &str,
        member_entry_ids: &[String],
    ) -> Result<(), LedgerError>;
}

/// The in-memory [`ClusterLedger`]: a documented TEST STAND-IN for the KB's
/// `kb_core.cluster_ledger`, so the pipeline, the filter, and the decline
/// write are exercisable with no live KB and no network.
///
/// The REAL rules live server-side and are deliberately NOT re-derived here:
/// somnus production code computes no overlap and holds no threshold
/// constant. This stand-in mirrors the shape the server contract requests —
/// subset containment for "is this cluster a previously declined one?", and
/// a ≥2× cardinality rule for the reopen escape — so the filter's behaviour
/// is testable while the authoritative arithmetic stays where it belongs.
#[derive(Debug, Default)]
pub struct InMemoryLedger {
    /// Per project, the stored declines in insertion order.
    declines: Mutex<HashMap<String, Vec<HashSet<String>>>>,
}

impl InMemoryLedger {
    /// An empty ledger.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The declines recorded for `project_ref`, in insertion order (an
    /// inspection helper so tests can assert what was written).
    ///
    /// # Panics
    /// Panics if the internal lock is poisoned (a fail-loudly, test-only
    /// posture).
    #[must_use]
    pub fn recorded_declines(&self, project_ref: &str) -> Vec<Vec<String>> {
        let declines = self.declines.lock().expect("declines lock poisoned");
        declines
            .get(project_ref)
            .map(|sets| {
                sets.iter()
                    .map(|set| {
                        let mut ids: Vec<String> = set.iter().cloned().collect();
                        ids.sort();
                        ids
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[async_trait]
impl ClusterLedger for InMemoryLedger {
    async fn match_candidates(
        &self,
        project_ref: &str,
        candidates: &[LedgerCandidate],
    ) -> Result<Vec<CandidateVerdict>, LedgerError> {
        let declines = self.declines.lock().expect("declines lock poisoned");
        let stored = declines.get(project_ref).cloned().unwrap_or_default();
        Ok(candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| {
                let member_set = candidate.member_set();
                // The FIRST stored decline (insertion order) whose member
                // set is a subset of the candidate's members is the match.
                let matched_position = stored
                    .iter()
                    .position(|stored_set| stored_set.is_subset(&member_set));
                match matched_position {
                    Some(position) => {
                        let stored_set = &stored[position];
                        CandidateVerdict {
                            index,
                            matched: true,
                            ledger_id: Some(format!("decline-{}", position + 1)),
                            status: Some("declined".to_string()),
                            jaccard: None,
                            reopen_eligible: candidate.member_entry_ids.len()
                                >= 2 * stored_set.len(),
                        }
                    }
                    None => CandidateVerdict {
                        index,
                        matched: false,
                        ledger_id: None,
                        status: None,
                        jaccard: None,
                        reopen_eligible: false,
                    },
                }
            })
            .collect())
    }

    async fn record_decline(
        &self,
        project_ref: &str,
        member_entry_ids: &[String],
    ) -> Result<(), LedgerError> {
        let mut declines = self.declines.lock().expect("declines lock poisoned");
        declines
            .entry(project_ref.to_string())
            .or_default()
            .push(member_entry_ids.iter().cloned().collect());
        Ok(())
    }
}

/// Split a night's pockets into `(kept, declined)` using the ledger's
/// verdicts, failing CLOSED on a malformed verdict vector.
///
/// The checks are ordered so a short (or long) verdict vector can never be
/// indexed out of range: [`LedgerError::VerdictCountMismatch`] fires FIRST,
/// then each verdict's `index` is checked against its position
/// ([`LedgerError::VerdictMisalignment`]). Only then is any pocket touched,
/// and a pocket is dropped exactly when its verdict is `matched` and NOT
/// `reopen_eligible`.
///
/// # Errors
/// [`LedgerError::VerdictCountMismatch`] when the counts disagree;
/// [`LedgerError::VerdictMisalignment`] when a verdict's claimed index is
/// not its position.
pub fn filter_pockets(
    pockets: &[LoopInputPocket],
    verdicts: &[CandidateVerdict],
) -> Result<(Vec<LoopInputPocket>, Vec<LoopInputPocket>), LedgerError> {
    if verdicts.len() != pockets.len() {
        return Err(LedgerError::VerdictCountMismatch {
            pockets: pockets.len(),
            verdicts: verdicts.len(),
        });
    }
    for (position, verdict) in verdicts.iter().enumerate() {
        if verdict.index != position {
            return Err(LedgerError::VerdictMisalignment {
                expected: verdict.index,
                got: position,
            });
        }
    }
    let mut kept = Vec::new();
    let mut declined = Vec::new();
    for (pocket, verdict) in pockets.iter().zip(verdicts) {
        if verdict.matched && !verdict.reopen_eligible {
            declined.push(pocket.clone());
        } else {
            kept.push(pocket.clone());
        }
    }
    Ok((kept, declined))
}

// ===== The real HTTP transport ===========================================

/// The per-call HTTP bound for BOTH ledger endpoints, enforced by the shared
/// `post` helper's `tokio::time::timeout` wrapper around the entire exchange
/// (send, status map, body read) — the same wrapper shape as
/// [`crate::observer::OBSERVE_BOUND`]'s around the pointer-count call.
///
/// Neither somnus HTTP transport carries a per-call bound today: the only
/// wrapper over the two `unit.rs` ledger call sites (steps (5) and (9)) is
/// the unit-level nightly wall clock ([`crate::NIGHTLY_WALL_CLOCK_SECS`],
/// 14,400s, armed by `unit.rs`'s `run_unit`), and the loop-input step-(3)
/// `fetch` sits in exactly the same situation (the observer's bound covers
/// only the pointer-count path, not the fetch). This client ADOPTS a 10s
/// bound — a healthy KB answers well inside ten seconds, the same
/// justification [`crate::observer::OBSERVE_BOUND`] carries — which
/// [`HttpLoopInputSource`](crate::loop_input::HttpLoopInputSource)
/// deliberately does not have.
pub const LEDGER_HTTP_BOUND: Duration = Duration::from_secs(10);

/// The pinned decline reason the transport supplies for EVERY decline write.
/// The trait's `record_decline` signature carries no reason parameter, but
/// the endpoint requires one — a trait-expressiveness finding, recorded here
/// rather than changed: the transport supplies this one pinned literal, so
/// every decline reads the same way on the KB side.
pub const DECLINE_REASON: &str = "real cluster with nothing chunky to point at";

/// The pinned endpoint paths. Exactly two routes, one request each, no
/// retry and no second endpoint anywhere.
const MATCH_PATH: &str = "/api/kb/cluster-ledger/match";
const DECLINE_PATH: &str = "/api/kb/cluster-ledger/decline";

/// The shared label-prefixed status-reason forms: `post` renders
/// `"{label}: {form}"`, so the match side and the decline side pin their own
/// strings by construction.
const NOT_FOUND_FORM: &str = "project_ref not found (404)";
const NOT_ELIGIBLE_FORM: &str = "project_ref not map-eligible (409)";
const UNAUTHORIZED_FORM: &str = "401 unauthorized";

/// The match request wire envelope. Serde serializes the fields in
/// declaration order, so the request body is byte-stable.
#[derive(Serialize)]
struct MatchRequest<'a> {
    project_ref: &'a str,
    candidates: &'a [LedgerCandidate],
}

/// The match response wire envelope: one `results` array, positionally
/// ordered.
#[derive(Deserialize)]
struct MatchResults {
    results: Vec<CandidateVerdict>,
}

/// The decline request wire envelope. The `reason` field is supplied by the
/// transport ([`DECLINE_REASON`]) because the trait's signature cannot
/// express one.
#[derive(Serialize)]
struct DeclineRequest<'a> {
    project_ref: &'a str,
    member_entry_ids: &'a [String],
    reason: &'a str,
}

/// The decline response wire envelope, parsed to pin the response shape and
/// catch contract drift; the id is then dropped because the trait's
/// `Result<(), LedgerError>` cannot express a returned id.
#[derive(Deserialize)]
#[allow(dead_code)] // `ledger_id` is parsed and pinned, never read: the trait cannot return it
struct DeclineResponse {
    ledger_id: String,
}

/// One candidate's verdict, zipped with its member ids for the run report:
/// the report alone reconstructs WHICH cluster was declined, against which
/// stored decline, at what jaccard.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LedgerVerdictRecord {
    /// The verdict's claimed index (its position in the batch).
    pub index: usize,
    /// The candidate's member entry ids.
    pub member_entry_ids: Vec<String>,
    /// Whether a stored decline matched the candidate.
    pub matched: bool,
    /// The matched decline's ledger row id, when matched.
    pub ledger_id: Option<String>,
    /// The matched decline's status, when matched.
    pub status: Option<String>,
    /// The jaccard the server computed, when it computed one — carried
    /// verbatim, never recomputed or overridden.
    pub jaccard: Option<f64>,
    /// The server's `reopened` predicate, verbatim.
    pub reopen_eligible: bool,
}

/// The production [`ClusterLedger`]: exactly ONE `POST` per call, to
/// `/api/kb/cluster-ledger/match` or `/api/kb/cluster-ledger/decline`, with
/// `Authorization: Bearer <token>` and `content-type: application/json`, no
/// retry, no second endpoint, and no overlap arithmetic of its own —
/// matching (the jaccard threshold, the reopen rule) stays server-side and
/// this client computes no overlap and holds no threshold.
///
/// The token is held as the result of reading it from disk (the same
/// `~/.config/somnus/kb-token` file [`HttpLoopInputSource`] reads — one
/// token seam, via [`read_token_from`] over
/// [`crate::loop_input::token_path`]), so a source whose token could not be
/// read can never turn into a request: both trait methods answer
/// [`LedgerError::Source`] without touching the network. The token is
/// deliberately omitted from the `Debug` rendering.
pub struct HttpClusterLedger {
    base_url: String,
    token: Result<String, TokenError>,
}

impl HttpClusterLedger {
    /// Build a transport over an already-read token.
    #[must_use]
    pub fn new(base_url: String, token: String) -> Self {
        Self {
            base_url,
            token: Ok(token),
        }
    }

    /// Build a transport by reading the token out of `home` (see
    /// [`read_token_from`], over [`crate::loop_input::token_path`] — the
    /// same `kb-token` file [`HttpLoopInputSource::from_home`] reads, so
    /// there is no second token seam). Constructing this from an `Err`
    /// token result is exactly the zero-request posture: no method dials.
    #[must_use]
    pub fn from_home(base_url: String, home: &Path) -> Self {
        Self {
            base_url,
            token: read_token_from(home),
        }
    }

    /// The pre-flight guards, shared by both trait methods and run in one
    /// order: the `project_ref` charset FIRST (via
    /// [`crate::loop_input::validate_project_ref`], which answers
    /// [`crate::loop_input::INVALID_PROJECT_REF_MSG`]), then the stored
    /// token result. Both refuse WITHOUT touching the wire.
    fn guards(&self, project_ref: &str) -> Result<(), LedgerError> {
        if let Err(reason) = crate::loop_input::validate_project_ref(project_ref) {
            return Err(LedgerError::Source { reason });
        }
        if let Err(err) = &self.token {
            return Err(LedgerError::Source {
                reason: err.to_string(),
            });
        }
        Ok(())
    }

    /// The ONE request path both trait methods share — no per-method
    /// duplicate request code exists: the token guard, then the ENTIRE
    /// exchange (reqwest send, status mapping, body read) wrapped in
    /// `tokio::time::timeout` with [`LEDGER_HTTP_BOUND`]. Every `Source`
    /// reason carries the `label` prefix (`"match"` or `"decline"`) so a
    /// failure names the endpoint.
    ///
    /// The ref charset guard has already run in [`guards`] before the
    /// caller serialized the body.
    async fn post(&self, label: &str, path: &str, body: String) -> Result<String, LedgerError> {
        let token = match &self.token {
            Ok(token) => token.as_str(),
            Err(err) => {
                return Err(LedgerError::Source {
                    reason: err.to_string(),
                });
            }
        };
        let url = format!("{}{path}", self.base_url.trim_end_matches('/'));
        // The pinned success status per label: the match endpoint answers
        // 200, the decline endpoint 201.
        let ok_status: u16 = if label == "match" { 200 } else { 201 };
        let label = label.to_string();
        let exchange = async {
            let status = |response: &reqwest::Response| response.status().as_u16();
            let response = reqwest::Client::new()
                .post(&url)
                .header(reqwest::header::AUTHORIZATION, format!("Bearer {token}"))
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(body)
                .send()
                .await
                .map_err(|err| LedgerError::Source {
                    reason: format!("{label}: transport failed: {err}"),
                })?;
            if status(&response) != ok_status {
                return Err(status_reason(&label, status(&response)));
            }
            // A failed body read is a TRANSPORT failure, the same terminal
            // form as a failed send — not a new reason kind.
            response.text().await.map_err(|err| LedgerError::Source {
                reason: format!("{label}: transport failed: {err}"),
            })
        };
        match tokio::time::timeout(LEDGER_HTTP_BOUND, exchange).await {
            Ok(result) => result,
            Err(_elapsed) => Err(LedgerError::Source {
                reason: format!("{label}: timed out after {}s", LEDGER_HTTP_BOUND.as_secs()),
            }),
        }
    }
}

impl std::fmt::Debug for HttpClusterLedger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never render the token, even under Debug.
        f.debug_struct("HttpClusterLedger")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

/// Map a non-success HTTP status to the pinned `Source` reason: `404` and
/// `409` are the admission contract (terminal and ordinary, overrides are
/// the human's call), `401` is auth, and anything else is a named surprise.
fn status_reason(label: &str, status: u16) -> LedgerError {
    let form = match status {
        404 => NOT_FOUND_FORM,
        409 => NOT_ELIGIBLE_FORM,
        401 => UNAUTHORIZED_FORM,
        other => {
            return LedgerError::Source {
                reason: format!("{label}: unexpected HTTP status {other}"),
            };
        }
    };
    LedgerError::Source {
        reason: format!("{label}: {form}"),
    }
}

#[async_trait]
impl ClusterLedger for HttpClusterLedger {
    async fn match_candidates(
        &self,
        project_ref: &str,
        candidates: &[LedgerCandidate],
    ) -> Result<Vec<CandidateVerdict>, LedgerError> {
        self.guards(project_ref)?;
        // The empty boundary: the trait contract is one verdict per
        // candidate, so zero candidates means zero verdicts and NO request
        // — the server's empty-batch shape exists nowhere in this clone, so
        // no empty-batch request is invented.
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let body = serde_json::to_string(&MatchRequest {
            project_ref,
            candidates,
        })
        .expect("a match request of plain strings serializes");
        let text = self.post("match", MATCH_PATH, body).await?;
        // The transport's single non-short-circuit `Ok` construction site.
        // Residual risk: a well-formed FABRICATED all-unmatched verdict set
        // would pass right through here — the downstream tripwires are
        // [`filter_pockets`]'s count/index checks and unit.rs's
        // `ledger_verdicts` report record; this transport's own job is
        // never to fabricate, which the one-request pins enforce.
        serde_json::from_str::<MatchResults>(&text)
            .map(|wire| wire.results)
            .map_err(|err| LedgerError::Source {
                reason: format!("match: malformed body ({err})"),
            })
    }

    async fn record_decline(
        &self,
        project_ref: &str,
        member_entry_ids: &[String],
    ) -> Result<(), LedgerError> {
        self.guards(project_ref)?;
        // The empty boundary: a decline with no members is a no-op with no
        // request — no empty-batch shape is invented for it either.
        if member_entry_ids.is_empty() {
            return Ok(());
        }
        let body = serde_json::to_string(&DeclineRequest {
            project_ref,
            member_entry_ids,
            reason: DECLINE_REASON,
        })
        .expect("a decline request of plain strings serializes");
        let text = self.post("decline", DECLINE_PATH, body).await?;
        // The response's `ledger_id` is parsed — pinning the response shape
        // and catching contract drift — and then dropped: the trait's
        // `Result<(), LedgerError>` cannot express a returned id.
        serde_json::from_str::<DeclineResponse>(&text).map_err(|err| LedgerError::Source {
            reason: format!("decline: malformed body ({err})"),
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loop_input::fixture;

    /// A verdict for position `index`, matched or not.
    fn verdict(index: usize, matched: bool, reopen_eligible: bool) -> CandidateVerdict {
        CandidateVerdict {
            index,
            matched,
            ledger_id: if matched {
                Some(format!("decline-{index}"))
            } else {
                None
            },
            status: if matched {
                Some("declined".to_string())
            } else {
                None
            },
            jaccard: None,
            reopen_eligible,
        }
    }

    fn candidate(ids: &[&str]) -> LedgerCandidate {
        LedgerCandidate {
            member_entry_ids: ids.iter().map(|id| (*id).to_string()).collect(),
        }
    }

    // --- LedgerError: the pinned Display strings -------------------------

    #[test]
    fn verdict_misalignment_renders_the_pinned_display() {
        assert_eq!(
            LedgerError::VerdictMisalignment {
                expected: 3,
                got: 1,
            }
            .to_string(),
            "somnus: ledger verdict at position 1 claims index 3"
        );
    }

    #[test]
    fn verdict_count_mismatch_renders_the_pinned_display() {
        assert_eq!(
            LedgerError::VerdictCountMismatch {
                pockets: 2,
                verdicts: 1,
            }
            .to_string(),
            "somnus: ledger returned 1 verdicts for 2 pockets"
        );
    }

    #[test]
    fn source_renders_the_pinned_display() {
        // [`HttpClusterLedger`] constructs this variant for every transport,
        // status, timeout, and parse failure; the rendering is pinned here
        // so the run-report line stays byte-stable.
        assert_eq!(
            LedgerError::Source {
                reason: "connection refused".to_string(),
            }
            .to_string(),
            "somnus: ledger source error: connection refused"
        );
    }

    // --- InMemoryLedger: the subset-containment and reopen rules ----------

    #[tokio::test]
    async fn stored_decline_matches_a_subset_candidate_with_the_pinned_verdict() {
        let ledger = InMemoryLedger::new();
        ledger
            .record_decline("demo-project", &["a".to_string(), "b".to_string()])
            .await
            .expect("record");

        let verdicts = ledger
            .match_candidates(
                "demo-project",
                &[
                    candidate(&["a", "b"]),
                    candidate(&["a", "b", "c", "d"]),
                    candidate(&["a", "b", "c"]),
                    candidate(&["x", "y"]),
                ],
            )
            .await
            .expect("match");

        assert_eq!(verdicts.len(), 4);
        // {a, b} → matched, NOT reopen-eligible (same cardinality).
        assert_eq!(
            verdicts[0],
            CandidateVerdict {
                index: 0,
                matched: true,
                ledger_id: Some("decline-1".to_string()),
                status: Some("declined".to_string()),
                jaccard: None,
                reopen_eligible: false,
            }
        );
        // {a, b, c, d} → matched and reopen-eligible (4 >= 2 × 2).
        assert!(verdicts[1].matched);
        assert!(verdicts[1].reopen_eligible);
        assert_eq!(verdicts[1].ledger_id.as_deref(), Some("decline-1"));
        // {a, b, c} → matched but NOT reopen-eligible (3 < 4).
        assert!(verdicts[2].matched);
        assert!(!verdicts[2].reopen_eligible);
        // {x, y} → no overlap, no match, every Option stays None.
        assert_eq!(
            verdicts[3],
            CandidateVerdict {
                index: 3,
                matched: false,
                ledger_id: None,
                status: None,
                jaccard: None,
                reopen_eligible: false,
            }
        );
    }

    #[tokio::test]
    async fn matching_is_scoped_per_project_and_ids_are_ordered_for_inspection() {
        let ledger = InMemoryLedger::new();
        ledger
            .record_decline("demo-project", &["a".to_string(), "b".to_string()])
            .await
            .expect("record");
        // A different project has no declines, so nothing matches there.
        let verdicts = ledger
            .match_candidates("other-project", &[candidate(&["a", "b"])])
            .await
            .expect("match");
        assert!(!verdicts[0].matched);
        // The inspection helper answers insertion order, ids sorted.
        assert_eq!(
            ledger.recorded_declines("demo-project"),
            vec![vec!["a".to_string(), "b".to_string(),]]
        );
        assert!(ledger.recorded_declines("other-project").is_empty());
    }

    // --- filter_pockets: fail closed BEFORE any element access ------------

    #[test]
    fn filter_drops_a_matched_pocket_that_is_not_reopen_eligible() {
        let pockets = fixture().pockets.clone();
        let verdicts = vec![verdict(0, true, false), verdict(1, false, false)];
        let (kept, declined) = filter_pockets(&pockets, &verdicts).expect("filter");
        assert_eq!(declined.len(), 1);
        assert_eq!(declined[0].member_entry_ids, pockets[0].member_entry_ids);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].member_entry_ids, pockets[1].member_entry_ids);
    }

    #[test]
    fn filter_keeps_a_matched_pocket_that_is_reopen_eligible() {
        let pockets = fixture().pockets.clone();
        let verdicts = vec![verdict(0, true, true), verdict(1, false, false)];
        let (kept, declined) = filter_pockets(&pockets, &verdicts).expect("filter");
        assert!(declined.is_empty());
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn filter_keeps_an_unmatched_pocket() {
        let pockets = fixture().pockets.clone();
        let (kept, declined) = filter_pockets(
            &pockets,
            &[verdict(0, false, false), verdict(1, false, false)],
        )
        .expect("filter");
        assert!(declined.is_empty());
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn filter_fails_closed_on_a_shorter_verdict_vector() {
        let pockets = fixture().pockets.clone();
        let error = filter_pockets(&pockets, &[verdict(0, false, false)])
            .expect_err("a short verdict vector must never be indexed");
        assert_eq!(
            error,
            LedgerError::VerdictCountMismatch {
                pockets: 2,
                verdicts: 1,
            }
        );
        assert_eq!(
            error.to_string(),
            "somnus: ledger returned 1 verdicts for 2 pockets"
        );
    }

    #[test]
    fn filter_fails_closed_on_a_longer_verdict_vector() {
        let pockets = fixture().pockets.clone();
        let error = filter_pockets(
            &pockets,
            &[
                verdict(0, false, false),
                verdict(1, false, false),
                verdict(2, false, false),
            ],
        )
        .expect_err("a long verdict vector is a count mismatch");
        assert_eq!(
            error,
            LedgerError::VerdictCountMismatch {
                pockets: 2,
                verdicts: 3,
            }
        );
    }

    #[test]
    fn filter_fails_closed_on_an_index_misalignment() {
        let pockets = fixture().pockets.clone();
        let error = filter_pockets(
            &pockets,
            &[verdict(5, false, false), verdict(1, false, false)],
        )
        .expect_err("a verdict claiming the wrong index is refused");
        assert_eq!(
            error,
            LedgerError::VerdictMisalignment {
                expected: 5,
                got: 0,
            }
        );
        assert_eq!(
            error.to_string(),
            "somnus: ledger verdict at position 0 claims index 5"
        );
    }

    #[test]
    fn filter_passes_the_e2e_pockets_through_a_declined_verdict() {
        // The exact e2e shape: P2 ([kb-10004, kb-10005]) was declined on a
        // previous night and is not yet reopen-eligible, so P1 is kept and
        // P2 is declined.
        let pockets = fixture().pockets.clone();
        let verdicts = vec![
            CandidateVerdict {
                index: 0,
                matched: false,
                ledger_id: None,
                status: None,
                jaccard: None,
                reopen_eligible: false,
            },
            CandidateVerdict {
                index: 1,
                matched: true,
                ledger_id: Some("decline-1".to_string()),
                status: Some("declined".to_string()),
                jaccard: None,
                reopen_eligible: false,
            },
        ];
        let (kept, declined) = filter_pockets(&pockets, &verdicts).expect("filter");
        assert_eq!(kept.len(), 1);
        assert_eq!(
            kept[0].member_entry_ids,
            vec![
                "kb-10001".to_string(),
                "kb-10002".to_string(),
                "kb-10003".to_string(),
            ]
        );
        assert_eq!(
            declined[0].member_entry_ids,
            vec!["kb-10004".to_string(), "kb-10005".to_string()]
        );
    }

    // --- the transport: wiremock, one request per call, the pinned map ---

    /// The pinned 200 match envelope for the two fixture pockets: P1
    /// unmatched, P2 matched by the stored `decline-1`.
    const ENVELOPE_P1_UNMATCHED: &str = "{\"results\":[{\"index\":0,\"matched\":false,\"ledger_id\":null,\"status\":null,\"jaccard\":null,\"reopen_eligible\":false},{\"index\":1,\"matched\":true,\"ledger_id\":\"decline-1\",\"status\":\"declined\",\"jaccard\":0.75,\"reopen_eligible\":false}]}";

    /// The pinned 200 match envelope with the verdicts swapped: P1 matched,
    /// P2 unmatched.
    const ENVELOPE_P1_MATCHED: &str = "{\"results\":[{\"index\":0,\"matched\":true,\"ledger_id\":\"decline-1\",\"status\":\"declined\",\"jaccard\":0.75,\"reopen_eligible\":false},{\"index\":1,\"matched\":false,\"ledger_id\":null,\"status\":null,\"jaccard\":null,\"reopen_eligible\":false}]}";

    mod transport {
        use super::*;
        use crate::loop_input::INVALID_PROJECT_REF_MSG;
        use wiremock::matchers::{body_string, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        /// The two fixture pockets as candidates.
        fn candidates() -> Vec<LedgerCandidate> {
            fixture()
                .pockets
                .iter()
                .map(|pocket| LedgerCandidate {
                    member_entry_ids: pocket.member_entry_ids.clone(),
                })
                .collect()
        }

        async fn mount_match(server: &MockServer, status: u16, body: &str) {
            Mock::given(method("POST"))
                .and(path("/api/kb/cluster-ledger/match"))
                .respond_with(ResponseTemplate::new(status).set_body_string(body))
                .mount(server)
                .await;
        }

        async fn mount_decline(server: &MockServer, status: u16, body: &str) {
            Mock::given(method("POST"))
                .and(path("/api/kb/cluster-ledger/decline"))
                .respond_with(ResponseTemplate::new(status).set_body_string(body))
                .mount(server)
                .await;
        }

        #[test]
        fn debug_never_renders_the_token() {
            let ledger =
                HttpClusterLedger::new("http://kb.invalid".to_string(), "secret".to_string());
            assert_eq!(
                format!("{ledger:?}"),
                "HttpClusterLedger { base_url: \"http://kb.invalid\", .. }"
            );
        }

        #[tokio::test]
        async fn match_happy_path_makes_exactly_one_pinned_request() {
            let server = MockServer::start().await;
            mount_match(&server, 200, ENVELOPE_P1_UNMATCHED).await;
            let ledger = HttpClusterLedger::new(server.uri(), "kb-token".to_string());

            let verdicts = ledger
                .match_candidates("demo-project", &candidates())
                .await
                .expect("match");
            assert_eq!(verdicts.len(), 2);

            let received = server.received_requests().await.expect("captured");
            assert_eq!(received.len(), 1, "no other route and no retry");
            let request = &received[0];
            assert_eq!(request.method.as_str(), "POST");
            assert_eq!(request.url.path(), "/api/kb/cluster-ledger/match");
            assert_eq!(
                request
                    .headers
                    .get("authorization")
                    .and_then(|value| value.to_str().ok()),
                Some("Bearer kb-token")
            );
            assert_eq!(
                request
                    .headers
                    .get("content-type")
                    .and_then(|value| value.to_str().ok()),
                Some("application/json")
            );
            // Byte-equal, candidates verbatim, no extra fields.
            assert_eq!(
                std::str::from_utf8(&request.body).expect("utf8"),
                "{\"project_ref\":\"demo-project\",\"candidates\":[{\"member_entry_ids\":[\"kb-10001\",\"kb-10002\",\"kb-10003\"]},{\"member_entry_ids\":[\"kb-10004\",\"kb-10005\"]}]}"
            );
        }

        #[tokio::test]
        async fn a_full_200_body_parses_to_the_pinned_verdicts() {
            let server = MockServer::start().await;
            mount_match(
                &server,
                200,
                "{\"results\":[{\"index\":0,\"matched\":true,\"ledger_id\":\"decline-1\",\"status\":\"declined\",\"jaccard\":0.75,\"reopen_eligible\":false},{\"index\":1,\"matched\":false,\"ledger_id\":null,\"status\":null,\"jaccard\":null,\"reopen_eligible\":false}]}",
            )
            .await;
            let ledger = HttpClusterLedger::new(server.uri(), "kb-token".to_string());

            let verdicts = ledger
                .match_candidates("demo-project", &candidates())
                .await
                .expect("match");
            assert_eq!(
                verdicts[0],
                CandidateVerdict {
                    index: 0,
                    matched: true,
                    ledger_id: Some("decline-1".to_string()),
                    status: Some("declined".to_string()),
                    jaccard: Some(0.75),
                    reopen_eligible: false,
                }
            );
            assert_eq!(
                verdicts[1],
                CandidateVerdict {
                    index: 1,
                    matched: false,
                    ledger_id: None,
                    status: None,
                    jaccard: None,
                    reopen_eligible: false,
                }
            );
        }

        #[tokio::test]
        async fn a_200_body_with_absent_keys_parses_to_none_options() {
            // Absence (not just explicit null) of the optional keys yields
            // `None` — serde's default behavior for absent `Option` fields.
            let server = MockServer::start().await;
            mount_match(
                &server,
                200,
                "{\"results\":[{\"index\":0,\"matched\":false,\"reopen_eligible\":false}]}",
            )
            .await;
            let ledger = HttpClusterLedger::new(server.uri(), "kb-token".to_string());

            let verdicts = ledger
                .match_candidates("demo-project", &candidates())
                .await
                .expect("match");
            assert_eq!(
                verdicts,
                vec![CandidateVerdict {
                    index: 0,
                    matched: false,
                    ledger_id: None,
                    status: None,
                    jaccard: None,
                    reopen_eligible: false,
                }]
            );
        }

        #[tokio::test]
        async fn a_200_body_round_trips_through_the_unchanged_filter() {
            let server = MockServer::start().await;
            mount_match(&server, 200, ENVELOPE_P1_UNMATCHED).await;
            let ledger = HttpClusterLedger::new(server.uri(), "kb-token".to_string());

            let verdicts = ledger
                .match_candidates("demo-project", &candidates())
                .await
                .expect("match");
            let (kept, declined) = filter_pockets(&fixture().pockets, &verdicts).expect("filter");
            assert_eq!(
                kept[0].member_entry_ids,
                vec![
                    "kb-10001".to_string(),
                    "kb-10002".to_string(),
                    "kb-10003".to_string(),
                ]
            );
            assert_eq!(
                declined[0].member_entry_ids,
                vec!["kb-10004".to_string(), "kb-10005".to_string()]
            );
        }

        #[tokio::test]
        async fn the_match_status_map_is_terminal_with_one_request_each() {
            for (status, expected) in [
                (
                    404u16,
                    LedgerError::Source {
                        reason: format!("match: {NOT_FOUND_FORM}"),
                    },
                ),
                (
                    409,
                    LedgerError::Source {
                        reason: format!("match: {NOT_ELIGIBLE_FORM}"),
                    },
                ),
                (
                    401,
                    LedgerError::Source {
                        reason: format!("match: {UNAUTHORIZED_FORM}"),
                    },
                ),
                (
                    500,
                    LedgerError::Source {
                        reason: "match: unexpected HTTP status 500".to_string(),
                    },
                ),
            ] {
                let server = MockServer::start().await;
                mount_match(&server, status, "ignored body").await;
                let ledger = HttpClusterLedger::new(server.uri(), "kb-token".to_string());
                assert_eq!(
                    ledger.match_candidates("demo-project", &candidates()).await,
                    Err(expected),
                    "status {status}"
                );
                assert_eq!(
                    server.received_requests().await.expect("captured").len(),
                    1,
                    "status {status} is terminal: no retry, no escalation"
                );
            }
        }

        #[tokio::test]
        async fn a_200_with_a_garbage_body_maps_to_malformed_body() {
            let server = MockServer::start().await;
            mount_match(&server, 200, "not a ledger payload at all").await;
            let ledger = HttpClusterLedger::new(server.uri(), "kb-token".to_string());
            match ledger.match_candidates("demo-project", &candidates()).await {
                Err(LedgerError::Source { reason }) => {
                    assert!(
                        reason.starts_with("match: malformed body ("),
                        "reason was {reason}"
                    );
                    assert!(reason.len() > "match: malformed body (".len());
                }
                other => panic!("expected a Source error, got {other:?}"),
            }
            assert_eq!(server.received_requests().await.expect("captured").len(), 1);
        }

        #[tokio::test]
        async fn a_dropped_server_maps_to_a_terminal_transport_failure() {
            // The transport branch, not a status branch. The server is built
            // over an EXPLICIT listener (and is therefore a bare server, not
            // a pooled facade) so that dropping it tears the listener down
            // and the connect is refused — wiremock's default
            // `MockServer::start()` hands its listener back to a process
            // pool on drop, which keeps serving instead of refusing.
            let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind a port");
            let address = listener.local_addr().expect("local address");
            let server = MockServer::builder().listener(listener).start().await;
            mount_match(&server, 200, ENVELOPE_P1_UNMATCHED).await;
            let ledger = HttpClusterLedger::new(server.uri(), "kb-token".to_string());
            drop(server);
            // The shutdown is graceful, so wait for the listener to actually
            // close before asserting on the transport outcome.
            let mut closed = false;
            for _ in 0..200 {
                if std::net::TcpStream::connect_timeout(
                    &address,
                    std::time::Duration::from_millis(25),
                )
                .is_err()
                {
                    closed = true;
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            assert!(closed, "the dropped server's listener must close");
            match ledger.match_candidates("demo-project", &candidates()).await {
                Err(LedgerError::Source { reason }) => {
                    assert!(
                        reason.starts_with("match: transport failed: "),
                        "reason was {reason}"
                    );
                }
                other => panic!("expected a transport failure, got {other:?}"),
            }
        }

        #[tokio::test]
        async fn a_failed_body_read_maps_to_the_transport_failure_form() {
            // A one-shot raw TCP server that answers 200 with a
            // `content-length` far larger than the body it writes, then
            // drops the connection — so `send()` succeeds on the headers and
            // `response.text().await` fails. That failed body read maps to
            // the SAME terminal transport-failure form.
            let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind a port");
            let address = listener.local_addr().expect("local address");
            let handle = std::thread::spawn(move || {
                if let Ok((mut stream, _)) = listener.accept() {
                    use std::io::Read;
                    let mut buf = [0u8; 4096];
                    let _ = stream.read(&mut buf);
                    let _ = std::io::Write::write_all(
                        &mut stream,
                        b"HTTP/1.1 200 OK\r\ncontent-length: 128\r\n\r\nshort",
                    );
                }
            });
            let ledger =
                HttpClusterLedger::new(format!("http://{address}"), "kb-token".to_string());

            match ledger.match_candidates("demo-project", &candidates()).await {
                Err(LedgerError::Source { reason }) => {
                    assert!(
                        reason.starts_with("match: transport failed: "),
                        "reason was {reason}"
                    );
                }
                other => panic!("expected a transport failure, got {other:?}"),
            }
            handle.join().expect("the one-shot server thread");
        }

        #[tokio::test(start_paused = true)]
        async fn a_slow_response_times_out_at_the_pinned_bound() {
            // wiremock delays via a tokio sleep, so the PAUSED runtime
            // auto-advances into the timeout: the pinned bound expires in
            // simulated time, with no wall-clock wait. (Technique from
            // observer.rs's start_paused tests, combined with wiremock's
            // `set_delay` — new to this crate.)
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/api/kb/cluster-ledger/match"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_string(ENVELOPE_P1_UNMATCHED)
                        .set_delay(Duration::from_mins(1)),
                )
                .mount(&server)
                .await;
            let ledger = HttpClusterLedger::new(server.uri(), "kb-token".to_string());

            let start = tokio::time::Instant::now();
            let outcome = ledger.match_candidates("demo-project", &candidates()).await;
            assert_eq!(start.elapsed(), LEDGER_HTTP_BOUND);
            assert_eq!(
                outcome,
                Err(LedgerError::Source {
                    reason: "match: timed out after 10s".to_string(),
                })
            );
        }

        #[tokio::test]
        async fn an_invalid_ref_never_reaches_the_wire_for_either_method() {
            let server = MockServer::start().await;
            mount_match(&server, 200, ENVELOPE_P1_UNMATCHED).await;
            mount_decline(&server, 201, "{\"ledger_id\":\"decline-1\"}").await;
            let ledger = HttpClusterLedger::new(server.uri(), "kb-token".to_string());
            for bad_ref in ["", "de mo", "demo/project"] {
                assert_eq!(
                    ledger.match_candidates(bad_ref, &candidates()).await,
                    Err(LedgerError::Source {
                        reason: INVALID_PROJECT_REF_MSG.to_string(),
                    })
                );
                assert_eq!(
                    ledger
                        .record_decline(bad_ref, &["a".to_string(), "b".to_string()])
                        .await,
                    Err(LedgerError::Source {
                        reason: INVALID_PROJECT_REF_MSG.to_string(),
                    })
                );
            }
            assert!(
                server
                    .received_requests()
                    .await
                    .expect("captured")
                    .is_empty(),
                "no request may exist for an invalid ref"
            );
        }

        #[tokio::test]
        async fn a_missing_token_never_reaches_the_wire() {
            let server = MockServer::start().await;
            mount_match(&server, 200, ENVELOPE_P1_UNMATCHED).await;
            let home = tempfile::tempdir().expect("tempdir");
            let ledger = HttpClusterLedger::from_home(server.uri(), home.path());
            assert_eq!(
                ledger.match_candidates("demo-project", &candidates()).await,
                Err(LedgerError::Source {
                    reason: TokenError::Missing.to_string(),
                })
            );
            assert!(
                server
                    .received_requests()
                    .await
                    .expect("captured")
                    .is_empty(),
                "no HTTP call may be made without a readable token"
            );
        }

        #[tokio::test]
        async fn an_empty_token_never_reaches_the_wire() {
            let server = MockServer::start().await;
            mount_match(&server, 200, ENVELOPE_P1_UNMATCHED).await;
            let home = tempfile::tempdir().expect("tempdir");
            let token_file = crate::loop_input::token_path(home.path());
            std::fs::create_dir_all(token_file.parent().expect("parent")).expect("create dirs");
            std::fs::write(&token_file, b"   \n\t").expect("write a whitespace-only token");
            let ledger = HttpClusterLedger::from_home(server.uri(), home.path());
            assert_eq!(
                ledger.match_candidates("demo-project", &candidates()).await,
                Err(LedgerError::Source {
                    reason: TokenError::Empty.to_string(),
                })
            );
            assert!(
                server
                    .received_requests()
                    .await
                    .expect("captured")
                    .is_empty(),
                "no HTTP call may be made with an empty token"
            );
        }

        #[tokio::test]
        async fn the_empty_boundary_short_circuits_with_zero_requests() {
            let server = MockServer::start().await;
            mount_match(&server, 200, ENVELOPE_P1_UNMATCHED).await;
            mount_decline(&server, 201, "{\"ledger_id\":\"decline-1\"}").await;
            let ledger = HttpClusterLedger::new(server.uri(), "kb-token".to_string());
            // Zero candidates means zero verdicts; zero members means no
            // decline. No empty-batch request shape is invented.
            assert_eq!(
                ledger.match_candidates("demo-project", &[]).await,
                Ok(vec![])
            );
            assert_eq!(ledger.record_decline("demo-project", &[]).await, Ok(()));
            assert!(
                server
                    .received_requests()
                    .await
                    .expect("captured")
                    .is_empty(),
                "the empty boundary must never reach the wire"
            );
        }

        #[tokio::test]
        async fn decline_happy_path_makes_exactly_one_pinned_request() {
            let server = MockServer::start().await;
            mount_decline(&server, 201, "{\"ledger_id\":\"decline-42\"}").await;
            let ledger = HttpClusterLedger::new(server.uri(), "kb-token".to_string());

            ledger
                .record_decline("demo-project", &["a".to_string(), "b".to_string()])
                .await
                .expect("decline");

            let received = server.received_requests().await.expect("captured");
            assert_eq!(received.len(), 1);
            let request = &received[0];
            assert_eq!(request.method.as_str(), "POST");
            assert_eq!(request.url.path(), "/api/kb/cluster-ledger/decline");
            assert_eq!(
                request
                    .headers
                    .get("authorization")
                    .and_then(|value| value.to_str().ok()),
                Some("Bearer kb-token")
            );
            assert_eq!(
                request
                    .headers
                    .get("content-type")
                    .and_then(|value| value.to_str().ok()),
                Some("application/json")
            );
            assert_eq!(
                std::str::from_utf8(&request.body).expect("utf8"),
                "{\"project_ref\":\"demo-project\",\"member_entry_ids\":[\"a\",\"b\"],\"reason\":\"real cluster with nothing chunky to point at\"}"
            );
        }

        #[tokio::test]
        async fn the_decline_status_map_is_terminal_with_one_request_each() {
            for (status, expected) in [
                (
                    404u16,
                    LedgerError::Source {
                        reason: format!("decline: {NOT_FOUND_FORM}"),
                    },
                ),
                (
                    409,
                    LedgerError::Source {
                        reason: format!("decline: {NOT_ELIGIBLE_FORM}"),
                    },
                ),
                (
                    401,
                    LedgerError::Source {
                        reason: format!("decline: {UNAUTHORIZED_FORM}"),
                    },
                ),
                (
                    500,
                    LedgerError::Source {
                        reason: "decline: unexpected HTTP status 500".to_string(),
                    },
                ),
            ] {
                let server = MockServer::start().await;
                mount_decline(&server, status, "ignored body").await;
                let ledger = HttpClusterLedger::new(server.uri(), "kb-token".to_string());
                assert_eq!(
                    ledger
                        .record_decline("demo-project", &["a".to_string()])
                        .await,
                    Err(expected),
                    "status {status}"
                );
                assert_eq!(
                    server.received_requests().await.expect("captured").len(),
                    1,
                    "status {status} is terminal: no retry, no escalation"
                );
            }
        }

        #[tokio::test]
        async fn a_201_with_a_garbage_body_maps_to_malformed_body() {
            let server = MockServer::start().await;
            mount_decline(&server, 201, "not a ledger id at all").await;
            let ledger = HttpClusterLedger::new(server.uri(), "kb-token".to_string());
            match ledger
                .record_decline("demo-project", &["a".to_string()])
                .await
            {
                Err(LedgerError::Source { reason }) => {
                    assert!(
                        reason.starts_with("decline: malformed body ("),
                        "reason was {reason}"
                    );
                }
                other => panic!("expected a Source error, got {other:?}"),
            }
            assert_eq!(server.received_requests().await.expect("captured").len(), 1);
        }

        #[tokio::test]
        async fn a_truncated_results_array_is_handed_through_and_fails_the_filter() {
            // The transport performs NO count or index validation of its
            // own — `filter_pockets` stays the single gate.
            let server = MockServer::start().await;
            mount_match(
                &server,
                200,
                "{\"results\":[{\"index\":0,\"matched\":false,\"reopen_eligible\":false}]}",
            )
            .await;
            let ledger = HttpClusterLedger::new(server.uri(), "kb-token".to_string());

            let verdicts = ledger
                .match_candidates("demo-project", &candidates())
                .await
                .expect("the transport hands the verdicts through unchanged");
            assert_eq!(verdicts.len(), 1);

            let error =
                filter_pockets(&fixture().pockets, &verdicts).expect_err("the filter is the gate");
            assert_eq!(
                error,
                LedgerError::VerdictCountMismatch {
                    pockets: 2,
                    verdicts: 1,
                }
            );
            assert_eq!(
                error.to_string(),
                "somnus: ledger returned 1 verdicts for 2 pockets"
            );
        }

        #[tokio::test]
        async fn the_match_request_is_mounted_against_its_exact_body() {
            // A belt-and-braces pin: the route matcher itself demands the
            // byte-pinned body, so a drifted serialization fails the request
            // outright rather than merely differing in a captured body.
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/api/kb/cluster-ledger/match"))
                .and(body_string(
                    "{\"project_ref\":\"demo-project\",\"candidates\":[{\"member_entry_ids\":[\"kb-10001\",\"kb-10002\",\"kb-10003\"]},{\"member_entry_ids\":[\"kb-10004\",\"kb-10005\"]}]}",
                ))
                .respond_with(ResponseTemplate::new(200).set_body_string(ENVELOPE_P1_MATCHED))
                .mount(&server)
                .await;
            let ledger = HttpClusterLedger::new(server.uri(), "kb-token".to_string());
            let verdicts = ledger
                .match_candidates("demo-project", &candidates())
                .await
                .expect("match");
            assert_eq!(verdicts.len(), 2);
            assert!(verdicts[0].matched, "P1 matched in the swapped envelope");
            assert!(!verdicts[1].matched);
            assert_eq!(server.received_requests().await.expect("captured").len(), 1);
        }
    }
}
