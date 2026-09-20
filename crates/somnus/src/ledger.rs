//! The cluster/decline ledger seam and the decline filter (the
//! expensive-if-missed part of the pipeline).
//!
//! The ledger table exists on the KB side (`map_cluster_ledger`, both
//! backends) but its HTTP endpoints are NOT written yet — nothing in this
//! repo implements `/api/kb/cluster-ledger` — so this module contains the
//! trait designed to the requested batch-match shape, the pure
//! [`filter_pockets`], and EXACTLY ONE implementation: the in-memory
//! [`InMemoryLedger`] test stand-in. There is no reqwest usage and no HTTP
//! code here; wiring the real transport is a follow-up.
//!
//! Why the ledger exists at all: a declined cluster is a real cluster whose
//! members have nothing chunky to point at, and paying Sonnet to re-propose
//! it every night is both waste and noise. A decline is permanent with one
//! escape — reopen when the member set doubles.
//!
//! Layering (pinned): this module references only [`crate::loop_input`]
//! types.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use async_trait::async_trait;

use crate::loop_input::LoopInputPocket;

/// One candidate cluster for the ledger's batch-match call: the member-id
/// set, and NOTHING beyond it — matching is keyed on member-set overlap, not
/// label, because labels drift between runs.
#[derive(Debug, Clone, PartialEq, Eq)]
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
/// verdict's position by [`filter_pockets`].
#[derive(Debug, Clone, PartialEq)]
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
    /// decline (the stored set doubled or more).
    pub reopen_eligible: bool,
}

/// Why a ledger interaction failed. [`LedgerError::VerdictMisalignment`] and
/// [`LedgerError::VerdictCountMismatch`] are constructed ONLY by
/// [`filter_pockets`] — a malformed verdict vector must never panic, it must
/// fail closed. [`LedgerError::Source`] is reserved for the future HTTP
/// transport and is constructed by nothing this cut.
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
/// declines, and record a new decline. The real transport is the KB's
/// cluster-ledger endpoints (unwritten); [`InMemoryLedger`] is the only
/// implementation this cut.
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
        // Reserved for the future transport; constructed by nothing this
        // cut, so the variant's rendering is pinned here instead.
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
}
