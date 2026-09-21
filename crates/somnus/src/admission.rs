//! Which clusters are fit to become maps.
//!
//! These rules discriminate QUALITY. They once also arbitrated scarcity —
//! the server capped a project at one new map per night, so something had to
//! choose which cluster spent it, and left to itself that was whichever
//! `create_map` was submitted first. Array order, which is to say chance.
//!
//! **The cap is gone** (server-side, 2026-09-21). Jason's reasoning: a cap
//! only ever bounded IRREVERSIBLE damage, and maps are deletable, so it
//! protected nothing — while costing plenty, since rung 2 already pays for
//! one inference per cluster and a capped run discarded all but one of them.
//! A project needing fourteen maps would have taken fourteen nights to become
//! navigable, and rationing maps rations exactly the agent orientation they
//! exist to provide.
//!
//! So every cluster passing these rules now mints. What survives from the
//! scarcity era is the ranking — rung 1 still emits in merit order — but that
//! is a spend-ordering concern now rather than a correctness one, and these
//! rules stayed because they never had anything to do with the cap: they
//! judge whether a cluster is a subject, not whether there is room for it.
//!
//! Layering (pinned): this module references [`crate::loop_input`] only.

use std::collections::HashMap;

use crate::loop_input::Cluster;

/// The fewest entries a cluster must hold before somnus will mint a NEW map
/// for it.
///
/// Set from measurement plus an asymmetry, not from taste. Across the 27 live
/// hand-authored maps the body-pointer counts run min 1, median 5, max 20 — so
/// humans do sometimes write a one-pointer map, but as a deliberate
/// set-completing choice made with the whole map set in view, which a machine
/// writing one map a night does not have.
///
/// What settles the number above the human minimum is that the two errors are
/// not symmetric. **A bad map cannot be withdrawn**: the KB's write API
/// refuses to deactivate a mental map, because deactivating one strips its
/// outbound edges and orphans every detail entry it pointed at. A missed small
/// cluster costs a `propose_gap` that a human can act on. Permanent damage on
/// one side, a recoverable note on the other.
pub const MIN_CLUSTER_MEMBERS_FOR_NEW_MAP: usize = 3;

/// The characters that turn a label into a union of subjects rather than one
/// subject. An ampersand or slash joins two nouns; a comma lists them.
const UNION_MARKERS: [char; 3] = ['&', '/', ','];

/// Whether a cluster may mint a map.
///
/// A [`Verdict::Refused`] cluster is structurally unfit, so its grouping
/// becomes a `propose_gap` and the pipeline records a decline — it stops
/// being re-proposed every night, and because the ledger keys on the exact
/// member set, a later re-split of the same entries is a different set and is
/// NOT suppressed.
///
/// There is deliberately no third "fit but not this time" verdict. One
/// existed while the cap did, and the distinction was load-bearing then: a
/// cluster that merely lost the night's single slot had to record NOTHING,
/// because declining it would have suppressed precisely the good clusters
/// that were waiting their turn while the junk got declined correctly. With
/// the cap gone there is no queue to wait in, so the case cannot arise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// This cluster may mint a map.
    Admitted,
    /// Structurally unfit to mint: decline it and record it.
    Refused {
        /// Plain-words prose naming which rule refused it. Crosses the same
        /// wire as model-written prose, so it obeys the map-lint's rules.
        reason: String,
    },
}

/// Rule 1 — the size floor.
#[must_use]
pub fn meets_size_floor(cluster: &Cluster) -> bool {
    cluster.member_entry_ids.len() >= MIN_CLUSTER_MEMBERS_FOR_NEW_MAP
}

/// Rule 2 — native majority: most of a cluster's members must belong to
/// exactly one cluster.
///
/// Overlap between clusters is expected and correct, so this is not a ban on
/// sharing. It is a test for a cluster that has no members of its OWN: one
/// assembled almost entirely out of entries that sit more naturally
/// elsewhere. On the first live run this deletes a three-member cluster two
/// of whose members were borrowed from another cluster and whose third
/// belonged with a third — a phantom subject, killed mechanically without
/// reading a single entry body.
/// **A cluster that is itself unfit does not count as a home.** The
/// membership tally runs over clusters that passed the two intrinsic rules,
/// not over every cluster rung 1 emitted. Without that, a phantom drags its
/// own victim down with it: the cluster it borrowed from sees those members
/// as shared, loses its majority, and is refused too — so the loop would
/// decline the good cluster on the strength of the bad one's existence.
#[must_use]
pub fn has_native_majority(cluster: &Cluster, all: &[Cluster]) -> bool {
    let mut memberships: HashMap<&str, usize> = HashMap::new();
    for other in all.iter().filter(|other| refusal(other).is_none()) {
        for id in &other.member_entry_ids {
            *memberships.entry(id.as_str()).or_insert(0) += 1;
        }
    }
    let members = cluster.member_entry_ids.len();
    let native = cluster
        .member_entry_ids
        .iter()
        .filter(|id| memberships.get(id.as_str()).copied().unwrap_or(0) <= 1)
        .count();
    native * 2 > members
}

/// Rule 3 — label shape: one subject per label.
///
/// A label that joins nouns with an ampersand or a slash, or lists them with
/// commas, is the model reporting a failed carve in its own words: it named
/// the union instead of re-splitting. The rule costs nothing to evaluate and
/// flagged the three worst clusters of the first live run on the label alone.
#[must_use]
pub fn label_names_one_subject(label: &str) -> bool {
    !label.contains(UNION_MARKERS)
}

/// The refusal prose for each rule, in plain words.
///
/// Written the way the map-lint demands of the model — whole numbers, no
/// backticks, no quotes, no dotted abbreviations — because this text crosses
/// the same wire model-written prose does.
fn refusal(cluster: &Cluster) -> Option<String> {
    if !meets_size_floor(cluster) {
        return Some(format!(
            "The cluster holds {} entries, fewer than the {MIN_CLUSTER_MEMBERS_FOR_NEW_MAP} a new map needs, so the grouping is recorded as a gap rather than mapped.",
            cluster.member_entry_ids.len()
        ));
    }
    if !label_names_one_subject(&cluster.label) {
        return Some(
            "The label joins several subjects rather than naming one, so the grouping is recorded as a gap for a later re-split rather than mapped."
                .to_string(),
        );
    }
    None
}

/// Judge every cluster, returning one verdict per cluster in the same order.
///
/// Every cluster that survives all three rules is admitted. Rung 1's merit
/// order still matters — it decides which maps are composed and submitted
/// first, so a run that runs out of token budget part-way spends what it had
/// on the clusters most worth mapping — but it no longer decides which single
/// cluster is allowed to exist.
#[must_use]
pub fn judge(clusters: &[Cluster]) -> Vec<Verdict> {
    clusters
        .iter()
        .map(|cluster| {
            if let Some(reason) = refusal(cluster) {
                return Verdict::Refused { reason };
            }
            if has_native_majority(cluster, clusters) {
                Verdict::Admitted
            } else {
                Verdict::Refused {
                    reason: "Most of the entries in this cluster belong to another cluster too, so it has no subject of its own and the grouping is recorded as a gap rather than mapped."
                        .to_string(),
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cluster(label: &str, members: &[&str]) -> Cluster {
        Cluster {
            label: label.to_string(),
            member_entry_ids: members.iter().map(|id| (*id).to_string()).collect(),
            owning_map_id: None,
            merit_reason: None,
        }
    }

    /// The cap is gone: a project needing many maps gets many maps in one
    /// run, rather than one a night for as many nights as it has subjects.
    #[test]
    fn every_fit_cluster_is_admitted() {
        let clusters = vec![
            cluster("Wireguard", &["kb-1", "kb-2", "kb-3"]),
            cluster("Lightroom", &["kb-4", "kb-5", "kb-6"]),
            cluster("Flickr", &["kb-7", "kb-8", "kb-9"]),
        ];
        assert_eq!(
            judge(&clusters),
            vec![Verdict::Admitted, Verdict::Admitted, Verdict::Admitted]
        );
    }

    #[test]
    fn a_refusal_does_not_affect_its_neighbours() {
        let clusters = vec![
            cluster("Backend & Frontend", &["kb-1", "kb-2", "kb-3"]),
            cluster("Lightroom", &["kb-4", "kb-5", "kb-6"]),
        ];
        let verdicts = judge(&clusters);
        assert!(matches!(verdicts[0], Verdict::Refused { .. }));
        assert_eq!(verdicts[1], Verdict::Admitted);
    }

    #[test]
    fn a_thin_cluster_is_refused_by_the_floor() {
        let clusters = vec![cluster("Testing", &["kb-1"])];
        let Verdict::Refused { reason } = &judge(&clusters)[0] else {
            panic!("a one-entry cluster is not a map");
        };
        assert!(reason.contains("fewer than the 3"), "{reason}");
    }

    #[test]
    fn a_union_label_is_refused_whatever_joins_it() {
        for label in [
            "Backend & Frontend",
            "Discovery / Recommendation Engine",
            "Rate Limiting, Quotas, Incidents",
        ] {
            let clusters = vec![cluster(label, &["kb-1", "kb-2", "kb-3"])];
            assert!(
                matches!(judge(&clusters)[0], Verdict::Refused { .. }),
                "{label} names a union, not a subject"
            );
        }
    }

    /// The phantom-cluster case: every member is also in another cluster, so
    /// this one has no subject of its own.
    #[test]
    fn a_cluster_built_from_borrowed_members_is_refused() {
        let clusters = vec![
            cluster("Stats Subsystem", &["kb-1", "kb-2", "kb-3", "kb-4", "kb-9"]),
            cluster("Discovery", &["kb-5", "kb-6", "kb-7"]),
            cluster("Flickr Quirks", &["kb-1", "kb-2", "kb-5"]),
        ];
        let verdicts = judge(&clusters);
        assert_eq!(verdicts[0], Verdict::Admitted);
        let Verdict::Refused { reason } = &verdicts[2] else {
            panic!("a cluster of borrowed members is a phantom");
        };
        assert!(reason.contains("belong to another cluster"), "{reason}");
    }

    /// The drag case, and the reason the tally skips unfit clusters. The
    /// phantom here borrows two of the owner's three members; counted
    /// naively the OWNER loses its majority and is refused on the strength of
    /// the phantom's existence, which would decline a good grouping.
    #[test]
    fn a_phantom_does_not_drag_down_the_cluster_it_borrowed_from() {
        let clusters = vec![
            cluster("Wireguard", &["kb-1", "kb-2", "kb-3"]),
            cluster("Borrowed & Union", &["kb-1", "kb-2", "kb-8"]),
        ];
        let verdicts = judge(&clusters);
        assert_eq!(
            verdicts[0],
            Verdict::Admitted,
            "the owner keeps its majority once the union-labelled phantom is discounted"
        );
        assert!(matches!(verdicts[1], Verdict::Refused { .. }));
    }

    /// Overlap is expected and correct — the rule is a majority test, not a
    /// ban on sharing. Two shared members out of five still leaves three of
    /// its own, so the broad cluster stands; the narrow one that is mostly a
    /// subset of it does not.
    #[test]
    fn a_broad_cluster_survives_overlap_that_sinks_a_near_subset() {
        let clusters = vec![
            cluster("Wireguard", &["kb-1", "kb-2", "kb-3"]),
            cluster("Networking", &["kb-1", "kb-2", "kb-7", "kb-8", "kb-9"]),
        ];
        let verdicts = judge(&clusters);
        assert!(
            matches!(verdicts[0], Verdict::Refused { .. }),
            "two of three members shared is not a majority of its own"
        );
        assert_eq!(verdicts[1], Verdict::Admitted);
    }

    /// An exact half is not a majority: a four-member cluster sharing two is
    /// as much someone else's as its own, so it does not mint.
    #[test]
    fn an_exact_half_is_not_a_majority() {
        let clusters = vec![
            cluster("Owner", &["kb-1", "kb-2", "kb-9", "kb-8"]),
            cluster("Half Borrowed", &["kb-1", "kb-2", "kb-3", "kb-4"]),
        ];
        assert!(matches!(judge(&clusters)[1], Verdict::Refused { .. }));
    }

    #[test]
    fn every_refusal_reason_obeys_the_map_lint_rules() {
        let clusters = vec![
            cluster("Thin", &["kb-1"]),
            cluster("A & B", &["kb-1", "kb-2", "kb-3"]),
            cluster("Owner", &["kb-4", "kb-5", "kb-6"]),
            cluster("Borrowed", &["kb-4", "kb-5", "kb-6"]),
        ];
        for verdict in judge(&clusters) {
            let Verdict::Refused { reason } = verdict else {
                continue;
            };
            assert!(!reason.contains('`'), "{reason}");
            assert!(!reason.contains('"'), "{reason}");
            assert!(!reason.contains('/'), "{reason}");
            assert!(!reason.contains('_'), "{reason}");
            assert!(!reason.contains("e.g."), "{reason}");
        }
    }
}
