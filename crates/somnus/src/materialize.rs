//! Rung 3: code composes the map bodies from the parsed ops.
//!
//! The model never writes a map body — there is no `rewrite_body` and no
//! `write_map` op — so every byte of a composed body is either code-composed
//! (the section skeletons and the pointer/gap line shapes), fixture-derived
//! (the existing map bodies, which are the read-modify-write substrate and
//! are never re-derived), or one of the two model strings (the orientation
//! prose and the per-pointer glosses). That is why a map is born lint-clean
//! by construction.
//!
//! The rendering grammar pinned here is somnus's OWN; the canonical kb
//! grammar stays an open question, and existing maps are never re-derived —
//! their bodies are edited in place, so a map written under any other
//! grammar survives this loop untouched.
//!
//! Layering (pinned): this module references [`crate::loop_input`] and
//! [`crate::ops`] only.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::admission::Verdict;
use crate::loop_input::{Cluster, LoopInput, LoopInputMap};
use crate::ops::Op;

/// The loop's author identity (provisional — the server's authorship-tier
/// field names are an open question). The authorship tier is enforced
/// against this constant: somnus may only edit maps it wrote.
pub const SOMNUS_WRITER_ID: &str = "somnus";

/// Structural cap: three eligible projects per night — the ONE cap with no
/// server analogue, so it is somnus's own worklist policy, enforced by the
/// nightly worklist slice ([`crate::worklist::select_first_projects`]) via
/// `take(3)`. The other two caps (one new map per project, three per KB,
/// both counted since UTC midnight) are SERVER-side at `POST /api/kb/map-op`;
/// somnus tracks neither.
pub const SOMNUS_MAX_PROJECTS_PER_NIGHT: u32 = 3;

/// The `Lives in` line's prefix, pinned so the value derivation and the
/// append path share one definition.
const LIVES_IN_PREFIX: &str = "Lives in ";

/// The detail-entries section header line.
const DETAIL_ENTRIES_HEADER: &str = "Detail entries:";

/// The not-yet-documented section header line.
const NOT_YET_HEADER: &str = "Not yet documented:";

/// The section separator of the somnus rendering grammar.
const SECTION_SEPARATOR: &str = "\n\n";

/// Why a body could not be composed. Every variant names the map or cluster
/// it refused to touch, so a refusal is traceable from the run report alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComposeError {
    /// The op targets a map the loop does not own (authorship tier).
    AuthorshipRefusal {
        /// The refused map.
        map_id: String,
    },
    /// A fresh map was requested with no pointers to list.
    EmptyPointerList {
        /// The fresh map that cannot exist pointerless.
        map_id: String,
    },
    /// The op targets a map that is neither in the loop-input maps nor the
    /// fresh map this run composes.
    UnknownMap {
        /// The unknown map id.
        map_id: String,
    },
    /// The target map's body has no `Detail entries:` section to append to.
    NoDetailSection {
        /// The map without a detail-entries section.
        map_id: String,
    },
    /// The cited gap line is not in the map's `Not yet documented:` section.
    GapNotFound {
        /// The map whose gap was cited.
        map_id: String,
        /// The gap text that was not found.
        gap_text: String,
    },
}

impl std::fmt::Display for ComposeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AuthorshipRefusal { map_id } => write!(
                f,
                "somnus: refusing to edit map {map_id}: not loop-owned (authorship tier; somnus only edits maps it wrote)"
            ),
            Self::EmptyPointerList { map_id } => write!(
                f,
                "somnus: refusing to compose map {map_id}: the create_map op set lists no pointers"
            ),
            Self::UnknownMap { map_id } => {
                write!(f, "somnus: op targets unknown map {map_id}")
            }
            Self::NoDetailSection { map_id } => write!(
                f,
                "somnus: map {map_id} has no `Detail entries:` section to append to"
            ),
            Self::GapNotFound { map_id, gap_text } => write!(
                f,
                "somnus: map {map_id} has no `Not yet documented:` line `- {gap_text}` to strike"
            ),
        }
    }
}

impl std::error::Error for ComposeError {}

/// Whether a map is loop-owned (the authorship tier): authored OR last
/// updated by [`SOMNUS_WRITER_ID`].
#[must_use]
pub fn is_loop_owned(map: &LoopInputMap) -> bool {
    map.contributor.as_deref() == Some(SOMNUS_WRITER_ID)
        || map.updated_by.as_deref() == Some(SOMNUS_WRITER_ID)
}

/// The map id the MODEL targets when pointing at the map its own op set is
/// creating. Deterministic so a cluster's `add_pointer` ops can name their
/// fresh map without a server round-trip.
///
/// Scoped to one cluster's op set and nothing wider: `cluster_id` is a string
/// the model invents, and rung 2 is one inference per cluster with no
/// knowledge of the others, so nothing stops every cluster in a run calling
/// its cluster `c1`.
#[must_use]
pub fn new_map_id(cluster_id: &str) -> String {
    format!("somnus-new-{cluster_id}")
}

/// The run-unique local id a fresh map is recorded under, before the server
/// assigns the real one.
///
/// The cluster INDEX is what makes it unique, and code owns that. This
/// mattered the moment the one-map-per-project cap was lifted: with several
/// maps composed in a run, ids derived from the model's `cluster_id` alone
/// collide as soon as two blind rung-2 calls both pick `c1` — and the
/// collision is silent, because the pairing of body to title is a lookup by
/// id, so the second map would have been created carrying the first map's
/// title. Under the cap only one create ever reached the server, so nothing
/// could expose it.
#[must_use]
pub fn fresh_map_id(cluster_index: usize, cluster_id: &str) -> String {
    format!("somnus-new-{cluster_index}-{cluster_id}")
}

/// How many members must share a directory before it may be named, when the
/// cluster spans two of them. One member is an anecdote, not a home.
const MIN_MEMBERS_PER_NAMED_DIRECTORY: usize = 2;

/// Derive the `Lives in` value for a cluster's fresh map from the cluster's
/// OWN members, or `None` when its members name no directory with a
/// plurality behind it.
///
/// `Lives in` is per-cluster, not per-project: the canonical line is
/// directory-level orientation for *that subject area*, so a project-level
/// value would be the wrong shape even where one exists. Each member's home
/// is the highest-ranked token the server extracted from its full body, and
/// the rule over those homes is:
///
/// 1. a majority of members share one home — name it;
/// 2. else a majority share one of the top two homes AND each of those two is
///    home to at least [`MIN_MEMBERS_PER_NAMED_DIRECTORY`] members — name
///    both, joined by the word `and`, which the map-lint accepts;
/// 3. else `None`.
///
/// **`None` means OMIT the line, not refuse the map** — a reversal of this
/// function's first behaviour, forced by measurement rather than taste. Run
/// against the real bodies, 9 of the 10 members of the cluster an adversarial
/// review picked as the best first map cite no repo directory at all: they
/// document a third-party SDK, not our code. Clusters like that are the ones
/// most worth mapping, so refusing them inverted the intent. The property
/// that mattered survives untouched — code still never invents a directory —
/// and a map that orients without naming one beats no map at all.
///
/// Members with no tokens still count toward the total, because an entry that
/// mentions no directory is evidence of nothing and should dilute a claim
/// rather than be quietly excluded from it.
#[must_use]
pub fn lives_in_value(cluster: &Cluster, input: &LoopInput) -> Option<String> {
    let mut homes: Vec<(String, usize)> = Vec::new();
    for id in &cluster.member_entry_ids {
        let Some(home) = input
            .entries
            .iter()
            .find(|entry| &entry.id == id)
            .and_then(|entry| entry.directory_tokens.first())
        else {
            continue;
        };
        match homes.iter_mut().find(|(token, _)| token == &home.token) {
            Some((_, count)) => *count += 1,
            None => homes.push((home.token.clone(), 1)),
        }
    }
    // Most-claimed first, ties broken by token so the answer is stable.
    homes.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let members = cluster.member_entry_ids.len();
    let (first, first_count) = homes.first().cloned()?;
    if first_count * 2 > members {
        return Some(first);
    }
    let (second, second_count) = homes.get(1).cloned()?;
    ((first_count + second_count) * 2 > members && second_count >= MIN_MEMBERS_PER_NAMED_DIRECTORY)
        .then(|| format!("{first} and {second}"))
}

/// Compose a FRESH map body: the first three sections ONLY (a new map has
/// no recorded gaps — the gap section appears only when a body already
/// carries it), with one `Detail entries:` line per `add_pointer` op in
/// `ops` that targets [`new_map_id(cluster_id)`], in op order.
///
/// The `Lives in` line is OMITTED when the cluster's members name no
/// directory with a plurality behind it — the commonest case for a cluster
/// documenting somebody else's code, and not a reason to withhold a map.
///
/// # Errors
/// [`ComposeError::EmptyPointerList`] when no op in `ops` points at the new
/// map — a map with no pointers is not a map.
pub fn compose_new_map_body(
    cluster: &Cluster,
    cluster_index: usize,
    cluster_id: &str,
    orientation_prose: &str,
    ops: &[Op],
    input: &LoopInput,
) -> Result<String, ComposeError> {
    let lives_in = lives_in_value(cluster, input);
    // Pointers are matched on the id the MODEL used; the map is RECORDED
    // under the run-unique one.
    let fresh_id = new_map_id(cluster_id);
    let mut detail_lines = Vec::new();
    for op in ops {
        if let Op::AddPointer {
            map_id,
            entry_id,
            gloss,
        } = op
            && map_id == &fresh_id
        {
            detail_lines.push(pointer_line(entry_id, gloss));
        }
    }
    if detail_lines.is_empty() {
        return Err(ComposeError::EmptyPointerList {
            map_id: fresh_map_id(cluster_index, cluster_id),
        });
    }
    let mut sections = Vec::with_capacity(3);
    if let Some(lives_in) = lives_in {
        sections.push(format!("{LIVES_IN_PREFIX}{lives_in}"));
    }
    sections.push(orientation_prose.to_string());
    sections.push(format!(
        "{DETAIL_ENTRIES_HEADER}\n{}",
        detail_lines.join("\n")
    ));
    Ok(sections.join(SECTION_SEPARATOR))
}

/// The pinned pointer-line shape: `- {entry_id} — {gloss}`.
fn pointer_line(entry_id: &str, gloss: &str) -> String {
    format!("- {entry_id} — {gloss}")
}

/// The pinned gap-line shape: `- {gap_text}`.
fn gap_line(gap_text: &str) -> String {
    format!("- {gap_text}")
}

/// Append one pointer line as the LAST line of `body`'s `Detail entries:`
/// section, leaving every other byte of the body identical.
///
/// Returns `None` when the body carries no `Detail entries:` section.
fn append_pointer_line(body: &str, entry_id: &str, gloss: &str) -> Option<String> {
    let mut lines: Vec<&str> = body.lines().collect();
    let header = lines
        .iter()
        .position(|line| *line == DETAIL_ENTRIES_HEADER)?;
    // The section ends at the first blank line after the header (the
    // grammar's `\n\n` section separator) or at the end of the body.
    let section_end = lines[header + 1..]
        .iter()
        .position(|line| line.is_empty())
        .map_or(lines.len(), |offset| header + 1 + offset);
    let pointer = pointer_line(entry_id, gloss);
    lines.insert(section_end, pointer.as_str());
    Some(lines.join("\n"))
}

/// Remove the exact gap line `- {gap_text}` from `body`'s
/// `Not yet documented:` section, header preserved.
///
/// Returns `None` when the body carries no such section or no such line.
fn strike_gap_line(body: &str, gap_text: &str) -> Option<String> {
    let lines: Vec<&str> = body.lines().collect();
    let header = lines.iter().position(|line| *line == NOT_YET_HEADER)?;
    let target = gap_line(gap_text);
    if !lines[header + 1..].contains(&target.as_str()) {
        return None;
    }
    Some(
        lines
            .iter()
            .filter(|line| **line != target)
            .copied()
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

/// The lead-disposition guard: a cluster whose rung-1 result carries an
/// `owning_map_id` must never mint a second map.
///
/// When a map is created covering 3 of 5 cluster members, the remaining 2
/// resurface next run as a SMALLER pocket, and the correct behaviour is to
/// add pointers to the owning map that loop-input already reports — that is
/// convergence, not a re-proposal. Without this guard the loop can mint a
/// duplicate map covering the leftovers of its own previous night's work,
/// which the one-new-map-per-project cap bounds but does not prevent. And
/// the decline ledger does not cover this case: a materialised cluster is
/// deliberately absent from the ledger (only monotonic, non-recomputable
/// state lives there), so the ledger filter will never stop a `create_map`
/// for an already-owned cluster — this validation is the only thing that
/// does.
///
/// The permitted ops for such a cluster are `add_pointer`, `strike_gap`,
/// `propose_gap`, and `no_change`.
///
/// # Errors
/// `Err` with the pinned refusal when the op set pairs an owned cluster
/// with a `create_map`.
pub fn validate_ops_for_cluster(cluster: &Cluster, ops: &[Op]) -> Result<(), String> {
    if let Some(owning_map_id) = &cluster.owning_map_id
        && ops.iter().any(|op| matches!(op, Op::CreateMap { .. }))
    {
        return Err(format!(
            "somnus: create_map is refused for cluster {} because rung 1 reports owning_map_id {owning_map_id} — partial coverage must converge by adding pointers to the owning map, not re-propose a second map for the remainder",
            cluster.label
        ));
    }
    Ok(())
}

/// Apply an admission [`Verdict`] to one cluster's op set, returning the ops
/// the pipeline should actually apply.
///
/// A floor, a union label and a phantom cluster are structural invariants, so
/// they live here rather than in the rung-2 instruction: a prompt makes a bad
/// map unlikely, and code makes it impossible.
///
/// Only the MINTING is affected. Pointers into maps that already exist, and
/// struck gaps, survive untouched: adding to an existing map is not the thing
/// admission guards against.
///
/// On a refusal the `create_map` and its pointers are replaced by a single
/// `propose_gap`, which is what the pipeline records as a decline, so the
/// observation survives instead of being re-proposed every night. A
/// `no_change` is dropped alongside, because leaving one would make the set
/// not-all-`propose_gap` and silently skip the decline write.
#[must_use]
pub fn apply_verdict(verdict: &Verdict, ops: Vec<Op>) -> Vec<Op> {
    let Verdict::Refused { reason } = verdict else {
        return ops;
    };
    let Some(cluster_id) = ops.iter().find_map(|op| match op {
        Op::CreateMap { cluster_id, .. } => Some(cluster_id.clone()),
        _ => None,
    }) else {
        return ops;
    };
    let minted = new_map_id(&cluster_id);
    let mut kept: Vec<Op> = ops
        .into_iter()
        .filter(|op| match op {
            Op::CreateMap { .. } | Op::NoChange { .. } => false,
            Op::AddPointer { map_id, .. } => map_id != &minted,
            Op::StrikeGap { .. } | Op::ProposeGap { .. } => true,
        })
        .collect();
    if !kept.iter().any(|op| matches!(op, Op::ProposeGap { .. })) {
        kept.push(Op::ProposeGap {
            reason: reason.clone(),
            cluster_id,
        });
    }
    kept
}

/// One composed body: the map it belongs to and the full body text.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ComposedBody {
    /// The map this body belongs to.
    pub map_id: String,
    /// The full body text.
    pub body: String,
}

/// One per-edit application step for an EXISTING map, as the pipeline's
/// map-op application needs it: the map, which edit, and the body state
/// AFTER that edit — the intermediate state each `add_pointer` POST must
/// carry (the server's `new − old == {added_entry_id}` set comparison makes
/// the FINAL composed body wrong for every call but the last).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MapEdit {
    /// The map being edited.
    pub map_id: String,
    /// The edit.
    pub kind: MapEditKind,
    /// The body as of this edit (the state after applying it).
    pub body_after: String,
}

/// Which edit a [`MapEdit`] carries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum MapEditKind {
    /// Point at `entry_id` from the map.
    AddPointer {
        /// The entry being pointed at.
        entry_id: String,
    },
    /// Strike `gap_text`, closed by `closing_entry_id`.
    StrikeGap {
        /// The gap text being struck.
        gap_text: String,
        /// The entry cited as closing it.
        closing_entry_id: String,
    },
}

/// The materialized result of one cluster's parsed op set. Never an error:
/// a refusal is recorded in the matching vector so one bad op can never
/// discard another op's paid work.
#[derive(Debug, Default, PartialEq)]
pub struct Materialized {
    /// The composed bodies, in first-touch order (a fresh map's body first
    /// when its op set created one).
    pub bodies: Vec<ComposedBody>,
    /// Rendered authorship refusals (an op targeted a map the loop does not
    /// own).
    pub authorship_refusals: Vec<String>,
    /// Rendered hard compose refusals (no body was composed for that op).
    pub compose_refusals: Vec<String>,
    /// The per-edit application chain, in op order per map, for the maps
    /// that survived (a map whose edit failed is dropped wholesale, its
    /// edits dropped with it). A fresh map's pointers are NOT here — they
    /// were already inlined into the create body.
    pub edits: Vec<MapEdit>,
}

/// Materialize one cluster's parsed op set into bodies, edits, and named
/// refusals. There is NO client-side new-map cap: both the per-project and
/// per-KB caps are SERVER-side at `POST /api/kb/map-op` (counted since UTC
/// midnight), so every `create_map` op composes, and a second map in one
/// op set meets the server's 409 as an ORDINARY ADMISSION OUTCOME at
/// application time.
#[allow(clippy::too_many_lines)]
// one op set, one linear pass: splitting it
// would scatter the read-modify-write chain across helpers.
#[must_use]
pub fn materialize_cluster(
    cluster: &Cluster,
    cluster_index: usize,
    ops: &[Op],
    input: &LoopInput,
) -> Materialized {
    let mut out = Materialized::default();

    // Every `create_map` op in the set composes (no client-side cap: the
    // server's per-night 409 is the ordinary admission outcome).
    for op in ops.iter().filter(|op| matches!(op, Op::CreateMap { .. })) {
        if let Op::CreateMap {
            cluster_id,
            orientation_prose,
            ..
        } = op
        {
            match compose_new_map_body(
                cluster,
                cluster_index,
                cluster_id,
                orientation_prose,
                ops,
                input,
            ) {
                Ok(body) => {
                    out.bodies.push(ComposedBody {
                        map_id: fresh_map_id(cluster_index, cluster_id),
                        body,
                    });
                }
                Err(err) => out.compose_refusals.push(err.to_string()),
            }
        }
    }

    // The working bodies for existing maps: read-modify-write, in
    // first-touch order, so two ops on the same map chain rather than
    // overwrite each other. A map whose edit failed is dropped from the
    // deliverables entirely — a partially applied op set is not a body.
    let mut working: Vec<ComposedBody> = Vec::new();
    let mut working_edits: Vec<Vec<MapEdit>> = Vec::new();
    let mut failed_maps: Vec<String> = Vec::new();
    let fresh_ids: Vec<String> = ops
        .iter()
        .filter_map(|op| match op {
            Op::CreateMap { cluster_id, .. } => Some(new_map_id(cluster_id)),
            _ => None,
        })
        .collect();

    for op in ops {
        let (map_id, edit) = match op {
            Op::AddPointer {
                map_id,
                entry_id,
                gloss,
            } => (map_id, Edit::AddPointer { entry_id, gloss }),
            Op::StrikeGap {
                map_id,
                gap_text,
                closing_entry_id,
            } => (
                map_id,
                Edit::StrikeGap {
                    gap_text,
                    closing_entry_id,
                },
            ),
            Op::CreateMap { .. } | Op::ProposeGap { .. } | Op::NoChange { .. } => continue,
        };
        // The fresh maps' pointers were already composed into their bodies
        // (inlined by the create), so they are never re-issued as calls.
        if fresh_ids.iter().any(|fresh_id| fresh_id == map_id) {
            continue;
        }
        let Some(map) = input.maps.iter().find(|map| &map.id == map_id) else {
            out.compose_refusals.push(
                ComposeError::UnknownMap {
                    map_id: map_id.clone(),
                }
                .to_string(),
            );
            continue;
        };
        if !is_loop_owned(map) {
            out.authorship_refusals.push(
                ComposeError::AuthorshipRefusal {
                    map_id: map_id.clone(),
                }
                .to_string(),
            );
            continue;
        }
        let position = working
            .iter()
            .position(|entry| entry.map_id == *map_id)
            .unwrap_or_else(|| {
                working.push(ComposedBody {
                    map_id: map_id.clone(),
                    body: map.body.clone(),
                });
                working_edits.push(Vec::new());
                working.len() - 1
            });
        let entry = &mut working[position];
        let edited = match &edit {
            Edit::AddPointer { entry_id, gloss } => {
                append_pointer_line(&entry.body, entry_id, gloss).ok_or_else(|| {
                    ComposeError::NoDetailSection {
                        map_id: map_id.clone(),
                    }
                    .to_string()
                })
            }
            Edit::StrikeGap { gap_text, .. } => {
                strike_gap_line(&entry.body, gap_text).ok_or_else(|| {
                    ComposeError::GapNotFound {
                        map_id: map_id.clone(),
                        gap_text: (*gap_text).clone(),
                    }
                    .to_string()
                })
            }
        };
        match edited {
            Ok(body) => {
                let kind = match &edit {
                    Edit::AddPointer { entry_id, .. } => MapEditKind::AddPointer {
                        entry_id: (*entry_id).clone(),
                    },
                    Edit::StrikeGap {
                        gap_text,
                        closing_entry_id,
                    } => MapEditKind::StrikeGap {
                        gap_text: (*gap_text).clone(),
                        closing_entry_id: (*closing_entry_id).clone(),
                    },
                };
                entry.body = body;
                working_edits[position].push(MapEdit {
                    map_id: map_id.clone(),
                    kind,
                    body_after: entry.body.clone(),
                });
            }
            Err(err) => {
                out.compose_refusals.push(err);
                failed_maps.push(map_id.clone());
            }
        }
    }

    for (entry, edits) in working.into_iter().zip(working_edits) {
        if failed_maps.contains(&entry.map_id) {
            continue;
        }
        out.bodies.push(entry);
        out.edits.extend(edits);
    }
    out
}

/// Which edit an op asks for on an existing map's body.
enum Edit<'a> {
    AddPointer {
        entry_id: &'a String,
        gloss: &'a String,
    },
    StrikeGap {
        gap_text: &'a String,
        closing_entry_id: &'a String,
    },
}

/// Where a composed map body is written: one file PER MAP (never one shared
/// `map-body.json`), under a project directory named by the
/// charset-validated project ref. `root` is a PARAMETER so parallel tests
/// cannot collide and no test depends on the machine tempdir.
#[must_use]
pub fn map_body_path(root: &Path, project_ref: &str, map_id: &str) -> PathBuf {
    root.join(project_ref).join(format!("{map_id}.json"))
}

/// Where the run report is written.
#[must_use]
pub fn run_report_path(root: &Path, project_ref: &str) -> PathBuf {
    root.join(project_ref).join("run-report.json")
}

/// Write one composed body to its [`map_body_path`], creating parent
/// directories. The file holds the composed body text EXACTLY (byte for
/// byte, no JSON envelope): it is both the read-modify-write substrate for
/// the next run and the file `curl --data-binary "@{path}"` posts to the
/// map-lint gate.
///
/// # Errors
/// Any filesystem failure (unwritable root, a non-directory in the path).
pub fn write_composed_body(
    root: &Path,
    project_ref: &str,
    composed: &ComposedBody,
) -> std::io::Result<PathBuf> {
    let path = map_body_path(root, project_ref, &composed.map_id);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, &composed.body)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loop_input::fixture;

    fn cluster(label: &str, members: &[&str], owning: Option<&str>) -> Cluster {
        Cluster {
            label: label.to_string(),
            member_entry_ids: members.iter().map(|id| (*id).to_string()).collect(),
            owning_map_id: owning.map(str::to_string),
            merit_reason: None,
        }
    }

    fn add_pointer(map_id: &str, entry_id: &str, gloss: &str) -> Op {
        Op::AddPointer {
            map_id: map_id.to_string(),
            entry_id: entry_id.to_string(),
            gloss: gloss.to_string(),
        }
    }

    fn create_map(cluster_id: &str, orientation_prose: &str) -> Op {
        Op::CreateMap {
            cluster_id: cluster_id.to_string(),
            title: "the title is not part of the body grammar".to_string(),
            orientation_prose: orientation_prose.to_string(),
        }
    }

    // --- the pinned constants --------------------------------------------

    #[test]
    fn the_structural_caps_and_writer_id_are_pinned() {
        assert_eq!(SOMNUS_WRITER_ID, "somnus");
        assert_eq!(SOMNUS_MAX_PROJECTS_PER_NIGHT, 3);
        assert_eq!(new_map_id("c1"), "somnus-new-c1");
    }

    // --- the authorship tier ---------------------------------------------

    #[test]
    fn only_loop_owned_maps_pass_the_predicate() {
        let input = fixture();
        assert!(is_loop_owned(&input.maps[0]), "kb-20001 is loop-owned");
        assert!(
            !is_loop_owned(&input.maps[1]),
            "kb-20002 was authored by a human and must be refused"
        );
        // A server null on either field is tolerated and simply not owned.
        let mut orphan = input.maps[0].clone();
        orphan.contributor = None;
        orphan.updated_by = None;
        assert!(!is_loop_owned(&orphan));
        let mut updated = input.maps[1].clone();
        updated.updated_by = Some("somnus".to_string());
        assert!(
            is_loop_owned(&updated),
            "last-updated-by also owns the edit"
        );
    }

    #[test]
    fn an_op_targeting_a_human_authored_map_is_refused() {
        let input = fixture();
        let owned = cluster("services", &["kb-10005"], None);
        let ops = vec![add_pointer("kb-20002", "kb-10005", "gloss")];
        let materialized = materialize_cluster(&owned, 0, &ops, &input);
        assert!(materialized.bodies.is_empty());
        assert_eq!(materialized.authorship_refusals.len(), 1);
        assert!(
            materialized.authorship_refusals[0].contains("kb-20002"),
            "{}",
            materialized.authorship_refusals[0]
        );
        assert!(
            materialized.authorship_refusals[0].contains("not loop-owned (authorship tier"),
            "{}",
            materialized.authorship_refusals[0]
        );
    }

    // --- the Lives in derivation -----------------------------------------

    #[test]
    fn a_majority_of_members_sharing_a_home_names_it() {
        let input = fixture();
        let owned = cluster("home-network", &["kb-10001"], Some("kb-20001"));
        assert_eq!(
            lives_in_value(&owned, &input),
            Some("knowledge/network".to_string())
        );
    }

    /// Two directories, each the home of at least two members and together a
    /// majority — a real two-directory subject, and the joined form is
    /// lint-clean.
    #[test]
    fn a_genuine_two_directory_split_names_both() {
        let mut input = fixture();
        for (index, entry) in input.entries.iter_mut().enumerate() {
            let token = if index < 2 {
                "apps/web"
            } else {
                "packages/core"
            };
            entry.directory_tokens = vec![crate::loop_input::DirectoryToken {
                token: token.to_string(),
                hits: 3,
            }];
        }
        let spread = cluster(
            "spread",
            &["kb-10001", "kb-10002", "kb-10003", "kb-10004"],
            None,
        );
        assert_eq!(
            lives_in_value(&spread, &input),
            Some("apps/web and packages/core".to_string()),
            "an even split is ordered by token so the answer is stable"
        );
    }

    /// The tie that must NOT be resolved. Three members, three homes, no
    /// plurality — rung 1 reporting a bad carve. Naming one of them would
    /// pass off an arbitrary pick as orientation.
    #[test]
    fn a_three_way_tie_names_no_directory() {
        let mut input = fixture();
        for (index, entry) in input.entries.iter_mut().enumerate() {
            entry.directory_tokens = vec![crate::loop_input::DirectoryToken {
                token: format!("dir{index}"),
                hits: 3,
            }];
        }
        let scattered = cluster("scattered", &["kb-10001", "kb-10002", "kb-10003"], None);
        assert_eq!(lives_in_value(&scattered, &input), None);
    }

    /// A member with no tokens is evidence of nothing, so it dilutes the
    /// claim rather than being quietly dropped from the denominator.
    #[test]
    fn untokened_members_dilute_the_majority() {
        let mut input = fixture();
        for entry in &mut input.entries {
            entry.directory_tokens.clear();
        }
        input.entries[0].directory_tokens = vec![crate::loop_input::DirectoryToken {
            token: "apps/web".to_string(),
            hits: 9,
        }];
        let thin = cluster("thin-evidence", &["kb-10001", "kb-10002", "kb-10003"], None);
        assert_eq!(
            lives_in_value(&thin, &input),
            None,
            "one member in nine directories is still one member"
        );
    }

    /// The case measurement forced: a cluster documenting somebody else's
    /// code names no directory of ours, and those are among the clusters most
    /// worth mapping. The body composes WITHOUT the line rather than the map
    /// being withheld.
    #[test]
    fn a_cluster_naming_no_directory_still_gets_a_body() {
        let mut input = fixture();
        for entry in &mut input.entries {
            entry.directory_tokens.clear();
        }
        let unowned = cluster("third-party-sdk", &["kb-10001"], None);
        assert_eq!(lives_in_value(&unowned, &input), None);
        let ops = vec![
            create_map("c1", "PROSE"),
            add_pointer("somnus-new-c1", "kb-10001", "GLOSS-1"),
        ];
        let body = compose_new_map_body(&unowned, 0, "c1", "PROSE", &ops, &input)
            .expect("no directory is not a reason to withhold a map");
        assert_eq!(body, "PROSE\n\nDetail entries:\n- kb-10001 — GLOSS-1");
        assert!(
            !body.contains("Lives in"),
            "the line is omitted, never invented: {body}"
        );
    }

    // --- the fresh-body sentinel: the model-text boundary -----------------

    #[test]
    fn a_fresh_body_is_composed_with_only_two_model_strings_in_it() {
        let input = fixture();
        let owned = cluster(
            "wireguard-and-dns",
            &["kb-10001", "kb-10002"],
            Some("kb-20001"),
        );
        let ops = vec![
            create_map("c1", "ORIENTATION-PROSE"),
            add_pointer("somnus-new-c1", "kb-10001", "GLOSS-1"),
            add_pointer("somnus-new-c1", "kb-10002", "GLOSS-2"),
            add_pointer("kb-20001", "kb-10005", "GLOSS-5"),
        ];
        let materialized = materialize_cluster(&owned, 0, &ops, &input);

        assert_eq!(materialized.bodies.len(), 2);
        // Every byte outside ORIENTATION-PROSE and the two glosses is
        // code-composed or fixture-derived.
        assert_eq!(
            materialized.bodies[0],
            ComposedBody {
                map_id: "somnus-new-0-c1".to_string(),
                body: "Lives in knowledge/network\n\nORIENTATION-PROSE\n\nDetail entries:\n- kb-10001 — GLOSS-1\n- kb-10002 — GLOSS-2"
                    .to_string(),
            }
        );
        // And the read-modify-write body on the existing loop-owned map.
        assert_eq!(
            materialized.bodies[1],
            ComposedBody {
                map_id: "kb-20001".to_string(),
                body: "Lives in knowledge/linux/network\n\nOrients the switch, DNS, and VPN entries for the home network.\n\nDetail entries:\n- kb-10001 — switch VLAN tagging baseline\n- kb-10002 — recursive DNS resolvers\n- kb-10003 — wireguard client config\n- kb-99999 — cross-project wireguard hub notes\n- kb-10004 — DHCP lease hygiene\n- kb-10005 — GLOSS-5\n\nNot yet documented:\n- Site-to-site wireguard topology"
                    .to_string(),
            }
        );
    }

    #[test]
    fn a_fresh_map_with_no_pointers_is_refused() {
        let input = fixture();
        let owned = cluster("home-network", &["kb-10001"], Some("kb-20001"));
        let ops = vec![create_map("c1", "ORIENTATION-PROSE")];
        let materialized = materialize_cluster(&owned, 0, &ops, &input);
        assert!(materialized.bodies.is_empty());
        assert_eq!(materialized.compose_refusals.len(), 1);
        assert!(
            materialized.compose_refusals[0].contains("no pointers"),
            "{}",
            materialized.compose_refusals[0]
        );
        assert_eq!(
            ComposeError::EmptyPointerList {
                map_id: "somnus-new-c1".to_string(),
            }
            .to_string(),
            "somnus: refusing to compose map somnus-new-c1: the create_map op set lists no pointers"
        );
    }

    #[test]
    fn a_fresh_map_has_no_gap_section_even_when_other_maps_carry_one() {
        // The gap section appears only when a body already carries it: a
        // fresh map is the first three sections ONLY.
        let input = fixture();
        let owned = cluster("home-network", &["kb-10001"], Some("kb-20001"));
        let ops = vec![
            create_map("c1", "PROSE"),
            add_pointer("somnus-new-c1", "kb-10001", "GLOSS-1"),
        ];
        let body = compose_new_map_body(&owned, 0, "c1", "PROSE", &ops, &input)
            .expect("composed with pointers");
        assert_eq!(
            body,
            "Lives in knowledge/network\n\nPROSE\n\nDetail entries:\n- kb-10001 — GLOSS-1"
        );
        assert!(!body.contains("Not yet documented:"));
    }

    #[test]
    fn detail_lines_follow_op_order_exactly() {
        let input = fixture();
        let owned = cluster("home-network", &["kb-10001", "kb-10002"], Some("kb-20001"));
        let ops = vec![
            create_map("c1", "PROSE"),
            add_pointer("somnus-new-c1", "kb-10002", "GLOSS-B"),
            add_pointer("somnus-new-c1", "kb-10001", "GLOSS-A"),
        ];
        let body = compose_new_map_body(&owned, 0, "c1", "PROSE", &ops, &input)
            .expect("composed with pointers");
        assert!(
            body.ends_with("Detail entries:\n- kb-10002 — GLOSS-B\n- kb-10001 — GLOSS-A"),
            "op order, not id order: {body}"
        );
    }

    // --- add_pointer: the byte-pinned append ------------------------------

    #[test]
    fn add_pointer_appends_as_the_last_detail_line_byte_for_byte() {
        let input = fixture();
        let owned = cluster("home-network", &["kb-10005"], Some("kb-20001"));
        let ops = vec![add_pointer("kb-20001", "kb-10005", "GLOSS-5")];
        let materialized = materialize_cluster(&owned, 0, &ops, &input);
        assert_eq!(materialized.bodies.len(), 1);
        assert_eq!(
            materialized.bodies[0].body,
            "Lives in knowledge/linux/network\n\nOrients the switch, DNS, and VPN entries for the home network.\n\nDetail entries:\n- kb-10001 — switch VLAN tagging baseline\n- kb-10002 — recursive DNS resolvers\n- kb-10003 — wireguard client config\n- kb-99999 — cross-project wireguard hub notes\n- kb-10004 — DHCP lease hygiene\n- kb-10005 — GLOSS-5\n\nNot yet documented:\n- Site-to-site wireguard topology"
        );
    }

    #[test]
    fn two_add_pointer_ops_on_one_map_chain_instead_of_overwriting() {
        let input = fixture();
        let owned = cluster("home-network", &["kb-10005", "kb-10006"], Some("kb-20001"));
        let ops = vec![
            add_pointer("kb-20001", "kb-10005", "GLOSS-5"),
            add_pointer("kb-20001", "kb-10006", "GLOSS-6"),
        ];
        let materialized = materialize_cluster(&owned, 0, &ops, &input);
        assert_eq!(materialized.bodies.len(), 1, "one body per map");
        assert!(materialized.bodies[0].body.contains(
            "- kb-10004 — DHCP lease hygiene\n- kb-10005 — GLOSS-5\n- kb-10006 — GLOSS-6"
        ));
    }

    #[test]
    fn an_op_targeting_an_unknown_map_is_refused_by_name() {
        let input = fixture();
        let owned = cluster("home-network", &["kb-10005"], Some("kb-20001"));
        let ops = vec![add_pointer("kb-99999", "kb-10005", "GLOSS-5")];
        let materialized = materialize_cluster(&owned, 0, &ops, &input);
        assert!(materialized.bodies.is_empty());
        assert_eq!(materialized.compose_refusals.len(), 1);
        assert!(
            materialized.compose_refusals[0].contains("unknown map kb-99999"),
            "{}",
            materialized.compose_refusals[0]
        );
    }

    #[test]
    fn a_body_with_no_detail_entries_section_refuses_the_append() {
        let mut input = fixture();
        input.maps[0].body = "Lives in knowledge/linux/network\n\nProse only.".to_string();
        let owned = cluster("home-network", &["kb-10005"], Some("kb-20001"));
        let ops = vec![add_pointer("kb-20001", "kb-10005", "GLOSS-5")];
        let materialized = materialize_cluster(&owned, 0, &ops, &input);
        assert!(materialized.bodies.is_empty());
        assert_eq!(materialized.compose_refusals.len(), 1);
        assert!(
            materialized.compose_refusals[0].contains("no `Detail entries:` section"),
            "{}",
            materialized.compose_refusals[0]
        );
    }

    // --- strike_gap: the byte-pinned removal ------------------------------

    #[test]
    fn strike_gap_removes_the_exact_line_and_preserves_the_header() {
        let input = fixture();
        let owned = cluster("home-network", &["kb-10001"], Some("kb-20001"));
        let ops = vec![Op::StrikeGap {
            map_id: "kb-20001".to_string(),
            gap_text: "Site-to-site wireguard topology".to_string(),
            closing_entry_id: "kb-10003".to_string(),
        }];
        let materialized = materialize_cluster(&owned, 0, &ops, &input);
        assert_eq!(materialized.bodies.len(), 1);
        assert_eq!(
            materialized.bodies[0].body,
            "Lives in knowledge/linux/network\n\nOrients the switch, DNS, and VPN entries for the home network.\n\nDetail entries:\n- kb-10001 — switch VLAN tagging baseline\n- kb-10002 — recursive DNS resolvers\n- kb-10003 — wireguard client config\n- kb-99999 — cross-project wireguard hub notes\n- kb-10004 — DHCP lease hygiene\n\nNot yet documented:"
        );
    }

    #[test]
    fn strike_gap_on_an_absent_line_refuses_by_name() {
        let input = fixture();
        let owned = cluster("home-network", &["kb-10001"], Some("kb-20001"));
        let ops = vec![Op::StrikeGap {
            map_id: "kb-20001".to_string(),
            gap_text: "a gap nobody recorded".to_string(),
            closing_entry_id: "kb-10003".to_string(),
        }];
        let materialized = materialize_cluster(&owned, 0, &ops, &input);
        assert!(materialized.bodies.is_empty());
        assert_eq!(materialized.compose_refusals.len(), 1);
        assert!(
            materialized.compose_refusals[0].contains("no `Not yet documented:` line"),
            "{}",
            materialized.compose_refusals[0]
        );
    }

    // --- the caps are SERVER-side: both create_map ops compose here ------

    #[test]
    fn two_create_maps_in_one_op_set_both_compose_and_both_gate() {
        // No client-side cap survives: the per-project and per-KB caps are
        // server-side at POST /api/kb/map-op, so both fresh bodies compose
        // and the second's application meets the ordinary 409 admission.
        let input = fixture();
        let owned = cluster("home-network", &["kb-10001"], Some("kb-20001"));
        let ops = vec![
            create_map("c1", "PROSE-1"),
            add_pointer("somnus-new-c1", "kb-10001", "GLOSS-1"),
            create_map("c2", "PROSE-2"),
            add_pointer("somnus-new-c2", "kb-10001", "GLOSS-1"),
        ];
        let materialized = materialize_cluster(&owned, 0, &ops, &input);
        assert_eq!(materialized.bodies.len(), 2);
        assert_eq!(materialized.bodies[0].map_id, "somnus-new-0-c1");
        assert_eq!(materialized.bodies[1].map_id, "somnus-new-0-c2");
        assert_eq!(materialized.compose_refusals, Vec::<String>::new());
        // A fresh map's add_pointers are inlined into the create body: no
        // edits for either fresh map.
        assert!(materialized.edits.is_empty());
    }

    #[test]
    fn fresh_maps_do_not_appear_in_the_edit_chain_and_existing_maps_do() {
        let input = fixture();
        let owned = cluster("home-network", &["kb-10005", "kb-10006"], Some("kb-20001"));
        let ops = vec![
            add_pointer("kb-20001", "kb-10005", "GLOSS-5"),
            Op::StrikeGap {
                map_id: "kb-20001".to_string(),
                gap_text: "Site-to-site wireguard topology".to_string(),
                closing_entry_id: "kb-10003".to_string(),
            },
        ];
        let materialized = materialize_cluster(&owned, 0, &ops, &input);
        assert_eq!(materialized.bodies.len(), 1);
        assert_eq!(materialized.edits.len(), 2, "one edit per op, in op order");
        assert_eq!(
            materialized.edits[0],
            MapEdit {
                map_id: "kb-20001".to_string(),
                kind: MapEditKind::AddPointer {
                    entry_id: "kb-10005".to_string(),
                },
                body_after: materialized.edits[0].body_after.clone(),
            }
        );
        assert!(
            materialized.edits[0]
                .body_after
                .contains("- kb-10005 — GLOSS-5"),
            "body_after carries the state after the FIRST edit"
        );
        assert!(!materialized.edits[0].body_after.contains("GLOSS-6"));
        assert_eq!(
            materialized.edits[1].kind,
            MapEditKind::StrikeGap {
                gap_text: "Site-to-site wireguard topology".to_string(),
                closing_entry_id: "kb-10003".to_string(),
            }
        );
        assert!(
            !materialized.edits[1]
                .body_after
                .contains("Site-to-site wireguard topology"),
            "body_after carries the state after the SECOND edit"
        );
    }

    // --- the lead-disposition guard --------------------------------------

    #[test]
    fn create_map_for_an_owned_cluster_is_rejected_before_any_op_is_applied() {
        let owned = cluster("home-network", &["kb-10001"], Some("kb-20001"));
        let ops = vec![
            create_map("c1", "PROSE"),
            add_pointer("somnus-new-c1", "kb-10001", "GLOSS-1"),
        ];
        let error = validate_ops_for_cluster(&owned, &ops)
            .expect_err("an owned cluster must never mint a second map");
        assert!(error.contains("create_map is refused"), "{error}");
        assert!(error.contains("owning_map_id kb-20001"), "{error}");
        assert!(error.contains("converge"), "{error}");
        // The permitted ops validate clean.
        for permitted in [
            vec![add_pointer("kb-20001", "kb-10001", "GLOSS-1")],
            vec![Op::StrikeGap {
                map_id: "kb-20001".to_string(),
                gap_text: "gap".to_string(),
                closing_entry_id: "kb-10003".to_string(),
            }],
            vec![Op::ProposeGap {
                cluster_id: "home-network".to_string(),
                reason: "r".to_string(),
            }],
            vec![Op::NoChange {
                cluster_id: "home-network".to_string(),
            }],
            vec![],
        ] {
            assert_eq!(
                validate_ops_for_cluster(&owned, &permitted),
                Ok(()),
                "permitted ops must validate: {permitted:?}"
            );
        }
        // And a cluster with NO owning map may create.
        let unowned = cluster("backup-drills", &["kb-10006"], None);
        assert_eq!(
            validate_ops_for_cluster(&unowned, &[create_map("c1", "PROSE")]),
            Ok(())
        );
    }

    // --- the disk paths ---------------------------------------------------

    #[test]
    fn the_disk_paths_are_pinned_per_map_and_per_project() {
        let root = Path::new("/tmp/somnus-root");
        assert_eq!(
            map_body_path(root, "demo-project", "kb-20001"),
            PathBuf::from("/tmp/somnus-root/demo-project/kb-20001.json")
        );
        assert_eq!(
            map_body_path(root, "demo-project", "somnus-new-c1"),
            PathBuf::from("/tmp/somnus-root/demo-project/somnus-new-c1.json")
        );
        assert_eq!(
            run_report_path(root, "demo-project"),
            PathBuf::from("/tmp/somnus-root/demo-project/run-report.json")
        );
    }

    #[test]
    fn write_composed_body_creates_the_project_directory_and_the_file() {
        let root = tempfile::tempdir().expect("tempdir");
        let composed = ComposedBody {
            map_id: "somnus-new-c1".to_string(),
            body: "{\"body\": \"Lives in x\"}".to_string(),
        };
        let path = write_composed_body(root.path(), "demo-project", &composed)
            .expect("the body lands on disk");
        assert_eq!(
            path,
            root.path().join("demo-project").join("somnus-new-c1.json")
        );
        assert_eq!(
            std::fs::read_to_string(&path).expect("the file exists"),
            composed.body
        );
    }

    #[test]
    fn write_composed_body_fails_closed_when_the_parent_cannot_be_created() {
        // A FILE where the project directory should be: `create_dir_all`
        // fails, the error propagates, and nothing is silently skipped.
        let root = tempfile::tempdir().expect("tempdir");
        std::fs::write(root.path().join("demo-project"), "not a directory")
            .expect("a file in the path");
        let composed = ComposedBody {
            map_id: "somnus-new-c1".to_string(),
            body: "x".to_string(),
        };
        assert!(write_composed_body(root.path(), "demo-project", &composed).is_err());
    }
}

#[cfg(test)]
mod verdict_tests {
    use super::*;

    fn create_map() -> Op {
        Op::CreateMap {
            cluster_id: "c1".to_string(),
            title: "Thin Cluster".to_string(),
            orientation_prose: "PROSE".to_string(),
        }
    }

    fn pointer(map_id: &str, entry_id: &str) -> Op {
        Op::AddPointer {
            map_id: map_id.to_string(),
            entry_id: entry_id.to_string(),
            gloss: "GLOSS".to_string(),
        }
    }

    fn refused() -> Verdict {
        Verdict::Refused {
            reason: "REASON".to_string(),
        }
    }

    #[test]
    fn an_admitted_cluster_passes_through_untouched() {
        let ops = vec![create_map(), pointer("somnus-new-c1", "kb-10001")];
        assert_eq!(apply_verdict(&Verdict::Admitted, ops.clone()), ops);
    }

    #[test]
    fn a_refused_cluster_gets_a_gap_instead_of_a_map() {
        let ops = apply_verdict(
            &refused(),
            vec![
                create_map(),
                pointer("somnus-new-c1", "kb-10001"),
                pointer("somnus-new-c1", "kb-10002"),
            ],
        );
        assert_eq!(
            ops,
            vec![Op::ProposeGap {
                cluster_id: "c1".to_string(),
                reason: "REASON".to_string(),
            }],
            "the map and every pointer into it must be gone"
        );
    }

    /// An op set that is ONLY `propose_gap` is what the pipeline records as a
    /// decline, so the substituted set has to be exactly that — a stray
    /// `no_change` would silently skip the decline write and the cluster
    /// would be re-proposed every night forever.
    #[test]
    fn the_refused_set_reads_as_a_clean_decline() {
        let ops = apply_verdict(
            &refused(),
            vec![
                create_map(),
                Op::NoChange {
                    cluster_id: "c1".to_string(),
                },
            ],
        );
        assert!(
            ops.iter().all(|op| matches!(op, Op::ProposeGap { .. })),
            "got {ops:?}"
        );
    }

    /// Admission guards MINTING, not contributing: a refusal leaves pointers
    /// into maps that already exist alone.
    #[test]
    fn pointers_into_an_existing_map_survive_a_refusal() {
        {
            let verdict = refused();
            let ops = apply_verdict(
                &verdict,
                vec![
                    create_map(),
                    pointer("somnus-new-c1", "kb-10001"),
                    pointer("kb-20001", "kb-10002"),
                ],
            );
            assert!(
                ops.contains(&pointer("kb-20001", "kb-10002")),
                "{verdict:?} dropped work on an existing map: {ops:?}"
            );
        }
    }

    #[test]
    fn a_cluster_the_model_already_declined_is_left_alone() {
        let ops = vec![Op::ProposeGap {
            cluster_id: "c1".to_string(),
            reason: "the model's own words".to_string(),
        }];
        assert_eq!(apply_verdict(&refused(), ops.clone()), ops);
    }
}
