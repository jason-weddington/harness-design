//! The per-project pipeline: one unit of nightly work, from the run-start
//! baseline to the run report on disk.
//!
//! Every collaborator is INJECTED through [`UnitDeps`] — the model backend,
//! the loop-input source, the decline ledger, the gate factory, the tool
//! context, and the body root — so the whole unit is exercisable with a
//! scripted backend, an in-memory ledger, a stub gate runner, and a
//! tempdir: no live KB, no model, no credentials. The production `ToolCtx`
//! construction rides with the write-transport follow-up (the CLI run body
//! stays a stub this cut), so every call site in tests passes
//! `ToolCtx::stub()`.
//!
//! [`run_unit`] NEVER panics and returns a [`UnitReport`], not a `Result`:
//! every failure mode is a named outcome on the report, the whole unit runs
//! under the nightly wall-clock budget, and the report is always written to
//! [`crate::materialize::run_report_path`].
//!
//! Layering (pinned): this module references every other somnus module.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use harness::exec::{ChangeEvidence, ChangeObserver, ChecksRunner, classify_change};
use harness::model::ModelBackend;
use harness::tool::ToolCtx;
use serde::Serialize;

use crate::ledger::{ClusterLedger, LedgerCandidate, LedgerVerdictRecord, filter_pockets};
use crate::loop_input::{
    Cluster, FetchOutcome, LoopInputCountSource, LoopInputSource, PocketsStatus, pockets_status,
    render_ledger_filter_line, render_pockets_not_computed_line, validate_project_ref,
};
use crate::materialize::{
    ComposedBody, Materialized, materialize_cluster, run_report_path, write_composed_body,
};
use crate::observer::MapPointerObserver;
use crate::ops::{Op, op_from_call};
use crate::rungs::{parse_clusters, rung1_raw_path, rung1_turn, rung2_raw_path, rung2_turn};

/// The pinned abort reason for a wall-clock expiry.
pub const WALL_CLOCK_ABORT_MSG: &str = "somnus: nightly wall-clock budget exhausted (14400s)";

/// The pinned abort reason for the per-unit inference budget.
pub const BUDGET_ABORT_MSG: &str = "somnus: per-unit inference budget exhausted (24 backend calls); aborting rung 2 for remaining clusters";

/// Every collaborator one unit needs, injected. The model is a `MockBackend`
/// in tests, the ledger an `InMemoryLedger`, the KB a wiremock server, the
/// gate a stub runner, and the body root a tempdir.
pub struct UnitDeps<'a> {
    /// The model backend both rungs call.
    pub backend: &'a dyn ModelBackend,
    /// The loop-input source (also the pointer-count source behind the
    /// leg-3 observer — one endpoint, one transport).
    pub source: Arc<dyn LoopInputSource>,
    /// The cluster/decline ledger.
    pub ledger: &'a dyn ClusterLedger,
    /// Builds the gate runner for one composed body's path.
    pub gate_for: &'a dyn Fn(&Path) -> ChecksRunner,
    /// The tool context the gate runner offloads through. `ToolCtx::stub()`
    /// in every test; the production construction rides with the
    /// write-transport follow-up.
    pub tool_ctx: &'a ToolCtx,
    /// The root every composed body, raw offload, and the run report are
    /// written under. A PARAMETER (a tempdir in tests) so parallel runs
    /// cannot collide.
    pub body_root: PathBuf,
}

/// How one unit ended.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum UnitOutcome {
    /// The unit ran to the end of its pipeline.
    Ready,
    /// The server answered `404`: terminal and ordinary, nothing to run.
    UnknownProject,
    /// The server answered `409`: terminal and ordinary, nothing to run.
    NotEligible,
    /// The bearer token was rejected.
    Unauthorized,
    /// The unit stopped with a named reason (never a retry).
    Aborted {
        /// Why the unit aborted.
        reason: String,
    },
}

/// Cost telemetry for one rung, accumulated per turn. Enforcement is NOT
/// this crate's job (the harness budget owns it); the run report still
/// carries the numbers so a night is auditable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct UsageTotals {
    /// Freshly-evaluated input tokens, summed.
    pub input: u64,
    /// Output tokens, summed.
    pub output: u64,
    /// Cache-read tokens, summed (absent counts as zero).
    pub cache_read: u64,
    /// Cache-write tokens, summed (absent counts as zero).
    pub cache_write: u64,
}

impl UsageTotals {
    /// Accumulate one turn's [`harness::model::Usage`].
    fn add(&mut self, usage: &harness::model::Usage) {
        self.input += u64::from(usage.input_tokens);
        self.output += u64::from(usage.output_tokens);
        self.cache_read += u64::from(usage.cache_read_tokens.unwrap_or(0));
        self.cache_write += u64::from(usage.cache_write_tokens.unwrap_or(0));
    }
}

/// How one cluster's rung-2 inference went.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Rung2Outcome {
    /// The turn's tool calls parsed into ops.
    Parsed,
    /// The cluster's op set was rejected (an unknown/ill-typed tool call, a
    /// backend error, or a `create_map` for an already-owned cluster). The
    /// cluster is skipped and the run continues — one bad cluster must not
    /// discard the other clusters' paid work.
    ParseError {
        /// Why the op set was rejected.
        reason: String,
    },
}

/// One rung-1 cluster and what rung 2 did with it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ClusterRecord {
    /// The cluster's model-chosen label.
    pub label: String,
    /// The cluster's member entries.
    pub member_entry_ids: Vec<String>,
    /// The existing map rung 1 says covers this cluster, if any.
    pub owning_map_id: Option<String>,
    /// The parsed ops rung 2 emitted for this cluster (`[]` on a
    /// `ParseError`).
    pub ops: Vec<Op>,
    /// How rung 2 went.
    pub rung2: Rung2Outcome,
    /// Where the cluster's raw turn text was offloaded on a parse error.
    pub raw_path: Option<PathBuf>,
}

impl ClusterRecord {
    /// Rebuild the rung-1 [`Cluster`] this record describes (the
    /// materializer takes the cluster, the report carries the record).
    #[must_use]
    pub fn cluster(&self) -> Cluster {
        Cluster {
            label: self.label.clone(),
            member_entry_ids: self.member_entry_ids.clone(),
            owning_map_id: self.owning_map_id.clone(),
        }
    }
}

/// One composed body, as the run report records it: the map, its path on
/// disk, and the full text, so the report alone reconstructs the
/// deliverable.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ComposedBodyRecord {
    /// The map this body belongs to.
    pub map_id: String,
    /// Where the body was written (and what the gate posted).
    pub body_path: PathBuf,
    /// The full body text.
    pub body: String,
}

/// One gate leg for one composed body. The HTTP status is the verdict: a
/// red report here means the map-lint refused the body.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GateReportRecord {
    /// The map that was gated.
    pub map_id: String,
    /// Whether the gate passed.
    pub passed: bool,
    /// The gate child's exit code, when it exited.
    pub exit_code: Option<i32>,
    /// Whether the gate child timed out.
    pub timed_out: bool,
}

/// The report one unit leaves on disk. Every refusal, decline, body, gate
/// verdict, and cost number the night produced is in here — including the
/// per-candidate ledger verdicts (`ledger_verdicts`), so the report alone
/// reconstructs which cluster was declined, against which stored decline,
/// at what jaccard.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UnitReport {
    /// The project this unit ran for.
    pub project_ref: String,
    /// How the unit ended.
    pub outcome: UnitOutcome,
    /// Pockets fetched from loop-input (pre-filter).
    pub pockets_fetched: usize,
    /// Pockets the ledger filter kept.
    pub pockets_kept: usize,
    /// Pockets the ledger filter declined.
    pub pockets_declined: usize,
    /// Verdicts the ledger matched.
    pub matched_verdicts: usize,
    /// One record per ledger verdict, the candidate's member ids zipped
    /// with its verdict by index.
    pub ledger_verdicts: Vec<LedgerVerdictRecord>,
    /// Why pockets were unavailable, if they were.
    pub pockets_status: PocketsStatus,
    /// One record per cluster rung 2 processed.
    pub clusters: Vec<ClusterRecord>,
    /// The member-id sets whose declines were recorded this run.
    pub declines_recorded: Vec<Vec<String>>,
    /// Rendered authorship refusals (an op targeted a map somnus does not
    /// own).
    pub authorship_refusals: Vec<String>,
    /// Rendered per-project new-map cap refusals.
    pub cap_refusals: Vec<String>,
    /// Rendered hard compose refusals (no body was composed for that op).
    pub compose_refusals: Vec<String>,
    /// The composed bodies, in materialization order.
    pub composed_bodies: Vec<ComposedBodyRecord>,
    /// One gate report per composed body.
    pub gate_reports: Vec<GateReportRecord>,
    /// The leg-3 change evidence.
    pub change: ChangeEvidence,
    /// How many `ModelBackend::turn` calls the unit made.
    pub backend_calls: u64,
    /// Rung-1 cost telemetry.
    pub usage_rung1: UsageTotals,
    /// Rung-2 cost telemetry.
    pub usage_rung2: UsageTotals,
    /// The single-shot tripwire: `Some` only when the call count exceeds
    /// 1 + cluster count, i.e. exactly when the discipline was violated.
    pub call_count_audit: Option<String>,
    /// Where a failed rung-1 parse offloaded the raw model text.
    pub rung1_raw_path: Option<PathBuf>,
    /// Where this report was written.
    pub report_path: PathBuf,
}

impl UnitReport {
    /// A fresh report with every count at zero and the outcome `Ready`.
    #[must_use]
    pub fn new(project_ref: &str, report_path: PathBuf) -> Self {
        Self {
            project_ref: project_ref.to_string(),
            outcome: UnitOutcome::Ready,
            pockets_fetched: 0,
            pockets_kept: 0,
            pockets_declined: 0,
            matched_verdicts: 0,
            ledger_verdicts: Vec::new(),
            pockets_status: PocketsStatus::NoneFound,
            clusters: Vec::new(),
            declines_recorded: Vec::new(),
            authorship_refusals: Vec::new(),
            cap_refusals: Vec::new(),
            compose_refusals: Vec::new(),
            composed_bodies: Vec::new(),
            gate_reports: Vec::new(),
            change: ChangeEvidence::default(),
            backend_calls: 0,
            usage_rung1: UsageTotals::default(),
            usage_rung2: UsageTotals::default(),
            call_count_audit: None,
            rung1_raw_path: None,
            report_path,
        }
    }
}

/// The single-shot discipline tripwire: `Some` iff `calls > 1 +
/// cluster_count` (one rung-1 inference plus one rung-2 inference per
/// cluster is the only shape a conforming run can have). Rendered as data
/// so the run report carries it without a conditional `eprintln!` site.
#[must_use]
pub fn render_call_count_audit_line(calls: u64, cluster_count: usize) -> Option<String> {
    let expected = 1 + cluster_count as u64;
    if calls > expected {
        Some(format!(
            "somnus: call-count audit: {calls} backend calls for {cluster_count} clusters (expected at most {expected}); the single-shot discipline was violated"
        ))
    } else {
        None
    }
}

/// The stderr line for one recorded decline, naming the cluster whose
/// decline was written. Pure so the shape is byte-pinned by a unit test;
/// the pipeline only `eprintln!`s it.
#[must_use]
pub fn render_decline_recorded_line(project_ref: &str, member_entry_ids: &[String]) -> String {
    format!(
        "somnus: decline recorded for {project_ref}: [{}]",
        member_entry_ids.join(", ")
    )
}

/// The stderr line for one decline write that failed, naming the cluster
/// whose decline was lost — so a multi-decline loop that aborts names which
/// cluster's decline did not land. Pure so the shape is byte-pinned by a
/// unit test; the pipeline only `eprintln!`s it.
#[must_use]
pub fn render_decline_write_failed_line(project_ref: &str, member_entry_ids: &[String]) -> String {
    format!(
        "somnus: decline write failed for {project_ref}: [{}]",
        member_entry_ids.join(", ")
    )
}

/// Write `text` to `path`, creating parent directories.
///
/// # Errors
/// Any filesystem failure (unwritable root, a non-directory in the path).
fn write_text(path: &Path, text: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, text)
}

/// Run one nightly unit for `project_ref`.
///
/// Never panics, never retries, and always returns a report. The whole unit
/// body is bounded by the nightly wall-clock budget
/// ([`crate::NIGHTLY_WALL_CLOCK_SECS`]); an expiry is a named abort, not a
/// hang.
pub async fn run_unit(deps: &UnitDeps<'_>, project_ref: &str) -> UnitReport {
    let report_path = run_report_path(&deps.body_root, project_ref);
    match tokio::time::timeout(
        Duration::from_secs(crate::NIGHTLY_WALL_CLOCK_SECS),
        run_unit_inner(deps, project_ref, report_path.clone()),
    )
    .await
    {
        Ok(report) => report,
        Err(_elapsed) => {
            let mut report = UnitReport::new(project_ref, report_path);
            report.outcome = UnitOutcome::Aborted {
                reason: WALL_CLOCK_ABORT_MSG.to_string(),
            };
            finalize(report)
        }
    }
}

/// The pipeline body, under the wall-clock budget.
///
/// The steps are pinned in one place (numbered (1)–(14), matching the
/// documented pipeline order); extracting them into helpers would hide the
/// early-return ordering the abort tests assert against.
#[allow(clippy::too_many_lines)]
async fn run_unit_inner(
    deps: &UnitDeps<'_>,
    project_ref: &str,
    report_path: PathBuf,
) -> UnitReport {
    let mut report = UnitReport::new(project_ref, report_path);

    // (1) The ref is charset-validated before anything else exists.
    if let Err(reason) = validate_project_ref(project_ref) {
        report.outcome = UnitOutcome::Aborted { reason };
        return finalize(report);
    }

    // (2) The leg-3 baseline, through the SAME loop-input source the run
    // fetches with (one endpoint, no second transport), guarded by the
    // run-start refusal: a loop that cannot prove work happened does not
    // get to run.
    let observer = MapPointerObserver::new(
        project_ref.to_string(),
        Arc::new(LoopInputCountSource::new(Arc::clone(&deps.source))),
    );
    let baseline = observer.observe(Path::new(".")).await;
    if let Err(reason) = crate::run_start_refusal(&baseline) {
        report.outcome = UnitOutcome::Aborted { reason };
        return finalize(report);
    }

    // (3) The fetch. 404/409 are terminal ORDINARY outcomes; everything
    // else that cannot produce a payload is a named abort.
    let input = match deps.source.fetch(project_ref).await {
        FetchOutcome::Ready(input) => input,
        FetchOutcome::UnknownProject => {
            report.outcome = UnitOutcome::UnknownProject;
            return finalize(report);
        }
        FetchOutcome::NotEligible => {
            report.outcome = UnitOutcome::NotEligible;
            return finalize(report);
        }
        FetchOutcome::Unauthorized => {
            report.outcome = UnitOutcome::Unauthorized;
            return finalize(report);
        }
        other => {
            report.outcome = UnitOutcome::Aborted {
                reason: format!(
                    "somnus: loop-input fetch failed: {}",
                    other.refusal_reason().unwrap_or_default()
                ),
            };
            return finalize(report);
        }
    };

    // (4) Pockets availability. A not-computed night skips the pair
    // statement (all three reasons) and still observes and reports.
    report.pockets_status = pockets_status(&input);
    if let PocketsStatus::NotComputed { reason } = &report.pockets_status {
        eprintln!("{}", render_pockets_not_computed_line(project_ref, reason));
        return observe_and_finalize(report, &observer, &baseline).await;
    }

    // (5) The decline ledger: match, then filter, then the loud count line.
    // A malformed ledger answer aborts the unit BEFORE rung 1 — running
    // unfiltered re-proposes declined clusters every night and pays Sonnet
    // for them, so it is never a fallback.
    let candidates: Vec<LedgerCandidate> = input
        .pockets
        .iter()
        .map(|pocket| LedgerCandidate {
            member_entry_ids: pocket.member_entry_ids.clone(),
        })
        .collect();
    let verdicts = match deps.ledger.match_candidates(project_ref, &candidates).await {
        Ok(verdicts) => verdicts,
        Err(err) => {
            report.outcome = UnitOutcome::Aborted {
                reason: err.to_string(),
            };
            return finalize(report);
        }
    };
    let (kept, declined) = match filter_pockets(&input.pockets, &verdicts) {
        Ok(pair) => pair,
        Err(err) => {
            report.outcome = UnitOutcome::Aborted {
                reason: err.to_string(),
            };
            return finalize(report);
        }
    };
    report.pockets_fetched = input.pockets.len();
    report.pockets_kept = kept.len();
    report.pockets_declined = declined.len();
    report.matched_verdicts = verdicts.iter().filter(|verdict| verdict.matched).count();
    report.ledger_verdicts = input
        .pockets
        .iter()
        .zip(verdicts.iter())
        .map(|(pocket, verdict)| LedgerVerdictRecord {
            index: verdict.index,
            member_entry_ids: pocket.member_entry_ids.clone(),
            matched: verdict.matched,
            ledger_id: verdict.ledger_id.clone(),
            status: verdict.status.clone(),
            jaccard: verdict.jaccard,
            reopen_eligible: verdict.reopen_eligible,
        })
        .collect();
    eprintln!(
        "{}",
        render_ledger_filter_line(
            project_ref,
            report.pockets_fetched,
            report.pockets_declined,
            report.matched_verdicts
        )
    );
    let mut filtered = input.clone();
    filtered.pockets = kept;

    // (6) Rung 1: one single-shot inference for the whole project.
    let turn = match rung1_turn(deps.backend, project_ref, &filtered).await {
        Ok(turn) => turn,
        Err(err) => {
            report.outcome = UnitOutcome::Aborted {
                reason: format!("somnus: rung-1 backend error: {err}"),
            };
            return finalize(report);
        }
    };
    report.backend_calls += 1;
    report.usage_rung1.add(&turn.usage);

    // A failed rung-1 parse aborts the unit (single-shot: no retry), with
    // the raw model text offloaded for a human to read.
    let clusters = match parse_clusters(&turn) {
        Ok(clusters) => clusters,
        Err(reason) => {
            let path = rung1_raw_path(&deps.body_root, project_ref);
            if let Err(err) = write_text(&path, &turn.text()) {
                eprintln!(
                    "somnus: could not offload the rung-1 raw text to {}: {err}",
                    path.display()
                );
            }
            report.rung1_raw_path = Some(path);
            report.outcome = UnitOutcome::Aborted { reason };
            return finalize(report);
        }
    };

    // (7)+(8) Rung 2: one single-shot inference per cluster, budget-checked
    // BEFORE each turn, with a per-cluster parse error skipping that
    // cluster and the run continuing.
    let mut budget_reason: Option<String> = None;
    for (index, cluster) in clusters.iter().enumerate() {
        if report.backend_calls >= u64::from(crate::SOMNUS_MAX_ITERATIONS) {
            budget_reason = Some(BUDGET_ABORT_MSG.to_string());
            break;
        }
        let mut record = ClusterRecord {
            label: cluster.label.clone(),
            member_entry_ids: cluster.member_entry_ids.clone(),
            owning_map_id: cluster.owning_map_id.clone(),
            ops: Vec::new(),
            rung2: Rung2Outcome::Parsed,
            raw_path: None,
        };
        let turn = match rung2_turn(deps.backend, project_ref, &filtered, cluster).await {
            Ok(turn) => turn,
            Err(err) => {
                record.rung2 = Rung2Outcome::ParseError {
                    reason: format!("somnus: rung-2 backend error: {err}"),
                };
                report.clusters.push(record);
                continue;
            }
        };
        report.backend_calls += 1;
        report.usage_rung2.add(&turn.usage);

        let mut ops = Vec::new();
        let mut parse_error = None;
        for call in turn.tool_calls() {
            match op_from_call(&call.name, &call.input) {
                Ok(op) => ops.push(op),
                Err(reason) => {
                    parse_error = Some(reason);
                    break;
                }
            }
        }
        // The lead-disposition guard rejects the whole op set BEFORE any
        // op is applied: an owned cluster must converge, never re-propose.
        let reason = parse_error
            .or_else(|| crate::materialize::validate_ops_for_cluster(cluster, &ops).err());
        if let Some(reason) = reason {
            let path = rung2_raw_path(&deps.body_root, project_ref, index);
            if let Err(err) = write_text(&path, &turn.text()) {
                eprintln!(
                    "somnus: could not offload the rung-2 raw text to {}: {err}",
                    path.display()
                );
            }
            record.raw_path = Some(path);
            record.rung2 = Rung2Outcome::ParseError { reason };
            report.clusters.push(record);
            continue;
        }
        record.ops = ops;
        report.clusters.push(record);
    }
    if let Some(reason) = budget_reason {
        report.outcome = UnitOutcome::Aborted { reason };
    }

    // (9) The decline write: a cluster whose op set is empty or only
    // `propose_gap` is recorded, so the ledger filter has something to
    // match on night two.
    for record in &report.clusters {
        if matches!(record.rung2, Rung2Outcome::ParseError { .. }) {
            continue;
        }
        let declined = record.ops.is_empty()
            || record
                .ops
                .iter()
                .all(|op| matches!(op, Op::ProposeGap { .. }));
        if !declined {
            continue;
        }
        match deps
            .ledger
            .record_decline(project_ref, &record.member_entry_ids)
            .await
        {
            Ok(()) => {
                eprintln!(
                    "{}",
                    render_decline_recorded_line(project_ref, &record.member_entry_ids)
                );
                report
                    .declines_recorded
                    .push(record.member_entry_ids.clone());
            }
            Err(err) => {
                // Name the cluster whose decline was lost BEFORE the
                // fail-closed abort return, so a multi-decline loop that
                // aborts says which one did not land.
                eprintln!(
                    "{}",
                    render_decline_write_failed_line(project_ref, &record.member_entry_ids)
                );
                // Fail closed: a decline that cannot be recorded would be
                // re-proposed every night, so the run does not claim success.
                report.outcome = UnitOutcome::Aborted {
                    reason: err.to_string(),
                };
                return finalize(report);
            }
        }
    }

    // (10) Rung 3: compose the bodies, recording every refusal by name.
    let mut new_maps_so_far = 0u32;
    let mut composed: Vec<ComposedBody> = Vec::new();
    for record in &report.clusters {
        if matches!(record.rung2, Rung2Outcome::ParseError { .. }) {
            continue;
        }
        let Materialized {
            bodies,
            authorship_refusals,
            cap_refusals,
            compose_refusals,
            new_maps,
        } = materialize_cluster(
            project_ref,
            &record.cluster(),
            &record.ops,
            &input,
            new_maps_so_far,
        );
        new_maps_so_far += new_maps;
        report.authorship_refusals.extend(authorship_refusals);
        report.cap_refusals.extend(cap_refusals);
        report.compose_refusals.extend(compose_refusals);
        composed.extend(bodies);
    }
    for refusal in &report.cap_refusals {
        eprintln!("{refusal}");
    }

    // (11) Every composed body goes on disk (the gate posts a file), and
    // (12) every composed body goes through the gate.
    for body in composed {
        match write_composed_body(&deps.body_root, project_ref, &body) {
            Ok(path) => report.composed_bodies.push(ComposedBodyRecord {
                map_id: body.map_id.clone(),
                body_path: path,
                body: body.body,
            }),
            Err(err) => {
                report.outcome = UnitOutcome::Aborted {
                    reason: format!(
                        "somnus: could not write the composed body for {}: {err}",
                        body.map_id
                    ),
                };
                return finalize(report);
            }
        }
    }
    for record in &report.composed_bodies {
        let runner = (deps.gate_for)(&record.body_path);
        let gate_report = runner.run(deps.tool_ctx).await;
        report.gate_reports.push(GateReportRecord {
            map_id: record.map_id.clone(),
            passed: gate_report.passed,
            exit_code: gate_report.exit_code,
            timed_out: gate_report.timed_out,
        });
    }

    // (13) The final observation through the SAME observer (cached
    // baseline, fail closed) and the leg-3 classification.
    observe_and_finalize(report, &observer, &baseline).await
}

/// Observe once more through the same observer, classify the change, audit
/// the call count, and write the report.
async fn observe_and_finalize(
    mut report: UnitReport,
    observer: &MapPointerObserver,
    baseline: &harness::exec::TreeObservation,
) -> UnitReport {
    let current = observer.observe(Path::new(".")).await;
    report.change = classify_change(baseline, &current);
    report.call_count_audit =
        render_call_count_audit_line(report.backend_calls, report.clusters.len());
    finalize(report)
}

/// Write the report to disk and hand it back. A report that cannot be
/// written is announced loudly — the run's outcome is already decided, so
/// the stderr line is the last resort, never a panic.
fn finalize(report: UnitReport) -> UnitReport {
    let text = serde_json::to_string_pretty(&report).expect(
        "the run report serializes by construction (every field is Serialize with no renames)",
    );
    if let Err(err) = write_text(&report.report_path, &text) {
        eprintln!(
            "somnus: could not write the run report at {}: {err}",
            report.report_path.display()
        );
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use harness::model::{
        AssistantTurn, BackendError, ContentBlock, StopReason, ToolCallRequest, TurnRequest, Usage,
    };
    use harness::test_support::MockBackend;
    use serde_json::{Value, json};

    use crate::ledger::InMemoryLedger;
    use crate::loop_input::FIXTURE;
    use crate::loop_input::{LoopInput, fixture};
    use crate::ops::op_tool_schemas;

    // ======================================================================
    // fakes: a scripted source, a failure-mode ledger, a never-resolving
    // backend, and a gate that runs `/bin/sh -c "exit N"`.
    // ======================================================================

    /// A scripted [`LoopInputSource`]: pops front-to-back, the last entry
    /// repeats. No network, no live KB.
    #[derive(Debug)]
    struct ScriptedSource(std::sync::Mutex<Vec<FetchOutcome>>);

    impl ScriptedSource {
        fn new(script: Vec<FetchOutcome>) -> Self {
            Self(std::sync::Mutex::new(script))
        }
    }

    #[async_trait]
    impl LoopInputSource for ScriptedSource {
        async fn fetch(&self, _project_ref: &str) -> FetchOutcome {
            let mut script = self.0.lock().expect("script lock poisoned");
            if script.len() > 1 {
                script.remove(0)
            } else {
                script[0].clone()
            }
        }
    }

    /// Which way the [`FakeLedger`] is broken, so one fake covers every
    /// ledger-failure branch the pipeline can take.
    #[derive(Debug, Clone, Copy)]
    enum FakeLedgerMode {
        /// `match_candidates` returns a transport error.
        MatchError,
        /// `match_candidates` answers fewer verdicts than it was asked for.
        ShortVerdicts,
        /// `match_candidates` answers verdicts with the wrong indices.
        MisalignedVerdicts,
        /// `record_decline` returns a transport error.
        DeclineError,
    }

    #[derive(Debug)]
    struct FakeLedger(FakeLedgerMode);

    #[async_trait]
    impl crate::ledger::ClusterLedger for FakeLedger {
        async fn match_candidates(
            &self,
            _project_ref: &str,
            candidates: &[crate::ledger::LedgerCandidate],
        ) -> Result<Vec<crate::ledger::CandidateVerdict>, crate::ledger::LedgerError> {
            match self.0 {
                FakeLedgerMode::MatchError => Err(crate::ledger::LedgerError::Source {
                    reason: "ledger endpoint unavailable".to_string(),
                }),
                FakeLedgerMode::ShortVerdicts => Ok(Vec::new()),
                FakeLedgerMode::MisalignedVerdicts => Ok(candidates
                    .iter()
                    .enumerate()
                    .map(|(index, _candidate)| crate::ledger::CandidateVerdict {
                        index: index + 7,
                        matched: false,
                        ledger_id: None,
                        status: None,
                        jaccard: None,
                        reopen_eligible: false,
                    })
                    .collect()),
                FakeLedgerMode::DeclineError => Ok(candidates
                    .iter()
                    .enumerate()
                    .map(|(index, _)| crate::ledger::CandidateVerdict {
                        index,
                        matched: false,
                        ledger_id: None,
                        status: None,
                        jaccard: None,
                        reopen_eligible: false,
                    })
                    .collect()),
            }
        }

        async fn record_decline(
            &self,
            _project_ref: &str,
            member_entry_ids: &[String],
        ) -> Result<(), crate::ledger::LedgerError> {
            let _ = member_entry_ids;
            if matches!(self.0, FakeLedgerMode::DeclineError) {
                return Err(crate::ledger::LedgerError::Source {
                    reason: "ledger write unavailable".to_string(),
                });
            }
            Ok(())
        }
    }

    /// A backend whose `turn` never resolves, for the paused-clock expiry
    /// test.
    #[derive(Debug)]
    struct PendBackend;

    #[async_trait]
    impl ModelBackend for PendBackend {
        async fn turn(&self, _req: &TurnRequest<'_>) -> Result<AssistantTurn, BackendError> {
            std::future::pending().await
        }
    }

    /// A gate factory whose children run `/bin/sh -c "{script}"`: `exit 0`
    /// is a green leg, `exit 22` is the `curl -f` 422 shape. No real curl is
    /// ever executed.
    fn gate_for_script(script: &'static str) -> impl Fn(&Path) -> ChecksRunner {
        move |_: &Path| {
            ChecksRunner::new(
                harness::exec::CheckCommand {
                    program: "/bin/sh".to_string(),
                    args: vec!["-c".to_string(), script.to_string()],
                },
                std::path::PathBuf::from("."),
                crate::gate::SOMNUS_GATE_TIMEOUT,
            )
        }
    }

    /// Run one unit against the injected fakes and return the report.
    async fn run_with(
        backend: &dyn ModelBackend,
        source: std::sync::Arc<dyn LoopInputSource>,
        ledger: &dyn ClusterLedger,
        gate_for: &dyn Fn(&Path) -> ChecksRunner,
        body_root: &Path,
        project_ref: &str,
    ) -> UnitReport {
        let tool_ctx = ToolCtx::stub();
        let deps = UnitDeps {
            backend,
            source,
            ledger,
            gate_for,
            tool_ctx: &tool_ctx,
            body_root: body_root.to_path_buf(),
        };
        run_unit(&deps, project_ref).await
    }

    fn usage(input: u32, output: u32, read: Option<u32>, write: Option<u32>) -> Usage {
        Usage {
            input_tokens: input,
            output_tokens: output,
            cache_read_tokens: read,
            cache_write_tokens: write,
            reasoning_tokens: None,
        }
    }

    fn text_turn(text: &str, usage: Usage) -> AssistantTurn {
        AssistantTurn {
            content: vec![ContentBlock::Text(text.to_string())],
            stop_reason: StopReason::EndTurn,
            usage,
        }
    }

    fn calls_turn(calls: &[(&str, Value)], usage: Usage) -> AssistantTurn {
        AssistantTurn {
            content: calls
                .iter()
                .map(|(name, input)| {
                    ContentBlock::ToolCall(ToolCallRequest {
                        id: format!("call-{name}"),
                        name: (*name).to_string(),
                        input: input.clone(),
                    })
                })
                .collect(),
            stop_reason: StopReason::ToolUse,
            usage,
        }
    }

    /// The two-cluster rung-1 answer the e2e scripts.
    ///
    /// Note on `owning_map_id`: the e2e sketch in the task description pairs
    /// the `create_map` cluster with an owning map, but the lead-disposition
    /// rule (see [`crate::materialize::validate_ops_for_cluster`]) rejects
    /// exactly that pairing BEFORE any op is applied, so a conforming e2e
    /// cannot carry both. The cluster here is unowned — the `Lives in` value
    /// still derives from `kb-20001` (the first map in loop-input order with
    /// such a line), so every pinned body literal holds — and the owned-
    /// cluster rejection is pinned by its own test:
    /// `create_map_for_an_owned_cluster_is_rejected_by_the_pipeline`.
    fn e2e_clusters_json() -> String {
        json!([
            {
                "label": "wireguard-and-dns",
                "member_entry_ids": ["kb-10001", "kb-10002", "kb-10003"],
                "owning_map_id": null
            },
            {
                "label": "backup-drills",
                "member_entry_ids": ["kb-10006", "kb-10007"],
                "owning_map_id": null
            }
        ])
        .to_string()
    }

    /// A fixture variant with a `pockets_omitted_reason`, for the
    /// not-computed night.
    fn not_computed_fixture(reason: &str) -> LoopInput {
        let mut input = fixture();
        input.pockets_omitted_reason = Some(reason.to_string());
        input.pockets.clear();
        input
    }

    /// The fixture with the KB having absorbed this run's writes: one new
    /// map plus the pointer added to `kb-20001`, so the leg-3 count moves.
    fn bumped_fixture_json() -> String {
        let mut value: Value = serde_json::from_str(FIXTURE).expect("fixture parses");
        value["maps"][0]["pointers"]
            .as_array_mut()
            .expect("pointers")
            .push(json!("kb-10005"));
        value["maps"].as_array_mut().expect("maps").push(json!({
            "id": "somnus-new-c1",
            "short_title": "Wireguard and DNS",
            "long_title": "Wireguard and DNS orientation map",
            "pointers": ["kb-10001", "kb-10002"],
            "body": "Lives in knowledge/linux/network\n\nORIENTATION-PROSE\n\nDetail entries:\n- kb-10001 — GLOSS-1\n- kb-10002 — GLOSS-2",
            "contributor": "somnus",
            "updated_by": "somnus"
        }));
        value.to_string()
    }

    /// A wiremock `Respond` that serves a scripted sequence of bodies,
    /// front-to-back, repeating the last.
    #[derive(Debug)]
    struct SequencedBodies {
        bodies: std::sync::Mutex<Vec<String>>,
    }

    impl wiremock::Respond for SequencedBodies {
        fn respond(&self, _request: &wiremock::Request) -> wiremock::ResponseTemplate {
            let mut bodies = self.bodies.lock().expect("bodies lock poisoned");
            if bodies.len() > 1 {
                wiremock::ResponseTemplate::new(200).set_body_string(bodies.remove(0))
            } else {
                wiremock::ResponseTemplate::new(200).set_body_string(bodies[0].clone())
            }
        }
    }

    // ======================================================================
    // the e2e night: wiremock + MockBackend + InMemoryLedger + tempdir
    // ======================================================================

    #[allow(clippy::too_many_lines)] // the e2e asserts every report field in one place
    #[tokio::test]
    async fn the_full_night_runs_end_to_end() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/api/kb/map-loop-input"))
            .respond_with(SequencedBodies {
                bodies: std::sync::Mutex::new(vec![
                    FIXTURE.to_string(),
                    FIXTURE.to_string(),
                    bumped_fixture_json(),
                ]),
            })
            .mount(&server)
            .await;
        let source = std::sync::Arc::new(crate::loop_input::HttpLoopInputSource::new(
            server.uri(),
            "kb-token".to_string(),
        ));

        // Pre-seeded decline: P2 ([kb-10004, kb-10005]) was declined on a
        // previous night and is not yet reopen-eligible.
        let ledger = InMemoryLedger::new();
        ledger
            .record_decline(
                "demo-project",
                &["kb-10004".to_string(), "kb-10005".to_string()],
            )
            .await
            .expect("seed the decline");

        let backend = MockBackend::from_turns(vec![
            text_turn(&e2e_clusters_json(), usage(1000, 50, Some(200), None)),
            calls_turn(
                &[
                    (
                        "create_map",
                        json!({"cluster_id": "c1", "title": "Wireguard and DNS", "orientation_prose": "ORIENTATION-PROSE"}),
                    ),
                    (
                        "add_pointer",
                        json!({"map_id": "somnus-new-c1", "entry_id": "kb-10001", "gloss": "GLOSS-1"}),
                    ),
                    (
                        "add_pointer",
                        json!({"map_id": "somnus-new-c1", "entry_id": "kb-10002", "gloss": "GLOSS-2"}),
                    ),
                    (
                        "add_pointer",
                        json!({"map_id": "kb-20001", "entry_id": "kb-10005", "gloss": "GLOSS-5"}),
                    ),
                ],
                usage(900, 30, Some(100), Some(10)),
            ),
            calls_turn(
                &[(
                    "propose_gap",
                    json!({"cluster_id": "c2", "reason": "needs an upstream authoring run"}),
                )],
                usage(900, 30, Some(100), Some(10)),
            ),
        ]);

        let body_root = tempfile::tempdir().expect("tempdir");
        let gate = gate_for_script("exit 0");
        let report = run_with(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
        )
        .await;

        // --- the inference discipline -----------------------------------
        assert_eq!(backend.calls(), 3, "1 rung-1 + 2 rung-2 turns");
        assert_eq!(
            backend.params_seen(),
            vec![128_000, 128_000, 128_000],
            "every turn runs at the pinned output cap"
        );
        assert_eq!(backend.systems_seen(), vec![None::<String>; 3]);
        let tools_seen = backend.tools_seen();
        assert_eq!(tools_seen.len(), 3);
        assert!(
            tools_seen[0].is_empty(),
            "rung 1 advertises NOTHING callable"
        );
        assert_eq!(tools_seen[1], op_tool_schemas());
        assert_eq!(tools_seen[2], op_tool_schemas());

        // --- the injected payload is the ledger-filtered one -------------
        let messages = backend.messages_seen();
        let second = &messages[0][1];
        let harness::model::Message::User { content } = second else {
            panic!("the second message is the tool result");
        };
        let harness::model::UserBlock::ToolResult { content, .. } = &content[0] else {
            panic!("the tool result carries the payload");
        };
        let payload: LoopInput = serde_json::from_str(content).expect("the payload parses");
        assert_eq!(
            payload.pockets.len(),
            1,
            "the declined pocket never reaches the model"
        );
        assert_eq!(
            payload.pockets[0].member_entry_ids,
            vec![
                "kb-10001".to_string(),
                "kb-10002".to_string(),
                "kb-10003".to_string(),
            ]
        );

        // --- the two composed bodies, byte-for-byte ----------------------
        assert_eq!(report.composed_bodies.len(), 2);
        assert_eq!(
            report.composed_bodies[0],
            ComposedBodyRecord {
                map_id: "somnus-new-c1".to_string(),
                body_path: crate::materialize::map_body_path(
                    body_root.path(),
                    "demo-project",
                    "somnus-new-c1"
                ),
                body: "Lives in knowledge/linux/network\n\nORIENTATION-PROSE\n\nDetail entries:\n- kb-10001 — GLOSS-1\n- kb-10002 — GLOSS-2".to_string(),
            }
        );
        assert_eq!(
            report.composed_bodies[1].map_id, "kb-20001",
            "the read-modify-write body"
        );
        assert_eq!(
            report.composed_bodies[1].body,
            "Lives in knowledge/linux/network\n\nOrients the switch, DNS, and VPN entries for the home network.\n\nDetail entries:\n- kb-10001 — switch VLAN tagging baseline\n- kb-10002 — recursive DNS resolvers\n- kb-10003 — wireguard client config\n- kb-99999 — cross-project wireguard hub notes\n- kb-10004 — DHCP lease hygiene\n- kb-10005 — GLOSS-5\n\nNot yet documented:\n- Site-to-site wireguard topology"
        );
        for record in &report.composed_bodies {
            assert_eq!(
                std::fs::read_to_string(&record.body_path).expect("the body is on disk"),
                record.body,
                "the gate posts a file, so the file must exist"
            );
        }

        // --- the gate legs ------------------------------------------------
        assert_eq!(report.gate_reports.len(), 2);
        for gate_report in &report.gate_reports {
            assert!(gate_report.passed, "the stub gate exits 0");
            assert_eq!(gate_report.exit_code, Some(0));
            assert!(!gate_report.timed_out);
        }

        // --- leg 3 and the ledger write -----------------------------------
        assert_eq!(report.change, ChangeEvidence::TreeChanged);
        assert_eq!(
            ledger.recorded_declines("demo-project"),
            vec![
                vec!["kb-10004".to_string(), "kb-10005".to_string()],
                vec!["kb-10006".to_string(), "kb-10007".to_string()],
            ],
            "the seeded decline plus this run's propose_gap decline"
        );
        assert_eq!(
            report.declines_recorded,
            vec![vec!["kb-10006".to_string(), "kb-10007".to_string(),]]
        );

        // --- the report -----------------------------------------------------
        assert_eq!(report.outcome, UnitOutcome::Ready);
        assert_eq!(report.project_ref, "demo-project");
        assert_eq!(report.pockets_fetched, 2);
        assert_eq!(report.pockets_kept, 1);
        assert_eq!(report.pockets_declined, 1);
        assert_eq!(report.matched_verdicts, 1);
        assert_eq!(
            report.ledger_verdicts,
            vec![
                LedgerVerdictRecord {
                    index: 0,
                    member_entry_ids: vec![
                        "kb-10001".to_string(),
                        "kb-10002".to_string(),
                        "kb-10003".to_string(),
                    ],
                    matched: false,
                    ledger_id: None,
                    status: None,
                    jaccard: None,
                    reopen_eligible: false,
                },
                LedgerVerdictRecord {
                    index: 1,
                    member_entry_ids: vec!["kb-10004".to_string(), "kb-10005".to_string(),],
                    matched: true,
                    ledger_id: Some("decline-1".to_string()),
                    status: Some("declined".to_string()),
                    jaccard: None,
                    reopen_eligible: false,
                },
            ]
        );
        assert_eq!(report.pockets_status, PocketsStatus::NoneFound);
        assert_eq!(report.backend_calls, 3);
        assert_eq!(
            report.usage_rung1,
            UsageTotals {
                input: 1000,
                output: 50,
                cache_read: 200,
                cache_write: 0,
            }
        );
        assert_eq!(
            report.usage_rung2,
            UsageTotals {
                input: 1800,
                output: 60,
                cache_read: 200,
                cache_write: 20,
            }
        );
        assert_eq!(
            report.call_count_audit, None,
            "3 calls for 2 clusters is exactly the single-shot shape"
        );
        assert!(report.authorship_refusals.is_empty());
        assert!(report.cap_refusals.is_empty());
        assert!(report.compose_refusals.is_empty());
        assert!(report.rung1_raw_path.is_none());
        assert_eq!(
            report.report_path,
            crate::materialize::run_report_path(body_root.path(), "demo-project")
        );
        let on_disk: Value = {
            let text = std::fs::read_to_string(&report.report_path).expect("the report is on disk");
            serde_json::from_str(&text).expect("the report parses as JSON")
        };
        // Every pinned field is present in the on-disk report.
        for field in [
            "project_ref",
            "outcome",
            "pockets_fetched",
            "pockets_kept",
            "pockets_declined",
            "matched_verdicts",
            "ledger_verdicts",
            "pockets_status",
            "clusters",
            "declines_recorded",
            "authorship_refusals",
            "cap_refusals",
            "compose_refusals",
            "composed_bodies",
            "gate_reports",
            "change",
            "backend_calls",
            "usage_rung1",
            "usage_rung2",
            "call_count_audit",
            "rung1_raw_path",
            "report_path",
        ] {
            assert!(
                on_disk.get(field).is_some(),
                "the on-disk report must carry `{field}`: {on_disk}"
            );
        }
        assert_eq!(on_disk["project_ref"], "demo-project");
        assert_eq!(on_disk["outcome"], "Ready");
        assert_eq!(on_disk["change"], "TreeChanged");
        assert_eq!(on_disk["usage_rung1"]["cache_read"], 200);
        assert_eq!(on_disk["usage_rung2"]["cache_write"], 20);
        assert_eq!(on_disk["clusters"].as_array().map(Vec::len), Some(2));
        assert_eq!(on_disk["composed_bodies"].as_array().map(Vec::len), Some(2));
        assert_eq!(on_disk["gate_reports"].as_array().map(Vec::len), Some(2));
        assert_eq!(
            on_disk["declines_recorded"][0],
            json!(["kb-10006", "kb-10007"])
        );
        assert_eq!(report.clusters.len(), 2);
        assert_eq!(report.clusters[0].label, "wireguard-and-dns");
        assert_eq!(report.clusters[0].rung2, Rung2Outcome::Parsed);
        assert_eq!(report.clusters[0].ops.len(), 4);
        assert_eq!(report.clusters[1].label, "backup-drills");
        assert_eq!(report.clusters[1].rung2, Rung2Outcome::Parsed);
    }

    #[tokio::test]
    async fn a_red_gate_stub_records_a_failed_gate_report() {
        let backend = MockBackend::from_turns(vec![
            text_turn(&e2e_clusters_json(), usage(1, 1, None, None)),
            calls_turn(
                &[(
                    "add_pointer",
                    json!({"map_id": "kb-20001", "entry_id": "kb-10005", "gloss": "GLOSS-5"}),
                )],
                usage(1, 1, None, None),
            ),
            calls_turn(
                &[("propose_gap", json!({"cluster_id": "c2", "reason": "r"}))],
                usage(1, 1, None, None),
            ),
        ]);
        let source = std::sync::Arc::new(ScriptedSource::new(vec![FetchOutcome::Ready(fixture())]));
        let ledger = InMemoryLedger::new();
        let body_root = tempfile::tempdir().expect("tempdir");
        let gate = gate_for_script("exit 22");
        let report = run_with(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
        )
        .await;
        assert_eq!(report.composed_bodies.len(), 1);
        assert_eq!(report.gate_reports.len(), 1);
        assert!(!report.gate_reports[0].passed, "`curl -f` maps 422 to red");
        assert_eq!(report.gate_reports[0].exit_code, Some(22));
        assert_eq!(report.outcome, UnitOutcome::Ready);
    }

    // ======================================================================
    // the terminal fetch outcomes
    // ======================================================================

    #[allow(clippy::too_many_lines)] // six outcomes, one table
    #[tokio::test]
    async fn every_fetch_outcome_maps_to_its_pinned_unit_outcome() {
        let terminal = [
            (FetchOutcome::UnknownProject, UnitOutcome::UnknownProject),
            (FetchOutcome::NotEligible, UnitOutcome::NotEligible),
            (FetchOutcome::Unauthorized, UnitOutcome::Unauthorized),
        ];
        for (fetch, expected) in terminal {
            let backend = MockBackend::from_turns(vec![]);
            let source = std::sync::Arc::new(ScriptedSource::new(vec![
                FetchOutcome::Ready(fixture()),
                fetch.clone(),
            ]));
            let ledger = InMemoryLedger::new();
            let body_root = tempfile::tempdir().expect("tempdir");
            let gate = gate_for_script("exit 0");
            let report = run_with(
                &backend,
                source,
                &ledger,
                &gate,
                body_root.path(),
                "demo-project",
            )
            .await;
            assert_eq!(report.outcome, expected, "{fetch:?}");
            assert!(
                backend.calls() == 0,
                "terminal outcomes are recorded with ZERO backend calls"
            );
        }
        for fetch in [
            FetchOutcome::BadStatus { status: 500 },
            FetchOutcome::MalformedBody {
                reason: "garbage".to_string(),
            },
            FetchOutcome::Unreachable {
                reason: "refused".to_string(),
            },
        ] {
            let backend = MockBackend::from_turns(vec![]);
            let source = std::sync::Arc::new(ScriptedSource::new(vec![
                FetchOutcome::Ready(fixture()),
                fetch.clone(),
            ]));
            let ledger = InMemoryLedger::new();
            let body_root = tempfile::tempdir().expect("tempdir");
            let gate = gate_for_script("exit 0");
            let report = run_with(
                &backend,
                source,
                &ledger,
                &gate,
                body_root.path(),
                "demo-project",
            )
            .await;
            let UnitOutcome::Aborted { reason } = &report.outcome else {
                panic!("{fetch:?} must abort the unit, got {:?}", report.outcome);
            };
            assert!(
                reason.starts_with("somnus: loop-input fetch failed:"),
                "{fetch:?} reason was {reason}"
            );
            assert_eq!(backend.calls(), 0);
        }
    }

    #[tokio::test]
    async fn an_invalid_project_ref_aborts_before_any_work() {
        let backend = MockBackend::from_turns(vec![]);
        let source = std::sync::Arc::new(ScriptedSource::new(vec![FetchOutcome::Ready(fixture())]));
        let ledger = InMemoryLedger::new();
        let body_root = tempfile::tempdir().expect("tempdir");
        let gate = gate_for_script("exit 0");
        let report = run_with(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "bad/ref",
        )
        .await;
        assert_eq!(
            report.outcome,
            UnitOutcome::Aborted {
                reason: crate::loop_input::INVALID_PROJECT_REF_MSG.to_string(),
            }
        );
        assert_eq!(backend.calls(), 0);
    }

    #[tokio::test]
    async fn an_unobservable_baseline_refuses_to_start() {
        // The very first count-source call fails, so the run-start baseline
        // is `Unobservable` and the run-start guard refuses to start: no
        // backend call, no fetch beyond the baseline.
        let backend = MockBackend::from_turns(vec![]);
        let source = std::sync::Arc::new(ScriptedSource::new(vec![FetchOutcome::UnknownProject]));
        let ledger = InMemoryLedger::new();
        let body_root = tempfile::tempdir().expect("tempdir");
        let gate = gate_for_script("exit 0");
        let report = run_with(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
        )
        .await;
        assert_eq!(
            report.outcome,
            UnitOutcome::Aborted {
                reason: "somnus: refusing to start: run-start baseline could not be observed (pointer count source failed: loop-input fetch: project_ref not found)".to_string(),
            }
        );
        assert_eq!(backend.calls(), 0);
    }

    #[tokio::test]
    async fn a_not_computed_night_skips_the_pair_statement() {
        for reason in [
            "too-few-unpointed-entries",
            "unpointed-set-too-large",
            "non-postgres-backend",
        ] {
            let backend = MockBackend::from_turns(vec![]);
            let source = std::sync::Arc::new(ScriptedSource::new(vec![FetchOutcome::Ready(
                not_computed_fixture(reason),
            )]));
            let ledger = InMemoryLedger::new();
            let body_root = tempfile::tempdir().expect("tempdir");
            let gate = gate_for_script("exit 0");
            let report = run_with(
                &backend,
                source,
                &ledger,
                &gate,
                body_root.path(),
                "demo-project",
            )
            .await;
            assert_eq!(report.outcome, UnitOutcome::Ready, "{reason}");
            assert_eq!(
                report.pockets_status,
                PocketsStatus::NotComputed {
                    reason: reason.to_string(),
                }
            );
            assert_eq!(backend.calls(), 0, "{reason}: nothing to cluster");
            assert!(report.clusters.is_empty());
            assert_eq!(report.change, ChangeEvidence::TreeUnchanged);
        }
    }

    // ======================================================================
    // the decline ledger: match, filter, and the decline write
    // ======================================================================

    #[tokio::test]
    async fn a_ledger_match_error_aborts_the_unit_before_rung1() {
        let backend = MockBackend::from_turns(vec![text_turn("[]", usage(1, 1, None, None))]);
        let source = std::sync::Arc::new(ScriptedSource::new(vec![FetchOutcome::Ready(fixture())]));
        let ledger = FakeLedger(FakeLedgerMode::MatchError);
        let body_root = tempfile::tempdir().expect("tempdir");
        let gate = gate_for_script("exit 0");
        let report = run_with(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
        )
        .await;
        assert_eq!(
            report.outcome,
            UnitOutcome::Aborted {
                reason: "somnus: ledger source error: ledger endpoint unavailable".to_string(),
            }
        );
        assert_eq!(backend.calls(), 0, "running unfiltered is never a fallback");
        // The report IS written by `finalize` on every path, so a ledger
        // abort is post-hoc greppable in the on-disk JSON.
        let text = std::fs::read_to_string(&report.report_path).expect("the report is on disk");
        assert!(
            text.contains("somnus: ledger source error: ledger endpoint unavailable"),
            "the abort reason must be greppable on disk: {text}"
        );
    }

    #[tokio::test]
    async fn a_malformed_verdict_vector_aborts_the_unit_before_rung1() {
        for mode in [
            FakeLedgerMode::ShortVerdicts,
            FakeLedgerMode::MisalignedVerdicts,
        ] {
            let backend = MockBackend::from_turns(vec![text_turn("[]", usage(1, 1, None, None))]);
            let source =
                std::sync::Arc::new(ScriptedSource::new(vec![FetchOutcome::Ready(fixture())]));
            let ledger = FakeLedger(mode);
            let body_root = tempfile::tempdir().expect("tempdir");
            let gate = gate_for_script("exit 0");
            let report = run_with(
                &backend,
                source,
                &ledger,
                &gate,
                body_root.path(),
                "demo-project",
            )
            .await;
            let UnitOutcome::Aborted { reason } = &report.outcome else {
                panic!("{mode:?} must abort, got {:?}", report.outcome);
            };
            assert!(reason.starts_with("somnus: ledger"), "{mode:?}: {reason}");
            assert_eq!(backend.calls(), 0, "{mode:?}: rung 1 never runs");
        }
    }

    #[tokio::test]
    async fn a_decline_write_failure_fails_closed() {
        let backend = MockBackend::from_turns(vec![
            text_turn(&e2e_clusters_json(), usage(1, 1, None, None)),
            calls_turn(
                &[("propose_gap", json!({"cluster_id": "c1", "reason": "r"}))],
                usage(1, 1, None, None),
            ),
            calls_turn(
                &[("propose_gap", json!({"cluster_id": "c2", "reason": "r"}))],
                usage(1, 1, None, None),
            ),
        ]);
        let source = std::sync::Arc::new(ScriptedSource::new(vec![FetchOutcome::Ready(fixture())]));
        let ledger = FakeLedger(FakeLedgerMode::DeclineError);
        let body_root = tempfile::tempdir().expect("tempdir");
        let gate = gate_for_script("exit 0");
        let report = run_with(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
        )
        .await;
        // The decline write happens before materialization: nothing is
        // composed, because a decline that cannot be recorded would be
        // re-proposed every night.
        assert_eq!(
            report.outcome,
            UnitOutcome::Aborted {
                reason: "somnus: ledger source error: ledger write unavailable".to_string(),
            }
        );
        assert!(report.composed_bodies.is_empty());
    }

    // ======================================================================
    // rung 1
    // ======================================================================

    #[tokio::test]
    async fn a_rung1_backend_error_aborts_the_unit() {
        let backend = MockBackend::new(vec![Err(BackendError::Terminal {
            kind: harness::model::TerminalKind::Auth,
            message: "bad key".to_string(),
        })]);
        let source = std::sync::Arc::new(ScriptedSource::new(vec![FetchOutcome::Ready(fixture())]));
        let ledger = InMemoryLedger::new();
        let body_root = tempfile::tempdir().expect("tempdir");
        let gate = gate_for_script("exit 0");
        let report = run_with(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
        )
        .await;
        let UnitOutcome::Aborted { reason } = &report.outcome else {
            panic!("expected an abort, got {:?}", report.outcome);
        };
        assert!(
            reason.contains("rung-1 backend error"),
            "reason was {reason}"
        );
        // `backend_calls` counts turns that returned: the failed call
        // produced no turn, no usage, and no cluster work.
        assert_eq!(report.backend_calls, 0);
    }

    #[tokio::test]
    async fn a_failed_rung1_parse_aborts_and_offloads_the_raw_text() {
        let backend = MockBackend::from_turns(vec![text_turn("not json", usage(1, 1, None, None))]);
        let source = std::sync::Arc::new(ScriptedSource::new(vec![FetchOutcome::Ready(fixture())]));
        let ledger = InMemoryLedger::new();
        let body_root = tempfile::tempdir().expect("tempdir");
        let gate = gate_for_script("exit 0");
        let report = run_with(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
        )
        .await;
        let UnitOutcome::Aborted { reason } = &report.outcome else {
            panic!("expected an abort, got {:?}", report.outcome);
        };
        assert!(
            reason.starts_with("somnus: rung-1 parse failed"),
            "reason was {reason}"
        );
        let raw_path = report
            .rung1_raw_path
            .as_ref()
            .expect("the raw text path is recorded");
        assert_eq!(
            raw_path,
            &crate::rungs::rung1_raw_path(body_root.path(), "demo-project")
        );
        assert_eq!(
            std::fs::read_to_string(raw_path).expect("the raw text is on disk"),
            "not json"
        );
        // Single-shot: no second rung-1 call.
        assert_eq!(backend.calls(), 1);
        assert!(report.clusters.is_empty());
    }

    // ======================================================================
    // the per-unit inference budget
    // ======================================================================

    #[tokio::test]
    async fn the_inference_budget_aborts_rung2_at_twenty_four_calls() {
        // A 30-cluster rung-1 answer: 1 rung-1 inference + at most 23
        // rung-2 inferences = the 24-call cap, count-based (no clock).
        let mut clusters = Vec::new();
        for index in 0..30 {
            clusters.push(json!({
                "label": format!("cluster-{index}"),
                "member_entry_ids": [format!("kb-1000{index}")],
                "owning_map_id": null
            }));
        }
        let mut script = vec![text_turn(
            &Value::Array(clusters).to_string(),
            usage(1, 1, None, None),
        )];
        for _ in 0..23 {
            script.push(calls_turn(
                &[("propose_gap", json!({"cluster_id": "c", "reason": "r"}))],
                usage(1, 1, None, None),
            ));
        }
        let backend = MockBackend::from_turns(script);
        let source = std::sync::Arc::new(ScriptedSource::new(vec![FetchOutcome::Ready(fixture())]));
        let ledger = InMemoryLedger::new();
        let body_root = tempfile::tempdir().expect("tempdir");
        let gate = gate_for_script("exit 0");
        let report = run_with(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
        )
        .await;
        assert_eq!(backend.calls(), 24, "1 rung-1 + 23 rung-2, then the cap");
        assert_eq!(report.backend_calls, 24);
        assert_eq!(
            report.outcome,
            UnitOutcome::Aborted {
                reason: BUDGET_ABORT_MSG.to_string(),
            }
        );
        // 24 calls for 23 processed clusters is exactly `1 + cluster_count`,
        // so the tripwire stays silent: the cap fired, the discipline did
        // not break.
        assert_eq!(report.call_count_audit, None);
        assert_eq!(
            report.declines_recorded.len(),
            23,
            "the paid work before the cap is still recorded"
        );
    }

    // ======================================================================
    // rung 2: per-cluster fates
    // ======================================================================

    #[tokio::test]
    async fn a_bad_cluster_never_discards_the_other_clusters_paid_work() {
        // Three clusters: the first turn fails at the backend, the second
        // names a tool outside the closed vocabulary, the third emits real
        // ops. Only the third composes a body, and the run continues.
        let clusters = json!([
            {"label": "a", "member_entry_ids": ["kb-10001"], "owning_map_id": null},
            {"label": "b", "member_entry_ids": ["kb-10002"], "owning_map_id": null},
            {"label": "c", "member_entry_ids": ["kb-10005"], "owning_map_id": null}
        ])
        .to_string();
        let backend = MockBackend::new(vec![
            Ok(text_turn(&clusters, usage(1, 1, None, None))),
            Err(BackendError::Terminal {
                kind: harness::model::TerminalKind::Other,
                message: "boom".to_string(),
            }),
            Ok(calls_turn(
                &[("finish", json!({}))],
                usage(1, 1, None, None),
            )),
            Ok(calls_turn(
                &[(
                    "add_pointer",
                    json!({"map_id": "kb-20001", "entry_id": "kb-10005", "gloss": "GLOSS-5"}),
                )],
                usage(1, 1, None, None),
            )),
        ]);
        let source = std::sync::Arc::new(ScriptedSource::new(vec![FetchOutcome::Ready(fixture())]));
        let ledger = InMemoryLedger::new();
        let body_root = tempfile::tempdir().expect("tempdir");
        let gate = gate_for_script("exit 0");
        let report = run_with(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
        )
        .await;
        assert_eq!(backend.calls(), 4);
        assert_eq!(report.clusters.len(), 3);
        assert!(matches!(
            &report.clusters[0].rung2,
            Rung2Outcome::ParseError { reason } if reason.contains("rung-2 backend error")
        ));
        assert!(matches!(
            &report.clusters[1].rung2,
            Rung2Outcome::ParseError { reason } if reason.contains("unknown tool `finish`")
        ));
        let raw_path = report.clusters[1]
            .raw_path
            .as_ref()
            .expect("the bad cluster's raw text is offloaded");
        assert_eq!(
            raw_path,
            &crate::rungs::rung2_raw_path(body_root.path(), "demo-project", 1)
        );
        assert_eq!(report.clusters[2].rung2, Rung2Outcome::Parsed);
        assert_eq!(report.clusters[2].ops.len(), 1);
        assert_eq!(report.composed_bodies.len(), 1);
        assert_eq!(report.composed_bodies[0].map_id, "kb-20001");
        assert_eq!(
            report.outcome,
            UnitOutcome::Ready,
            "one bad cluster does not abort the unit"
        );
        // The declined nothing: clusters a and b never parsed ops, and c
        // acted.
        assert!(report.declines_recorded.is_empty());
    }

    #[tokio::test]
    async fn create_map_for_an_owned_cluster_is_rejected_by_the_pipeline() {
        // The lead-disposition guard: an owned cluster paired with a
        // create_map is rejected BEFORE any op is applied — partial
        // coverage must converge, not re-propose.
        let clusters = json!([
            {"label": "home-network", "member_entry_ids": ["kb-10001", "kb-10002"], "owning_map_id": "kb-20001"}
        ])
        .to_string();
        let backend = MockBackend::from_turns(vec![
            text_turn(&clusters, usage(1, 1, None, None)),
            calls_turn(
                &[
                    (
                        "create_map",
                        json!({"cluster_id": "c1", "title": "t", "orientation_prose": "p"}),
                    ),
                    (
                        "add_pointer",
                        json!({"map_id": "somnus-new-c1", "entry_id": "kb-10001", "gloss": "g"}),
                    ),
                ],
                usage(1, 1, None, None),
            ),
        ]);
        let source = std::sync::Arc::new(ScriptedSource::new(vec![FetchOutcome::Ready(fixture())]));
        let ledger = InMemoryLedger::new();
        let body_root = tempfile::tempdir().expect("tempdir");
        let gate = gate_for_script("exit 0");
        let report = run_with(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
        )
        .await;
        assert!(report.composed_bodies.is_empty(), "no op was applied");
        assert!(matches!(
            &report.clusters[0].rung2,
            Rung2Outcome::ParseError { reason } if reason.contains("create_map is refused")
                && reason.contains("owning_map_id kb-20001")
        ));
        assert!(report.clusters[0].raw_path.as_ref().is_some());
        assert!(report.declines_recorded.is_empty());
        assert_eq!(backend.calls(), 2);
    }

    #[tokio::test]
    async fn a_second_create_map_hits_the_per_project_cap_and_is_recorded() {
        let clusters = json!([
            {"label": "a", "member_entry_ids": ["kb-10001"], "owning_map_id": null}
        ])
        .to_string();
        let backend = MockBackend::from_turns(vec![
            text_turn(&clusters, usage(1, 1, None, None)),
            calls_turn(
                &[
                    (
                        "create_map",
                        json!({"cluster_id": "c1", "title": "t1", "orientation_prose": "PROSE-1"}),
                    ),
                    (
                        "add_pointer",
                        json!({"map_id": "somnus-new-c1", "entry_id": "kb-10001", "gloss": "GLOSS-1"}),
                    ),
                    (
                        "create_map",
                        json!({"cluster_id": "c2", "title": "t2", "orientation_prose": "PROSE-2"}),
                    ),
                ],
                usage(1, 1, None, None),
            ),
        ]);
        let source = std::sync::Arc::new(ScriptedSource::new(vec![FetchOutcome::Ready(fixture())]));
        let ledger = InMemoryLedger::new();
        let body_root = tempfile::tempdir().expect("tempdir");
        let gate = gate_for_script("exit 0");
        let report = run_with(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
        )
        .await;
        assert_eq!(
            report.composed_bodies.len(),
            1,
            "the first create_map composes"
        );
        assert_eq!(report.composed_bodies[0].map_id, "somnus-new-c1");
        assert_eq!(
            report.cap_refusals,
            vec![crate::materialize::render_cap_refusal_line("demo-project")]
        );
        assert_eq!(report.gate_reports.len(), 1);
    }

    #[tokio::test]
    async fn a_hard_compose_refusal_is_recorded_without_a_body() {
        // A create_map with no pointers targeting the new map: the
        // EmptyPointerList refusal lands in the report and nothing is
        // composed or gated.
        let clusters = json!([
            {"label": "a", "member_entry_ids": ["kb-10001"], "owning_map_id": null}
        ])
        .to_string();
        let backend = MockBackend::from_turns(vec![
            text_turn(&clusters, usage(1, 1, None, None)),
            calls_turn(
                &[(
                    "create_map",
                    json!({"cluster_id": "c1", "title": "t", "orientation_prose": "p"}),
                )],
                usage(1, 1, None, None),
            ),
        ]);
        let source = std::sync::Arc::new(ScriptedSource::new(vec![FetchOutcome::Ready(fixture())]));
        let ledger = InMemoryLedger::new();
        let body_root = tempfile::tempdir().expect("tempdir");
        let gate = gate_for_script("exit 0");
        let report = run_with(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
        )
        .await;
        assert!(report.composed_bodies.is_empty());
        assert_eq!(report.compose_refusals.len(), 1);
        assert!(
            report.compose_refusals[0].contains("no pointers"),
            "{}",
            report.compose_refusals[0]
        );
        assert!(report.gate_reports.is_empty());
    }

    // ======================================================================
    // the wall-clock budget
    // ======================================================================

    #[tokio::test(start_paused = true)]
    async fn the_wall_clock_budget_expires_into_a_named_abort() {
        // The paused runtime auto-advances the clock while the pending
        // future is outstanding, so the 4-hour budget expires with no
        // wall-clock wait (the `observer.rs` precedent).
        let backend = PendBackend;
        let source = std::sync::Arc::new(ScriptedSource::new(vec![FetchOutcome::Ready(fixture())]));
        let ledger = InMemoryLedger::new();
        let body_root = tempfile::tempdir().expect("tempdir");
        let gate = gate_for_script("exit 0");
        let report = run_with(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
        )
        .await;
        assert_eq!(
            report.outcome,
            UnitOutcome::Aborted {
                reason: WALL_CLOCK_ABORT_MSG.to_string(),
            }
        );
        assert!(
            std::fs::read_to_string(crate::materialize::run_report_path(
                body_root.path(),
                "demo-project"
            ))
            .expect("the expired run still leaves a report")
            .contains("nightly wall-clock budget exhausted")
        );
    }

    // ======================================================================
    // unwritable roots: every write failure is loud and fail-closed
    // ======================================================================

    /// A body root whose project directory is a FILE, so every write under
    /// it fails.
    fn broken_root() -> tempfile::TempDir {
        let root = tempfile::tempdir().expect("tempdir");
        std::fs::write(root.path().join("demo-project"), "not a directory")
            .expect("a file where the project directory should be");
        root
    }

    #[tokio::test]
    async fn an_unwritable_body_root_aborts_the_unit() {
        let backend = MockBackend::from_turns(vec![
            text_turn(&e2e_clusters_json(), usage(1, 1, None, None)),
            calls_turn(
                &[
                    (
                        "create_map",
                        json!({"cluster_id": "c1", "title": "t", "orientation_prose": "p"}),
                    ),
                    (
                        "add_pointer",
                        json!({"map_id": "somnus-new-c1", "entry_id": "kb-10001", "gloss": "g"}),
                    ),
                ],
                usage(1, 1, None, None),
            ),
            calls_turn(
                &[("propose_gap", json!({"cluster_id": "c2", "reason": "r"}))],
                usage(1, 1, None, None),
            ),
        ]);
        let source = std::sync::Arc::new(ScriptedSource::new(vec![FetchOutcome::Ready(fixture())]));
        let ledger = InMemoryLedger::new();
        let root = broken_root();
        let gate = gate_for_script("exit 0");
        let report = run_with(
            &backend,
            source,
            &ledger,
            &gate,
            root.path(),
            "demo-project",
        )
        .await;
        let UnitOutcome::Aborted { reason } = &report.outcome else {
            panic!("expected an abort, got {:?}", report.outcome);
        };
        assert!(
            reason.contains("could not write the composed body"),
            "reason was {reason}"
        );
        // The decline for the propose_gap cluster was still recorded: the
        // abort happens at materialization, after the paid work.
        assert_eq!(report.declines_recorded.len(), 1);
    }

    #[tokio::test]
    async fn a_failed_rung1_offload_is_announced_and_the_unit_aborts() {
        let backend = MockBackend::from_turns(vec![text_turn("not json", usage(1, 1, None, None))]);
        let source = std::sync::Arc::new(ScriptedSource::new(vec![FetchOutcome::Ready(fixture())]));
        let ledger = InMemoryLedger::new();
        let root = broken_root();
        let gate = gate_for_script("exit 0");
        let report = run_with(
            &backend,
            source,
            &ledger,
            &gate,
            root.path(),
            "demo-project",
        )
        .await;
        let UnitOutcome::Aborted { reason } = &report.outcome else {
            panic!("expected an abort, got {:?}", report.outcome);
        };
        assert!(
            reason.starts_with("somnus: rung-1 parse failed"),
            "reason was {reason}"
        );
        assert!(report.rung1_raw_path.is_some());
        // The raw file does not exist (the root is broken), and the report
        // write failed loudly too — but the run still returned a report.
        assert!(!report.rung1_raw_path.as_ref().unwrap().exists());
    }

    #[tokio::test]
    async fn a_failed_rung2_offload_is_announced_and_the_run_continues() {
        let clusters = json!([
            {"label": "a", "member_entry_ids": ["kb-10001"], "owning_map_id": null},
            {"label": "b", "member_entry_ids": ["kb-10002"], "owning_map_id": null}
        ])
        .to_string();
        let backend = MockBackend::from_turns(vec![
            text_turn(&clusters, usage(1, 1, None, None)),
            calls_turn(&[("finish", json!({}))], usage(1, 1, None, None)),
            calls_turn(
                &[("propose_gap", json!({"cluster_id": "c", "reason": "r"}))],
                usage(1, 1, None, None),
            ),
        ]);
        let source = std::sync::Arc::new(ScriptedSource::new(vec![FetchOutcome::Ready(fixture())]));
        let ledger = InMemoryLedger::new();
        let root = broken_root();
        let gate = gate_for_script("exit 0");
        let report = run_with(
            &backend,
            source,
            &ledger,
            &gate,
            root.path(),
            "demo-project",
        )
        .await;
        // The bad cluster is offloaded (loudly, to an unwritable path) and
        // skipped; the good cluster still declines and the run continues.
        assert!(matches!(
            report.clusters[0].rung2,
            Rung2Outcome::ParseError { .. }
        ));
        assert!(!report.clusters[0].raw_path.as_ref().unwrap().exists());
        assert_eq!(report.clusters[1].rung2, Rung2Outcome::Parsed);
        assert_eq!(report.declines_recorded.len(), 1);
        assert_eq!(report.outcome, UnitOutcome::Ready);
    }

    // ======================================================================
    // the pure report helpers
    // ======================================================================

    #[test]
    fn the_call_count_audit_line_is_some_exactly_when_the_discipline_breaks() {
        assert_eq!(render_call_count_audit_line(1, 0), None);
        assert_eq!(render_call_count_audit_line(3, 2), None);
        assert_eq!(
            render_call_count_audit_line(4, 2).as_deref(),
            Some(
                "somnus: call-count audit: 4 backend calls for 2 clusters (expected at most 3); the single-shot discipline was violated"
            )
        );
        assert_eq!(
            render_call_count_audit_line(25, 23).as_deref(),
            Some(
                "somnus: call-count audit: 25 backend calls for 23 clusters (expected at most 24); the single-shot discipline was violated"
            )
        );
    }

    #[test]
    fn the_decline_recorded_line_is_byte_pinned() {
        assert_eq!(
            render_decline_recorded_line("demo-project", &["a".to_string(), "b".to_string()]),
            "somnus: decline recorded for demo-project: [a, b]"
        );
        assert_eq!(
            render_decline_recorded_line("solo", &[]),
            "somnus: decline recorded for solo: []"
        );
    }

    #[test]
    fn the_decline_write_failed_line_is_byte_pinned() {
        assert_eq!(
            render_decline_write_failed_line(
                "demo-project",
                &["kb-10001".to_string(), "kb-10002".to_string()]
            ),
            "somnus: decline write failed for demo-project: [kb-10001, kb-10002]"
        );
    }

    #[test]
    fn the_budget_and_wall_clock_abort_messages_are_pinned() {
        assert_eq!(
            BUDGET_ABORT_MSG,
            "somnus: per-unit inference budget exhausted (24 backend calls); aborting rung 2 for remaining clusters"
        );
        assert_eq!(
            WALL_CLOCK_ABORT_MSG,
            "somnus: nightly wall-clock budget exhausted (14400s)"
        );
        assert_eq!(crate::NIGHTLY_WALL_CLOCK_SECS, 14_400);
        assert_eq!(crate::SOMNUS_MAX_ITERATIONS, 24);
    }
}
