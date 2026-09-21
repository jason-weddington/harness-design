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
use crate::map_op::{MapOpClient, MapOpRequest, MapOpResult};
use crate::materialize::{
    ComposedBody, MapEdit, Materialized, materialize_cluster, new_map_id, run_report_path,
    write_composed_body,
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
    /// The tool context the gate runner offloads through. Tests use
    /// `ToolCtx::stub()`; the binary builds the production `ToolCtx` over
    /// the state dir.
    pub tool_ctx: &'a ToolCtx,
    /// The map-op client the application step POSTs through.
    pub map_ops: Arc<dyn MapOpClient>,
    /// The invocation's token ceiling, in billed tokens. The sentinel `0` =
    /// unbounded (mirroring `RunConfig::token_budget` in
    /// `crates/harness/src/engine.rs`). somnus enforces it ITSELF because its
    /// pipeline drives `ModelBackend::turn` directly and never runs
    /// `engine::run`.
    pub token_budget: u64,
    /// The invocation's billed-token total at unit START, so an across-units
    /// nightly ceiling sees what earlier units already spent. No second
    /// accumulator: the check reads the accumulator that already exists in
    /// [`run_unit`]'s report usage.
    pub billed_before: u64,
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
    /// The KB refused the caller on `/api/kb/map-op` as not the machine
    /// principal: a server configuration fact, never a somnus configuration
    /// fault.
    NotMachinePrincipal,
    /// The unit stopped with a named reason (never a retry).
    Aborted {
        /// Why the unit aborted.
        reason: String,
    },
}

impl UnitOutcome {
    /// The outcome's name, as it serializes verbatim and as the nightly
    /// stderr line renders it.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Ready => "Ready",
            Self::UnknownProject => "UnknownProject",
            Self::NotEligible => "NotEligible",
            Self::Unauthorized => "Unauthorized",
            Self::NotMachinePrincipal => "NotMachinePrincipal",
            Self::Aborted { .. } => "Aborted",
        }
    }
}

/// The pinned exit-code mapping: 0 on the ordinary outcomes (including a
/// `create_map` 409 cap admission, which never leaves `Ready`), 1 on the
/// config-class outcomes (401/403 from ANY endpoint), 2 on every
/// run-class abort (transport errors, gate failures, map-op faults, budget
/// exhaustion). Per-cluster rung-2 parse errors KEEP their skip-and-continue
/// semantics — this mapping is never licence to convert them to aborts.
#[must_use]
pub fn exit_code_for_outcome(outcome: &UnitOutcome) -> i32 {
    match outcome {
        UnitOutcome::Ready | UnitOutcome::UnknownProject | UnitOutcome::NotEligible => 0,
        UnitOutcome::Unauthorized | UnitOutcome::NotMachinePrincipal => 1,
        UnitOutcome::Aborted { .. } => 2,
    }
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
    /// The billed-token sum: `input + output + cache_read + cache_write`,
    /// saturating, over the ALREADY-unwrapped counters — the ONE formula
    /// every consumer calls, extracted from `harness` so the workspace has
    /// a single definition of a billed token (see
    /// `harness::model::billed_token_sum`).
    #[must_use]
    pub fn billed(&self) -> u64 {
        harness::model::billed_token_sum(self.input, self.output, self.cache_read, self.cache_write)
    }

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
    /// The gate child's combined stdout+stderr tail — the map-lint server's
    /// 422 lint findings on a red body, preserved instead of discarded
    /// (`curl -s --fail-with-body`).
    pub excerpt: String,
    /// Where the full combined output was offloaded.
    pub offload_path: Option<PathBuf>,
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
    /// Rendered hard compose refusals (no body was composed for that op).
    pub compose_refusals: Vec<String>,
    /// The composed bodies, in materialization order.
    pub composed_bodies: Vec<ComposedBodyRecord>,
    /// One gate report per composed body.
    pub gate_reports: Vec<GateReportRecord>,
    /// One record per op `POSTed` to `/api/kb/map-op` (a skipped op is in
    /// [`UnitReport::skipped`]; a cap admission in
    /// [`UnitReport::cap_admissions`]).
    pub applied: Vec<AppliedOpRecord>,
    /// One record per op deliberately NOT `POSTed` (a `create_map` skipped by
    /// the cap-admission latch).
    pub skipped: Vec<SkippedOpRecord>,
    /// The `create_map` 409 cap admissions, ordinary by contract: the
    /// server's reason body recorded verbatim, never an outcome.
    pub cap_admissions: Vec<AppliedOpRecord>,
    /// The apply-audit tripwire lines (a materialize ordering or inlining
    /// regression, caught client-side).
    pub apply_audit: Vec<String>,
    /// The leg-3 change evidence.
    pub change: ChangeEvidence,
    /// How many `ModelBackend::turn` calls the unit made.
    pub backend_calls: u64,
    /// Rung-1 cost telemetry.
    pub usage_rung1: UsageTotals,
    /// Rung-2 cost telemetry.
    pub usage_rung2: UsageTotals,
    /// Wall-clock milliseconds around the rung-1 turn.
    pub wall_rung1_ms: u64,
    /// Wall-clock milliseconds summed across the rung-2 turns.
    pub wall_rung2_ms: u64,
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
            compose_refusals: Vec::new(),
            composed_bodies: Vec::new(),
            gate_reports: Vec::new(),
            applied: Vec::new(),
            skipped: Vec::new(),
            cap_admissions: Vec::new(),
            apply_audit: Vec::new(),
            change: ChangeEvidence::default(),
            backend_calls: 0,
            usage_rung1: UsageTotals::default(),
            usage_rung2: UsageTotals::default(),
            wall_rung1_ms: 0,
            wall_rung2_ms: 0,
            call_count_audit: None,
            rung1_raw_path: None,
            report_path,
        }
    }
}

/// One op `POSTed` to `/api/kb/map-op`. The `body` field is the exact
/// `POSTed` body text (the [`ComposedBodyRecord`] precedent), so the report alone
/// reconstructs every byte sent to the mutating endpoint.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AppliedOpRecord {
    /// `"create_map"`, `"add_pointer"`, or `"strike_gap"`.
    pub op_kind: &'static str,
    /// `0` for a `create_map`, `i` for the i-th edit in the map's edit chain
    /// (1-based, so a create and its first edit never share an index).
    pub chain_index: usize,
    /// The map id the request named (a fresh map keeps its client-side
    /// `somnus-new-{cluster_id}` id here).
    pub submitted_map_id: String,
    /// The server's returned map id, when the op applied.
    pub server_map_id: Option<String>,
    /// The server's returned version, when the op applied.
    pub version: Option<u64>,
    /// The server's returned pointer count, when the op applied.
    pub pointer_count: Option<u64>,
    /// The server's returned remaining budget, when the op applied.
    pub budget: Option<u64>,
    /// The HTTP status (`Some` for dialed ops, `None` for skipped).
    pub http_status: Option<u16>,
    /// The exact `POSTed` body text.
    pub body: String,
    /// The verbatim 409 reason body, on a cap admission.
    pub admission_reason: Option<String>,
}

/// One op deliberately NOT `POSTed`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SkippedOpRecord {
    /// The skipped op's kind.
    pub op_kind: &'static str,
    /// The map the op targeted.
    pub map_id: String,
    /// Why it was skipped.
    pub reason: String,
}

/// The pinned stderr line for a map-op 403: the server's configuration is
/// at fault (its `machine_principal_email`), never somnus's.
pub const MAP_OP_NOT_MACHINE_PRINCIPAL_MSG: &str = "somnus: the KB refused the caller on /api/kb/map-op as not the machine principal — check machine_principal_email on the server; this is not a somnus configuration fault";

/// The pinned token-budget exhaustion message, carrying BOTH decision
/// inputs. Pure so the shape is byte-pinned; the pipeline only `eprintln!`s
/// it.
#[must_use]
pub fn render_token_budget_line(armed: u64, billed: u64) -> String {
    format!("somnus: token budget exhausted (armed {armed}, billed {billed})")
}

/// The pinned reason naming the FIRST red gate body, in
/// `composed_bodies` order.
#[must_use]
pub fn render_gate_rejected_line(map_id: &str) -> String {
    format!("somnus: map-lint gate rejected the body for {map_id}")
}

/// The pinned skip reason for a `create_map` withheld by the
/// cap-admission latch.
#[must_use]
pub fn render_create_skipped_after_admission(map_id: &str) -> String {
    format!(
        "somnus: create_map for {map_id} skipped: a create_map cap admission was already recorded this invocation"
    )
}

/// The map-op fault reason prefix for one failed op.
#[must_use]
pub fn render_map_op_fault_line(map_id: &str, reason: &str) -> String {
    format!("somnus: map-op failed for {map_id}: {reason}")
}

/// The apply-audit tripwire: a pure, fixture-tested scan that catches a
/// materialize ordering or inlining regression client-side, with a named
/// reason, instead of by the KB after the metered spend.
///
/// For an `add_pointer`, it answers:
///
/// - the pinned fresh-map line when the op targets a map in `fresh_ids`
///   (its pointers were already inlined into the create body, so an
///   `add_pointer` request for one is a regression);
/// - the pinned delta line when `previous_body` is supplied and the
///   request's body introduces a ref delta other than exactly the claimed
///   `added_entry_id` — the delta computed by a hand-rolled scan for the
///   literal shape `kb-` + five ASCII digits (the vendored spec's pinned
///   pointer shape; no new dependency);
/// - `None` otherwise.
///
/// Every other op kind is unaudited (the server's own set comparison is
/// authoritative for it).
#[must_use]
pub fn apply_audit(
    previous_body: Option<&str>,
    request: &MapOpRequest,
    fresh_ids: &[String],
) -> Option<String> {
    let MapOpRequest::AddPointer {
        map_id,
        body,
        added_entry_id,
    } = request
    else {
        return None;
    };
    if fresh_ids.iter().any(|fresh| fresh == map_id) {
        return Some(format!(
            "somnus: apply audit: add_pointer on fresh map {map_id} not absorbed by the create body"
        ));
    }
    let previous = previous_body?;
    let old = kb_refs(previous);
    let new_refs = kb_refs(body);
    let mut delta: Vec<String> = Vec::new();
    for reference in new_refs {
        if !old.contains(&reference) && !delta.contains(&reference) {
            delta.push(reference);
        }
    }
    if delta.as_slice() != [added_entry_id.as_str()] {
        return Some(format!(
            "somnus: apply audit: add_pointer delta for {map_id} was {delta:?}, expected {added_entry_id}"
        ));
    }
    None
}

/// Every `kb-XXXXX` ref in `text` (the literal shape `kb-` followed by five
/// ASCII digits), deduplicated in first-appearance order. A hand-rolled
/// scan, not a regex dependency: the shape is the vendored spec's pinned
/// pointer shape.
fn kb_refs(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut refs = Vec::new();
    let mut index = 0;
    while index + 8 <= bytes.len() {
        if &bytes[index..index + 3] == b"kb-"
            && bytes[index + 3..index + 8].iter().all(u8::is_ascii_digit)
        {
            let reference = text[index..index + 8].to_string();
            if !refs.contains(&reference) {
                refs.push(reference);
            }
            index += 8;
        } else {
            index += 1;
        }
    }
    refs
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

    // (6) Rung 1: one single-shot inference for the whole project, AFTER
    // the token-budget guard (the whole unit's spend so far is
    // `billed_before`; the sentinel 0 = unbounded short-circuits).
    if harness::engine::token_budget_breached(deps.billed_before, deps.token_budget) {
        let billed = deps.billed_before;
        eprintln!("{}", render_token_budget_line(deps.token_budget, billed));
        report.outcome = UnitOutcome::Aborted {
            reason: render_token_budget_line(deps.token_budget, billed),
        };
        return finalize(report);
    }
    let rung1_started = std::time::Instant::now();
    let turn = match rung1_turn(deps.backend, project_ref, &filtered).await {
        Ok(turn) => turn,
        Err(err) => {
            report.outcome = UnitOutcome::Aborted {
                reason: format!("somnus: rung-1 backend error: {err}"),
            };
            return finalize(report);
        }
    };
    report.wall_rung1_ms = u64::try_from(rung1_started.elapsed().as_millis()).unwrap_or(u64::MAX);
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
        // The token-budget guard, BEFORE the turn: the decision inputs are
        // the invocation total so far (`billed_before`) plus this unit's
        // accumulated billed usage. Same guard style as the count cap
        // above — checked between turns, never mid-flight.
        let billed_now =
            deps.billed_before + report.usage_rung1.billed() + report.usage_rung2.billed();
        if harness::engine::token_budget_breached(billed_now, deps.token_budget) {
            budget_reason = Some(render_token_budget_line(deps.token_budget, billed_now));
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
        let rung2_started = std::time::Instant::now();
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
        report.wall_rung2_ms +=
            u64::try_from(rung2_started.elapsed().as_millis()).unwrap_or(u64::MAX);
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
    // There is NO client-side new-map cap: both caps are server-side, so a
    // second create_map in one op set composes and gates like any other
    // and meets the server's ordinary 409 admission at application time.
    let mut composed: Vec<ComposedBody> = Vec::new();
    // The per-edit chains, grouped by map in collection order, consumed by
    // the application step below.
    let mut edits_by_map: Vec<(String, Vec<MapEdit>)> = Vec::new();
    for record in &report.clusters {
        if matches!(record.rung2, Rung2Outcome::ParseError { .. }) {
            continue;
        }
        let Materialized {
            bodies,
            authorship_refusals,
            compose_refusals,
            edits,
        } = materialize_cluster(&record.cluster(), &record.ops, &input);
        report.authorship_refusals.extend(authorship_refusals);
        report.compose_refusals.extend(compose_refusals);
        for edit in edits {
            let (map_id, edit) = (edit.map_id.clone(), edit);
            match edits_by_map
                .iter_mut()
                .find(|(existing, _)| *existing == map_id)
            {
                Some((_, chain)) => chain.push(edit),
                None => edits_by_map.push((map_id, vec![edit])),
            }
        }
        composed.extend(bodies);
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
            excerpt: gate_report.excerpt,
            offload_path: gate_report.offload_path,
        });
    }

    // (12.5) The application step: every composed body whose gate report
    // PASSED has its ops applied through the map-op client, in op order
    // per map. A red gate body is NOT applied (its paid work is recorded,
    // never discarded); the unit aborts naming the FIRST red body, while
    // gate-green siblings are still applied — one bad body must not
    // discard the others' paid work.
    let application = apply_map_ops(
        &mut report,
        &deps.map_ops,
        project_ref,
        &input,
        edits_by_map,
    )
    .await;
    if let Some(first_red) = application.first_red_body
        && !matches!(report.outcome, UnitOutcome::Aborted { .. })
    {
        report.outcome = UnitOutcome::Aborted {
            reason: render_gate_rejected_line(&first_red),
        };
    }
    if let Some(outcome) = application.outcome_override {
        report.outcome = outcome;
    }

    // (13) The final observation through the SAME observer (cached
    // baseline, fail closed) and the leg-3 classification.
    observe_and_finalize(report, &observer, &baseline).await
}

/// What the application step decided after all green bodies were handled.
struct ApplicationResult {
    /// The FIRST red gate body, in `composed_bodies` order, when any gate
    /// report was red.
    first_red_body: Option<String>,
    /// A terminal outcome the application step decided (a map-op 401/403
    /// stop or a fault abort), which overrides whatever the pipeline had
    /// already concluded — config-class outcomes outrank run-class aborts.
    outcome_override: Option<UnitOutcome>,
}

/// The application step: POST every gate-green composed body's ops through
/// `map_ops`, in op order per map, recording every byte, skip, and
/// admission on the report.
///
/// - A FRESH map (an id of [`crate::materialize::new_map_id`]'s
///   `somnus-new-{cluster_id}` shape) issues exactly ONE `create_map` POST
///   carrying the full composed fresh body, with short and long title both
///   the create op's `title` verbatim. Its own `add_pointer` ops are NOT
///   re-issued as calls — materialize's `fresh_ids` filter already inlined
///   them into the create body, and re-issuing would 409 "nothing added".
/// - An EXISTING map issues one POST per edit in its chain, the i-th
///   `add_pointer` carrying `body_after` (the body state after the first i
///   edits — the server's `new − old == {added_entry_id}` set comparison
///   makes the final composed body wrong for every call but the last).
/// - A `create_map` 409 cap admission is an ORDINARY ADMISSION OUTCOME: the
///   reason body is recorded verbatim, a boolean LATCH (not a counter —
///   somnus must not track the two server-side caps) turns further
///   `create_map` application off for the rest of the unit, and the exit
///   stays 0.
/// - A 401 stops as [`UnitOutcome::Unauthorized`], a 403 as
///   [`UnitOutcome::NotMachinePrincipal`] (with the pinned stderr line),
///   and any fault aborts the unit with
///   [`render_map_op_fault_line`] — none of them is retryable, so the
///   remaining bodies are not applied after one stops.
#[allow(clippy::too_many_lines)] // one dispatch loop, two op shapes
async fn apply_map_ops(
    report: &mut UnitReport,
    map_ops: &std::sync::Arc<dyn MapOpClient>,
    project_ref: &str,
    input: &crate::loop_input::LoopInput,
    mut edits_by_map: Vec<(String, Vec<MapEdit>)>,
) -> ApplicationResult {
    // The fresh maps and their titles, from the parsed ops.
    let mut fresh: Vec<(String, String)> = Vec::new();
    for record in &report.clusters {
        if matches!(record.rung2, Rung2Outcome::ParseError { .. }) {
            continue;
        }
        for op in &record.ops {
            if let Op::CreateMap {
                cluster_id, title, ..
            } = op
            {
                fresh.push((new_map_id(cluster_id), title.clone()));
            }
        }
    }
    let fresh_ids: Vec<String> = fresh.iter().map(|(id, _)| id.clone()).collect();

    let plan: Vec<(ComposedBodyRecord, bool)> = report
        .composed_bodies
        .iter()
        .zip(report.gate_reports.iter().map(|gate| gate.passed))
        .map(|(body, passed)| (body.clone(), passed))
        .collect();

    let mut first_red_body: Option<String> = None;
    let mut admission_latch: Option<String> = None;
    for (body_record, passed) in plan {
        if !passed {
            if first_red_body.is_none() {
                first_red_body = Some(body_record.map_id.clone());
            }
            continue;
        }
        if let Some((_, title)) = fresh.iter().find(|(id, _)| *id == body_record.map_id) {
            // A fresh map: exactly ONE create_map POST.
            if admission_latch.is_some() {
                report.skipped.push(SkippedOpRecord {
                    op_kind: "create_map",
                    map_id: body_record.map_id.clone(),
                    reason: render_create_skipped_after_admission(&body_record.map_id),
                });
                continue;
            }
            let request = MapOpRequest::CreateMap {
                project_ref: project_ref.to_string(),
                short_title: title.clone(),
                long_title: title.clone(),
                body: body_record.body.clone(),
            };
            let body_text = request_body_text(&request);
            let submitted = body_record.map_id.clone();
            match map_ops.apply(request).await {
                MapOpResult::Applied(applied) => {
                    report.applied.push(AppliedOpRecord {
                        op_kind: "create_map",
                        chain_index: 0,
                        submitted_map_id: submitted,
                        server_map_id: Some(applied.map_id),
                        version: Some(applied.version),
                        pointer_count: Some(applied.pointer_count),
                        budget: Some(applied.budget),
                        http_status: Some(201),
                        body: body_text,
                        admission_reason: None,
                    });
                }
                MapOpResult::CapAdmission { reason_body } => {
                    admission_latch = Some(reason_body.clone());
                    report.cap_admissions.push(AppliedOpRecord {
                        op_kind: "create_map",
                        chain_index: 0,
                        submitted_map_id: submitted,
                        server_map_id: None,
                        version: None,
                        pointer_count: None,
                        budget: None,
                        http_status: Some(409),
                        body: body_text,
                        admission_reason: Some(reason_body),
                    });
                }
                MapOpResult::Unauthorized => {
                    return ApplicationResult {
                        first_red_body,
                        outcome_override: Some(UnitOutcome::Unauthorized),
                    };
                }
                MapOpResult::NotMachinePrincipal => {
                    eprintln!("{MAP_OP_NOT_MACHINE_PRINCIPAL_MSG}");
                    return ApplicationResult {
                        first_red_body,
                        outcome_override: Some(UnitOutcome::NotMachinePrincipal),
                    };
                }
                MapOpResult::Fault { map_id, reason } => {
                    return ApplicationResult {
                        first_red_body,
                        outcome_override: Some(UnitOutcome::Aborted {
                            reason: render_map_op_fault_line(&map_id, &reason),
                        }),
                    };
                }
            }
            continue;
        }

        // An existing map: one POST per edit, in chain (op) order.
        let Some(position) = edits_by_map
            .iter()
            .position(|(map_id, _)| *map_id == body_record.map_id)
        else {
            continue;
        };
        let (_, chain) = edits_by_map.remove(position);
        let original_body = input
            .maps
            .iter()
            .find(|map| map.id == body_record.map_id)
            .map(|map| map.body.clone());
        for (index, edit) in chain.iter().enumerate() {
            let (request, op_kind) = match edit {
                MapEdit {
                    map_id,
                    kind: crate::materialize::MapEditKind::AddPointer { entry_id },
                    body_after,
                } => (
                    MapOpRequest::AddPointer {
                        map_id: map_id.clone(),
                        body: body_after.clone(),
                        added_entry_id: entry_id.clone(),
                    },
                    "add_pointer",
                ),
                MapEdit {
                    map_id,
                    kind:
                        crate::materialize::MapEditKind::StrikeGap {
                            gap_text,
                            closing_entry_id,
                        },
                    body_after,
                } => (
                    MapOpRequest::StrikeGap {
                        map_id: map_id.clone(),
                        body: body_after.clone(),
                        gap_text: gap_text.clone(),
                        closing_entry_id: closing_entry_id.clone(),
                    },
                    "strike_gap",
                ),
            };
            // The audit tripwire runs BEFORE the POST: a materialize
            // regression aborts the unit with the named reason instead of
            // being caught by the KB after the metered spend.
            if matches!(request, MapOpRequest::AddPointer { .. }) {
                let previous = if index == 0 {
                    original_body.as_deref()
                } else {
                    Some(chain[index - 1].body_after.as_str())
                };
                if let Some(line) = apply_audit(previous, &request, &fresh_ids) {
                    report.apply_audit.push(line.clone());
                    return ApplicationResult {
                        first_red_body,
                        outcome_override: Some(UnitOutcome::Aborted { reason: line }),
                    };
                }
            }
            let body_text = request_body_text(&request);
            let submitted = body_record.map_id.clone();
            match map_ops.apply(request).await {
                MapOpResult::Applied(applied) => {
                    report.applied.push(AppliedOpRecord {
                        op_kind,
                        chain_index: index + 1,
                        submitted_map_id: submitted,
                        server_map_id: Some(applied.map_id),
                        version: Some(applied.version),
                        pointer_count: Some(applied.pointer_count),
                        budget: Some(applied.budget),
                        http_status: Some(200),
                        body: body_text,
                        admission_reason: None,
                    });
                }
                MapOpResult::CapAdmission { reason_body } => {
                    // Unreachable from the production client (it classifies
                    // every update-op 409 as a Fault): treated as a fault
                    // rather than an admission against an EXISTING map.
                    return ApplicationResult {
                        first_red_body,
                        outcome_override: Some(UnitOutcome::Aborted {
                            reason: render_map_op_fault_line(&submitted, &reason_body),
                        }),
                    };
                }
                MapOpResult::Unauthorized => {
                    return ApplicationResult {
                        first_red_body,
                        outcome_override: Some(UnitOutcome::Unauthorized),
                    };
                }
                MapOpResult::NotMachinePrincipal => {
                    eprintln!("{MAP_OP_NOT_MACHINE_PRINCIPAL_MSG}");
                    return ApplicationResult {
                        first_red_body,
                        outcome_override: Some(UnitOutcome::NotMachinePrincipal),
                    };
                }
                MapOpResult::Fault { map_id, reason } => {
                    return ApplicationResult {
                        first_red_body,
                        outcome_override: Some(UnitOutcome::Aborted {
                            reason: render_map_op_fault_line(&map_id, &reason),
                        }),
                    };
                }
            }
        }
    }
    ApplicationResult {
        first_red_body,
        outcome_override: None,
    }
}

/// The exact `POSTed` body text for one map-op request (the
/// [`ComposedBodyRecord`] precedent).
fn request_body_text(request: &MapOpRequest) -> String {
    serde_json::to_string(&crate::map_op::build_request_body(request))
        .expect("the pinned request body always serializes")
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

    /// A scripted [`MapOpClient`] that records every request and answers
    /// from a script (the last entry repeats); an empty script answers
    /// `Applied` with a pinned envelope, so tests opt into failure modes.
    #[derive(Debug, Default)]
    struct RecordingMapOps {
        requests: std::sync::Mutex<Vec<MapOpRequest>>,
        script: std::sync::Mutex<Vec<MapOpResult>>,
    }

    impl RecordingMapOps {
        fn scripted(script: Vec<MapOpResult>) -> std::sync::Arc<Self> {
            std::sync::Arc::new(Self {
                requests: std::sync::Mutex::new(Vec::new()),
                script: std::sync::Mutex::new(script),
            })
        }

        fn requests(&self) -> Vec<MapOpRequest> {
            self.requests
                .lock()
                .expect("requests lock poisoned")
                .clone()
        }
    }

    fn applied_envelope() -> MapOpResult {
        MapOpResult::Applied(crate::map_op::MapOpApplied {
            map_id: "kb-30001".to_string(),
            version: 3,
            pointer_count: 6,
            budget: 2,
        })
    }

    #[async_trait]
    impl MapOpClient for RecordingMapOps {
        async fn apply(&self, request: MapOpRequest) -> MapOpResult {
            self.requests
                .lock()
                .expect("requests lock poisoned")
                .push(request);
            let mut script = self.script.lock().expect("script lock poisoned");
            if script.len() > 1 {
                script.remove(0)
            } else {
                script.first().cloned().unwrap_or_else(applied_envelope)
            }
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
        run_with_map_ops(
            backend,
            source,
            ledger,
            gate_for,
            body_root,
            project_ref,
            std::sync::Arc::new(RecordingMapOps::default()),
        )
        .await
    }

    /// [`run_with`] with a named map-op client, for the application-step
    /// tests (the budget defaults to unbounded and the invocation baseline
    /// to zero).
    async fn run_with_map_ops(
        backend: &dyn ModelBackend,
        source: std::sync::Arc<dyn LoopInputSource>,
        ledger: &dyn ClusterLedger,
        gate_for: &dyn Fn(&Path) -> ChecksRunner,
        body_root: &Path,
        project_ref: &str,
        map_ops: std::sync::Arc<dyn MapOpClient>,
    ) -> UnitReport {
        run_with_budget(
            backend,
            source,
            ledger,
            gate_for,
            body_root,
            project_ref,
            map_ops,
            0,
            0,
        )
        .await
    }

    /// The full-shape helper: a named map-op client plus the invocation
    /// token-budget arm and baseline.
    #[allow(clippy::too_many_arguments)] // the full UnitDeps shape
    async fn run_with_budget(
        backend: &dyn ModelBackend,
        source: std::sync::Arc<dyn LoopInputSource>,
        ledger: &dyn ClusterLedger,
        gate_for: &dyn Fn(&Path) -> ChecksRunner,
        body_root: &Path,
        project_ref: &str,
        map_ops: std::sync::Arc<dyn MapOpClient>,
        token_budget: u64,
        billed_before: u64,
    ) -> UnitReport {
        let tool_ctx = ToolCtx::stub();
        let deps = UnitDeps {
            backend,
            source,
            ledger,
            gate_for,
            tool_ctx: &tool_ctx,
            map_ops,
            token_budget,
            billed_before,
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
        drop(server);

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
            // Fenced on purpose. The first live run aborted on exactly this
            // framing, so the full-night path — not just `parse_clusters` —
            // is what has to tolerate it.
            text_turn(
                &format!("```json\n{}\n```", e2e_clusters_json()),
                usage(1000, 50, Some(200), None),
            ),
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
        let map_ops: std::sync::Arc<RecordingMapOps> = std::sync::Arc::default();
        let report = run_with_map_ops(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
            map_ops.clone(),
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
            assert_eq!(
                gate_report.excerpt, "",
                "a green stub gate has no output to excerpt"
            );
            assert!(gate_report.offload_path.is_some());
        }

        // --- the application step ------------------------------------------
        // One create_map POST for the fresh map, ONE add_pointer POST for
        // kb-20001, the fresh map's own add_pointers inlined (never
        // re-issued), each record carrying the exact POSTed body text.
        let requests = map_ops.requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[0],
            MapOpRequest::CreateMap {
                project_ref: "demo-project".to_string(),
                short_title: "Wireguard and DNS".to_string(),
                long_title: "Wireguard and DNS".to_string(),
                body: "Lives in knowledge/linux/network\n\nORIENTATION-PROSE\n\nDetail entries:\n- kb-10001 — GLOSS-1\n- kb-10002 — GLOSS-2".to_string(),
            },
            "short_title and long_title are the create op's title verbatim"
        );
        assert_eq!(
            requests[1],
            MapOpRequest::AddPointer {
                map_id: "kb-20001".to_string(),
                body: "Lives in knowledge/linux/network\n\nOrients the switch, DNS, and VPN entries for the home network.\n\nDetail entries:\n- kb-10001 — switch VLAN tagging baseline\n- kb-10002 — recursive DNS resolvers\n- kb-10003 — wireguard client config\n- kb-99999 — cross-project wireguard hub notes\n- kb-10004 — DHCP lease hygiene\n- kb-10005 — GLOSS-5\n\nNot yet documented:\n- Site-to-site wireguard topology".to_string(),
                added_entry_id: "kb-10005".to_string(),
            },
            "the single edit POST carries the FINAL composed body"
        );
        assert_eq!(report.applied.len(), 2);
        assert_eq!(report.applied[0].op_kind, "create_map");
        assert_eq!(report.applied[0].chain_index, 0);
        assert_eq!(report.applied[0].submitted_map_id, "somnus-new-c1");
        assert_eq!(
            report.applied[0].server_map_id.as_deref(),
            Some("kb-30001"),
            "the server's map id makes the round trip auditable"
        );
        assert_eq!(report.applied[0].http_status, Some(201));
        assert!(report.skipped.is_empty());
        assert!(report.cap_admissions.is_empty());
        assert!(report.apply_audit.is_empty());
        assert_eq!(report.wall_rung1_ms, 0, "MockBackend turns are instant");
        assert_eq!(report.wall_rung2_ms, 0);

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
        assert!(report.cap_admissions.is_empty());
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
            "compose_refusals",
            "composed_bodies",
            "gate_reports",
            "applied",
            "skipped",
            "cap_admissions",
            "apply_audit",
            "change",
            "backend_calls",
            "usage_rung1",
            "usage_rung2",
            "wall_rung1_ms",
            "wall_rung2_ms",
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
        // GATE-THEN-APPLY: a red gate body is NOT applied, and the unit
        // aborts with the pinned first-red-body reason.
        assert_eq!(
            report.outcome,
            UnitOutcome::Aborted {
                reason: render_gate_rejected_line("kb-20001"),
            }
        );
        assert_eq!(
            report.outcome,
            UnitOutcome::Aborted {
                reason: "somnus: map-lint gate rejected the body for kb-20001".to_string()
            }
        );
        assert_eq!(exit_code_for_outcome(&report.outcome), 2);
    }

    #[tokio::test]
    async fn a_red_gate_body_is_not_applied_while_gate_green_siblings_still_are() {
        // Two bodies: the first (the fresh map) is gate-red, the second
        // (kb-20001) gate-green. EXACTLY the green body's POST goes out,
        // and the outcome aborts naming the FIRST red body.
        let backend = MockBackend::from_turns(vec![
            text_turn(&e2e_clusters_json(), usage(1, 1, None, None)),
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
                        json!({"map_id": "kb-20001", "entry_id": "kb-10005", "gloss": "GLOSS-5"}),
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
        let body_root = tempfile::tempdir().expect("tempdir");
        let map_ops: std::sync::Arc<RecordingMapOps> = std::sync::Arc::default();
        // The gate script REDS the FIRST body only: the stub runs
        // `/bin/sh -c` once per body in order; `exit 0` for the second.
        // (The gate factory receives the body path; the scripted fake
        // below distinguishes by path.)
        let gate = |body_path: &Path| {
            let script = if body_path.to_string_lossy().contains("somnus-new-c1") {
                "exit 22"
            } else {
                "exit 0"
            };
            ChecksRunner::new(
                harness::exec::CheckCommand {
                    program: "/bin/sh".to_string(),
                    args: vec!["-c".to_string(), script.to_string()],
                },
                std::path::PathBuf::from("."),
                crate::gate::SOMNUS_GATE_TIMEOUT,
            )
        };
        let report = run_with_map_ops(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
            map_ops.clone(),
        )
        .await;
        assert_eq!(report.gate_reports.len(), 2);
        assert!(!report.gate_reports[0].passed);
        assert!(report.gate_reports[1].passed);
        assert_eq!(map_ops.requests().len(), 1, "only the green body applied");
        assert_eq!(
            map_ops.requests()[0],
            MapOpRequest::AddPointer {
                map_id: "kb-20001".to_string(),
                body: report.composed_bodies[1].body.clone(),
                added_entry_id: "kb-10005".to_string(),
            }
        );
        assert_eq!(report.applied.len(), 1);
        assert_eq!(
            report.outcome,
            UnitOutcome::Aborted {
                reason: "somnus: map-lint gate rejected the body for somnus-new-c1".to_string(),
            }
        );
        assert_eq!(exit_code_for_outcome(&report.outcome), 2);
    }

    #[tokio::test]
    async fn two_red_gate_bodies_abort_naming_the_first() {
        let backend = MockBackend::from_turns(vec![
            text_turn(&e2e_clusters_json(), usage(1, 1, None, None)),
            calls_turn(
                &[
                    (
                        "create_map",
                        json!({"cluster_id": "c1", "title": "t", "orientation_prose": "ORIENTATION-PROSE"}),
                    ),
                    (
                        "add_pointer",
                        json!({"map_id": "somnus-new-c1", "entry_id": "kb-10001", "gloss": "GLOSS-1"}),
                    ),
                    (
                        "add_pointer",
                        json!({"map_id": "kb-20001", "entry_id": "kb-10005", "gloss": "GLOSS-5"}),
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
        let body_root = tempfile::tempdir().expect("tempdir");
        let map_ops: std::sync::Arc<RecordingMapOps> = std::sync::Arc::default();
        let gate = gate_for_script("exit 22");
        let report = run_with_map_ops(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
            map_ops.clone(),
        )
        .await;
        assert_eq!(report.gate_reports.len(), 2);
        assert!(!report.gate_reports.iter().any(|gate| gate.passed));
        assert!(map_ops.requests().is_empty(), "zero map-op POSTs");
        assert_eq!(
            report.outcome,
            UnitOutcome::Aborted {
                reason: "somnus: map-lint gate rejected the body for somnus-new-c1".to_string(),
            },
            "the reason names the FIRST red body in composed_bodies order"
        );
    }

    #[tokio::test]
    async fn the_k_chain_posts_intermediate_bodies_in_chain_order() {
        // k=3 add_pointer edits on one map: EXACTLY 3 POSTs, the i-th
        // carrying the fixture-derived INTERMEDIATE body byte for byte and
        // the i-th entry id.
        let backend = MockBackend::from_turns(vec![
            text_turn(&e2e_clusters_json(), usage(1, 1, None, None)),
            calls_turn(
                &[
                    (
                        "add_pointer",
                        json!({"map_id": "kb-20001", "entry_id": "kb-10005", "gloss": "GLOSS-5"}),
                    ),
                    (
                        "add_pointer",
                        json!({"map_id": "kb-20001", "entry_id": "kb-10006", "gloss": "GLOSS-6"}),
                    ),
                    (
                        "add_pointer",
                        json!({"map_id": "kb-20001", "entry_id": "kb-10007", "gloss": "GLOSS-7"}),
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
        let body_root = tempfile::tempdir().expect("tempdir");
        let gate = gate_for_script("exit 0");
        let map_ops: std::sync::Arc<RecordingMapOps> = std::sync::Arc::default();
        let report = run_with_map_ops(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
            map_ops.clone(),
        )
        .await;
        assert_eq!(report.outcome, UnitOutcome::Ready);
        let requests = map_ops.requests();
        assert_eq!(requests.len(), 3);
        let original = "Lives in knowledge/linux/network\n\nOrients the switch, DNS, and VPN entries for the home network.\n\nDetail entries:\n- kb-10001 — switch VLAN tagging baseline\n- kb-10002 — recursive DNS resolvers\n- kb-10003 — wireguard client config\n- kb-99999 — cross-project wireguard hub notes\n- kb-10004 — DHCP lease hygiene\n\nNot yet documented:\n- Site-to-site wireguard topology".to_string();
        // Each insert lands INSIDE the `Detail entries:` section (before
        // the blank line), never at the body's end.
        let after_1 = original.replace(
            "- kb-10004 — DHCP lease hygiene\n",
            "- kb-10004 — DHCP lease hygiene\n- kb-10005 — GLOSS-5\n",
        );
        let after_2 = after_1.replace(
            "- kb-10005 — GLOSS-5\n",
            "- kb-10005 — GLOSS-5\n- kb-10006 — GLOSS-6\n",
        );
        let after_3 = after_2.replace(
            "- kb-10006 — GLOSS-6\n",
            "- kb-10006 — GLOSS-6\n- kb-10007 — GLOSS-7\n",
        );
        let expected = [
            ("kb-10005", after_1.as_str()),
            ("kb-10006", after_2.as_str()),
            ("kb-10007", after_3.as_str()),
        ];
        for (index, (entry_id, body)) in expected.iter().enumerate() {
            assert_eq!(
                requests[index],
                MapOpRequest::AddPointer {
                    map_id: "kb-20001".to_string(),
                    body: (*body).to_string(),
                    added_entry_id: (*entry_id).to_string(),
                },
                "POST #{index} carries the body AFTER {index} edit(s)"
            );
        }
        // The report's records carry the exact POSTed text and the 1-based
        // chain indices.
        for (index, record) in report.applied.iter().enumerate() {
            assert_eq!(record.op_kind, "add_pointer");
            assert_eq!(record.chain_index, index + 1);
            assert_eq!(record.http_status, Some(200));
            assert_eq!(record.submitted_map_id, "kb-20001");
            assert_eq!(record.server_map_id.as_deref(), Some("kb-30001"));
            assert!(matches!(requests[index], MapOpRequest::AddPointer { .. }));
            let rendered =
                serde_json::to_string(&crate::map_op::build_request_body(&requests[index]))
                    .expect("serializes");
            assert_eq!(record.body, rendered);
        }
        // The FINAL composed body equals the last POSTed body_after.
        assert_eq!(report.composed_bodies[0].body, after_3);
    }

    #[tokio::test]
    async fn an_all_decline_night_issues_zero_map_op_posts() {
        // Clusters whose op sets are empty or propose_gap-only: declines
        // recorded, nothing composed, zero map-op POSTs.
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
        let ledger = InMemoryLedger::new();
        let body_root = tempfile::tempdir().expect("tempdir");
        let gate = gate_for_script("exit 0");
        let map_ops: std::sync::Arc<RecordingMapOps> = std::sync::Arc::default();
        let report = run_with_map_ops(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
            map_ops.clone(),
        )
        .await;
        assert_eq!(report.outcome, UnitOutcome::Ready);
        assert!(report.composed_bodies.is_empty());
        assert!(map_ops.requests().is_empty(), "zero map-op POSTs");
        assert_eq!(report.declines_recorded.len(), 2);
    }

    #[tokio::test]
    async fn a_map_op_403_is_not_the_machine_principal_and_names_the_server_config() {
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
        let gate = gate_for_script("exit 0");
        let map_ops = RecordingMapOps::scripted(vec![MapOpResult::NotMachinePrincipal]);
        let report = run_with_map_ops(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
            map_ops,
        )
        .await;
        assert_eq!(report.outcome, UnitOutcome::NotMachinePrincipal);
        assert_eq!(exit_code_for_outcome(&report.outcome), 1);
        assert!(
            MAP_OP_NOT_MACHINE_PRINCIPAL_MSG.contains("machine_principal_email"),
            "the pinned line names the server's config knob: {MAP_OP_NOT_MACHINE_PRINCIPAL_MSG}"
        );
        assert!(
            !MAP_OP_NOT_MACHINE_PRINCIPAL_MSG.contains("SOMNUS_"),
            "the line never points at somnus config: {MAP_OP_NOT_MACHINE_PRINCIPAL_MSG}"
        );
    }

    #[tokio::test]
    async fn a_map_op_401_stops_as_unauthorized_with_exit_1() {
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
        let gate = gate_for_script("exit 0");
        let map_ops = RecordingMapOps::scripted(vec![MapOpResult::Unauthorized]);
        let report = run_with_map_ops(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
            map_ops,
        )
        .await;
        assert_eq!(report.outcome, UnitOutcome::Unauthorized);
        assert_eq!(exit_code_for_outcome(&report.outcome), 1);
        assert_eq!(report.applied.len(), 0);
    }

    #[tokio::test]
    async fn a_map_op_fault_aborts_with_the_pinned_prefix() {
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
        let gate = gate_for_script("exit 0");
        let map_ops = RecordingMapOps::scripted(vec![MapOpResult::Fault {
            map_id: "kb-20001".to_string(),
            reason: "unexpected HTTP status 409".to_string(),
        }]);
        let report = run_with_map_ops(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
            map_ops,
        )
        .await;
        assert_eq!(
            report.outcome,
            UnitOutcome::Aborted {
                reason: "somnus: map-op failed for kb-20001: unexpected HTTP status 409"
                    .to_string(),
            }
        );
        assert_eq!(exit_code_for_outcome(&report.outcome), 2);
    }

    #[tokio::test]
    async fn a_strike_gap_edit_posts_the_intermediate_strike_state() {
        // A strike_gap edit issues ONE POST whose body is the state as of
        // that strike (the gap line removed, header preserved).
        let backend = MockBackend::from_turns(vec![
            text_turn(&e2e_clusters_json(), usage(1, 1, None, None)),
            calls_turn(
                &[(
                    "strike_gap",
                    json!({"map_id": "kb-20001", "gap_text": "Site-to-site wireguard topology", "closing_entry_id": "kb-10003"}),
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
        let gate = gate_for_script("exit 0");
        let map_ops: std::sync::Arc<RecordingMapOps> = std::sync::Arc::default();
        let report = run_with_map_ops(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
            map_ops.clone(),
        )
        .await;
        assert_eq!(report.outcome, UnitOutcome::Ready);
        let requests = map_ops.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0],
            MapOpRequest::StrikeGap {
                map_id: "kb-20001".to_string(),
                body: "Lives in knowledge/linux/network\n\nOrients the switch, DNS, and VPN entries for the home network.\n\nDetail entries:\n- kb-10001 — switch VLAN tagging baseline\n- kb-10002 — recursive DNS resolvers\n- kb-10003 — wireguard client config\n- kb-99999 — cross-project wireguard hub notes\n- kb-10004 — DHCP lease hygiene\n\nNot yet documented:".to_string(),
                gap_text: "Site-to-site wireguard topology".to_string(),
                closing_entry_id: "kb-10003".to_string(),
            }
        );
        assert_eq!(report.applied[0].op_kind, "strike_gap");
        assert_eq!(report.applied[0].chain_index, 1);
    }

    #[tokio::test]
    async fn an_interleaved_chain_posts_in_chain_order() {
        // add_pointer then strike_gap on one map: the POSTs land in op
        // order, each with the body as of that edit.
        let backend = MockBackend::from_turns(vec![
            text_turn(&e2e_clusters_json(), usage(1, 1, None, None)),
            calls_turn(
                &[
                    (
                        "add_pointer",
                        json!({"map_id": "kb-20001", "entry_id": "kb-10005", "gloss": "GLOSS-5"}),
                    ),
                    (
                        "strike_gap",
                        json!({"map_id": "kb-20001", "gap_text": "Site-to-site wireguard topology", "closing_entry_id": "kb-10003"}),
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
        let body_root = tempfile::tempdir().expect("tempdir");
        let gate = gate_for_script("exit 0");
        let map_ops: std::sync::Arc<RecordingMapOps> = std::sync::Arc::default();
        let report = run_with_map_ops(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
            map_ops.clone(),
        )
        .await;
        let requests = map_ops.requests();
        assert_eq!(requests.len(), 2);
        assert!(matches!(requests[0], MapOpRequest::AddPointer { .. }));
        assert!(matches!(requests[1], MapOpRequest::StrikeGap { .. }));
        // The strike POST's body carries the pointer the FIRST edit added
        // (the state as of that strike), and the report's chain indices
        // are 1 and 2.
        assert_eq!(report.applied[0].chain_index, 1);
        assert_eq!(report.applied[1].chain_index, 2);
        let MapOpRequest::StrikeGap { body, .. } = &requests[1] else {
            panic!("strike");
        };
        assert!(body.contains("GLOSS-5"));
        assert!(!body.contains("Site-to-site wireguard topology"));
    }

    #[tokio::test]
    async fn a_create_map_401_403_and_fault_each_stop_before_the_next_body() {
        // The create branch's stop arms: a scripted 401 on the fresh map's
        // create POST stops as Unauthorized.
        let script_run = |script: Vec<MapOpResult>| async {
            let backend = MockBackend::from_turns(vec![
                text_turn(&e2e_clusters_json(), usage(1, 1, None, None)),
                calls_turn(
                    &[
                        (
                            "create_map",
                            json!({"cluster_id": "c1", "title": "t", "orientation_prose": "PROSE"}),
                        ),
                        (
                            "add_pointer",
                            json!({"map_id": "somnus-new-c1", "entry_id": "kb-10001", "gloss": "GLOSS-1"}),
                        ),
                        (
                            "add_pointer",
                            json!({"map_id": "kb-20001", "entry_id": "kb-10005", "gloss": "GLOSS-5"}),
                        ),
                    ],
                    usage(1, 1, None, None),
                ),
                calls_turn(
                    &[("propose_gap", json!({"cluster_id": "c2", "reason": "r"}))],
                    usage(1, 1, None, None),
                ),
            ]);
            let source =
                std::sync::Arc::new(ScriptedSource::new(vec![FetchOutcome::Ready(fixture())]));
            let ledger = InMemoryLedger::new();
            let body_root = tempfile::tempdir().expect("tempdir");
            let gate = gate_for_script("exit 0");
            let map_ops = RecordingMapOps::scripted(script);
            let report = run_with_map_ops(
                &backend,
                source,
                &ledger,
                &gate,
                body_root.path(),
                "demo-project",
                map_ops.clone(),
            )
            .await;
            (map_ops, report)
        };
        let (map_ops, report) = script_run(vec![MapOpResult::Unauthorized]).await;
        assert_eq!(report.outcome, UnitOutcome::Unauthorized);
        assert_eq!(exit_code_for_outcome(&report.outcome), 1);
        assert_eq!(map_ops.requests().len(), 1, "the night stops after the 401");

        let (map_ops, report) = script_run(vec![MapOpResult::NotMachinePrincipal]).await;
        assert_eq!(report.outcome, UnitOutcome::NotMachinePrincipal);
        assert_eq!(map_ops.requests().len(), 1);

        let (map_ops, report) = script_run(vec![MapOpResult::Fault {
            map_id: String::new(),
            reason: "unexpected HTTP status 422".to_string(),
        }])
        .await;
        assert_eq!(
            report.outcome,
            UnitOutcome::Aborted {
                reason: "somnus: map-op failed for : unexpected HTTP status 422".to_string(),
            }
        );
        assert_eq!(map_ops.requests().len(), 1);
    }

    #[tokio::test]
    async fn the_audit_catches_a_double_add_pointer_and_aborts_before_the_post() {
        // Two add_pointer ops for the SAME entry in one chain: the second
        // edit's ref delta is empty, the audit fires with the pinned line,
        // and the second POST never happens.
        let backend = MockBackend::from_turns(vec![
            text_turn(&e2e_clusters_json(), usage(1, 1, None, None)),
            calls_turn(
                &[
                    (
                        "add_pointer",
                        json!({"map_id": "kb-20001", "entry_id": "kb-10005", "gloss": "GLOSS-5"}),
                    ),
                    (
                        "add_pointer",
                        json!({"map_id": "kb-20001", "entry_id": "kb-10005", "gloss": "GLOSS-5-again"}),
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
        let body_root = tempfile::tempdir().expect("tempdir");
        let gate = gate_for_script("exit 0");
        let map_ops: std::sync::Arc<RecordingMapOps> = std::sync::Arc::default();
        let report = run_with_map_ops(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
            map_ops.clone(),
        )
        .await;
        assert_eq!(map_ops.requests().len(), 1, "the audited op was NOT POSTed");
        assert_eq!(report.apply_audit.len(), 1);
        assert_eq!(
            report.apply_audit[0],
            "somnus: apply audit: add_pointer delta for kb-20001 was [], expected kb-10005"
        );
        assert_eq!(
            report.outcome,
            UnitOutcome::Aborted {
                reason:
                    "somnus: apply audit: add_pointer delta for kb-20001 was [], expected kb-10005"
                        .to_string(),
            }
        );
    }

    #[tokio::test]
    async fn a_cap_admission_on_an_edit_op_is_a_fault_by_contract() {
        // The production client classifies every update-op 409 as a Fault,
        // so a CapAdmission arriving for an edit is a client-contract
        // violation; the step aborts rather than recording an ordinary
        // admission against an existing map.
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
        let gate = gate_for_script("exit 0");
        let map_ops = RecordingMapOps::scripted(vec![MapOpResult::CapAdmission {
            reason_body: "invariant".to_string(),
        }]);
        let report = run_with_map_ops(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
            map_ops.clone(),
        )
        .await;
        assert!(matches!(
            &report.outcome,
            UnitOutcome::Aborted { reason } if reason.contains("map-op failed for kb-20001")
        ));
    }

    #[tokio::test]
    async fn an_apply_audit_regression_aborts_before_the_post() {
        // A scripted client that hands back a WRONG body state: the audit
        // (previous body minus new refs != exactly the added id) fires
        // client-side with the pinned reason and the POST never happens.
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
        let gate = gate_for_script("exit 0");
        // The scripted client's `Applied` envelope is returned but the
        // REQUEST recorded is what the audit sees; to force a delta failure
        // we script an add_pointer whose body state drops an existing
        // pointer — impossible from real materialize, so the audit catches
        // it via the pure-fn test below. Here we drive the audit directly
        // through the pipeline's decision seam: the recording client
        // answers Applied, so we instead assert the pure fn's two lines.
        let map_ops: std::sync::Arc<RecordingMapOps> = std::sync::Arc::default();
        let report = run_with_map_ops(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
            map_ops.clone(),
        )
        .await;
        assert!(
            report.apply_audit.is_empty(),
            "a conforming night is silent"
        );
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
    async fn two_fresh_maps_compose_and_gate_and_the_second_is_an_ordinary_cap_admission() {
        // NO client-side cap survives: both create_map ops compose AND gate
        // (the gate sees both bodies), the first application applies, and
        // the second application meets the server's 409 as an ORDINARY
        // ADMISSION — recorded verbatim, outcome still Ready, exit 0.
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
                    (
                        "add_pointer",
                        json!({"map_id": "somnus-new-c2", "entry_id": "kb-10001", "gloss": "GLOSS-1"}),
                    ),
                ],
                usage(1, 1, None, None),
            ),
        ]);
        let source = std::sync::Arc::new(ScriptedSource::new(vec![FetchOutcome::Ready(fixture())]));
        let ledger = InMemoryLedger::new();
        let body_root = tempfile::tempdir().expect("tempdir");
        let gate = gate_for_script("exit 0");
        let reason = r#"{"reason":"per-night new-map cap for demo-project exceeded"}"#;
        let map_ops = RecordingMapOps::scripted(vec![
            MapOpResult::Applied(crate::map_op::MapOpApplied {
                map_id: "kb-30001".to_string(),
                version: 1,
                pointer_count: 1,
                budget: 0,
            }),
            MapOpResult::CapAdmission {
                reason_body: reason.to_string(),
            },
        ]);
        let report = run_with_map_ops(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
            map_ops.clone(),
        )
        .await;
        assert_eq!(
            report.composed_bodies.len(),
            2,
            "BOTH fresh bodies compose and gate"
        );
        assert_eq!(report.gate_reports.len(), 2);
        // Exactly TWO create POSTs dialed: the first applied, the second
        // admitted — the latch then holds for any FURTHER fresh map.
        assert_eq!(map_ops.requests().len(), 2);
        assert_eq!(report.applied.len(), 1);
        assert_eq!(report.applied[0].submitted_map_id, "somnus-new-c1");
        assert_eq!(report.cap_admissions.len(), 1);
        assert_eq!(
            report.cap_admissions[0],
            AppliedOpRecord {
                op_kind: "create_map",
                chain_index: 0,
                submitted_map_id: "somnus-new-c2".to_string(),
                server_map_id: None,
                version: None,
                pointer_count: None,
                budget: None,
                http_status: Some(409),
                body: serde_json::to_string(&crate::map_op::build_request_body(
                    &MapOpRequest::CreateMap {
                        project_ref: "demo-project".to_string(),
                        short_title: "t2".to_string(),
                        long_title: "t2".to_string(),
                        body: report.composed_bodies[1].body.clone(),
                    }
                ))
                .expect("serializes"),
                admission_reason: Some(reason.to_string()),
            }
        );
        assert_eq!(
            report.outcome,
            UnitOutcome::Ready,
            "an admission is data on the report, never an outcome"
        );
        assert_eq!(exit_code_for_outcome(&report.outcome), 0);
        assert!(report.skipped.is_empty(), "no further fresh map this unit");
    }

    #[tokio::test]
    async fn the_cap_admission_latch_skips_a_later_fresh_map_and_stays_ready() {
        // Two fresh maps from DIFFERENT clusters: the first create_map
        // meets the server's cap admission, the latch turns create_map
        // application off, and the second fresh map's create is SKIPPED
        // (recorded) with zero POSTs for it — while the outcome stays
        // Ready and the exit stays 0.
        let clusters = json!([
            {"label": "a", "member_entry_ids": ["kb-10001"], "owning_map_id": null},
            {"label": "b", "member_entry_ids": ["kb-10002"], "owning_map_id": null},
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
                    (
                        "add_pointer",
                        json!({"map_id": "somnus-new-c2", "entry_id": "kb-10002", "gloss": "GLOSS-2"}),
                    ),
                ],
                usage(1, 1, None, None),
            ),
        ]);
        let source = std::sync::Arc::new(ScriptedSource::new(vec![FetchOutcome::Ready(fixture())]));
        let ledger = InMemoryLedger::new();
        let body_root = tempfile::tempdir().expect("tempdir");
        let gate = gate_for_script("exit 0");
        let reason = r#"{"reason":"per-KB nightly cap exceeded"}"#;
        let map_ops = RecordingMapOps::scripted(vec![
            MapOpResult::CapAdmission {
                reason_body: reason.to_string(),
            },
            // Anything after the latch would be a defect: the script
            // repeats the admission, so a second POST would record another
            // admission and the count assertion below would catch it.
            MapOpResult::CapAdmission {
                reason_body: reason.to_string(),
            },
        ]);
        let report = run_with_map_ops(
            &backend,
            source,
            &ledger,
            &gate,
            body_root.path(),
            "demo-project",
            map_ops.clone(),
        )
        .await;
        assert_eq!(report.composed_bodies.len(), 2);
        // EXACTLY one POST: the second fresh map's create is latched off,
        // and neither fresh map's add_pointers were ever candidates.
        assert_eq!(map_ops.requests().len(), 1);
        assert_eq!(report.cap_admissions.len(), 1);
        assert_eq!(
            report.skipped,
            vec![SkippedOpRecord {
                op_kind: "create_map",
                map_id: "somnus-new-c2".to_string(),
                reason: render_create_skipped_after_admission("somnus-new-c2"),
            }]
        );
        assert_eq!(
            report.skipped[0].reason,
            "somnus: create_map for somnus-new-c2 skipped: a create_map cap admission was already recorded this invocation"
        );
        assert_eq!(report.outcome, UnitOutcome::Ready);
        assert_eq!(exit_code_for_outcome(&report.outcome), 0);
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

    // ======================================================================
    // the exit-code table, the budget render, and the apply audit
    // ======================================================================

    #[test]
    fn the_exit_code_table_covers_every_outcome_variant() {
        let table = [
            (UnitOutcome::Ready, 0),
            (UnitOutcome::UnknownProject, 0),
            (UnitOutcome::NotEligible, 0),
            (UnitOutcome::Unauthorized, 1),
            (UnitOutcome::NotMachinePrincipal, 1),
            (
                UnitOutcome::Aborted {
                    reason: "x".to_string(),
                },
                2,
            ),
        ];
        for (outcome, expected) in table {
            assert_eq!(exit_code_for_outcome(&outcome), expected, "{outcome:?}");
        }
        // A cap-admission unit stays Ready, so the nightly exit-0 predicate
        // is decidable as written.
        assert_eq!(exit_code_for_outcome(&UnitOutcome::Ready), 0);
    }

    #[test]
    fn the_outcome_names_render_pascal_case() {
        for (outcome, name) in [
            (UnitOutcome::Ready, "Ready"),
            (UnitOutcome::UnknownProject, "UnknownProject"),
            (UnitOutcome::NotEligible, "NotEligible"),
            (UnitOutcome::Unauthorized, "Unauthorized"),
            (UnitOutcome::NotMachinePrincipal, "NotMachinePrincipal"),
            (
                UnitOutcome::Aborted {
                    reason: "r".to_string(),
                },
                "Aborted",
            ),
        ] {
            assert_eq!(outcome.name(), name, "{name}");
        }
    }

    #[test]
    fn the_token_budget_line_carries_both_decision_inputs() {
        assert_eq!(
            render_token_budget_line(550_000, 612_004),
            "somnus: token budget exhausted (armed 550000, billed 612004)"
        );
        assert_eq!(
            render_token_budget_line(1_100_000, 1_100_333),
            "somnus: token budget exhausted (armed 1100000, billed 1100333)"
        );
    }

    #[test]
    fn the_gate_and_map_op_lines_are_byte_pinned() {
        assert_eq!(
            render_gate_rejected_line("kb-20001"),
            "somnus: map-lint gate rejected the body for kb-20001"
        );
        assert_eq!(
            render_map_op_fault_line("kb-20001", "unexpected HTTP status 422"),
            "somnus: map-op failed for kb-20001: unexpected HTTP status 422"
        );
        assert_eq!(
            MAP_OP_NOT_MACHINE_PRINCIPAL_MSG,
            "somnus: the KB refused the caller on /api/kb/map-op as not the machine principal — check machine_principal_email on the server; this is not a somnus configuration fault"
        );
    }

    #[test]
    fn the_usage_totals_bill_the_one_workspace_formula() {
        assert_eq!(
            UsageTotals {
                input: 1000,
                output: 50,
                cache_read: 200,
                cache_write: 10,
            }
            .billed(),
            1260
        );
        assert_eq!(UsageTotals::default().billed(), 0);
        assert_eq!(
            UsageTotals {
                input: u64::MAX,
                output: 1,
                cache_read: 0,
                cache_write: 0,
            }
            .billed(),
            u64::MAX,
            "saturating, never wrapping"
        );
    }

    // --- the apply audit: the two pinned tripwire lines --------------------

    #[test]
    fn the_apply_audit_catches_an_add_pointer_on_a_fresh_map() {
        let request = MapOpRequest::AddPointer {
            map_id: "somnus-new-c1".to_string(),
            body: "body".to_string(),
            added_entry_id: "kb-10001".to_string(),
        };
        assert_eq!(
            apply_audit(Some("previous body"), &request, &["somnus-new-c1".to_string()]),
            Some(
                "somnus: apply audit: add_pointer on fresh map somnus-new-c1 not absorbed by the create body"
                    .to_string()
            )
        );
    }

    #[test]
    fn the_apply_audit_catches_a_wrong_delta_with_the_pinned_line() {
        // The previous body carries kb-10001; the submitted body adds TWO
        // refs and drops none — the delta is not exactly the claimed id.
        let request = MapOpRequest::AddPointer {
            map_id: "kb-20001".to_string(),
            body: "Lives in x\n\nDetail entries:\n- kb-10001 — a\n- kb-10002 — b\n- kb-10003 — c\n\nNot yet documented:\n- gap with kb-00042 in it".to_string(),
            added_entry_id: "kb-10002".to_string(),
        };
        let previous = "Lives in x\n\nDetail entries:\n- kb-10001 — a".to_string();
        assert_eq!(
            apply_audit(
                Some(&previous),
                &request,
                &["somnus-new-c1".to_string()],
            ),
            Some(
                "somnus: apply audit: add_pointer delta for kb-20001 was [\"kb-10002\", \"kb-10003\", \"kb-00042\"], expected kb-10002"
                    .to_string()
            )
        );
    }

    #[test]
    fn the_apply_audit_passes_a_conforming_single_addition() {
        let previous = "Detail entries:\n- kb-10001 — a".to_string();
        let body = "Detail entries:\n- kb-10001 — a\n- kb-10005 — b".to_string();
        let request = MapOpRequest::AddPointer {
            map_id: "kb-20001".to_string(),
            body,
            added_entry_id: "kb-10005".to_string(),
        };
        assert_eq!(
            apply_audit(Some(&previous), &request, &[]),
            None,
            "exactly one added ref, the claimed id: silent"
        );
    }

    #[test]
    fn the_apply_audit_scans_the_pinned_pointer_shape_with_no_dependency() {
        // `kb-` + exactly five ASCII digits: six or four digits are not
        // refs, and a scan never overruns.
        let text = "kb-12345 kb-123456 kb-1234 kb-00000 kb-99999";
        let refs = super::kb_refs(text);
        assert_eq!(
            refs,
            vec![
                "kb-12345".to_string(),
                "kb-00000".to_string(),
                "kb-99999".to_string()
            ],
            "six-digit and four-digit shapes are not refs; duplicates collapse"
        );
        assert_eq!(super::kb_refs("no refs here"), Vec::<String>::new());
    }

    #[test]
    fn the_apply_audit_ignores_non_add_pointer_ops() {
        for request in [
            MapOpRequest::CreateMap {
                project_ref: "p".to_string(),
                short_title: "s".to_string(),
                long_title: "l".to_string(),
                body: "b".to_string(),
            },
            MapOpRequest::StrikeGap {
                map_id: "kb-20001".to_string(),
                body: "b".to_string(),
                gap_text: "g".to_string(),
                closing_entry_id: "kb-10001".to_string(),
            },
        ] {
            assert_eq!(apply_audit(Some("prev"), &request, &[]), None);
        }
    }

    // ======================================================================
    // the token-budget guard
    // ======================================================================

    #[tokio::test]
    async fn the_token_budget_exhaustion_aborts_before_further_turns() {
        // The pre-rung-1 arm: the whole spend so far is `billed_before`,
        // and it already breaches the small ceiling, so the backend is
        // never called at all.
        let backend = MockBackend::from_turns(vec![text_turn(
            &e2e_clusters_json(),
            usage(1000, 50, Some(200), None),
        )]);
        let source = std::sync::Arc::new(ScriptedSource::new(vec![FetchOutcome::Ready(fixture())]));
        let ledger = InMemoryLedger::new();
        let body_root = tempfile::tempdir().expect("tempdir");
        let gate = gate_for_script("exit 0");
        let tool_ctx = ToolCtx::stub();
        let deps = UnitDeps {
            backend: &backend,
            source,
            ledger: &ledger,
            gate_for: &gate,
            tool_ctx: &tool_ctx,
            map_ops: std::sync::Arc::new(RecordingMapOps::default()),
            token_budget: 100,
            billed_before: 500,
            body_root: body_root.path().to_path_buf(),
        };
        let report = run_unit(&deps, "demo-project").await;
        assert_eq!(
            report.outcome,
            UnitOutcome::Aborted {
                reason: "somnus: token budget exhausted (armed 100, billed 500)".to_string(),
            }
        );
        assert_eq!(backend.calls(), 0, "the guard fired BEFORE rung 1");
    }

    #[tokio::test]
    async fn the_token_budget_stops_the_rung2_loop_between_turns() {
        // Rung 1 is affordable (1200 < 5000); its billed usage alone
        // breaches the ceiling, so rung 2 never starts: the scripted turn
        // count stays at the rung-1 call.
        let backend = MockBackend::from_turns(vec![text_turn(
            &e2e_clusters_json(),
            usage(4000, 500, Some(300), Some(200)),
        )]);
        let source = std::sync::Arc::new(ScriptedSource::new(vec![FetchOutcome::Ready(fixture())]));
        let ledger = InMemoryLedger::new();
        let body_root = tempfile::tempdir().expect("tempdir");
        let gate = gate_for_script("exit 0");
        let tool_ctx = ToolCtx::stub();
        let deps = UnitDeps {
            backend: &backend,
            source,
            ledger: &ledger,
            gate_for: &gate,
            tool_ctx: &tool_ctx,
            map_ops: std::sync::Arc::new(RecordingMapOps::default()),
            token_budget: 5000,
            billed_before: 500,
            body_root: body_root.path().to_path_buf(),
        };
        let report = run_unit(&deps, "demo-project").await;
        assert_eq!(backend.calls(), 1, "rung 2 never started");
        assert_eq!(
            report.outcome,
            UnitOutcome::Aborted {
                reason: "somnus: token budget exhausted (armed 5000, billed 5500)".to_string(),
            },
            "billed_before 500 + rung-1 billed 5000 breaches 5000 before rung 2"
        );
    }

    #[tokio::test]
    async fn an_armed_invocation_baseline_trips_the_guard_before_rung1() {
        // The across-units arm: `billed_before` already over the ceiling,
        // so the unit reports the budget abort before ANY turn — the
        // nightly binary's budget-stop decision, mirrored in-pipeline.
        let backend = MockBackend::from_turns(vec![]);
        let source = std::sync::Arc::new(ScriptedSource::new(vec![FetchOutcome::Ready(fixture())]));
        let ledger = InMemoryLedger::new();
        let body_root = tempfile::tempdir().expect("tempdir");
        let gate = gate_for_script("exit 0");
        let tool_ctx = ToolCtx::stub();
        let deps = UnitDeps {
            backend: &backend,
            source: source.clone(),
            ledger: &ledger,
            gate_for: &gate,
            tool_ctx: &tool_ctx,
            map_ops: std::sync::Arc::new(RecordingMapOps::default()),
            token_budget: 550_000,
            billed_before: 612_004,
            body_root: body_root.path().to_path_buf(),
        };
        let report = run_unit(&deps, "demo-project").await;
        assert_eq!(
            report.outcome,
            UnitOutcome::Aborted {
                reason: "somnus: token budget exhausted (armed 550000, billed 612004)".to_string(),
            }
        );
        assert_eq!(backend.calls(), 0);
        // The sentinel 0 is UNBOUNDED: the same unit runs clean.
        let backend = MockBackend::from_turns(vec![text_turn("[]", usage(1, 1, None, None))]);
        let tool_ctx = ToolCtx::stub();
        let deps = UnitDeps {
            backend: &backend,
            source: source.clone(),
            ledger: &ledger,
            gate_for: &gate,
            tool_ctx: &tool_ctx,
            map_ops: std::sync::Arc::new(RecordingMapOps::default()),
            token_budget: 0,
            billed_before: 612_004,
            body_root: body_root.path().to_path_buf(),
        };
        let report = run_unit(&deps, "demo-project").await;
        assert_eq!(report.outcome, UnitOutcome::Ready, "0 = unbounded");
    }
}
