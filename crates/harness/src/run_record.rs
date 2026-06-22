//! Core run-record data model — the serializable state the inner loop is a
//! function of.
//!
//! This module is **types only** by design. It defines the run record itself,
//! the per-criterion checklist (claim vs. verify), the finish disposition, and
//! the event-log entries. Persistence (the `RunStore` trait + `SQLite`),
//! the tool layer, and the loop engine all live elsewhere; the
//! illegal-states-unrepresentable encoding here is what those layers build on.
//!
//! See `docs/design/02-run-record-schema.md` for the design rationale —
//! especially the split between durable state and disposable context, and the
//! "agent claims, harness verifies" mechanism enforced by the
//! [`CriterionStatus::Verified`] variant requiring [`Evidence`].
//!
//! ## Determinism
//!
//! Maps use [`BTreeMap`], never [`std::collections::HashMap`]. Serialized JSON
//! is therefore byte-stable across runs — the discipline that keeps the prompt
//! cache hitting and makes the event log replayable.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Current run-record schema version. Bump deliberately when the on-disk
/// format changes; the value is carried on every [`RunRecord`] so a loader
/// can migrate or refuse stale data.
pub const SCHEMA_VERSION: u32 = 1;

// ===== Top-level run record ============================================

/// The single serializable state the inner loop reduces over.
///
/// The critical structural choice is the split between **durable state**
/// (survives a context reset — `phase`, `durable_facts`, `budgets`,
/// `last_gate_result`, `disposition`) and **disposable context** (`messages`).
/// That split is what lets the harness do a fresh-context restart before
/// quality degrades, without losing task progress.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunRecord {
    /// Deterministic id: `hash(task_id + attempt_n)`. Stable across restarts,
    /// so re-dispatch of the same attempt addresses the same record
    /// (idempotent dispatch).
    pub run_id: String,
    /// Schema tag for migration. See [`SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Which attempt this is for the underlying task.
    pub attempt_n: u32,

    // ---- frozen at dispatch (the seam from GTD) ----
    /// Groomed task snapshot — what the harness was dispatched against.
    pub task: Task,
    /// Per-project knobs: gate commands + model-routing hint.
    pub project_config: ProjectConfig,

    // ---- DURABLE STATE (survives a context reset) ----
    /// Outer control-flow position. Driven by hard-coded Rust, not the model.
    pub phase: Phase,
    /// Cross-window carrier: the AC checklist + append-only findings.
    pub durable_facts: DurableFacts,
    /// Consumed + limits for the bounded inner loop.
    pub budgets: Budgets,
    /// Latest structured `run_checks` result (`None` before the first run).
    pub last_gate_result: Option<GateResult>,
    /// `None` until the run terminates with a disposition.
    pub disposition: Option<Disposition>,

    // ---- DISPOSABLE CONTEXT (scratch; may be dropped/compacted) ----
    /// Current model context window. Rebuildable from the event log on
    /// crash-resume; intentionally dropped on a fresh-context restart.
    pub messages: Vec<Message>,
}

// ===== Task & project config ==========================================

/// Groomed task snapshot, frozen at dispatch. The agent never edits the AC
/// itself — that anti-drift rule is enforced by the checklist mechanism
/// (see [`ChecklistItem`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Task {
    pub task_id: String,
    pub title: String,
    pub description: String,
    /// One entry per groomed acceptance criterion.
    pub acceptance_criteria: Vec<AcceptanceCriterion>,
    /// Files the task is scoped to (read/edit allowed by convention).
    pub files_in_scope: Vec<String>,
    /// What the task explicitly does NOT cover.
    pub scope_out: Vec<String>,
}

/// One groomed AC, paired with the check that verifies it (if any).
///
/// A `check` shipped with the AC is the gold-standard verification path
/// (`Verified(test)` evidence). When `check` is `None` the agent may need
/// to write its own check, or the AC may genuinely need a human reviewer
/// (`ClaimedDone` + `Evidence::NeedsHuman`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcceptanceCriterion {
    pub id: String,
    pub criterion: String,
    /// Optional shipped check command (gold-standard verification).
    pub check: Option<String>,
}

/// Per-project knobs the harness reads at dispatch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectConfig {
    /// Gate commands `run_checks` should run, keyed by gate name. `BTreeMap`
    /// (not `HashMap`) for deterministic JSON ordering — what keeps the
    /// prompt cache byte-stable across windows.
    pub run_checks: BTreeMap<String, String>,
    /// Routing hint for the model backend (e.g. `"sonnet"`, `"ollama:llama"`).
    pub model_routing_hint: Option<String>,
}

// ===== Phase ===========================================================

/// Outer control-flow position. The predictable outer sequence is hard-coded
/// Rust; the open, bounded inner loop is the `InnerLoop` phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    Init,
    Orient,
    InnerLoop,
    Checks,
    Finalize,
    Done,
}

// ===== Durable facts & the AC checklist ===============================

/// The cross-window carrier — a `passes:false` AC checklist plus a free-form
/// `findings` log of established facts / decisions / ruled-out approaches.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct DurableFacts {
    /// One entry per AC. Items are **immutable** by policy (the harness never
    /// edits ids or criterion text — drift attempts are a `Blocked`
    /// disposition, not a silent rewrite).
    pub checklist: Vec<ChecklistItem>,
    /// Append-only free-form memory carried across context resets.
    pub findings: Vec<String>,
}

/// One AC criterion, tracked through its state machine. The `status` is the
/// only field the inner loop mutates; `id` and `criterion` are fixed at
/// dispatch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChecklistItem {
    pub id: String,
    pub criterion: String,
    pub status: CriterionStatus,
}

/// Per-criterion state machine.
///
/// The agent may move a criterion up to `ClaimedDone` (a *claim* — "I think
/// this is met"). Only the harness writes `Verified`, and only after running
/// the criterion's check and seeing it pass.
///
/// **Illegal-states-unrepresentable:** the `Verified` variant carries
/// [`Evidence`], so a verified criterion without backing evidence simply
/// cannot exist as a value of this type. The
/// `verified_without_evidence_fails_to_deserialize` test pins this guarantee
/// at the wire boundary too.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CriterionStatus {
    NotStarted,
    InProgress,
    ClaimedDone,
    Verified(Evidence),
}

/// What backs a `Verified` status, in descending trust.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Evidence {
    /// A deterministic check passed. Strongest. The check is either shipped
    /// with the AC (gold standard) or written by the agent (the
    /// outer-review-backstopped case).
    Test { name: String, command: String },
    /// A calibrated, rubric-based LLM-judge passed. Probabilistic; weaker
    /// than `Test`.
    Judge { judge_id: String, rationale: String },
    /// No automatable check exists (genuinely subjective AC). The harness
    /// refuses to auto-verify; the outer review must decide.
    NeedsHuman,
}

// ===== Budgets =========================================================

/// Consumed + limits for the bounded inner loop.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budgets {
    pub consumed: BudgetConsumed,
    pub limits: BudgetLimits,
    /// Wall-clock start. Stored as a string (e.g. RFC 3339) so the type
    /// stays dependency-free; the format is the caller's contract, the
    /// record cares only that it round-trips byte-stably.
    pub wall_clock_start: String,
}

/// Spend so far, ticked by the loop after each step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct BudgetConsumed {
    pub iterations: u32,
    pub tokens: u64,
    /// Cost in micro-dollars (millionths of a dollar) to avoid floats and
    /// keep arithmetic deterministic.
    pub cost_micros: u64,
}

/// Caps the inner loop must not exceed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BudgetLimits {
    pub iterations: u32,
    pub tokens: u64,
    pub cost_micros: u64,
}

// ===== Gate result =====================================================

/// Latest structured `run_checks` output — per-gate pass/fail with bounded
/// failure extracts. The done-oracle reads from this, not from any model
/// self-report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateResult {
    /// `true` only if every gate in `gates` passed.
    pub passed: bool,
    /// Per-gate outcomes, keyed by gate name. `BTreeMap` for deterministic
    /// ordering.
    pub gates: BTreeMap<String, GateOutcome>,
}

/// Outcome of a single quality gate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateOutcome {
    pub passed: bool,
    pub summary: String,
    /// Bounded extract of the failure for steering. The full log lives on
    /// disk; the harness advertises that path via the tool layer.
    pub failure_extract: Option<String>,
}

// ===== Disposition =====================================================

/// Terminal status of a run. The discriminator is "does running the same
/// thing again have any chance of working?" — `Blocked` no, `Failed` maybe,
/// `Done` already worked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Disposition {
    /// Gates green and every criterion `Verified`.
    Done,
    /// The spec or environment is the problem; retrying unchanged cannot
    /// help (ambiguous AC, missing access, out-of-scope ask).
    Blocked { decision_needed: String },
    /// The run is the problem, the spec is fine; retrying might work.
    Failed { mode: FailureMode, summary: String },
}

/// Why a `Failed` run failed. Drives the outer harness's retry decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureMode {
    /// The agent looped / made no progress.
    Loop,
    /// A budget cap was hit (iterations, tokens, cost, or wall clock).
    BudgetExhausted,
    /// A tool kept erroring across retries (not transient).
    PersistentToolError,
    /// Network / infra blip; almost certainly worth retrying.
    TransientInfra,
}

/// Structured report attached to a disposition — also the eval-case seed
/// (every failed dispatch becomes a replayable eval case from its log).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DispositionReport {
    /// Per-AC final status + evidence.
    pub checklist_final: Vec<ChecklistItem>,
    /// What the run spent.
    pub budget_spent: BudgetConsumed,
    /// Pointer to the trajectory in the event log.
    pub event_log_ref: String,
}

// ===== Event log =======================================================

/// One message exchanged with the model. Disposable context: the loop can
/// drop or compact this without losing durable progress.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// Conversational role of a [`Message`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Append-only audit-trail entries. Each carries a monotonic `seq` — that
/// `seq` is the idempotency key for resume, so a tool side effect attempted
/// at `seq=N` will not be re-applied twice when the run resumes from an
/// interrupted step.
///
/// `Eq` is not derived because [`Event::ToolCallStarted`] carries
/// [`serde_json::Value`] (which can hold floats); `PartialEq` is enough for
/// the comparisons we want.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Event {
    /// The harness called the model.
    ModelCall {
        seq: u64,
        model: String,
        prompt_tokens: u64,
        completion_tokens: u64,
    },
    /// A tool call was started (recorded *before* execution — pairs with
    /// `ToolCallResult` by `seq`; an unpaired `started` on resume means
    /// "interrupted, re-run idempotently").
    ToolCallStarted {
        seq: u64,
        name: String,
        args: serde_json::Value,
    },
    /// A tool call completed (success or steering error).
    ToolCallResult {
        seq: u64,
        name: String,
        is_error: bool,
        summary: String,
        offload_path: Option<String>,
    },
    /// The outer phase advanced.
    PhaseTransition { seq: u64, from: Phase, to: Phase },
    /// A budget tick — periodic spend update.
    BudgetTick { seq: u64, consumed: BudgetConsumed },
    /// The terminal disposition was written.
    DispositionSet { seq: u64, disposition: Disposition },
}

// =======================================================================
// Tests
// =======================================================================

#[cfg(test)]
mod tests {
    use super::{
        AcceptanceCriterion, BudgetConsumed, BudgetLimits, Budgets, ChecklistItem, CriterionStatus,
        Disposition, DispositionReport, DurableFacts, Event, Evidence, FailureMode, GateOutcome,
        GateResult, Message, Phase, ProjectConfig, Role, RunRecord, SCHEMA_VERSION, Task,
    };
    use std::collections::BTreeMap;

    fn sample_project_config() -> ProjectConfig {
        let mut run_checks = BTreeMap::new();
        run_checks.insert("fmt".to_string(), "cargo fmt --check".to_string());
        run_checks.insert(
            "clippy".to_string(),
            "cargo clippy -- -D warnings".to_string(),
        );
        run_checks.insert("test".to_string(), "cargo nextest run".to_string());
        ProjectConfig {
            run_checks,
            model_routing_hint: Some("sonnet".to_string()),
        }
    }

    fn sample_task() -> Task {
        Task {
            task_id: "task-42".to_string(),
            title: "Add run-record types".to_string(),
            description: "Implement the serde-serializable run record.".to_string(),
            acceptance_criteria: vec![
                AcceptanceCriterion {
                    id: "ac1".to_string(),
                    criterion: "RunRecord exists with all fields.".to_string(),
                    check: Some("cargo nextest run -p harness".to_string()),
                },
                AcceptanceCriterion {
                    id: "ac2".to_string(),
                    criterion: "JSON round-trips byte-stably.".to_string(),
                    check: None,
                },
            ],
            files_in_scope: vec!["crates/harness/src/run_record.rs".to_string()],
            scope_out: vec!["RunStore / SQLite persistence".to_string()],
        }
    }

    fn sample_durable_facts() -> DurableFacts {
        DurableFacts {
            checklist: vec![
                ChecklistItem {
                    id: "ac1".to_string(),
                    criterion: "RunRecord exists with all fields.".to_string(),
                    status: CriterionStatus::Verified(Evidence::Test {
                        name: "round_trip_run_record".to_string(),
                        command: "cargo nextest run".to_string(),
                    }),
                },
                ChecklistItem {
                    id: "ac2".to_string(),
                    criterion: "JSON round-trips byte-stably.".to_string(),
                    status: CriterionStatus::InProgress,
                },
                ChecklistItem {
                    id: "ac3".to_string(),
                    criterion: "Docs accurate".to_string(),
                    status: CriterionStatus::ClaimedDone,
                },
                ChecklistItem {
                    id: "ac4".to_string(),
                    criterion: "Pending".to_string(),
                    status: CriterionStatus::NotStarted,
                },
            ],
            findings: vec![
                "Used BTreeMap, not HashMap, for determinism.".to_string(),
                "serde_json is the only structural dep.".to_string(),
            ],
        }
    }

    fn sample_budgets() -> Budgets {
        Budgets {
            consumed: BudgetConsumed {
                iterations: 7,
                tokens: 12_345,
                cost_micros: 6_780,
            },
            limits: BudgetLimits {
                iterations: 50,
                tokens: 1_000_000,
                cost_micros: 5_000_000,
            },
            wall_clock_start: "2026-06-22T00:00:00Z".to_string(),
        }
    }

    fn sample_gate_result() -> GateResult {
        let mut gates = BTreeMap::new();
        gates.insert(
            "fmt".to_string(),
            GateOutcome {
                passed: true,
                summary: "fmt clean".to_string(),
                failure_extract: None,
            },
        );
        gates.insert(
            "clippy".to_string(),
            GateOutcome {
                passed: false,
                summary: "1 warning".to_string(),
                failure_extract: Some("warning: unused variable `x`".to_string()),
            },
        );
        GateResult {
            passed: false,
            gates,
        }
    }

    fn sample_messages() -> Vec<Message> {
        vec![
            Message {
                role: Role::System,
                content: "You are a build engine.".to_string(),
            },
            Message {
                role: Role::User,
                content: "Do the task.".to_string(),
            },
            Message {
                role: Role::Assistant,
                content: "Calling tool...".to_string(),
            },
            Message {
                role: Role::Tool,
                content: "ok".to_string(),
            },
        ]
    }

    fn sample_run_record() -> RunRecord {
        RunRecord {
            run_id: "run-2026-06-22-abc".to_string(),
            schema_version: SCHEMA_VERSION,
            attempt_n: 1,
            task: sample_task(),
            project_config: sample_project_config(),
            phase: Phase::InnerLoop,
            durable_facts: sample_durable_facts(),
            budgets: sample_budgets(),
            last_gate_result: Some(sample_gate_result()),
            disposition: None,
            messages: sample_messages(),
        }
    }

    fn round_trip<T>(value: &T)
    where
        T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        let json = serde_json::to_string(value).expect("serialize");
        let back: T = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(value, &back, "round-trip should be lossless");
    }

    // ---- round-trip tests ----

    #[test]
    fn round_trip_run_record() {
        round_trip(&sample_run_record());
    }

    #[test]
    fn round_trip_all_phases() {
        for phase in [
            Phase::Init,
            Phase::Orient,
            Phase::InnerLoop,
            Phase::Checks,
            Phase::Finalize,
            Phase::Done,
        ] {
            round_trip(&phase);
        }
    }

    #[test]
    fn round_trip_all_criterion_statuses() {
        round_trip(&CriterionStatus::NotStarted);
        round_trip(&CriterionStatus::InProgress);
        round_trip(&CriterionStatus::ClaimedDone);
        round_trip(&CriterionStatus::Verified(Evidence::Test {
            name: "t".to_string(),
            command: "cargo test".to_string(),
        }));
        round_trip(&CriterionStatus::Verified(Evidence::Judge {
            judge_id: "j1".to_string(),
            rationale: "looks good".to_string(),
        }));
        round_trip(&CriterionStatus::Verified(Evidence::NeedsHuman));
    }

    #[test]
    fn round_trip_disposition_all_variants() {
        round_trip(&Disposition::Done);
        round_trip(&Disposition::Blocked {
            decision_needed: "Which API version?".to_string(),
        });
        for mode in [
            FailureMode::Loop,
            FailureMode::BudgetExhausted,
            FailureMode::PersistentToolError,
            FailureMode::TransientInfra,
        ] {
            round_trip(&Disposition::Failed {
                mode,
                summary: "stuck".to_string(),
            });
        }
    }

    #[test]
    fn round_trip_disposition_report() {
        let report = DispositionReport {
            checklist_final: sample_durable_facts().checklist,
            budget_spent: sample_budgets().consumed,
            event_log_ref: "run-2026-06-22-abc/events".to_string(),
        };
        round_trip(&report);
    }

    #[test]
    fn round_trip_all_event_variants() {
        let events = vec![
            Event::ModelCall {
                seq: 1,
                model: "claude-sonnet-4-6".to_string(),
                prompt_tokens: 1024,
                completion_tokens: 256,
            },
            Event::ToolCallStarted {
                seq: 2,
                name: "edit_file".to_string(),
                args: serde_json::json!({
                    "path": "crates/harness/src/run_record.rs",
                    "old_string": "foo",
                    "new_string": "bar",
                }),
            },
            Event::ToolCallResult {
                seq: 3,
                name: "edit_file".to_string(),
                is_error: false,
                summary: "1 occurrence replaced".to_string(),
                offload_path: Some("/run/log/3.txt".to_string()),
            },
            Event::PhaseTransition {
                seq: 4,
                from: Phase::Orient,
                to: Phase::InnerLoop,
            },
            Event::BudgetTick {
                seq: 5,
                consumed: BudgetConsumed {
                    iterations: 1,
                    tokens: 100,
                    cost_micros: 10,
                },
            },
            Event::DispositionSet {
                seq: 6,
                disposition: Disposition::Done,
            },
        ];
        for ev in &events {
            // Event has only PartialEq (Value blocks Eq); round_trip needs PartialEq.
            let json = serde_json::to_string(ev).expect("serialize");
            let back: Event = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(ev, &back);
        }
    }

    #[test]
    fn round_trip_run_record_with_disposition_set() {
        let mut r = sample_run_record();
        r.disposition = Some(Disposition::Blocked {
            decision_needed: "AC #2 conflicts with AC #5".to_string(),
        });
        round_trip(&r);
    }

    #[test]
    fn round_trip_run_record_with_no_gate_result_or_routing_hint() {
        let mut r = sample_run_record();
        r.last_gate_result = None;
        r.project_config.model_routing_hint = None;
        round_trip(&r);
    }

    #[test]
    fn round_trip_durable_facts_default_is_empty() {
        let df = DurableFacts::default();
        assert!(df.checklist.is_empty());
        assert!(df.findings.is_empty());
        round_trip(&df);
    }

    #[test]
    fn round_trip_budget_consumed_default_is_zero() {
        let bc = BudgetConsumed::default();
        assert_eq!(bc.iterations, 0);
        assert_eq!(bc.tokens, 0);
        assert_eq!(bc.cost_micros, 0);
        round_trip(&bc);
    }

    // ---- determinism ----

    #[test]
    fn json_key_ordering_is_deterministic_via_btreemap() {
        // Insert in a non-alphabetical order; BTreeMap reorders to sorted, so
        // the serialized JSON should put keys in alphabetical order — and
        // every serialization of the same value should be byte-identical.
        let mut run_checks = BTreeMap::new();
        run_checks.insert("zeta".to_string(), "z".to_string());
        run_checks.insert("alpha".to_string(), "a".to_string());
        run_checks.insert("mid".to_string(), "m".to_string());
        let cfg = ProjectConfig {
            run_checks,
            model_routing_hint: None,
        };

        let s1 = serde_json::to_string(&cfg).expect("serialize");
        let s2 = serde_json::to_string(&cfg).expect("serialize");
        assert_eq!(s1, s2, "serialization must be byte-stable");

        let alpha = s1.find("alpha").expect("alpha present");
        let mid = s1.find("mid").expect("mid present");
        let zeta = s1.find("zeta").expect("zeta present");
        assert!(alpha < mid, "alpha must precede mid in {s1}");
        assert!(mid < zeta, "mid must precede zeta in {s1}");
    }

    #[test]
    fn run_record_serialization_is_byte_stable_across_calls() {
        let r = sample_run_record();
        let s1 = serde_json::to_string(&r).expect("serialize");
        let s2 = serde_json::to_string(&r).expect("serialize");
        let s3 = serde_json::to_string(&r).expect("serialize");
        assert_eq!(s1, s2);
        assert_eq!(s2, s3);
    }

    // ---- illegal states unrepresentable ----

    #[test]
    fn verified_without_evidence_fails_to_deserialize() {
        // The `Verified` variant takes `Evidence`. Trying to parse it as a
        // bare unit variant (no payload) must fail — that is the
        // "illegal-states-unrepresentable" guarantee surfaced at the wire
        // boundary so external data can't smuggle in a status the type
        // system would reject.
        let bare = r#""Verified""#;
        let res: Result<CriterionStatus, _> = serde_json::from_str(bare);
        assert!(
            res.is_err(),
            "deserializing `Verified` without Evidence must fail, got Ok({:?})",
            res.ok()
        );

        // And the valid forms (every Evidence variant) parse fine.
        let json_needs_human = r#"{"Verified":"NeedsHuman"}"#;
        let parsed: CriterionStatus =
            serde_json::from_str(json_needs_human).expect("Verified+NeedsHuman is valid");
        assert!(matches!(
            parsed,
            CriterionStatus::Verified(Evidence::NeedsHuman)
        ));

        let json_test = r#"{"Verified":{"Test":{"name":"t","command":"cargo test"}}}"#;
        let parsed: CriterionStatus =
            serde_json::from_str(json_test).expect("Verified+Test is valid");
        assert!(matches!(
            parsed,
            CriterionStatus::Verified(Evidence::Test { .. })
        ));

        let json_judge = r#"{"Verified":{"Judge":{"judge_id":"j","rationale":"r"}}}"#;
        let parsed: CriterionStatus =
            serde_json::from_str(json_judge).expect("Verified+Judge is valid");
        assert!(matches!(
            parsed,
            CriterionStatus::Verified(Evidence::Judge { .. })
        ));

        // And the agent-writable claim states still parse as bare unit
        // variants — that's the agent's lane.
        for unit in ["NotStarted", "InProgress", "ClaimedDone"] {
            let json = format!("\"{unit}\"");
            let parsed: CriterionStatus =
                serde_json::from_str(&json).expect("unit variant is valid");
            assert!(!matches!(parsed, CriterionStatus::Verified(_)));
        }
    }

    #[test]
    fn schema_version_constant_is_what_we_publish() {
        assert_eq!(SCHEMA_VERSION, 1);
        // And the field on RunRecord defaults to this in our sample.
        assert_eq!(sample_run_record().schema_version, SCHEMA_VERSION);
    }

    // ---- copy/clone/derive smoke ----

    #[test]
    fn small_types_are_copy_and_clone() {
        // Compile-time-ish smoke: these types are `Copy`, so let-binding
        // through a value doesn't move it.
        let p = Phase::Done;
        let p2 = p;
        assert_eq!(p, p2);

        let m = FailureMode::TransientInfra;
        let m2 = m;
        assert_eq!(m, m2);

        let r = Role::Assistant;
        let r2 = r;
        assert_eq!(r, r2);

        let bc = BudgetConsumed {
            iterations: 1,
            tokens: 2,
            cost_micros: 3,
        };
        let bc2 = bc;
        assert_eq!(bc, bc2);

        let bl = BudgetLimits {
            iterations: 10,
            tokens: 20,
            cost_micros: 30,
        };
        let bl2 = bl;
        assert_eq!(bl, bl2);
    }

    #[test]
    fn record_clone_is_equal_to_source() {
        let r = sample_run_record();
        let cloned = r.clone();
        assert_eq!(r, cloned);
    }
}
