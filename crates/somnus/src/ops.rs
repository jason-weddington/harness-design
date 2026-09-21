//! The op vocabulary, the op tools, the op-execution seam, and the
//! op→disposition mapping.
//!
//! Rung 2 of `reference/somnus-functional-spec.md`: the model emits ops from
//! a **closed vocabulary** and nothing else. The four externally-callable ops
//! are registered as tools whose schemas pin exactly that vocabulary;
//! `no_change` exists ONLY as a variant of [`Op`] — the model declines a
//! cluster by not acting, and code can represent the decline without giving
//! the model a tool for it.
//!
//! Op execution stays behind a [`MapOpSink`] seam. The op tools are
//! SCHEMA-ONLY in the somnus run path: rung-2 tool calls are parsed via
//! [`op_from_call`], never executed through a registry, so no agent can
//! invent an execution leg — ops are applied through the map-op client
//! ([`crate::map_op`]) after the gate, by code, in the pipeline's pinned
//! application step. The registered sink exists only to answer a stray
//! direct execution with a loud, named refusal.
//!
//! [`map_disposition`] maps the loop's inputs to the HARNESS
//! [`Disposition`] enum (not a somnus twin, so there is one disposition
//! vocabulary), and [`decision_record`] serializes every decision plus its
//! driving inputs — the op→disposition mapping happens outside the engine
//! and would otherwise leave zero post-hoc trace.

use std::sync::Arc;

use async_trait::async_trait;
use harness::exec::{ChangeEvidence, CheckReport};
use harness::run_record::{Disposition, FailureMode, Verification};
use harness::tool::{Tool, ToolCtx, ToolResult};
use serde::Serialize;
use serde_json::{Value, json};

/// The closed vocabulary of rung-2 ops (plus the decline variant). Field
/// names are IDENTICAL to the pinned tool schemas in
/// [`crate::gate::build_registry`], so the tool→Op mapping is mechanical.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    /// Point at an existing entry from a map.
    AddPointer {
        /// The map to point from.
        map_id: String,
        /// The entry to point at.
        entry_id: String,
        /// The pointer gloss (model-written prose).
        gloss: String,
    },
    /// Create a new map for a cluster.
    CreateMap {
        /// The cluster the map covers.
        cluster_id: String,
        /// The map title.
        title: String,
        /// The model-written orientation prose (the ONLY prose the model
        /// writes on a map).
        orientation_prose: String,
    },
    /// Strike a gap, citing the entry that closed it.
    StrikeGap {
        /// The map whose gap is struck.
        map_id: String,
        /// The gap text.
        gap_text: String,
        /// The entry that closed it (citation required).
        closing_entry_id: String,
    },
    /// Decline a cluster: it is real, but nothing chunky exists to point at.
    ProposeGap {
        /// The declined cluster.
        cluster_id: String,
        /// Why it needs an upstream authoring run.
        reason: String,
    },
    /// The model declines a cluster without an external action. NEVER a
    /// registered tool (see [`crate::gate::build_registry`]'s registry test).
    NoChange {
        /// The declined cluster.
        cluster_id: String,
    },
}

/// What the model claimed at the finish terminal, pre-parse into the harness
/// disposition mapping.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishClaim {
    /// The model claimed the run is complete.
    Done,
    /// The model claimed the task turned out to be already satisfied.
    AlreadySatisfied {
        /// What it checked and why.
        reason: String,
    },
}

/// The op-execution seam. Every op the model picks is applied through this
/// sink; the production transport is the KB's write endpoints, whose
/// contract is not pinned in this repo (the remaining gap this cut).
#[async_trait]
pub trait MapOpSink: Send + Sync {
    /// Apply one op.
    async fn apply(&self, op: Op) -> ToolResult;
}

/// The production [`MapOpSink`]: a loud, named refusal — the op tools are
/// SCHEMA-ONLY in the somnus run path (only `schema()` is ever called on
/// them), because ops are applied through the map-op client after the gate,
/// by code, never through a registry execution.
#[derive(Debug, Default, Clone, Copy)]
pub struct SchemaOnlySink;

pub(crate) const OP_APPLY_REFUSAL: &str = "somnus: op tools are schema-only in the somnus run path; ops are applied through the map-op client after the gate";

#[async_trait]
impl MapOpSink for SchemaOnlySink {
    async fn apply(&self, _op: Op) -> ToolResult {
        ToolResult::error(OP_APPLY_REFUSAL)
    }
}

/// One op tool: a closed-vocabulary op exposed to the model, bound to the
/// [`MapOpSink`] its picks flow into.
pub struct OpTool {
    kind: OpKind,
    sink: Arc<dyn MapOpSink>,
}

impl std::fmt::Debug for OpTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpTool")
            .field("kind", &self.kind)
            .finish_non_exhaustive()
    }
}

impl Clone for OpTool {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind,
            sink: Arc::clone(&self.sink),
        }
    }
}

/// The four externally-callable ops, each with a pinned name and schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpKind {
    AddPointer,
    CreateMap,
    StrikeGap,
    ProposeGap,
}

impl OpKind {
    /// The tool name the kind is registered and invoked under.
    fn name(self) -> &'static str {
        match self {
            Self::AddPointer => "add_pointer",
            Self::CreateMap => "create_map",
            Self::StrikeGap => "strike_gap",
            Self::ProposeGap => "propose_gap",
        }
    }

    /// Parse a tool name into its kind; `None` for anything outside the
    /// closed vocabulary.
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "add_pointer" => Some(Self::AddPointer),
            "create_map" => Some(Self::CreateMap),
            "strike_gap" => Some(Self::StrikeGap),
            "propose_gap" => Some(Self::ProposeGap),
            _ => None,
        }
    }
}

impl OpTool {
    /// Build the tool for `name` over `sink`. Panics only on a programmer
    /// error (a name outside the closed vocabulary), never on model input.
    ///
    /// # Panics
    /// Panics if `name` is not one of the four pinned op-tool names.
    pub fn new(name: &'static str, sink: Arc<dyn MapOpSink>) -> Self {
        let kind = OpKind::from_name(name).unwrap_or_else(|| {
            panic!("OpTool constructed with a name outside the closed vocabulary: {name}")
        });
        Self { kind, sink }
    }

    /// Parse a tool-call input into the corresponding [`Op`] by delegating
    /// the per-kind field extraction to [`parse_op_fields`] with
    /// `self.kind`.
    fn parse_op(&self, input: &Value) -> Result<Op, String> {
        parse_op_fields(self.kind, input)
    }
}

/// Read a required string field from a tool-call input.
fn req_str(input: &Value, key: &str) -> Result<String, String> {
    input
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("missing required string field `{key}`"))
}

/// The per-kind field extraction behind [`OpTool::parse_op`]: routing an
/// already-resolved [`OpKind`] over a raw tool-call input. Shared with the
/// free function the rung-2 loop parses model turns through
/// ([`op_from_call`]), so the tool's registered schema and the loop's parse
/// seam can never drift apart.
fn parse_op_fields(kind: OpKind, input: &Value) -> Result<Op, String> {
    match kind {
        OpKind::AddPointer => Ok(Op::AddPointer {
            map_id: req_str(input, "map_id")?,
            entry_id: req_str(input, "entry_id")?,
            gloss: req_str(input, "gloss")?,
        }),
        OpKind::CreateMap => Ok(Op::CreateMap {
            cluster_id: req_str(input, "cluster_id")?,
            title: req_str(input, "title")?,
            orientation_prose: req_str(input, "orientation_prose")?,
        }),
        OpKind::StrikeGap => Ok(Op::StrikeGap {
            map_id: req_str(input, "map_id")?,
            gap_text: req_str(input, "gap_text")?,
            closing_entry_id: req_str(input, "closing_entry_id")?,
        }),
        OpKind::ProposeGap => Ok(Op::ProposeGap {
            cluster_id: req_str(input, "cluster_id")?,
            reason: req_str(input, "reason")?,
        }),
    }
}

/// Parse a rung-2 tool call (`name`, `input`) into an [`Op`], routing the
/// NAME first: any string outside the four op names is a named `Err` before
/// any field is read and before any [`OpTool`] is constructed — `finish`,
/// `run_checks`, `no_change`, and anything else the model might reach for
/// are parse errors, never panics, never silently applied.
///
/// # Errors
/// `Err` naming the unknown tool, or naming the missing/ill-typed field.
pub fn op_from_call(name: &str, input: &Value) -> Result<Op, String> {
    let kind = OpKind::from_name(name).ok_or_else(|| {
        format!("unknown tool `{name}`: the rung-2 closed vocabulary is add_pointer, create_map, strike_gap, propose_gap")
    })?;
    parse_op_fields(kind, input)
}

/// The four op tool schemas, byte-identical to what
/// [`crate::gate::build_registry`] registers — built over the same
/// [`SchemaOnlySink`] and the same [`OpTool`] construction, so the rung-2
/// tool union and the registry can never drift apart.
#[must_use]
pub fn op_tool_schemas() -> Vec<Value> {
    let sink: Arc<dyn MapOpSink> = Arc::new(SchemaOnlySink);
    ["add_pointer", "create_map", "strike_gap", "propose_gap"]
        .into_iter()
        .map(|name| OpTool::new(name, Arc::clone(&sink)).schema())
        .collect()
}

#[async_trait]
impl Tool for OpTool {
    fn name(&self) -> &str {
        self.kind.name()
    }

    fn schema(&self) -> Value {
        let (description, properties, required): (&str, Value, &[&str]) = match self.kind {
            OpKind::AddPointer => (
                "Point at an existing entry from a map.",
                json!({
                    "map_id": { "type": "string" },
                    "entry_id": { "type": "string" },
                    "gloss": { "type": "string" }
                }),
                &["map_id", "entry_id", "gloss"],
            ),
            OpKind::CreateMap => (
                "Create a new map for a cluster.",
                json!({
                    "cluster_id": { "type": "string" },
                    "title": { "type": "string" },
                    "orientation_prose": { "type": "string" }
                }),
                &["cluster_id", "title", "orientation_prose"],
            ),
            OpKind::StrikeGap => (
                "Strike a gap, citing the entry that closed it.",
                json!({
                    "map_id": { "type": "string" },
                    "gap_text": { "type": "string" },
                    "closing_entry_id": { "type": "string" }
                }),
                &["map_id", "gap_text", "closing_entry_id"],
            ),
            OpKind::ProposeGap => (
                "Decline a cluster: it is real, but nothing chunky exists to point at and no amount of retrying fixes it.",
                json!({
                    "cluster_id": { "type": "string" },
                    "reason": { "type": "string" }
                }),
                &["cluster_id", "reason"],
            ),
        };
        json!({
            "name": self.kind.name(),
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        })
    }

    async fn run(&self, input: Value, _ctx: &ToolCtx) -> ToolResult {
        match self.parse_op(&input) {
            Ok(op) => self.sink.apply(op).await,
            // The op path is unwired this cut end to end; an unparsable input
            // surfaces the same named refusal so a stray call never looks
            // like it landed.
            Err(_) => ToolResult::error(OP_APPLY_REFUSAL),
        }
    }
}

/// The summary recorded on a legitimate `Done` disposition.
pub const SOMNUS_DONE_SUMMARY: &str = "ops applied, map-lint gate green, pointer count moved";

/// Map the loop's four inputs to the HARNESS disposition enum, in pinned
/// order:
///
/// 1. `Some(Op::ProposeGap { reason, .. })` →
///    [`Disposition::Blocked`] — regardless of claim/gate/change, and the
///    ONLY input that yields `Blocked`. The cluster is real, nothing chunky
///    exists to point at, and no amount of retrying fixes it because it
///    needs an upstream authoring run.
/// 2. `Some(FinishClaim::Done)` AND a passing gate AND
///    `ChangeEvidence::TreeChanged` → [`Disposition::Done`]. No code path
///    may fabricate `Done` without the gate and count legs.
/// 3. `Some(FinishClaim::AlreadySatisfied { .. })` AND a passing gate AND
///    `ChangeEvidence::TreeUnchanged` → [`Disposition::AlreadySatisfied`].
/// 4. EVERYTHING else → [`Disposition::Failed`] (`FailureMode::Loop`).
#[must_use]
pub fn map_disposition(
    op: Option<&Op>,
    claim: Option<&FinishClaim>,
    gate: Option<&CheckReport>,
    change: &ChangeEvidence,
) -> Disposition {
    if let Some(Op::ProposeGap { reason, .. }) = op {
        return Disposition::Blocked {
            decision_needed: reason.clone(),
        };
    }
    let gate_passing = gate.is_some_and(|report| report.passed);
    if gate_passing
        && claim == Some(&FinishClaim::Done)
        && *change == ChangeEvidence::TreeChanged
        && let Some(report) = gate
    {
        return Disposition::Done {
            summary: SOMNUS_DONE_SUMMARY.to_string(),
            verification: Verification::Checks(report.clone()),
            change: change.clone(),
        };
    }
    if gate_passing
        && let (Some(FinishClaim::AlreadySatisfied { reason }), Some(report)) = (claim, gate)
        && *change == ChangeEvidence::TreeUnchanged
    {
        return Disposition::AlreadySatisfied {
            reason: reason.clone(),
            verification: Verification::Checks(report.clone()),
            change: change.clone(),
        };
    }
    Disposition::Failed {
        mode: FailureMode::Loop,
        summary: format!(
            "somnus: no disposition matched (op={op:?}, claim={claim:?}, gate={gate:?}, change={change:?})"
        ),
    }
}

/// Every disposition decision plus its driving inputs, emittable as
/// structured data. `gate_passed` is the raw verdict (`gate.map(|r|
/// r.passed)`); the full report rides inside the disposition's verification.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MapDecisionRecord {
    /// The op the model picked, if one was picked.
    op: Option<Op>,
    /// The finish claim, if one was made.
    claim: Option<FinishClaim>,
    /// Whether the gate passed, if a gate report exists.
    gate_passed: Option<bool>,
    /// The leg-3 change evidence the decision was made against.
    change: ChangeEvidence,
    /// The mapped disposition, serialized with a `snake_case` variant key
    /// (see [`serialize_disposition_snake_case`]).
    #[serde(serialize_with = "serialize_disposition_snake_case")]
    disposition: Disposition,
}

/// Serialize a [`Disposition`] with its top-level variant key mapped from the
/// harness enum's `PascalCase` wire shape to the `snake_case` shape this
/// record's decision trace is pinned to (`done`, `already_satisfied`,
/// `blocked`, `failed`). The RUST type stays the harness enum — one disposition
/// vocabulary — only the emitted key is mapped; inner payloads pass through
/// untouched. An unrecognized key passes through verbatim, so a future
/// harness variant cannot break this silently.
fn serialize_disposition_snake_case<S: serde::Serializer>(
    disposition: &Disposition,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let raw = serde_json::to_value(disposition).map_err(serde::ser::Error::custom)?;
    let obj = raw
        .as_object()
        .expect("an externally-tagged enum serializes to an object");
    let mut out = serde_json::Map::new();
    for (key, value) in obj {
        out.insert(snake_case_variant(key).to_string(), value.clone());
    }
    serde_json::Value::Object(out).serialize(serializer)
}

/// The pinned `PascalCase` → `snake_case` variant-key map. An unrecognized
/// key passes through verbatim, so a future harness variant cannot break
/// this silently.
#[cfg_attr(not(test), allow(dead_code))]
fn snake_case_variant(key: &str) -> &str {
    match key {
        "Done" => "done",
        "AlreadySatisfied" => "already_satisfied",
        "Answer" => "answer",
        "Blocked" => "blocked",
        "Failed" => "failed",
        other => other,
    }
}

/// Build a [`MapDecisionRecord`] for one finish-time decision.
#[must_use]
pub fn decision_record(
    op: Option<&Op>,
    claim: Option<&FinishClaim>,
    gate: Option<&CheckReport>,
    change: &ChangeEvidence,
) -> MapDecisionRecord {
    MapDecisionRecord {
        op: op.cloned(),
        claim: claim.cloned(),
        gate_passed: gate.map(|report| report.passed),
        change: change.clone(),
        disposition: map_disposition(op, claim, gate, change),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// The green gate fixture: a passing, non-timed-out report.
    fn green_report() -> CheckReport {
        CheckReport {
            passed: true,
            exit_code: Some(0),
            timed_out: false,
            excerpt: String::new(),
            offload_path: None,
            duration: Duration::ZERO,
        }
    }

    /// The red gate fixture: `curl -f` mapped a 422 to a failed report.
    fn red_report() -> CheckReport {
        CheckReport {
            passed: false,
            exit_code: Some(22),
            timed_out: false,
            excerpt: String::new(),
            offload_path: None,
            duration: Duration::ZERO,
        }
    }

    // --- the Op enum: externally-tagged serde shape ---

    #[test]
    fn op_serializes_externally_tagged_with_snake_case_variants() {
        let value = serde_json::to_value(Op::ProposeGap {
            cluster_id: "c1".to_string(),
            reason: "r".to_string(),
        })
        .expect("serialize");
        assert_eq!(
            value,
            serde_json::json!({"propose_gap": {"cluster_id": "c1", "reason": "r"}})
        );
        let value = serde_json::to_value(Op::NoChange {
            cluster_id: "c1".to_string(),
        })
        .expect("serialize");
        assert_eq!(
            value,
            serde_json::json!({"no_change": {"cluster_id": "c1"}})
        );
    }

    // --- the op tools: the schema-only sink is the only registered sink ---

    #[tokio::test]
    async fn the_schema_only_sink_refuses_loudly_before_any_execution() {
        let result = SchemaOnlySink
            .apply(Op::AddPointer {
                map_id: "kb-20001".to_string(),
                entry_id: "kb-10001".to_string(),
                gloss: "g".to_string(),
            })
            .await;
        assert!(result.is_error);
        assert_eq!(
            result.summary,
            "somnus: op tools are schema-only in the somnus run path; ops are applied through the map-op client after the gate"
        );
        // The refusal names where application really happens, and never
        // claims a write landed.
        assert!(result.summary.contains("map-op client after the gate"));
    }

    #[tokio::test]
    async fn all_four_op_tools_refuse_through_the_registry_at_the_schema_only_sink() {
        let registry = crate::gate::build_registry(
            "http://kb.invalid",
            std::path::Path::new("/tmp/somnus-body.json"),
            std::path::Path::new("/tmp/kb-token"),
            std::sync::Arc::new(SchemaOnlySink),
        );
        for name in ["add_pointer", "create_map", "strike_gap", "propose_gap"] {
            let result = registry
                .invoke(name, serde_json::json!({}), &harness::tool::ToolCtx::stub())
                .await;
            assert!(result.is_error, "{name} must refuse");
            assert_eq!(
                result.summary,
                "somnus: op tools are schema-only in the somnus run path; ops are applied through the map-op client after the gate",
                "{name} must refuse through the schema-only sink: {}",
                result.summary
            );
        }
    }

    #[tokio::test]
    async fn op_tool_parses_valid_input_into_the_op_vocabulary() {
        let registry = crate::gate::build_registry(
            "http://kb.invalid",
            std::path::Path::new("/tmp/somnus-body.json"),
            std::path::Path::new("/tmp/somnus/kb-token"),
            std::sync::Arc::new(SchemaOnlySink),
        );
        let result = registry
            .invoke(
                "propose_gap",
                serde_json::json!({"cluster_id": "c1", "reason": "r"}),
                &harness::tool::ToolCtx::stub(),
            )
            .await;
        // Still an error (the sink is schema-only), but the op WAS parsed
        // and handed to the sink — the mapping is mechanical, not a schema
        // error.
        assert!(result.is_error);
        assert_eq!(
            result.summary,
            "somnus: op tools are schema-only in the somnus run path; ops are applied through the map-op client after the gate"
        );
    }

    // --- map_disposition: the pinned rows ---

    #[test]
    fn propose_gap_yields_blocked_regardless_of_claim_gate_and_change() {
        let op = Op::ProposeGap {
            cluster_id: "c1".to_string(),
            reason: "needs an authoring run".to_string(),
        };
        let disposition = map_disposition(
            Some(&op),
            Some(&FinishClaim::Done),
            Some(&green_report()),
            &ChangeEvidence::TreeChanged,
        );
        assert_eq!(
            disposition,
            Disposition::Blocked {
                decision_needed: "needs an authoring run".to_string(),
            }
        );
        // And Blocked even with NO gate and NO claim.
        let disposition = map_disposition(Some(&op), None, None, &ChangeEvidence::TreeUnchanged);
        assert_eq!(
            disposition,
            Disposition::Blocked {
                decision_needed: "needs an authoring run".to_string(),
            }
        );
    }

    #[test]
    fn done_claim_with_green_gate_and_changed_tree_yields_done() {
        let disposition = map_disposition(
            Some(&Op::AddPointer {
                map_id: "m1".to_string(),
                entry_id: "e1".to_string(),
                gloss: "g".to_string(),
            }),
            Some(&FinishClaim::Done),
            Some(&green_report()),
            &ChangeEvidence::TreeChanged,
        );
        let green = green_report();
        assert_eq!(
            disposition,
            Disposition::Done {
                summary: SOMNUS_DONE_SUMMARY.to_string(),
                verification: Verification::Checks(green),
                change: ChangeEvidence::TreeChanged,
            }
        );
    }

    #[test]
    fn done_claim_with_red_gate_yields_failed() {
        let disposition = map_disposition(
            None,
            Some(&FinishClaim::Done),
            Some(&red_report()),
            &ChangeEvidence::TreeChanged,
        );
        assert!(
            summary_starts_with_no_disposition_matched(&disposition),
            "expected Failed, got {disposition:?}"
        );
    }

    #[test]
    fn done_claim_with_green_gate_but_unchanged_tree_yields_failed() {
        let disposition = map_disposition(
            None,
            Some(&FinishClaim::Done),
            Some(&green_report()),
            &ChangeEvidence::TreeUnchanged,
        );
        assert!(matches!(disposition, Disposition::Failed { .. }));
    }

    #[test]
    fn already_satisfied_with_green_gate_and_unchanged_tree_yields_already_satisfied() {
        let green = green_report();
        let disposition = map_disposition(
            None,
            Some(&FinishClaim::AlreadySatisfied {
                reason: "map already current".to_string(),
            }),
            Some(&green_report()),
            &ChangeEvidence::TreeUnchanged,
        );
        assert_eq!(
            disposition,
            Disposition::AlreadySatisfied {
                reason: "map already current".to_string(),
                verification: Verification::Checks(green),
                change: ChangeEvidence::TreeUnchanged,
            }
        );
    }

    #[test]
    fn no_claim_with_no_op_yields_failed() {
        let disposition = map_disposition(
            None,
            None,
            Some(&green_report()),
            &ChangeEvidence::TreeChanged,
        );
        assert!(
            matches!(disposition, Disposition::Failed { mode: FailureMode::Loop, .. }
                if summary_starts_with_no_disposition_matched(&disposition)),
            "expected the everything-else Failed row, got {disposition:?}"
        );
    }

    /// Assert the summary of a `Failed` disposition starts with the pinned
    /// prefix (a helper so the tests never carry an uncovered panic arm).
    fn summary_starts_with_no_disposition_matched(disposition: &Disposition) -> bool {
        matches!(disposition, Disposition::Failed { summary, .. }
            if summary.starts_with("somnus: no disposition matched"))
    }

    #[test]
    fn no_change_with_a_done_claim_yields_failed_everything_else_row() {
        let op = Op::NoChange {
            cluster_id: "c1".to_string(),
        };
        let disposition = map_disposition(
            Some(&op),
            Some(&FinishClaim::Done),
            Some(&green_report()),
            &ChangeEvidence::TreeUnchanged,
        );
        assert!(
            summary_starts_with_no_disposition_matched(&disposition),
            "expected Failed, got {disposition:?}"
        );
    }

    // --- MapDecisionRecord: the serialized decision trace ---

    #[test]
    fn decision_record_serializes_the_full_decision_trace() {
        let op = Op::ProposeGap {
            cluster_id: "c1".to_string(),
            reason: "needs an authoring run".to_string(),
        };
        let record = decision_record(Some(&op), None, None, &ChangeEvidence::TreeUnchanged);
        let value = serde_json::to_value(&record).expect("serialize");
        assert_eq!(
            value["op"]["propose_gap"]["reason"],
            "needs an authoring run"
        );
        assert!(value["claim"].is_null());
        assert!(value["gate_passed"].is_null());
        assert_eq!(value["change"], "TreeUnchanged");
        assert_eq!(
            value["disposition"]["blocked"]["decision_needed"],
            "needs an authoring run"
        );
    }

    #[test]
    fn decision_record_records_the_gate_verdict() {
        let record = decision_record(
            None,
            Some(&FinishClaim::Done),
            Some(&red_report()),
            &ChangeEvidence::TreeChanged,
        );
        assert_eq!(record.gate_passed, Some(false));
        let value = serde_json::to_value(&record).expect("serialize");
        assert_eq!(value["gate_passed"], false);
        assert_eq!(value["claim"], "done");
    }

    // --- OpTool plumbing: Debug, Clone, and the mechanical input mapping ---

    #[test]
    fn op_tool_debug_and_clone_are_wired() {
        let tool = OpTool::new("add_pointer", Arc::new(SchemaOnlySink));
        let rendered = format!("{tool:?}");
        assert!(rendered.contains("OpTool"), "{rendered}");
        let cloned = tool.clone();
        assert_eq!(cloned.name(), "add_pointer");
        assert_eq!(cloned.kind, tool.kind);
    }

    #[tokio::test]
    async fn each_op_tool_maps_its_input_to_the_matching_op() {
        // The mechanical mapping, tool name → kind → Op: parse the input
        // through the same deserialization the tool's run path uses and
        // assert the expected op fields came out the other side.
        let scripts: [(&'static str, serde_json::Value, Op); 4] = [
            (
                "add_pointer",
                serde_json::json!({"map_id": "m1", "entry_id": "e1", "gloss": "g"}),
                Op::AddPointer {
                    map_id: "m1".to_string(),
                    entry_id: "e1".to_string(),
                    gloss: "g".to_string(),
                },
            ),
            (
                "create_map",
                serde_json::json!({"cluster_id": "c1", "title": "t", "orientation_prose": "p"}),
                Op::CreateMap {
                    cluster_id: "c1".to_string(),
                    title: "t".to_string(),
                    orientation_prose: "p".to_string(),
                },
            ),
            (
                "strike_gap",
                serde_json::json!({"map_id": "m1", "gap_text": "g", "closing_entry_id": "e1"}),
                Op::StrikeGap {
                    map_id: "m1".to_string(),
                    gap_text: "g".to_string(),
                    closing_entry_id: "e1".to_string(),
                },
            ),
            (
                "propose_gap",
                serde_json::json!({"cluster_id": "c1", "reason": "r"}),
                Op::ProposeGap {
                    cluster_id: "c1".to_string(),
                    reason: "r".to_string(),
                },
            ),
        ];
        for (name, input, expected_op) in scripts {
            let tool = OpTool::new(name, Arc::new(SchemaOnlySink));
            let parsed = tool.parse_op(&input).expect("valid input parses");
            assert_eq!(parsed, expected_op, "{name} mapping must be mechanical");
        }
    }

    #[test]
    fn snake_case_variant_maps_every_harness_variant() {
        assert_eq!(snake_case_variant("Done"), "done");
        assert_eq!(snake_case_variant("AlreadySatisfied"), "already_satisfied");
        assert_eq!(snake_case_variant("Answer"), "answer");
        assert_eq!(snake_case_variant("Blocked"), "blocked");
        assert_eq!(snake_case_variant("Failed"), "failed");
        // Unrecognized keys pass through verbatim.
        assert_eq!(snake_case_variant("SomeFutureVariant"), "SomeFutureVariant");
    }

    #[test]
    fn decision_record_serializes_done_and_already_satisfied_keys() {
        let record = decision_record(
            None,
            Some(&FinishClaim::Done),
            Some(&green_report()),
            &ChangeEvidence::TreeChanged,
        );
        let value = serde_json::to_value(&record).expect("serialize");
        assert_eq!(value["disposition"]["done"]["summary"], SOMNUS_DONE_SUMMARY);
        assert_eq!(value["claim"], "done");

        let record = decision_record(
            None,
            Some(&FinishClaim::AlreadySatisfied {
                reason: "map already current".to_string(),
            }),
            Some(&green_report()),
            &ChangeEvidence::TreeUnchanged,
        );
        let value = serde_json::to_value(&record).expect("serialize");
        assert_eq!(
            value["disposition"]["already_satisfied"]["reason"],
            "map already current"
        );
    }

    #[test]
    fn req_str_rejects_a_missing_or_non_string_field() {
        let tool = OpTool::new("add_pointer", Arc::new(SchemaOnlySink));
        let error = tool
            .parse_op(&serde_json::json!({"map_id": "m1", "entry_id": 3}))
            .expect_err("non-string field is rejected");
        assert_eq!(error, "missing required string field `entry_id`");
        let error = tool
            .parse_op(&serde_json::json!({}))
            .expect_err("absent field is rejected");
        assert_eq!(error, "missing required string field `map_id`");
    }

    // --- op_from_call: the rung-2 parse seam, names routed FIRST ----------

    #[test]
    fn op_from_call_routes_the_name_before_any_field_is_read() {
        // Every one of the four op names parses through the same per-kind
        // extraction the registered tool uses.
        let op = op_from_call(
            "add_pointer",
            &serde_json::json!({"map_id": "m1", "entry_id": "e1", "gloss": "g"}),
        )
        .expect("a known op name with valid fields parses");
        assert_eq!(
            op,
            Op::AddPointer {
                map_id: "m1".to_string(),
                entry_id: "e1".to_string(),
                gloss: "g".to_string(),
            }
        );
        let op = op_from_call(
            "create_map",
            &serde_json::json!({"cluster_id": "c1", "title": "t", "orientation_prose": "p"}),
        )
        .expect("create_map parses");
        assert_eq!(
            op,
            Op::CreateMap {
                cluster_id: "c1".to_string(),
                title: "t".to_string(),
                orientation_prose: "p".to_string(),
            }
        );
        // A missing field inside a KNOWN op is a named field error.
        let error = op_from_call("strike_gap", &serde_json::json!({"map_id": "m1"}))
            .expect_err("a missing field is a named error");
        assert_eq!(error, "missing required string field `gap_text`");
    }

    #[test]
    fn op_from_call_refuses_every_name_outside_the_closed_vocabulary() {
        // `finish`, `run_checks`, `no_change`, and anything else the model
        // might reach for: a named Err, never a panic, never an op.
        for name in ["finish", "run_checks", "no_change", "bogus"] {
            let error = op_from_call(name, &serde_json::json!({}))
                .expect_err("every non-op tool name must be refused");
            assert!(
                error.contains("unknown tool"),
                "{name} must be named as an unknown tool, got {error}"
            );
            assert!(error.contains(name), "the error must name {name}: {error}");
        }
    }

    // --- op_tool_schemas: the rung-2 tool union ---------------------------

    #[test]
    fn op_tool_schemas_are_exactly_the_four_registered_op_schemas() {
        let schemas = op_tool_schemas();
        assert_eq!(schemas.len(), 4);
        let registry = crate::gate::build_registry(
            "http://kb.invalid",
            std::path::Path::new("/tmp/somnus-body.json"),
            std::path::Path::new("/tmp/somnus/kb-token"),
            std::sync::Arc::new(SchemaOnlySink),
        );
        let mut names: Vec<String> = schemas
            .iter()
            .map(|schema| schema["name"].as_str().expect("a name").to_string())
            .collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "add_pointer".to_string(),
                "create_map".to_string(),
                "propose_gap".to_string(),
                "strike_gap".to_string(),
            ]
        );
        for schema in &schemas {
            let name = schema["name"].as_str().expect("a name");
            let tool = registry.get(name).expect("registered in the registry");
            assert_eq!(tool.schema(), *schema, "{name} must be byte-identical");
        }
    }
}
