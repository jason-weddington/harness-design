//! The tool abstraction: the [`Tool`] trait, its [`ToolResult`] output
//! contract, the [`ToolRegistry`] that maps names to tools, and the
//! [`ToolCtx`] that carries per-run plumbing (the offload sink) into a tool's
//! `run`.
//!
//! This is the *shape* only — no concrete tools (`read_file`, `run_command`,
//! …) live here; those land in later items. The pieces here establish the
//! contract every tool obeys:
//!
//! - **Signal over firehose.** A [`ToolResult`]'s `summary` is always small and
//!   its inline `detail` is bounded to [`DETAIL_CAP`] characters. The full
//!   output is offloaded through a [`ToolCtx`]-provided sink and its path
//!   advertised, so trimming the inline copy stays safe — the agent can read
//!   the offload path for the rest.
//! - **The registry is the boundary.** The agent can only invoke what's been
//!   registered; an unregistered name yields a structured `is_error`
//!   [`ToolResult`], never a panic.
//! - **Deterministic ordering.** The registry is a [`BTreeMap`] so `list()`
//!   output is stable across runs (byte-exact prompt-cache friendliness).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Inline-`detail` size cap, in characters (~25K). Anything longer is
/// truncated in place and the full copy is offloaded via the [`ToolCtx`] sink.
pub const DETAIL_CAP: usize = 25_000;

/// Sink that persists a tool's full output to a run-scoped location and returns
/// the path to advertise back to the agent.
///
/// This is the seam to the run-scoped working directory. The real
/// disk-writing implementation lands with the run-record work; v1 ships only
/// [`StubOffloadSink`] so the [`ToolResult`] cap logic can be wired and tested
/// without a filesystem.
pub trait OffloadSink: Send + Sync {
    /// Persist `contents` somewhere durable and return the path it now lives
    /// at. Implementations must be infallible from the caller's perspective —
    /// offloading is a best-effort safety net, not a failure surface.
    fn offload(&self, contents: &str) -> PathBuf;
}

/// A no-op [`OffloadSink`] for tests and pre-wiring: it never touches disk and
/// hands back a deterministic placeholder path. Real disk wiring is a later
/// item; this keeps the cap logic exercisable today.
#[derive(Debug, Default, Clone, Copy)]
pub struct StubOffloadSink;

impl OffloadSink for StubOffloadSink {
    fn offload(&self, _contents: &str) -> PathBuf {
        PathBuf::from("<offload-stub>")
    }
}

/// Per-run context handed to every [`Tool::run`] call.
///
/// v1 carries only the offload sink; future items thread the run-scoped working
/// directory, project config, and budgets through here.
#[derive(Clone)]
pub struct ToolCtx {
    sink: Arc<dyn OffloadSink>,
}

impl ToolCtx {
    /// Build a context backed by the given offload sink.
    #[must_use]
    pub fn new(sink: Arc<dyn OffloadSink>) -> Self {
        Self { sink }
    }

    /// Build a context backed by the [`StubOffloadSink`] — the default for
    /// tests and the not-yet-wired loop.
    #[must_use]
    pub fn stub() -> Self {
        Self::new(Arc::new(StubOffloadSink))
    }

    /// Offload `contents` through the configured sink, returning the path.
    #[must_use]
    pub fn offload(&self, contents: &str) -> PathBuf {
        self.sink.offload(contents)
    }
}

/// The result of running a [`Tool`].
///
/// `summary` is always small and goes straight into the model context.
/// `detail` is an optional, bounded ([`DETAIL_CAP`]) inline elaboration.
/// `offload_path` points at the full, untruncated output when it was offloaded.
/// `is_error` flags a steering error (a tool failure the agent should react
/// to) versus a successful call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResult {
    /// Always-small headline that lands in the model context.
    pub summary: String,
    /// Optional inline elaboration, bounded to [`DETAIL_CAP`] characters.
    pub detail: Option<String>,
    /// Path to the full, untruncated output when it was offloaded to disk.
    pub offload_path: Option<PathBuf>,
    /// Whether this result represents a (steering) error rather than success.
    pub is_error: bool,
}

impl ToolResult {
    /// A successful result with just a summary.
    pub fn ok(summary: impl Into<String>) -> Self {
        Self {
            summary: summary.into(),
            detail: None,
            offload_path: None,
            is_error: false,
        }
    }

    /// An error result with just a summary. Errors are a *steering* surface —
    /// actionable messages the agent reacts to, not loop-crashing exceptions.
    pub fn error(summary: impl Into<String>) -> Self {
        Self {
            summary: summary.into(),
            detail: None,
            offload_path: None,
            is_error: true,
        }
    }

    /// Build a successful result whose `detail` is bounded to [`DETAIL_CAP`]
    /// characters.
    ///
    /// If `detail` fits within the cap it passes through unchanged and nothing
    /// is offloaded. If it exceeds the cap, the full text is offloaded through
    /// the `ctx` sink, the inline copy is truncated to the cap and flagged with
    /// a pointer to the offload path, and `offload_path` is recorded.
    pub fn with_detail(
        summary: impl Into<String>,
        detail: impl Into<String>,
        ctx: &ToolCtx,
    ) -> Self {
        let summary = summary.into();
        let detail = detail.into();

        if detail.chars().count() <= DETAIL_CAP {
            return Self {
                summary,
                detail: Some(detail),
                offload_path: None,
                is_error: false,
            };
        }

        let offload_path = ctx.offload(&detail);
        let truncated: String = detail.chars().take(DETAIL_CAP).collect();
        let marked = format!(
            "{truncated}\n…[truncated at {DETAIL_CAP} chars; full output at {}]",
            offload_path.display()
        );
        Self {
            summary,
            detail: Some(marked),
            offload_path: Some(offload_path),
            is_error: false,
        }
    }
}

/// A capability the agent can invoke.
///
/// **Dyn-compatibility:** the agent holds tools as `Arc<dyn Tool>` in the
/// [`ToolRegistry`], so the trait must be object-safe. A bare `async fn` in a
/// trait is not yet dyn-compatible on this toolchain, so we use the
/// [`mod@async_trait`] crate, which desugars `async fn run` into a method
/// returning a boxed future — keeping `dyn Tool` usable. (The alternative,
/// hand-writing `-> Pin<Box<dyn Future>>`, is the same desugaring spelled out;
/// the macro is chosen for readability.)
#[async_trait]
pub trait Tool: Send + Sync {
    /// The tool's stable name — the key it's registered and invoked under.
    fn name(&self) -> &str;

    /// The tool's JSON input schema, as advertised to the model.
    fn schema(&self) -> Value;

    /// Execute the tool against `input`, using `ctx` for run-scoped plumbing.
    async fn run(&self, input: Value, ctx: &ToolCtx) -> ToolResult;
}

/// Maps tool names to tools. The registered set *is* the agent's capability
/// boundary — there is no separate permission layer. Backed by a [`BTreeMap`]
/// so `list()` is deterministically ordered.
#[derive(Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `tool` under `name`, replacing any existing tool with that
    /// name.
    pub fn register(&mut self, name: impl Into<String>, tool: Arc<dyn Tool>) {
        self.tools.insert(name.into(), tool);
    }

    /// Fetch a registered tool by name, if any.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// The schemas of all registered tools, in deterministic name order.
    #[must_use]
    pub fn list(&self) -> Vec<Value> {
        self.tools.values().map(|tool| tool.schema()).collect()
    }

    /// Invoke a registered tool by name.
    ///
    /// Invoking an unregistered name is *not* a panic — it returns a structured
    /// `is_error` [`ToolResult`], because an unknown tool is a steering signal
    /// the agent should see and recover from, not a crash.
    pub async fn invoke(&self, name: &str, input: Value, ctx: &ToolCtx) -> ToolResult {
        match self.get(name) {
            Some(tool) => tool.run(input, ctx).await,
            None => ToolResult::error(format!(
                "unknown tool `{name}`: not registered (registered tools are the only callable surface)"
            )),
        }
    }
}

/// A trivial tool that echoes its input back as the result summary. Exists to
/// exercise the [`Tool`] trait and [`ToolRegistry`] in tests; not a real v1
/// tool.
#[derive(Debug, Default, Clone, Copy)]
pub struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    // The trait fixes the return type as `&str`; returning a `&'static str`
    // here would diverge from the trait signature, so the lint doesn't apply.
    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "echo"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "echo",
            "description": "Returns its JSON input unchanged as the result summary.",
            "input_schema": { "type": "object" },
        })
    }

    async fn run(&self, input: Value, _ctx: &ToolCtx) -> ToolResult {
        ToolResult::ok(input.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DETAIL_CAP, EchoTool, OffloadSink, StubOffloadSink, Tool, ToolCtx, ToolRegistry, ToolResult,
    };
    use serde_json::json;
    use std::sync::Arc;

    #[test]
    fn ok_and_error_constructors_set_flags() {
        let ok = ToolResult::ok("done");
        assert!(!ok.is_error);
        assert_eq!(ok.summary, "done");
        assert!(ok.detail.is_none());
        assert!(ok.offload_path.is_none());

        let err = ToolResult::error("boom");
        assert!(err.is_error);
        assert_eq!(err.summary, "boom");
    }

    #[test]
    fn detail_under_cap_passes_through() {
        let ctx = ToolCtx::stub();
        let small = "a".repeat(DETAIL_CAP); // exactly at the cap is still inline
        let result = ToolResult::with_detail("sum", small.clone(), &ctx);

        assert_eq!(result.detail.as_deref(), Some(small.as_str()));
        assert!(result.offload_path.is_none());
        assert!(!result.is_error);
    }

    #[test]
    fn detail_over_cap_truncates_and_flags() {
        let ctx = ToolCtx::stub();
        let big = "b".repeat(DETAIL_CAP + 500);
        let result = ToolResult::with_detail("sum", big, &ctx);

        let detail = result.detail.expect("detail present");
        assert!(
            detail.contains("truncated"),
            "detail should be flagged truncated"
        );
        assert!(
            detail.starts_with(&"b".repeat(DETAIL_CAP)),
            "inline copy keeps the first {DETAIL_CAP} chars"
        );
        // The retained content (before the marker) is bounded to the cap.
        let kept = detail.split('\n').next().unwrap();
        assert_eq!(kept.chars().count(), DETAIL_CAP);
        assert_eq!(result.offload_path, Some("<offload-stub>".into()));
    }

    #[test]
    fn stub_sink_returns_placeholder_path() {
        let sink = StubOffloadSink;
        assert_eq!(
            sink.offload("anything"),
            std::path::PathBuf::from("<offload-stub>")
        );
        // Via a ctx built from a custom Arc sink too.
        let ctx = ToolCtx::new(Arc::new(StubOffloadSink));
        assert_eq!(ctx.offload("x"), std::path::PathBuf::from("<offload-stub>"));
    }

    #[test]
    fn result_round_trips_through_serde() {
        let result = ToolResult::error("nope");
        let text = serde_json::to_string(&result).expect("serialize");
        let back: ToolResult = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(result, back);
    }

    #[test]
    fn schema_is_parseable_json_object() {
        let schema = EchoTool.schema();
        assert!(schema.is_object(), "schema must be a JSON object");
        // Re-parse the serialized form to confirm it is valid JSON.
        let text = serde_json::to_string(&schema).expect("serialize schema");
        let reparsed: serde_json::Value = serde_json::from_str(&text).expect("reparse schema");
        assert_eq!(reparsed["name"], "echo");
    }

    #[tokio::test]
    async fn echo_tool_returns_input_as_summary() {
        let ctx = ToolCtx::stub();
        let out = EchoTool.run(json!({"hello": "world"}), &ctx).await;
        assert!(!out.is_error);
        assert_eq!(out.summary, json!({"hello": "world"}).to_string());
    }

    #[tokio::test]
    async fn registry_register_get_list_and_invoke() {
        let mut registry = ToolRegistry::new();
        registry.register("echo", Arc::new(EchoTool));

        // get returns the registered tool.
        assert!(registry.get("echo").is_some());

        // list returns each tool's schema.
        let schemas = registry.list();
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0]["name"], "echo");

        // invoke routes to the tool.
        let ctx = ToolCtx::stub();
        let out = registry.invoke("echo", json!({"k": 1}), &ctx).await;
        assert!(!out.is_error);
        assert_eq!(out.summary, json!({"k": 1}).to_string());
    }

    #[tokio::test]
    async fn unknown_tool_returns_structured_error_not_panic() {
        let registry = ToolRegistry::new();
        let ctx = ToolCtx::stub();
        let out = registry.invoke("does_not_exist", json!(null), &ctx).await;

        assert!(out.is_error);
        assert!(out.summary.contains("does_not_exist"));
        assert!(registry.get("does_not_exist").is_none());
    }

    #[test]
    fn register_replaces_existing_name() {
        let mut registry = ToolRegistry::new();
        registry.register("echo", Arc::new(EchoTool));
        registry.register("echo", Arc::new(EchoTool));
        assert_eq!(registry.list().len(), 1);
    }
}
