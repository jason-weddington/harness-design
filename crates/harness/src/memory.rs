//! The **memory seam**: how a knowledge base reaches into the engine loop.
//!
//! Phase 1 of `docs/design/11-memory.md`. A [`Memory`] implementation can do
//! three things to a run, and nothing else:
//!
//! - contribute a **session slice** — a short block of known gotchas that the
//!   engine appends to the rendered system prompt under
//!   [`MEMORY_SLICE_HEADING`] (and that therefore survives compaction, since
//!   the system prompt is never compacted);
//! - **advise** on each dispatched non-finish tool call via
//!   [`Memory::before_tool`] — either annotate the tool's result with a fact,
//!   or deny the call once (see [`Advice`]); the engine, not the adapter,
//!   decides what is actually applied (see [`MemoryMode`] and the
//!   crate-internal decision precedence documented on [`Memory`]);
//! - register extra **tools** (typically read-only KB tools) for the run.
//!
//! Memory is OFF by default ([`MemoryMode::Off`] with [`NoMemory`]), and an
//! Off run is byte-identical to a run on an engine without this seam.

use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;

use serde_json::Value;

use crate::tool::Tool;

/// The most calls a single loop invocation will deny. Once this many denies
/// have been applied, further `DenyOnce` advice is downgraded to an
/// annotation ([`DecisionReason::DenyCapReached`]).
pub const MAX_MEMORY_DENIES: u32 = 2;

/// The heading the engine places above the session slice when it appends the
/// slice to the system prompt.
pub const MEMORY_SLICE_HEADING: &str = "# Knowledge base memory";

/// The sentence that ends every deny message: the promise that an unchanged
/// retry of a denied call will be allowed. Starts with a space.
pub const MEMORY_DENY_RETRY_SUFFIX: &str =
    " If you still intend this exact call, retry it unchanged and it will be allowed.";

/// A knowledge-base memory consulted by the engine loop.
///
/// **Off by default.** [`crate::engine::RunConfig::new`] installs
/// [`NoMemory`] with [`MemoryMode::Off`], and in Off mode the engine never
/// calls ANY method of this trait — not even [`Memory::label`].
///
/// **No blocking, no I/O.** Every method is called synchronously on the async
/// executor, inside the engine loop. An implementation MUST NOT block or
/// perform I/O — no network, no `block_on` (a nested `block_on` inside the
/// loop panics). An adapter does every fetch before the run, typically in its
/// own async constructor, and reports a failed fetch through
/// [`Memory::degraded`].
///
/// **Call order.** When memory is on, the engine calls `degraded()` →
/// `label()` → `tools()` → `session_slice()`, each at most once per loop
/// invocation and all before the first model request; then `before_tool()`
/// once per dispatched non-finish tool call. A `Some` from `degraded()` forces
/// Off behaviour for the invocation: `tools()`, `session_slice()` and
/// `before_tool()` are then never called, and the engine prints one warning.
///
/// **Decision precedence.** Each [`Advice`] is resolved against the run's
/// [`MemoryMode`] (first match wins): Off → nothing; `Advice::None` →
/// nothing; a blank fact or entry id → suppressed; an unchanged retry of a
/// call denied earlier this invocation → annotate (the deny text promised it
/// would be allowed); `Annotate` → annotate; `DenyOnce` → annotate if the
/// mode is [`MemoryMode::Annotate`] (the mode is a ceiling), if the entry was
/// already denied once, or if [`MAX_MEMORY_DENIES`] is reached; otherwise
/// deny.
///
/// **Compaction.** Only the session slice is compaction-protected (it lives
/// in the system prompt). Annotations live in tool results and may be elided
/// like any other tool output.
///
/// **Scope.** Deny state, the deny cap and the memory counters are per loop
/// invocation, which equals per run for `talos run`.
///
/// **Spelling.** [`MemoryMode::as_str`] and its [`FromStr`] impl share the
/// CLI spelling: `off`, `annotate`, `deny-once`.
pub trait Memory: Send + Sync {
    /// The session slice to append to the system prompt, or `None` for none.
    /// A blank slice is treated as `None`.
    fn session_slice(&self) -> Option<String>;

    /// Advice for one dispatched non-finish tool call. Never called for a
    /// call named `finish`.
    fn before_tool(&self, call: &ToolCallView) -> Advice;

    /// Extra tools to register for the run. A tool whose name collides with
    /// an already-registered tool is dropped (the registered one is kept).
    fn tools(&self) -> Vec<Box<dyn Tool>> {
        Vec::new()
    }

    /// `Some(reason)` when the memory could not be loaded and the run must
    /// fall back to Off behaviour (fail open).
    fn degraded(&self) -> Option<String> {
        None
    }

    /// The telemetry label recorded as `memory_slice.source`. Must not carry
    /// secrets.
    fn label(&self) -> String {
        "custom".to_string()
    }

    /// The session id this memory reports to its backing store, recorded as
    /// `memory_slice.kb_session_id` when `Some`. Default `None` (no store).
    fn session_id(&self) -> Option<String> {
        None
    }
}

impl fmt::Debug for dyn Memory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("dyn Memory")
    }
}

/// A read-only view of a tool call handed to [`Memory::before_tool`].
#[derive(Debug, Clone, Copy)]
pub struct ToolCallView<'a> {
    /// The tool name the model called.
    pub name: &'a str,
    /// The call's JSON arguments, as the model sent them.
    pub args: &'a Value,
}

/// What a [`Memory`] advises for one tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Advice {
    /// No advice; the call runs exactly as without memory.
    None,
    /// Run the call and append the fact to its result.
    Annotate {
        /// The fact to show the model.
        fact: String,
        /// The KB entry the fact comes from.
        entry_id: String,
    },
    /// Refuse the call once, showing the fact; an unchanged retry is allowed.
    DenyOnce {
        /// The fact to show the model.
        fact: String,
        /// The KB entry the fact comes from.
        entry_id: String,
    },
}

impl Advice {
    /// The wire spelling: `none`, `annotate` or `deny_once`.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Annotate { .. } => "annotate",
            Self::DenyOnce { .. } => "deny_once",
        }
    }
}

/// The memory that knows nothing: no slice, no advice, no tools.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoMemory;

impl Memory for NoMemory {
    fn session_slice(&self) -> Option<String> {
        None
    }

    fn before_tool(&self, _call: &ToolCallView) -> Advice {
        Advice::None
    }

    fn label(&self) -> String {
        "none".to_string()
    }
}

/// How much of a [`Memory`]'s advice the engine may apply. The mode is a
/// ceiling: `Annotate` never denies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemoryMode {
    /// Memory is never consulted.
    #[default]
    Off,
    /// Slice, tools and annotations; `DenyOnce` advice is downgraded.
    Annotate,
    /// Everything, including denying a call once.
    DenyOnce,
}

impl MemoryMode {
    /// The CLI spelling: `off`, `annotate` or `deny-once` (hyphenated).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Annotate => "annotate",
            Self::DenyOnce => "deny-once",
        }
    }
}

impl FromStr for MemoryMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "off" => Ok(Self::Off),
            "annotate" => Ok(Self::Annotate),
            "deny-once" => Ok(Self::DenyOnce),
            _ => Err(format!(
                "unknown memory mode '{s}' (expected off, annotate or deny-once)"
            )),
        }
    }
}

/// What the engine actually applied for one piece of advice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppliedMode {
    /// The call ran and its result was annotated.
    Annotate,
    /// The call was not executed.
    DenyOnce,
    /// The advice was ignored; the call ran unannotated.
    Suppressed,
}

impl AppliedMode {
    /// The wire spelling: `annotate`, `deny_once` or `suppressed`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Annotate => "annotate",
            Self::DenyOnce => "deny_once",
            Self::Suppressed => "suppressed",
        }
    }
}

/// Why the engine applied what it applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionReason {
    /// Applied exactly as advised.
    AsAdvised,
    /// A deny was downgraded because the mode is `Annotate`.
    ModeCeiling,
    /// A deny was downgraded because its entry was already denied once.
    EntryAlreadyDenied,
    /// A deny was downgraded because [`MAX_MEMORY_DENIES`] was reached.
    DenyCapReached,
    /// The advice had a blank fact or entry id.
    BlankAdvice,
    /// The call is an unchanged retry of a denied call, so it is allowed.
    RetryOfDenied,
}

impl DecisionReason {
    /// The wire spelling, e.g. `as_advised` or `retry_of_denied`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AsAdvised => "as_advised",
            Self::ModeCeiling => "mode_ceiling",
            Self::EntryAlreadyDenied => "entry_already_denied",
            Self::DenyCapReached => "deny_cap_reached",
            Self::BlankAdvice => "blank_advice",
            Self::RetryOfDenied => "retry_of_denied",
        }
    }
}

/// The block appended to an annotated tool result.
#[must_use]
pub fn annotation_block(fact: &str, entry_id: &str) -> String {
    format!("[KB memory] {fact} (kb: {entry_id})")
}

/// The error content fed back for a denied call.
#[must_use]
pub fn deny_content(fact: &str, entry_id: &str) -> String {
    format!(
        "[KB memory] This call was not executed. {fact} (kb: {entry_id}){MEMORY_DENY_RETRY_SUFFIX}"
    )
}

/// Append a session slice to a rendered system prompt under
/// [`MEMORY_SLICE_HEADING`].
#[must_use]
pub fn append_session_slice(system: &str, slice: &str) -> String {
    format!(
        "{}\n\n{}\n\n{}",
        system.trim_end(),
        MEMORY_SLICE_HEADING,
        slice.trim()
    )
}

/// The number of items in a slice: lines whose `trim_start()` starts with
/// `- `.
#[must_use]
pub fn slice_item_count(slice: &str) -> usize {
    slice
        .lines()
        .filter(|line| line.trim_start().starts_with("- "))
        .count()
}

/// Per-loop-invocation deny bookkeeping.
#[derive(Debug, Default)]
pub(crate) struct MemoryRunState {
    pub(crate) denied_entry_ids: HashSet<String>,
    pub(crate) denied_calls: Vec<(String, Value)>,
    pub(crate) denies: u32,
}

/// The engine's resolution of one piece of advice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Decision {
    pub(crate) applied: AppliedMode,
    pub(crate) reason: DecisionReason,
}

/// Resolve `advice` for `call` under `mode` and the run's deny `state`.
/// `None` means there is nothing to apply or record.
pub(crate) fn decide(
    mode: MemoryMode,
    advice: &Advice,
    call: &ToolCallView,
    state: &MemoryRunState,
) -> Option<Decision> {
    let make = |applied, reason| Some(Decision { applied, reason });
    if mode == MemoryMode::Off {
        return None;
    }
    let (fact, entry_id, is_deny) = match advice {
        Advice::None => return None,
        Advice::Annotate { fact, entry_id } => (fact, entry_id, false),
        Advice::DenyOnce { fact, entry_id } => (fact, entry_id, true),
    };
    if fact.trim().is_empty() || entry_id.trim().is_empty() {
        return make(AppliedMode::Suppressed, DecisionReason::BlankAdvice);
    }
    if state
        .denied_calls
        .iter()
        .any(|(name, args)| name == call.name && args == call.args)
    {
        return make(AppliedMode::Annotate, DecisionReason::RetryOfDenied);
    }
    if !is_deny {
        return make(AppliedMode::Annotate, DecisionReason::AsAdvised);
    }
    if mode == MemoryMode::Annotate {
        make(AppliedMode::Annotate, DecisionReason::ModeCeiling)
    } else if state.denied_entry_ids.contains(entry_id) {
        make(AppliedMode::Annotate, DecisionReason::EntryAlreadyDenied)
    } else if state.denies >= MAX_MEMORY_DENIES {
        make(AppliedMode::Annotate, DecisionReason::DenyCapReached)
    } else {
        make(AppliedMode::DenyOnce, DecisionReason::AsAdvised)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn helpers_produce_exact_strings() {
        assert_eq!(
            annotation_block("use cargo nextest", "kb-1"),
            "[KB memory] use cargo nextest (kb: kb-1)"
        );
        assert_eq!(
            deny_content("use cargo nextest", "kb-1"),
            "[KB memory] This call was not executed. use cargo nextest (kb: kb-1) If you still intend this exact call, retry it unchanged and it will be allowed."
        );
        assert_eq!(
            append_session_slice("SYS\n", "\n- a\n"),
            "SYS\n\n# Knowledge base memory\n\n- a"
        );
        assert_eq!(slice_item_count("Header:\n- a [x; kb-1]\n- b [x; kb-2]"), 2);
        assert_eq!(slice_item_count(""), 0);
    }

    #[test]
    fn as_str_mappings() {
        let f = || ("f".to_string(), "e".to_string());
        assert_eq!(Advice::None.as_str(), "none");
        let (fact, entry_id) = f();
        assert_eq!(Advice::Annotate { fact, entry_id }.as_str(), "annotate");
        let (fact, entry_id) = f();
        assert_eq!(Advice::DenyOnce { fact, entry_id }.as_str(), "deny_once");
        assert_eq!(MemoryMode::Off.as_str(), "off");
        assert_eq!(MemoryMode::Annotate.as_str(), "annotate");
        assert_eq!(MemoryMode::DenyOnce.as_str(), "deny-once");
        assert_eq!(AppliedMode::Annotate.as_str(), "annotate");
        assert_eq!(AppliedMode::DenyOnce.as_str(), "deny_once");
        assert_eq!(AppliedMode::Suppressed.as_str(), "suppressed");
        assert_eq!(DecisionReason::AsAdvised.as_str(), "as_advised");
        assert_eq!(DecisionReason::ModeCeiling.as_str(), "mode_ceiling");
        assert_eq!(
            DecisionReason::EntryAlreadyDenied.as_str(),
            "entry_already_denied"
        );
        assert_eq!(DecisionReason::DenyCapReached.as_str(), "deny_cap_reached");
        assert_eq!(DecisionReason::BlankAdvice.as_str(), "blank_advice");
        assert_eq!(DecisionReason::RetryOfDenied.as_str(), "retry_of_denied");
    }

    #[test]
    fn memory_mode_round_trips_and_rejects_unknown() {
        for m in [MemoryMode::Off, MemoryMode::Annotate, MemoryMode::DenyOnce] {
            assert_eq!(m.as_str().parse::<MemoryMode>(), Ok(m));
        }
        assert_eq!(
            "deny_once".parse::<MemoryMode>(),
            Err(
                "unknown memory mode 'deny_once' (expected off, annotate or deny-once)".to_string()
            )
        );
        assert!("".parse::<MemoryMode>().is_err());
        assert_eq!(MemoryMode::default(), MemoryMode::Off);
    }

    #[test]
    fn no_memory_is_inert() {
        let args = json!({});
        let view = ToolCallView {
            name: "echo",
            args: &args,
        };
        let m = NoMemory;
        assert_eq!(m.session_slice(), None);
        assert_eq!(m.before_tool(&view), Advice::None);
        assert!(m.tools().is_empty());
        assert_eq!(m.degraded(), None);
        assert_eq!(m.label(), "none");
    }

    struct Bare;

    impl Memory for Bare {
        fn session_slice(&self) -> Option<String> {
            Some("x".into())
        }
        fn before_tool(&self, _call: &ToolCallView) -> Advice {
            Advice::None
        }
    }

    #[test]
    fn provided_defaults() {
        let b = Bare;
        assert_eq!(b.label(), "custom");
        assert!(b.tools().is_empty());
        assert_eq!(b.degraded(), None);
        assert_eq!(b.session_slice().as_deref(), Some("x"));
        let args = json!(null);
        assert_eq!(
            b.before_tool(&ToolCallView {
                name: "n",
                args: &args
            }),
            Advice::None
        );
    }

    #[test]
    fn dyn_memory_debug() {
        assert_eq!(format!("{:?}", &NoMemory as &dyn Memory), "dyn Memory");
    }

    fn deny(fact: &str, entry_id: &str) -> Advice {
        Advice::DenyOnce {
            fact: fact.into(),
            entry_id: entry_id.into(),
        }
    }

    fn annotate(fact: &str, entry_id: &str) -> Advice {
        Advice::Annotate {
            fact: fact.into(),
            entry_id: entry_id.into(),
        }
    }

    fn d(applied: AppliedMode, reason: DecisionReason) -> Decision {
        Decision { applied, reason }
    }

    #[test]
    fn decide_table() {
        use AppliedMode as A;
        use DecisionReason as R;
        use MemoryMode as M;

        let args = json!({"command":"cargo test"});
        let other = json!({"command":"cargo build"});
        let view = ToolCallView {
            name: "bash",
            args: &args,
        };
        let other_view = ToolCallView {
            name: "bash",
            args: &other,
        };
        let fresh = MemoryRunState::default();

        // R0
        assert_eq!(decide(M::Off, &deny("f", "kb-1"), &view, &fresh), None);
        // R1
        assert_eq!(decide(M::Annotate, &Advice::None, &view, &fresh), None);
        // R2
        assert_eq!(
            decide(M::Annotate, &annotate("  ", "kb-1"), &view, &fresh),
            Some(d(A::Suppressed, R::BlankAdvice))
        );
        assert_eq!(
            decide(M::DenyOnce, &deny("f", " "), &view, &fresh),
            Some(d(A::Suppressed, R::BlankAdvice))
        );
        // R3
        let retried = MemoryRunState {
            denied_calls: vec![("bash".into(), json!({"command":"cargo test"}))],
            ..MemoryRunState::default()
        };
        assert_eq!(
            decide(M::DenyOnce, &deny("f", "kb-2"), &view, &retried),
            Some(d(A::Annotate, R::RetryOfDenied))
        );
        // R4
        assert_eq!(
            decide(M::DenyOnce, &annotate("f", "kb-1"), &view, &fresh),
            Some(d(A::Annotate, R::AsAdvised))
        );
        // R5a
        assert_eq!(
            decide(M::Annotate, &deny("f", "kb-1"), &view, &fresh),
            Some(d(A::Annotate, R::ModeCeiling))
        );
        // R5b
        let denied_entry = MemoryRunState {
            denied_entry_ids: HashSet::from(["kb-1".to_string()]),
            ..MemoryRunState::default()
        };
        assert_eq!(
            decide(M::DenyOnce, &deny("f", "kb-1"), &other_view, &denied_entry),
            Some(d(A::Annotate, R::EntryAlreadyDenied))
        );
        // R5c
        let capped = MemoryRunState {
            denies: 2,
            ..MemoryRunState::default()
        };
        assert_eq!(
            decide(M::DenyOnce, &deny("f", "kb-7"), &view, &capped),
            Some(d(A::Annotate, R::DenyCapReached))
        );
        // R5d
        assert_eq!(
            decide(M::DenyOnce, &deny("f", "kb-1"), &view, &fresh),
            Some(d(A::DenyOnce, R::AsAdvised))
        );
    }
}
