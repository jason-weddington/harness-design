//! The agent loop: drive a [`model::ModelBackend`] and a [`ToolRegistry`]
//! through a conversation until the agent calls the `finish` tool with a claim
//! the harness accepts, or a hard iteration cap is hit.
//!
//! ## Claim vs. verify — the load-bearing invariant
//!
//! `finish(done)` is a **claim** the model makes; the harness verifies it
//! mechanically before honoring it. When the run has [`ChecksRunner`] wired in,
//! a `finish(done)` triggers the harness to re-run those exact checks itself
//! — and the disposition is only accepted if they come back green. A red
//! verification is **steering, not termination**: the fed-back tool result is
//! `is_error=true`, the loop continues, and a subsequent turn can react
//! (typically by fixing whatever failed and finishing again). The [`Done`]
//! variant is **constructed by the loop only**, and only alongside the
//! [`Verification`] evidence that justifies it — the two are unified in a
//! single struct so a `Done` value in the outcome always carries proof.
//!
//! ## Leg 3 — work demonstrably happened
//!
//! Green checks are not enough: on a repo whose gate was *already* green, a
//! run that touched nothing could otherwise claim a fully "verified" `Done`.
//! So the completion contract has three legs — **the agent claimed**, **the
//! checks verified**, and **work demonstrably happened**. The loop observes
//! the workspace once at start ([`crate::exec::observe_tree`]) and again when
//! a `done` claim passes verification, and
//! [`crate::exec::classify_change`] turns the pair into
//! [`ChangeEvidence`]. An unchanged tree rejects the claim back to the model
//! exactly like a red gate; an *unobservable* tree (a non-git workspace, a
//! missing `git`, a timed-out status call) fails **open** with a loud stderr
//! warning, because under-enforcing beats rejecting honest work.
//!
//! `finish(already_satisfied)` is the deliberate off-ramp for a task that
//! turned out to need no change: it requires a non-empty `reason`, is still
//! held to the configured checks, imposes no tree constraint, and terminates
//! with [`Disposition::AlreadySatisfied`] — which is never pushable.
//!
//! `finish(blocked)` and `finish(failed)` are **not** verified — the model
//! declaring defeat needs no proof; those still terminate the loop as the
//! declaration states.
//!
//! A `finish` call whose `disposition` is missing, non-string, or not one of
//! `done`/`blocked`/`failed`/`already_satisfied` after trimming and
//! ASCII-lowercasing is rejected: it is fed back as an `is_error=true` tool
//! result and never terminates the loop. So is an `already_satisfied` claim
//! with no `reason`.
//!
//! ## Answer mode — a second deliverable shape
//!
//! When [`RunConfig::answer_schema`] is set, `finish` additionally accepts
//! `answer`: the run's deliverable is a **schema-validated payload** rather
//! than a changed workspace. The same three legs apply — the agent claims
//! (`finish(answer)`), something mechanical verifies (the
//! [`AnswerSchema`]), and the deliverable demonstrably exists (the validated
//! `result` itself). A missing or schema-invalid `result` is fed back as an
//! `is_error=true` tool result exactly like a red gate, and the loop
//! continues. With NO schema configured, `answer` is not advertised and is
//! rejected as an unrecognized disposition — build mode is byte-identical to
//! what it was.
//!
//! ## What lives here
//!
//! - [`RunConfig`] — the shape a caller hands to [`run`]: task, iteration cap,
//!   optional [`ChecksRunner`], per-turn output cap.
//! - [`run`] — the loop itself, generic over any [`model::ModelBackend`].
//! - [`LoopOutcome`] — the four ways the loop can end.
//! - [`FinishTool`] — the tool the model calls to end the run. Its schema is
//!   what the model sees; the loop is what parses the input and (for `done`)
//!   verifies it. The `answer_mode` flag gates whether the `answer`
//!   disposition and its `result` property appear in that schema at all; with
//!   it `false` (the default) the schema is byte-identical to build mode's.
//! - [`LoopOutcome::into_disposition`] — converts a terminal outcome to a
//!   [`crate::run_record::Disposition`] for storage.
//!
//! What does **not** live here yet (tracked separately): token / cost budget
//! enforcement, loop / no-progress detection, and context assembly /
//! compaction. Wall-clock budget enforcement and persistence / checkpointing
//! are implemented. The hard `max_iterations` cap and wall-clock cap are the
//! two non-finish stopping conditions.
//!
//! ## Loop shape
//!
//! 1. Render the system prompt via [`prompt::render_system_prompt`] and the
//!    task seed via [`prompt::render_task_prompt`]. Both are computed **once**
//!    before the loop and reused verbatim on every iteration — the
//!    prompt-cache correctness invariant.
//! 2. Each iteration: build a [`TurnRequest`] and call
//!    [`model::ModelBackend::turn`].
//! 3. Append the assistant turn to history (via `From<AssistantTurn>`).
//! 4. If the turn made no tool calls, stop ([`LoopOutcome::StoppedWithoutFinish`]).
//! 5. Otherwise execute each call in order, collecting fed-back results into a
//!    single [`Message::User`]. When an executed call is `finish`, the loop
//!    verifies (or accepts) the claim per the invariant above.
//! 6. If the cap is reached before the model reaches an accepted finish, stop
//!    ([`LoopOutcome::MaxIterations`]).
//!
//! A retryable backend error ([`model::BackendError::Transient`]) is retried
//! up to [`RunConfig::max_retries`] additional times with deterministic
//! exponential backoff before surfacing [`LoopOutcome::BackendError`]. A
//! non-retryable error ([`model::BackendError::Terminal`],
//! [`model::BackendError::Protocol`],
//! [`model::BackendError::ContextLengthExceeded`]) is surfaced on first
//! occurrence.
//!
//! [`ChecksRunner`]: crate::exec::ChecksRunner
//! [`TurnRequest`]: crate::model::TurnRequest

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::time::sleep;

use crate::exec::{
    self, ChangeEvidence, CheckReport, ChecksRunner, TREE_OBSERVE_TIMEOUT, TreeObservation,
};
use crate::model::{self, Message, SamplingParams, TurnRequest, UserBlock};
use crate::prompt;
use crate::run_record::{
    BackendSettings, BudgetConsumed, BudgetLimits, Budgets, Disposition, DurableFacts, Event,
    FailureMode, Phase, ProjectConfig, RecoveryFacts, RunRecord, SCHEMA_VERSION, Task,
    Verification,
};
use crate::store::{RunStore, StoreError};
use crate::time::{Clock, SystemClock, format_rfc3339};
use crate::tool::{Tool, ToolCtx, ToolRegistry, ToolResult};
use crate::transcript::{TranscriptConfig, TranscriptWriter};

/// The registered name of the finish tool — the loop recognizes termination by
/// matching an executed call's name against this.
pub const FINISH_TOOL_NAME: &str = "finish";

/// A compiled JSON Schema every `finish(answer)` payload is validated
/// against — answer mode's mechanical verifier, playing the role a green gate
/// plays for `finish(done)`.
///
/// Holds BOTH the compiled `jsonschema::Validator` and the
/// [`serde_json::Value`] it was compiled from, so the transcript can record
/// the schema itself and a reader can re-verify the harness's verdict off the
/// record. `jsonschema::Validator` is `Debug + Clone + Send + Sync +
/// 'static`, so no `Arc` and no hand-written `Debug` impl are needed and
/// [`RunConfig`]'s `#[derive(Debug, Clone)]` survives.
#[derive(Debug, Clone)]
pub struct AnswerSchema {
    validator: jsonschema::Validator,
    source: Value,
}

/// A result schema that would not compile — the caller handed [`AnswerSchema::compile`]
/// something that is not a valid JSON Schema (e.g. `{"type": 12345}`).
#[derive(Debug, thiserror::Error)]
#[error("invalid answer result schema: {message}")]
pub struct AnswerSchemaError {
    /// The underlying compile failure as `jsonschema` reported it.
    pub message: String,
}

impl AnswerSchema {
    /// Compile `schema` into a reusable validator.
    ///
    /// Built on `jsonschema::validator_for`, so anything that crate rejects
    /// as a schema — a non-object, a `"type"` that is not a string or array
    /// of strings, an unresolvable `$ref` — yields
    /// [`AnswerSchemaError`] rather than a validator that silently accepts
    /// everything.
    pub fn compile(schema: &Value) -> Result<Self, AnswerSchemaError> {
        let validator = jsonschema::validator_for(schema).map_err(|e| AnswerSchemaError {
            message: e.to_string(),
        })?;
        Ok(Self {
            validator,
            source: schema.clone(),
        })
    }

    /// The schema value this validator was compiled from — recorded on the
    /// transcript's `run_start` so a reviewer can re-check any verdict.
    pub fn source(&self) -> &Value {
        &self.source
    }

    /// Validate `instance`, returning an EMPTY vec when it conforms and one
    /// `"<path>: <message>"` string per error otherwise.
    ///
    /// `jsonschema` renders the instance path of a ROOT-level error as the
    /// empty string, which would produce a leading-colon line like
    /// `": 1 is not of type \"object\""`. That is normalized here to `/`, so
    /// every line the model sees names a JSON pointer.
    pub fn validation_errors(&self, instance: &Value) -> Vec<String> {
        self.validator
            .iter_errors(instance)
            .map(|err| {
                let path = err.instance_path().to_string();
                let path = if path.is_empty() { "/" } else { path.as_str() };
                format!("{path}: {err}")
            })
            .collect()
    }
}

/// Configuration for one call to [`run`].
///
/// Bundles the task text, iteration cap, optional [`ChecksRunner`], and the
/// per-turn `max_tokens` cap into one struct so the [`run`] signature stays
/// tight and adding a knob later doesn't force every caller to change. When
/// `checks` is `Some`, `finish(done)` is verified against the runner before
/// being honored (see the module docs).
///
/// Build via [`RunConfig::new`] (which sets the [`DEFAULT_MAX_TOKENS`] default)
/// and layer optional knobs with [`RunConfig::with_checks`] /
/// [`RunConfig::with_max_tokens`].
#[derive(Debug, Clone)]
pub struct RunConfig {
    /// The seed user message: what the agent is being asked to do.
    pub task: String,
    /// Hard cap on model turns before the loop gives up
    /// ([`LoopOutcome::MaxIterations`]).
    pub max_iterations: u32,
    /// The checks the harness re-runs itself to verify a `finish(done)` claim.
    /// `None` means no automated verification — see
    /// [`crate::run_record::Verification::NoChecksConfigured`].
    pub checks: Option<ChecksRunner>,
    /// Per-turn output cap threaded into [`SamplingParams::max_tokens`].
    pub max_tokens: u32,
    /// Static-tree threshold K for finish-recovery: how many consecutive
    /// non-mutating iterations must accumulate AFTER a green `run_checks`
    /// before the harness considers the run "done-but-unclaimed" and injects
    /// a nudge. A successful `edit_file`/`bash` resets the counter. See
    /// [`DEFAULT_STATIC_TREE_K`].
    pub static_tree_k: u32,
    /// Maximum nudges the harness will inject before taking the recovery
    /// terminal ([`FailureMode::FinishDiscipline`]). `0` DISABLES
    /// finish-recovery entirely — no nudge is ever injected and the recovery
    /// terminal is never taken. See [`DEFAULT_MAX_NUDGES`].
    pub max_nudges: u32,
    /// Number of ADDITIONAL attempts after the first try when a
    /// [`model::BackendError::Transient`] failure is returned. With the
    /// default 3, one logical turn calls `backend.turn` at most `1 + 3 = 4`
    /// times. Set to `0` to disable retries.
    ///
    /// **Panic-safety bound:** the exponential schedule
    /// (`retry_backoff_base * 2^attempt`) is panic-safe only for small values;
    /// the default (max exponent 2) is safe. A caller configuring a very large
    /// `max_retries` owns the `Duration`-multiply overflow — saturating math is
    /// deferred (YAGNI at the pinned default).
    ///
    /// See [`DEFAULT_MAX_RETRIES`].
    pub max_retries: u32,
    /// Base delay for the deterministic exponential backoff schedule. The
    /// delay before retry attempt `i` (0-indexed) is `retry_backoff_base *
    /// 2^i`. Set to [`Duration::ZERO`] in tests to run retries with no sleep.
    ///
    /// See [`DEFAULT_RETRY_BACKOFF_BASE`].
    pub retry_backoff_base: Duration,
    /// Wall-clock budget in seconds. `0` means unbounded — the loop only
    /// stops when `max_iterations` is hit or the agent calls `finish`.
    ///
    /// When non-zero, the loop checks elapsed wall-clock time at the end of
    /// each iteration (after tool execution, before the non-terminal
    /// checkpoint) and self-terminates with [`LoopOutcome::BudgetExhausted`]
    /// when elapsed ≥ this value. Recovery facts are persisted so the outer
    /// harness can decide whether to resume.
    pub wall_clock_secs: u64,
    /// The clock implementation used to read "now" inside the loop. Inject a
    /// [`crate::time::FakeClock`] (test-only) for deterministic timing tests
    /// with zero real sleeping; production code uses the [`SystemClock`]
    /// default set by [`RunConfig::new`].
    ///
    /// `Arc<dyn Clock>` satisfies the `Clone` requirement on `RunConfig`
    /// (via `Arc::clone`) and the `Debug` supertrait on [`Clock`] satisfies
    /// `#[derive(Debug)]`.
    pub clock: Arc<dyn Clock>,
    /// Opt-in full run transcript sink (see [`crate::transcript`]). `None`
    /// (the default set by [`RunConfig::new`]) means no transcript is
    /// written and the loop does zero transcript-related filesystem I/O or
    /// event-payload construction. Set via [`RunConfig::with_transcript`].
    pub transcript: Option<TranscriptConfig>,
    /// The result schema that turns on **answer mode**. `None` (the default
    /// set by [`RunConfig::new`]) means the `answer` disposition is neither
    /// advertised in [`FinishTool`]'s schema nor accepted by the parser — a
    /// `finish(answer)` is rejected as an unrecognized disposition, exactly
    /// as it is today. `Some` advertises `answer` and makes the schema the
    /// mechanical verifier for its `result`. Set via
    /// [`RunConfig::with_answer_schema`].
    pub answer_schema: Option<AnswerSchema>,
}

/// The default per-turn output cap. Sized for reasoning models: a model whose
/// thinking counts toward completion tokens (glm-5.2, qwen) can exceed a small
/// cap mid-turn and get truncated *before* it emits a tool call, which the loop
/// then sees as a no-tool-call turn and reports as `StoppedWithoutFinish` (a
/// silent, misdiagnosed stall — see kb-03104). 32768 is safe across every
/// backend: it is half of Claude Haiku 4.5's 64K output ceiling (Sonnet/Opus
/// allow 128K) and well within Ollama's `num_predict`; there is no cost
/// downside since billing is on actual output, not the cap. Override per-run
/// with [`RunConfig::with_max_tokens`].
pub const DEFAULT_MAX_TOKENS: u32 = 32768;

/// Default retry cap: how many ADDITIONAL attempts are made after the first
/// try on a retryable [`model::BackendError::Transient`] failure. With the
/// default of 3, one logical turn calls `backend.turn` at most 4 times before
/// giving up. The exponential backoff schedule (`base * 2^attempt`) is
/// panic-safe for small values; at the pinned default the max exponent is 2
/// (0.5 s / 1 s / 2 s). A caller configuring a very large `max_retries` owns
/// the `Duration`-multiply overflow (saturating math is deferred — YAGNI at
/// the pinned default).
pub const DEFAULT_MAX_RETRIES: u32 = 3;

/// Default base delay for the exponential backoff schedule. The delay before
/// attempt `i` (0-indexed) is `DEFAULT_RETRY_BACKOFF_BASE * 2^i`:
/// 500 ms, 1 000 ms, 2 000 ms for attempts 0/1/2 (with `max_retries = 3`).
pub const DEFAULT_RETRY_BACKOFF_BASE: Duration = Duration::from_millis(500);

/// Default static-tree threshold K — a starting guess, to be tuned against
/// 0.4.0 run data. After K consecutive non-mutating iterations with a green
/// gate, the harness considers the run probably-done-but-unclaimed.
pub const DEFAULT_STATIC_TREE_K: u32 = 3;

/// Default nudge cap N — a starting guess, to be tuned against 0.4.0 run
/// data. After N nudges the harness force-terminates via
/// [`FailureMode::FinishDiscipline`]. Setting `max_nudges == 0` disables
/// finish-recovery entirely.
pub const DEFAULT_MAX_NUDGES: u32 = 2;

impl RunConfig {
    /// Build a config with the given `task` and iteration cap. Defaults
    /// `checks` to `None`, `max_tokens` to [`DEFAULT_MAX_TOKENS`],
    /// `max_retries` to [`DEFAULT_MAX_RETRIES`], and `retry_backoff_base` to
    /// [`DEFAULT_RETRY_BACKOFF_BASE`].
    #[must_use]
    pub fn new(task: impl Into<String>, max_iterations: u32) -> Self {
        Self {
            task: task.into(),
            max_iterations,
            checks: None,
            max_tokens: DEFAULT_MAX_TOKENS,
            static_tree_k: DEFAULT_STATIC_TREE_K,
            max_nudges: DEFAULT_MAX_NUDGES,
            max_retries: DEFAULT_MAX_RETRIES,
            retry_backoff_base: DEFAULT_RETRY_BACKOFF_BASE,
            wall_clock_secs: 0,
            clock: Arc::new(SystemClock),
            transcript: None,
            answer_schema: None,
        }
    }

    /// Attach a [`ChecksRunner`] — the loop will verify `finish(done)` claims
    /// against it and reject any that come back red.
    #[must_use]
    pub fn with_checks(mut self, checks: ChecksRunner) -> Self {
        self.checks = Some(checks);
        self
    }

    /// Turn on **answer mode**: advertise the `answer` disposition and
    /// validate every `finish(answer)` payload against `schema`. Off
    /// (`None`) by default — see [`RunConfig::answer_schema`].
    #[must_use]
    pub fn with_answer_schema(mut self, schema: AnswerSchema) -> Self {
        self.answer_schema = Some(schema);
        self
    }

    /// Override the per-turn output cap ([`DEFAULT_MAX_TOKENS`] by default).
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Override the static-tree threshold K ([`DEFAULT_STATIC_TREE_K`] by
    /// default). See [`RunConfig::static_tree_k`].
    #[must_use]
    pub fn with_static_tree_k(mut self, static_tree_k: u32) -> Self {
        self.static_tree_k = static_tree_k;
        self
    }

    /// Override the nudge cap N ([`DEFAULT_MAX_NUDGES`] by default). `0`
    /// disables finish-recovery entirely. See [`RunConfig::max_nudges`].
    #[must_use]
    pub fn with_max_nudges(mut self, max_nudges: u32) -> Self {
        self.max_nudges = max_nudges;
        self
    }

    /// Override the retry cap ([`DEFAULT_MAX_RETRIES`] by default). `0`
    /// disables retries. See [`RunConfig::max_retries`].
    #[must_use]
    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    /// Override the backoff base delay ([`DEFAULT_RETRY_BACKOFF_BASE`] by
    /// default). Set to [`Duration::ZERO`] in tests to skip the sleep. See
    /// [`RunConfig::retry_backoff_base`].
    #[must_use]
    pub fn with_retry_backoff_base(mut self, base: Duration) -> Self {
        self.retry_backoff_base = base;
        self
    }

    /// Set the wall-clock budget in seconds (`0` = unbounded, the default).
    ///
    /// When non-zero, `run_loop_impl` checks elapsed time at the end of each
    /// iteration and self-terminates with [`LoopOutcome::BudgetExhausted`]
    /// when `elapsed >= wall_clock_secs`.
    #[must_use]
    pub fn with_wall_clock_secs(mut self, secs: u64) -> Self {
        self.wall_clock_secs = secs;
        self
    }

    /// Inject a custom [`Clock`] implementation. Use
    /// [`crate::time::FakeClock`] in tests for zero-sleep deterministic
    /// timing. Production code uses the [`SystemClock`] default set by
    /// [`RunConfig::new`].
    #[must_use]
    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    /// Turn on the opt-in full run transcript (see [`crate::transcript`]),
    /// writing to `path` with the given `label`. Off (`None`) by default —
    /// see [`RunConfig::transcript`].
    #[must_use]
    pub fn with_transcript(mut self, path: impl Into<PathBuf>, label: impl Into<String>) -> Self {
        self.transcript = Some(TranscriptConfig {
            path: path.into(),
            label: label.into(),
        });
        self
    }
}

/// Run-specific persistence bundle passed to [`run_persisted`].
///
/// Kept separate from [`RunConfig`] because `Arc<dyn RunStore>` is not
/// `Debug` (the [`RunStore`] trait has no `Debug` bound), so it cannot be
/// placed in a `derive(Debug, Clone)` struct without breaking those derives on
/// [`RunConfig`].
pub struct Persistence {
    /// Store to write events and checkpoints to.
    pub store: Arc<dyn RunStore>,
    /// Task id — used to compute the [`run_id`] and seed the [`RunRecord`].
    pub task_id: String,
    /// Which attempt number this is for the task (used in the run id).
    pub attempt_n: u32,
    /// Human-readable label for the model backend (e.g. `"claude-sonnet-5"`).
    /// Carried on every [`Event::ModelCall`]; the model backend deliberately
    /// does not expose its own id at the trait level.
    pub model_label: String,
    /// The resolved backend the run was CONSTRUCTED with (see
    /// [`crate::run_record::BackendSettings`] — construction-time settings,
    /// NOT proof of served identity), stamped on the [`RunRecord`] and the
    /// transcript's `run_start` line. Invariant: when this is `Some`,
    /// `model_label` MUST equal `backend_settings.model_label()`.
    pub backend_settings: Option<BackendSettings>,
}

impl std::fmt::Debug for Persistence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Persistence")
            .field("task_id", &self.task_id)
            .field("attempt_n", &self.attempt_n)
            .field("model_label", &self.model_label)
            .field("backend_settings", &self.backend_settings)
            .field("store", &"<dyn RunStore>")
            .finish()
    }
}

/// Produce a run id from `task_id` and `attempt_n`.
///
/// The id is the plain join `"{task_id}:{attempt_n}"` — NOT a hash. Stable
/// across restarts so the same attempt always addresses the same record
/// (idempotent dispatch).
///
/// # Example
///
/// ```
/// use harness::engine::run_id;
/// assert_eq!(run_id("task-42", 1), "task-42:1");
/// ```
pub fn run_id(task_id: &str, attempt_n: u32) -> String {
    format!("{task_id}:{attempt_n}")
}

/// Mode of resume — how to reconstruct context on re-entry after an
/// interruption.
///
/// This is a **runtime call argument**, never persisted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeMode {
    /// **D6 — Crash**: reload `messages` from the last checkpoint and
    /// reconcile any dangling [`Event::ToolCallStarted`] in the log tail.
    /// For each interrupted (or never-started) call, a synthetic
    /// `is_error=true` [`UserBlock::ToolResult`] with content
    /// `"interrupted by host restart"` is fed back — the tool is **never
    /// re-executed** (side effects may have already happened). Continues
    /// checkpointing under the same `run_id`.
    Crash,
    /// **D7**: drop the reloaded `messages` and restart from a freshly-rendered
    /// task seed (byte-identical to what [`run`] would produce). Carries
    /// `phase`, `durable_facts`, and `budgets.consumed` forward. Checkpoints
    /// under a new `run_id` = `"{task_id}:{attempt_n+1}"`, leaving the prior
    /// record intact.
    FreshContext,
}

/// Errors surfaced by [`resume`].
#[derive(Debug)]
pub enum ResumeError {
    /// No checkpoint exists for the requested `run_id`. [`resume`] returns
    /// this immediately — no [`model::ModelBackend::turn`] call is made.
    UnknownRunId(String),
    /// A store load, append, or checkpoint operation failed.
    Store(StoreError),
}

impl std::fmt::Display for ResumeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownRunId(id) => write!(f, "no checkpoint found for run_id {id:?}"),
            Self::Store(e) => write!(f, "store error: {e}"),
        }
    }
}

impl std::error::Error for ResumeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::UnknownRunId(_) => None,
            Self::Store(e) => Some(e),
        }
    }
}

impl From<StoreError> for ResumeError {
    fn from(e: StoreError) -> Self {
        Self::Store(e)
    }
}

/// Private in-loop persistence context — bundles the computed run id with the
/// live [`RunRecord`] being mutated as the loop progresses. Only constructed
/// (and used) when a [`Persistence`] is supplied to [`run_persisted`].
struct RunPersist {
    rid: String,
    record: RunRecord,
}

/// A claim the model made in a `finish` call, parsed from its raw JSON input.
///
/// This is the *pre-verification* view: it captures what the model said, and
/// the loop then decides whether to accept it as a [`Disposition`]. Kept
/// internal because callers should only ever see the post-verification
/// [`Disposition`].
///
/// `Eq` is NOT derived: [`Self::Answer`] carries a [`serde_json::Value`],
/// which is `PartialEq` but not `Eq`. Every existing use compares with
/// `assert_eq!`, which needs only `PartialEq`.
#[derive(Debug, PartialEq)]
enum FinishClaim {
    Done {
        summary: String,
    },
    /// The task was already complete and nothing needed changing. Carries the
    /// non-empty `reason` the model supplied; any `summary` on this branch is
    /// deliberately discarded.
    AlreadySatisfied {
        reason: String,
    },
    Blocked {
        decision_needed: String,
    },
    Failed {
        summary: String,
    },
    /// Answer mode's claim: the deliverable is the `result` payload, which
    /// the loop validates against the run's configured [`AnswerSchema`].
    /// Selected ONLY when answer mode is enabled; with no schema configured a
    /// `disposition` of `answer` falls through to [`Self::Invalid`] like any
    /// other unrecognized value. `result` is `None` when the key is absent
    /// and `Some` for ANY present JSON value (including `null`); any
    /// `summary` on this branch is deliberately discarded.
    Answer {
        result: Option<Value>,
    },
    /// The model's `disposition` was missing, non-string, or unrecognized;
    /// `raw` is its JSON serialization, or `<missing>` when the key is
    /// absent or the input is not an object. `"complete"`, `"success"` and
    /// `"finished"` all land here. Never terminates the loop.
    Invalid {
        raw: String,
    },
    /// The disposition was `already_satisfied` but the `reason` key was
    /// missing, non-string, or blank after trimming. Never terminates the
    /// loop; counted as a malformed finish call.
    MissingReason,
}

impl FinishClaim {
    /// Parse the `finish` tool's raw JSON input into a claim.
    ///
    /// `disposition` selects the variant: a JSON string that, after
    /// trimming and ASCII-lowercasing, equals `done`, `already_satisfied`,
    /// `blocked`, or `failed` selects that variant; anything else — a string
    /// outside that set, a non-string value, or a missing/non-object input —
    /// yields [`Self::Invalid`]. `summary` and `decision_needed` are read as
    /// strings on the accepted variants (absent or non-string → empty).
    ///
    /// The `already_satisfied` branch reads ONLY the `reason` key — a
    /// supplied `summary` is deliberately discarded, because
    /// [`Disposition::AlreadySatisfied`] carries no summary field and
    /// `reason` is the human-readable text. A missing, non-string, or
    /// blank-after-trimming `reason` yields [`Self::MissingReason`].
    ///
    /// `answer_enabled` is the answer-mode gate — `true` exactly when the run
    /// has an [`AnswerSchema`] configured. Only then does a `disposition` of
    /// `answer` (after the same trim + ASCII-lowercase) select
    /// [`Self::Answer`], reading `result` as `input.get("result").cloned()`
    /// (present-but-any-JSON-type is `Some`, absent is `None`). With
    /// `answer_enabled == false` the value falls through to the ordinary
    /// [`Self::Invalid`] arm, so the rejection wording, the
    /// `invalid_finish_calls` bump and the recorded `raw` are inherited
    /// byte-for-byte. Like `already_satisfied`, the `answer` branch DISCARDS
    /// any supplied `summary` — [`Self::Answer`] carries only `result`.
    ///
    /// No other normalization is performed: no Unicode case folding, no
    /// synonyms. `"complete"`, `"success"`, and `"finished"` are all
    /// [`Self::Invalid`].
    fn from_input(input: &Value, answer_enabled: bool) -> Self {
        let field = |key: &str| {
            input
                .get(key)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        let Some(disposition) = input.get("disposition") else {
            return Self::Invalid {
                raw: "<missing>".to_string(),
            };
        };
        match disposition.as_str().map(|s| s.trim().to_ascii_lowercase()) {
            Some(s) if s == "done" => Self::Done {
                summary: field("summary"),
            },
            Some(s) if s == "already_satisfied" => {
                let reason = field("reason");
                if reason.trim().is_empty() {
                    Self::MissingReason
                } else {
                    Self::AlreadySatisfied { reason }
                }
            }
            Some(s) if s == "blocked" => Self::Blocked {
                decision_needed: field("decision_needed"),
            },
            Some(s) if s == "failed" => Self::Failed {
                summary: field("summary"),
            },
            Some(s) if answer_enabled && s == "answer" => Self::Answer {
                result: input.get("result").cloned(),
            },
            _ => Self::Invalid {
                raw: disposition.to_string(),
            },
        }
    }
}

/// Mechanical statistics accumulated by the loop, carried alongside every
/// [`LoopOutcome`] on the [`RunResult`] a call to [`run`] returns.
///
/// - `iterations`: how many **logical** iterations the loop executed — one per
///   `for`-loop pass. A turn that fails transiently and is retried within the
///   same pass still counts as **one** logical iteration; retry draws within a
///   single pass are NOT counted separately. A
///   [`LoopOutcome::BackendError`] on the FIRST turn yields `iterations = 1`.
///   [`LoopOutcome::MaxIterations`] always yields
///   `iterations == config.max_iterations`. Raw `backend.turn` draw count
///   (including retries within a pass) is observable in tests via
///   [`crate::test_support::MockBackend::calls`].
/// - `input_tokens` / `output_tokens`: the sum of
///   [`AssistantTurn.usage.input_tokens`](crate::model::Usage::input_tokens) /
///   [`output_tokens`](crate::model::Usage::output_tokens) across every
///   SUCCESSFUL turn. Turns that returned a
///   [`BackendError`](crate::model::BackendError) contribute nothing. Per-turn
///   `u32` values sum into `u64` so a long run can't overflow.
/// - `cache_read_tokens` / `cache_write_tokens`: the sum of
///   [`Usage::cache_read_tokens`](crate::model::Usage::cache_read_tokens) /
///   [`Usage::cache_write_tokens`](crate::model::Usage::cache_write_tokens)
///   across every SUCCESSFUL turn, treating `None` as 0 (a provider that
///   doesn't report cache tokens contributes nothing). With prompt caching
///   on, Anthropic MOVES cached input out of `input_tokens` into these two
///   buckets — so the harness-overhead number comparable to an UNCACHED run
///   is `input_tokens + cache_read_tokens + cache_write_tokens` (the
///   "raw input"), NOT `input_tokens` alone.
/// - `wall_clock`: measured across the whole [`run`] call — from just before
///   the loop starts to just after it returns.
/// - `nudges_fired`: finish-recovery nudges injected this loop invocation.
///   Counted since THIS loop invocation — a resumed run starts from zero;
///   pre-crash values live in `RunRecord::recovery_facts`.
/// - `tree_dirty`: a LATCH set by the first successful `edit_file`/`bash`
///   and NEVER cleared for the rest of the run. Counted since THIS loop
///   invocation — a resumed run starts from zero; pre-crash values live in
///   `RunRecord::recovery_facts`.
/// - `iters_since_tree_change_at_exit`: the consecutive-non-mutating-iteration
///   counter's value at the terminal. On the `Finished` terminal
///   (engine.rs:1414) reflects state as of the END of the PREVIOUS iteration,
///   because that return precedes the counter-update block at
///   engine.rs:1420-1424. Counted since THIS loop invocation — a resumed run
///   starts from zero; pre-crash values live in `RunRecord::recovery_facts`.
/// - `peak_iters_since_tree_change`: the maximum that the
///   consecutive-non-mutating-iteration counter EVER reached during the run.
///   On the `Finished` terminal (engine.rs:1414) reflects state as of the END
///   of the PREVIOUS iteration, because that return precedes the
///   counter-update block at engine.rs:1420-1424. Counted since THIS loop
///   invocation — a resumed run starts from zero; pre-crash values live in
///   `RunRecord::recovery_facts`.
/// - `mutating_iters`: count of loop iterations whose end-of-iteration
///   classification was mutating. On the `Finished` terminal (engine.rs:1414)
///   reflects state as of the END of the PREVIOUS iteration, because that
///   return precedes the counter-update block at engine.rs:1420-1424. Counted
///   since THIS loop invocation — a resumed run starts from zero; pre-crash
///   values live in `RunRecord::recovery_facts`.
/// - `bash_calls_ok` / `edit_file_calls_ok`: count of individual SUCCESSFUL
///   (`!is_error`) `bash` / `edit_file` tool calls. Counted since THIS loop
///   invocation — a resumed run starts from zero; pre-crash values live in
///   `RunRecord::recovery_facts`.
/// - `invalid_finish_calls`: count of MALFORMED `finish` calls — rejected as
///   [`FinishClaim::Invalid`] (missing, non-string, or unrecognized
///   `disposition`) or as [`FinishClaim::MissingReason`] (an
///   `already_satisfied` with no usable `reason`). A red-verification or
///   unchanged-tree rejection is NOT counted here — those have their own
///   fields (`no_change_rejections`, `already_satisfied_check_rejections`).
///   Counted since THIS loop invocation — a resumed run starts from zero.
/// - `first_invalid_finish_raw`: the untruncated `raw` of the FIRST
///   malformed claim this loop invocation (the literal
///   `already_satisfied without reason` for a
///   [`FinishClaim::MissingReason`]); never overwritten after being set.
///   Counted since THIS loop invocation — a resumed run starts from zero.
/// - `no_change_rejections`: count of `finish(done)` claims rejected because
///   the working tree was unchanged since the run started (leg 3). Counted
///   since THIS loop invocation — a resumed run starts from zero.
/// - `already_satisfied_check_rejections`: count of
///   `finish(already_satisfied)` claims rejected by a red gate. Counted
///   since THIS loop invocation — a resumed run starts from zero.
/// - `answer_schema_rejections`: count of `finish(answer)` claims rejected
///   because the supplied `result` did not validate against the configured
///   [`AnswerSchema`]. A missing `result` is NOT counted here — that is a
///   malformed finish call and lands in `invalid_finish_calls`. Counted
///   since THIS loop invocation — a resumed run starts from zero.
/// - `modified_workspace_rejections`: count of `finish(answer)` claims
///   rejected because the working tree CHANGED since the run started (answer
///   mode's INVERTED leg 3 — an answer run must not modify the workspace).
///   A wrong-mode `done` / `already_satisfied` claim on an answer run is NOT
///   counted here (or anywhere): it is a steering rejection, not evidence
///   about the workspace. Counted since THIS loop invocation — a resumed run
///   starts from zero.
/// - `tree_baseline_unobservable`: whether the run-start tree observation
///   failed, which makes the leg-3 precondition INERT for the whole run (it
///   fails open). The inert-detector: without it, a precondition that
///   silently disabled itself in production is observationally identical to
///   one that is working. Counted since THIS loop invocation — a resumed run
///   always reports `true`, since `resume` supplies an unobservable baseline
///   by construction.
///
/// Deliberately NOT `serde`: persistence wiring
/// (into [`crate::run_record`]) is a later milestone; this type is the
/// in-memory shape the loop hands its caller today.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunStats {
    /// Logical iterations executed — one per `for`-loop pass. Retry draws
    /// within one pass count once here; use `MockBackend::calls()` in tests
    /// to observe the raw `backend.turn` draw count including retries.
    pub iterations: u32,
    /// Sum of `usage.input_tokens` across successful turns.
    pub input_tokens: u64,
    /// Sum of `usage.output_tokens` across successful turns.
    pub output_tokens: u64,
    /// Sum of `usage.cache_read_tokens` across successful turns (`None` → 0).
    /// With prompt caching on, Anthropic moves cached input out of
    /// `input_tokens` into this bucket — so the harness-overhead number
    /// comparable to an UNCACHED run is `input_tokens + cache_read_tokens +
    /// cache_write_tokens` (raw input), NOT `input_tokens` alone.
    pub cache_read_tokens: u64,
    /// Sum of `usage.cache_write_tokens` across successful turns (`None` → 0).
    /// See [`Self::cache_read_tokens`] — populated by Anthropic when a turn
    /// writes a fresh cache entry.
    pub cache_write_tokens: u64,
    /// Wall-clock elapsed across the whole [`run`] call.
    pub wall_clock: Duration,
    /// Whether the harness's last in-loop gate (`run_checks`) was GREEN at the
    /// terminal — i.e. the value of `last_gate_green` at exit. Lets a caller
    /// distinguish a [`LoopOutcome::StoppedWithoutFinish`] where the model had
    /// verified green in-loop then stopped (a finish-discipline miss that
    /// finish-recovery's precondition *could* target) from one where the gate
    /// was never green in-loop (correctly out of scope — nudging toward
    /// `finish` on an unverified state would risk a false claim). Best-effort
    /// telemetry: an error-propagation (`?`) exit reports the last observed
    /// value.
    pub gates_green_at_exit: bool,
    /// Finish-recovery nudges injected this loop invocation. Counted since
    /// THIS loop invocation — a resumed run (`resume`, engine.rs:1808) starts
    /// from zero; pre-crash values live in `RunRecord::recovery_facts`
    /// (`crates/harness/src/run_record.rs`).
    pub nudges_fired: u32,
    /// A LATCH set by the first successful `edit_file`/`bash`
    /// (engine.rs:1362) and NEVER cleared for the rest of the run. Counted
    /// since THIS loop invocation — a resumed run (`resume`, engine.rs:1808)
    /// starts from zero; pre-crash values live in `RunRecord::recovery_facts`
    /// (`crates/harness/src/run_record.rs`).
    pub tree_dirty: bool,
    /// The consecutive-non-mutating-iteration counter's value at the
    /// terminal. On the `Finished` terminal (engine.rs:1414) this value
    /// reflects state as of the END of the PREVIOUS iteration, because that
    /// return precedes the counter-update block at engine.rs:1420-1424.
    /// Counted since THIS loop invocation — a resumed run (`resume`,
    /// engine.rs:1808) starts from zero; pre-crash values live in
    /// `RunRecord::recovery_facts` (`crates/harness/src/run_record.rs`).
    pub iters_since_tree_change_at_exit: u32,
    /// The maximum that the consecutive-non-mutating-iteration counter EVER
    /// reached during the run. On the `Finished` terminal (engine.rs:1414)
    /// this value reflects state as of the END of the PREVIOUS iteration,
    /// because that return precedes the counter-update block at
    /// engine.rs:1420-1424. Counted since THIS loop invocation — a resumed
    /// run (`resume`, engine.rs:1808) starts from zero; pre-crash values live
    /// in `RunRecord::recovery_facts` (`crates/harness/src/run_record.rs`).
    pub peak_iters_since_tree_change: u32,
    /// Count of loop iterations whose end-of-iteration classification was
    /// mutating. On the `Finished` terminal (engine.rs:1414) this value
    /// reflects state as of the END of the PREVIOUS iteration, because that
    /// return precedes the counter-update block at engine.rs:1420-1424.
    /// Counted since THIS loop invocation — a resumed run (`resume`,
    /// engine.rs:1808) starts from zero; pre-crash values live in
    /// `RunRecord::recovery_facts` (`crates/harness/src/run_record.rs`).
    pub mutating_iters: u32,
    /// Count of individual SUCCESSFUL (`!is_error`) `bash` tool calls.
    /// Counted since THIS loop invocation — a resumed run (`resume`,
    /// engine.rs:1808) starts from zero; pre-crash values live in
    /// `RunRecord::recovery_facts` (`crates/harness/src/run_record.rs`).
    pub bash_calls_ok: u32,
    /// Count of individual SUCCESSFUL (`!is_error`) `edit_file` tool calls.
    /// Counted since THIS loop invocation — a resumed run (`resume`,
    /// engine.rs:1808) starts from zero; pre-crash values live in
    /// `RunRecord::recovery_facts` (`crates/harness/src/run_record.rs`).
    pub edit_file_calls_ok: u32,
    /// Count of MALFORMED `finish` calls — rejected as
    /// [`FinishClaim::Invalid`] (a missing, non-string, or unrecognized
    /// `disposition`) or as [`FinishClaim::MissingReason`] (an
    /// `already_satisfied` with no usable `reason`). A red-verification or
    /// unchanged-tree rejection is NOT counted here; see
    /// [`Self::no_change_rejections`] and
    /// [`Self::already_satisfied_check_rejections`]. Counted since THIS loop
    /// invocation — a resumed run (`resume`, engine.rs:1808) starts from
    /// zero.
    pub invalid_finish_calls: u32,
    /// The untruncated `raw` of the FIRST malformed `finish` call this loop
    /// invocation — the literal `already_satisfied without reason` when it
    /// was a [`FinishClaim::MissingReason`]; never overwritten once set.
    /// Counted since THIS loop invocation — a resumed run (`resume`,
    /// engine.rs:1808) starts from zero.
    pub first_invalid_finish_raw: Option<String>,
    /// Count of `finish(done)` claims rejected because the working tree was
    /// unchanged since the run started (leg 3 of the completion contract).
    /// Counted since THIS loop invocation — a resumed run (`resume`,
    /// engine.rs:1808) starts from zero.
    pub no_change_rejections: u32,
    /// Count of `finish(already_satisfied)` claims rejected by a red gate.
    /// Counted since THIS loop invocation — a resumed run (`resume`,
    /// engine.rs:1808) starts from zero.
    pub already_satisfied_check_rejections: u32,
    /// Count of `finish(answer)` claims rejected because the supplied
    /// `result` failed to validate against the configured [`AnswerSchema`].
    /// A missing `result` is NOT counted here — it is a malformed finish call
    /// and bumps [`Self::invalid_finish_calls`] instead. Counted since THIS
    /// loop invocation — a resumed run (`resume`, engine.rs:1808) starts from
    /// zero.
    pub answer_schema_rejections: u32,
    /// Count of `finish(answer)` claims rejected because the working tree
    /// CHANGED since the run started — answer mode's INVERTED leg-3
    /// precondition. A wrong-mode `done` / `already_satisfied` claim on an
    /// answer run bumps NOTHING; it is steering, not evidence. Counted since
    /// THIS loop invocation — a resumed run (`resume`) starts from zero.
    pub modified_workspace_rejections: u32,
    /// Whether the run-start tree observation failed, making the leg-3
    /// precondition INERT for this run (it fails open). The inert-detector:
    /// a precondition that silently disabled itself in production would
    /// otherwise be observationally identical to one that is working.
    /// Counted since THIS loop invocation — a resumed run (`resume`,
    /// engine.rs:1808) always reports `true`, since it supplies an
    /// unobservable baseline by construction.
    pub tree_baseline_unobservable: bool,
}

/// The full result of one [`run`] call: the terminal [`LoopOutcome`] plus the
/// mechanical [`RunStats`] carried alongside it.
///
/// Not `PartialEq` — inherited from [`LoopOutcome`], whose
/// [`model::BackendError`] variant is a runtime error that doesn't compare.
#[derive(Debug)]
pub struct RunResult {
    /// Why the loop stopped.
    pub outcome: LoopOutcome,
    /// Mechanical stats accumulated over the run.
    pub stats: RunStats,
}

/// Why the agent loop stopped.
///
/// Not `PartialEq` because [`model::BackendError`] is a runtime error type that
/// doesn't compare; tests match on the variant instead.
#[derive(Debug)]
pub enum LoopOutcome {
    /// The agent called the `finish` tool AND the harness accepted the
    /// claim. Carries the post-verification [`Disposition`]; a `Done`
    /// here has evidence by construction (see [`Verification`]).
    Finished(Disposition),
    /// A turn produced no tool calls (the model ended its turn without
    /// finishing). The loop has nothing to feed back, so it stops.
    StoppedWithoutFinish,
    /// The hard `max_iterations` cap was reached before the agent reached an
    /// accepted finish. Repeated `finish(done)` claims that fail verification
    /// bottom out here — loop/rejection-counter detection lands with a
    /// separate item. Repeated `finish` calls with an unrecognized or missing
    /// disposition also end here; `RunStats::invalid_finish_calls > 0` tells
    /// that case apart from repeated red-verification `done` rejections.
    MaxIterations,
    /// The wall-clock budget ([`RunConfig::wall_clock_secs`]) expired before
    /// the agent finished. The summary is always
    /// `"wall-clock budget exhausted"`. Recovery facts are written to the
    /// persisted run record so the outer harness can decide whether to resume.
    /// Distinct from [`LoopOutcome::MaxIterations`] so the outer harness can
    /// recognise a time-bounded termination from an iteration-bounded one.
    BudgetExhausted { summary: String },
    /// The backend returned an error that exhausted the retry budget. Carries
    /// the **last** attempt's error. Retryable errors
    /// ([`model::BackendError::Transient`]) are retried up to
    /// [`RunConfig::max_retries`] additional times with deterministic
    /// exponential backoff before reaching this variant. Non-retryable errors
    /// ([`model::BackendError::Terminal`], [`model::BackendError::Protocol`],
    /// [`model::BackendError::ContextLengthExceeded`]) reach this variant on
    /// first occurrence.
    BackendError(model::BackendError),
}

impl LoopOutcome {
    /// Convert a terminal [`LoopOutcome`] into its [`Disposition`].
    ///
    /// Maps every loop-exit reason to the appropriate run-record disposition:
    /// - `Finished(d)` → `d` (already a `Disposition`)
    /// - `MaxIterations` → `Failed { mode: BudgetExhausted, .. }` (summary:
    ///   `"iteration cap reached before the agent finished"`)
    /// - `BudgetExhausted { summary }` → `Failed { mode: BudgetExhausted,
    ///   summary }` (summary: `"wall-clock budget exhausted"`)
    /// - `StoppedWithoutFinish` → `Failed { mode: StoppedWithoutFinish, .. }`
    /// - `BackendError(e)` → `Failed { mode: TransientInfra }` if
    ///   `e.is_retryable()`, else `Failed { mode: PersistentToolError }`
    ///
    /// Both `MaxIterations` and `BudgetExhausted` yield
    /// `mode: FailureMode::BudgetExhausted`; they are distinguishable by their
    /// `summary` strings.
    ///
    /// The `into_` prefix (consuming `self` by value) satisfies
    /// `clippy::wrong_self_convention` for a non-`Copy` type.
    pub fn into_disposition(self) -> Disposition {
        match self {
            LoopOutcome::Finished(d) => d,
            LoopOutcome::MaxIterations => Disposition::Failed {
                mode: FailureMode::BudgetExhausted,
                summary: "iteration cap reached before the agent finished".to_string(),
            },
            LoopOutcome::BudgetExhausted { summary } => Disposition::Failed {
                mode: FailureMode::BudgetExhausted,
                summary,
            },
            LoopOutcome::StoppedWithoutFinish => Disposition::Failed {
                mode: FailureMode::StoppedWithoutFinish,
                summary: "agent stopped generating tool calls without calling finish".to_string(),
            },
            LoopOutcome::BackendError(e) => Disposition::Failed {
                mode: if e.is_retryable() {
                    FailureMode::TransientInfra
                } else {
                    FailureMode::PersistentToolError
                },
                summary: format!("backend error: {e:?}"),
            },
        }
    }
}

/// The `finish` tool: the agent calls it to end the run.
///
/// Its input is `{ disposition: "done" | "blocked" | "failed", summary:
/// string, decision_needed?: string }`.
///
/// `answer_mode` gates the answer-mode surface: with it `false` (the
/// [`Default`], and every build-mode run) [`Tool::schema`] returns exactly
/// the schema it always has — the string `answer` appears nowhere in it. With
/// it `true` the `disposition` enum gains `answer` (LAST, so the existing
/// members keep their order), a `result` property is advertised, and the
/// prose describes both. `"required"` is `["disposition", "summary"]` in BOTH
/// modes: `result`'s presence is enforced by the harness's pinned
/// missing-result rejection, exactly the way `reason` is enforced for
/// `already_satisfied` today.
///
/// [`Tool::run`] here just returns an ok acknowledgment; the LOOP is what
/// recognizes the name, parses the input, verifies a `done` claim against
/// the configured [`ChecksRunner`], and (when the claim is accepted) builds
/// the terminal [`LoopOutcome::Finished`]. The tool's `.run` result is only
/// used when no checks are configured or when the disposition is
/// `blocked`/`failed`; a verified-red `done` bypasses it entirely and the
/// loop synthesizes an `is_error=true` [`ToolResult`] the model can react to.
/// An unrecognized or missing `disposition` likewise never reaches
/// `.run`: the loop rejects it with an `is_error=true` result and the loop
/// continues.
///
/// [`ChecksRunner`]: crate::exec::ChecksRunner
#[derive(Debug, Default, Clone, Copy)]
pub struct FinishTool {
    /// Whether to advertise the `answer` disposition and its `result`
    /// property. `false` (the [`Default`]) yields the byte-identical
    /// build-mode schema.
    pub answer_mode: bool,
}

#[async_trait]
impl Tool for FinishTool {
    // The trait fixes the return type as `&str`; a `&'static str` here would
    // diverge from the trait signature, so the lint doesn't apply.
    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        FINISH_TOOL_NAME
    }

    #[allow(clippy::too_many_lines)]
    fn schema(&self) -> Value {
        if self.answer_mode {
            return json!({
                "name": FINISH_TOOL_NAME,
                "description": "End the run. Call it when the task is complete, \
                                already satisfied, answered, blocked on a decision, or has \
                                failed. A `done` claim is verified by the harness re-running \
                                the configured checks AND requiring that the working tree \
                                changed since the run started; an `answer` claim is verified \
                                by the harness validating your `result` against the run's \
                                configured result schema. An unchanged tree, a failed \
                                verification, an absent or schema-invalid `result`, or a \
                                disposition other than \
                                done/already_satisfied/answer/blocked/failed is fed back as \
                                a tool-result error you can react to, not a termination.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "disposition": {
                            "type": "string",
                            "enum": ["done", "blocked", "failed", "already_satisfied", "answer"],
                            "description": "done = task complete and you changed something \
                                            (harness verifies via checks AND requires a changed \
                                            working tree); already_satisfied = the task was \
                                            already complete and nothing needed changing \
                                            (requires `reason`; harness still verifies via \
                                            checks); answer = the deliverable is data, not a \
                                            change: supply `result` conforming to the run's \
                                            configured result schema (an absent or \
                                            schema-invalid `result` is fed back as a \
                                            tool-result error, not a termination); blocked = \
                                            needs a decision before retrying; failed = you \
                                            could not complete the task in this attempt and a \
                                            fresh attempt might succeed."
                        },
                        "summary": {
                            "type": "string",
                            "description": "A short summary of the outcome."
                        },
                        "decision_needed": {
                            "type": "string",
                            "description": "Required when blocked: the decision a human must make."
                        },
                        "reason": {
                            "type": "string",
                            "description": "Required when the disposition is already_satisfied: \
                                            what you checked and why the task was already \
                                            complete."
                        },
                        "result": {
                            "description": "Required when the disposition is answer: the \
                                            deliverable payload. It MUST conform to the run's \
                                            configured result schema; the harness validates it \
                                            and feeds any validation errors back to you as a \
                                            tool-result error rather than terminating the run."
                        }
                    },
                    "required": ["disposition", "summary"]
                }
            });
        }
        json!({
            "name": FINISH_TOOL_NAME,
            "description": "End the run. Call it when the task is complete, \
                            already satisfied, blocked on a decision, or has failed. A \
                            `done` claim is verified by the harness re-running the \
                            configured checks AND requiring that the working tree \
                            changed since the run started; an unchanged tree, a failed \
                            verification, or a disposition other than \
                            done/already_satisfied/blocked/failed is fed back as a \
                            tool-result error you can react to, not a termination.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "disposition": {
                        "type": "string",
                        "enum": ["done", "blocked", "failed", "already_satisfied"],
                        "description": "done = task complete and you changed something \
                                        (harness verifies via checks AND requires a changed \
                                        working tree); already_satisfied = the task was \
                                        already complete and nothing needed changing \
                                        (requires `reason`; harness still verifies via \
                                        checks); blocked = needs a decision before retrying; \
                                        failed = you could not complete the task in this \
                                        attempt and a fresh attempt might succeed."
                    },
                    "summary": {
                        "type": "string",
                        "description": "A short summary of the outcome."
                    },
                    "decision_needed": {
                        "type": "string",
                        "description": "Required when blocked: the decision a human must make."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Required when the disposition is already_satisfied: \
                                        what you checked and why the task was already \
                                        complete."
                    }
                },
                "required": ["disposition", "summary"]
            }
        })
    }

    async fn run(&self, _input: Value, _ctx: &ToolCtx) -> ToolResult {
        ToolResult::ok("finish acknowledged")
    }
}

/// Render a [`ToolResult`] into the string fed back to the model as a
/// [`UserBlock::ToolResult`]'s content: the always-small `summary`, plus the
/// bounded `detail` when present (already capped at
/// [`crate::tool::DETAIL_CAP`] by the result constructor).
fn render_tool_result(result: &ToolResult) -> String {
    match &result.detail {
        Some(detail) => format!("{}\n{detail}", result.summary),
        None => result.summary.clone(),
    }
}

/// Build the `is_error=true` fed-back content for a finish claim REJECTED by
/// a red verification: the "rejected" header + the check report's excerpt +
/// a pointer to the full offloaded output when the runner offloaded it.
///
/// `claim` is the disposition label that was rejected (`done` or
/// `already_satisfied`), so the header names what the model actually asked
/// for. The wording ("finish(<claim>) rejected: verification failed") is
/// load-bearing — a test in this module pins the substring "rejected" as the
/// steering signal the model must see.
fn rejection_content(claim: &str, report: &CheckReport) -> String {
    use std::fmt::Write as _;
    let mut content = format!("finish({claim}) rejected: verification failed");
    if !report.excerpt.is_empty() {
        content.push('\n');
        content.push_str(&report.excerpt);
    }
    if let Some(path) = &report.offload_path {
        // `write!` into a String is infallible (its `write_str` cannot fail),
        // so `.expect` here is a lint-satisfying no-op, not a real recovery
        // path.
        write!(content, "\n\n[full check output: {}]", path.display())
            .expect("write! into String is infallible");
    }
    content
}

/// The `is_error=true` fed-back content for a `finish(done)` rejected by the
/// leg-3 precondition: the checks were green, but the working tree is exactly
/// what the run started from. Points at both ways forward — do the work, or
/// take the `already_satisfied` off-ramp.
fn no_change_rejection_content() -> String {
    "finish(done) rejected: the working tree is unchanged since this run started — no work \
     was done. Make the change the task requires and finish again, or call finish with \
     disposition `already_satisfied` and a `reason` if the task was already complete."
        .to_string()
}

/// The `is_error=true` fed-back content for an `already_satisfied` claim with
/// no usable `reason`.
fn missing_reason_rejection_content() -> String {
    "finish rejected: already_satisfied requires a non-empty `reason` explaining what you \
     checked and why the task was already complete. Call finish again with a reason, or with \
     a different disposition."
        .to_string()
}

/// The `is_error=true` fed-back content for a `finish(answer)` that supplied
/// no `result` at all. The sibling of [`missing_reason_rejection_content`]:
/// answer mode's `result` is enforced by the harness, not by the tool schema's
/// `required` list, so the model needs wording that names both ways forward.
fn missing_result_rejection_content() -> String {
    "finish rejected: answer requires a `result` that conforms to the configured result \
     schema. Call finish again with disposition `answer` and a `result`, or with a different \
     disposition."
        .to_string()
}

/// The `is_error=true` fed-back content for a `finish(answer)` whose `result`
/// validated but whose working tree CHANGED since the run started — answer
/// mode's INVERTED leg-3 precondition.
///
/// Build mode requires evidence that work happened; answer mode requires
/// evidence that it did NOT. The wording therefore has to do two jobs: say
/// why the claim bounced, and tell the model concretely how to get back to an
/// acceptable state (revert, then finish again). The changed paths are
/// appended as evidence — `git status --porcelain` of the CURRENT observation,
/// capped at [`TREE_PORCELAIN_RENDER_CAP`] characters like every other
/// model-facing tree rendering.
///
/// An observation with an EMPTY porcelain that still classified as
/// `TreeChanged` means `HEAD` moved (a commit, a checkout, a reset) with a
/// clean tree, so there are no paths to list; the lead-in says that instead.
fn modified_workspace_rejection_content(current: &TreeObservation) -> String {
    let evidence = match current {
        TreeObservation::Observed { porcelain, .. } if porcelain.trim().is_empty() => {
            "HEAD moved".to_string()
        }
        TreeObservation::Observed { porcelain, .. } => format!(
            "changed paths:\n{}",
            exec::tail(porcelain, TREE_PORCELAIN_RENDER_CAP)
        ),
        // Unreachable in practice — `TreeChanged` requires BOTH observations
        // to be `Observed`. Expressed as an arm rather than an `expect` so a
        // future classifier change degrades to a bare rejection, not a panic.
        TreeObservation::Unobservable { .. } => "changed paths:".to_string(),
    };
    format!(
        "finish(answer) rejected: the working tree changed since this run started — an \
         answer run must not modify the workspace. Revert your edits (restore tracked files \
         and delete files you created) and call finish(answer) again.\n{evidence}"
    )
}

/// The `is_error=true` fed-back content for a build-mode terminal disposition
/// (`done` / `already_satisfied`) claimed on an ANSWER-mode run.
///
/// `claim` is the disposition as the model named it, so the rejection opens
/// with the same `finish(<claim>) rejected:` shape every other rejection uses.
/// Returned BEFORE any checks run and before the tree is observed — a
/// wrong-mode claim costs nothing to diagnose, and running a gate for it would
/// be pure waste.
fn wrong_mode_rejection_content(claim: &str) -> String {
    format!(
        "finish({claim}) rejected: this run is in answer mode — the only accepted terminal \
         dispositions are answer, blocked, failed. End it with disposition `answer` and a \
         `result` matching the schema in the task."
    )
}

/// Character cap on the rendered schema-error block fed back to the model.
/// Mirrors `exec::CHECK_EXCERPT_CAP` (also `4_000`), the precedent for
/// model-facing bounded text: a pathological schema can produce megabytes of
/// errors, and an unbounded feed-back would blow the context window the
/// rejection exists to steer.
const ANSWER_SCHEMA_ERRORS_CAP: usize = 4_000;

/// Cap on how many individual schema-error LINES are rendered before an
/// `…and N more` tail. A line cap as well as a character cap so a few
/// enormous messages cannot crowd out the count of how many problems there
/// actually are.
const ANSWER_SCHEMA_ERRORS_MAX_LINES: usize = 20;

/// The `is_error=true` fed-back content for a `finish(answer)` whose `result`
/// did not validate: the pinned header line plus the (bounded) error lines,
/// one per line.
///
/// Bounded twice — at most [`ANSWER_SCHEMA_ERRORS_MAX_LINES`] error lines
/// (then an `…and N more` line), and the whole rendered string truncated to
/// [`ANSWER_SCHEMA_ERRORS_CAP`] characters with an interpolated marker so the
/// marker cannot drift from the constant.
fn answer_schema_rejection_content(errors: &[String]) -> String {
    let mut content =
        "finish(answer) rejected: result does not conform to the configured schema:".to_string();
    for err in errors.iter().take(ANSWER_SCHEMA_ERRORS_MAX_LINES) {
        content.push('\n');
        content.push_str(err);
    }
    if errors.len() > ANSWER_SCHEMA_ERRORS_MAX_LINES {
        use std::fmt::Write as _;
        // `write!` into a String is infallible (its `write_str` cannot fail),
        // so `.expect` here is a lint-satisfying no-op, not a real recovery
        // path — the same idiom `rejection_content` uses.
        write!(
            content,
            "\n…and {} more",
            errors.len() - ANSWER_SCHEMA_ERRORS_MAX_LINES
        )
        .expect("write! into String is infallible");
    }
    if content.chars().count() > ANSWER_SCHEMA_ERRORS_CAP {
        let head: String = content.chars().take(ANSWER_SCHEMA_ERRORS_CAP).collect();
        content = format!("{head}…[truncated at {ANSWER_SCHEMA_ERRORS_CAP} chars]");
    }
    content
}

/// Which rejection a well-formed-but-unaccepted `finish` call produced. Read
/// by the loop to increment the matching [`RunStats`] counter — the same
/// discriminator-on-[`FinishOutcome`] shape `invalid_raw` already uses. A
/// checks-rejected `done` has no variant here: it is the pre-existing
/// rejection and has never been counted. Neither does a `finish(answer)` with
/// no `result` at all — that is a MALFORMED call, counted through
/// `invalid_raw` like every other malformed finish; only a `result` that was
/// supplied and FAILED validation lands on [`Self::AnswerSchema`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FinishRejection {
    /// A `done` claim whose checks were green but whose working tree was
    /// unchanged since the run started.
    NoChange,
    /// An `already_satisfied` claim rejected by a red gate.
    AlreadySatisfiedChecks,
    /// An `answer` claim whose `result` failed to validate against the
    /// configured [`AnswerSchema`]. A `finish(answer)` with NO `result` is
    /// not here — it is a malformed finish call and bumps
    /// `invalid_finish_calls`.
    AnswerSchema,
    /// A `finish(answer)` claim whose `result` validated but whose working
    /// tree changed since the run started — answer mode's INVERTED leg-3
    /// precondition. Counted by `RunStats::modified_workspace_rejections`.
    ModifiedWorkspace,
    /// A build-mode terminal disposition (`done` / `already_satisfied`)
    /// claimed on an answer-mode run. Bumps no counter — it is a steering
    /// rejection, not evidence about the workspace — but it IS surfaced on
    /// the transcript via [`Self::as_str`].
    WrongModeDisposition,
}

impl FinishRejection {
    /// The stable transcript label for this rejection — the value of the
    /// `tool_result` event's `finish_rejection` field.
    ///
    /// Hand-written rather than derived so the wire strings are pinned
    /// independently of the Rust variant names; a rename must not silently
    /// change a transcript key a jq audit query greps for.
    fn as_str(self) -> &'static str {
        match self {
            Self::NoChange => "no_change",
            Self::AlreadySatisfiedChecks => "already_satisfied_checks",
            Self::AnswerSchema => "schema_invalid",
            Self::ModifiedWorkspace => "modified_workspace",
            Self::WrongModeDisposition => "wrong_mode_disposition",
        }
    }
}

/// Which branch of the answer-mode gate a `finish(answer)` call took —
/// recorded on the transcript so a reader can tell "the model never supplied
/// a result" from "the model supplied one and it failed validation" from
/// "accepted", rather than only validity-vs-not.
#[derive(Debug, Clone, serde::Serialize)]
struct AnswerVerdict {
    /// `"missing_result"`, `"invalid"`, or `"valid"`.
    branch: &'static str,
    /// The SAME bounded error list the model was shown — empty on every
    /// branch except `"invalid"`.
    errors: Vec<String>,
}

/// One dispatched `finish` call's outcome from the loop's point of view: the
/// fed-back [`UserBlock::ToolResult`] to hand back to the model, and — when
/// the loop should terminate — the accepted [`Disposition`].
///
/// Extracted out of [`run`] so the main loop stays readable; keeps the
/// per-call plumbing (call id, `is_error`, content wording) in one place.
struct FinishOutcome {
    result: UserBlock,
    finish: Option<Disposition>,
    /// `Some(raw)` when this call was rejected as [`FinishClaim::Invalid`];
    /// `None` on every other arm (including a checks-rejected `done`).
    invalid_raw: Option<String>,
    /// The [`CheckReport`] `handle_finish_call` ran against `config.checks`,
    /// when it ran one — `Some` on the seven arms that ran the checks:
    /// accepted-`Done`-with-checks, checks-rejected-`done`,
    /// tree-unchanged-rejected-`done`-with-checks,
    /// accepted-`AlreadySatisfied`-with-checks,
    /// checks-rejected-`already_satisfied`, accepted-`Answer`-with-checks and
    /// modified-workspace-rejected-`Answer`-with-checks (both of which record
    /// a RED report too — see [`Verification`]). `None` on every no-checks
    /// path, on `blocked`/`failed`, on `Invalid`, on `MissingReason`, on both
    /// wrong-mode arms (an answer-mode `done` / `already_satisfied`), and on
    /// the missing-result / schema-invalid `Answer` arms — all of which return
    /// before running anything.
    /// Transcript-only: the `tool_result` event's `finish_verification` field
    /// is derived from this, never from `FinishClaim` internals.
    report: Option<CheckReport>,
    /// The leg-3 [`ChangeEvidence`] this call computed, when it observed the
    /// tree at all — `Some` on exactly the arms that reached the observation
    /// (accepted `Done`, tree-unchanged-rejected `done`, accepted
    /// `AlreadySatisfied`, accepted `Answer`, and
    /// modified-workspace-rejected `Answer`), `None` on
    /// `blocked`/`failed`/`Invalid`/`MissingReason`, on a checks-rejected
    /// claim, on a wrong-mode `done` / `already_satisfied`, and on a
    /// missing-result / schema-invalid `Answer` — all of which return before
    /// observing. Transcript-only.
    change: Option<ChangeEvidence>,
    /// The [`TreeObservation`] taken at the moment of the claim, paired with
    /// `change`. `Some` on exactly the same arms. Transcript-only.
    current_tree: Option<TreeObservation>,
    /// Which [`RunStats`] rejection counter this call should bump, if any.
    rejection: Option<FinishRejection>,
    /// Which answer-mode branch this call took — `Some` on exactly the four
    /// [`FinishClaim::Answer`] arms (missing-result, schema-invalid,
    /// modified-workspace-rejected, accepted), `None` on every other claim
    /// INCLUDING a build-mode `answer` that parsed as
    /// [`FinishClaim::Invalid`]. `branch` describes the SCHEMA verdict only,
    /// so a modified-workspace rejection still reports `"valid"` — read
    /// `rejection` (the transcript's `finish_rejection`) to see why an
    /// otherwise-valid answer was not accepted. Transcript-only.
    answer: Option<AnswerVerdict>,
}

impl FinishOutcome {
    /// An accepted claim: the standard acknowledgement plus the terminal
    /// `disposition`, with no checks run and no tree observed.
    fn accepted(call_id: &str, disposition: Disposition) -> Self {
        Self {
            result: ack(call_id),
            finish: Some(disposition),
            invalid_raw: None,
            report: None,
            change: None,
            current_tree: None,
            rejection: None,
            answer: None,
        }
    }

    /// A rejected claim fed back as an `is_error=true` tool result. The loop
    /// continues; `invalid_raw` and `rejection` select which [`RunStats`]
    /// counter (if any) the loop bumps.
    fn rejected(
        call_id: &str,
        content: String,
        invalid_raw: Option<String>,
        rejection: Option<FinishRejection>,
    ) -> Self {
        Self {
            result: UserBlock::ToolResult {
                call_id: call_id.to_string(),
                content,
                is_error: true,
            },
            finish: None,
            invalid_raw,
            report: None,
            change: None,
            current_tree: None,
            rejection,
            answer: None,
        }
    }
}

/// The outcome of running (or skipping) `config.checks` for a finish claim.
enum ChecksVerdict {
    /// The checks passed, or there were none to run.
    Green {
        report: Option<CheckReport>,
        verification: Verification,
    },
    /// The checks ran and came back red.
    Red(CheckReport),
}

/// Run `checks` for a finish claim, or record [`Verification::NoChecksConfigured`]
/// when the run has none wired in.
async fn verify_against_checks(checks: Option<&ChecksRunner>, ctx: &ToolCtx) -> ChecksVerdict {
    match checks {
        Some(runner) => {
            let report = runner.run(ctx).await;
            if report.passed {
                let verification = Verification::Checks(report.clone());
                ChecksVerdict::Green {
                    report: Some(report),
                    verification,
                }
            } else {
                ChecksVerdict::Red(report)
            }
        }
        None => ChecksVerdict::Green {
            report: None,
            verification: Verification::NoChecksConfigured,
        },
    }
}

/// The red-gate rejection for `claim`, carrying the report as evidence.
fn checks_rejected(
    call_id: &str,
    claim: &str,
    report: CheckReport,
    rejection: Option<FinishRejection>,
) -> FinishOutcome {
    FinishOutcome {
        result: UserBlock::ToolResult {
            call_id: call_id.to_string(),
            content: rejection_content(claim, &report),
            is_error: true,
        },
        finish: None,
        invalid_raw: None,
        report: Some(report),
        change: None,
        current_tree: None,
        rejection,
        answer: None,
    }
}

/// Handle a `finish` call: verify a `done` claim against `config.checks`
/// when configured, or accept it on trust when not — and, on a green verdict,
/// additionally require that the workspace actually moved since the run
/// started (leg 3 of the completion contract). `already_satisfied` is held to
/// the same checks but imposes no tree constraint. `blocked` and `failed`
/// terminate as declared with no verification. An unrecognized or missing
/// disposition ([`FinishClaim::Invalid`]) and an `already_satisfied` with no
/// `reason` ([`FinishClaim::MissingReason`]) are rejected back to the model as
/// an `is_error=true` tool result — `finish = None`, no checks run — and the
/// loop continues.
///
/// `answer_schema` is the answer-mode gate: `Some` advertises and accepts the
/// `answer` disposition, whose `result` must validate against it; `None`
/// (build mode) makes `answer` an unrecognized disposition handled entirely
/// by the pre-existing [`FinishClaim::Invalid`] arm. An accepted `answer`
/// validates FIRST (so an invalid answer never pays for a gate run), then
/// runs the checks — recording BOTH verdicts as [`Verification`] telemetry
/// without ever rejecting on a red one — then observes the tree and applies
/// the INVERTED leg-3 precondition.
///
/// **Answer mode inverts leg 3.** Build mode requires evidence that the
/// workspace MOVED; answer mode requires evidence that it did not. A
/// `finish(answer)` whose `result` validated but whose tree is
/// [`ChangeEvidence::TreeChanged`] is REJECTED (with the changed paths as
/// evidence) and the loop continues; `TreeUnchanged` is accepted;
/// `Unobservable` fails OPEN and is recorded. Symmetrically, `answer_schema
/// == Some` makes the BUILD-mode terminals (`done`, `already_satisfied`)
/// wrong-mode claims: both are rejected up front, before any checks run and
/// before the tree is observed, with steering toward `answer`.
///
/// Returns the fed-back [`UserBlock::ToolResult`] plus, when the loop should
/// terminate, the accepted [`Disposition`]. A rejected `done` returns
/// `finish = None`, an `is_error=true` result, and the loop continues.
#[allow(clippy::too_many_lines)]
async fn handle_finish_call(
    call_id: &str,
    input: &Value,
    checks: Option<&ChecksRunner>,
    answer_schema: Option<&AnswerSchema>,
    baseline: &TreeObservation,
    ctx: &ToolCtx,
) -> FinishOutcome {
    match FinishClaim::from_input(input, answer_schema.is_some()) {
        FinishClaim::Done { summary } => {
            // Answer mode rejects the build-mode terminals BEFORE anything
            // else: no checks are run, the tree is not observed, no counter
            // moves. `done` means "I changed the workspace", which is exactly
            // what an answer run must not have done — accepting it would
            // invert the guarantee the inverted precondition exists to make.
            if answer_schema.is_some() {
                return FinishOutcome::rejected(
                    call_id,
                    wrong_mode_rejection_content("done"),
                    None,
                    Some(FinishRejection::WrongModeDisposition),
                );
            }
            // Ordering is load-bearing and unchanged: the checks run FIRST,
            // and a red report still returns exactly the rejection it always
            // did. Only a green verdict — including the
            // no-checks-configured path, because "no gate configured" is not
            // a reason to accept work that did not happen — reaches the tree
            // precondition.
            let (report, verification) = match verify_against_checks(checks, ctx).await {
                ChecksVerdict::Red(report) => {
                    return checks_rejected(call_id, "done", report, None);
                }
                ChecksVerdict::Green {
                    report,
                    verification,
                } => (report, verification),
            };
            let current = exec::observe_tree(ctx.workspace().root(), TREE_OBSERVE_TIMEOUT).await;
            let change = exec::classify_change(baseline, &current);
            // The `Unobservable` arm is a DELIBERATE fail-open and must not
            // be "fixed" into a rejection: every existing engine / `eval.rs` /
            // `crates/talos/tests/cli.rs` test workspace is a non-git temp
            // directory, and — more importantly — a workspace the harness
            // cannot observe is no evidence that work did not happen.
            let accepted = change != ChangeEvidence::TreeUnchanged;
            FinishOutcome {
                result: if accepted {
                    ack(call_id)
                } else {
                    UserBlock::ToolResult {
                        call_id: call_id.to_string(),
                        content: no_change_rejection_content(),
                        is_error: true,
                    }
                },
                finish: accepted.then(|| Disposition::Done {
                    summary,
                    verification,
                    change: change.clone(),
                }),
                invalid_raw: None,
                report,
                change: Some(change),
                current_tree: Some(current),
                rejection: (!accepted).then_some(FinishRejection::NoChange),
                answer: None,
            }
        }
        FinishClaim::AlreadySatisfied { reason } => {
            // Same wrong-mode guard as `done` — see that arm. An answer run's
            // deliverable is the payload, so "nothing needed changing" is not
            // a terminal it can reach.
            if answer_schema.is_some() {
                return FinishOutcome::rejected(
                    call_id,
                    wrong_mode_rejection_content("already_satisfied"),
                    None,
                    Some(FinishRejection::WrongModeDisposition),
                );
            }
            // A no-op claim on a red repo is never legitimate, so the checks
            // still gate it. The tree does NOT: a `TreeChanged` observation
            // is RECORDED, not rejected — rejecting it would trap an agent
            // that wrote an incidental scratch file, and the run is
            // non-pushable either way.
            let (report, verification) = match verify_against_checks(checks, ctx).await {
                ChecksVerdict::Red(report) => {
                    return checks_rejected(
                        call_id,
                        "already_satisfied",
                        report,
                        Some(FinishRejection::AlreadySatisfiedChecks),
                    );
                }
                ChecksVerdict::Green {
                    report,
                    verification,
                } => (report, verification),
            };
            let current = exec::observe_tree(ctx.workspace().root(), TREE_OBSERVE_TIMEOUT).await;
            let change = exec::classify_change(baseline, &current);
            FinishOutcome {
                result: ack(call_id),
                finish: Some(Disposition::AlreadySatisfied {
                    reason,
                    verification,
                    change: change.clone(),
                }),
                invalid_raw: None,
                report,
                change: Some(change),
                current_tree: Some(current),
                rejection: None,
                answer: None,
            }
        }
        FinishClaim::Answer { result } => {
            // `answer_enabled` was `answer_schema.is_some()`, so this arm is
            // unreachable without a schema — but express it as a `let else`
            // rather than an `expect`, so a future parser change degrades to
            // the inherited invalid-disposition rejection instead of panicking.
            let Some(schema) = answer_schema else {
                return FinishOutcome::rejected(
                    call_id,
                    missing_result_rejection_content(),
                    Some("answer without result".to_string()),
                    None,
                );
            };
            let Some(result) = result else {
                let mut outcome = FinishOutcome::rejected(
                    call_id,
                    missing_result_rejection_content(),
                    Some("answer without result".to_string()),
                    None,
                );
                outcome.answer = Some(AnswerVerdict {
                    branch: "missing_result",
                    errors: Vec::new(),
                });
                return outcome;
            };
            // Ordering is load-bearing: the SCHEMA is answer mode's
            // mechanical verifier, so it runs first and an invalid answer
            // never pays for a gate run.
            let errors = schema.validation_errors(&result);
            if !errors.is_empty() {
                let shown: Vec<String> = errors
                    .iter()
                    .take(ANSWER_SCHEMA_ERRORS_MAX_LINES)
                    .cloned()
                    .collect();
                let mut outcome = FinishOutcome::rejected(
                    call_id,
                    answer_schema_rejection_content(&errors),
                    None,
                    Some(FinishRejection::AnswerSchema),
                );
                outcome.answer = Some(AnswerVerdict {
                    branch: "invalid",
                    errors: shown,
                });
                return outcome;
            }
            // Checks in answer mode: RUN if configured, NEVER reject. For
            // `Answer` the mechanical verifier is the SCHEMA, not the gate
            // (docs/design/06-answer-mode-and-workflows.md:23 — "for answer
            // mode leg 3 is the payload"). The gate describes the WORKSPACE,
            // which an answering agent did not author, so a red gate is
            // inherited state, not evidence against the answer — and
            // rejecting on it would trap an answer agent in an unfixable loop
            // to the iteration cap. Both verdicts therefore map to a
            // `Verification` that is RECORDED; `checks_rejected` is never
            // called here.
            let (report, verification) = match verify_against_checks(checks, ctx).await {
                ChecksVerdict::Green {
                    report,
                    verification,
                } => (report, verification),
                ChecksVerdict::Red(report) => (Some(report.clone()), Verification::Checks(report)),
            };
            // The INVERTED leg-3 precondition, and the LAST gate in the
            // arm: schema first (item 1), then checks (item 1), then this.
            // The ordering matters — an invalid `result` on a changed tree
            // gets the schema rejection, which is the more actionable of the
            // two, and never pays for a tree observation.
            //
            // Build mode requires evidence that work HAPPENED; answer mode
            // requires evidence that it did NOT. `TreeChanged` is therefore
            // rejected and the loop continues. `Unobservable` fails OPEN —
            // deliberately, and for the same reason build mode does: a
            // workspace the harness cannot observe is no evidence that the
            // agent modified it, and every non-git temp workspace in the test
            // suite would otherwise become unanswerable. The fail-open is
            // RECORDED (`RunStats::tree_baseline_unobservable`, and the
            // `Unobservable` evidence on the accepted `Disposition::Answer`
            // itself) so a caller can tell a verified read-only answer from
            // an unverifiable one.
            let current = exec::observe_tree(ctx.workspace().root(), TREE_OBSERVE_TIMEOUT).await;
            let change = exec::classify_change(baseline, &current);
            if change == ChangeEvidence::TreeChanged {
                return FinishOutcome {
                    result: UserBlock::ToolResult {
                        call_id: call_id.to_string(),
                        content: modified_workspace_rejection_content(&current),
                        is_error: true,
                    },
                    finish: None,
                    invalid_raw: None,
                    report,
                    change: Some(change),
                    current_tree: Some(current),
                    rejection: Some(FinishRejection::ModifiedWorkspace),
                    answer: Some(AnswerVerdict {
                        branch: "valid",
                        errors: Vec::new(),
                    }),
                };
            }
            FinishOutcome {
                result: ack(call_id),
                finish: Some(Disposition::Answer {
                    result,
                    verification,
                    change: change.clone(),
                }),
                invalid_raw: None,
                report,
                change: Some(change),
                current_tree: Some(current),
                rejection: None,
                answer: Some(AnswerVerdict {
                    branch: "valid",
                    errors: Vec::new(),
                }),
            }
        }
        FinishClaim::Blocked { decision_needed } => {
            FinishOutcome::accepted(call_id, Disposition::Blocked { decision_needed })
        }
        FinishClaim::Failed { summary } => FinishOutcome::accepted(
            call_id,
            Disposition::Failed {
                mode: FailureMode::Loop,
                summary,
            },
        ),
        FinishClaim::Invalid { raw } => FinishOutcome::rejected(
            call_id,
            format!(
                "finish rejected: disposition must be one of: done, blocked, failed, \
                 already_satisfied; got {raw}. Call finish again with one of those values."
            ),
            Some(raw),
            None,
        ),
        FinishClaim::MissingReason => FinishOutcome::rejected(
            call_id,
            missing_reason_rejection_content(),
            Some("already_satisfied without reason".to_string()),
            None,
        ),
    }
}

/// The standard `finish acknowledged` fed-back [`UserBlock::ToolResult`] the
/// loop hands the model for an accepted `finish` — for an accepted
/// `done` / `already_satisfied` / `answer` / `blocked` / `failed` claim (or a
/// `done` with no checks configured). Kept factored so the wording matches
/// exactly across the five accepted paths.
fn ack(call_id: &str) -> UserBlock {
    UserBlock::ToolResult {
        call_id: call_id.to_string(),
        content: "finish acknowledged".to_string(),
        is_error: false,
    }
}

/// Bound on a rendered [`TreeObservation`]'s `porcelain`, in characters.
/// RENDERING ONLY — the in-memory value [`exec::classify_change`] compares is
/// never truncated.
const TREE_PORCELAIN_RENDER_CAP: usize = 4_000;

/// Serialize a [`TreeObservation`] for the transcript, capping `porcelain` at
/// [`TREE_PORCELAIN_RENDER_CAP`] characters and emitting the untruncated
/// character count alongside it so a reader can tell the excerpt is partial.
fn render_tree_observation(obs: &TreeObservation) -> Value {
    match obs {
        TreeObservation::Observed { porcelain, head } => json!({
            "Observed": {
                "porcelain": exec::tail(porcelain, TREE_PORCELAIN_RENDER_CAP),
                "porcelain_chars": porcelain.chars().count(),
                "head": head,
            }
        }),
        TreeObservation::Unobservable { reason } => json!({
            "Unobservable": { "reason": reason }
        }),
    }
}

/// The tripwire text appended to the run-start "tree observation unavailable"
/// warning, naming the precondition that just went INERT.
///
/// Mode-aware because the two modes enforce OPPOSITE invariants off the same
/// observation: build mode requires the tree to have CHANGED before it will
/// accept a `done`, answer mode requires it to be UNCHANGED before it will
/// accept an `answer`. A warning that named the wrong one would send the next
/// reader of the log hunting the wrong invariant.
///
/// Pure and `&'static str`-returning so both branches are unit-testable
/// without capturing stderr.
fn inert_precondition_warning(answer_mode: bool) -> &'static str {
    if answer_mode {
        "the answer-requires-unchanged-tree precondition is INERT for this run"
    } else {
        "the done-requires-change precondition is INERT for this run"
    }
}

/// Drive `backend` + `tools` through a conversation until the agent finishes
/// or `config.max_iterations` is hit.
///
/// The system prompt and task seed are rendered ONCE via the [`crate::prompt`]
/// layer before the loop starts and then reused verbatim every iteration —
/// this is the prompt-cache correctness invariant, and the reason the caller
/// hands in a [`RunConfig`] rather than a pre-rendered `&str` system prompt.
///
/// A `finish(done)` claim is verified via [`ChecksRunner::run`] when
/// `config.checks` is `Some`; see the module docs for the full claim-vs-verify
/// contract.
///
/// The returned [`RunResult`] pairs the terminal [`LoopOutcome`] with the
/// mechanical [`RunStats`] accumulated over the run — see [`RunStats`] for
/// how each field is counted (in particular: an erroring last turn still
/// contributes to `iterations` but not to the token totals).
///
/// **No persistence:** this entry point passes `None` for the store, so no
/// events are appended and no checkpoints are written. Use [`run_persisted`]
/// when durability is needed.
///
/// # Panics
///
/// This function is infallible in practice, but internally calls `.expect()` on
/// a `Result` that is structurally `Ok` when no persistence is wired in (no
/// store calls are made). If that invariant were violated, the function would
/// panic with a diagnostic message.
///
/// [`ChecksRunner::run`]: crate::exec::ChecksRunner::run
pub async fn run(
    backend: &impl model::ModelBackend,
    tools: &ToolRegistry,
    ctx: &ToolCtx,
    config: &RunConfig,
) -> RunResult {
    // `wall_clock` covers the whole call (prompt rendering included, since
    // that's real work the loop did). `stats` is threaded into the loop by
    // mut-ref so every termination path picks up the same in-progress totals.
    let start = Instant::now();
    let mut stats = RunStats {
        iterations: 0,
        input_tokens: 0,
        output_tokens: 0,
        wall_clock: Duration::ZERO,
        gates_green_at_exit: false,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        nudges_fired: 0,
        tree_dirty: false,
        iters_since_tree_change_at_exit: 0,
        peak_iters_since_tree_change: 0,
        mutating_iters: 0,
        bash_calls_ok: 0,
        edit_file_calls_ok: 0,
        invalid_finish_calls: 0,
        first_invalid_finish_raw: None,
        no_change_rejections: 0,
        already_satisfied_check_rejections: 0,
        answer_schema_rejections: 0,
        modified_workspace_rejections: 0,
        tree_baseline_unobservable: false,
    };
    let task_message = prompt::render_task_prompt(&config.task);
    let initial_messages = vec![Message::User {
        content: vec![UserBlock::Text(task_message)],
    }];
    // No persistence — the Result::Err path is structurally unreachable when
    // persistence is None (no store calls are made), so the expect is a
    // compile-time invariant, not a runtime safety net.
    let outcome = run_loop_impl(
        backend,
        tools,
        ctx,
        config,
        None,
        &mut stats,
        initial_messages,
        BudgetConsumed::default(),
        None,
        None,
    )
    .await
    .expect("no-persistence run cannot produce a StoreError");
    stats.wall_clock = start.elapsed();
    RunResult { outcome, stats }
}

/// Drive `backend` + `tools` with full durability: append events and write
/// checkpoints to `persistence.store` as the loop progresses.
///
/// ## Persistence discipline
///
/// - A [`RunRecord`] is constructed at run start and kept current throughout.
/// - After each successful model turn: [`Event::ModelCall`] then
///   [`Event::BudgetTick`] are appended, followed by a mid-iteration
///   checkpoint (snapshot includes the assistant turn in `messages`).
/// - For each non-`finish` tool call: [`Event::ToolCallStarted`] (before
///   execution) and [`Event::ToolCallResult`] (after execution) are appended.
/// - At end of each loop iteration: a full checkpoint is written.
/// - On every terminal path: [`Event::DispositionSet`] is appended, then a
///   final checkpoint is written with `disposition` set.
///
/// The first [`StoreError`] from any append or checkpoint immediately aborts
/// the loop and is returned as `Err`. The no-store [`run`] path keeps its
/// bare `RunResult` return type unchanged.
pub async fn run_persisted(
    backend: &impl model::ModelBackend,
    tools: &ToolRegistry,
    ctx: &ToolCtx,
    config: &RunConfig,
    persistence: &Persistence,
) -> Result<RunResult, StoreError> {
    let start = Instant::now();
    let mut stats = RunStats {
        iterations: 0,
        input_tokens: 0,
        output_tokens: 0,
        wall_clock: Duration::ZERO,
        gates_green_at_exit: false,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        nudges_fired: 0,
        tree_dirty: false,
        iters_since_tree_change_at_exit: 0,
        peak_iters_since_tree_change: 0,
        mutating_iters: 0,
        bash_calls_ok: 0,
        edit_file_calls_ok: 0,
        invalid_finish_calls: 0,
        first_invalid_finish_raw: None,
        no_change_rejections: 0,
        already_satisfied_check_rejections: 0,
        answer_schema_rejections: 0,
        modified_workspace_rejections: 0,
        tree_baseline_unobservable: false,
    };
    let task_message = prompt::render_task_prompt(&config.task);
    let initial_messages = vec![Message::User {
        content: vec![UserBlock::Text(task_message)],
    }];
    let outcome = run_loop_impl(
        backend,
        tools,
        ctx,
        config,
        Some(persistence),
        &mut stats,
        initial_messages,
        BudgetConsumed::default(),
        None,
        None,
    )
    .await?;
    stats.wall_clock = start.elapsed();
    Ok(RunResult { outcome, stats })
}

/// The engine loop body, shared by [`run`], [`run_persisted`], and [`resume`].
///
/// `stats` is mutated in place as the loop progresses:
/// - `iterations` is incremented once per `for`-loop iteration, BEFORE the
///   first (possibly-retried) `backend.turn` call of that iteration. A turn
///   that fails transiently and is retried within the same pass still counts
///   as ONE logical iteration. A backend error on the first iteration yields
///   `iterations = 1`.
/// - `input_tokens` / `output_tokens` accumulate the SUCCESSFUL turn's
///   [`crate::model::Usage`] only; errored turns contribute nothing.
///
/// `initial_messages` is the starting conversation history (task seed for
/// fresh runs; reloaded/reconciled messages for crash-resume; fresh task seed
/// for fresh-context resume).
///
/// `initial_consumed` offsets all `budgets.consumed` computations — zero for
/// fresh runs; the loaded record's consumed for resume (budget carry-over,
/// accounting-only in 0.3.0).
///
/// `override_persist` when `Some` bypasses the record-construction step and
/// uses the provided [`RunPersist`] directly — used by [`resume`] to inject
/// a pre-loaded (and possibly reconciled) record.
///
/// When `persistence` is `Some`, the loop appends events and writes
/// checkpoints per the durability contract documented on [`run_persisted`].
/// When `persistence` is `None`, no store calls are made and the function
/// returns `Ok(outcome)` (the `Err` arm is structurally unreachable).
///
/// **Transcript:** a thin wrapper around [`run_loop_body`] — it opens the
/// [`TranscriptWriter`] from `config.transcript`, awaits the body, emits the
/// `run_end` event from the body's `Result` (`Ok` and `Err` alike) using the
/// post-loop `stats`, and returns that `Result` unchanged. No transcript
/// error is ever propagated.
#[allow(clippy::too_many_arguments)]
async fn run_loop_impl(
    backend: &impl model::ModelBackend,
    tools: &ToolRegistry,
    ctx: &ToolCtx,
    config: &RunConfig,
    persistence: Option<&Persistence>,
    stats: &mut RunStats,
    initial_messages: Vec<Message>,
    initial_consumed: BudgetConsumed,
    override_persist: Option<RunPersist>,
    baseline_override: Option<TreeObservation>,
) -> Result<LoopOutcome, StoreError> {
    let mut writer = TranscriptWriter::open(config.transcript.as_ref());
    // Mirrors `run_loop_body`'s own rid derivation (override first, else
    // derived from the persistence handle) so the `contract_violation` audit
    // can name the run without threading state back out of the body.
    let run_id_hint = override_persist
        .as_ref()
        .map(|p| p.rid.clone())
        .or_else(|| persistence.map(|p| run_id(&p.task_id, p.attempt_n)));
    let result = run_loop_body(
        backend,
        tools,
        ctx,
        config,
        persistence,
        stats,
        initial_messages,
        initial_consumed,
        override_persist,
        baseline_override,
        &mut writer,
    )
    .await;
    emit_run_end(
        &mut writer,
        &result,
        stats,
        run_id_hint.as_deref(),
        config.answer_schema.as_ref(),
    );
    result
}

/// Build and emit the `run_end` transcript event from `result` (the
/// [`run_loop_body`] return value) and the post-loop `stats`. A no-op when
/// the writer is disabled — no `serde_json::json!` payload is built in that
/// case.
fn emit_run_end(
    writer: &mut TranscriptWriter,
    result: &Result<LoopOutcome, StoreError>,
    stats: &RunStats,
    run_id: Option<&str>,
    answer_schema: Option<&AnswerSchema>,
) {
    // The cheap runtime audit of the leg-3 invariant. Unlike `Verification`,
    // `Disposition::Done` is a public struct variant with a public `change`
    // field constructed at many sites, so "unconstructible without observed
    // change" is enforced by one code path plus convention — not by the type
    // system. This is the single choke point every terminal passes through,
    // so a regression that lets a `TreeUnchanged` `Done` escape shows up here
    // rather than silently reaching a push.
    if let Ok(LoopOutcome::Finished(Disposition::Done {
        change: change @ ChangeEvidence::TreeUnchanged,
        ..
    })) = result
    {
        eprintln!(
            "warning: contract violation — a Done disposition reached the terminal with an \
             unchanged working tree; the done-requires-change precondition did not hold"
        );
        writer.emit(
            "contract_violation",
            json!({
                "kind": "done_with_unchanged_tree",
                "run_id": run_id,
                "change": serde_json::to_value(change).unwrap_or(Value::Null),
            }),
        );
    }
    // The `Answer` counterpart of the same tripwire. `Disposition::Answer` is
    // a public struct variant with a public `result` field constructible
    // anywhere, so "an Answer only reaches the terminal when a schema was
    // configured AND its result validated" is enforced by one code path plus
    // convention. Re-check it at the single choke point every terminal
    // passes through.
    if let Ok(LoopOutcome::Finished(Disposition::Answer { result, .. })) = result {
        match answer_schema {
            None => {
                eprintln!(
                    "warning: contract violation — an Answer disposition reached the terminal \
                     on a run with no answer schema configured; the schema-validated-result \
                     precondition did not hold"
                );
                writer.emit(
                    "contract_violation",
                    json!({
                        "kind": "answer_without_schema",
                        "run_id": run_id,
                        "errors": Vec::<String>::new(),
                    }),
                );
            }
            Some(schema) => {
                let errors = schema.validation_errors(result);
                if !errors.is_empty() {
                    eprintln!(
                        "warning: contract violation — an Answer disposition reached the \
                         terminal carrying a result that does not validate against the \
                         configured schema; the schema-validated-result precondition did not \
                         hold"
                    );
                    writer.emit(
                        "contract_violation",
                        json!({
                            "kind": "answer_result_fails_schema",
                            "run_id": run_id,
                            "errors": errors,
                        }),
                    );
                }
            }
        }
    }
    if !writer.is_enabled() {
        return;
    }
    let (outcome_str, disposition, detail) = match result {
        Ok(LoopOutcome::Finished(disposition)) => (
            "Finished",
            Some(serde_json::to_value(disposition).unwrap_or(Value::Null)),
            None,
        ),
        Ok(LoopOutcome::StoppedWithoutFinish) => ("StoppedWithoutFinish", None, None),
        Ok(LoopOutcome::MaxIterations) => ("MaxIterations", None, None),
        Ok(LoopOutcome::BudgetExhausted { summary }) => {
            ("BudgetExhausted", None, Some(summary.clone()))
        }
        Ok(LoopOutcome::BackendError(err)) => ("BackendError", None, Some(format!("{err}"))),
        Err(err) => ("StoreError", None, Some(err.to_string())),
    };
    writer.emit(
        "run_end",
        json!({
            "outcome": outcome_str,
            "disposition": disposition,
            "detail": detail,
            "stats": render_run_end_stats(stats),
        }),
    );
}

/// Serialize [`RunStats`] for the `run_end` event.
///
/// `wall_clock` is intentionally omitted — the caller
/// (`run`/`run_persisted`/`resume`) sets it only AFTER `run_loop_impl` (and
/// therefore this event) returns, so it would always be zero here. Every
/// other field is emitted, and the transcript module doc enumerates them.
fn render_run_end_stats(stats: &RunStats) -> Value {
    json!({
        "iterations": stats.iterations,
        "input_tokens": stats.input_tokens,
        "output_tokens": stats.output_tokens,
        "cache_read_tokens": stats.cache_read_tokens,
        "cache_write_tokens": stats.cache_write_tokens,
        "gates_green_at_exit": stats.gates_green_at_exit,
        "nudges_fired": stats.nudges_fired,
        "tree_dirty": stats.tree_dirty,
        "iters_since_tree_change_at_exit": stats.iters_since_tree_change_at_exit,
        "peak_iters_since_tree_change": stats.peak_iters_since_tree_change,
        "mutating_iters": stats.mutating_iters,
        "bash_calls_ok": stats.bash_calls_ok,
        "edit_file_calls_ok": stats.edit_file_calls_ok,
        "no_change_rejections": stats.no_change_rejections,
        "already_satisfied_check_rejections": stats.already_satisfied_check_rejections,
        "answer_schema_rejections": stats.answer_schema_rejections,
        "modified_workspace_rejections": stats.modified_workspace_rejections,
        "tree_baseline_unobservable": stats.tree_baseline_unobservable,
    })
}

/// The engine loop body proper — the renamed former `run_loop_impl`, now
/// taking an extra `writer` so it can emit per-iteration transcript events.
/// See [`run_loop_impl`] for the wrapper that opens `writer` and emits
/// `run_end`.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
async fn run_loop_body(
    backend: &impl model::ModelBackend,
    tools: &ToolRegistry,
    ctx: &ToolCtx,
    config: &RunConfig,
    persistence: Option<&Persistence>,
    stats: &mut RunStats,
    initial_messages: Vec<Message>,
    initial_consumed: BudgetConsumed,
    override_persist: Option<RunPersist>,
    baseline_override: Option<TreeObservation>,
    writer: &mut TranscriptWriter,
) -> Result<LoopOutcome, StoreError> {
    // Capture the loop-start instant ONCE via the injected clock. Used both
    // for the `wall_clock_start` string on the run record and for the
    // per-iteration wall-clock breach check. All "now" reads inside this
    // function go through `config.clock` — never via `SystemTime::now()`
    // directly — so tests can drive time with a FakeClock.
    let loop_start = config.clock.now();

    // Answer mode IS `config.answer_schema.is_some()` — there is deliberately
    // no second mode field to drift out of sync with it.
    let answer_mode = config.answer_schema.is_some();

    // Render the system prompt ONCE and reuse verbatim every iteration —
    // the prompt-cache correctness invariant (D9). Answer mode renders its
    // OWN template (`answer_system_prompt.md`): a read-only analyst frame
    // with no verification section and only `answer`/`blocked`/`failed` as
    // terminals. `system_prompt.md` is left byte-identical, so the
    // tier-1/tier-2 eval-parity rule gains no new surface.
    let system = if answer_mode {
        prompt::render_answer_system_prompt(&prompt::tool_lines(tools))
    } else {
        prompt::render_system_prompt(
            &prompt::tool_lines(tools),
            config
                .checks
                .as_ref()
                .map(ChecksRunner::command_display)
                .as_deref(),
        )
    };

    let mut messages = initial_messages;
    let tool_schemas = tools.list();
    let params = SamplingParams {
        max_tokens: config.max_tokens,
        temperature: None,
        stop_sequences: Vec::new(),
    };

    // Captured before `override_persist` is moved into the match below —
    // `resume` is true for both ResumeMode::Crash and ResumeMode::FreshContext
    // (both go through `resume`, which always supplies `Some(pre_persist)`).
    let is_resume = override_persist.is_some();

    // Build the run-record if persistence is configured. This is kept as
    // Option<RunPersist> so the non-persistent path has zero overhead.
    // When override_persist is Some (resume path), use it directly — the
    // caller (resume) already loaded/constructed the record.
    let mut persist: Option<RunPersist> = if let Some(pre) = override_persist {
        Some(pre)
    } else if let Some(p) = persistence {
        let rid = run_id(&p.task_id, p.attempt_n);
        let wall_clock_start = format_rfc3339(loop_start);

        // run_checks: a single "checks" entry when a ChecksRunner is wired in,
        // empty BTreeMap otherwise (D7 honesty: only what we actually know).
        let run_checks = match &config.checks {
            None => BTreeMap::new(),
            Some(runner) => {
                let mut m = BTreeMap::new();
                m.insert("checks".to_string(), runner.command_display());
                m
            }
        };

        let record = RunRecord {
            run_id: rid.clone(),
            schema_version: SCHEMA_VERSION,
            attempt_n: p.attempt_n,
            task: Task {
                task_id: p.task_id.clone(),
                title: String::new(),
                description: config.task.clone(),
                acceptance_criteria: vec![],
                files_in_scope: vec![],
                scope_out: vec![],
            },
            project_config: ProjectConfig {
                run_checks,
                model_routing_hint: None,
            },
            phase: Phase::InnerLoop,
            durable_facts: DurableFacts::default(),
            budgets: Budgets {
                consumed: BudgetConsumed::default(),
                limits: BudgetLimits {
                    iterations: config.max_iterations,
                    tokens: 0,
                    cost_micros: 0,
                    wall_clock_secs: config.wall_clock_secs,
                },
                wall_clock_start,
            },
            last_gate_result: None,
            disposition: None,
            recovery_facts: None,
            backend_settings: p.backend_settings.clone(),
            messages: messages.clone(),
        };
        Some(RunPersist { rid, record })
    } else {
        None
    };

    // ---- leg-3 baseline: observed ONCE per loop invocation ----
    // `baseline_override` is `Some` only on the resume path, which supplies
    // its own `Unobservable` baseline (see `resume`).
    let tree_baseline = match baseline_override {
        Some(obs) => obs,
        None => exec::observe_tree(ctx.workspace().root(), TREE_OBSERVE_TIMEOUT).await,
    };
    if let TreeObservation::Unobservable { reason } = &tree_baseline {
        stats.tree_baseline_unobservable = true;
        eprintln!(
            "warning: tree observation unavailable ({reason}) — {}",
            inert_precondition_warning(answer_mode)
        );
    }

    // `run_start` — emitted once per invocation, now that the run id (if any)
    // is known and before the first iteration. All of `system`/`tool_schemas`/
    // `messages` are only serialized when the writer is enabled.
    if writer.is_enabled() {
        writer.emit(
            "run_start",
            json!({
                "transcript_version": crate::transcript::TRANSCRIPT_VERSION,
                "harness_version": env!("CARGO_PKG_VERSION"),
                "label": config.transcript.as_ref().map(|t| t.label.clone()),
                "run_id": persist.as_ref().map(|p| p.rid.clone()),
                // The SAME value stamped on the [`RunRecord`] (single source)
                // — `null` for the non-persisted `engine::run` path.
                "backend_settings": persist
                    .as_ref()
                    .and_then(|p| p.record.backend_settings.as_ref())
                    .map(|s| serde_json::to_value(s).unwrap_or(Value::Null)),
                "resume": is_resume,
                "tree_baseline": render_tree_observation(&tree_baseline),
                "system": system,
                "tools": Value::Array(tool_schemas.clone()),
                "messages": serde_json::to_value(&messages).unwrap_or(Value::Null),
                "config": {
                    "max_iterations": config.max_iterations,
                    "max_tokens": config.max_tokens,
                    "mode": if answer_mode { "answer" } else { "build" },
                    "checks": config.checks.as_ref().map(ChecksRunner::command_display),
                    "answer_schema": config.answer_schema.as_ref().map(AnswerSchema::source),
                    "wall_clock_secs": config.wall_clock_secs,
                    "static_tree_k": config.static_tree_k,
                    "max_nudges": config.max_nudges,
                    "max_retries": config.max_retries,
                },
            }),
        );
    }

    // ---- finish-recovery detection state (loop-local) ----
    // The done-oracle is `last_gate_green`, driven ONLY by `run_checks`'s
    // `is_error` flag (never a model self-report). A successful mutating tool
    // call (`edit_file`/`bash` with `!is_error`) invalidates the green — this
    // closes the stale-green false-trip window so a nudge's "gates are
    // currently green" is always true at trip time. `iters_since_tree_change`
    // counts consecutive non-mutating iterations; reset wins over increment
    // when a mutation and the per-iteration tick collide. `nudges_fired`
    // bounds how many times the harness will nudge before force-terminating;
    // `max_nudges == 0` disables the feature entirely (no nudge is ever
    // injected and the recovery terminal is never taken). `nudge_awaiting_status`
    // gates telemetry capture of the assistant reply text from the turn that
    // follows a nudge without producing an accepted `finish(done)`.
    let mut last_gate_green: bool = false;
    let mut iters_since_tree_change: u32 = 0;
    let mut tree_dirty: bool = false;
    let mut nudges_fired: u32 = 0;
    let mut nudge_awaiting_status: bool = false;
    let mut nudge_statuses: Vec<String> = Vec::new();

    for _ in 0..config.max_iterations {
        let req = TurnRequest {
            system: Some(&system),
            messages: &messages,
            tools: &tool_schemas,
            params: &params,
        };

        // Count the logical iteration BEFORE the retry loop so an error on
        // the first iteration still shows `iterations = 1`. A transient error
        // retried within the same for-loop pass counts as ONE logical
        // iteration — `stats.iterations` is NOT re-incremented per retry.
        stats.iterations += 1;
        if writer.is_enabled() {
            let block_count: usize = messages
                .iter()
                .map(|m| match m {
                    Message::User { content } => content.len(),
                    Message::Assistant { content } => content.len(),
                })
                .sum();
            writer.emit(
                "model_request",
                json!({
                    "iteration": stats.iterations,
                    "message_count": messages.len(),
                    "block_count": block_count,
                }),
            );
        }
        let mut attempt = 0u32;
        // `(AssistantTurn, attempts_made, latency_of_the_successful_call)` on
        // success — `attempts_made` is `attempt`'s value AT THE TIME of the
        // successful call, i.e. the number of PRIOR failed attempts.
        let turn_result = loop {
            let call_start = Instant::now();
            let call_result = backend.turn(&req).await;
            let call_latency = call_start.elapsed();
            match call_result {
                Ok(turn) => break Ok((turn, attempt, call_latency)),
                Err(err) => {
                    let retryable = err.is_retryable();
                    let will_retry = retryable && attempt < config.max_retries;
                    if writer.is_enabled() {
                        let retry_delay_ms = if will_retry {
                            Some(
                                u64::try_from(
                                    retry_delay(config.retry_backoff_base, attempt).as_millis(),
                                )
                                .unwrap_or(u64::MAX),
                            )
                        } else {
                            None
                        };
                        writer.emit(
                            "backend_error",
                            json!({
                                "iteration": stats.iterations,
                                "attempt": attempt,
                                "retryable": retryable,
                                "will_retry": will_retry,
                                "error": format!("{err}"),
                                "error_debug": format!("{err:?}"),
                                "latency_ms": u64::try_from(call_latency.as_millis())
                                    .unwrap_or(u64::MAX),
                                "retry_delay_ms": retry_delay_ms,
                            }),
                        );
                    }
                    if will_retry {
                        sleep(retry_delay(config.retry_backoff_base, attempt)).await;
                        attempt += 1;
                    } else {
                        break Err(err);
                    }
                }
            }
        };
        let (turn, attempts_made, success_latency) = match turn_result {
            Ok((turn, prior_attempts, latency)) => (turn, prior_attempts + 1, latency),
            Err(err) => {
                // Terminal path: BackendError (retries exhausted or
                // non-retryable). Persist the disposition before returning —
                // this path exits early, bypassing the normal end-of-iteration
                // checkpoint. `err` is the LAST attempt's error.
                if let (Some(ctx), Some(p)) = (persist.as_mut(), persistence) {
                    let mode = if err.is_retryable() {
                        FailureMode::TransientInfra
                    } else {
                        FailureMode::PersistentToolError
                    };
                    let disposition = Disposition::Failed {
                        mode,
                        summary: format!("backend error: {err:?}"),
                    };
                    ctx.record.budgets.consumed = BudgetConsumed {
                        iterations: initial_consumed.iterations + stats.iterations,
                        tokens: initial_consumed.tokens + stats.input_tokens + stats.output_tokens,
                        cost_micros: initial_consumed.cost_micros,
                    };
                    ctx.record.messages.clone_from(&messages);
                    ctx.record.disposition = Some(disposition.clone());
                    p.store
                        .append_event(
                            &ctx.rid,
                            Event::DispositionSet {
                                seq: 0,
                                disposition,
                            },
                        )
                        .await?;
                    p.store.checkpoint(&ctx.rid, &ctx.record).await?;
                }
                return Ok(LoopOutcome::BackendError(err));
            }
        };

        // Successful turn — capture per-turn usage BEFORE moving the turn
        // into history (Message::from consumes it).
        let per_turn_input = u64::from(turn.usage.input_tokens);
        let per_turn_output = u64::from(turn.usage.output_tokens);
        // Cache tokens are `Option<u32>`: `None` means "this provider didn't
        // report it", treated as 0 for accumulation (a provider that never
        // reports cache tokens — Ollama today — simply contributes 0).
        let per_turn_cache_read = u64::from(turn.usage.cache_read_tokens.unwrap_or(0));
        let per_turn_cache_write = u64::from(turn.usage.cache_write_tokens.unwrap_or(0));

        // Accumulate into run totals. Per-turn u32 values sum into u64 so a
        // long run can't overflow.
        stats.input_tokens += per_turn_input;
        stats.output_tokens += per_turn_output;
        stats.cache_read_tokens += per_turn_cache_read;
        stats.cache_write_tokens += per_turn_cache_write;

        // Snapshot the calls before moving the turn into history (the `From`
        // impl consumes `turn.content`).
        let calls: Vec<_> = turn.tool_calls().into_iter().cloned().collect();
        // Capture the assistant reply text BEFORE `Message::from(turn)`
        // consumes the turn. `AssistantTurn::text` concatenates every
        // `ContentBlock::Text` in order (skipping Reasoning/ToolCall) — used
        // for nudge-status telemetry if this turn follows a nudge without
        // producing an accepted finish(done).
        let turn_text = turn.text();
        if writer.is_enabled() {
            writer.emit(
                "model_response",
                json!({
                    "iteration": stats.iterations,
                    "attempts": attempts_made,
                    "latency_ms": u64::try_from(success_latency.as_millis()).unwrap_or(u64::MAX),
                    "stop_reason": serde_json::to_value(&turn.stop_reason).unwrap_or(Value::Null),
                    "usage": serde_json::to_value(turn.usage).unwrap_or(Value::Null),
                    "content": serde_json::to_value(&turn.content).unwrap_or(Value::Null),
                }),
            );
        }
        messages.push(Message::from(turn));

        // Append ModelCall + BudgetTick events, then write the mid-iteration
        // checkpoint (after assistant turn, BEFORE any tools.invoke). This
        // guarantees that a mid-iteration crash always leaves a snapshot whose
        // messages include the in-flight assistant turn.
        if let (Some(ctx), Some(p)) = (persist.as_mut(), persistence) {
            let consumed = BudgetConsumed {
                iterations: initial_consumed.iterations + stats.iterations,
                tokens: initial_consumed.tokens + stats.input_tokens + stats.output_tokens,
                cost_micros: initial_consumed.cost_micros,
            };
            p.store
                .append_event(
                    &ctx.rid,
                    Event::ModelCall {
                        seq: 0,
                        model: p.model_label.clone(),
                        prompt_tokens: per_turn_input,
                        completion_tokens: per_turn_output,
                    },
                )
                .await?;
            p.store
                .append_event(&ctx.rid, Event::BudgetTick { seq: 0, consumed })
                .await?;
            // Mid-iteration checkpoint: assistant turn is in messages,
            // tool calls have NOT been invoked yet.
            ctx.record.budgets.consumed = consumed;
            ctx.record.messages.clone_from(&messages);
            p.store.checkpoint(&ctx.rid, &ctx.record).await?;
        }

        if calls.is_empty() {
            // Finish-recovery at the stop terminal: when the last in-loop gate
            // was green, nudge the model toward finish before giving up.
            // Guard: max_nudges > 0 AND last_gate_green.
            // When false (red/never-green gate OR max_nudges == 0), falls
            // through to the unchanged StoppedWithoutFinish path below.
            if config.max_nudges > 0 && last_gate_green {
                // AC4 — Telemetry capture inside the guard. The normal
                // telemetry path (lines below the is_empty() return) is
                // unreachable for a no-tool-call turn, so we capture here.
                // Must be inside the guard: a mutate-after-nudge-then-stop
                // case clears last_gate_green, intentionally fails the guard,
                // and does NOT record the unverified stop text.
                // AC4: if this stop follows a prior nudge, capture the model's
                // reply text. The `= false` clear is intentionally OMITTED here:
                // the nudge branch (AC2) unconditionally sets it to `true`
                // (avoiding a write-without-read that clippy flags as
                // `unused_assignments`), and the exhaustion branch (AC3) returns
                // immediately so the value is never read again.
                if nudge_awaiting_status {
                    nudge_statuses.push(turn_text.clone());
                }

                if nudges_fired < config.max_nudges {
                    // AC2 — Inject a FRESH user message (NOT last_mut append).
                    // At this terminal the trailing message is the assistant
                    // stop turn (no tool calls), so the correct wire shape is
                    // assistant → fresh-user. Unlike the green-static site,
                    // where the trailing message is the tool-results
                    // Message::User and appending avoids a 400-inducing second
                    // adjacent user turn, here pushing a new Message::User is
                    // correct: the sequence assistant → user is a valid
                    // alternating pair and does NOT trigger a 400.
                    let nudge_text = prompt::render_nudge_prompt();
                    messages.push(Message::User {
                        content: vec![UserBlock::Text(nudge_text.clone())],
                    });
                    nudges_fired += 1;
                    stats.nudges_fired += 1;
                    nudge_awaiting_status = true;
                    if writer.is_enabled() {
                        writer.emit(
                            "harness_message",
                            json!({
                                "iteration": stats.iterations,
                                "kind": "nudge",
                                "placement": "new_user_message",
                                "text": nudge_text,
                                "last_gate_green": last_gate_green,
                                "iters_since_tree_change": iters_since_tree_change,
                                "static_tree_k": config.static_tree_k,
                                "nudge_number": nudges_fired,
                                "max_nudges": config.max_nudges,
                            }),
                        );
                    }
                    continue;
                }
                // AC3 — Exhaustion terminal: unified with the green-static
                // exhaustion terminal (engine.rs:1344–1381). Same summary
                // literal, same persistence discipline, same return type.
                let summary = format!(
                    "gates green but agent did not call finish after {} nudges",
                    config.max_nudges
                );
                let disposition = Disposition::Failed {
                    mode: FailureMode::FinishDiscipline,
                    summary,
                };
                if let (Some(ctx), Some(p)) = (persist.as_mut(), persistence) {
                    ctx.record.messages.clone_from(&messages);
                    ctx.record.budgets.consumed = BudgetConsumed {
                        iterations: initial_consumed.iterations + stats.iterations,
                        tokens: initial_consumed.tokens + stats.input_tokens + stats.output_tokens,
                        cost_micros: initial_consumed.cost_micros,
                    };
                    ctx.record.recovery_facts = Some(RecoveryFacts {
                        gates_green_at_exit: last_gate_green,
                        tree_dirty,
                        nudge_statuses: nudge_statuses.clone(),
                    });
                    ctx.record.disposition = Some(disposition.clone());
                    p.store
                        .append_event(
                            &ctx.rid,
                            Event::DispositionSet {
                                seq: 0,
                                disposition: disposition.clone(),
                            },
                        )
                        .await?;
                    p.store.checkpoint(&ctx.rid, &ctx.record).await?;
                }
                return Ok(LoopOutcome::Finished(disposition));
            }

            // Terminal path: StoppedWithoutFinish.
            // Reached when: last_gate_green == false (never verified in-loop,
            // or gate was invalidated by a later mutation) OR max_nudges == 0
            // (finish-recovery disabled). All paths below are UNCHANGED.
            if let (Some(ctx), Some(p)) = (persist.as_mut(), persistence) {
                let disposition = Disposition::Failed {
                    mode: FailureMode::StoppedWithoutFinish,
                    summary: "agent stopped generating tool calls without calling finish"
                        .to_string(),
                };
                ctx.record.disposition = Some(disposition.clone());
                p.store
                    .append_event(
                        &ctx.rid,
                        Event::DispositionSet {
                            seq: 0,
                            disposition,
                        },
                    )
                    .await?;
                p.store.checkpoint(&ctx.rid, &ctx.record).await?;
            }
            return Ok(LoopOutcome::StoppedWithoutFinish);
        }

        // Execute every requested call in order, collecting fed-back tool
        // results into a single user message. `finish` is special-cased in
        // [`handle_finish_call`]: a `done` claim triggers the harness
        // re-running the configured checks, and only a green verification
        // (or no checks at all) sets the terminal `finish` slot. A red
        // verification is fed back as an `is_error=true` result and the loop
        // CONTINUES with the remaining calls in the same batch. An
        // unrecognized or missing disposition is likewise fed back as an
        // `is_error=true` result and the batch continues.
        let mut results = Vec::with_capacity(calls.len());
        let mut finish: Option<Disposition> = None;
        // Per-iteration mutation flag — reset before the per-call loop. A
        // successful `edit_file`/`bash` latches it (driving the end-of-iteration
        // tree-counter reset) AND clears `last_gate_green`.
        let mut mutated_this_iter: bool = false;
        for call in &calls {
            // Only the FIRST accepted finish in a batch gets to terminate;
            // a later finish (or a finish while one is already accepted)
            // still executes as a normal tool invocation so its
            // acknowledgement lands in the fed-back batch alongside the
            // earlier calls.
            if call.name == FINISH_TOOL_NAME && finish.is_none() {
                let call_start = Instant::now();
                let outcome = handle_finish_call(
                    &call.id,
                    &call.input,
                    config.checks.as_ref(),
                    config.answer_schema.as_ref(),
                    &tree_baseline,
                    ctx,
                )
                .await;
                let duration_ms =
                    u64::try_from(call_start.elapsed().as_millis()).unwrap_or(u64::MAX);
                if writer.is_enabled() {
                    let UserBlock::ToolResult {
                        call_id,
                        content,
                        is_error,
                    } = &outcome.result
                    else {
                        unreachable!("handle_finish_call always returns a ToolResult");
                    };
                    writer.emit(
                        "tool_result",
                        json!({
                            "iteration": stats.iterations,
                            "call_id": call_id,
                            "tool_name": call.name,
                            "is_error": is_error,
                            "content": content,
                            "offload_path": Value::Null,
                            "duration_ms": duration_ms,
                            "finish_accepted": outcome.finish.is_some(),
                            "finish_verification": outcome
                                .report
                                .as_ref()
                                .map(|r| serde_json::to_value(r).unwrap_or(Value::Null)),
                            "finish_change": outcome
                                .change
                                .as_ref()
                                .map(|c| serde_json::to_value(c).unwrap_or(Value::Null)),
                            "tree_current": outcome
                                .current_tree
                                .as_ref()
                                .map(render_tree_observation),
                            "finish_answer": outcome
                                .answer
                                .as_ref()
                                .map(|a| serde_json::to_value(a).unwrap_or(Value::Null)),
                            "finish_rejection": outcome
                                .rejection
                                .map(FinishRejection::as_str),
                        }),
                    );
                }
                // Discriminate the two verified-but-rejected outcomes off
                // `FinishOutcome`, the same shape `invalid_raw` already uses.
                match outcome.rejection {
                    Some(FinishRejection::NoChange) => stats.no_change_rejections += 1,
                    Some(FinishRejection::AlreadySatisfiedChecks) => {
                        stats.already_satisfied_check_rejections += 1;
                    }
                    Some(FinishRejection::AnswerSchema) => stats.answer_schema_rejections += 1,
                    Some(FinishRejection::ModifiedWorkspace) => {
                        stats.modified_workspace_rejections += 1;
                    }
                    // Steering, not evidence — deliberately uncounted.
                    Some(FinishRejection::WrongModeDisposition) | None => {}
                }
                results.push(outcome.result);
                finish = outcome.finish;
                if let Some(raw) = outcome.invalid_raw {
                    stats.invalid_finish_calls += 1;
                    if stats.first_invalid_finish_raw.is_none() {
                        stats.first_invalid_finish_raw = Some(raw);
                    }
                }
            } else {
                // Non-finish tool call: append ToolCallStarted before invoke,
                // ToolCallResult after invoke (log-then-snapshot discipline).
                if let (Some(ctx), Some(p)) = (persist.as_ref(), persistence) {
                    p.store
                        .append_event(
                            &ctx.rid,
                            Event::ToolCallStarted {
                                seq: 0,
                                name: call.name.clone(),
                                args: call.input.clone(),
                                call_id: call.id.clone(),
                            },
                        )
                        .await?;
                }

                let call_start = Instant::now();
                let result = tools.invoke(&call.name, call.input.clone(), ctx).await;
                let duration_ms =
                    u64::try_from(call_start.elapsed().as_millis()).unwrap_or(u64::MAX);

                if let (Some(ctx), Some(p)) = (persist.as_ref(), persistence) {
                    p.store
                        .append_event(
                            &ctx.rid,
                            Event::ToolCallResult {
                                seq: 0,
                                name: call.name.clone(),
                                is_error: result.is_error,
                                summary: result.summary.clone(),
                                // summary only — detail is NOT concatenated (audit record)
                                offload_path: result
                                    .offload_path
                                    .as_ref()
                                    .map(|path| path.display().to_string()),
                            },
                        )
                        .await?;
                }

                let content = render_tool_result(&result);
                if writer.is_enabled() {
                    writer.emit(
                        "tool_result",
                        json!({
                            "iteration": stats.iterations,
                            "call_id": call.id,
                            "tool_name": call.name,
                            "is_error": result.is_error,
                            "content": content,
                            "offload_path": result
                                .offload_path
                                .as_ref()
                                .map(|path| path.display().to_string()),
                            "duration_ms": duration_ms,
                        }),
                    );
                }

                results.push(UserBlock::ToolResult {
                    call_id: call.id.clone(),
                    content,
                    is_error: result.is_error,
                });

                // Observe the actual tool ToolResult for finish-recovery
                // detection — the done-oracle is `run_checks`'s `is_error`
                // flag (the run_checks tool sets is_error = !report.passed),
                // NEVER a model self-report. A successful mutating tool call
                // (`edit_file`/`bash` with `!is_error`) latches `tree_dirty`,
                // marks `mutated_this_iter`, and CLEARS `last_gate_green` — a
                // mutation after a green check invalidates the green, closing
                // the stale-green false-trip window so the nudge's "gates are
                // currently green" is always true at trip time.
                let is_error = result.is_error;
                if call.name == "run_checks" {
                    last_gate_green = !is_error;
                } else if (call.name == "edit_file" || call.name == "bash") && !is_error {
                    mutated_this_iter = true;
                    tree_dirty = true;
                    last_gate_green = false;
                    if call.name == "bash" {
                        stats.bash_calls_ok += 1;
                    } else {
                        stats.edit_file_calls_ok += 1;
                    }
                }
            }
        }
        messages.push(Message::User { content: results });

        // Mirror the finish-recovery done-oracle into RunStats so EVERY
        // terminal (StoppedWithoutFinish, MaxIterations, Finished, the recovery
        // terminals, and `?` error exits) reports whether the last in-loop gate
        // was green at exit. Placed right after the per-call loop so it reflects
        // this iteration's final gate state; a no-tool-call StoppedWithoutFinish
        // (top of the NEXT iteration) correctly reads the prior iteration's
        // value, since a turn with no calls cannot change the gate.
        stats.gates_green_at_exit = last_gate_green;
        stats.tree_dirty = tree_dirty;

        // Nudge-status telemetry: this turn followed a nudge iff
        // `nudge_awaiting_status` was set. If the turn produced an accepted
        // `finish(done)` the loop terminates normally below — clear the flag
        // without pushing (a Done-after-nudge stays a clean success; its
        // nudge_statuses are recoverable from the event log/messages if later
        // wanted). Otherwise push the (possibly empty for a tool-calls-only
        // turn) captured text and clear.
        if nudge_awaiting_status {
            // An accepted `already_satisfied` after a nudge is a clean
            // success on the newly-advertised off-ramp, not a failed nudge —
            // and, decided here rather than inherited, so is an accepted
            // `answer`: it is a clean success on answer mode's off-ramp, not
            // a nudge the model failed to act on.
            let is_done = matches!(
                finish,
                Some(
                    Disposition::Done { .. }
                        | Disposition::AlreadySatisfied { .. }
                        | Disposition::Answer { .. }
                )
            );
            if !is_done {
                nudge_statuses.push(turn_text);
            }
            nudge_awaiting_status = false;
        }

        if let Some(disposition) = finish {
            // Terminal path: Finished. Write DispositionSet + terminal checkpoint.
            if let (Some(ctx), Some(p)) = (persist.as_mut(), persistence) {
                ctx.record.messages.clone_from(&messages);
                ctx.record.budgets.consumed = BudgetConsumed {
                    iterations: initial_consumed.iterations + stats.iterations,
                    tokens: initial_consumed.tokens + stats.input_tokens + stats.output_tokens,
                    cost_micros: initial_consumed.cost_micros,
                };
                ctx.record.disposition = Some(disposition.clone());
                p.store
                    .append_event(
                        &ctx.rid,
                        Event::DispositionSet {
                            seq: 0,
                            disposition: disposition.clone(),
                        },
                    )
                    .await?;
                p.store.checkpoint(&ctx.rid, &ctx.record).await?;
            }
            return Ok(LoopOutcome::Finished(disposition));
        }

        // End-of-iteration tree-counter update — reached only when `finish`
        // was None. Reset wins over increment when a mutation and the
        // per-iteration tick collide.
        if mutated_this_iter {
            iters_since_tree_change = 0;
            stats.mutating_iters += 1;
        } else {
            iters_since_tree_change += 1;
        }
        stats.iters_since_tree_change_at_exit = iters_since_tree_change;
        // Peak update BEFORE the trip check: the green-static trip resets the
        // counter to 0 at engine.rs:1451, so a peak update placed after the
        // trip would silently under-report.
        stats.peak_iters_since_tree_change = stats
            .peak_iters_since_tree_change
            .max(iters_since_tree_change);

        // Detection / high-precision trip — evaluated after the counter
        // update, only when finish-recovery is enabled (`max_nudges > 0`).
        // A RED gate (`last_gate_green == false`) MUST NOT trip — that case
        // falls through unchanged to the existing MaxIterations cap.
        if config.max_nudges > 0
            && last_gate_green
            && iters_since_tree_change >= config.static_tree_k
        {
            if nudges_fired < config.max_nudges {
                // Inject the nudge by APPENDING a `UserBlock::Text` onto the
                // content vec of the EXISTING tool-results `Message::User`
                // (the `results` batch just pushed above) — NOT a new
                // `Message::User`. `anthropic::map_message` maps each
                // `Message` 1:1 with NO same-role merge, so two adjacent
                // `Message::User` reach the wire as two `role:"user"` blocks
                // and 400. Appending keeps the conversation a single user
                // turn from the API's view.
                let nudge_text = prompt::render_nudge_prompt();
                if let Some(Message::User { content }) = messages.last_mut() {
                    content.push(UserBlock::Text(nudge_text.clone()));
                    nudges_fired += 1;
                    stats.nudges_fired += 1;
                    if writer.is_enabled() {
                        writer.emit(
                            "harness_message",
                            json!({
                                "iteration": stats.iterations,
                                "kind": "nudge",
                                "placement": "appended_to_tool_results",
                                "text": nudge_text,
                                "last_gate_green": last_gate_green,
                                "iters_since_tree_change": iters_since_tree_change,
                                "static_tree_k": config.static_tree_k,
                                "nudge_number": nudges_fired,
                                "max_nudges": config.max_nudges,
                            }),
                        );
                    }
                }
                nudge_awaiting_status = true;
                // Reset so K static iterations must re-accumulate before the
                // next trip.
                iters_since_tree_change = 0;
            } else {
                // Recovery terminal: gates green but agent did not call
                // `finish` after `max_nudges` nudges. Mirror the MaxIterations
                // block's persistence discipline (messages, budgets.consumed,
                // disposition, DispositionSet, checkpoint) AND write
                // `recovery_facts`. The harness NEVER constructs Done here —
                // claim-vs-verify is preserved; the loop is the sole Done
                // constructor (only after a green Verification::Checks via
                // handle_finish_call).
                let summary = format!(
                    "gates green but agent did not call finish after {} nudges",
                    config.max_nudges
                );
                let disposition = Disposition::Failed {
                    mode: FailureMode::FinishDiscipline,
                    summary,
                };
                if let (Some(ctx), Some(p)) = (persist.as_mut(), persistence) {
                    ctx.record.messages.clone_from(&messages);
                    ctx.record.budgets.consumed = BudgetConsumed {
                        iterations: initial_consumed.iterations + stats.iterations,
                        tokens: initial_consumed.tokens + stats.input_tokens + stats.output_tokens,
                        cost_micros: initial_consumed.cost_micros,
                    };
                    ctx.record.recovery_facts = Some(RecoveryFacts {
                        gates_green_at_exit: last_gate_green,
                        tree_dirty,
                        nudge_statuses: nudge_statuses.clone(),
                    });
                    ctx.record.disposition = Some(disposition.clone());
                    p.store
                        .append_event(
                            &ctx.rid,
                            Event::DispositionSet {
                                seq: 0,
                                disposition: disposition.clone(),
                            },
                        )
                        .await?;
                    p.store.checkpoint(&ctx.rid, &ctx.record).await?;
                }
                // When persistence is None (the `run` path), the facts are
                // computed but there is no record to write (acceptable — `run`
                // returns only LoopOutcome/RunStats). The returned
                // `Finished(Failed{FinishDiscipline})` still distinguishes the
                // recovery terminal from a model finish(failed) by its `mode`.
                return Ok(LoopOutcome::Finished(disposition));
            }
        }

        // `iteration_end` — fires exactly once per iteration that reaches
        // end-of-iteration bookkeeping (not for iterations that returned
        // earlier above, nor for a `continue`d stop-site nudge). Reads no
        // clock.
        if writer.is_enabled() {
            writer.emit(
                "iteration_end",
                json!({
                    "iteration": stats.iterations,
                    "mutated": mutated_this_iter,
                    "last_gate_green": last_gate_green,
                    "tree_dirty": tree_dirty,
                    "iters_since_tree_change": iters_since_tree_change,
                    "nudges_fired": nudges_fired,
                }),
            );
        }

        // Wall-clock breach check: evaluated BEFORE the non-terminal
        // end-of-iteration checkpoint so the Finished terminal (and the
        // FinishDiscipline recovery terminal above) take precedence. A breach
        // writes exactly ONE terminal checkpoint; no non-terminal checkpoint
        // is written for the same iteration.
        //
        // Sentinel: `wall_clock_secs == 0` means UNBOUNDED — the check is
        // skipped entirely when the budget is not set.
        //
        // Per-process semantics: `loop_start` is captured fresh at
        // `run_loop_impl` entry, giving each resumed process its own budget
        // window — NOT whole-run elapsed. This matches the worker's per-process
        // hard-kill behaviour.
        if config.wall_clock_secs != 0
            && config
                .clock
                .now()
                .duration_since(loop_start)
                .unwrap_or(Duration::ZERO)
                .as_secs()
                >= config.wall_clock_secs
        {
            let summary = "wall-clock budget exhausted".to_string();
            if let (Some(ctx), Some(p)) = (persist.as_mut(), persistence) {
                ctx.record.messages.clone_from(&messages);
                ctx.record.budgets.consumed = BudgetConsumed {
                    iterations: initial_consumed.iterations + stats.iterations,
                    tokens: initial_consumed.tokens + stats.input_tokens + stats.output_tokens,
                    cost_micros: initial_consumed.cost_micros,
                };
                // Recovery facts mirror the FinishDiscipline terminal exactly:
                // same `last_gate_green` / `tree_dirty` / `nudge_statuses`
                // loop-locals so the outer harness sees a consistent shape
                // regardless of which recovery terminal fired.
                ctx.record.recovery_facts = Some(RecoveryFacts {
                    gates_green_at_exit: last_gate_green,
                    tree_dirty,
                    nudge_statuses: nudge_statuses.clone(),
                });
                let disposition = Disposition::Failed {
                    mode: FailureMode::BudgetExhausted,
                    summary: summary.clone(),
                };
                ctx.record.disposition = Some(disposition.clone());
                p.store
                    .append_event(
                        &ctx.rid,
                        Event::DispositionSet {
                            seq: 0,
                            disposition,
                        },
                    )
                    .await?;
                p.store.checkpoint(&ctx.rid, &ctx.record).await?;
            }
            // UNCONDITIONAL return — mirrors the MaxIterations pattern so the
            // non-persistent path still terminates on breach.
            return Ok(LoopOutcome::BudgetExhausted { summary });
        }

        // Non-terminal end of iteration: write the end-of-iteration checkpoint
        // so a crash here loses at most the current iteration's tool results
        // (already in messages).
        if let (Some(ctx), Some(p)) = (persist.as_mut(), persistence) {
            ctx.record.messages.clone_from(&messages);
            ctx.record.budgets.consumed = BudgetConsumed {
                iterations: initial_consumed.iterations + stats.iterations,
                tokens: initial_consumed.tokens + stats.input_tokens + stats.output_tokens,
                cost_micros: initial_consumed.cost_micros,
            };
            p.store.checkpoint(&ctx.rid, &ctx.record).await?;
        }
    }

    // Terminal path: MaxIterations.
    if let (Some(ctx), Some(p)) = (persist.as_mut(), persistence) {
        let disposition = Disposition::Failed {
            mode: FailureMode::BudgetExhausted,
            summary: "iteration cap reached before the agent finished".to_string(),
        };
        ctx.record.messages.clone_from(&messages);
        ctx.record.budgets.consumed = BudgetConsumed {
            iterations: initial_consumed.iterations + stats.iterations,
            tokens: initial_consumed.tokens + stats.input_tokens + stats.output_tokens,
            cost_micros: initial_consumed.cost_micros,
        };
        // Recovery facts: a GREEN-static MaxIterations (gates green, tree
        // static, model never called finish — and finish-recovery disabled via
        // `max_nudges == 0` OR the FinishDiscipline terminal simply did not
        // trip) carries WIP the recovery feature exists to preserve. Mirror
        // the FinishDiscipline and BudgetExhausted terminals exactly so the
        // outer harness sees a consistent shape. A RED gate
        // (`last_gate_green == false`) has nothing worth preserving, so the
        // write is conditioned on `last_gate_green` — not unconditional. The
        // disposition, FailureMode, LoopOutcome, and exit-code mapping are
        // UNCHANGED; this adds recovery_facts ONLY.
        if last_gate_green {
            ctx.record.recovery_facts = Some(RecoveryFacts {
                gates_green_at_exit: last_gate_green,
                tree_dirty,
                nudge_statuses: nudge_statuses.clone(),
            });
        }
        ctx.record.disposition = Some(disposition.clone());
        p.store
            .append_event(
                &ctx.rid,
                Event::DispositionSet {
                    seq: 0,
                    disposition,
                },
            )
            .await?;
        p.store.checkpoint(&ctx.rid, &ctx.record).await?;
    }

    Ok(LoopOutcome::MaxIterations)
}

// =============================================================================
// Crash-resume helpers
// =============================================================================

/// Reconcile a potential dangling log tail for [`ResumeMode::Crash`].
///
/// Loads all events for `run_id`. If the last event is an
/// [`Event::ToolCallStarted`] (the only reliable "interrupted mid-execution"
/// signal — D5 appends Started then Result serially), this is a dangling tail:
/// the call was started but its Result was never recorded.
///
/// Returns the reconstructed initial message history:
///
/// - **Clean tail** (last event is not `ToolCallStarted`): returns
///   `record.messages` verbatim.
/// - **Dangling tail**: returns `record.messages` plus a synthetic
///   `Message::User` covering EVERY tool call in the last assistant turn.
///   Pairing is by `call_id` (D6): each [`Event::ToolCallStarted`]'s `call_id`
///   is matched against the [`model::ToolCallRequest`] ids in the snapshot's
///   in-flight assistant turn. Calls with a real logged
///   [`Event::ToolCallResult`] keep that real `{is_error, summary}`; the
///   interrupted call (and any that were never started) get a synthetic
///   `is_error=true` result with content `"interrupted by host restart"`.
///   Non-tool events between [`Event::ModelCall`] and the first
///   [`Event::ToolCallStarted`] (e.g. [`Event::BudgetTick`]) are filtered out
///   before the walk. Also appends one synthetic [`Event::ToolCallResult`] to
///   the log for the dangling Started, so the log is clean on the next resume.
async fn reconcile_crash_tail(
    record: &RunRecord,
    store: &Arc<dyn RunStore>,
    run_id: &str,
) -> Result<Vec<Message>, StoreError> {
    // Gate on SNAPSHOT shape, not log-tail shape: reconciliation is needed
    // exactly when the reloaded snapshot ends mid-turn — its last message is
    // an Assistant turn carrying tool calls whose results never made it into
    // `messages`. (After a clean end-of-iteration checkpoint the last message
    // is the User results batch; a stopped-without-finish turn carries no
    // tool calls.) Gating on `events.last()` being a `ToolCallStarted` is
    // WRONG: a crash between a `ToolCallResult` and the next `Started` — or
    // after the final result but before the end-of-iteration checkpoint —
    // leaves the log ending in a Result while the snapshot still dangles,
    // and an un-reconciled transcript ending in tool calls is a malformed
    // request on every backend.
    let last_assistant_calls: Vec<model::ToolCallRequest> = match record.messages.last() {
        Some(Message::Assistant { content }) => content
            .iter()
            .filter_map(|b| match b {
                model::ContentBlock::ToolCall(req) => Some(req.clone()),
                model::ContentBlock::Text(_) | model::ContentBlock::Reasoning { .. } => None,
            })
            .collect(),
        _ => return Ok(record.messages.clone()), // clean: ends with results batch
    };
    if last_assistant_calls.is_empty() {
        return Ok(record.messages.clone()); // assistant turn with no tool calls
    }

    let events = store.list_events(run_id).await?;

    // Isolate events after the last ModelCall (the current iteration's tail),
    // then filter to tool events only — ToolCallStarted and ToolCallResult.
    // This skips BudgetTick and any other non-tool events the engine emits
    // between ModelCall and the first ToolCallStarted (e.g. the BudgetTick
    // appended at engine.rs:895 before tool dispatch begins).
    let post_model_idx = events
        .iter()
        .rposition(|e| matches!(e, Event::ModelCall { .. }))
        .map_or(0, |i| i + 1);
    let tool_tail: Vec<&Event> = events[post_model_idx..]
        .iter()
        .filter(|e| {
            matches!(
                e,
                Event::ToolCallStarted { .. } | Event::ToolCallResult { .. }
            )
        })
        .collect();

    // Build a map: call_id → Option<(is_error, summary)>.
    //   None  = ToolCallStarted seen, no matching Result yet (dangling).
    //   Some  = both Started + Result seen (call completed).
    //
    // ToolCallResult carries no call_id field; pair it positionally with the
    // most-recently-unmatched ToolCallStarted (pending_cid). Malformed-tail
    // safety: an orphaned Result (no unmatched Started) is silently ignored;
    // a duplicate call_id in Started leaves the earlier entry unchanged.
    let mut completion_map: HashMap<String, Option<(bool, String)>> = HashMap::new();
    let mut started_names: Vec<(String, String)> = Vec::new(); // (call_id, name)
    let mut pending_cid: Option<String> = None;
    for event in &tool_tail {
        match event {
            Event::ToolCallStarted { call_id, name, .. } => {
                // If a previous Started is still unmatched, leave it as None
                // in the map. Update the pending slot. Record the name on
                // first sighting only (duplicate call_ids degrade safely).
                pending_cid = Some(call_id.clone());
                if !completion_map.contains_key(call_id.as_str()) {
                    started_names.push((call_id.clone(), name.clone()));
                }
                completion_map.entry(call_id.clone()).or_insert(None);
            }
            Event::ToolCallResult {
                is_error, summary, ..
            } => {
                if let Some(cid) = pending_cid.take() {
                    // Pair this Result to the most-recently-unmatched Started.
                    completion_map.insert(cid, Some((*is_error, summary.clone())));
                }
                // Orphaned Result (no unmatched Started): degrade safely, ignore.
            }
            _ => {} // unreachable after filter; never panic on edge cases
        }
    }

    // Build the reconciled UserBlock list: for each call in the last assistant
    // turn, look up by call_id. Calls with a real logged Result keep it; calls
    // with no matching Result (dangling Started or never started) get the
    // synthetic is_error "interrupted by host restart" block.
    //
    // Known fidelity limit: the event log stores `ToolResult.summary` only
    // (the audit record deliberately omits the full rendered detail), so a
    // reconciled completed call feeds the model a terser — but true — result
    // than an uninterrupted run would have. The model can re-read workspace
    // state if it needs the detail.
    let mut results: Vec<UserBlock> = Vec::with_capacity(last_assistant_calls.len());
    for call in &last_assistant_calls {
        let completed = match completion_map.get(&call.id) {
            Some(Some((is_error, summary))) => Some((*is_error, summary.clone())),
            _ => None, // not started at all, or started but no Result (dangling)
        };
        results.push(match completed {
            Some((is_error, summary)) => UserBlock::ToolResult {
                call_id: call.id.clone(),
                content: summary,
                is_error,
            },
            None => UserBlock::ToolResult {
                call_id: call.id.clone(),
                content: "interrupted by host restart".to_string(),
                is_error: true,
            },
        });
    }

    // Append a synthetic ToolCallResult for every unmatched Started so the
    // log is paired on the next resume (avoids double-reconciliation). Calls
    // that never reached a Started event have nothing to pair in the log;
    // their transcript block above is synthesis enough, and re-running this
    // reconciliation is idempotent.
    for (call_id, name) in &started_names {
        if matches!(completion_map.get(call_id.as_str()), Some(None)) {
            store
                .append_event(
                    run_id,
                    Event::ToolCallResult {
                        seq: 0,
                        name: name.clone(),
                        is_error: true,
                        summary: "interrupted by host restart".to_string(),
                        offload_path: None,
                    },
                )
                .await?;
        }
    }

    // Return the original messages + reconciliation user message.
    let mut messages = record.messages.clone();
    messages.push(Message::User { content: results });
    Ok(messages)
}

/// Resume a previously-interrupted run from its last checkpoint.
///
/// Loads the [`RunRecord`] for `run_id` from `store`. If no checkpoint exists
/// (`store.load` returns `Ok(None)`), returns
/// [`ResumeError::UnknownRunId`] immediately — no [`model::ModelBackend::turn`]
/// call is made.
///
/// See [`ResumeMode`] for the two resumption strategies (D6/D7).
///
/// The system and task prompts are **RE-RENDERED** from `config + tools` (not
/// read from the record — D9 byte-identity invariant).
///
/// **Budget carry-over (0.3.0):** `budgets.consumed` accumulates across
/// resume in the [`RunRecord`], but remaining-budget enforcement is deferred
/// to 0.4.0. No enforcement logic is added here.
pub async fn resume(
    backend: &impl model::ModelBackend,
    tools: &ToolRegistry,
    ctx: &ToolCtx,
    config: &RunConfig,
    store: Arc<dyn RunStore>,
    run_id_arg: &str,
    mode: ResumeMode,
) -> Result<RunResult, ResumeError> {
    let start = Instant::now();
    let mut stats = RunStats {
        iterations: 0,
        input_tokens: 0,
        output_tokens: 0,
        wall_clock: Duration::ZERO,
        gates_green_at_exit: false,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        nudges_fired: 0,
        tree_dirty: false,
        iters_since_tree_change_at_exit: 0,
        peak_iters_since_tree_change: 0,
        mutating_iters: 0,
        bash_calls_ok: 0,
        edit_file_calls_ok: 0,
        invalid_finish_calls: 0,
        first_invalid_finish_raw: None,
        no_change_rejections: 0,
        already_satisfied_check_rejections: 0,
        answer_schema_rejections: 0,
        modified_workspace_rejections: 0,
        tree_baseline_unobservable: false,
    };

    // Load the checkpoint. Return UnknownRunId immediately — no backend call —
    // when no checkpoint exists for the requested run_id.
    let Some(record) = store.load(run_id_arg).await.map_err(ResumeError::Store)? else {
        return Err(ResumeError::UnknownRunId(run_id_arg.to_string()));
    };

    // Budget carry-over: accounting-only in 0.3.0.
    let initial_consumed = record.budgets.consumed;

    let (initial_messages, pre_persist) = match mode {
        ResumeMode::Crash => {
            // D6: reconcile log tail, continue under the same run_id.
            let messages = reconcile_crash_tail(&record, &store, run_id_arg)
                .await
                .map_err(ResumeError::Store)?;
            let pre = RunPersist {
                rid: run_id_arg.to_string(),
                record: {
                    let mut r = record.clone();
                    r.messages.clone_from(&messages);
                    r
                },
            };
            (messages, pre)
        }
        ResumeMode::FreshContext => {
            // D7: drop messages, fresh task seed, new run_id = task_id:(attempt_n+1).
            let task_message = prompt::render_task_prompt(&config.task);
            let messages = vec![Message::User {
                content: vec![UserBlock::Text(task_message)],
            }];
            let new_attempt_n = record.attempt_n + 1;
            let new_rid = run_id(&record.task.task_id, new_attempt_n);
            let pre = RunPersist {
                rid: new_rid.clone(),
                record: {
                    let mut r = record.clone();
                    r.run_id = new_rid;
                    r.attempt_n = new_attempt_n;
                    // Clear any prior terminal disposition — this is a new
                    // attempt, even though it carries durable state forward.
                    r.disposition = None;
                    r.messages.clone_from(&messages);
                    r
                },
            };
            (messages, pre)
        }
    };

    // Build a Persistence solely for store access inside run_loop_impl.
    // task_id / attempt_n are only used to compute the run_id when
    // override_persist is None; since we always pass Some(pre_persist),
    // those fields are irrelevant and set to empty/zero.
    let pers = Persistence {
        store,
        task_id: String::new(),
        attempt_n: 0,
        model_label: String::new(),
        backend_settings: None,
    };

    let outcome = run_loop_impl(
        backend,
        tools,
        ctx,
        config,
        Some(&pers),
        &mut stats,
        initial_messages,
        initial_consumed,
        Some(pre_persist),
        // A resumed run's TRUE starting tree predates the crash. Observing at
        // resume time would fold the pre-crash edits into the baseline and
        // then reject a run that legitimately needs no further edit — so the
        // baseline is deliberately `Unobservable`, which under
        // `classify_change` makes every resumed `done` claim accepted on
        // trust. Honest under-enforcement for a crash-recovery path.
        Some(TreeObservation::Unobservable {
            reason: "resumed run — the pre-crash starting tree is unavailable".to_string(),
        }),
    )
    .await
    .map_err(ResumeError::Store)?;

    stats.wall_clock = start.elapsed();
    Ok(RunResult { outcome, stats })
}

/// Compute the delay before retry attempt `attempt` (0-indexed) using a
/// deterministic exponential backoff schedule:
///
/// ```text
/// delay = base * 2^attempt
/// ```
///
/// Examples with `base = 500 ms`: attempt 0 → 500 ms, attempt 1 → 1 000 ms,
/// attempt 2 → 2 000 ms.
///
/// **No jitter, no RNG, no wall-clock read** — the delay is a pure function
/// of the attempt index (determinism invariant 3). Set `base =
/// Duration::ZERO` in tests to run retries with no sleep.
fn retry_delay(base: Duration, attempt: u32) -> Duration {
    base * 2u32.pow(attempt)
}

#[cfg(test)]
mod tests {
    use super::{
        ANSWER_SCHEMA_ERRORS_CAP, ANSWER_SCHEMA_ERRORS_MAX_LINES, AnswerSchema, FINISH_TOOL_NAME,
        FinishClaim, FinishRejection, FinishTool, LoopOutcome, Persistence, ResumeError,
        ResumeMode, RunConfig, RunResult, RunStats, answer_schema_rejection_content, emit_run_end,
        inert_precondition_warning, missing_reason_rejection_content,
        missing_result_rejection_content, no_change_rejection_content, rejection_content,
        render_tool_result, resume, retry_delay, run, run_id, run_persisted,
    };
    use crate::exec::{ChangeEvidence, CheckCommand, CheckReport, ChecksRunner};
    use crate::model::{
        AssistantTurn, BackendError, ContentBlock, Message, StopReason, TerminalKind,
        ToolCallRequest, TransientKind, Usage, UserBlock,
    };
    use crate::prompt;
    use crate::run_record::{
        BackendKind, BackendSettings, BudgetConsumed, BudgetLimits, Budgets, Disposition,
        DurableFacts, Event, FailureMode, Phase, ProjectConfig, RunRecord, SCHEMA_VERSION, Task,
        Verification,
    };
    use crate::store::{RunStore, SqliteRunStore, StoreError};
    use crate::test_support::MockBackend;
    use crate::time::{Clock, FakeClock};
    use crate::tool::{EchoTool, Tool, ToolCtx, ToolRegistry, ToolResult};
    use crate::tools::edit_file::EditFileTool;
    use crate::tools::standard_registry;
    use crate::transcript::{TranscriptConfig, TranscriptWriter};
    use crate::workspace::Workspace;
    use async_trait::async_trait;
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, UNIX_EPOCH};
    use tempfile::TempDir;

    // ---- test-only store doubles ----------------------------------------

    /// Enumerates the observable store operations. Used by [`RecordingStore`]
    /// to verify log-then-snapshot ordering.
    #[derive(Debug, Clone)]
    enum StoreCall {
        AppendEvent { kind: String },
        Checkpoint,
    }

    /// A [`RunStore`] that delegates to an in-memory `SQLite` store and records
    /// every call in order. Used to assert the log-then-snapshot ordering
    /// discipline (events before checkpoints within each iteration).
    struct RecordingStore {
        inner: SqliteRunStore,
        calls: Mutex<Vec<StoreCall>>,
    }

    impl RecordingStore {
        fn new() -> Self {
            Self {
                inner: SqliteRunStore::open_in_memory().expect("in-memory SQLite"),
                calls: Mutex::new(Vec::new()),
            }
        }

        fn recorded_calls(&self) -> Vec<StoreCall> {
            self.calls.lock().expect("calls lock").clone()
        }
    }

    #[async_trait]
    impl RunStore for RecordingStore {
        async fn load(&self, rid: &str) -> Result<Option<RunRecord>, StoreError> {
            self.inner.load(rid).await
        }

        async fn append_event(&self, rid: &str, event: Event) -> Result<u64, StoreError> {
            let kind = match &event {
                Event::ModelCall { .. } => "ModelCall",
                Event::ToolCallStarted { .. } => "ToolCallStarted",
                Event::ToolCallResult { .. } => "ToolCallResult",
                Event::PhaseTransition { .. } => "PhaseTransition",
                Event::BudgetTick { .. } => "BudgetTick",
                Event::DispositionSet { .. } => "DispositionSet",
            };
            self.calls
                .lock()
                .expect("calls lock")
                .push(StoreCall::AppendEvent {
                    kind: kind.to_string(),
                });
            self.inner.append_event(rid, event).await
        }

        async fn checkpoint(&self, rid: &str, record: &RunRecord) -> Result<(), StoreError> {
            self.calls
                .lock()
                .expect("calls lock")
                .push(StoreCall::Checkpoint);
            self.inner.checkpoint(rid, record).await
        }

        async fn list_events(&self, rid: &str) -> Result<Vec<Event>, StoreError> {
            self.inner.list_events(rid).await
        }
    }

    /// A [`RunStore`] that captures every [`RunRecord`] snapshot passed to
    /// [`RunStore::checkpoint`]. Used to inspect intermediate (mid-iteration)
    /// checkpoint states that are overwritten by subsequent checkpoints.
    struct SnapshotStore {
        inner: SqliteRunStore,
        snapshots: Mutex<Vec<RunRecord>>,
    }

    impl SnapshotStore {
        fn new() -> Self {
            Self {
                inner: SqliteRunStore::open_in_memory().expect("in-memory SQLite"),
                snapshots: Mutex::new(Vec::new()),
            }
        }

        fn all_snapshots(&self) -> Vec<RunRecord> {
            self.snapshots.lock().expect("snapshots lock").clone()
        }
    }

    #[async_trait]
    impl RunStore for SnapshotStore {
        async fn load(&self, rid: &str) -> Result<Option<RunRecord>, StoreError> {
            self.inner.load(rid).await
        }

        async fn append_event(&self, rid: &str, event: Event) -> Result<u64, StoreError> {
            self.inner.append_event(rid, event).await
        }

        async fn checkpoint(&self, rid: &str, record: &RunRecord) -> Result<(), StoreError> {
            self.snapshots
                .lock()
                .expect("snapshots lock")
                .push(record.clone());
            self.inner.checkpoint(rid, record).await
        }

        async fn list_events(&self, rid: &str) -> Result<Vec<Event>, StoreError> {
            self.inner.list_events(rid).await
        }
    }

    /// A [`RunStore`] whose [`RunStore::append_event`] always returns an error.
    /// Used to assert that the first store error aborts [`run_persisted`].
    struct FailingStore;

    #[async_trait]
    impl RunStore for FailingStore {
        async fn load(&self, _rid: &str) -> Result<Option<RunRecord>, StoreError> {
            Ok(None)
        }

        async fn append_event(&self, _rid: &str, _event: Event) -> Result<u64, StoreError> {
            Err(StoreError::LockPoisoned)
        }

        async fn checkpoint(&self, _rid: &str, _record: &RunRecord) -> Result<(), StoreError> {
            Err(StoreError::LockPoisoned)
        }

        async fn list_events(&self, _rid: &str) -> Result<Vec<Event>, StoreError> {
            Ok(vec![])
        }
    }

    /// Build a [`Persistence`] backed by the given store.
    fn make_persistence(store: Arc<dyn RunStore>) -> Persistence {
        Persistence {
            store,
            task_id: "task-t".to_string(),
            attempt_n: 1,
            model_label: "test-model".to_string(),
            backend_settings: None,
        }
    }

    // Expected run_id for the make_persistence fixture above.
    const FIXTURE_RID: &str = "task-t:1";

    fn usage() -> Usage {
        Usage {
            input_tokens: 0,
            output_tokens: 0,
            cache_read_tokens: None,
            cache_write_tokens: None,
            reasoning_tokens: None,
        }
    }

    /// Build a [`Usage`] with the given `input_tokens`/`output_tokens` and
    /// every optional field cleared — the smallest thing tests need to script
    /// known per-turn token amounts.
    fn usage_with(input_tokens: u32, output_tokens: u32) -> Usage {
        Usage {
            input_tokens,
            output_tokens,
            cache_read_tokens: None,
            cache_write_tokens: None,
            reasoning_tokens: None,
        }
    }

    fn tool_call(id: &str, name: &str, input: serde_json::Value) -> ContentBlock {
        ContentBlock::ToolCall(ToolCallRequest {
            id: id.to_string(),
            name: name.to_string(),
            input,
        })
    }

    fn turn_with(content: Vec<ContentBlock>, stop_reason: StopReason) -> AssistantTurn {
        AssistantTurn {
            content,
            stop_reason,
            usage: usage(),
        }
    }

    /// Like [`turn_with`], but with an explicit [`Usage`] — for tests that
    /// pin the accumulated [`RunStats`] token totals.
    fn turn_with_usage(
        content: Vec<ContentBlock>,
        stop_reason: StopReason,
        usage: Usage,
    ) -> AssistantTurn {
        AssistantTurn {
            content,
            stop_reason,
            usage,
        }
    }

    fn finish_call(id: &str, input: serde_json::Value) -> AssistantTurn {
        turn_with(
            vec![tool_call(id, FINISH_TOOL_NAME, input)],
            StopReason::ToolUse,
        )
    }

    fn registry_with_finish_and_echo() -> ToolRegistry {
        let mut registry = ToolRegistry::new();
        registry.register("echo", Arc::new(EchoTool));
        registry.register(FINISH_TOOL_NAME, Arc::new(FinishTool::default()));
        registry
    }

    /// The last message in history must be the fed-back tool-result user
    /// message whose first block's `call_id` matches `expected_id`.
    fn assert_last_is_tool_result(messages: &[Message], expected_id: &str) {
        let last = messages.last().expect("at least one message");
        match last {
            Message::User { content } => match &content[0] {
                UserBlock::ToolResult { call_id, .. } => {
                    assert_eq!(
                        call_id, expected_id,
                        "fed-back call_id must match request id"
                    );
                }
                UserBlock::Text(_) => panic!("expected a ToolResult block, got Text"),
            },
            Message::Assistant { .. } => panic!("expected a User message, got Assistant"),
        }
    }

    fn passing_runner() -> ChecksRunner {
        ChecksRunner::new(
            CheckCommand {
                program: "/bin/sh".to_string(),
                args: vec!["-c".to_string(), "exit 0".to_string()],
            },
            PathBuf::from("/"),
            Duration::from_secs(10),
        )
    }

    fn failing_runner() -> ChecksRunner {
        ChecksRunner::new(
            CheckCommand {
                program: "/bin/sh".to_string(),
                args: vec!["-c".to_string(), "echo FAIL_DETAIL; exit 3".to_string()],
            },
            PathBuf::from("/"),
            Duration::from_secs(10),
        )
    }

    #[tokio::test]
    async fn single_finish_done_with_no_checks_terminates_with_no_checks_verification() {
        let backend = MockBackend::from_turns(vec![finish_call(
            "c1",
            serde_json::json!({ "disposition": "done", "summary": "all set" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 10);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Done {
                summary,
                verification: Verification::NoChecksConfigured,
                change: _,
            }) => {
                assert_eq!(summary, "all set");
            }
            other => panic!("expected Finished(Done{{NoChecksConfigured}}), got {other:?}"),
        }
        assert_eq!(backend.calls(), 1, "should finish in a single iteration");
        // The Finished variant carries stats too: one drawn turn, zero tokens
        // (the mock's usage() helper is all zeros).
        assert_eq!(stats.iterations, 1);
        assert_eq!(stats.input_tokens, 0);
        assert_eq!(stats.output_tokens, 0);
        assert_eq!(stats.invalid_finish_calls, 0);
        assert_eq!(stats.first_invalid_finish_raw, None);
    }

    #[tokio::test]
    async fn finish_done_with_passing_checks_terminates_with_checks_verification() {
        let backend = MockBackend::from_turns(vec![finish_call(
            "c1",
            serde_json::json!({ "disposition": "done", "summary": "shipped" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 10).with_checks(passing_runner());

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Done {
                summary,
                verification: Verification::Checks(report),
                change: _,
            }) => {
                assert_eq!(summary, "shipped");
                assert!(report.passed, "checks report must be green");
                assert_eq!(report.exit_code, Some(0));
            }
            other => panic!("expected Finished(Done{{Checks(green)}}), got {other:?}"),
        }
        assert_eq!(backend.calls(), 1);
        assert_eq!(stats.iterations, 1);
    }

    #[tokio::test]
    async fn finish_done_with_failing_checks_is_rejected_and_loop_continues() {
        // Two scripted turns: (1) finish(done) → rejected; (2) a non-finish
        // tool call so the second draw shows the loop went past the rejection.
        // We stop by drawing (2) then over-drawing the empty script.
        let backend = MockBackend::from_turns(vec![
            finish_call(
                "call-finish",
                serde_json::json!({ "disposition": "done", "summary": "premature" }),
            ),
            turn_with(
                vec![tool_call("c-echo", "echo", serde_json::json!({ "k": 1 }))],
                StopReason::ToolUse,
            ),
        ]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5).with_checks(failing_runner());

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        // With the second turn being a non-finish call and no third turn
        // scripted, the loop hits the over-draw path — the important
        // assertion is that we did NOT terminate at the finish, we drew
        // beyond it.
        assert!(
            matches!(outcome, LoopOutcome::BackendError(_)),
            "loop should continue past the rejected finish; got {outcome:?}"
        );
        assert!(
            backend.calls() >= 2,
            "second turn must have been drawn — loop didn't terminate at rejected finish; got {} calls",
            backend.calls()
        );

        // The rejected finish's fed-back tool result must appear somewhere
        // in the history the loop later sent to the backend: an
        // is_error=true UserBlock::ToolResult whose call_id matches
        // "call-finish" and whose content contains "rejected" + the check's
        // excerpt. Search every user message — the last one holds the
        // subsequent turn's echo result, which is the whole point (the loop
        // did NOT terminate at the rejection).
        let seen = backend.last_messages();
        let rejection = seen.iter().find_map(|m| match m {
            Message::User { content } => content.iter().find_map(|b| match b {
                UserBlock::ToolResult {
                    call_id,
                    content,
                    is_error,
                } if call_id == "call-finish" => Some((content.clone(), *is_error)),
                _ => None,
            }),
            Message::Assistant { .. } => None,
        });
        let (content, is_error) = rejection.expect("fed-back rejection tool-result present");
        assert!(is_error, "rejected finish result is is_error=true");
        assert!(
            content.contains("rejected"),
            "content must announce rejection; got:\n{content}"
        );
        assert!(
            content.contains("FAIL_DETAIL"),
            "content must include the check excerpt; got:\n{content}"
        );
        // The BackendError variant carries stats too. The mock overdraws on
        // the third turn, so `iterations` counts all three drawn turns.
        assert_eq!(stats.iterations, 3);
        // A checks rejection is NOT counted as an invalid disposition.
        assert_eq!(stats.invalid_finish_calls, 0);
    }

    #[tokio::test]
    async fn max_iterations_hits_when_model_keeps_claiming_done_with_red_checks() {
        // The model claims done twice in a row against a failing runner; both
        // are rejected; the loop hits max_iterations rather than terminating.
        let backend = MockBackend::from_turns(vec![
            finish_call(
                "c-1",
                serde_json::json!({ "disposition": "done", "summary": "first claim" }),
            ),
            finish_call(
                "c-2",
                serde_json::json!({ "disposition": "done", "summary": "second claim" }),
            ),
        ]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 2).with_checks(failing_runner());

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert!(
            matches!(outcome, LoopOutcome::MaxIterations),
            "repeated red claims must hit MaxIterations; got {outcome:?}"
        );
        assert_eq!(
            backend.calls(),
            2,
            "drew exactly max_iterations turns before giving up"
        );
        // MaxIterations carries stats — iterations equals the cap exactly.
        assert_eq!(stats.iterations, 2);
    }

    #[tokio::test]
    async fn finish_unknown_disposition_is_rejected_and_loop_continues() {
        let backend = MockBackend::from_turns(vec![
            finish_call(
                "c-bad",
                serde_json::json!({ "disposition": "complete", "summary": "huh" }),
            ),
            finish_call(
                "c-good",
                serde_json::json!({ "disposition": "done", "summary": "ok" }),
            ),
        ]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Done {
                summary,
                verification: Verification::NoChecksConfigured,
                change: _,
            }) => {
                assert_eq!(summary, "ok");
            }
            other => panic!("expected Finished(Done{{NoChecksConfigured}}), got {other:?}"),
        }
        assert_eq!(backend.calls(), 2);
        assert_eq!(stats.iterations, 2);
        assert_eq!(stats.invalid_finish_calls, 1);
        assert_eq!(
            stats.first_invalid_finish_raw,
            Some("\"complete\"".to_string())
        );

        let seen = backend.last_messages();
        let rejection = seen.iter().find_map(|m| match m {
            Message::User { content } => content.iter().find_map(|b| match b {
                UserBlock::ToolResult {
                    call_id,
                    content,
                    is_error,
                } if call_id == "c-bad" => Some((content.clone(), *is_error)),
                _ => None,
            }),
            Message::Assistant { .. } => None,
        });
        let (content, is_error) = rejection.expect("fed-back rejection tool-result present");
        assert!(is_error, "rejected finish result is is_error=true");
        assert_eq!(
            content,
            "finish rejected: disposition must be one of: done, blocked, failed, \
             already_satisfied; got \"complete\". Call finish again with one of those values."
        );
    }

    #[tokio::test]
    async fn finish_missing_disposition_is_rejected_and_loop_continues() {
        let backend = MockBackend::from_turns(vec![
            finish_call(
                "c-bad",
                serde_json::json!({ "summary": "no disposition field" }),
            ),
            finish_call(
                "c-good",
                serde_json::json!({ "disposition": "done", "summary": "ok" }),
            ),
        ]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Done {
                summary,
                verification: Verification::NoChecksConfigured,
                change: _,
            }) => {
                assert_eq!(summary, "ok");
            }
            other => panic!("expected Finished(Done{{NoChecksConfigured}}), got {other:?}"),
        }
        assert_eq!(backend.calls(), 2);
        assert_eq!(stats.invalid_finish_calls, 1);
        assert_eq!(
            stats.first_invalid_finish_raw,
            Some("<missing>".to_string())
        );

        let seen = backend.last_messages();
        let rejection = seen.iter().find_map(|m| match m {
            Message::User { content } => content.iter().find_map(|b| match b {
                UserBlock::ToolResult {
                    call_id,
                    content,
                    is_error,
                } if call_id == "c-bad" => Some((content.clone(), *is_error)),
                _ => None,
            }),
            Message::Assistant { .. } => None,
        });
        let (content, is_error) = rejection.expect("fed-back rejection tool-result present");
        assert!(is_error, "rejected finish result is is_error=true");
        assert_eq!(
            content,
            "finish rejected: disposition must be one of: done, blocked, failed, \
             already_satisfied; got <missing>. Call finish again with one of those values."
        );
    }

    #[tokio::test]
    async fn invalid_dispositions_only_end_at_max_iterations() {
        let backend = MockBackend::from_turns(vec![
            finish_call("c1", serde_json::json!({ "disposition": "complete" })),
            finish_call("c2", serde_json::json!({ "disposition": null })),
            finish_call("c3", serde_json::json!({ "summary": "x" })),
        ]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 3);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert!(
            matches!(outcome, LoopOutcome::MaxIterations),
            "repeated invalid dispositions must hit MaxIterations, not Finished(Failed); got {outcome:?}"
        );
        assert_eq!(backend.calls(), 3);
        assert_eq!(stats.iterations, 3);
        assert_eq!(stats.invalid_finish_calls, 3);
        assert_eq!(
            stats.first_invalid_finish_raw,
            Some("\"complete\"".to_string()),
            "the first invalid raw must win over the later null and <missing>"
        );
    }

    #[tokio::test]
    async fn invalid_disposition_does_not_run_checks() {
        let dir = TempDir::new().expect("tempdir");
        let runner = ChecksRunner::new(
            CheckCommand {
                program: "/bin/sh".to_string(),
                args: vec![
                    "-c".to_string(),
                    format!("echo ran >> {}/checks_ran; exit 3", dir.path().display()),
                ],
            },
            dir.path().to_path_buf(),
            Duration::from_secs(10),
        );
        let backend = MockBackend::from_turns(vec![
            finish_call(
                "c-bad",
                serde_json::json!({ "disposition": "success", "summary": "x" }),
            ),
            finish_call(
                "c-red",
                serde_json::json!({ "disposition": "done", "summary": "y" }),
            ),
            finish_call(
                "c-blk",
                serde_json::json!({ "disposition": "blocked", "decision_needed": "q" }),
            ),
        ]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10).with_checks(runner);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Blocked { decision_needed }) => {
                assert_eq!(decision_needed, "q");
            }
            other => panic!("expected Finished(Blocked), got {other:?}"),
        }
        assert_eq!(backend.calls(), 3);
        let checks_ran = std::fs::read_to_string(dir.path().join("checks_ran"))
            .expect("checks_ran file written by the sentinel check");
        assert_eq!(
            checks_ran.lines().count(),
            1,
            "checks must run exactly once, from the red `done` — the invalid claim ran none"
        );
        assert_eq!(stats.invalid_finish_calls, 1);

        let seen = backend.last_messages();
        let rejection = seen.iter().find_map(|m| match m {
            Message::User { content } => content.iter().find_map(|b| match b {
                UserBlock::ToolResult {
                    call_id,
                    content,
                    is_error,
                } if call_id == "c-bad" => Some((content.clone(), *is_error)),
                _ => None,
            }),
            Message::Assistant { .. } => None,
        });
        let (content, is_error) = rejection.expect("fed-back rejection tool-result present");
        assert!(is_error, "rejected finish result is is_error=true");
        assert_eq!(
            content,
            "finish rejected: disposition must be one of: done, blocked, failed, \
             already_satisfied; got \"success\". Call finish again with one of those values."
        );
    }

    #[tokio::test]
    async fn blocked_terminates_without_verification() {
        // The runner is set to fail — if the loop were verifying blocked,
        // this test would not terminate. It does verify termination is
        // unconditional for `blocked`.
        let backend = MockBackend::from_turns(vec![finish_call(
            "c-blocked",
            serde_json::json!({
                "disposition": "blocked",
                "summary": "ambiguous spec",
                "decision_needed": "which API version?"
            }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5).with_checks(failing_runner());

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Blocked { decision_needed }) => {
                assert_eq!(decision_needed, "which API version?");
            }
            other => panic!("expected Finished(Blocked), got {other:?}"),
        }
        assert_eq!(stats.iterations, 1);
    }

    #[tokio::test]
    async fn failed_terminates_without_verification() {
        let backend = MockBackend::from_turns(vec![finish_call(
            "c-failed",
            serde_json::json!({ "disposition": "failed", "summary": "tool kept erroring" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5).with_checks(failing_runner());

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Failed { summary, .. }) => {
                assert_eq!(summary, "tool kept erroring");
            }
            other => panic!("expected Finished(Failed), got {other:?}"),
        }
        assert_eq!(stats.iterations, 1);
    }

    #[tokio::test]
    async fn full_claim_vs_verify_integration_flag_file() {
        // The full loop: checks look for a file `flag` that doesn't exist
        // initially. Scripted turns:
        //   (1) finish(done) → REJECTED (flag missing → checks fail).
        //   (2) edit_file creates `flag` in the workspace.
        //   (3) finish(done) → VERIFIED GREEN → Finished(Done).
        // This is the "harness itself re-ran the checks" assertion — the
        // model never called run_checks, but a red claim was rejected and a
        // subsequent green claim was accepted.
        let root = TempDir::new().expect("workspace tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize root");
        let workspace = Workspace::new(&root_path, None).expect("workspace");
        let ctx = ToolCtx::new(Arc::new(workspace), Arc::new(crate::tool::StubOffloadSink));

        let runner = ChecksRunner::new(
            CheckCommand {
                program: "/bin/sh".to_string(),
                args: vec!["-c".to_string(), "test -f flag".to_string()],
            },
            root_path.clone(),
            Duration::from_secs(10),
        );

        let mut tools = ToolRegistry::new();
        tools.register("edit_file", Arc::new(EditFileTool));
        tools.register(FINISH_TOOL_NAME, Arc::new(FinishTool::default()));

        let backend = MockBackend::from_turns(vec![
            // 1: claim done — flag doesn't exist, should be rejected.
            finish_call(
                "c-premature",
                serde_json::json!({ "disposition": "done", "summary": "premature" }),
            ),
            // 2: create the flag file.
            turn_with(
                vec![tool_call(
                    "c-edit",
                    "edit_file",
                    serde_json::json!({
                        "path": "flag",
                        "old_string": "",
                        "new_string": "planted\n",
                    }),
                )],
                StopReason::ToolUse,
            ),
            // 3: claim done again — flag exists, should be verified.
            finish_call(
                "c-verified",
                serde_json::json!({ "disposition": "done", "summary": "flag planted" }),
            ),
        ]);

        let config = RunConfig::new("plant the flag", 5).with_checks(runner);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Done {
                summary,
                verification: Verification::Checks(report),
                change: _,
            }) => {
                assert_eq!(summary, "flag planted");
                assert!(
                    report.passed,
                    "second claim's report must be green (flag now exists)"
                );
            }
            other => panic!(
                "expected Finished(Done{{Checks(green)}}) after edit_file plants the flag; got {other:?}"
            ),
        }
        assert_eq!(
            backend.calls(),
            3,
            "loop should have drawn: rejected claim, edit, verified claim"
        );
        assert!(
            root_path.join("flag").exists(),
            "flag file should have been created by edit_file"
        );
        assert_eq!(stats.iterations, 3, "rejected claim, edit, verified claim");
    }

    #[tokio::test]
    async fn system_prompt_rendered_once_and_identical_every_turn() {
        // Three iterations: assert every turn's `system` string is identical
        // (byte-equal) — the prompt-cache correctness invariant. Also assert
        // it contains the check command display so the checks configuration
        // flows through into the prompt.
        let script: Vec<AssistantTurn> = (0..3)
            .map(|i| {
                turn_with(
                    vec![tool_call(
                        &format!("c{i}"),
                        "echo",
                        serde_json::json!({ "i": i }),
                    )],
                    StopReason::ToolUse,
                )
            })
            .collect();
        let backend = MockBackend::from_turns(script);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 3).with_checks(passing_runner());

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;
        assert!(matches!(outcome, LoopOutcome::MaxIterations));
        assert_eq!(backend.calls(), 3);

        let systems = backend.systems_seen();
        assert_eq!(systems.len(), 3, "one system entry per turn");
        let first = systems[0].as_ref().expect("system prompt was sent");
        assert!(
            first.contains("/bin/sh -c exit 0"),
            "system prompt must include the check command display; got:\n{first}"
        );
        for (i, entry) in systems.iter().enumerate() {
            let s = entry.as_ref().expect("system prompt was sent");
            assert_eq!(
                s.as_bytes(),
                first.as_bytes(),
                "turn {i} system prompt drifted from turn 0 — prompt cache invariant broken"
            );
        }
        assert_eq!(stats.iterations, 3);
    }

    #[tokio::test]
    async fn echo_then_finish_feeds_result_back_and_finishes() {
        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call(
                    "call-echo",
                    "echo",
                    serde_json::json!({ "x": 1 }),
                )],
                StopReason::ToolUse,
            ),
            finish_call(
                "call-finish",
                serde_json::json!({ "disposition": "done", "summary": "done after echo" }),
            ),
        ]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert!(matches!(
            outcome,
            LoopOutcome::Finished(Disposition::Done { .. })
        ));
        // Two model turns: echo, then finish.
        assert_eq!(backend.calls(), 2, "echo turn then finish turn");
        assert_eq!(stats.iterations, 2);
    }

    #[tokio::test]
    async fn fed_back_message_is_user_tool_result_with_matching_call_id() {
        // Turn 1: an echo call. Turn 2: finish. By turn 2, the messages the
        // loop sent to the backend include the fed-back tool-result user
        // message — assert it is a Message::User carrying a UserBlock::ToolResult
        // whose call_id matches the echo request id.
        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call(
                    "call-echo",
                    "echo",
                    serde_json::json!({ "k": 1 }),
                )],
                StopReason::ToolUse,
            ),
            finish_call(
                "call-finish",
                serde_json::json!({ "disposition": "done", "summary": "ok" }),
            ),
        ]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;
        assert!(matches!(
            outcome,
            LoopOutcome::Finished(Disposition::Done { .. })
        ));

        // The most recent turn (finish) saw history ending in the fed-back
        // echo tool result.
        let seen = backend.last_messages();
        assert_last_is_tool_result(&seen, "call-echo");

        // And the fed-back content is echo's rendered result (its input as a
        // string), proving echo actually executed.
        match seen.last().expect("history non-empty") {
            Message::User { content } => match &content[0] {
                UserBlock::ToolResult {
                    content, is_error, ..
                } => {
                    assert!(!is_error);
                    assert_eq!(content, &serde_json::json!({ "k": 1 }).to_string());
                }
                UserBlock::Text(_) => panic!("expected ToolResult"),
            },
            Message::Assistant { .. } => panic!("expected User message"),
        }
        assert_eq!(stats.iterations, 2);
    }

    #[tokio::test]
    async fn never_finishing_script_hits_max_iterations() {
        // Three non-finishing (plain tool_use that isn't finish) turns; cap at 3
        // means the loop draws exactly three turns and stops at the cap.
        let non_finish = || {
            turn_with(
                vec![tool_call("c", "echo", serde_json::json!({}))],
                StopReason::ToolUse,
            )
        };
        let backend = MockBackend::from_turns(vec![non_finish(), non_finish(), non_finish()]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 3);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert!(matches!(outcome, LoopOutcome::MaxIterations));
        assert_eq!(backend.calls(), 3, "drew exactly max_iterations turns");
        assert_eq!(stats.iterations, 3);
    }

    #[tokio::test]
    async fn first_turn_error_surfaces_backend_error_without_retry() {
        let backend = MockBackend::new(vec![Err(BackendError::Terminal {
            kind: TerminalKind::Auth,
            message: "no creds".to_string(),
        })]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert!(matches!(outcome, LoopOutcome::BackendError(_)));
        // Surfaced on the first call — no retry.
        assert_eq!(backend.calls(), 1);
        // BackendError-on-first-turn: iterations = 1 (the erroring draw counts),
        // both token totals stay zero (errored turns contribute nothing).
        assert_eq!(stats.iterations, 1);
        assert_eq!(stats.input_tokens, 0);
        assert_eq!(stats.output_tokens, 0);
    }

    #[tokio::test]
    async fn plain_text_turn_stops_without_finish() {
        let backend = MockBackend::from_turns(vec![turn_with(
            vec![ContentBlock::Text("I am just talking".to_string())],
            StopReason::EndTurn,
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert!(matches!(outcome, LoopOutcome::StoppedWithoutFinish));
        assert_eq!(backend.calls(), 1);
        assert_eq!(stats.iterations, 1);
        // No run_checks ran, so the gate was never green in-loop: this is the
        // "stopped before verifying" case, which finish-recovery correctly does
        // NOT target (nudging toward finish on an unverified state would risk a
        // false claim).
        assert!(
            !stats.gates_green_at_exit,
            "gate never green in-loop -> gates_green_at_exit must be false"
        );
    }

    #[tokio::test]
    async fn stopped_without_finish_reports_gate_green_when_verified_then_stopped() {
        // The counterpart to the never-verified stop above, and the exact
        // classification `gates_green_at_exit` exists to expose: the model runs
        // run_checks GREEN in-loop, then stops WITHOUT calling finish. This is a
        // done-but-unclaimed finish-discipline miss whose green precondition
        // finish-recovery COULD target. finish-recovery is disabled
        // (max_nudges = 0) so the stop itself — not a nudge — is the terminal.
        let root = TempDir::new().expect("workspace tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize root");
        let workspace = Workspace::new(&root_path, None).expect("workspace");
        let ctx = ToolCtx::new(Arc::new(workspace), Arc::new(crate::tool::StubOffloadSink));

        let runner = passing_runner();
        let tools = standard_registry(Some(runner.clone()));
        let config = RunConfig::new("do the task", 10)
            .with_checks(runner)
            .with_max_nudges(0);

        let backend = MockBackend::from_turns(vec![
            run_checks_turn("c1"),
            turn_with(
                vec![ContentBlock::Text("looks fixed to me".to_string())],
                StopReason::EndTurn,
            ),
        ]);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert!(
            matches!(outcome, LoopOutcome::StoppedWithoutFinish),
            "green run_checks then a no-tool-call turn must stop without finish; got {outcome:?}"
        );
        assert!(
            stats.gates_green_at_exit,
            "the last in-loop gate was green before the stop -> gates_green_at_exit must be true"
        );
    }

    #[tokio::test]
    async fn overdrawn_mock_surfaces_backend_error() {
        // A non-finishing turn followed by an empty script: the loop draws a
        // second turn that over-draws the mock, surfacing a BackendError.
        let backend = MockBackend::from_turns(vec![turn_with(
            vec![tool_call("c", "echo", serde_json::json!({}))],
            StopReason::ToolUse,
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert!(matches!(outcome, LoopOutcome::BackendError(_)));
        assert_eq!(backend.calls(), 2, "second draw over-draws the script");
        // Both draws count: the successful echo turn AND the second-draw
        // over-draw error.
        assert_eq!(stats.iterations, 2);
    }

    #[tokio::test]
    async fn finish_blocked_parses_decision_needed() {
        let backend = MockBackend::from_turns(vec![finish_call(
            "c1",
            serde_json::json!({
                "disposition": "blocked",
                "summary": "ambiguous spec",
                "decision_needed": "which API version?"
            }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Blocked { decision_needed }) => {
                assert_eq!(decision_needed, "which API version?");
            }
            other => panic!("expected Finished(Blocked), got {other:?}"),
        }
        assert_eq!(stats.iterations, 1);
    }

    #[tokio::test]
    async fn finish_failed_parses_summary() {
        let backend = MockBackend::from_turns(vec![finish_call(
            "c1",
            serde_json::json!({ "disposition": "failed", "summary": "tool kept erroring" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Failed { summary, .. }) => {
                assert_eq!(summary, "tool kept erroring");
            }
            other => panic!("expected Finished(Failed), got {other:?}"),
        }
        assert_eq!(stats.iterations, 1);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn from_input_classifies_every_disposition_shape() {
        // Built by a closure rather than a `let` binding so the SAME table of
        // pre-existing rows can be materialized twice — once per
        // `answer_enabled` value. `FinishClaim` is deliberately not `Clone`
        // (see its derive), so the rows cannot simply be reused.
        let shared_cases = || -> Vec<(serde_json::Value, FinishClaim)> {
            vec![
                (
                    serde_json::json!({ "disposition": "done", "summary": "s" }),
                    FinishClaim::Done {
                        summary: "s".to_string(),
                    },
                ),
                (
                    serde_json::json!({ "disposition": "blocked", "decision_needed": "d" }),
                    FinishClaim::Blocked {
                        decision_needed: "d".to_string(),
                    },
                ),
                (
                    serde_json::json!({ "disposition": "failed", "summary": "s" }),
                    FinishClaim::Failed {
                        summary: "s".to_string(),
                    },
                ),
                (
                    serde_json::json!({ "disposition": " Done \n" }),
                    FinishClaim::Done {
                        summary: String::new(),
                    },
                ),
                (
                    serde_json::json!({ "disposition": "FAILED" }),
                    FinishClaim::Failed {
                        summary: String::new(),
                    },
                ),
                (
                    serde_json::json!({ "disposition": "done", "summary": 5 }),
                    FinishClaim::Done {
                        summary: String::new(),
                    },
                ),
                (
                    serde_json::json!({ "disposition": "blocked" }),
                    FinishClaim::Blocked {
                        decision_needed: String::new(),
                    },
                ),
                (
                    serde_json::json!({ "disposition": "already_satisfied", "reason": "r" }),
                    FinishClaim::AlreadySatisfied {
                        reason: "r".to_string(),
                    },
                ),
                (
                    serde_json::json!({ "disposition": " Already_Satisfied ", "reason": "r" }),
                    FinishClaim::AlreadySatisfied {
                        reason: "r".to_string(),
                    },
                ),
                (
                    // The schema still requires `summary`, so a conforming
                    // already_satisfied call supplies one — it is DISCARDED, never
                    // smuggled into `reason`.
                    serde_json::json!({
                        "disposition": "already_satisfied",
                        "summary": "s",
                        "reason": "r",
                    }),
                    FinishClaim::AlreadySatisfied {
                        reason: "r".to_string(),
                    },
                ),
                (
                    serde_json::json!({ "disposition": "already_satisfied" }),
                    FinishClaim::MissingReason,
                ),
                (
                    serde_json::json!({ "disposition": "already_satisfied", "reason": "" }),
                    FinishClaim::MissingReason,
                ),
                (
                    serde_json::json!({ "disposition": "already_satisfied", "reason": "  \t " }),
                    FinishClaim::MissingReason,
                ),
                (
                    serde_json::json!({ "disposition": "already_satisfied", "reason": 7 }),
                    FinishClaim::MissingReason,
                ),
                (
                    serde_json::json!({ "disposition": "complete" }),
                    FinishClaim::Invalid {
                        raw: "\"complete\"".to_string(),
                    },
                ),
                (
                    serde_json::json!({ "disposition": "success" }),
                    FinishClaim::Invalid {
                        raw: "\"success\"".to_string(),
                    },
                ),
                (
                    serde_json::json!({ "disposition": "" }),
                    FinishClaim::Invalid {
                        raw: "\"\"".to_string(),
                    },
                ),
                (
                    serde_json::json!({ "disposition": 1 }),
                    FinishClaim::Invalid {
                        raw: "1".to_string(),
                    },
                ),
                (
                    serde_json::json!({ "disposition": true }),
                    FinishClaim::Invalid {
                        raw: "true".to_string(),
                    },
                ),
                (
                    serde_json::json!({ "disposition": null }),
                    FinishClaim::Invalid {
                        raw: "null".to_string(),
                    },
                ),
                (
                    serde_json::json!({ "disposition": ["done"] }),
                    FinishClaim::Invalid {
                        raw: "[\"done\"]".to_string(),
                    },
                ),
                (
                    serde_json::json!({ "summary": "x" }),
                    FinishClaim::Invalid {
                        raw: "<missing>".to_string(),
                    },
                ),
                (
                    serde_json::json!("done"),
                    FinishClaim::Invalid {
                        raw: "<missing>".to_string(),
                    },
                ),
                (
                    serde_json::json!(null),
                    FinishClaim::Invalid {
                        raw: "<missing>".to_string(),
                    },
                ),
                (
                    serde_json::json!(["done"]),
                    FinishClaim::Invalid {
                        raw: "<missing>".to_string(),
                    },
                ),
            ]
        };
        // Every pre-existing row must classify identically in BOTH modes —
        // turning answer mode on changes nothing about the other five shapes.
        for answer_enabled in [true, false] {
            for (input, expected) in shared_cases() {
                assert_eq!(
                    FinishClaim::from_input(&input, answer_enabled),
                    expected,
                    "input: {input:?} (answer_enabled: {answer_enabled})"
                );
            }
        }

        // The answer-mode rows, where `answer_enabled` IS the discriminator.
        let answer_cases: Vec<(serde_json::Value, bool, FinishClaim)> = vec![
            (
                serde_json::json!({ "disposition": "answer", "result": { "v": 1 } }),
                true,
                FinishClaim::Answer {
                    result: Some(serde_json::json!({ "v": 1 })),
                },
            ),
            (
                serde_json::json!({ "disposition": " ANSWER " }),
                true,
                FinishClaim::Answer { result: None },
            ),
            (
                serde_json::json!({ "disposition": "answer", "result": null }),
                true,
                FinishClaim::Answer {
                    result: Some(serde_json::Value::Null),
                },
            ),
            (
                serde_json::json!({
                    "disposition": "answer",
                    "summary": "ignored",
                    "result": {}
                }),
                true,
                FinishClaim::Answer {
                    result: Some(serde_json::json!({})),
                },
            ),
            // With answer mode OFF the value falls through the pre-existing
            // `Invalid` arm, so `raw` is the UNTRIMMED original JSON — not a
            // synthesized literal.
            (
                serde_json::json!({ "disposition": " ANSWER " }),
                false,
                FinishClaim::Invalid {
                    raw: "\" ANSWER \"".to_string(),
                },
            ),
            (
                serde_json::json!({ "disposition": "answer" }),
                false,
                FinishClaim::Invalid {
                    raw: "\"answer\"".to_string(),
                },
            ),
        ];
        for (input, answer_enabled, expected) in answer_cases {
            assert_eq!(
                FinishClaim::from_input(&input, answer_enabled),
                expected,
                "input: {input:?} (answer_enabled: {answer_enabled})"
            );
        }
    }

    #[test]
    fn render_tool_result_includes_bounded_detail() {
        let ctx = ToolCtx::stub();
        let with_detail = ToolResult::with_detail("summary line", "the detail body", &ctx);
        let rendered = render_tool_result(&with_detail);
        assert!(rendered.starts_with("summary line"));
        assert!(rendered.contains("the detail body"));

        let plain = ToolResult::ok("just a summary");
        assert_eq!(render_tool_result(&plain), "just a summary");
    }

    #[test]
    fn rejection_content_composes_header_excerpt_and_offload() {
        // Report with excerpt + offload path → all three pieces flow into
        // the content the model sees on rejection.
        let report = CheckReport {
            passed: false,
            exit_code: Some(1),
            timed_out: false,
            excerpt: "FAILURE OUTPUT".to_string(),
            offload_path: Some(PathBuf::from("/tmp/offload-0001.txt")),
            duration: Duration::from_millis(100),
        };
        let content = rejection_content("done", &report);
        assert!(
            content.starts_with("finish(done) rejected: verification failed"),
            "rejection header first; got:\n{content}"
        );
        assert!(content.contains("FAILURE OUTPUT"));
        assert!(content.contains("/tmp/offload-0001.txt"));

        // With no excerpt and no offload path, the header is the whole content.
        let bare = CheckReport {
            passed: false,
            exit_code: None,
            timed_out: true,
            excerpt: String::new(),
            offload_path: None,
            duration: Duration::from_secs(1),
        };
        let bare_content = rejection_content("done", &bare);
        assert_eq!(
            bare_content, "finish(done) rejected: verification failed",
            "bare report yields just the header"
        );
    }

    #[tokio::test]
    async fn finish_tool_metadata_and_run() {
        let tool = FinishTool::default();
        assert_eq!(tool.name(), FINISH_TOOL_NAME);
        let schema = tool.schema();
        assert_eq!(schema["name"], FINISH_TOOL_NAME);
        assert_eq!(schema["input_schema"]["type"], "object");

        let ctx = ToolCtx::stub();
        let result = tool
            .run(
                serde_json::json!({ "disposition": "done", "summary": "s" }),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        assert!(result.summary.contains("finish"));
    }

    #[test]
    fn finish_schema_disposition_description_is_pinned() {
        let schema = FinishTool::default().schema();
        assert_eq!(
            schema["description"],
            "End the run. Call it when the task is complete, already satisfied, blocked \
             on a decision, or has failed. A `done` claim is verified by the harness \
             re-running the configured checks AND requiring that the working tree \
             changed since the run started; an unchanged tree, a failed verification, or \
             a disposition other than done/already_satisfied/blocked/failed is fed back \
             as a tool-result error you can react to, not a termination."
        );
        assert_eq!(
            schema["input_schema"]["properties"]["disposition"]["description"],
            "done = task complete and you changed something (harness verifies via checks \
             AND requires a changed working tree); already_satisfied = the task was \
             already complete and nothing needed changing (requires `reason`; harness \
             still verifies via checks); blocked = needs a decision before retrying; \
             failed = you could not complete the task in this attempt and a fresh \
             attempt might succeed."
        );
        assert_eq!(
            schema["input_schema"]["properties"]["reason"]["description"],
            "Required when the disposition is already_satisfied: what you checked and \
             why the task was already complete."
        );
        let disposition_desc = schema["input_schema"]["properties"]["disposition"]["description"]
            .as_str()
            .expect("disposition description is a string");
        assert!(!disposition_desc.contains("the run is the problem"));
        assert_eq!(
            schema["input_schema"]["properties"]["disposition"]["enum"],
            serde_json::json!(["done", "blocked", "failed", "already_satisfied"])
        );
        assert_eq!(
            schema["input_schema"]["required"],
            serde_json::json!(["disposition", "summary"])
        );
    }

    #[tokio::test]
    async fn run_stats_accumulate_exact_per_turn_usage_across_a_scripted_run() {
        // Three scripted turns, each with a KNOWN, distinct `Usage`. After the
        // run finishes we assert `stats` sums those exact per-turn values.
        // - turn 1: echo call, usage(input=100, output=10)
        // - turn 2: echo call, usage(input=200, output=20)
        // - turn 3: finish(done), usage(input=300, output=30)
        // Expected: iterations = 3, input_tokens = 600, output_tokens = 60.
        let script = vec![
            turn_with_usage(
                vec![tool_call("c1", "echo", serde_json::json!({ "i": 1 }))],
                StopReason::ToolUse,
                usage_with(100, 10),
            ),
            turn_with_usage(
                vec![tool_call("c2", "echo", serde_json::json!({ "i": 2 }))],
                StopReason::ToolUse,
                usage_with(200, 20),
            ),
            turn_with_usage(
                vec![tool_call(
                    "c-finish",
                    FINISH_TOOL_NAME,
                    serde_json::json!({ "disposition": "done", "summary": "ok" }),
                )],
                StopReason::ToolUse,
                usage_with(300, 30),
            ),
        ];
        let backend = MockBackend::from_turns(script);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 10);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert!(matches!(
            outcome,
            LoopOutcome::Finished(Disposition::Done { .. })
        ));
        assert_eq!(stats.iterations, 3);
        assert_eq!(stats.input_tokens, 600);
        assert_eq!(stats.output_tokens, 60);
    }

    #[tokio::test]
    async fn run_stats_errored_turn_contributes_no_tokens_but_counts_the_iteration() {
        // Two turns: (1) a successful echo with non-zero usage, (2) a terminal
        // backend error. `iterations` counts both draws (2); `input_tokens` /
        // `output_tokens` include ONLY the first (successful) turn — errored
        // turns contribute nothing.
        let script: Vec<Result<AssistantTurn, BackendError>> = vec![
            Ok(turn_with_usage(
                vec![tool_call("c1", "echo", serde_json::json!({ "i": 1 }))],
                StopReason::ToolUse,
                usage_with(500, 50),
            )),
            Err(BackendError::Terminal {
                kind: TerminalKind::Other,
                message: "second turn boom".to_string(),
            }),
        ];
        let backend = MockBackend::new(script);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 10);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert!(matches!(outcome, LoopOutcome::BackendError(_)));
        assert_eq!(
            stats.iterations, 2,
            "both draws count, including the erroring one"
        );
        assert_eq!(
            stats.input_tokens, 500,
            "only the successful turn's usage sums"
        );
        assert_eq!(stats.output_tokens, 50);
    }

    #[tokio::test]
    async fn run_stats_wall_clock_populated_across_the_run() {
        // The `wall_clock` field must always be populated (non-zero after any
        // real call) — pin the invariant that it's finalized on every return
        // path. A single finish turn is the smallest case that still measures
        // real elapsed time.
        let backend = MockBackend::from_turns(vec![finish_call(
            "c1",
            serde_json::json!({ "disposition": "done", "summary": "ok" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("t", 5);

        let RunResult { outcome: _, stats } = run(&backend, &tools, &ctx, &config).await;

        // Wall clock is monotonic Instant-based, so any real call produces a
        // duration strictly greater than zero.
        assert!(
            stats.wall_clock > Duration::ZERO,
            "wall_clock must be measured across the run; got {:?}",
            stats.wall_clock,
        );
    }

    #[test]
    fn run_stats_is_debug_clone_and_eq() {
        // RunStats derives Clone / Debug / PartialEq / Eq — a value can be
        // moved into a report, cloned into a log line, and compared exactly.
        // The task spec explicitly requires these traits.
        let a = RunStats {
            iterations: 3,
            input_tokens: 100,
            output_tokens: 10,
            wall_clock: Duration::from_millis(250),
            gates_green_at_exit: false,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            nudges_fired: 0,
            tree_dirty: false,
            iters_since_tree_change_at_exit: 0,
            peak_iters_since_tree_change: 0,
            mutating_iters: 0,
            bash_calls_ok: 0,
            edit_file_calls_ok: 0,
            invalid_finish_calls: 0,
            first_invalid_finish_raw: None,
            no_change_rejections: 0,
            already_satisfied_check_rejections: 0,
            answer_schema_rejections: 0,
            modified_workspace_rejections: 0,
            tree_baseline_unobservable: false,
        };
        let printed = format!("{a:?}");
        assert!(printed.contains("RunStats"));
        let b = a.clone();
        assert_eq!(a, b);
        let c = RunStats {
            iterations: 4,
            ..b.clone()
        };
        assert_ne!(a, c);
        let d = RunStats {
            nudges_fired: 1,
            ..b.clone()
        };
        assert_ne!(a, d);
    }

    #[tokio::test]
    async fn finish_failed_disposition_yields_failed_with_loop_mode() {
        // A model that self-declares finish(failed) should yield
        // Disposition::Failed { mode: FailureMode::Loop, .. } — the pinned
        // default for "agent gave up / no productive progress".
        let backend = MockBackend::from_turns(vec![finish_call(
            "c1",
            serde_json::json!({ "disposition": "failed", "summary": "gave up" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);

        let RunResult { outcome, .. } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Failed { mode, .. }) => {
                assert_eq!(
                    mode,
                    FailureMode::Loop,
                    "model-declared failed must yield FailureMode::Loop"
                );
            }
            other => panic!("expected Finished(Failed{{Loop}}), got {other:?}"),
        }
    }

    #[test]
    fn into_disposition_maps_all_arms() {
        use crate::model::{BackendError, TerminalKind, TransientKind};

        // Finished(d) → d (pass-through)
        let d = Disposition::Done {
            summary: "ok".to_string(),
            verification: Verification::NoChecksConfigured,
            change: ChangeEvidence::default(),
        };
        let out = LoopOutcome::Finished(d.clone()).into_disposition();
        assert_eq!(out, d);

        // MaxIterations → Failed { mode: BudgetExhausted }
        // The summary string is pinned byte-for-byte so the outer harness can
        // distinguish a MaxIterations termination from a wall-clock one.
        let out = LoopOutcome::MaxIterations.into_disposition();
        match &out {
            Disposition::Failed {
                mode: FailureMode::BudgetExhausted,
                summary,
            } => {
                assert_eq!(
                    summary, "iteration cap reached before the agent finished",
                    "MaxIterations summary must be exactly this literal"
                );
            }
            other => panic!("MaxIterations must map to BudgetExhausted; got {other:?}"),
        }

        // BudgetExhausted → Failed { mode: BudgetExhausted }
        // The summary string must contain "wall-clock budget exhausted" so it is
        // distinguishable from the MaxIterations summary above.
        let out = LoopOutcome::BudgetExhausted {
            summary: "wall-clock budget exhausted".to_string(),
        }
        .into_disposition();
        match &out {
            Disposition::Failed {
                mode: FailureMode::BudgetExhausted,
                summary,
            } => {
                assert!(
                    summary.contains("wall-clock budget exhausted"),
                    "BudgetExhausted summary must contain \"wall-clock budget exhausted\"; got {summary:?}"
                );
            }
            other => panic!("BudgetExhausted must map to Failed{{BudgetExhausted}}; got {other:?}"),
        }

        // StoppedWithoutFinish → Failed { mode: StoppedWithoutFinish }
        let out = LoopOutcome::StoppedWithoutFinish.into_disposition();
        assert!(
            matches!(
                out,
                Disposition::Failed {
                    mode: FailureMode::StoppedWithoutFinish,
                    ..
                }
            ),
            "StoppedWithoutFinish must map to FailureMode::StoppedWithoutFinish; got {out:?}"
        );

        // BackendError(Transient) → Failed { mode: TransientInfra }
        let out = LoopOutcome::BackendError(BackendError::Transient {
            kind: TransientKind::RateLimit,
            retry_after: None,
        })
        .into_disposition();
        assert!(
            matches!(
                out,
                Disposition::Failed {
                    mode: FailureMode::TransientInfra,
                    ..
                }
            ),
            "Transient BackendError must map to TransientInfra; got {out:?}"
        );

        // BackendError(Terminal) → Failed { mode: PersistentToolError }
        let out = LoopOutcome::BackendError(BackendError::Terminal {
            kind: TerminalKind::Auth,
            message: "no creds".to_string(),
        })
        .into_disposition();
        assert!(
            matches!(
                out,
                Disposition::Failed {
                    mode: FailureMode::PersistentToolError,
                    ..
                }
            ),
            "Terminal BackendError must map to PersistentToolError; got {out:?}"
        );
    }

    #[test]
    fn run_config_defaults_and_builders_compose() {
        // new() sets checks=None and max_tokens=DEFAULT_MAX_TOKENS.
        let config = RunConfig::new("do a thing", 7);
        assert_eq!(config.task, "do a thing");
        assert_eq!(config.max_iterations, 7);
        assert!(config.checks.is_none());
        assert_eq!(config.max_tokens, super::DEFAULT_MAX_TOKENS);

        // Builders layer on top.
        let with_checks = RunConfig::new("t", 1).with_checks(passing_runner());
        assert!(with_checks.checks.is_some());
        let with_mt = RunConfig::new("t", 1).with_max_tokens(1234);
        assert_eq!(with_mt.max_tokens, 1234);

        // wall_clock_secs defaults 0; with_wall_clock_secs overrides it.
        assert_eq!(RunConfig::new("t", 1).wall_clock_secs, 0);
        let with_wc = RunConfig::new("t", 1).with_wall_clock_secs(120);
        assert_eq!(with_wc.wall_clock_secs, 120);

        // with_clock replaces the clock.
        let fake: Arc<dyn crate::time::Clock> = Arc::new(FakeClock::new(UNIX_EPOCH));
        let with_clk = RunConfig::new("t", 1).with_clock(fake);
        // Arc<dyn Clock> in RunConfig must be Debug and Clone.
        let _ = format!("{with_clk:?}");
        let _cloned = with_clk.clone();

        // transcript defaults to None; with_transcript sets it.
        assert!(RunConfig::new("t", 1).transcript.is_none());
        let with_transcript = RunConfig::new("t", 1).with_transcript("/x/t.jsonl", "lbl");
        assert_eq!(
            with_transcript.transcript,
            Some(crate::transcript::TranscriptConfig {
                path: "/x/t.jsonl".into(),
                label: "lbl".to_string(),
            })
        );
    }

    // =====================================================================
    // Wall-clock budget tests
    // =====================================================================

    /// SENTINEL: `wall_clock_secs = 0` means UNBOUNDED.
    ///
    /// A [`FakeClock`] with a large auto-advance step and `wall_clock_secs = 0`
    /// must reach `MaxIterations` (not `BudgetExhausted`) — the sentinel value
    /// `0` short-circuits the breach check entirely.
    #[tokio::test]
    async fn wall_clock_sentinel_zero_never_terminates_on_budget() {
        // Script: two echo turns, exhausts max_iterations (2).
        let backend = MockBackend::from_turns(vec![
            turn_with_usage(
                vec![tool_call("c1", "echo", serde_json::json!({ "i": 1 }))],
                StopReason::ToolUse,
                usage_with(5, 5),
            ),
            turn_with_usage(
                vec![tool_call("c2", "echo", serde_json::json!({ "i": 2 }))],
                StopReason::ToolUse,
                usage_with(5, 5),
            ),
        ]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        // Auto-advance ~10 years per now() call — would breach any real budget.
        // 87_600 hours = 10 * 365 * 24 h.
        let fake = Arc::new(FakeClock::new_auto_advance(
            UNIX_EPOCH,
            Duration::from_hours(87_600),
        ));
        let config = RunConfig::new("task", 2)
            .with_wall_clock_secs(0) // sentinel: unbounded
            .with_clock(fake as Arc<dyn crate::time::Clock>);

        let RunResult { outcome, .. } = run(&backend, &tools, &ctx, &config).await;
        assert!(
            matches!(outcome, LoopOutcome::MaxIterations),
            "sentinel 0 must yield MaxIterations, not BudgetExhausted; got {outcome:?}"
        );
    }

    /// POSITIVE TEST (non-persisted path): auto-advance [`FakeClock`] fires the
    /// breach on the first iteration.
    ///
    /// The [`FakeClock`] auto-advances by 31 s on each `now()` call. With
    /// budget = 30 s: `loop_start = T0`, breach-check reads `T0 + 31s`, elapsed
    /// = 31 s ≥ 30 s → `BudgetExhausted`.
    #[tokio::test]
    async fn wall_clock_breach_non_persistent_path_returns_budget_exhausted() {
        // Script: one echo iteration (breach fires before the next turn).
        let backend = MockBackend::from_turns(vec![turn_with_usage(
            vec![tool_call("c1", "echo", serde_json::json!({ "i": 1 }))],
            StopReason::ToolUse,
            usage_with(5, 5),
        )]);

        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        // Each now() call auto-advances 31s. Budget = 30s.
        //   Call 1: loop_start = T0, clock → T0+31s
        //   Call 2 (breach check): now = T0+31s, elapsed = 31s ≥ 30s → BREACH
        let fake = Arc::new(FakeClock::new_auto_advance(
            UNIX_EPOCH,
            Duration::from_secs(31),
        ));
        let config = RunConfig::new("task", 10)
            .with_wall_clock_secs(30)
            .with_clock(fake as Arc<dyn crate::time::Clock>);

        let RunResult { outcome, .. } = run(&backend, &tools, &ctx, &config).await;
        match outcome {
            LoopOutcome::BudgetExhausted { ref summary } => {
                assert!(
                    summary.contains("wall-clock budget exhausted"),
                    "summary must contain the pinned literal; got {summary:?}"
                );
            }
            other => panic!("expected BudgetExhausted; got {other:?}"),
        }
    }

    /// POSITIVE TEST (persisted path): [`FakeClock`] + scripted backend returns
    /// `BudgetExhausted` AND the checkpointed record carries `recovery_facts`.
    #[tokio::test]
    async fn wall_clock_breach_persisted_path_writes_recovery_facts() {
        let backend = MockBackend::from_turns(vec![turn_with_usage(
            vec![tool_call("c1", "echo", serde_json::json!({ "i": 1 }))],
            StopReason::ToolUse,
            usage_with(5, 5),
        )]);

        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        // Auto-advance 31s per call. Budget = 30s → breach on first iteration.
        let fake = Arc::new(FakeClock::new_auto_advance(
            UNIX_EPOCH,
            Duration::from_secs(31),
        ));
        let config = RunConfig::new("task", 10)
            .with_wall_clock_secs(30)
            .with_clock(fake as Arc<dyn crate::time::Clock>);

        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        let pers = make_persistence(Arc::clone(&store));

        let RunResult { outcome, .. } = run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no store error");

        // Outcome must be BudgetExhausted.
        match &outcome {
            LoopOutcome::BudgetExhausted { summary } => {
                assert!(summary.contains("wall-clock budget exhausted"));
            }
            other => panic!("expected BudgetExhausted; got {other:?}"),
        }

        // into_disposition must yield Failed { BudgetExhausted }.
        let disp = outcome.into_disposition();
        assert!(
            matches!(
                disp,
                Disposition::Failed {
                    mode: FailureMode::BudgetExhausted,
                    ..
                }
            ),
            "into_disposition must yield Failed{{BudgetExhausted}}; got {disp:?}"
        );

        // The persisted record must have recovery_facts = Some.
        let record = store
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("record must exist");
        assert!(
            record.recovery_facts.is_some(),
            "recovery_facts must be Some on a wall-clock breach terminal"
        );
        let rf = record.recovery_facts.unwrap();
        // tree_dirty = false: no edit_file/bash was called.
        assert!(
            !rf.tree_dirty,
            "tree_dirty must be false — no mutations ran"
        );
        // gates_green_at_exit = false: no run_checks was called.
        assert!(!rf.gates_green_at_exit);
    }

    /// RESUME DETERMINISM: a Crash-mode resume with a [`FakeClock`] still
    /// reconciles a dangling tool-call tail correctly, and `loop_start` is
    /// captured FRESH at `run_loop_impl` entry — not from persisted state.
    ///
    /// Uses a zero-step `FakeClock` so `loop_start` and every breach check
    /// both return the same instant (elapsed = 0) — the budget is never
    /// breached and the run completes normally via the scripted finish turn.
    #[tokio::test]
    async fn resume_with_fake_clock_reconciles_crash_tail_and_uses_fresh_budget() {
        // ---- Set up a run record with a dangling tail ----
        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));

        // Phase 1: a single echo turn — the record gets a mid-turn checkpoint
        // (assistant turn in messages, tool results NOT yet in messages because
        // MockBackend exhausts after one turn and the loop hits BackendError on
        // the next turn attempt). The StoppedWithoutFinish / BackendError path
        // leaves the record with a disposition set, so we overwrite it below.
        let phase1_backend = MockBackend::from_turns(vec![turn_with_usage(
            vec![tool_call("c1", "echo", serde_json::json!({ "i": 1 }))],
            StopReason::ToolUse,
            usage_with(5, 5),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config_p1 = RunConfig::new("task", 10);
        let pers = make_persistence(Arc::clone(&store));
        run_persisted(&phase1_backend, &tools, &ctx, &config_p1, &pers)
            .await
            .expect("phase 1 ok");

        // Rewrite the record to simulate a mid-iteration crash: remove the
        // last User message (tool results) so messages end with an Assistant
        // turn carrying pending tool calls.
        let mut record = store
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("must exist");
        // Clear any terminal disposition — crash means no clean terminal.
        record.disposition = None;
        record.recovery_facts = None;
        // Remove last User message to leave a dangling Assistant turn.
        if matches!(record.messages.last(), Some(Message::User { .. })) {
            record.messages.pop();
        }
        // Append a dangling ToolCallStarted to the event log so
        // reconcile_crash_tail has something to reconcile.
        store
            .append_event(
                FIXTURE_RID,
                Event::ToolCallStarted {
                    seq: 0,
                    name: "echo".to_string(),
                    args: serde_json::json!({ "i": 1 }),
                    call_id: "c1".to_string(),
                },
            )
            .await
            .expect("append");
        store
            .checkpoint(FIXTURE_RID, &record)
            .await
            .expect("checkpoint");

        // ---- Resume with a zero-step FakeClock ----
        // loop_start = UNIX_EPOCH; every breach check returns UNIX_EPOCH;
        // elapsed = 0 < 3600 → no breach. The resumed run finishes normally.
        let fake = Arc::new(FakeClock::new(UNIX_EPOCH));
        let phase2_backend = MockBackend::from_turns(vec![finish_call(
            "cf",
            serde_json::json!({ "disposition": "done", "summary": "recovered ok" }),
        )]);
        let config_p2 = RunConfig::new("task", 10)
            .with_wall_clock_secs(3600) // 1 hour budget; FakeClock at epoch → elapsed = 0
            .with_clock(fake as Arc<dyn crate::time::Clock>);

        let RunResult { outcome, .. } = resume(
            &phase2_backend,
            &tools,
            &ctx,
            &config_p2,
            Arc::clone(&store),
            FIXTURE_RID,
            ResumeMode::Crash,
        )
        .await
        .expect("resume ok");

        // The resumed run must finish normally (not BudgetExhausted).
        assert!(
            matches!(outcome, LoopOutcome::Finished(Disposition::Done { .. })),
            "resume must finish normally when budget is not breached; got {outcome:?}"
        );
    }

    // =====================================================================
    // Persistence tests (run identity + checkpoint wiring)
    // =====================================================================

    #[test]
    fn run_id_helper_formats_task_id_colon_attempt_n() {
        // Pinned vector: the join must be exactly "task-42:1", never a hash.
        assert_eq!(run_id("task-42", 1_u32), "task-42:1");
        // Also check the fixture we use throughout the persistence tests.
        assert_eq!(run_id("task-t", 1), FIXTURE_RID);
    }

    #[tokio::test]
    async fn run_persisted_produces_same_outcome_and_stats_as_run() {
        // No-store parity: run() and run_persisted() on the same scripted
        // trajectory should produce an identical LoopOutcome variant and
        // identical RunStats (except wall_clock, which is timing-dependent
        // and is not compared).
        let script = || {
            vec![
                turn_with_usage(
                    vec![tool_call("c1", "echo", serde_json::json!({ "i": 1 }))],
                    StopReason::ToolUse,
                    usage_with(10, 5),
                ),
                finish_call(
                    "cf",
                    serde_json::json!({ "disposition": "done", "summary": "ok" }),
                ),
            ]
        };

        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);

        let RunResult {
            outcome: o1,
            stats: s1,
        } = run(&MockBackend::from_turns(script()), &tools, &ctx, &config).await;

        let store = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        let pers = make_persistence(store);
        let RunResult {
            outcome: o2,
            stats: s2,
        } = run_persisted(
            &MockBackend::from_turns(script()),
            &tools,
            &ctx,
            &config,
            &pers,
        )
        .await
        .expect("run_persisted must succeed");

        // LoopOutcome variant
        assert!(
            matches!(o1, LoopOutcome::Finished(Disposition::Done { .. })),
            "run() outcome: {o1:?}"
        );
        assert!(
            matches!(o2, LoopOutcome::Finished(Disposition::Done { .. })),
            "run_persisted() outcome: {o2:?}"
        );
        // RunStats (not wall_clock)
        assert_eq!(s1.iterations, s2.iterations, "iterations must match");
        assert_eq!(s1.input_tokens, s2.input_tokens, "input_tokens must match");
        assert_eq!(
            s1.output_tokens, s2.output_tokens,
            "output_tokens must match"
        );
    }

    #[tokio::test]
    async fn tool_call_events_recorded_before_and_after_invoke() {
        // A scripted echo turn followed by finish. The event log must contain
        // ToolCallStarted("echo") immediately before ToolCallResult("echo").
        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call(
                    "call-echo",
                    "echo",
                    serde_json::json!({ "x": 1 }),
                )],
                StopReason::ToolUse,
            ),
            finish_call(
                "call-finish",
                serde_json::json!({ "disposition": "done", "summary": "done" }),
            ),
        ]);

        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);
        let store = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        let pers = make_persistence(store.clone());

        run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        let events = store.list_events(FIXTURE_RID).await.expect("list events");

        // Find the ToolCallStarted and ToolCallResult indices.
        let started_idx = events
            .iter()
            .position(|e| matches!(e, Event::ToolCallStarted { name, .. } if name == "echo"))
            .expect("ToolCallStarted(echo) must be in event log");
        let result_idx = events
            .iter()
            .position(|e| matches!(e, Event::ToolCallResult { name, .. } if name == "echo"))
            .expect("ToolCallResult(echo) must be in event log");

        assert!(
            started_idx < result_idx,
            "ToolCallStarted must precede ToolCallResult; started={started_idx}, result={result_idx}"
        );
    }

    #[tokio::test]
    async fn tool_call_started_event_carries_matching_call_id() {
        // The ToolCallStarted.call_id must equal the ToolCallRequest.id from
        // the model response — this is what resume pairs against.
        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call(
                    "my-unique-call-id",
                    "echo",
                    serde_json::json!({ "x": 1 }),
                )],
                StopReason::ToolUse,
            ),
            finish_call(
                "cf",
                serde_json::json!({ "disposition": "done", "summary": "ok" }),
            ),
        ]);

        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);
        let store = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        let pers = make_persistence(store.clone());

        run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        let events = store.list_events(FIXTURE_RID).await.expect("list events");
        let started = events.iter().find_map(|e| {
            if let Event::ToolCallStarted { name, call_id, .. } = e
                && name == "echo"
            {
                return Some(call_id.clone());
            }
            None
        });
        assert_eq!(
            started.as_deref(),
            Some("my-unique-call-id"),
            "ToolCallStarted.call_id must equal the model's ToolCallRequest.id"
        );
    }

    #[tokio::test]
    async fn two_successful_turns_produce_exactly_two_model_call_and_budget_tick_events() {
        // [echo turn, finish turn] = 2 model draws → 2 ModelCall + 2 BudgetTick.
        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call("c1", "echo", serde_json::json!({}))],
                StopReason::ToolUse,
            ),
            finish_call(
                "cf",
                serde_json::json!({ "disposition": "done", "summary": "done" }),
            ),
        ]);

        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);
        let store = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        let pers = make_persistence(store.clone());

        run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        let events = store.list_events(FIXTURE_RID).await.expect("list");
        let model_calls = events
            .iter()
            .filter(|e| matches!(e, Event::ModelCall { .. }))
            .count();
        let budget_ticks = events
            .iter()
            .filter(|e| matches!(e, Event::BudgetTick { .. }))
            .count();

        assert_eq!(
            model_calls, 2,
            "expected 2 ModelCall events, got {model_calls}"
        );
        assert_eq!(
            budget_ticks, 2,
            "expected 2 BudgetTick events, got {budget_ticks}"
        );
    }

    /// `backend_settings` is stamped onto the persisted record, and the
    /// persisted `Event::ModelCall`'s `model` equals
    /// `backend_settings.model_label()` — the twin identity sources must
    /// agree from a single construction site.
    #[tokio::test]
    async fn run_persisted_stamps_backend_settings_on_record_and_model_calls() {
        let s = BackendSettings {
            kind: BackendKind::Ollama,
            model: "m".to_string(),
            think: Some("on".to_string()),
            num_ctx: Some(32768),
            num_ctx_source: Some("localhost_default".to_string()),
        };
        let backend = MockBackend::from_turns(vec![finish_call(
            "c-fin",
            serde_json::json!({ "disposition": "done", "summary": "ok" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5);
        let store = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        let pers = Persistence {
            store: store.clone(),
            task_id: "task-42".to_string(),
            attempt_n: 1,
            model_label: s.model_label(),
            backend_settings: Some(s.clone()),
        };

        run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        let rid = run_id("task-42", 1);
        let rec = store
            .load(&rid)
            .await
            .expect("load")
            .expect("record present");
        assert_eq!(
            rec.backend_settings,
            Some(s.clone()),
            "the record must carry the constructed backend settings"
        );

        let events = store.list_events(&rid).await.expect("list");
        let first_call = events
            .iter()
            .find_map(|e| match e {
                Event::ModelCall { model, .. } => Some(model.clone()),
                _ => None,
            })
            .expect("at least one ModelCall event");
        assert_eq!(
            first_call,
            s.model_label(),
            "the persisted ModelCall's model must equal backend_settings.model_label()"
        );
    }

    #[tokio::test]
    async fn loaded_checkpoint_has_correct_run_id_schema_version_phase_and_run_checks() {
        // Verifies RunRecord construction: run_id format, schema_version == 2,
        // phase == InnerLoop, and run_checks populated from the ChecksRunner.
        let backend = MockBackend::from_turns(vec![finish_call(
            "cf",
            serde_json::json!({ "disposition": "done", "summary": "ok" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let runner = passing_runner();
        let runner_display = runner.command_display();
        let config = RunConfig::new("do the task", 5).with_checks(runner);
        let store = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        let pers = Persistence {
            store: store.clone(),
            task_id: "task-42".to_string(),
            attempt_n: 1,
            model_label: "test-model".to_string(),
            backend_settings: None,
        };

        run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        let rid = run_id("task-42", 1);
        let rec = store.load(&rid).await.expect("load").expect("present");

        assert_eq!(rec.run_id, "task-42:1", "run_id must be task_id:attempt_n");
        assert_eq!(
            rec.schema_version,
            crate::run_record::SCHEMA_VERSION,
            "schema_version must be SCHEMA_VERSION (= 2)"
        );
        assert_eq!(
            rec.phase,
            crate::run_record::Phase::InnerLoop,
            "phase must be InnerLoop"
        );
        assert_eq!(
            rec.project_config
                .run_checks
                .get("checks")
                .map(String::as_str),
            Some(runner_display.as_str()),
            "run_checks['checks'] must equal runner.command_display()"
        );
    }

    #[tokio::test]
    async fn finish_done_produces_disposition_done_in_persisted_record() {
        // A finish(done) terminal path must write a checkpoint whose
        // disposition is Some(Disposition::Done{..}) and emit a DispositionSet
        // event.
        let backend = MockBackend::from_turns(vec![finish_call(
            "cf",
            serde_json::json!({ "disposition": "done", "summary": "task complete" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);
        let store = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        let pers = make_persistence(store.clone());

        run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        let rec = store
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        assert!(
            matches!(rec.disposition, Some(Disposition::Done { .. })),
            "disposition must be Done; got {:?}",
            rec.disposition
        );

        let events = store.list_events(FIXTURE_RID).await.expect("list");
        let has_disposition_set = events
            .iter()
            .any(|e| matches!(e, Event::DispositionSet { .. }));
        assert!(has_disposition_set, "DispositionSet event must be in log");
    }

    #[tokio::test]
    async fn same_batch_invalid_then_valid_finish_terminates_on_valid() {
        // Both finish calls land in the SAME batch (one scripted turn); the
        // first is invalid, the second is a valid `done` — the loop must
        // reject the first and terminate on the second within that one turn.
        let backend = MockBackend::from_turns(vec![turn_with(
            vec![
                tool_call(
                    "c-bad",
                    FINISH_TOOL_NAME,
                    serde_json::json!({ "disposition": "complete" }),
                ),
                tool_call(
                    "c-good",
                    FINISH_TOOL_NAME,
                    serde_json::json!({ "disposition": "done", "summary": "ok" }),
                ),
            ],
            StopReason::ToolUse,
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);
        let store = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        let pers = make_persistence(store.clone());

        let RunResult { outcome, stats } = run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        match outcome {
            LoopOutcome::Finished(Disposition::Done {
                summary,
                verification: Verification::NoChecksConfigured,
                change: _,
            }) => {
                assert_eq!(summary, "ok");
            }
            other => panic!("expected Finished(Done{{NoChecksConfigured}}), got {other:?}"),
        }
        assert_eq!(backend.calls(), 1);
        assert_eq!(stats.iterations, 1);
        assert_eq!(stats.invalid_finish_calls, 1);

        let rec = store
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        let last = rec.messages.last().expect("at least one message");
        match last {
            Message::User { content } => {
                assert_eq!(content.len(), 2, "both fed-back results in one message");
                match &content[0] {
                    UserBlock::ToolResult {
                        call_id,
                        content,
                        is_error,
                    } => {
                        assert_eq!(call_id, "c-bad");
                        assert!(*is_error);
                        assert_eq!(
                            content,
                            "finish rejected: disposition must be one of: done, blocked, \
                             failed, already_satisfied; got \"complete\". Call finish again \
                             with one of those values."
                        );
                    }
                    UserBlock::Text(_) => panic!("expected ToolResult, got Text"),
                }
                match &content[1] {
                    UserBlock::ToolResult {
                        call_id,
                        content,
                        is_error,
                    } => {
                        assert_eq!(call_id, "c-good");
                        assert!(!is_error);
                        assert_eq!(content, "finish acknowledged");
                    }
                    UserBlock::Text(_) => panic!("expected ToolResult, got Text"),
                }
            }
            Message::Assistant { .. } => panic!("expected a User message, got Assistant"),
        }
    }

    #[tokio::test]
    async fn invalid_finish_is_persisted_only_in_messages() {
        let backend = MockBackend::from_turns(vec![
            finish_call(
                "c-bad",
                serde_json::json!({ "disposition": "complete", "summary": "x" }),
            ),
            finish_call(
                "c-good",
                serde_json::json!({ "disposition": "done", "summary": "ok" }),
            ),
        ]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);
        let store = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        let pers = make_persistence(store.clone());

        run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        let rec = store
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        assert!(
            matches!(rec.disposition, Some(Disposition::Done { .. })),
            "disposition must be Done; got {:?}",
            rec.disposition
        );

        let events = store.list_events(FIXTURE_RID).await.expect("list");
        let disposition_set_count = events
            .iter()
            .filter(|e| matches!(e, Event::DispositionSet { .. }))
            .count();
        assert_eq!(disposition_set_count, 1, "exactly one DispositionSet");
        let finish_tool_call_started = events
            .iter()
            .any(|e| matches!(e, Event::ToolCallStarted { name, .. } if name == FINISH_TOOL_NAME));
        assert!(
            !finish_tool_call_started,
            "finish calls must never emit ToolCallStarted"
        );

        let has_rejected = rec.messages.iter().any(|m| match m {
            Message::User { content } => content.iter().any(|b| {
                matches!(b, UserBlock::ToolResult { call_id, is_error, .. }
                    if call_id == "c-bad" && *is_error)
            }),
            Message::Assistant { .. } => false,
        });
        assert!(
            has_rejected,
            "rejected finish must be persisted in record.messages"
        );
    }

    #[tokio::test]
    async fn stopped_without_finish_produces_failed_stopped_without_finish_disposition() {
        // A plain-text turn (no tool calls) stops as StoppedWithoutFinish.
        // The persisted disposition must be Failed{mode: StoppedWithoutFinish}.
        let backend = MockBackend::from_turns(vec![turn_with(
            vec![ContentBlock::Text("I am just talking".to_string())],
            StopReason::EndTurn,
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);
        let store = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        let pers = make_persistence(store.clone());

        let RunResult { outcome, .. } = run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        assert!(
            matches!(outcome, LoopOutcome::StoppedWithoutFinish),
            "outcome must be StoppedWithoutFinish; got {outcome:?}"
        );

        let rec = store
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        assert!(
            matches!(
                rec.disposition,
                Some(Disposition::Failed {
                    mode: FailureMode::StoppedWithoutFinish,
                    ..
                })
            ),
            "persisted disposition must be Failed{{StoppedWithoutFinish}}; got {:?}",
            rec.disposition
        );
    }

    #[tokio::test]
    #[allow(clippy::too_many_lines)]
    async fn log_then_snapshot_ordering_and_two_turn_iteration_count() {
        // [echo turn, finish turn] = 2 model draws.
        // Asserts:
        //   1. The terminal DispositionSet event is recorded before the last
        //      checkpoint (log-then-snapshot discipline).
        //   2. No checkpoint is recorded before the first AppendEvent
        //      (the very first call must be an event, not a checkpoint).
        //   3. The loaded record's budgets.consumed.iterations == 2.
        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call("c1", "echo", serde_json::json!({}))],
                StopReason::ToolUse,
            ),
            finish_call(
                "cf",
                serde_json::json!({ "disposition": "done", "summary": "done" }),
            ),
        ]);

        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);
        let recording = Arc::new(RecordingStore::new());
        let pers = make_persistence(recording.clone());

        run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        let calls = recording.recorded_calls();
        assert!(!calls.is_empty(), "should have recorded some store calls");

        // (2) For EVERY iteration (segment bounded by Checkpoint entries), the
        // first call must be AppendEvent — not a Checkpoint.  Walk the full
        // sequence rather than checking only calls[0] so a regression in any
        // iteration (e.g. iteration 2 starting with a Checkpoint) would be caught.
        {
            let mut iteration_start_idx = 0usize;
            for (idx, call) in calls.iter().enumerate() {
                if matches!(call, StoreCall::Checkpoint) {
                    let first_in_iter = &calls[iteration_start_idx];
                    assert!(
                        matches!(first_in_iter, StoreCall::AppendEvent { .. }),
                        "iteration starting at call index {iteration_start_idx} must begin \
                         with AppendEvent, not Checkpoint; got {first_in_iter:?}"
                    );
                    iteration_start_idx = idx + 1;
                }
            }
        }

        // (1) Last DispositionSet must precede last Checkpoint.
        let last_ds = calls
            .iter()
            .rposition(|c| matches!(c, StoreCall::AppendEvent { kind } if kind == "DispositionSet"))
            .expect("DispositionSet event must have been appended");
        let last_ckpt = calls
            .iter()
            .rposition(|c| matches!(c, StoreCall::Checkpoint))
            .expect("at least one Checkpoint must have been written");
        assert!(
            last_ds < last_ckpt,
            "DispositionSet (idx {last_ds}) must precede the terminal Checkpoint (idx {last_ckpt})"
        );

        // (3) Final checkpoint has iterations == 2.
        let rec = recording
            .inner
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        assert_eq!(
            rec.budgets.consumed.iterations, 2,
            "loaded record must have iterations == 2; got {:?}",
            rec.budgets.consumed
        );
    }

    #[tokio::test]
    async fn mid_turn_checkpoint_contains_assistant_message_before_tool_execution() {
        // LEAD ADDITION: the mid-iteration checkpoint (written after the
        // assistant turn is appended to messages but BEFORE any tools.invoke)
        // must have the assistant turn as its last message. Specifically, that
        // assistant message must carry the ToolCallRequest with the started
        // call's id.
        //
        // Script: [echo call], [finish]. The first checkpoint (snapshot 0) is
        // the mid-iteration checkpoint of iteration 1.
        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call(
                    "call-echo-123",
                    "echo",
                    serde_json::json!({ "x": 1 }),
                )],
                StopReason::ToolUse,
            ),
            finish_call(
                "cf",
                serde_json::json!({ "disposition": "done", "summary": "ok" }),
            ),
        ]);

        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);
        let snap_store = Arc::new(SnapshotStore::new());
        let pers = make_persistence(snap_store.clone());

        run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        let snapshots = snap_store.all_snapshots();
        assert!(
            !snapshots.is_empty(),
            "at least one checkpoint must have been written"
        );

        // Snapshot 0 is the mid-iteration checkpoint of iteration 1:
        // messages = [task_seed_user, assistant_turn_with_echo_call]
        let first = &snapshots[0];
        let last_msg = first.messages.last().expect("messages must be non-empty");

        match last_msg {
            Message::Assistant { content } => {
                let has_echo_call = content.iter().any(|block| {
                    matches!(
                        block,
                        ContentBlock::ToolCall(ToolCallRequest { id, name, .. })
                        if id == "call-echo-123" && name == "echo"
                    )
                });
                assert!(
                    has_echo_call,
                    "first checkpoint's last message must be the assistant turn \
                     carrying call_id 'call-echo-123'; got content: {content:?}"
                );
            }
            other @ Message::User { .. } => {
                panic!("first checkpoint's last message must be Message::Assistant; got {other:?}")
            }
        }
    }

    #[tokio::test]
    async fn failing_store_causes_run_persisted_to_return_err() {
        // The first store error (on append_event) must abort run_persisted
        // and propagate as Err(StoreError).
        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call("c1", "echo", serde_json::json!({}))],
                StopReason::ToolUse,
            ),
            finish_call(
                "cf",
                serde_json::json!({ "disposition": "done", "summary": "ok" }),
            ),
        ]);

        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);
        let pers = make_persistence(Arc::new(FailingStore));

        let result = run_persisted(&backend, &tools, &ctx, &config, &pers).await;
        assert!(
            result.is_err(),
            "run_persisted with a failing store must return Err"
        );
    }

    #[tokio::test]
    async fn budget_consumed_in_record_reflects_current_iteration() {
        // The BudgetTick event and end-of-iteration checkpoint must carry
        // budgets.consumed matching the loop's current running totals.
        let backend = MockBackend::from_turns(vec![
            turn_with_usage(
                vec![tool_call("c1", "echo", serde_json::json!({}))],
                StopReason::ToolUse,
                usage_with(100, 50),
            ),
            finish_call(
                "cf",
                serde_json::json!({ "disposition": "done", "summary": "ok" }),
            ),
        ]);

        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);
        let store = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        let pers = make_persistence(store.clone());

        run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        let events = store.list_events(FIXTURE_RID).await.expect("list");
        // Second BudgetTick (iter 2, the finish turn) should have
        // accumulated tokens from BOTH turns.
        let budget_ticks: Vec<_> = events
            .iter()
            .filter_map(|e| {
                if let Event::BudgetTick { consumed, .. } = e {
                    Some(consumed)
                } else {
                    None
                }
            })
            .collect();

        // iter 1 BudgetTick: iterations=1, tokens=100+50=150
        let BudgetConsumed {
            iterations: i1,
            tokens: t1,
            ..
        } = budget_ticks[0];
        assert_eq!(*i1, 1, "first BudgetTick iterations must be 1");
        assert_eq!(*t1, 150, "first BudgetTick tokens must be 150 (100+50)");
    }

    // =====================================================================
    // Resume tests (AC-2 through AC-10)
    // =====================================================================

    /// Build a minimal [`RunRecord`] for resume tests — enough structure to
    /// satisfy the type but no interesting content beyond what each test sets.
    fn make_minimal_record(task_id: &str, attempt_n: u32) -> RunRecord {
        RunRecord {
            run_id: run_id(task_id, attempt_n),
            schema_version: SCHEMA_VERSION,
            attempt_n,
            task: Task {
                task_id: task_id.to_string(),
                title: String::new(),
                description: "test task".to_string(),
                acceptance_criteria: vec![],
                files_in_scope: vec![],
                scope_out: vec![],
            },
            project_config: ProjectConfig {
                run_checks: BTreeMap::new(),
                model_routing_hint: None,
            },
            phase: Phase::InnerLoop,
            durable_facts: DurableFacts::default(),
            budgets: Budgets {
                consumed: BudgetConsumed::default(),
                limits: BudgetLimits {
                    iterations: 10,
                    tokens: 0,
                    cost_micros: 0,
                    wall_clock_secs: 0,
                },
                wall_clock_start: "2026-01-01T00:00:00Z".to_string(),
            },
            last_gate_result: None,
            disposition: None,
            recovery_facts: None,
            backend_settings: None,
            messages: vec![],
        }
    }

    // AC-3: UnknownRunId — no backend calls.
    #[tokio::test]
    async fn resume_unknown_run_id_returns_error_without_backend_calls() {
        let backend = MockBackend::from_turns(vec![]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 5);
        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));

        let err = resume(
            &backend,
            &tools,
            &ctx,
            &config,
            store,
            "no-such-run",
            ResumeMode::Crash,
        )
        .await
        .expect_err("unknown run_id must return Err");

        assert!(
            matches!(&err, ResumeError::UnknownRunId(id) if id == "no-such-run"),
            "must be UnknownRunId(\"no-such-run\"); got {err:?}"
        );
        assert_eq!(
            backend.calls(),
            0,
            "no backend turn must be drawn for an unknown run_id"
        );
    }

    // AC-4: Crash mode clean tail — first turn sees exact reloaded messages.
    #[tokio::test]
    async fn crash_resume_clean_tail_first_turn_sees_exact_reloaded_messages() {
        // Seed a record with a known two-message history (user task + assistant
        // + user tool-result). The log's last event is NOT a ToolCallStarted
        // (clean tail). The resumed run finishes in one turn.
        let task_seed = Message::User {
            content: vec![UserBlock::Text("do the task".to_string())],
        };
        let asst = Message::Assistant {
            content: vec![ContentBlock::ToolCall(ToolCallRequest {
                id: "c-echo".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({}),
            })],
        };
        let tool_result = Message::User {
            content: vec![UserBlock::ToolResult {
                call_id: "c-echo".to_string(),
                content: "{}".to_string(),
                is_error: false,
            }],
        };
        let pre_messages = vec![task_seed.clone(), asst.clone(), tool_result.clone()];

        let mut record = make_minimal_record("clean-task", 1);
        record.messages = pre_messages.clone();

        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        store
            .checkpoint("clean-task:1", &record)
            .await
            .expect("checkpoint");
        // Append a ModelCall (not ToolCallStarted) so the tail is clean.
        store
            .append_event(
                "clean-task:1",
                Event::ModelCall {
                    seq: 0,
                    model: "test".to_string(),
                    prompt_tokens: 1,
                    completion_tokens: 1,
                },
            )
            .await
            .expect("append");

        // One-turn script: finish immediately.
        let backend = MockBackend::from_turns(vec![finish_call(
            "c-fin",
            serde_json::json!({ "disposition": "done", "summary": "ok" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5);

        let result = resume(
            &backend,
            &tools,
            &ctx,
            &config,
            store,
            "clean-task:1",
            ResumeMode::Crash,
        )
        .await
        .expect("resume must succeed");

        // First turn must see EXACTLY the reloaded messages — no reconciliation
        // message added on a clean tail.
        let seen = backend.messages_seen();
        assert!(!seen.is_empty(), "at least one turn was drawn");
        assert_eq!(
            seen[0], pre_messages,
            "first turn must see exactly the reloaded messages (clean-tail crash resume)"
        );
        assert!(
            matches!(
                result.outcome,
                LoopOutcome::Finished(Disposition::Done { .. })
            ),
            "outcome must be Done; got {:?}",
            result.outcome
        );
    }

    // AC-5: Crash mode dangling tail — two-call reconciliation.
    #[tokio::test]
    #[allow(clippy::too_many_lines)]
    async fn crash_resume_dangling_tail_reconciles_two_calls_no_reinvocation() {
        // A record whose last assistant turn has TWO tool calls: "count1" and
        // "count2". The event log has a ToolCallResult for call1 + a dangling
        // ToolCallStarted for call2. Resume must:
        //   (a) feed back a User message with 2 ToolResults (call1 real, call2 synthetic)
        //   (b) not invoke either tool during reconciliation
        //   (c) append a synthetic ToolCallResult for call2 to the log

        // --- build a counting tool so we can assert 0 new invocations ---
        struct CountingTool {
            count: Mutex<u32>,
        }
        impl CountingTool {
            fn invocations(&self) -> u32 {
                *self.count.lock().unwrap()
            }
        }
        #[async_trait]
        impl Tool for CountingTool {
            #[allow(clippy::unnecessary_literal_bound)]
            fn name(&self) -> &str {
                "count"
            }
            fn schema(&self) -> serde_json::Value {
                serde_json::json!({
                    "name": "count",
                    "input_schema": { "type": "object", "properties": {}, "required": [] }
                })
            }
            async fn run(&self, _input: serde_json::Value, _ctx: &ToolCtx) -> ToolResult {
                *self.count.lock().unwrap() += 1;
                ToolResult::ok("counted")
            }
        }

        let counter = Arc::new(CountingTool {
            count: Mutex::new(0),
        });
        let mut tools = ToolRegistry::new();
        tools.register("count", counter.clone());
        tools.register(FINISH_TOOL_NAME, Arc::new(FinishTool::default()));

        // Build the interrupted record: task seed + assistant(call1, call2).
        let task_seed = Message::User {
            content: vec![UserBlock::Text("do the task".to_string())],
        };
        let asst = Message::Assistant {
            content: vec![
                ContentBlock::ToolCall(ToolCallRequest {
                    id: "id-call1".to_string(),
                    name: "count".to_string(),
                    input: serde_json::json!({}),
                }),
                ContentBlock::ToolCall(ToolCallRequest {
                    id: "id-call2".to_string(),
                    name: "count".to_string(),
                    input: serde_json::json!({}),
                }),
            ],
        };
        let mut record = make_minimal_record("dangling-task", 1);
        record.messages = vec![task_seed.clone(), asst.clone()];

        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        store
            .checkpoint("dangling-task:1", &record)
            .await
            .expect("checkpoint");

        // Log seeds the REAL engine-emitted tail shape:
        //   ModelCall → BudgetTick → Started1(call_id=id-call1) → Result1 → Started2(call_id=id-call2) [dangling]
        // The BudgetTick is what the engine always emits (engine.rs:895) between
        // ModelCall and the first ToolCallStarted. The old test omitted it,
        // which caused the positional walk to never advance and mark EVERY call
        // as interrupted — this rewrite closes that seam.
        store
            .append_event(
                "dangling-task:1",
                Event::ModelCall {
                    seq: 0,
                    model: "test".to_string(),
                    prompt_tokens: 1,
                    completion_tokens: 1,
                },
            )
            .await
            .expect("mc");
        store
            .append_event(
                "dangling-task:1",
                Event::BudgetTick {
                    seq: 0,
                    consumed: BudgetConsumed {
                        iterations: 1,
                        tokens: 2,
                        cost_micros: 0,
                    },
                },
            )
            .await
            .expect("budget_tick");
        store
            .append_event(
                "dangling-task:1",
                Event::ToolCallStarted {
                    seq: 0,
                    name: "count".to_string(),
                    args: serde_json::json!({}),
                    call_id: "id-call1".to_string(),
                },
            )
            .await
            .expect("started1");
        store
            .append_event(
                "dangling-task:1",
                Event::ToolCallResult {
                    seq: 0,
                    name: "count".to_string(),
                    is_error: false,
                    summary: "counted".to_string(),
                    offload_path: None,
                },
            )
            .await
            .expect("result1");
        store
            .append_event(
                "dangling-task:1",
                Event::ToolCallStarted {
                    seq: 0,
                    name: "count".to_string(),
                    args: serde_json::json!({}),
                    call_id: "id-call2".to_string(),
                },
            )
            .await
            .expect("started2_dangling");

        // Resume script: one turn that calls finish immediately (so the loop
        // doesn't re-invoke count after reconciliation).
        let backend = MockBackend::from_turns(vec![finish_call(
            "c-fin",
            serde_json::json!({ "disposition": "done", "summary": "ok" }),
        )]);
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5);

        let _result = resume(
            &backend,
            &tools,
            &ctx,
            &config,
            store.clone(),
            "dangling-task:1",
            ResumeMode::Crash,
        )
        .await
        .expect("resume must succeed");

        // (a) First turn must see task seed + assistant + reconciled user msg
        //     (2 ToolResults: call1 real, call2 synthetic is_error).
        let seen = backend.messages_seen();
        assert!(!seen.is_empty());
        let first_turn = &seen[0];
        // messages[0] = task seed, [1] = assistant, [2] = reconciled user
        assert_eq!(
            first_turn.len(),
            3,
            "task_seed + asst + reconciled user; got {} msgs",
            first_turn.len()
        );
        let reconciled_user = &first_turn[2];
        match reconciled_user {
            Message::User { content } => {
                assert_eq!(content.len(), 2, "must have 2 ToolResult blocks");
                // call1: real result (not error, summary = "counted")
                match &content[0] {
                    UserBlock::ToolResult {
                        call_id,
                        content: c,
                        is_error,
                    } => {
                        assert_eq!(call_id, "id-call1");
                        assert!(!is_error, "call1 was completed successfully");
                        assert_eq!(c, "counted");
                    }
                    other @ UserBlock::Text(_) => {
                        panic!("expected ToolResult block 0; got {other:?}")
                    }
                }
                // call2: synthetic is_error=true
                match &content[1] {
                    UserBlock::ToolResult {
                        call_id,
                        content: c,
                        is_error,
                    } => {
                        assert_eq!(call_id, "id-call2");
                        assert!(is_error, "call2 must be is_error=true (interrupted)");
                        assert_eq!(c, "interrupted by host restart");
                    }
                    other @ UserBlock::Text(_) => {
                        panic!("expected ToolResult block 1; got {other:?}")
                    }
                }
            }
            other @ Message::Assistant { .. } => {
                panic!("expected User message for reconciled results; got {other:?}")
            }
        }

        // (b) CountingTool must not have been invoked during reconciliation.
        assert_eq!(
            counter.invocations(),
            0,
            "counting tool must not be invoked during reconciliation"
        );

        // (c) The log must contain a synthetic ToolCallResult(is_error=true, name="count")
        //     appended during reconciliation (the exact position is after the dangling
        //     Started; more events may follow from the resumed run itself).
        let events = store.list_events("dangling-task:1").await.expect("list");
        let synthetic = events.iter().find(|e| {
            matches!(
                e,
                Event::ToolCallResult {
                    is_error: true,
                    summary,
                    name,
                    ..
                }
                if summary == "interrupted by host restart" && name == "count"
            )
        });
        assert!(
            synthetic.is_some(),
            "synthetic ToolCallResult(is_error=true, summary='interrupted by host restart') \
             must be in the event log; events: {events:?}"
        );
    }

    // AC-6: FreshContext drops messages and carries budgets.consumed forward.
    #[tokio::test]
    async fn fresh_context_resume_drops_messages_carries_consumed() {
        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));

        // Build a record with non-empty messages and non-zero consumed.
        let old_msg = Message::User {
            content: vec![UserBlock::Text("old context".to_string())],
        };
        let pre_consumed = BudgetConsumed {
            iterations: 7,
            tokens: 500,
            cost_micros: 100,
        };
        let mut record = make_minimal_record("fc-task", 1);
        record.messages = vec![old_msg.clone()];
        record.budgets.consumed = pre_consumed;
        record
            .durable_facts
            .findings
            .push("prior finding".to_string());

        store
            .checkpoint("fc-task:1", &record)
            .await
            .expect("checkpoint");

        // One-turn script: finish immediately.
        let snap = Arc::new(SnapshotStore::new());
        // We need a combined store so we can observe snapshots AND have the
        // same data available for load. Use a wrapper.
        // For simplicity, use an SqliteStore for load and a SnapshotStore for
        // observing snapshots. But they need to share data...
        // Instead: manually checkpoint into the SnapshotStore's inner store.
        snap.inner
            .checkpoint("fc-task:1", &record)
            .await
            .expect("snap checkpoint");

        let store2: Arc<dyn RunStore> = snap.clone();

        let backend = MockBackend::from_turns(vec![finish_call(
            "c-fin",
            serde_json::json!({ "disposition": "done", "summary": "ok" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5);

        resume(
            &backend,
            &tools,
            &ctx,
            &config,
            store2,
            "fc-task:1",
            ResumeMode::FreshContext,
        )
        .await
        .expect("resume must succeed");

        // First turn must see ONLY the fresh task seed (old messages absent).
        let seen = backend.messages_seen();
        assert!(!seen.is_empty());
        let first = &seen[0];
        assert_eq!(first.len(), 1, "only task seed; old messages absent");
        match &first[0] {
            Message::User { content } => match &content[0] {
                UserBlock::Text(t) => assert!(
                    !t.contains("old context"),
                    "old context must not appear in fresh seed"
                ),
                other @ UserBlock::ToolResult { .. } => {
                    panic!("expected Text; got {other:?}")
                }
            },
            other @ Message::Assistant { .. } => {
                panic!("expected User message; got {other:?}")
            }
        }

        // The checkpointed record must carry the pre-restart consumed value.
        let snaps = snap.all_snapshots();
        assert!(!snaps.is_empty(), "at least one checkpoint written");
        let final_snap = snaps.last().unwrap();
        assert!(
            final_snap.budgets.consumed.iterations >= pre_consumed.iterations,
            "consumed.iterations must carry forward (>= pre-restart value); got {:?}",
            final_snap.budgets.consumed
        );
        // The durable finding from before the restart must still be there.
        assert!(
            final_snap
                .durable_facts
                .findings
                .contains(&"prior finding".to_string()),
            "durable_facts.findings must carry forward; got {:?}",
            final_snap.durable_facts.findings
        );
    }

    // AC-7: FreshContext run identity — new run_id, old intact.
    #[tokio::test]
    async fn fresh_context_run_identity_new_run_id_old_intact() {
        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));

        let mut record = make_minimal_record("id-task", 1);
        record.messages = vec![Message::User {
            content: vec![UserBlock::Text("old".to_string())],
        }];
        record.durable_facts.findings.push("carried".to_string());

        let old_rid = "id-task:1";
        store.checkpoint(old_rid, &record).await.expect("cp");

        let backend = MockBackend::from_turns(vec![finish_call(
            "cf",
            serde_json::json!({ "disposition": "done", "summary": "ok" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5);

        resume(
            &backend,
            &tools,
            &ctx,
            &config,
            store.clone(),
            old_rid,
            ResumeMode::FreshContext,
        )
        .await
        .expect("resume must succeed");

        // old run_id must still hold the original record (unchanged).
        let old_rec = store.load(old_rid).await.expect("load").expect("present");
        assert_eq!(
            old_rec.messages, record.messages,
            "prior run_id messages must be unmodified by FreshContext resume"
        );

        // new run_id = "id-task:2" must hold the continued record.
        let new_rid = "id-task:2";
        let new_rec = store
            .load(new_rid)
            .await
            .expect("load")
            .expect("new run_id must be checkpointed");
        assert_eq!(new_rec.attempt_n, 2, "new record must have attempt_n == 2");
        assert!(
            new_rec
                .durable_facts
                .findings
                .contains(&"carried".to_string()),
            "new record must carry durable_facts forward"
        );
    }

    // AC-7 complement: Crash resume checkpoints under the SAME run_id only.
    #[tokio::test]
    async fn crash_resume_checkpoints_under_same_run_id_only() {
        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));

        let mut record = make_minimal_record("crash-id-task", 1);
        // Minimal messages so reconcile finds no dangling tail.
        record.messages = vec![Message::User {
            content: vec![UserBlock::Text("do the task".to_string())],
        }];
        store
            .checkpoint("crash-id-task:1", &record)
            .await
            .expect("cp");
        // Append a non-ToolCallStarted event so the tail is clean.
        store
            .append_event(
                "crash-id-task:1",
                Event::ModelCall {
                    seq: 0,
                    model: "t".to_string(),
                    prompt_tokens: 0,
                    completion_tokens: 0,
                },
            )
            .await
            .expect("mc");

        let backend = MockBackend::from_turns(vec![finish_call(
            "cf",
            serde_json::json!({ "disposition": "done", "summary": "ok" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5);

        resume(
            &backend,
            &tools,
            &ctx,
            &config,
            store.clone(),
            "crash-id-task:1",
            ResumeMode::Crash,
        )
        .await
        .expect("resume must succeed");

        // Only the original run_id should have a checkpoint.
        let orig = store
            .load("crash-id-task:1")
            .await
            .expect("load")
            .expect("original run_id must still be checkpointed");
        assert_eq!(orig.attempt_n, 1, "same attempt_n on crash resume");

        // No new run_id must have been created.
        let new_rid = store.load("crash-id-task:2").await.expect("load");
        assert!(
            new_rid.is_none(),
            "crash resume must not create a new run_id; found: {new_rid:?}"
        );
    }

    // Resume must carry the constructed backend settings forward and never
    // clear them — FreshContext clears ONLY `disposition`.

    /// `FreshContext` resume carries `backend_settings` forward into the new
    /// run id and clears ONLY `disposition` — visible on the resumed run's
    /// first checkpoint, where the disposition is still pre-terminal.
    #[tokio::test]
    async fn fresh_context_resume_carries_backend_settings_and_clears_only_disposition() {
        let snap = Arc::new(SnapshotStore::new());
        let s = BackendSettings {
            kind: BackendKind::Ollama,
            model: "m".to_string(),
            think: Some("on".to_string()),
            num_ctx: Some(32768),
            num_ctx_source: Some("localhost_default".to_string()),
        };
        let mut record = make_minimal_record("bs-fc-task", 1);
        record.backend_settings = Some(s.clone());
        record.messages = vec![Message::User {
            content: vec![UserBlock::Text("old context".to_string())],
        }];
        snap.inner
            .checkpoint("bs-fc-task:1", &record)
            .await
            .expect("cp");
        let store: Arc<dyn RunStore> = snap.clone();

        let backend = MockBackend::from_turns(vec![finish_call(
            "c-fin",
            serde_json::json!({ "disposition": "done", "summary": "ok" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5);

        resume(
            &backend,
            &tools,
            &ctx,
            &config,
            store.clone(),
            "bs-fc-task:1",
            ResumeMode::FreshContext,
        )
        .await
        .expect("resume must succeed");

        let new_rid = run_id("bs-fc-task", 2);
        // The FINAL record under the new run id keeps the carried settings.
        let final_rec = store
            .load(&new_rid)
            .await
            .expect("load")
            .expect("new run_id must be checkpointed");
        assert_eq!(
            final_rec.backend_settings,
            Some(s.clone()),
            "FreshContext resume must carry backend_settings forward, never clear it"
        );
        // ...and the resumed run's FIRST checkpoint (pre-terminal) proves the
        // arm cleared ONLY `disposition`: `None` there, settings carried.
        let snaps: Vec<RunRecord> = snap
            .all_snapshots()
            .into_iter()
            .filter(|rec| rec.run_id == new_rid)
            .collect();
        assert!(
            !snaps.is_empty(),
            "resumed run must checkpoint under the new run id"
        );
        let first = snaps.into_iter().next().expect("just asserted non-empty");
        assert_eq!(
            first.backend_settings,
            Some(s),
            "the first post-resume checkpoint must already carry backend_settings"
        );
        assert_eq!(
            first.disposition, None,
            "FreshContext must clear disposition on the new attempt"
        );
    }

    /// `Crash` resume carries `backend_settings` forward under the SAME
    /// run id — the crash arm clones the record and leaves the field
    /// untouched.
    #[tokio::test]
    async fn crash_resume_carries_backend_settings() {
        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        let s = BackendSettings {
            kind: BackendKind::Ollama,
            model: "m".to_string(),
            think: Some("on".to_string()),
            num_ctx: Some(32768),
            num_ctx_source: Some("localhost_default".to_string()),
        };
        let mut record = make_minimal_record("bs-crash-task", 1);
        record.backend_settings = Some(s.clone());
        // Minimal messages so reconcile finds no dangling tail.
        record.messages = vec![Message::User {
            content: vec![UserBlock::Text("do the task".to_string())],
        }];
        store
            .checkpoint("bs-crash-task:1", &record)
            .await
            .expect("cp");
        // Append a non-ToolCallStarted event so the tail is clean.
        store
            .append_event(
                "bs-crash-task:1",
                Event::ModelCall {
                    seq: 0,
                    model: "t".to_string(),
                    prompt_tokens: 0,
                    completion_tokens: 0,
                },
            )
            .await
            .expect("mc");

        let backend = MockBackend::from_turns(vec![finish_call(
            "c-fin",
            serde_json::json!({ "disposition": "done", "summary": "ok" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5);

        resume(
            &backend,
            &tools,
            &ctx,
            &config,
            store.clone(),
            "bs-crash-task:1",
            ResumeMode::Crash,
        )
        .await
        .expect("resume must succeed");

        let rec = store
            .load("bs-crash-task:1")
            .await
            .expect("load")
            .expect("original run_id must still be checkpointed");
        assert_eq!(
            rec.backend_settings,
            Some(s),
            "Crash resume must carry backend_settings forward, never clear it"
        );
    }

    // AC-8: Prompt byte-identity on resume — system prompt rendered fresh.
    #[tokio::test]
    async fn resume_system_prompt_byte_identical_to_fresh_run() {
        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));

        let mut record = make_minimal_record("prompt-task", 1);
        record.messages = vec![Message::User {
            content: vec![UserBlock::Text("do the task".to_string())],
        }];
        store
            .checkpoint("prompt-task:1", &record)
            .await
            .expect("cp");
        store
            .append_event(
                "prompt-task:1",
                Event::ModelCall {
                    seq: 0,
                    model: "t".to_string(),
                    prompt_tokens: 0,
                    completion_tokens: 0,
                },
            )
            .await
            .expect("mc");

        // Use a non-trivial tools setup and a checks runner so the system
        // prompt includes the check command display.
        let tools = registry_with_finish_and_echo();
        let runner = passing_runner();
        let runner_display = runner.command_display();
        let config = RunConfig::new("do the task", 5).with_checks(runner);

        // Three-turn script so we see multiple system-prompt entries.
        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call("c1", "echo", serde_json::json!({}))],
                StopReason::ToolUse,
            ),
            finish_call(
                "cf",
                serde_json::json!({ "disposition": "done", "summary": "ok" }),
            ),
        ]);
        let ctx = ToolCtx::stub();

        resume(
            &backend,
            &tools,
            &ctx,
            &config,
            store,
            "prompt-task:1",
            ResumeMode::Crash,
        )
        .await
        .expect("resume must succeed");

        let systems = backend.systems_seen();
        assert!(!systems.is_empty(), "at least one turn drawn");

        // The expected system prompt: re-rendered from the same tools + checks.
        let expected = prompt::render_system_prompt(
            &prompt::tool_lines(&tools),
            Some(runner_display.as_str()),
        );

        for (i, entry) in systems.iter().enumerate() {
            let s = entry.as_ref().expect("system prompt must be sent");
            assert_eq!(
                s.as_bytes(),
                expected.as_bytes(),
                "turn {i} system prompt must be byte-identical to fresh render"
            );
        }
    }

    // AC-10: Crash resume happy path reaches verified Done.
    #[tokio::test]
    async fn crash_resume_happy_path_reaches_verified_done() {
        // Seed a record with a clean log tail, then resume. The script ends
        // with finish(done) against a passing ChecksRunner, proving the
        // claim-vs-verify loop still fires on resume.
        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));

        let task_seed = Message::User {
            content: vec![UserBlock::Text("do the task".to_string())],
        };
        let mut record = make_minimal_record("happy-task", 1);
        record.messages = vec![task_seed.clone()];
        store.checkpoint("happy-task:1", &record).await.expect("cp");
        // Clean tail.
        store
            .append_event(
                "happy-task:1",
                Event::ModelCall {
                    seq: 0,
                    model: "t".to_string(),
                    prompt_tokens: 0,
                    completion_tokens: 0,
                },
            )
            .await
            .expect("mc");

        let backend = MockBackend::from_turns(vec![finish_call(
            "cf",
            serde_json::json!({ "disposition": "done", "summary": "task complete" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5).with_checks(passing_runner());

        let result = resume(
            &backend,
            &tools,
            &ctx,
            &config,
            store,
            "happy-task:1",
            ResumeMode::Crash,
        )
        .await
        .expect("resume must succeed");

        // Must reach a verified Done — the claim-vs-verify loop fired.
        assert!(
            matches!(
                result.outcome,
                LoopOutcome::Finished(Disposition::Done {
                    verification: crate::run_record::Verification::Checks(_),
                    ..
                })
            ),
            "crash resume must reach Finished(Done{{Checks(green)}}); got {:?}",
            result.outcome
        );
    }

    // 0.3.0-4: Kill-and-resume proof — the release-gate deterministic integration test.
    //
    // This test proves the crash-resume capability introduced by items 0.3.0-1/2/3.
    // It runs WITHOUT #[ignore] and WITHOUT an env-var gate — it is always in CI.
    //
    // Trajectory:
    //   Leg 1 (pre-crash):  model calls panicky-tool → tool panics → process "dies"
    //   Leg 2 (post-resume): resume() reconciles the dangling tail, model calls finish
    //
    // All storage is FILE-BACKED (tempfile::TempDir + SqliteRunStore::open).
    // No live model, no network.
    #[tokio::test]
    #[allow(clippy::too_many_lines)]
    async fn kill_and_resume_proof_deterministic_integration() {
        use std::sync::atomic::{AtomicU32, Ordering};

        // ---- Test-local panicking tool ------------------------------------
        // Follows the EchoTool test pattern (tool.rs:~388).
        // On its FIRST invocation it increments a shared counter then panics,
        // simulating a host crash mid-tool-execution.
        struct PanickyTool {
            counter: Arc<AtomicU32>,
        }

        #[async_trait]
        impl Tool for PanickyTool {
            #[allow(clippy::unnecessary_literal_bound)]
            fn name(&self) -> &str {
                "panicky-tool"
            }

            fn schema(&self) -> serde_json::Value {
                serde_json::json!({
                    "name": "panicky-tool",
                    "description": "test tool that panics on first invocation (kill-resume proof)",
                    "input_schema": { "type": "object", "properties": {}, "required": [] }
                })
            }

            async fn run(&self, _input: serde_json::Value, _ctx: &ToolCtx) -> ToolResult {
                self.counter.fetch_add(1, Ordering::SeqCst);
                panic!("simulated host kill — crash-resume proof leg 1");
            }
        }

        // ---- Constants ---------------------------------------------------
        // run_id format is {task_id}:{attempt_n} (D4).
        const TASK_ID: &str = "kill-resume-task";
        const ATTEMPT_N: u32 = 1;
        const RUN_ID_LITERAL: &str = "kill-resume-task:1";
        const PANICKY_TOOL_NAME: &str = "panicky-tool";
        const PANICKY_CALL_ID: &str = "c-panicky-1";
        // Per-turn token amounts (pinned so totals are computable constants).
        const INPUT_TOKENS: u32 = 100;
        const OUTPUT_TOKENS: u32 = 10;
        const TOKENS_PER_TURN: u64 = (INPUT_TOKENS + OUTPUT_TOKENS) as u64; // 110

        // ---- Setup: file-backed store + shared side-effect counter -------
        let db_dir = TempDir::new().expect("create temp dir for SQLite DB");
        let db_path = db_dir.path().join("harness.db");
        let side_effect_counter = Arc::new(AtomicU32::new(0));

        // ---- Leg 1: run until crash (AC-2) -------------------------------
        // Drive leg 1 inside tokio::spawn so we can catch the panic.
        // D5 guarantees ToolCallStarted is appended BEFORE tools.invoke, so
        // the log ends with a dangling ToolCallStarted when the tool panics.
        //
        // The backend is created OUTSIDE the spawn and shared via Arc so that
        // after the crash we can call leg1_backend.systems_seen() and compare
        // it to leg 2 — asserting byte-identity against what leg 1 ACTUALLY
        // sent, not against a fresh in-test reference render (D9 fix).
        let leg1_backend = Arc::new(MockBackend::from_turns(vec![turn_with_usage(
            vec![tool_call(
                PANICKY_CALL_ID,
                PANICKY_TOOL_NAME,
                serde_json::json!({}),
            )],
            StopReason::ToolUse,
            usage_with(INPUT_TOKENS, OUTPUT_TOKENS),
        )]));
        let leg1_backend_for_spawn = leg1_backend.clone();

        let db_path_for_spawn = db_path.clone();
        let counter_for_spawn = side_effect_counter.clone();

        let leg1_handle = tokio::spawn(async move {
            let mut tools = ToolRegistry::new();
            tools.register(
                PANICKY_TOOL_NAME,
                Arc::new(PanickyTool {
                    counter: counter_for_spawn,
                }),
            );
            tools.register(FINISH_TOOL_NAME, Arc::new(FinishTool::default()));
            let ctx = ToolCtx::stub();
            let config = RunConfig::new("do the task", 10).with_checks(passing_runner());
            let store: Arc<dyn RunStore> =
                Arc::new(SqliteRunStore::open(&db_path_for_spawn).expect("open SQLite for leg 1"));
            let pers = Persistence {
                store,
                task_id: TASK_ID.to_string(),
                attempt_n: ATTEMPT_N,
                model_label: "test-model".to_string(),
                backend_settings: None,
            };
            // PanickyTool::run() panics here; the spawned task catches it.
            run_persisted(&*leg1_backend_for_spawn, &tools, &ctx, &config, &pers).await
        });

        // AC-2: Leg 1 must panic.
        let leg1_join = leg1_handle.await;
        assert!(
            leg1_join.is_err(),
            "leg 1 tokio::spawn must return Err (panicked task)"
        );
        assert!(
            leg1_join.unwrap_err().is_panic(),
            "leg 1 JoinError must be a panic, not a cancellation"
        );

        // ---- Verify crash state: log ends with dangling ToolCallStarted --
        // Open a FRESH handle — simulates the harness opening the DB after
        // a process restart.
        let store_post_crash: Arc<dyn RunStore> =
            Arc::new(SqliteRunStore::open(&db_path).expect("open SQLite post-crash"));
        let events_pre_resume = store_post_crash
            .list_events(RUN_ID_LITERAL)
            .await
            .expect("list events after crash");
        assert!(
            !events_pre_resume.is_empty(),
            "event log must not be empty after leg 1 crash"
        );
        let last_pre_crash = events_pre_resume.last().expect("non-empty — just asserted");
        // AC-2: the LAST event in list_events at crash time is ToolCallStarted.
        assert!(
            matches!(
                last_pre_crash,
                Event::ToolCallStarted { name, .. } if name == PANICKY_TOOL_NAME
            ),
            "last event after crash must be ToolCallStarted({PANICKY_TOOL_NAME}); \
             got {last_pre_crash:?}"
        );
        let pre_crash_last_seq = match last_pre_crash {
            Event::ToolCallStarted { seq, .. } => *seq,
            _ => unreachable!("just matched above"),
        };

        // ---- Leg 2: crash-resume (AC-3) ----------------------------------
        // Fresh store handle over the same file — durability proof: no in-memory dodge.
        let store_leg2: Arc<dyn RunStore> =
            Arc::new(SqliteRunStore::open(&db_path).expect("open SQLite for leg 2"));

        // Scripted backend for leg 2: after reconcile feeds the synthetic
        // ToolResult back, the model calls finish immediately.
        let backend_leg2 = MockBackend::from_turns(vec![turn_with_usage(
            vec![tool_call(
                "c-finish-resume",
                FINISH_TOOL_NAME,
                serde_json::json!({
                    "disposition": "done",
                    "summary": "task complete after crash-resume"
                }),
            )],
            StopReason::ToolUse,
            usage_with(INPUT_TOKENS, OUTPUT_TOKENS),
        )]);

        // Same tool registry as leg 1 — D9 byte-identity requires identical
        // tool_lines. Including panicky-tool here: its schema must match so
        // the system prompt is byte-identical. The scripted backend does NOT
        // call it in leg 2, so the counter stays at 1.
        let mut tools_leg2 = ToolRegistry::new();
        tools_leg2.register(
            PANICKY_TOOL_NAME,
            Arc::new(PanickyTool {
                counter: side_effect_counter.clone(),
            }),
        );
        tools_leg2.register(FINISH_TOOL_NAME, Arc::new(FinishTool::default()));

        let ctx_leg2 = ToolCtx::stub();
        let config_leg2 = RunConfig::new("do the task", 10).with_checks(passing_runner());

        let resume_result = resume(
            &backend_leg2,
            &tools_leg2,
            &ctx_leg2,
            &config_leg2,
            store_leg2,
            RUN_ID_LITERAL,
            ResumeMode::Crash,
        )
        .await
        .expect("crash-resume must succeed without error");

        // AC-3: Terminal disposition must be Done{Checks(green)}.
        // A Blocked or Failed terminal fails the test.
        let is_verified_done = matches!(
            &resume_result.outcome,
            LoopOutcome::Finished(Disposition::Done {
                verification: Verification::Checks(report),
                ..
            }) if report.passed
        );
        assert!(
            is_verified_done,
            "crash-resume must terminate in Done{{Checks(green)}}; got {:?}",
            resume_result.outcome
        );

        // ---- AC-4a: Resume-leg transcript contains synthetic ToolResult --
        // D6: reconcile feeds a model::UserBlock::ToolResult{is_error=true,
        // content contains "interrupted"} for the panicky call.
        // The `content` field (not summary) is the assertion target.
        let leg2_messages_seen = backend_leg2.messages_seen();
        assert!(
            !leg2_messages_seen.is_empty(),
            "leg 2 must draw at least one model turn"
        );
        let first_turn = &leg2_messages_seen[0];
        let synthetic_in_transcript = first_turn.iter().any(|msg| match msg {
            Message::User { content } => content.iter().any(|b| {
                matches!(
                    b,
                    UserBlock::ToolResult {
                        is_error: true,
                        call_id,
                        content: c,
                        ..
                    } if call_id == PANICKY_CALL_ID && c.contains("interrupted")
                )
            }),
            Message::Assistant { .. } => false,
        });
        assert!(
            synthetic_in_transcript,
            "first turn of leg 2 must include \
             UserBlock::ToolResult{{is_error=true, call_id={PANICKY_CALL_ID:?}, \
             content contains 'interrupted'}} (D6); messages: {first_turn:?}"
        );

        // ---- Collect final event log for remaining assertions ------------
        let store_final: Arc<dyn RunStore> =
            Arc::new(SqliteRunStore::open(&db_path).expect("open SQLite for final assertions"));
        let all_events = store_final
            .list_events(RUN_ID_LITERAL)
            .await
            .expect("list all events after both legs");

        // ---- AC-4b: Exactly ONE synthetic ToolCallResult (is_error=true, "interrupted") --
        // reconcile_crash_tail appends exactly one synthetic ToolCallResult for
        // the dangling ToolCallStarted.
        let synthetic_results: Vec<_> = all_events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    Event::ToolCallResult {
                        is_error: true,
                        name,
                        summary,
                        ..
                    } if name == PANICKY_TOOL_NAME && summary.contains("interrupted")
                )
            })
            .collect();
        assert_eq!(
            synthetic_results.len(),
            1,
            "event log must contain exactly ONE synthetic ToolCallResult \
             (is_error=true, name={PANICKY_TOOL_NAME:?}, summary contains 'interrupted'); \
             found {}: {synthetic_results:?}",
            synthetic_results.len()
        );

        // ---- AC-4c: Side-effect counter == 1 (no blind re-execution) ----
        assert_eq!(
            side_effect_counter.load(Ordering::SeqCst),
            1,
            "panicky-tool side-effect counter must be exactly 1 after both legs; \
             the interrupted call must never be blindly re-executed"
        );

        // ---- AC-4d: Every ToolCallStarted has a matching ToolCallResult --
        // After reconcile, the final log has no unpaired ToolCallStarted events:
        // every Started is followed by a Result before the next ModelCall or
        // end-of-log. Walk the events to verify.
        {
            let mut dangling: Option<&str> = None;
            for event in &all_events {
                match event {
                    Event::ToolCallStarted { name, .. } => {
                        assert!(
                            dangling.is_none(),
                            "ToolCallStarted({name}) found while {dangling:?} is still unpaired"
                        );
                        dangling = Some(name.as_str());
                    }
                    Event::ToolCallResult { name, .. } => {
                        assert!(
                            dangling.is_some(),
                            "ToolCallResult({name}) has no matching ToolCallStarted"
                        );
                        dangling = None;
                    }
                    Event::ModelCall { .. } => {
                        assert!(
                            dangling.is_none(),
                            "ModelCall arrived while ToolCallStarted({dangling:?}) is still unpaired"
                        );
                    }
                    Event::PhaseTransition { .. }
                    | Event::BudgetTick { .. }
                    | Event::DispositionSet { .. } => {}
                }
            }
            assert!(
                dangling.is_none(),
                "event log ends with an unpaired ToolCallStarted: {dangling:?}"
            );
        }

        // ---- AC-5: Budget continuity — pinned integer literals -----------
        // Leg 1: 1 turn × (INPUT + OUTPUT) tokens = TOKENS_PER_TURN, 1 iteration.
        // Leg 2: 1 turn × (INPUT + OUTPUT) tokens = TOKENS_PER_TURN, 1 iteration.
        // Total: iterations = 2, tokens = 2 × TOKENS_PER_TURN = 220, cost_micros = 0.
        // item 0.3.0-3 wires no pricing, so cost_micros == 0.
        let final_record = store_final
            .load(RUN_ID_LITERAL)
            .await
            .expect("load final record")
            .expect("record must exist after both legs");
        assert_eq!(
            final_record.budgets.consumed.iterations, 2,
            "consumed.iterations must be exactly 2 (1 pre-crash + 1 post-resume); \
             got {:?}",
            final_record.budgets.consumed
        );
        assert_eq!(
            final_record.budgets.consumed.tokens,
            2 * TOKENS_PER_TURN,
            "consumed.tokens must be exactly {} ({}×{} per turn × 2 turns); got {:?}",
            2 * TOKENS_PER_TURN,
            INPUT_TOKENS + OUTPUT_TOKENS,
            2,
            final_record.budgets.consumed
        );
        assert_eq!(
            final_record.budgets.consumed.cost_micros, 0,
            "consumed.cost_micros must be 0 (no pricing wired in 0.3.0); \
             got {:?}",
            final_record.budgets.consumed
        );

        // ---- AC-6: Seq monotonicity — no reset at the resume boundary ----
        // Extract the monotonic seq from every event in the final log.
        let seqs: Vec<u64> = all_events
            .iter()
            .map(|e| match e {
                Event::ModelCall { seq, .. }
                | Event::ToolCallStarted { seq, .. }
                | Event::ToolCallResult { seq, .. }
                | Event::PhaseTransition { seq, .. }
                | Event::BudgetTick { seq, .. }
                | Event::DispositionSet { seq, .. } => *seq,
            })
            .collect();

        // Strictly increasing — no duplicates, no resets at the resume boundary.
        for (i, window) in seqs.windows(2).enumerate() {
            assert!(
                window[1] > window[0],
                "seq is not strictly increasing at position {i}: {} -> {}; \
                 full seq list: {seqs:?}",
                window[0],
                window[1]
            );
        }

        // The first event appended by the resumed loop has seq exactly one
        // greater than the last event persisted before the crash.
        // grounded in store.rs:322-323 (COALESCE(MAX(seq),-1)+1 per run).
        // events_pre_resume.len() events were written in leg 1;
        // seqs[events_pre_resume.len()] is the first seq added during resume.
        let first_resumed_seq = seqs[events_pre_resume.len()];
        assert_eq!(
            first_resumed_seq,
            pre_crash_last_seq + 1,
            "first event appended by resumed loop (seq={first_resumed_seq}) must be \
             exactly one greater than the last pre-crash event \
             (seq={pre_crash_last_seq})"
        );

        // ---- AC-7: D9 System prompt byte-identity across the restart -----
        // Leg 2's system prompt must be byte-identical to what leg 1 ACTUALLY
        // sent — not to a fresh in-test reference render.  leg1_backend was
        // captured via Arc before the spawn, so systems_seen() reflects the
        // real bytes that the engine transmitted during leg 1.
        let leg1_systems = leg1_backend.systems_seen();
        assert!(
            !leg1_systems.is_empty(),
            "leg 1 must have sent at least one system prompt"
        );
        let leg1_first_system = leg1_systems[0]
            .as_ref()
            .expect("leg 1 turn 0 must send a non-None system prompt");
        let leg2_systems = backend_leg2.systems_seen();
        assert!(
            !leg2_systems.is_empty(),
            "leg 2 must observe at least one system prompt"
        );
        for (i, entry) in leg2_systems.iter().enumerate() {
            let seen = entry
                .as_ref()
                .unwrap_or_else(|| panic!("leg 2 turn {i} must send a non-None system prompt"));
            assert_eq!(
                seen.as_bytes(),
                leg1_first_system.as_bytes(),
                "leg 2 turn {i} system prompt must be byte-identical to what \
                 leg 1 ACTUALLY sent (D9 prompt-cache invariant)"
            );
        }
    }

    // Two-call interrupted turn: first call completes (real result logged),
    // crash before the second call's result. Post-resume the completed call's
    // REAL result content survives in the transcript; the second call gets
    // exactly one synthetic is_error result. Run completes Done{Checks(green)}.
    //
    // This is the sibling of kill_and_resume_proof_deterministic_integration
    // extended to the two-tool-call scenario that the original test skipped
    // (a single-call turn where the buggy all-synthetic output coincides with
    // the correct output — the bug was invisible).
    #[tokio::test]
    #[allow(clippy::too_many_lines)]
    async fn kill_and_resume_two_call_interrupted_turn() {
        use std::sync::atomic::{AtomicU32, Ordering};

        // ---- Test-local panicky tool (always panics on invocation) ---------
        struct PanickyTool {
            counter: Arc<AtomicU32>,
        }

        #[async_trait]
        impl Tool for PanickyTool {
            #[allow(clippy::unnecessary_literal_bound)]
            fn name(&self) -> &str {
                "panicky-tool"
            }

            fn schema(&self) -> serde_json::Value {
                serde_json::json!({
                    "name": "panicky-tool",
                    "description": "test tool that always panics (two-call kill-resume proof)",
                    "input_schema": { "type": "object", "properties": {}, "required": [] }
                })
            }

            async fn run(&self, _input: serde_json::Value, _ctx: &ToolCtx) -> ToolResult {
                self.counter.fetch_add(1, Ordering::SeqCst);
                panic!("simulated host kill — two-call crash-resume proof");
            }
        }

        // ---- Constants ---------------------------------------------------
        const TASK_ID: &str = "two-call-kill-resume-task";
        const ATTEMPT_N: u32 = 1;
        const RUN_ID_LITERAL: &str = "two-call-kill-resume-task:1";
        const ECHO_CALL_ID: &str = "c-echo-1";
        const PANICKY_CALL_ID: &str = "c-panicky-1";
        const PANICKY_TOOL_NAME: &str = "panicky-tool";
        // EchoTool returns input.to_string() as its summary.
        // Called with json!({}) the summary stored in the event is "{}".
        const ECHO_EXPECTED_SUMMARY: &str = "{}";

        // ---- Setup -------------------------------------------------------
        let db_dir = TempDir::new().expect("create temp dir");
        let db_path = db_dir.path().join("harness.db");
        let panicky_counter = Arc::new(AtomicU32::new(0));

        // ---- Leg 1: run until crash ----------------------------------------
        // Model turn: [echo-call, panicky-call]. EchoTool completes; PanickyTool panics.
        let db_path_for_spawn = db_path.clone();
        let counter_for_spawn = panicky_counter.clone();

        let leg1_handle = tokio::spawn(async move {
            let backend = MockBackend::from_turns(vec![turn_with(
                vec![
                    tool_call(ECHO_CALL_ID, "echo", serde_json::json!({})),
                    tool_call(PANICKY_CALL_ID, PANICKY_TOOL_NAME, serde_json::json!({})),
                ],
                StopReason::ToolUse,
            )]);
            let mut tools = ToolRegistry::new();
            tools.register("echo", Arc::new(EchoTool));
            tools.register(
                PANICKY_TOOL_NAME,
                Arc::new(PanickyTool {
                    counter: counter_for_spawn,
                }),
            );
            tools.register(FINISH_TOOL_NAME, Arc::new(FinishTool::default()));
            let ctx = ToolCtx::stub();
            let config = RunConfig::new("do the task", 10).with_checks(passing_runner());
            let store: Arc<dyn RunStore> =
                Arc::new(SqliteRunStore::open(&db_path_for_spawn).expect("open SQLite for leg 1"));
            let pers = Persistence {
                store,
                task_id: TASK_ID.to_string(),
                attempt_n: ATTEMPT_N,
                model_label: "test-model".to_string(),
                backend_settings: None,
            };
            run_persisted(&backend, &tools, &ctx, &config, &pers).await
        });

        // Leg 1 must panic (PanickyTool always panics).
        let leg1_join = leg1_handle.await;
        assert!(leg1_join.is_err(), "leg 1 must return Err (panicked task)");
        assert!(
            leg1_join.unwrap_err().is_panic(),
            "leg 1 JoinError must be a panic"
        );

        // ---- Verify crash state ------------------------------------------
        // Log must end with a dangling ToolCallStarted for panicky-tool.
        let store_post_crash: Arc<dyn RunStore> =
            Arc::new(SqliteRunStore::open(&db_path).expect("open post-crash"));
        let events_pre_resume = store_post_crash
            .list_events(RUN_ID_LITERAL)
            .await
            .expect("list events");
        let last_pre_crash = events_pre_resume.last().expect("non-empty log after crash");
        assert!(
            matches!(
                last_pre_crash,
                Event::ToolCallStarted { name, .. } if name == PANICKY_TOOL_NAME
            ),
            "last event after crash must be ToolCallStarted({PANICKY_TOOL_NAME}); \
             got {last_pre_crash:?}"
        );

        // The echo call must have a ToolCallResult with is_error=false in the log.
        let echo_result_in_log = events_pre_resume.iter().any(|e| {
            matches!(
                e,
                Event::ToolCallResult {
                    name,
                    is_error: false,
                    ..
                } if name == "echo"
            )
        });
        assert!(
            echo_result_in_log,
            "echo ToolCallResult(is_error=false) must be in log before resume; \
             events: {events_pre_resume:?}"
        );

        // ---- Leg 2: crash-resume -----------------------------------------
        let store_leg2: Arc<dyn RunStore> =
            Arc::new(SqliteRunStore::open(&db_path).expect("open SQLite for leg 2"));

        let backend_leg2 = MockBackend::from_turns(vec![turn_with(
            vec![tool_call(
                "c-finish-resume",
                FINISH_TOOL_NAME,
                serde_json::json!({
                    "disposition": "done",
                    "summary": "task complete after two-call crash-resume"
                }),
            )],
            StopReason::ToolUse,
        )]);

        let mut tools_leg2 = ToolRegistry::new();
        tools_leg2.register("echo", Arc::new(EchoTool));
        tools_leg2.register(
            PANICKY_TOOL_NAME,
            Arc::new(PanickyTool {
                counter: panicky_counter.clone(),
            }),
        );
        tools_leg2.register(FINISH_TOOL_NAME, Arc::new(FinishTool::default()));

        let ctx_leg2 = ToolCtx::stub();
        let config_leg2 = RunConfig::new("do the task", 10).with_checks(passing_runner());

        let resume_result = resume(
            &backend_leg2,
            &tools_leg2,
            &ctx_leg2,
            &config_leg2,
            store_leg2,
            RUN_ID_LITERAL,
            ResumeMode::Crash,
        )
        .await
        .expect("crash-resume must succeed");

        // Terminal disposition must be Done{Checks(green)}.
        let is_verified_done = matches!(
            &resume_result.outcome,
            LoopOutcome::Finished(Disposition::Done {
                verification: Verification::Checks(report),
                ..
            }) if report.passed
        );
        assert!(
            is_verified_done,
            "two-call crash-resume must terminate in Done{{Checks(green)}}; \
             got {:?}",
            resume_result.outcome
        );

        // First leg-2 turn must see the reconciled user message with two blocks:
        //   [0] echo-call: REAL result (content = ECHO_EXPECTED_SUMMARY, is_error=false)
        //   [1] panicky-call: synthetic is_error=true, content contains "interrupted"
        let leg2_messages = backend_leg2.messages_seen();
        assert!(
            !leg2_messages.is_empty(),
            "leg 2 must draw at least one turn"
        );
        let first_turn = &leg2_messages[0];

        // Find the reconciled User message (last User before the first Assistant).
        let reconciled_user = first_turn
            .iter()
            .find(|m| {
                matches!(m, Message::User { content }
                    if content.iter().any(|b| matches!(b, UserBlock::ToolResult { .. })))
            })
            .expect("first leg-2 turn must include a User message with ToolResult blocks");

        let Message::User { content: blocks } = reconciled_user else {
            panic!("expected User message");
        };
        assert_eq!(
            blocks.len(),
            2,
            "reconciled User must have exactly 2 ToolResult blocks; got {blocks:?}"
        );

        // Block 0: echo-call, real result.
        match &blocks[0] {
            UserBlock::ToolResult {
                call_id,
                content,
                is_error,
            } => {
                assert_eq!(call_id, ECHO_CALL_ID, "block[0] call_id must be echo");
                assert!(
                    !is_error,
                    "echo call completed successfully — is_error must be false"
                );
                assert_eq!(
                    content, ECHO_EXPECTED_SUMMARY,
                    "echo call must carry its REAL result content, not synthetic"
                );
            }
            UserBlock::Text(_) => panic!("expected ToolResult block 0; got Text"),
        }

        // Block 1: panicky-call, synthetic is_error.
        match &blocks[1] {
            UserBlock::ToolResult {
                call_id,
                content,
                is_error,
            } => {
                assert_eq!(call_id, PANICKY_CALL_ID, "block[1] call_id must be panicky");
                assert!(
                    *is_error,
                    "panicky call was interrupted — is_error must be true"
                );
                assert!(
                    content.contains("interrupted"),
                    "panicky call content must contain 'interrupted'; got {content:?}"
                );
            }
            UserBlock::Text(_) => panic!("expected ToolResult block 1; got Text"),
        }

        // PanickyTool must have been invoked exactly once (leg 1).
        // Resume does NOT re-execute it.
        assert_eq!(
            panicky_counter.load(Ordering::SeqCst),
            1,
            "panicky-tool must be invoked exactly once (leg 1 only, never re-executed on resume)"
        );
    }

    /// Builds the mid-turn snapshot + log shape shared by the entry-gate
    /// regression tests: a record whose last message is an assistant turn
    /// with the given tool calls, and a log of
    /// `[ModelCall, BudgetTick, Started(a), Result(a)]` — i.e. call `a`
    /// completed, and the log's LAST event is a `ToolCallResult`.
    async fn seed_result_tail_store(
        task_id: &str,
        calls: &[(&str, &str)], // (call_id, name)
    ) -> (RunRecord, Arc<dyn RunStore>, String) {
        let run_id = format!("{task_id}:1");
        let asst = Message::Assistant {
            content: calls
                .iter()
                .map(|(id, name)| {
                    ContentBlock::ToolCall(ToolCallRequest {
                        id: (*id).to_string(),
                        name: (*name).to_string(),
                        input: serde_json::json!({}),
                    })
                })
                .collect(),
        };
        let mut record = make_minimal_record(task_id, 1);
        record.messages = vec![
            Message::User {
                content: vec![UserBlock::Text("do the task".to_string())],
            },
            asst,
        ];
        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        store
            .checkpoint(&run_id, &record)
            .await
            .expect("checkpoint");
        store
            .append_event(
                &run_id,
                Event::ModelCall {
                    seq: 0,
                    model: "t".to_string(),
                    prompt_tokens: 1,
                    completion_tokens: 1,
                },
            )
            .await
            .expect("mc");
        store
            .append_event(
                &run_id,
                Event::BudgetTick {
                    seq: 0,
                    consumed: BudgetConsumed::default(),
                },
            )
            .await
            .expect("bt");
        store
            .append_event(
                &run_id,
                Event::ToolCallStarted {
                    seq: 0,
                    name: calls[0].1.to_string(),
                    args: serde_json::json!({}),
                    call_id: calls[0].0.to_string(),
                },
            )
            .await
            .expect("started-a");
        store
            .append_event(
                &run_id,
                Event::ToolCallResult {
                    seq: 0,
                    name: calls[0].1.to_string(),
                    is_error: false,
                    summary: "real output a".to_string(),
                    offload_path: None,
                },
            )
            .await
            .expect("result-a");
        (record, store, run_id)
    }

    // Entry-gate regression (review follow-up): a crash can land BETWEEN a
    // ToolCallResult and the next ToolCallStarted. The log then ends in a
    // Result, but the snapshot still dangles (assistant turn, no results
    // batch). Gating on `events.last() == ToolCallStarted` skipped
    // reconciliation here and returned a transcript ending in unanswered
    // tool calls — malformed on every backend. The gate is snapshot-shape.
    #[tokio::test]
    async fn crash_between_result_and_next_started_reconciles() {
        let (record, store, run_id) =
            seed_result_tail_store("gate-task", &[("id-a", "work"), ("id-b", "work")]).await;
        // Log deliberately ends at Result(a): call b never reached Started.

        let msgs = super::reconcile_crash_tail(&record, &store, &run_id)
            .await
            .expect("reconcile");

        assert_eq!(msgs.len(), 3, "seed + asst + reconciled results batch");
        let Message::User { content } = &msgs[2] else {
            panic!("third message must be User");
        };
        assert_eq!(content.len(), 2, "one block per assistant tool call");
        match &content[0] {
            UserBlock::ToolResult {
                call_id,
                content,
                is_error,
            } => {
                assert_eq!(call_id, "id-a");
                assert_eq!(
                    content, "real output a",
                    "completed call keeps its real result"
                );
                assert!(!is_error);
            }
            UserBlock::Text(_) => panic!("expected ToolResult"),
        }
        match &content[1] {
            UserBlock::ToolResult {
                call_id, is_error, ..
            } => {
                assert_eq!(call_id, "id-b");
                assert!(*is_error, "never-started call gets synthetic is_error");
            }
            UserBlock::Text(_) => panic!("expected ToolResult"),
        }
    }

    // Entry-gate regression, all-completed flavor: crash after the final
    // Result but before the end-of-iteration checkpoint. Every call has a
    // real logged result; reconciliation must rebuild the results batch with
    // no synthetic blocks and append no synthetic log events.
    #[tokio::test]
    async fn crash_after_all_results_before_iteration_checkpoint_reconciles() {
        let (record, store, run_id) =
            seed_result_tail_store("gate-task-2", &[("id-a", "work")]).await;
        let events_before = store.list_events(&run_id).await.expect("events").len();

        let msgs = super::reconcile_crash_tail(&record, &store, &run_id)
            .await
            .expect("reconcile");

        let Message::User { content } = &msgs[2] else {
            panic!("third message must be User");
        };
        assert_eq!(content.len(), 1);
        match &content[0] {
            UserBlock::ToolResult {
                call_id,
                content,
                is_error,
            } => {
                assert_eq!(call_id, "id-a");
                assert_eq!(content, "real output a");
                assert!(!is_error, "completed call must NOT be marked interrupted");
            }
            UserBlock::Text(_) => panic!("expected ToolResult"),
        }
        let events_after = store.list_events(&run_id).await.expect("events").len();
        assert_eq!(
            events_before, events_after,
            "no synthetic event when nothing dangles"
        );
    }

    // Entry-gate clean cases: a snapshot ending in a User results batch (the
    // end-of-iteration checkpoint landed) or an assistant turn with NO tool
    // calls must pass through untouched.
    #[tokio::test]
    async fn clean_snapshots_are_not_reconciled() {
        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        let mut record = make_minimal_record("clean-task", 1);
        record.messages = vec![
            Message::User {
                content: vec![UserBlock::Text("do the task".to_string())],
            },
            Message::Assistant {
                content: vec![ContentBlock::Text("thinking out loud".to_string())],
            },
        ];
        let msgs = super::reconcile_crash_tail(&record, &store, "clean-task:1")
            .await
            .expect("reconcile");
        assert_eq!(msgs, record.messages, "no-tool-call turn passes through");
    }

    // Malformed-tail protection: reconcile_crash_tail must not panic or
    // index-out-of-bounds when the log contains:
    //   (a) a ToolCallResult with no preceding ToolCallStarted (orphaned result)
    //   (b) duplicate call_ids in ToolCallStarted events
    //
    // In both cases the reconciler must degrade safely and return a valid
    // (possibly all-synthetic) message list.
    #[allow(clippy::too_many_lines)]
    #[tokio::test]
    async fn reconcile_crash_tail_malformed_tail_no_panic() {
        // Build a record with one assistant tool call.
        let task_seed = Message::User {
            content: vec![UserBlock::Text("do the task".to_string())],
        };
        let asst = Message::Assistant {
            content: vec![ContentBlock::ToolCall(ToolCallRequest {
                id: "id-a".to_string(),
                name: "work".to_string(),
                input: serde_json::json!({}),
            })],
        };
        let mut record = make_minimal_record("malformed-task", 1);
        record.messages = vec![task_seed, asst];

        // ---- Case (a): orphaned ToolCallResult (no matching Started) -------
        // Log ends with ToolCallStarted (so dangling check fires) but before it
        // there's a ToolCallResult with no Started. The reconciler must not panic.
        {
            let store: Arc<dyn RunStore> =
                Arc::new(SqliteRunStore::open_in_memory().expect("open"));
            store
                .checkpoint("malformed-task:1", &record)
                .await
                .expect("checkpoint");

            // Seed an orphaned ToolCallResult followed by a dangling Started.
            store
                .append_event(
                    "malformed-task:1",
                    Event::ModelCall {
                        seq: 0,
                        model: "t".to_string(),
                        prompt_tokens: 1,
                        completion_tokens: 1,
                    },
                )
                .await
                .expect("mc");
            // Orphaned result — no Started before it.
            store
                .append_event(
                    "malformed-task:1",
                    Event::ToolCallResult {
                        seq: 0,
                        name: "work".to_string(),
                        is_error: false,
                        summary: "orphaned".to_string(),
                        offload_path: None,
                    },
                )
                .await
                .expect("orphaned result");
            // Dangling Started — makes events.last() a ToolCallStarted.
            store
                .append_event(
                    "malformed-task:1",
                    Event::ToolCallStarted {
                        seq: 0,
                        name: "work".to_string(),
                        args: serde_json::json!({}),
                        call_id: "id-a".to_string(),
                    },
                )
                .await
                .expect("dangling started");

            // Must not panic.
            let msgs = super::reconcile_crash_tail(&record, &store, "malformed-task:1")
                .await
                .expect("reconcile must not fail on orphaned result");

            // Should return task_seed + asst + reconciled user (all-synthetic for id-a).
            assert_eq!(
                msgs.len(),
                3,
                "must return 3 messages (seed + asst + reconciled)"
            );
            let Message::User { content } = &msgs[2] else {
                panic!("third message must be User");
            };
            assert_eq!(content.len(), 1, "one ToolResult block for id-a");
            match &content[0] {
                UserBlock::ToolResult {
                    call_id, is_error, ..
                } => {
                    assert_eq!(call_id, "id-a");
                    // The orphaned result is ignored; the dangling Started has
                    // call_id=id-a with no matching Result → synthetic is_error.
                    assert!(
                        *is_error,
                        "id-a must be synthetic is_error (dangling Started)"
                    );
                }
                UserBlock::Text(_) => panic!("expected ToolResult block"),
            }
        }

        // ---- Case (b): duplicate call_ids in ToolCallStarted ---------------
        {
            let store: Arc<dyn RunStore> =
                Arc::new(SqliteRunStore::open_in_memory().expect("open"));
            store
                .checkpoint("malformed-task:1", &record)
                .await
                .expect("checkpoint");

            store
                .append_event(
                    "malformed-task:1",
                    Event::ModelCall {
                        seq: 0,
                        model: "t".to_string(),
                        prompt_tokens: 1,
                        completion_tokens: 1,
                    },
                )
                .await
                .expect("mc");
            // First Started(id-a).
            store
                .append_event(
                    "malformed-task:1",
                    Event::ToolCallStarted {
                        seq: 0,
                        name: "work".to_string(),
                        args: serde_json::json!({}),
                        call_id: "id-a".to_string(),
                    },
                )
                .await
                .expect("started1");
            // Duplicate Started(id-a) — makes events.last() a ToolCallStarted.
            store
                .append_event(
                    "malformed-task:1",
                    Event::ToolCallStarted {
                        seq: 0,
                        name: "work".to_string(),
                        args: serde_json::json!({}),
                        call_id: "id-a".to_string(),
                    },
                )
                .await
                .expect("started2_duplicate");

            // Must not panic.
            let msgs = super::reconcile_crash_tail(&record, &store, "malformed-task:1")
                .await
                .expect("reconcile must not fail on duplicate call_id");

            assert_eq!(msgs.len(), 3, "must return 3 messages");
            let Message::User { content } = &msgs[2] else {
                panic!("third message must be User");
            };
            assert_eq!(content.len(), 1, "one ToolResult block for id-a");
            match &content[0] {
                UserBlock::ToolResult {
                    call_id, is_error, ..
                } => {
                    assert_eq!(call_id, "id-a");
                    assert!(
                        *is_error,
                        "id-a must be synthetic is_error with duplicate Started"
                    );
                }
                UserBlock::Text(_) => panic!("expected ToolResult block"),
            }
        }
    }

    // =====================================================================
    // Finish-recovery protocol tests (detection, nudge, recovery terminal)
    // =====================================================================

    /// The exact nudge wording the harness injects — pinned by a test so a
    /// wording pass that drifts fails loudly. Must match
    /// `crates/harness/templates/nudge_prompt.md` verbatim.
    const NUDGE_TEXT: &str = "The quality gates are currently green. \
        If the acceptance criteria are met, call `finish(done)` now. \
        If nothing needed changing because the task was already complete, \
        call `finish(already_satisfied)` with a `reason`. \
        If they are not yet met, reply with a one-sentence status: \
        what remains, and why you are still working.";

    /// Count how many `UserBlock::Text` blocks in `messages` carry the nudge
    /// text — the harness injects the nudge by APPENDING onto the existing
    /// tool-results `Message::User`, so this counts nudge injections (not new
    /// messages).
    fn count_nudge_injections(messages: &[Message]) -> usize {
        messages
            .iter()
            .map(|m| match m {
                Message::User { content } => content
                    .iter()
                    .filter(|b| matches!(b, UserBlock::Text(t) if t == NUDGE_TEXT))
                    .count(),
                Message::Assistant { .. } => 0,
            })
            .sum()
    }

    /// Assert no two adjacent `Message::User` appear in `messages` — the
    /// nudge must be APPENDED onto the existing tool-results `Message::User`,
    /// never pushed as a new `Message::User` (which would hit the Anthropic
    /// wire as two `role:"user"` blocks and 400).
    fn assert_no_adjacent_user_messages(messages: &[Message]) {
        for i in 0..messages.len().saturating_sub(1) {
            let a = matches!(messages[i], Message::User { .. });
            let b = matches!(messages[i + 1], Message::User { .. });
            assert!(
                !(a && b),
                "two adjacent Message::User at indices {i} and {} — nudge \
                 must be appended, not pushed as a new message",
                i + 1
            );
        }
    }

    /// A turn that calls `run_checks` (the done-oracle). Used to drive the
    /// green/red gate signal in detection tests.
    fn run_checks_turn(id: &str) -> AssistantTurn {
        turn_with(
            vec![tool_call(id, "run_checks", serde_json::json!({}))],
            StopReason::ToolUse,
        )
    }

    /// A turn that emits a one-sentence status text and a non-mutating echo
    /// call — the post-nudge "status reply" (no green re-observation).
    fn status_echo_turn(id: &str, status: &str) -> AssistantTurn {
        turn_with(
            vec![
                ContentBlock::Text(status.to_string()),
                tool_call(id, "echo", serde_json::json!({})),
            ],
            StopReason::ToolUse,
        )
    }

    /// A non-mutating echo turn (no text — `turn.text()` is empty).
    fn echo_turn(id: &str) -> AssistantTurn {
        turn_with(
            vec![tool_call(id, "echo", serde_json::json!({}))],
            StopReason::ToolUse,
        )
    }

    // AC-1 detection test: green run_checks then K static non-mutating
    // iterations -> exactly one nudge appended (pinned text appears once,
    // no two adjacent Message::User).
    #[tokio::test]
    async fn green_static_spin_injects_exactly_one_nudge_with_pinned_text() {
        // K = DEFAULT_STATIC_TREE_K = 3. The counter increments every
        // non-mutating iteration; the run_checks iteration counts as one.
        //   iter 1: run_checks (green) — iters 0->1
        //   iter 2: echo — iters 1->2
        //   iter 3: echo — iters 2->3 -> trip, nudge injected (nudges_fired 0->1)
        //   iter 4: echo — iters 0->1 (post-nudge; nudge_awaiting_status cleared)
        //   max_iterations=4 -> MaxIterations
        let runner = passing_runner();
        let tools = standard_registry(Some(runner.clone()));
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 4)
            .with_checks(runner)
            .with_max_nudges(2);

        let backend = MockBackend::from_turns(vec![
            run_checks_turn("c1"),
            echo_turn("c2"),
            echo_turn("c3"),
            echo_turn("c4"),
        ]);

        let snap_store = Arc::new(SnapshotStore::new());
        let pers = make_persistence(snap_store.clone());
        let RunResult { outcome, stats } = run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        // The nudge is NOT a terminal — after one nudge the loop continues and
        // hits MaxIterations (no finish, cap reached).
        assert!(
            matches!(outcome, LoopOutcome::MaxIterations),
            "one-nudge spin should hit MaxIterations, not the recovery terminal; got {outcome:?}"
        );
        assert_eq!(stats.nudges_fired, 1, "exactly one nudge fired");
        assert!(
            !stats.tree_dirty,
            "no edit_file/bash ran — tree_dirty must be false"
        );
        assert_eq!(stats.mutating_iters, 0, "no mutating iterations");
        assert_eq!(stats.bash_calls_ok, 0, "no bash calls");
        assert_eq!(stats.edit_file_calls_ok, 0, "no edit_file calls");
        assert_eq!(
            stats.peak_iters_since_tree_change, 3,
            "trip fires at iters==3 then resets — peak must be 3"
        );

        let rec = snap_store
            .inner
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        let nudge_count = count_nudge_injections(&rec.messages);
        assert_eq!(
            nudge_count, 1,
            "exactly one nudge should be injected; got {nudge_count}"
        );
        assert_no_adjacent_user_messages(&rec.messages);
        // GREEN-static MaxIterations now captures recovery_facts (hardened):
        // finish-recovery was on but only one nudge fired, so the
        // FinishDiscipline terminal did NOT trip and the loop fell through to
        // MaxIterations on a green gate. The WIP is preserved.
        let facts = rec
            .recovery_facts
            .as_ref()
            .expect("green-static MaxIterations must capture recovery_facts");
        assert!(
            facts.gates_green_at_exit,
            "gates_green_at_exit must be true on a green-static MaxIterations"
        );
        assert!(
            !facts.tree_dirty,
            "only echo turns ran, so tree_dirty must be false"
        );
    }

    // AC-2 detection test: finish(done) on the turn after a nudge -> the run
    // terminates as a NORMAL Finished(Done{..}) through the existing
    // handle_finish_call path. The recovery terminal is NOT taken, and
    // recovery_facts stays None (a Done-after-nudge is a clean success).
    #[tokio::test]
    async fn finish_done_after_nudge_terminates_normally_without_recovery() {
        //   iter 1: run_checks (green) — iters 0->1
        //   iter 2: echo — iters 1->2
        //   iter 3: echo — iters 2->3 -> trip, nudge injected. nudge_awaiting_status=true.
        //   iter 4: finish(done) — accepted (green checks). nudge_awaiting_status
        //          cleared WITHOUT pushing (is_done). Terminal: Finished(Done).
        let runner = passing_runner();
        let tools = standard_registry(Some(runner.clone()));
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5)
            .with_checks(runner)
            .with_max_nudges(2);

        let backend = MockBackend::from_turns(vec![
            run_checks_turn("c1"),
            echo_turn("c2"),
            echo_turn("c3"),
            finish_call(
                "c4",
                serde_json::json!({ "disposition": "done", "summary": "done after nudge" }),
            ),
        ]);

        let snap_store = Arc::new(SnapshotStore::new());
        let pers = make_persistence(snap_store.clone());
        let RunResult { outcome, .. } = run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        match outcome {
            LoopOutcome::Finished(Disposition::Done { summary, .. }) => {
                assert_eq!(summary, "done after nudge");
            }
            other => panic!("expected Finished(Done), got {other:?}"),
        }

        let rec = snap_store
            .inner
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        assert_eq!(rec.recovery_facts, None);
        assert!(
            matches!(rec.disposition, Some(Disposition::Done { .. })),
            "persisted disposition must be Done; got {:?}",
            rec.disposition
        );
        // The nudge was injected (once) before the finish.
        assert_eq!(count_nudge_injections(&rec.messages), 1);
    }

    // AC-3 detection test: N nudges then a further K static-green iterations
    // with no finish -> LoopOutcome::Finished(Disposition::Failed{
    // FinishDiscipline }) and record.recovery_facts == Some(RecoveryFacts{
    // gates_green_at_exit: true, tree_dirty: <reflects edits>, nudge_statuses:
    // <non-empty> }).
    #[tokio::test]
    async fn n_nudges_then_static_green_spin_takes_recovery_terminal() {
        // N = DEFAULT_MAX_NUDGES = 2, K = DEFAULT_STATIC_TREE_K = 3.
        //   iter 1: run_checks (green) — iters 0->1
        //   iter 2: echo — iters 1->2
        //   iter 3: echo — iters 2->3 -> trip, nudge 1. iters=0. awaiting=true.
        //   iter 4: status+echo — push status. iters 0->1. awaiting=false.
        //   iter 5: echo — iters 1->2.
        //   iter 6: echo — iters 2->3 -> trip, nudge 2. iters=0. awaiting=true.
        //   iter 7: status+echo — push status. iters 0->1. awaiting=false.
        //   iter 8: echo — iters 1->2.
        //   iter 9: echo — iters 2->3 -> trip. nudges_fired(2)==max(2) -> RECOVERY TERMINAL.
        //   max_iterations=10 (never reached — recovery returns at iter 9).
        let runner = passing_runner();
        let tools = standard_registry(Some(runner.clone()));
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 10)
            .with_checks(runner)
            .with_max_nudges(2);

        let backend = MockBackend::from_turns(vec![
            run_checks_turn("c1"),
            echo_turn("c2"),
            echo_turn("c3"),
            status_echo_turn("c4", "still writing the failing-case test"),
            echo_turn("c5"),
            echo_turn("c6"),
            status_echo_turn("c7", "still fixing the off-by-one"),
            echo_turn("c8"),
            echo_turn("c9"),
            // iter 10 would over-draw; recovery returns at iter 9 before this.
            echo_turn("c10"),
        ]);

        let snap_store = Arc::new(SnapshotStore::new());
        let pers = make_persistence(snap_store.clone());
        let RunResult { outcome, stats } = run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        match outcome {
            LoopOutcome::Finished(Disposition::Failed { mode, summary }) => {
                assert_eq!(
                    mode,
                    FailureMode::FinishDiscipline,
                    "recovery terminal must be FinishDiscipline; got {mode:?}"
                );
                assert!(
                    summary.contains("gates green but agent did not call finish"),
                    "summary must name the recovery; got {summary}"
                );
                assert!(
                    summary.contains("2 nudges"),
                    "summary must name the nudge count; got {summary}"
                );
            }
            other => panic!("expected Finished(Failed{{FinishDiscipline}}), got {other:?}"),
        }

        let rec = snap_store
            .inner
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        let facts = rec
            .recovery_facts
            .as_ref()
            .expect("recovery_facts must be Some on the FinishDiscipline terminal");
        assert!(
            facts.gates_green_at_exit,
            "gates were green at the recovery terminal"
        );
        assert!(
            !facts.tree_dirty,
            "no edit_file/bash was called in this script, so tree_dirty must be false"
        );
        assert!(
            !facts.nudge_statuses.is_empty(),
            "nudge_statuses must be non-empty (one status per nudge that didn't finish)"
        );
        assert_eq!(
            facts.nudge_statuses.len(),
            2,
            "exactly two status replies (one per nudge); got {:?}",
            facts.nudge_statuses
        );
        assert_eq!(count_nudge_injections(&rec.messages), 2);
        assert_no_adjacent_user_messages(&rec.messages);
        assert_eq!(
            stats.nudges_fired, 2,
            "nudges_fired must equal DEFAULT_MAX_NUDGES=2"
        );
    }

    // AC-4 detection test: a successful edit_file between green-static iters
    // resets iters_since_tree_change AND clears last_gate_green, so no nudge
    // fires until a fresh green run_checks is observed post-edit.
    #[tokio::test]
    async fn edit_file_resets_counter_and_clears_green_until_fresh_run_checks() {
        //   iter 1: run_checks (green) — iters 0->1
        //   iter 2: edit_file (success) — mutated=true, tree_dirty=true,
        //           last_gate_green=false. iters reset to 0.
        //   iter 3: echo — iters 0->1. green=false -> no trip.
        //   iter 4: echo — iters 1->2. green=false -> no trip.
        //   iter 5: echo — iters 2->3. green=false -> no trip (stale-green window closed).
        //   iter 6: run_checks (green) — last_gate_green=true. iters 3->4. trip! nudge.
        //   iter 7: echo — iters 0->1 (post-nudge).
        //   max_iterations=7 -> MaxIterations
        let root = TempDir::new().expect("workspace tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize root");
        let workspace = Workspace::new(&root_path, None).expect("workspace");
        let ctx = ToolCtx::new(Arc::new(workspace), Arc::new(crate::tool::StubOffloadSink));

        let runner = passing_runner();
        let tools = standard_registry(Some(runner.clone()));
        let config = RunConfig::new("do the task", 7)
            .with_checks(runner)
            .with_max_nudges(2);

        let backend = MockBackend::from_turns(vec![
            run_checks_turn("c1"),
            turn_with(
                vec![tool_call(
                    "c2",
                    "edit_file",
                    serde_json::json!({
                        "path": "flag",
                        "old_string": "",
                        "new_string": "planted\n",
                    }),
                )],
                StopReason::ToolUse,
            ),
            echo_turn("c3"),
            echo_turn("c4"),
            echo_turn("c5"),
            run_checks_turn("c6"),
            echo_turn("c7"),
        ]);

        let snap_store = Arc::new(SnapshotStore::new());
        let pers = make_persistence(snap_store.clone());
        let RunResult { outcome, stats } = run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        assert!(
            matches!(outcome, LoopOutcome::MaxIterations),
            "expected MaxIterations (only one nudge, no recovery); got {outcome:?}"
        );

        let rec = snap_store
            .inner
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        // Exactly one nudge — fired at iter 6 (the fresh green run_checks),
        // NOT at iter 5 (where iters=3 but green=false after the edit).
        assert_eq!(
            count_nudge_injections(&rec.messages),
            1,
            "exactly one nudge, fired only after the fresh green run_checks"
        );
        assert_no_adjacent_user_messages(&rec.messages);
        // GREEN-static MaxIterations now captures recovery_facts (hardened):
        // the last iteration before the cap was a fresh green run_checks
        // (iter 6) followed by a non-mutating echo (iter 7), so
        // last_gate_green is true at the terminal. tree_dirty is true because
        // a successful edit_file ran at iter 2.
        let facts = rec
            .recovery_facts
            .as_ref()
            .expect("green-static MaxIterations must capture recovery_facts");
        assert!(
            facts.gates_green_at_exit,
            "gates_green_at_exit must be true on a green-static MaxIterations"
        );
        assert!(
            facts.tree_dirty,
            "an edit_file ran, so tree_dirty must be true"
        );
        assert!(
            root_path.join("flag").exists(),
            "edit_file must have created the flag file"
        );
        assert_eq!(stats.edit_file_calls_ok, 1, "one successful edit_file call");
        assert_eq!(stats.bash_calls_ok, 0, "no bash calls");
        assert_eq!(
            stats.tree_dirty, facts.tree_dirty,
            "RunStats.tree_dirty and RecoveryFacts.tree_dirty must agree"
        );
        assert_eq!(
            stats.gates_green_at_exit, facts.gates_green_at_exit,
            "RunStats.gates_green_at_exit and RecoveryFacts.gates_green_at_exit must agree"
        );
    }

    // AC-5 detection test: RED run_checks + spin does NOT nudge and
    // terminates MaxIterations.
    #[tokio::test]
    async fn red_run_checks_spin_does_not_nudge_and_terminates_max_iterations() {
        //   iter 1: run_checks (RED) — last_gate_green=false. iters 0->1. no trip.
        //   iter 2-5: echo — iters 1->2->3->4->5. green=false -> never trips.
        //   max_iterations=5 -> MaxIterations.
        let runner = failing_runner();
        let tools = standard_registry(Some(runner.clone()));
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5)
            .with_checks(runner)
            .with_max_nudges(2);

        let backend = MockBackend::from_turns(vec![
            run_checks_turn("c1"),
            echo_turn("c2"),
            echo_turn("c3"),
            echo_turn("c4"),
            echo_turn("c5"),
        ]);

        let snap_store = Arc::new(SnapshotStore::new());
        let pers = make_persistence(snap_store.clone());
        let RunResult { outcome, .. } = run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        assert!(
            matches!(outcome, LoopOutcome::MaxIterations),
            "red-gate spin must fall through to MaxIterations; got {outcome:?}"
        );

        let rec = snap_store
            .inner
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        assert_eq!(
            count_nudge_injections(&rec.messages),
            0,
            "red gate must never inject a nudge"
        );
        // Symmetric guard: a RED-gate MaxIterations yields recovery_facts ==
        // None — the recovery_facts write is conditioned on last_gate_green,
        // NOT unconditional. A red gate has nothing worth preserving.
        assert_eq!(
            rec.recovery_facts, None,
            "red-gate MaxIterations must NOT capture recovery_facts"
        );
        let disp = rec
            .disposition
            .as_ref()
            .expect("persisted record must carry a disposition at MaxIterations");
        assert!(
            matches!(
                disp,
                Disposition::Failed {
                    mode: FailureMode::BudgetExhausted,
                    ..
                }
            ),
            "MaxIterations disposition must be Failed{{BudgetExhausted}}; got {disp:?}"
        );
    }

    // AC-6 detection test: max_nudges == 0 disables the feature entirely —
    // no nudge is ever injected and the recovery terminal is never taken;
    // a green-gates + static-tree spin falls through to MaxIterations.
    #[tokio::test]
    async fn max_nudges_zero_disables_finish_recovery_entirely() {
        //   iter 1: run_checks (green) — iters 0->1. max_nudges=0 -> no trip.
        //   iter 2-4: echo — iters 1->2->3->4. no trip (feature off).
        //   max_iterations=4 -> MaxIterations.
        let runner = passing_runner();
        let tools = standard_registry(Some(runner.clone()));
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 4)
            .with_checks(runner)
            .with_max_nudges(0);

        let backend = MockBackend::from_turns(vec![
            run_checks_turn("c1"),
            echo_turn("c2"),
            echo_turn("c3"),
            echo_turn("c4"),
        ]);

        let snap_store = Arc::new(SnapshotStore::new());
        let pers = make_persistence(snap_store.clone());
        let RunResult { outcome, stats } = run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        assert!(
            matches!(outcome, LoopOutcome::MaxIterations),
            "max_nudges=0 must fall through to MaxIterations; got {outcome:?}"
        );

        let rec = snap_store
            .inner
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        assert_eq!(
            count_nudge_injections(&rec.messages),
            0,
            "max_nudges=0 must never inject a nudge"
        );
        // GREEN-static MaxIterations now captures recovery_facts (hardened):
        // the WIP the recovery feature exists to preserve is no longer silently
        // forfeited when finish-recovery is disabled and a green-static spin
        // reaches the iteration cap.
        let facts = rec
            .recovery_facts
            .as_ref()
            .expect("green-static MaxIterations must capture recovery_facts");
        assert!(
            facts.gates_green_at_exit,
            "gates_green_at_exit must be true on a green-static MaxIterations"
        );
        // The disposition is unchanged by the hardening — still
        // Failed{BudgetExhausted}, outcome MaxIterations.
        let disp = rec
            .disposition
            .as_ref()
            .expect("persisted record must carry a disposition at MaxIterations");
        assert!(
            matches!(
                disp,
                Disposition::Failed {
                    mode: FailureMode::BudgetExhausted,
                    ..
                }
            ),
            "MaxIterations disposition must be Failed{{BudgetExhausted}}; got {disp:?}"
        );
        assert_eq!(
            stats.nudges_fired, 0,
            "max_nudges=0 must never fire a nudge"
        );
        assert_eq!(
            stats.iters_since_tree_change_at_exit, 4,
            "counter must be 4 at exit (all 4 iters non-mutating)"
        );
        assert_eq!(
            stats.peak_iters_since_tree_change, 4,
            "peak must be 4 (monotone non-mutating run)"
        );
        assert_eq!(stats.mutating_iters, 0, "no mutating iterations");
        assert!(
            !stats.tree_dirty,
            "no edit_file/bash ran — tree_dirty must be false"
        );
    }

    // =====================================================================
    // Finish-recovery at the StoppedWithoutFinish terminal (the stop-terminal
    // trip site added alongside the existing green-static-spin trip site).
    // =====================================================================

    // AC5 — green→stop→nudge→finish ⇒ Done (harness never constructs Done).
    // Sequence:
    //   iter1: run_checks (green) → last_gate_green=true.
    //   iter2: no-tool-call text stop → guard fires (green+nudges>0) →
    //          fresh user nudge pushed, nudges_fired=1, continue.
    //   iter3: finish(done) → handle_finish_call verifies green → Done.
    // recovery_facts must be None (Done-after-nudge is a clean success).
    #[tokio::test]
    async fn green_stop_nudge_finish_done() {
        let runner = passing_runner();
        let tools = standard_registry(Some(runner.clone()));
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 3)
            .with_checks(runner)
            .with_max_nudges(1);

        let backend = MockBackend::from_turns(vec![
            run_checks_turn("c1"),
            turn_with(
                vec![ContentBlock::Text("looks fixed".into())],
                StopReason::EndTurn,
            ),
            finish_call(
                "c3",
                serde_json::json!({"disposition": "done", "summary": "..."}),
            ),
        ]);

        let snap_store = Arc::new(SnapshotStore::new());
        let pers = make_persistence(snap_store.clone());
        let RunResult { outcome, stats } = run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        assert!(
            matches!(outcome, LoopOutcome::Finished(Disposition::Done { .. })),
            "expected Done after green-stop nudge + finish; got {outcome:?}"
        );
        assert_eq!(
            stats.nudges_fired, 1,
            "exactly one nudge fired at the stop terminal"
        );

        let rec = snap_store
            .inner
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        // Done-after-nudge is a clean success: recovery_facts must be None.
        assert_eq!(
            rec.recovery_facts, None,
            "Done-after-nudge must NOT produce recovery_facts"
        );
        // Exactly one nudge was injected (as a fresh Message::User after the
        // assistant stop turn).
        assert_eq!(
            count_nudge_injections(&rec.messages),
            1,
            "exactly one nudge must be injected before the finish"
        );
        // No adjacent user messages: the nudge is a fresh Message::User pushed
        // after the assistant stop turn (assistant → user), not a second
        // adjacent user (which would 400 on the real wire).
        assert_no_adjacent_user_messages(&rec.messages);
    }

    // AC6 — green→stop→nudge→stop ⇒ FinishDiscipline + recovery_facts.
    // Sequence:
    //   iter1: run_checks (green).
    //   iter2: text stop → nudge (nudges_fired=1), continue.
    //   iter3: text stop → guard: nudge_awaiting_status → push "almost done
    //          here" onto nudge_statuses; nudges_fired==max_nudges →
    //          exhaustion terminal: Finished(Failed{FinishDiscipline}).
    // max_iterations MUST be ≥ 3 so iter3's stop reaches exhaustion rather
    // than the loop cap or an over-draw BackendError.
    #[tokio::test]
    async fn green_stop_nudge_exhaustion_finish_discipline() {
        let runner = passing_runner();
        let tools = standard_registry(Some(runner.clone()));
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 3)
            .with_checks(runner)
            .with_max_nudges(1);

        let backend = MockBackend::from_turns(vec![
            run_checks_turn("c1"),
            turn_with(
                vec![ContentBlock::Text("still working on it".into())],
                StopReason::EndTurn,
            ),
            turn_with(
                vec![ContentBlock::Text("almost done here".into())],
                StopReason::EndTurn,
            ),
        ]);

        let snap_store = Arc::new(SnapshotStore::new());
        let pers = make_persistence(snap_store.clone());
        let RunResult { outcome, stats } = run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        match &outcome {
            LoopOutcome::Finished(Disposition::Failed { mode, summary }) => {
                assert_eq!(
                    *mode,
                    FailureMode::FinishDiscipline,
                    "stop-terminal exhaustion must be FinishDiscipline; got {mode:?}"
                );
                assert!(
                    summary.contains("gates green but agent did not call finish"),
                    "summary must name the recovery; got {summary}"
                );
                assert!(
                    summary.contains("1 nudges"),
                    "summary must name the nudge count; got {summary}"
                );
            }
            other => panic!("expected Finished(Failed{{FinishDiscipline}}); got {other:?}"),
        }

        let rec = snap_store
            .inner
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        // AC6 primary: exactly one nudge injected as a fresh Message::User.
        // This is the LOUD guard against a wrong-shape append silently dropping
        // the nudge (see AC9).
        assert_eq!(
            count_nudge_injections(&rec.messages),
            1,
            "exactly one nudge must be injected (as a fresh Message::User)"
        );
        // AC9 wire-shape: no adjacent user messages, and the nudge lives in
        // its own Message::User immediately preceded by a Message::Assistant.
        assert_no_adjacent_user_messages(&rec.messages);
        let nudge_pos = rec.messages.iter().position(|m| {
            matches!(m, Message::User { content }
                if content.iter().any(|b| matches!(b, UserBlock::Text(t) if t == NUDGE_TEXT)))
        });
        let nudge_idx = nudge_pos.expect("nudge message must exist in rec.messages");
        assert!(
            nudge_idx > 0,
            "nudge message must have a predecessor (the assistant stop turn)"
        );
        assert!(
            matches!(rec.messages[nudge_idx - 1], Message::Assistant { .. }),
            "the message immediately before the nudge must be Message::Assistant \
             (stop turn); got {:?}",
            rec.messages[nudge_idx - 1]
        );
        // recovery_facts: gates_green_at_exit, no edit_file/bash (tree_dirty
        // false), and nudge_statuses contains iter3's text.
        let facts = rec
            .recovery_facts
            .as_ref()
            .expect("recovery_facts must be Some on the stop-terminal FinishDiscipline");
        assert!(
            facts.gates_green_at_exit,
            "gates were green at the stop-terminal recovery terminal"
        );
        assert!(
            !facts.tree_dirty,
            "no edit_file/bash ran, so tree_dirty must be false"
        );
        assert!(
            !facts.nudge_statuses.is_empty(),
            "nudge_statuses must contain the iter3 stop text"
        );
        assert!(
            facts
                .nudge_statuses
                .iter()
                .any(|s| s.contains("almost done here")),
            "nudge_statuses must contain the iter3 stop text 'almost done here'; \
             got {:?}",
            facts.nudge_statuses
        );
        assert_eq!(
            stats.nudges_fired, 1,
            "exactly one nudge fired at the stop terminal"
        );
        assert!(
            stats.gates_green_at_exit,
            "gates_green_at_exit must be true on the stop-terminal FinishDiscipline terminal"
        );
    }

    // AC5 — successful bash latches tree_dirty, increments bash_calls_ok,
    // clears last_gate_green, and marks the iteration as mutating.
    // NOTE: no engine test exercises the `call.name == "bash"` half of the
    // tool-classification arm today — this test is additive, not a duplicate
    // of `edit_file_resets_counter_and_clears_green_until_fresh_run_checks`
    // (which covers only the edit_file half).
    #[tokio::test]
    async fn successful_bash_latches_tree_dirty_and_clears_green() {
        // Sequence:
        //   iter 1: run_checks (green) — last_gate_green=true, iters 0->1.
        //   iter 2: bash "true" (success, mutating) — tree_dirty=true,
        //           last_gate_green=false, iters 1->0 (mutated). bash_calls_ok=1.
        //   max_iterations=2 -> MaxIterations.
        let runner = passing_runner();
        let tools = standard_registry(Some(runner.clone()));
        let ctx = ToolCtx::stub();
        // Leave max_nudges at DEFAULT_MAX_NUDGES=2 (do not call with_max_nudges).
        let config = RunConfig::new("do the task", 2).with_checks(runner);

        let backend = MockBackend::from_turns(vec![
            run_checks_turn("c1"),
            turn_with(
                vec![tool_call(
                    "c2",
                    "bash",
                    serde_json::json!({"command": "true"}),
                )],
                StopReason::ToolUse,
            ),
        ]);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;
        assert!(
            matches!(outcome, LoopOutcome::MaxIterations),
            "expected MaxIterations; got {outcome:?}"
        );
        assert_eq!(stats.bash_calls_ok, 1, "one successful bash call");
        assert_eq!(stats.edit_file_calls_ok, 0, "no edit_file calls");
        assert!(stats.tree_dirty, "bash must latch tree_dirty");
        assert_eq!(stats.mutating_iters, 1, "bash iter is mutating");
        assert_eq!(
            stats.iters_since_tree_change_at_exit, 0,
            "counter reset by bash on iter 2"
        );
        assert_eq!(
            stats.peak_iters_since_tree_change, 1,
            "run_checks iter accumulated 1 before bash reset the counter"
        );
        assert!(
            !stats.gates_green_at_exit,
            "bash must clear gates_green_at_exit (last_gate_green)"
        );
    }

    // AC7 — red/never-green stop with recovery ENABLED ⇒ StoppedWithoutFinish
    // unchanged. The model never calls run_checks so last_gate_green stays
    // false. The guard `&& last_gate_green` blocks the nudge even though
    // max_nudges > 0.
    #[tokio::test]
    async fn red_never_green_stop_with_recovery_enabled_stops_without_finish() {
        let runner = passing_runner();
        let tools = standard_registry(Some(runner.clone()));
        let ctx = ToolCtx::stub();
        // Recovery is ENABLED (max_nudges=1) but the model never calls run_checks.
        let config = RunConfig::new("do the task", 10)
            .with_checks(runner)
            .with_max_nudges(1);

        let backend = MockBackend::from_turns(vec![turn_with(
            vec![ContentBlock::Text("I am just talking".into())],
            StopReason::EndTurn,
        )]);

        let snap_store = Arc::new(SnapshotStore::new());
        let pers = make_persistence(snap_store.clone());
        let RunResult { outcome, stats } = run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        assert!(
            matches!(outcome, LoopOutcome::StoppedWithoutFinish),
            "never-green stop must yield StoppedWithoutFinish even with recovery on; \
             got {outcome:?}"
        );
        assert!(
            !stats.gates_green_at_exit,
            "gate was never green in-loop -> gates_green_at_exit must be false"
        );

        let rec = snap_store
            .inner
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        // The && last_gate_green guard must block the nudge.
        assert_eq!(
            count_nudge_injections(&rec.messages),
            0,
            "no nudge must be injected when last_gate_green is false"
        );
    }

    // ---- retry / backoff tests ------------------------------------------

    /// Load-bearing unit test: pins the exponential schedule with a NON-zero
    /// base so the shape is actually tested. The async retry tests all use
    /// `base = Duration::ZERO` and cannot discriminate between exponential,
    /// linear, or constant schedules.
    #[test]
    fn retry_delay_schedule_is_exponential() {
        assert_eq!(
            retry_delay(Duration::from_millis(500), 0),
            Duration::from_millis(500)
        );
        assert_eq!(
            retry_delay(Duration::from_millis(500), 1),
            Duration::from_secs(1)
        );
        assert_eq!(
            retry_delay(Duration::from_millis(500), 2),
            Duration::from_secs(2)
        );
    }

    /// A single transient error followed by a success completes the run.
    /// Verifies: `backend.calls() == 2`, `stats.iterations == 1`, and
    /// `outcome == LoopOutcome::Finished(Disposition::Done{..})`.
    #[tokio::test]
    async fn transient_then_success_proceeds() {
        let transient = BackendError::Transient {
            kind: TransientKind::RateLimit,
            retry_after: None,
        };
        let done_turn = finish_call(
            "c1",
            serde_json::json!({ "disposition": "done", "summary": "all good" }),
        );

        let backend = MockBackend::new(vec![Err(transient), Ok(done_turn)]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 10).with_retry_backoff_base(Duration::ZERO);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert_eq!(backend.calls(), 2, "one transient + one success = 2 draws");
        assert_eq!(stats.iterations, 1, "single logical iteration");
        match outcome {
            LoopOutcome::Finished(Disposition::Done { verification, .. }) => {
                assert_eq!(
                    verification,
                    Verification::NoChecksConfigured,
                    "no checks configured"
                );
            }
            other => panic!("expected Finished(Done), got {other:?}"),
        }
    }

    /// A retried turn is counted as ONE logical iteration even when
    /// `backend.turn` was called twice.
    #[tokio::test]
    async fn iteration_counted_once_across_retried_turn() {
        let transient = BackendError::Transient {
            kind: TransientKind::RateLimit,
            retry_after: None,
        };
        // Plain turn with no tool calls → StoppedWithoutFinish.
        let no_tools_turn = turn_with(
            vec![ContentBlock::Text("thinking...".to_string())],
            StopReason::EndTurn,
        );

        let backend = MockBackend::new(vec![Err(transient), Ok(no_tools_turn)]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 10).with_retry_backoff_base(Duration::ZERO);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert_eq!(backend.calls(), 2, "one transient + one success = 2 draws");
        assert_eq!(
            stats.iterations, 1,
            "retry does NOT double-count the iteration"
        );
        assert!(
            matches!(outcome, LoopOutcome::StoppedWithoutFinish),
            "expected StoppedWithoutFinish, got {outcome:?}"
        );
    }

    /// When all retries are exhausted the loop gives up and returns
    /// `LoopOutcome::BackendError` carrying the last attempt's error, with
    /// `stats.iterations == 1` (all retries stay inside iteration 1).
    #[tokio::test]
    async fn consecutive_transients_exhaust_and_give_up_as_transient_infra() {
        // max_retries = 3 (default) → 4 total attempts.
        let script: Vec<Result<AssistantTurn, BackendError>> = (0..4)
            .map(|_| {
                Err(BackendError::Transient {
                    kind: TransientKind::RateLimit,
                    retry_after: None,
                })
            })
            .collect();

        let backend = MockBackend::new(script);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 10).with_retry_backoff_base(Duration::ZERO);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert_eq!(backend.calls(), 4, "1 first try + 3 retries = 4 draws");
        assert_eq!(stats.iterations, 1, "all retries stay inside iteration 1");
        // The outcome must be BackendError carrying a Transient error.
        match &outcome {
            LoopOutcome::BackendError(BackendError::Transient { .. }) => {}
            other => panic!("expected BackendError(Transient), got {other:?}"),
        }
        // into_disposition maps a retryable final error to TransientInfra.
        match outcome.into_disposition() {
            Disposition::Failed {
                mode: FailureMode::TransientInfra,
                ..
            } => {}
            other => panic!("expected Failed(TransientInfra), got {other:?}"),
        }
    }

    /// Terminal errors are not retried — `backend.calls() == 1`.
    #[tokio::test]
    async fn terminal_fails_on_first_occurrence_no_retry() {
        let backend = MockBackend::new(vec![Err(BackendError::Terminal {
            kind: TerminalKind::Other,
            message: "bad request".to_string(),
        })]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 10).with_retry_backoff_base(Duration::ZERO);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert_eq!(backend.calls(), 1, "terminal error: no retries");
        assert_eq!(stats.iterations, 1);
        match outcome.into_disposition() {
            Disposition::Failed {
                mode: FailureMode::PersistentToolError,
                ..
            } => {}
            other => panic!("expected Failed(PersistentToolError), got {other:?}"),
        }
    }

    /// `ContextLengthExceeded` is NOT retried (deferred out of 0.4.0) —
    /// it maps to `PersistentToolError` on first occurrence.
    #[tokio::test]
    async fn context_length_exceeded_not_retried() {
        let backend = MockBackend::new(vec![Err(BackendError::ContextLengthExceeded)]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 10).with_retry_backoff_base(Duration::ZERO);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert_eq!(backend.calls(), 1, "ContextLengthExceeded: no retries");
        assert_eq!(stats.iterations, 1);
        match outcome.into_disposition() {
            Disposition::Failed {
                mode: FailureMode::PersistentToolError,
                ..
            } => {}
            other => panic!("expected Failed(PersistentToolError), got {other:?}"),
        }
    }

    // ---- ResumeError: Display + Error::source + From<StoreError> --------
    // These impls are the only surface a caller sees when `resume` fails, so
    // their wording and error-chaining are observable contracts: a regression
    // here would silently change how the outer harness logs/routes resume
    // failures.

    #[test]
    fn resume_error_display_and_source_chain_are_stable() {
        let unknown = ResumeError::UnknownRunId("task-9:2".to_string());
        assert_eq!(
            unknown.to_string(),
            "no checkpoint found for run_id \"task-9:2\"",
            "UnknownRunId Display must quote the run_id",
        );
        assert!(
            std::error::Error::source(&unknown).is_none(),
            "UnknownRunId must have no underlying source",
        );

        let store_err = StoreError::LockPoisoned;
        let wrapped = ResumeError::Store(store_err);
        assert_eq!(
            wrapped.to_string(),
            "store error: internal connection mutex was poisoned",
            "Store variant must delegate to the StoreError Display via the \
             `store error: ` prefix",
        );
        let src = std::error::Error::source(&wrapped).expect("Store must chain its StoreError");
        assert!(
            src.to_string().contains("mutex was poisoned"),
            "source() must expose the underlying StoreError, got {src}",
        );
    }

    #[test]
    fn resume_error_from_store_error_is_the_store_variant() {
        // The `?`-driven conversion in `resume` relies on this `From` impl
        // mapping every StoreError into ResumeError::Store — a regression to
        // (say) UnknownRunId would mislead the harness into treating a store
        // fault as a missing checkpoint.
        let err: ResumeError = StoreError::LockPoisoned.into();
        assert!(
            matches!(err, ResumeError::Store(StoreError::LockPoisoned)),
            "From<StoreError> must produce ResumeError::Store verbatim, got {err:?}",
        );
    }

    // ---- Persistence Debug redacts the store pointer ---------------------
    // `RunStore` has no Debug bound, so the manual Debug impl must substitute
    // a fixed label — never leak a pointer/addr (which would make run logs
    // non-deterministic) or panic.

    #[test]
    fn persistence_debug_redacts_the_store_as_a_fixed_label() {
        let pers = make_persistence(Arc::new(SnapshotStore::new()));
        let dbg = format!("{pers:?}");
        assert!(
            dbg.contains("task_id") && dbg.contains("\"task-t\""),
            "Debug must surface task_id, got {dbg}",
        );
        assert!(
            dbg.contains("attempt_n"),
            "Debug must surface attempt_n, got {dbg}"
        );
        assert!(
            dbg.contains("model_label") && dbg.contains("\"test-model\""),
            "Debug must surface model_label, got {dbg}",
        );
        assert!(
            dbg.contains("backend_settings"),
            "Debug must surface backend_settings, got {dbg}"
        );
        assert!(
            dbg.contains("\"<dyn RunStore>\""),
            "Debug must substitute the fixed `<dyn RunStore>` label, got {dbg}",
        );
        // Guard: the Debug output must NOT carry a memory address for the
        // store (the whole point of the manual impl).
        assert!(
            !dbg.contains("0x"),
            "Debug must not leak a store pointer address, got {dbg}",
        );
    }

    // ---- RunConfig builders override the documented defaults ------------
    // `static_tree_k` and `max_retries` feed finish-recovery detection and
    // the retry schedule; a no-op builder would silently change both.

    #[test]
    fn with_static_tree_k_and_with_max_retries_override_their_defaults() {
        let default = RunConfig::new("t", 5);
        assert_eq!(default.static_tree_k, super::DEFAULT_STATIC_TREE_K);
        assert_eq!(default.max_retries, super::DEFAULT_MAX_RETRIES);

        let cfg = RunConfig::new("t", 5)
            .with_static_tree_k(7)
            .with_max_retries(0);
        assert_eq!(
            cfg.static_tree_k, 7,
            "with_static_tree_k must set the field"
        );
        assert_eq!(cfg.max_retries, 0, "with_max_retries must set the field");

        // `0` is the documented "disable" sentinel for both knobs — make sure
        // the builders accept it without clamping.
        let disabled = RunConfig::new("t", 1)
            .with_static_tree_k(0)
            .with_max_retries(0);
        assert_eq!(disabled.static_tree_k, 0);
        assert_eq!(disabled.max_retries, 0);
    }

    // =====================================================================
    // Opt-in full run transcript (crate::transcript)
    // =====================================================================

    fn read_transcript_lines(path: &std::path::Path) -> Vec<serde_json::Value> {
        let contents = std::fs::read_to_string(path).expect("read transcript file");
        contents
            .lines()
            .map(|l| serde_json::from_str(l).expect("each transcript line is valid JSON"))
            .collect()
    }

    fn assert_ts_is_20_chars_ending_in_z(line: &serde_json::Value) {
        let ts = line["ts"].as_str().expect("ts is a string");
        assert_eq!(
            ts.len(),
            20,
            "ts must be YYYY-MM-DDTHH:MM:SSZ (20 chars): {ts}"
        );
        assert!(ts.ends_with('Z'), "ts must end in Z: {ts}");
    }

    fn message_block_count(m: &Message) -> usize {
        match m {
            Message::User { content } => content.len(),
            Message::Assistant { content } => content.len(),
        }
    }

    /// One entry per `model_request` event encountered while reconstructing a
    /// transcript's lines.
    struct ReconstructedRequest {
        iteration: u64,
        messages: Vec<Message>,
        message_count: usize,
        block_count: usize,
    }

    /// Rebuild message history from `lines` (one run block; `lines[0]` must
    /// be `run_start`) per the reconstruction contract documented on
    /// [`crate::transcript`], snapshotting the rebuilt history at every
    /// `model_request`.
    fn reconstruct_transcript(
        lines: &[serde_json::Value],
    ) -> (Vec<ReconstructedRequest>, serde_json::Value) {
        let run_start = lines.first().expect("at least one line").clone();
        assert_eq!(run_start["event"], "run_start");
        let mut history: Vec<Message> = serde_json::from_value(run_start["messages"].clone())
            .expect("run_start.messages deserializes as Vec<Message>");
        let mut pending: Vec<UserBlock> = Vec::new();
        let mut requests = Vec::new();

        for line in &lines[1..] {
            let event = line["event"].as_str().expect("event is a string");
            match event {
                "model_request" => {
                    if !pending.is_empty() {
                        history.push(Message::User {
                            content: std::mem::take(&mut pending),
                        });
                    }
                    requests.push(ReconstructedRequest {
                        iteration: line["iteration"].as_u64().expect("iteration"),
                        messages: history.clone(),
                        message_count: usize::try_from(
                            line["message_count"].as_u64().expect("message_count"),
                        )
                        .expect("message_count fits usize"),
                        block_count: usize::try_from(
                            line["block_count"].as_u64().expect("block_count"),
                        )
                        .expect("block_count fits usize"),
                    });
                }
                "model_response" => {
                    if !pending.is_empty() {
                        history.push(Message::User {
                            content: std::mem::take(&mut pending),
                        });
                    }
                    let content: Vec<ContentBlock> =
                        serde_json::from_value(line["content"].clone())
                            .expect("model_response.content deserializes as Vec<ContentBlock>");
                    history.push(Message::Assistant { content });
                }
                "tool_result" => {
                    pending.push(UserBlock::ToolResult {
                        call_id: line["call_id"].as_str().expect("call_id").to_string(),
                        content: line["content"].as_str().expect("content").to_string(),
                        is_error: line["is_error"].as_bool().expect("is_error"),
                    });
                }
                "harness_message" => {
                    let placement = line["placement"].as_str().expect("placement");
                    let text = line["text"].as_str().expect("text").to_string();
                    match placement {
                        "appended_to_tool_results" => pending.push(UserBlock::Text(text)),
                        "new_user_message" => {
                            if !pending.is_empty() {
                                history.push(Message::User {
                                    content: std::mem::take(&mut pending),
                                });
                            }
                            history.push(Message::User {
                                content: vec![UserBlock::Text(text)],
                            });
                        }
                        other => panic!("unknown harness_message placement {other}"),
                    }
                }
                "iteration_end" | "backend_error" | "run_end" => {}
                other => panic!("unexpected event kind in reconstruction: {other}"),
            }
        }
        (requests, run_start)
    }

    /// Assert the reconstruction contract holds for `lines` against what
    /// `backend` actually saw: at every `model_request`, the rebuilt history
    /// equals `backend.messages_seen()[iteration-1]`, and its length/block
    /// total equal the event's `message_count`/`block_count`. Also checks
    /// `run_start.system` against the first turn's system prompt.
    fn assert_reconstruction_matches(lines: &[serde_json::Value], backend: &MockBackend) {
        let (requests, run_start) = reconstruct_transcript(lines);
        let seen = backend.messages_seen();
        let systems = backend.systems_seen();
        assert_eq!(
            run_start["system"],
            serde_json::Value::String(systems[0].clone().expect("system prompt was sent")),
            "run_start.system must match the first turn's system prompt"
        );
        assert!(!requests.is_empty(), "at least one model_request event");
        for req in &requests {
            let idx = usize::try_from(req.iteration - 1).expect("iteration fits usize");
            assert_eq!(
                req.messages, seen[idx],
                "rebuilt history at iteration {} must equal messages_seen()[{idx}]",
                req.iteration
            );
            assert_eq!(
                req.messages.len(),
                req.message_count,
                "message_count must match the rebuilt length at iteration {}",
                req.iteration
            );
            let block_total: usize = req.messages.iter().map(message_block_count).sum();
            assert_eq!(
                block_total, req.block_count,
                "block_count must match the rebuilt total at iteration {}",
                req.iteration
            );
        }
    }

    #[tokio::test]
    #[allow(clippy::too_many_lines)]
    async fn ordered_events_transcript_has_pinned_13_lines_and_reconstructs() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("t.jsonl");
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5)
            .with_checks(failing_runner())
            .with_transcript(path.clone(), "t");
        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call("c1", "echo", serde_json::json!({"i": 1}))],
                StopReason::ToolUse,
            ),
            finish_call(
                "c2",
                serde_json::json!({"disposition": "done", "summary": "s"}),
            ),
            finish_call(
                "c3",
                serde_json::json!({"disposition": "blocked", "decision_needed": "need input"}),
            ),
        ]);

        let RunResult { outcome, .. } = run(&backend, &tools, &ctx, &config).await;
        assert!(
            matches!(outcome, LoopOutcome::Finished(Disposition::Blocked { .. })),
            "expected Finished(Blocked); got {outcome:?}"
        );

        let lines = read_transcript_lines(&path);
        assert_eq!(lines.len(), 13, "expected exactly 13 transcript lines");
        for line in &lines {
            let event = line["event"].as_str().expect("event is a string");
            assert!(
                crate::transcript::EVENT_KINDS.contains(&event),
                "unexpected event kind {event}"
            );
            assert_ts_is_20_chars_ending_in_z(line);
        }

        // 1: run_start
        assert_eq!(lines[0]["event"], "run_start");
        assert_eq!(lines[0]["resume"], false);
        assert!(lines[0]["run_id"].is_null());
        assert_eq!(lines[0]["label"], "t");
        assert_eq!(
            lines[0]["tools"],
            serde_json::Value::Array(tools.list()),
            "run_start.tools must carry the exact tool schema array"
        );

        // 2: model_request (iteration 1)
        assert_eq!(lines[1]["event"], "model_request");
        assert_eq!(lines[1]["iteration"], 1);
        assert_eq!(lines[1]["message_count"], 1);
        assert_eq!(lines[1]["block_count"], 1);

        // 3: model_response (iteration 1) — the echo input appears verbatim.
        assert_eq!(lines[2]["event"], "model_response");
        assert_eq!(lines[2]["iteration"], 1);
        assert_eq!(lines[2]["attempts"], 1);
        assert!(
            lines[2]["content"].to_string().contains(r#"{"i":1}"#),
            "line 3's content must carry the echo input verbatim: {}",
            lines[2]["content"]
        );

        // 4: tool_result (echo) — content equals the ToolResult content the
        // backend actually saw on the NEXT turn.
        assert_eq!(lines[3]["event"], "tool_result");
        assert_eq!(lines[3]["tool_name"], "echo");
        assert_eq!(lines[3]["call_id"], "c1");
        assert_eq!(lines[3]["is_error"], false);
        assert!(lines[3].get("finish_accepted").is_none());
        assert!(lines[3].get("finish_verification").is_none());
        let seen = backend.messages_seen();
        let Message::User { content } = &seen[1][2] else {
            panic!("expected the fed-back tool-result message");
        };
        let UserBlock::ToolResult {
            content: seen_content,
            ..
        } = &content[0]
        else {
            panic!("expected a ToolResult block");
        };
        assert_eq!(lines[3]["content"].as_str().unwrap(), seen_content);

        // 5: iteration_end (iteration 1)
        assert_eq!(lines[4]["event"], "iteration_end");
        assert_eq!(lines[4]["iteration"], 1);
        assert_eq!(lines[4]["mutated"], false);
        assert_eq!(lines[4]["last_gate_green"], false);
        assert_eq!(lines[4]["iters_since_tree_change"], 1);

        // 6: model_request (iteration 2)
        assert_eq!(lines[5]["event"], "model_request");
        assert_eq!(lines[5]["iteration"], 2);
        assert_eq!(lines[5]["message_count"], 3);
        assert_eq!(lines[5]["block_count"], 3);

        // 7: model_response (iteration 2)
        assert_eq!(lines[6]["event"], "model_response");
        assert_eq!(lines[6]["iteration"], 2);

        // 8: tool_result (finish rejected by failing checks)
        assert_eq!(lines[7]["event"], "tool_result");
        assert_eq!(lines[7]["tool_name"], "finish");
        assert_eq!(lines[7]["call_id"], "c2");
        assert_eq!(lines[7]["is_error"], true);
        assert_eq!(lines[7]["finish_accepted"], false);
        assert_eq!(lines[7]["finish_verification"]["passed"], false);
        assert_eq!(lines[7]["finish_verification"]["exit_code"], 3);
        assert!(
            lines[7]["content"]
                .as_str()
                .unwrap()
                .starts_with("finish(done) rejected: verification failed"),
            "got {}",
            lines[7]["content"]
        );

        // 9: iteration_end (iteration 2)
        assert_eq!(lines[8]["event"], "iteration_end");
        assert_eq!(lines[8]["iteration"], 2);
        assert_eq!(lines[8]["iters_since_tree_change"], 2);

        // 10: model_request (iteration 3)
        assert_eq!(lines[9]["event"], "model_request");
        assert_eq!(lines[9]["iteration"], 3);
        assert_eq!(lines[9]["message_count"], 5);
        assert_eq!(lines[9]["block_count"], 5);

        // 11: model_response (iteration 3)
        assert_eq!(lines[10]["event"], "model_response");
        assert_eq!(lines[10]["iteration"], 3);

        // 12: tool_result (finish blocked, accepted)
        assert_eq!(lines[11]["event"], "tool_result");
        assert_eq!(lines[11]["call_id"], "c3");
        assert_eq!(lines[11]["is_error"], false);
        assert_eq!(lines[11]["finish_accepted"], true);
        assert!(lines[11]["finish_verification"].is_null());
        assert_eq!(lines[11]["content"], "finish acknowledged");

        // 13: run_end
        assert_eq!(lines[12]["event"], "run_end");
        assert_eq!(lines[12]["outcome"], "Finished");
        assert_eq!(
            lines[12]["disposition"],
            serde_json::json!({"Blocked": {"decision_needed": "need input"}})
        );

        assert_reconstruction_matches(&lines, &backend);
    }

    #[tokio::test]
    async fn finish_accepted_tool_result_carries_passing_verification() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("t.jsonl");
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5)
            .with_checks(passing_runner())
            .with_transcript(path.clone(), "t");
        let backend = MockBackend::from_turns(vec![finish_call(
            "c1",
            serde_json::json!({"disposition": "done", "summary": "ok"}),
        )]);

        let RunResult { outcome, .. } = run(&backend, &tools, &ctx, &config).await;
        assert!(matches!(
            outcome,
            LoopOutcome::Finished(Disposition::Done { .. })
        ));

        let lines = read_transcript_lines(&path);
        let tool_result = lines
            .iter()
            .find(|l| l["event"] == "tool_result")
            .expect("a tool_result line exists");
        assert_eq!(tool_result["finish_accepted"], true);
        assert_eq!(tool_result["finish_verification"]["passed"], true);
    }

    #[tokio::test]
    async fn transcript_reconstruction_matches_green_stop_nudge_finish_done() {
        let runner = passing_runner();
        let tools = standard_registry(Some(runner.clone()));
        let ctx = ToolCtx::stub();
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("t.jsonl");
        let config = RunConfig::new("do the task", 3)
            .with_checks(runner)
            .with_max_nudges(1)
            .with_transcript(path.clone(), "t");

        let backend = MockBackend::from_turns(vec![
            run_checks_turn("c1"),
            turn_with(
                vec![ContentBlock::Text("looks fixed".into())],
                StopReason::EndTurn,
            ),
            finish_call(
                "c3",
                serde_json::json!({"disposition": "done", "summary": "..."}),
            ),
        ]);

        let snap_store = Arc::new(SnapshotStore::new());
        let pers = make_persistence(snap_store.clone());
        let RunResult { outcome, stats } = run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");
        assert!(matches!(
            outcome,
            LoopOutcome::Finished(Disposition::Done { .. })
        ));
        assert_eq!(stats.nudges_fired, 1);

        let lines = read_transcript_lines(&path);
        assert_reconstruction_matches(&lines, &backend);

        let nudges: Vec<&serde_json::Value> = lines
            .iter()
            .filter(|l| l["event"] == "harness_message")
            .collect();
        assert_eq!(nudges.len(), 1, "exactly one nudge event");
        let nudge = nudges[0];
        assert_eq!(nudge["placement"], "new_user_message");
        assert_eq!(
            nudge["text"].as_str().unwrap(),
            prompt::render_nudge_prompt()
        );
        assert_eq!(nudge["last_gate_green"], true);
        assert_eq!(nudge["nudge_number"], 1);
        assert_eq!(nudge["max_nudges"], config.max_nudges);
    }

    #[tokio::test]
    async fn transcript_reconstruction_matches_green_static_spin_one_nudge() {
        let runner = passing_runner();
        let tools = standard_registry(Some(runner.clone()));
        let ctx = ToolCtx::stub();
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("t.jsonl");
        let config = RunConfig::new("do the task", 4)
            .with_checks(runner)
            .with_max_nudges(2)
            .with_transcript(path.clone(), "t");

        let backend = MockBackend::from_turns(vec![
            run_checks_turn("c1"),
            echo_turn("c2"),
            echo_turn("c3"),
            echo_turn("c4"),
        ]);

        let snap_store = Arc::new(SnapshotStore::new());
        let pers = make_persistence(snap_store.clone());
        let RunResult { outcome, stats } = run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");
        assert!(matches!(outcome, LoopOutcome::MaxIterations));
        assert_eq!(stats.nudges_fired, 1);

        let lines = read_transcript_lines(&path);
        assert_reconstruction_matches(&lines, &backend);

        let nudges: Vec<&serde_json::Value> = lines
            .iter()
            .filter(|l| l["event"] == "harness_message")
            .collect();
        assert_eq!(nudges.len(), 1, "exactly one nudge event");
        let nudge = nudges[0];
        assert_eq!(nudge["placement"], "appended_to_tool_results");
        assert_eq!(
            nudge["text"].as_str().unwrap(),
            prompt::render_nudge_prompt()
        );
        assert_eq!(nudge["last_gate_green"], true);
        assert_eq!(nudge["nudge_number"], 1);
        assert_eq!(nudge["max_nudges"], config.max_nudges);
        assert_eq!(nudge["iters_since_tree_change"], config.static_tree_k);
    }

    // ---- Clock-discrimination: transcript timestamps never read config.clock

    #[tokio::test]
    async fn transcript_on_reads_the_clock_the_same_number_of_times_as_off() {
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let script = || {
            vec![
                echo_turn("c1"),
                echo_turn("c2"),
                finish_call(
                    "c3",
                    serde_json::json!({"disposition": "blocked", "decision_needed": "x"}),
                ),
            ]
        };

        let fake_off = Arc::new(FakeClock::new_auto_advance(
            UNIX_EPOCH,
            Duration::from_secs(1),
        ));
        let dyn_off: Arc<dyn crate::time::Clock> = fake_off.clone();
        let config_off = RunConfig::new("task", 10)
            .with_wall_clock_secs(1_000)
            .with_clock(dyn_off);
        let backend_off = MockBackend::from_turns(script());
        run(&backend_off, &tools, &ctx, &config_off).await;

        let fake_on = Arc::new(FakeClock::new_auto_advance(
            UNIX_EPOCH,
            Duration::from_secs(1),
        ));
        let dyn_on: Arc<dyn crate::time::Clock> = fake_on.clone();
        let dir = TempDir::new().expect("tempdir");
        let config_on = RunConfig::new("task", 10)
            .with_wall_clock_secs(1_000)
            .with_clock(dyn_on)
            .with_transcript(dir.path().join("t.jsonl"), "t");
        let backend_on = MockBackend::from_turns(script());
        run(&backend_on, &tools, &ctx, &config_on).await;

        assert_eq!(
            fake_off.now(),
            fake_on.now(),
            "the transcript must not read config.clock at all — equal read counts \
             mean equal post-run clock values"
        );
    }

    #[tokio::test]
    async fn transcript_on_does_not_change_wall_clock_breach_timing() {
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let script = || {
            vec![
                echo_turn("c1"),
                echo_turn("c2"),
                echo_turn("c3"),
                echo_turn("c4"),
                echo_turn("c5"),
            ]
        };

        for transcript_on in [false, true] {
            let fake = Arc::new(FakeClock::new_auto_advance(
                UNIX_EPOCH,
                Duration::from_secs(10),
            ));
            let dyn_clock: Arc<dyn crate::time::Clock> = fake;
            let mut config = RunConfig::new("task", 10)
                .with_wall_clock_secs(25)
                .with_clock(dyn_clock);
            let dir = TempDir::new().expect("tempdir");
            if transcript_on {
                config = config.with_transcript(dir.path().join("t.jsonl"), "t");
            }
            let backend = MockBackend::from_turns(script());

            let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;
            assert!(
                matches!(outcome, LoopOutcome::BudgetExhausted { .. }),
                "transcript_on={transcript_on}: expected BudgetExhausted, got {outcome:?}"
            );
            assert_eq!(
                stats.iterations, 3,
                "transcript_on={transcript_on}: breach must fire at the same iteration"
            );
        }
    }

    // ---- Zero behaviour change: transcript on vs off ----------------------

    #[tokio::test]
    async fn transcript_on_vs_off_does_not_change_run_outcome_or_stats() {
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let script = || {
            vec![
                echo_turn("c1"),
                echo_turn("c2"),
                finish_call(
                    "c3",
                    serde_json::json!({"disposition": "done", "summary": "ok"}),
                ),
            ]
        };

        let config_off = RunConfig::new("task", 10);
        let backend_off = MockBackend::from_turns(script());
        let RunResult {
            outcome: outcome_off,
            stats: mut stats_off,
        } = run(&backend_off, &tools, &ctx, &config_off).await;

        let dir = TempDir::new().expect("tempdir");
        let config_on = RunConfig::new("task", 10).with_transcript(dir.path().join("t.jsonl"), "t");
        let backend_on = MockBackend::from_turns(script());
        let RunResult {
            outcome: outcome_on,
            stats: mut stats_on,
        } = run(&backend_on, &tools, &ctx, &config_on).await;

        assert_eq!(backend_off.messages_seen(), backend_on.messages_seen());
        assert_eq!(backend_off.systems_seen(), backend_on.systems_seen());
        assert_eq!(
            outcome_off.into_disposition(),
            outcome_on.into_disposition()
        );
        stats_off.wall_clock = Duration::ZERO;
        stats_on.wall_clock = Duration::ZERO;
        assert_eq!(stats_off, stats_on);
    }

    #[tokio::test]
    async fn transcript_on_vs_off_persisted_produces_equal_events_and_record() {
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let script = || {
            vec![
                echo_turn("c1"),
                finish_call(
                    "c2",
                    serde_json::json!({"disposition": "done", "summary": "ok"}),
                ),
            ]
        };

        let store_off = Arc::new(SqliteRunStore::open_in_memory().expect("in-memory sqlite"));
        let pers_off = make_persistence(store_off.clone());
        let backend_off = MockBackend::from_turns(script());
        let config_off = RunConfig::new("task", 10);
        run_persisted(&backend_off, &tools, &ctx, &config_off, &pers_off)
            .await
            .expect("ok");

        let store_on = Arc::new(SqliteRunStore::open_in_memory().expect("in-memory sqlite"));
        let pers_on = make_persistence(store_on.clone());
        let backend_on = MockBackend::from_turns(script());
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("t.jsonl");
        let config_on = RunConfig::new("task", 10).with_transcript(path.clone(), "t");
        run_persisted(&backend_on, &tools, &ctx, &config_on, &pers_on)
            .await
            .expect("ok");

        let events_off = store_off
            .list_events(FIXTURE_RID)
            .await
            .expect("list events");
        let events_on = store_on
            .list_events(FIXTURE_RID)
            .await
            .expect("list events");
        assert_eq!(events_off, events_on);

        let rec_off = store_off
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        let rec_on = store_on
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        assert_eq!(rec_off.disposition, rec_on.disposition);
        assert_eq!(rec_off.messages, rec_on.messages);

        let lines = read_transcript_lines(&path);
        assert_eq!(lines[0]["run_id"], FIXTURE_RID);
        assert_eq!(lines.last().unwrap()["event"], "run_end");
    }

    // ---- StoreError exit -----------------------------------------------

    #[tokio::test]
    async fn store_error_exit_writes_run_end_store_error() {
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let backend = MockBackend::from_turns(vec![echo_turn("c1")]);
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("t.jsonl");
        let config = RunConfig::new("task", 5).with_transcript(path.clone(), "t");
        let pers = make_persistence(Arc::new(FailingStore));

        let err = run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect_err("FailingStore must abort the run");
        assert!(matches!(err, StoreError::LockPoisoned));

        let lines = read_transcript_lines(&path);
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0]["event"], "run_start");
        assert_eq!(lines[1]["event"], "model_request");
        assert_eq!(lines[1]["iteration"], 1);
        assert_eq!(lines[2]["event"], "model_response");
        assert_eq!(lines[2]["iteration"], 1);
        assert_eq!(lines[3]["event"], "run_end");
        assert_eq!(lines[3]["outcome"], "StoreError");
        assert!(lines[3]["disposition"].is_null());
        assert_eq!(
            lines[3]["detail"].as_str().unwrap(),
            StoreError::LockPoisoned.to_string()
        );
    }

    // ---- Append and resume ------------------------------------------------

    #[tokio::test]
    async fn repeated_run_calls_append_two_run_blocks() {
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let script = || {
            vec![
                echo_turn("c1"),
                finish_call(
                    "c2",
                    serde_json::json!({"disposition": "done", "summary": "ok"}),
                ),
            ]
        };
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("t.jsonl");
        let config = RunConfig::new("task", 10).with_transcript(path.clone(), "t");

        let backend1 = MockBackend::from_turns(script());
        run(&backend1, &tools, &ctx, &config).await;
        let single_run_lines = read_transcript_lines(&path);

        let backend2 = MockBackend::from_turns(script());
        run(&backend2, &tools, &ctx, &config).await;
        let two_run_lines = read_transcript_lines(&path);

        let run_start_count = two_run_lines
            .iter()
            .filter(|l| l["event"] == "run_start")
            .count();
        let run_end_count = two_run_lines
            .iter()
            .filter(|l| l["event"] == "run_end")
            .count();
        assert_eq!(run_start_count, 2);
        assert_eq!(run_end_count, 2);
        assert_eq!(
            two_run_lines.len(),
            single_run_lines.len() * 2,
            "two runs double the single-run line count"
        );

        let single_events: Vec<&str> = single_run_lines
            .iter()
            .map(|l| l["event"].as_str().unwrap())
            .collect();
        let first_block_events: Vec<&str> = two_run_lines[..single_run_lines.len()]
            .iter()
            .map(|l| l["event"].as_str().unwrap())
            .collect();
        assert_eq!(
            single_events, first_block_events,
            "the first block's event sequence is unchanged by the second run appending"
        );
    }

    #[tokio::test]
    async fn resumed_run_transcript_begins_with_reloaded_messages_and_ends_with_run_end() {
        let task_seed = Message::User {
            content: vec![UserBlock::Text("do the task".to_string())],
        };
        let asst = Message::Assistant {
            content: vec![ContentBlock::ToolCall(ToolCallRequest {
                id: "c-echo".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({}),
            })],
        };
        let tool_result = Message::User {
            content: vec![UserBlock::ToolResult {
                call_id: "c-echo".to_string(),
                content: "{}".to_string(),
                is_error: false,
            }],
        };
        let pre_messages = vec![task_seed.clone(), asst.clone(), tool_result.clone()];

        let mut record = make_minimal_record("clean-task", 1);
        record.messages = pre_messages.clone();

        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        store
            .checkpoint("clean-task:1", &record)
            .await
            .expect("checkpoint");
        store
            .append_event(
                "clean-task:1",
                Event::ModelCall {
                    seq: 0,
                    model: "test".to_string(),
                    prompt_tokens: 1,
                    completion_tokens: 1,
                },
            )
            .await
            .expect("append");

        let backend = MockBackend::from_turns(vec![finish_call(
            "c-fin",
            serde_json::json!({ "disposition": "done", "summary": "ok" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("t.jsonl");
        let config = RunConfig::new("do the task", 5).with_transcript(path.clone(), "t");

        let result = resume(
            &backend,
            &tools,
            &ctx,
            &config,
            store,
            "clean-task:1",
            ResumeMode::Crash,
        )
        .await
        .expect("resume must succeed");
        assert!(matches!(
            result.outcome,
            LoopOutcome::Finished(Disposition::Done { .. })
        ));

        let lines = read_transcript_lines(&path);
        assert_eq!(lines[0]["event"], "run_start");
        assert_eq!(lines[0]["resume"], true);
        assert_eq!(lines[0]["run_id"], "clean-task:1");
        let reloaded: Vec<Message> = serde_json::from_value(lines[0]["messages"].clone())
            .expect("messages deserialize as Vec<Message>");
        assert_eq!(reloaded, pre_messages);
        assert_eq!(lines.last().unwrap()["event"], "run_end");
    }

    // ---- Backend errors -----------------------------------------------

    #[tokio::test]
    async fn transient_exhaustion_records_every_failed_attempt() {
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let max_retries = super::DEFAULT_MAX_RETRIES;
        let script: Vec<Result<AssistantTurn, BackendError>> = (0..=max_retries)
            .map(|_| {
                Err(BackendError::Transient {
                    kind: TransientKind::Network,
                    retry_after: None,
                })
            })
            .collect();
        let backend = MockBackend::new(script);
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("t.jsonl");
        let config = RunConfig::new("task", 5)
            .with_retry_backoff_base(Duration::ZERO)
            .with_transcript(path.clone(), "t");

        let RunResult { outcome, .. } = run(&backend, &tools, &ctx, &config).await;
        assert!(matches!(outcome, LoopOutcome::BackendError(_)));
        assert_eq!(backend.calls(), max_retries + 1);

        let lines = read_transcript_lines(&path);
        let backend_errors: Vec<&serde_json::Value> = lines
            .iter()
            .filter(|l| l["event"] == "backend_error")
            .collect();
        assert_eq!(backend_errors.len(), (max_retries + 1) as usize);
        for (i, ev) in backend_errors.iter().enumerate() {
            let attempt = u32::try_from(i).unwrap();
            assert_eq!(ev["attempt"], attempt);
            assert_eq!(ev["retryable"], true);
            let is_last = attempt == max_retries;
            assert_eq!(ev["will_retry"], !is_last);
            if is_last {
                assert!(ev["retry_delay_ms"].is_null());
            } else {
                assert!(ev["retry_delay_ms"].is_number());
            }
            assert!(ev["error_debug"].as_str().unwrap().contains("Transient"));
        }
        assert!(
            !lines.iter().any(|l| l["event"] == "model_response"),
            "no successful turn was ever drawn"
        );
        let run_end = lines.last().unwrap();
        assert_eq!(run_end["event"], "run_end");
        assert_eq!(run_end["outcome"], "BackendError");
    }

    #[tokio::test]
    async fn terminal_backend_error_records_a_single_non_retryable_entry() {
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let backend = MockBackend::new(vec![Err(BackendError::Terminal {
            kind: TerminalKind::Other,
            message: "boom".to_string(),
        })]);
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("t.jsonl");
        let config = RunConfig::new("task", 5).with_transcript(path.clone(), "t");

        let RunResult { outcome, .. } = run(&backend, &tools, &ctx, &config).await;
        assert!(matches!(outcome, LoopOutcome::BackendError(_)));

        let lines = read_transcript_lines(&path);
        let backend_errors: Vec<&serde_json::Value> = lines
            .iter()
            .filter(|l| l["event"] == "backend_error")
            .collect();
        assert_eq!(backend_errors.len(), 1);
        let ev = backend_errors[0];
        assert_eq!(ev["iteration"], 1);
        assert_eq!(ev["attempt"], 0);
        assert_eq!(ev["retryable"], false);
        assert_eq!(ev["will_retry"], false);
        assert!(ev["retry_delay_ms"].is_null());
        assert_eq!(ev["error"], "terminal backend failure (Other): boom");

        let run_end = lines.last().unwrap();
        assert_eq!(run_end["event"], "run_end");
        assert_eq!(run_end["outcome"], "BackendError");
        assert_eq!(run_end["detail"], "terminal backend failure (Other): boom");
    }

    // ---- Engine-level best-effort failure: never changes the outcome ------

    #[tokio::test]
    async fn best_effort_failure_parent_is_file_does_not_change_outcome() {
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let script = || {
            vec![
                echo_turn("c1"),
                finish_call(
                    "c2",
                    serde_json::json!({"disposition": "done", "summary": "ok"}),
                ),
            ]
        };

        let config_off = RunConfig::new("task", 10);
        let backend_off = MockBackend::from_turns(script());
        let RunResult {
            outcome: outcome_off,
            stats: mut stats_off,
        } = run(&backend_off, &tools, &ctx, &config_off).await;

        let parent_file = tempfile::NamedTempFile::new().expect("temp file");
        let path = parent_file.path().join("t.jsonl");
        let config_bad = RunConfig::new("task", 10).with_transcript(path, "t");
        let backend_bad = MockBackend::from_turns(script());
        let RunResult {
            outcome: outcome_bad,
            stats: mut stats_bad,
        } = run(&backend_bad, &tools, &ctx, &config_bad).await;

        assert_eq!(
            outcome_off.into_disposition(),
            outcome_bad.into_disposition()
        );
        stats_off.wall_clock = Duration::ZERO;
        stats_bad.wall_clock = Duration::ZERO;
        assert_eq!(stats_off, stats_bad);
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn best_effort_failure_dev_full_does_not_change_outcome() {
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let script = || {
            vec![
                echo_turn("c1"),
                finish_call(
                    "c2",
                    serde_json::json!({"disposition": "done", "summary": "ok"}),
                ),
            ]
        };

        let config_off = RunConfig::new("task", 10);
        let backend_off = MockBackend::from_turns(script());
        let RunResult {
            outcome: outcome_off,
            stats: mut stats_off,
        } = run(&backend_off, &tools, &ctx, &config_off).await;

        let config_bad = RunConfig::new("task", 10).with_transcript("/dev/full", "t");
        let backend_bad = MockBackend::from_turns(script());
        let RunResult {
            outcome: outcome_bad,
            stats: mut stats_bad,
        } = run(&backend_bad, &tools, &ctx, &config_bad).await;

        assert_eq!(
            outcome_off.into_disposition(),
            outcome_bad.into_disposition()
        );
        stats_off.wall_clock = Duration::ZERO;
        stats_bad.wall_clock = Duration::ZERO;
        assert_eq!(stats_off, stats_bad);
    }

    // =====================================================================
    // Leg 3 of the completion contract — `done` requires observed change
    // =====================================================================

    /// A `RunStats` with every counter at zero — for unit-testing
    /// `emit_run_end` directly.
    fn zero_stats() -> RunStats {
        RunStats {
            iterations: 0,
            input_tokens: 0,
            output_tokens: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            wall_clock: Duration::ZERO,
            gates_green_at_exit: false,
            nudges_fired: 0,
            tree_dirty: false,
            iters_since_tree_change_at_exit: 0,
            peak_iters_since_tree_change: 0,
            mutating_iters: 0,
            bash_calls_ok: 0,
            edit_file_calls_ok: 0,
            invalid_finish_calls: 0,
            first_invalid_finish_raw: None,
            no_change_rejections: 0,
            already_satisfied_check_rejections: 0,
            answer_schema_rejections: 0,
            modified_workspace_rejections: 0,
            tree_baseline_unobservable: false,
        }
    }

    /// `git init` `root` and return a `ToolCtx` rooted there.
    fn git_ctx(root: &std::path::Path) -> ToolCtx {
        let out = std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(root)
            .output()
            .expect("git init runs");
        assert!(out.status.success(), "git init failed: {out:?}");
        let workspace = Workspace::new(root, None).expect("workspace");
        ToolCtx::new(Arc::new(workspace), Arc::new(crate::tool::StubOffloadSink))
    }

    /// Run `git` with `args` in `dir`, asserting success. Commits carry an
    /// explicit identity so no global git config is required.
    fn git_in(dir: &std::path::Path, args: &[&str]) {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .env("GIT_AUTHOR_NAME", "t")
            .env("GIT_AUTHOR_EMAIL", "t@example.com")
            .env("GIT_COMMITTER_NAME", "t")
            .env("GIT_COMMITTER_EMAIL", "t@example.com")
            .output()
            .expect("git runs");
        assert!(out.status.success(), "git {args:?} failed: {out:?}");
    }

    fn registry_with_finish_and_edit() -> ToolRegistry {
        let mut registry = ToolRegistry::new();
        registry.register("edit_file", Arc::new(EditFileTool));
        registry.register(FINISH_TOOL_NAME, Arc::new(FinishTool::default()));
        registry
    }

    #[tokio::test]
    async fn bare_finish_done_on_unchanged_git_tree_is_rejected_and_loop_continues() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        let ctx = git_ctx(&root_path);
        let tools = registry_with_finish_and_edit();

        let backend = MockBackend::from_turns(vec![
            finish_call(
                "c-bare",
                serde_json::json!({ "disposition": "done", "summary": "nothing" }),
            ),
            finish_call(
                "c-bare-2",
                serde_json::json!({ "disposition": "done", "summary": "nothing" }),
            ),
        ]);
        let config = RunConfig::new("do the work", 2);
        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert!(
            matches!(outcome, LoopOutcome::MaxIterations),
            "an unchanged tree must NOT terminate as Finished(Done); got {outcome:?}"
        );
        assert_eq!(stats.no_change_rejections, 2);
        assert!(!stats.tree_baseline_unobservable);
        let fed_back = backend.last_messages();
        assert!(
            fed_back.iter().any(|m| matches!(
                m,
                Message::User { content }
                    if content.iter().any(|b| matches!(
                        b,
                        UserBlock::ToolResult { content, is_error, .. }
                            if *is_error && *content == no_change_rejection_content()
                    ))
            )),
            "the fed-back result must be the pinned no-change rejection"
        );
    }

    #[tokio::test]
    async fn finish_done_after_an_edit_is_accepted_with_tree_changed_evidence() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        let ctx = git_ctx(&root_path);
        let tools = registry_with_finish_and_edit();

        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call(
                    "c-edit",
                    "edit_file",
                    serde_json::json!({
                        "path": "work.txt",
                        "old_string": "",
                        "new_string": "done\n",
                    }),
                )],
                StopReason::ToolUse,
            ),
            finish_call(
                "c-done",
                serde_json::json!({ "disposition": "done", "summary": "edited" }),
            ),
        ]);
        let config = RunConfig::new("do the work", 5);
        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Done { change, .. }) => {
                assert_eq!(change, ChangeEvidence::TreeChanged);
            }
            other => panic!("expected Finished(Done{{TreeChanged}}); got {other:?}"),
        }
        assert_eq!(stats.no_change_rejections, 0);
    }

    #[tokio::test]
    async fn pre_dirty_attachments_dir_does_not_mask_a_no_work_run() {
        // THE PRODUCTION SEAM. `agent-gtd-dispatch`'s `stage_attachments`
        // creates an untracked `<run_id>-attachments/` directory inside the
        // clone root BEFORE the agent starts, and tells the agent never to
        // commit it. A bare `current`-dirtiness test would read TreeChanged
        // for the whole run on every attachment-carrying item — exactly the
        // code path where a false Done costs a real push. The comparison is
        // baseline-RELATIVE, so the pre-existing entry cancels.
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        std::fs::create_dir(root_path.join("run-123-attachments")).expect("mkdir attachments");
        std::fs::write(root_path.join("run-123-attachments/spec.md"), "spec\n").expect("write");
        let ctx = git_ctx(&root_path);
        let tools = registry_with_finish_and_edit();

        let backend = MockBackend::from_turns(vec![finish_call(
            "c-bare",
            serde_json::json!({ "disposition": "done", "summary": "nothing" }),
        )]);
        let config = RunConfig::new("do the work", 1);
        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;
        assert!(
            matches!(outcome, LoopOutcome::MaxIterations),
            "a pre-dirty workspace must not mask a no-work run; got {outcome:?}"
        );
        assert_eq!(stats.no_change_rejections, 1);

        // Mirror case: the same pre-dirty workspace, but the agent edits a
        // tracked file — accepted.
        let root2 = TempDir::new().expect("tempdir");
        let root2_path = root2.path().canonicalize().expect("canonicalize");
        std::fs::write(root2_path.join("tracked.txt"), "before\n").expect("write");
        let ctx2 = git_ctx(&root2_path);
        git_in(&root2_path, &["add", "tracked.txt"]);
        git_in(&root2_path, &["commit", "-qm", "seed"]);
        std::fs::create_dir(root2_path.join("run-123-attachments")).expect("mkdir attachments");
        std::fs::write(root2_path.join("run-123-attachments/spec.md"), "spec\n").expect("write");

        let backend2 = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call(
                    "c-edit",
                    "edit_file",
                    serde_json::json!({
                        "path": "tracked.txt",
                        "old_string": "before",
                        "new_string": "after",
                    }),
                )],
                StopReason::ToolUse,
            ),
            finish_call(
                "c-done",
                serde_json::json!({ "disposition": "done", "summary": "edited" }),
            ),
        ]);
        let config2 = RunConfig::new("do the work", 5);
        let RunResult { outcome, .. } = run(&backend2, &tools, &ctx2, &config2).await;
        match outcome {
            LoopOutcome::Finished(Disposition::Done { change, .. }) => {
                assert_eq!(change, ChangeEvidence::TreeChanged);
            }
            other => panic!("expected Finished(Done{{TreeChanged}}); got {other:?}"),
        }
    }

    #[tokio::test]
    async fn non_git_workspace_fails_open_and_accepts_a_bare_done() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        // Pin the load-bearing fail-open assumption with a test rather than
        // leaving it to the ambient environment: this root really is NOT a
        // git work tree.
        let probe = std::process::Command::new("git")
            .args(["rev-parse", "--is-inside-work-tree"])
            .current_dir(&root_path)
            .output()
            .expect("git runs");
        assert!(
            !probe.status.success(),
            "the fail-open assumption requires a non-git workspace root"
        );

        let workspace = Workspace::new(&root_path, None).expect("workspace");
        let ctx = ToolCtx::new(Arc::new(workspace), Arc::new(crate::tool::StubOffloadSink));
        let tools = registry_with_finish_and_edit();
        let backend = MockBackend::from_turns(vec![finish_call(
            "c-bare",
            serde_json::json!({ "disposition": "done", "summary": "nothing" }),
        )]);
        let config = RunConfig::new("do the work", 2);
        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Done { change, .. }) => {
                assert!(
                    matches!(change, ChangeEvidence::Unobservable { .. }),
                    "an unobservable workspace must fail OPEN; got {change:?}"
                );
            }
            other => panic!("expected Finished(Done{{Unobservable}}); got {other:?}"),
        }
        assert!(stats.tree_baseline_unobservable);
        assert_eq!(stats.no_change_rejections, 0);
    }

    #[tokio::test]
    async fn committed_work_moves_head_and_reads_as_tree_changed() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        std::fs::write(root_path.join("seed.txt"), "seed\n").expect("write");
        let ctx = git_ctx(&root_path);
        git_in(&root_path, &["add", "seed.txt"]);
        git_in(&root_path, &["commit", "-qm", "seed"]);

        let mut tools = ToolRegistry::new();
        tools.register("bash", Arc::new(crate::tools::bash::BashTool));
        tools.register(FINISH_TOOL_NAME, Arc::new(FinishTool::default()));

        // The agent commits its own change: the tree is CLEAN at finish time
        // but HEAD moved. No template instructs a commit, but `bash` permits
        // one — and the moved-HEAD arm must read as TreeChanged.
        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call(
                    "c-bash",
                    "bash",
                    serde_json::json!({
                        "command": "printf 'more\\n' >> seed.txt && \
                                    git -c user.name=t -c user.email=t@example.com \
                                    commit -qam agent",
                    }),
                )],
                StopReason::ToolUse,
            ),
            finish_call(
                "c-done",
                serde_json::json!({ "disposition": "done", "summary": "committed" }),
            ),
        ]);
        let config = RunConfig::new("commit the work", 5);
        let RunResult { outcome, .. } = run(&backend, &tools, &ctx, &config).await;
        match outcome {
            LoopOutcome::Finished(Disposition::Done { change, .. }) => {
                assert_eq!(change, ChangeEvidence::TreeChanged, "HEAD moved");
            }
            other => panic!("expected Finished(Done{{TreeChanged}}); got {other:?}"),
        }
    }

    #[tokio::test]
    async fn already_satisfied_with_green_checks_terminates_and_records_the_tree() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        let ctx = git_ctx(&root_path);
        let tools = registry_with_finish_and_edit();
        let runner = ChecksRunner::new(
            CheckCommand {
                program: "/bin/sh".to_string(),
                args: vec!["-c".to_string(), "exit 0".to_string()],
            },
            root_path.clone(),
            Duration::from_secs(10),
        );

        let backend = MockBackend::from_turns(vec![finish_call(
            "c-as",
            serde_json::json!({
                "disposition": "already_satisfied",
                "summary": "s",
                "reason": "the flag was already set",
            }),
        )]);
        let config = RunConfig::new("confirm the flag", 3).with_checks(runner);
        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::AlreadySatisfied {
                reason,
                verification: Verification::Checks(report),
                change,
            }) => {
                assert_eq!(reason, "the flag was already set");
                assert!(report.passed);
                assert_eq!(change, ChangeEvidence::TreeUnchanged);
            }
            other => panic!("expected Finished(AlreadySatisfied); got {other:?}"),
        }
        assert_eq!(stats.already_satisfied_check_rejections, 0);
        assert_eq!(stats.invalid_finish_calls, 0);
    }

    #[tokio::test]
    async fn already_satisfied_records_an_incidental_tree_change_rather_than_rejecting() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        let ctx = git_ctx(&root_path);
        let tools = registry_with_finish_and_edit();

        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call(
                    "c-edit",
                    "edit_file",
                    serde_json::json!({
                        "path": "scratch.txt",
                        "old_string": "",
                        "new_string": "incidental\n",
                    }),
                )],
                StopReason::ToolUse,
            ),
            finish_call(
                "c-as",
                serde_json::json!({
                    "disposition": "already_satisfied",
                    "reason": "nothing needed changing",
                }),
            ),
        ]);
        let config = RunConfig::new("confirm", 3);
        let RunResult { outcome, .. } = run(&backend, &tools, &ctx, &config).await;
        match outcome {
            LoopOutcome::Finished(Disposition::AlreadySatisfied { change, .. }) => {
                assert_eq!(
                    change,
                    ChangeEvidence::TreeChanged,
                    "an incidental scratch file is RECORDED, not rejected"
                );
            }
            other => panic!("expected Finished(AlreadySatisfied); got {other:?}"),
        }
    }

    #[tokio::test]
    async fn already_satisfied_on_a_red_gate_is_rejected_and_counted() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        let ctx = git_ctx(&root_path);
        let tools = registry_with_finish_and_edit();
        let runner = ChecksRunner::new(
            CheckCommand {
                program: "/bin/sh".to_string(),
                args: vec!["-c".to_string(), "exit 1".to_string()],
            },
            root_path.clone(),
            Duration::from_secs(10),
        );

        let backend = MockBackend::from_turns(vec![
            finish_call(
                "c-as",
                serde_json::json!({
                    "disposition": "already_satisfied",
                    "reason": "nothing needed changing",
                }),
            ),
            finish_call(
                "c-as-2",
                serde_json::json!({
                    "disposition": "already_satisfied",
                    "reason": "still nothing",
                }),
            ),
        ]);
        let config = RunConfig::new("confirm", 2).with_checks(runner);
        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;
        assert!(
            matches!(outcome, LoopOutcome::MaxIterations),
            "a no-op claim on a red repo is never legitimate; got {outcome:?}"
        );
        assert_eq!(stats.already_satisfied_check_rejections, 2);
    }

    #[tokio::test]
    async fn already_satisfied_without_reason_is_a_malformed_finish_call() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        let ctx = git_ctx(&root_path);
        let tools = registry_with_finish_and_edit();

        let backend = MockBackend::from_turns(vec![
            finish_call(
                "c-as",
                serde_json::json!({ "disposition": "already_satisfied", "reason": "" }),
            ),
            finish_call(
                "c-as-2",
                serde_json::json!({ "disposition": "already_satisfied", "reason": "   " }),
            ),
        ]);
        let config = RunConfig::new("confirm", 2);
        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;
        assert!(matches!(outcome, LoopOutcome::MaxIterations));
        assert_eq!(stats.invalid_finish_calls, 2);
        assert_eq!(
            stats.first_invalid_finish_raw.as_deref(),
            Some("already_satisfied without reason")
        );
        let fed_back = backend.last_messages();
        assert!(fed_back.iter().any(|m| matches!(
            m,
            Message::User { content }
                if content.iter().any(|b| matches!(
                    b,
                    UserBlock::ToolResult { content, is_error, .. }
                        if *is_error && *content == missing_reason_rejection_content()
                ))
        )));
    }

    #[test]
    fn rejection_content_strings_are_pinned_and_all_say_rejected() {
        assert_eq!(
            no_change_rejection_content(),
            "finish(done) rejected: the working tree is unchanged since this run started — \
             no work was done. Make the change the task requires and finish again, or call \
             finish with disposition `already_satisfied` and a `reason` if the task was \
             already complete."
        );
        assert_eq!(
            missing_reason_rejection_content(),
            "finish rejected: already_satisfied requires a non-empty `reason` explaining what \
             you checked and why the task was already complete. Call finish again with a \
             reason, or with a different disposition."
        );
        assert_eq!(
            missing_result_rejection_content(),
            "finish rejected: answer requires a `result` that conforms to the configured \
             result schema. Call finish again with disposition `answer` and a `result`, or \
             with a different disposition."
        );
        assert!(no_change_rejection_content().contains("rejected"));
        assert!(missing_reason_rejection_content().contains("rejected"));
        assert!(missing_result_rejection_content().contains("rejected"));
        let answer_errors = answer_schema_rejection_content(&["/verdict: bad".to_string()]);
        assert!(
            answer_errors.starts_with(
                "finish(answer) rejected: result does not conform to the configured schema:"
            ),
            "got {answer_errors}"
        );
        assert!(answer_errors.contains("rejected"));
        let report = CheckReport {
            passed: false,
            exit_code: Some(1),
            timed_out: false,
            excerpt: String::new(),
            offload_path: None,
            duration: Duration::from_millis(1),
        };
        assert!(rejection_content("already_satisfied", &report).contains("rejected"));
        assert!(
            rejection_content("already_satisfied", &report)
                .starts_with("finish(already_satisfied) rejected:")
        );
    }

    #[tokio::test]
    async fn resume_supplies_an_unobservable_baseline_so_a_bare_done_is_accepted() {
        let dir = TempDir::new().expect("tempdir");
        let root_path = dir.path().canonicalize().expect("canonicalize");
        let ctx = git_ctx(&root_path);
        // The workspace already carries edits, as it would after a crash.
        std::fs::write(root_path.join("pre_crash.txt"), "wip\n").expect("write");
        let tools = registry_with_finish_and_edit();

        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        let mut record = make_minimal_record("resumed-leg3", 1);
        record.messages = vec![Message::User {
            content: vec![UserBlock::Text("do the task".to_string())],
        }];
        store
            .checkpoint(&record.run_id, &record)
            .await
            .expect("checkpoint");
        let rid = record.run_id.clone();

        let backend = MockBackend::from_turns(vec![finish_call(
            "c-done",
            serde_json::json!({ "disposition": "done", "summary": "already fixed pre-crash" }),
        )]);
        let config = RunConfig::new("finish the pre-crash work", 2);
        let RunResult { outcome, stats } = resume(
            &backend,
            &tools,
            &ctx,
            &config,
            Arc::clone(&store),
            &rid,
            ResumeMode::Crash,
        )
        .await
        .expect("resume");

        match outcome {
            LoopOutcome::Finished(Disposition::Done { change, .. }) => {
                assert!(
                    matches!(change, ChangeEvidence::Unobservable { .. }),
                    "a resumed run's pre-crash baseline is unavailable, so it fails open; \
                     got {change:?}"
                );
            }
            other => panic!("expected Finished(Done{{Unobservable}}); got {other:?}"),
        }
        assert!(stats.tree_baseline_unobservable);
    }

    #[tokio::test]
    async fn transcript_records_the_no_change_decision_inputs_and_the_baseline() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        let ctx = git_ctx(&root_path);
        let tools = registry_with_finish_and_edit();
        let runner = ChecksRunner::new(
            CheckCommand {
                program: "/bin/sh".to_string(),
                args: vec!["-c".to_string(), "exit 0".to_string()],
            },
            root_path.clone(),
            Duration::from_secs(10),
        );

        let transcript = root_path.join("t.jsonl");
        let backend = MockBackend::from_turns(vec![finish_call(
            "c-bare",
            serde_json::json!({ "disposition": "done", "summary": "nothing" }),
        )]);
        let config = RunConfig::new("do the work", 1)
            .with_checks(runner)
            .with_transcript(transcript.clone(), "leg3");
        let _ = run(&backend, &tools, &ctx, &config).await;

        let lines = read_transcript_lines(&transcript);
        assert_eq!(lines[0]["event"], "run_start");
        assert!(
            lines[0]["tree_baseline"]["Observed"].is_object(),
            "run_start must carry the observed baseline: {}",
            lines[0]
        );
        let tool_result = lines
            .iter()
            .find(|l| l["event"] == "tool_result")
            .expect("a tool_result event");
        assert_eq!(tool_result["finish_accepted"], false);
        assert_eq!(
            tool_result["finish_change"],
            serde_json::json!("TreeUnchanged")
        );
        assert!(
            tool_result["tree_current"]["Observed"].is_object(),
            "tool_result must carry the observation it classified: {tool_result}"
        );
        assert_eq!(
            tool_result["finish_verification"]["passed"], true,
            "the checks passed; only the tree precondition failed"
        );
        let run_end = lines
            .iter()
            .find(|l| l["event"] == "run_end")
            .expect("a run_end event");
        assert_eq!(run_end["stats"]["no_change_rejections"], 1);
        assert_eq!(run_end["stats"]["tree_baseline_unobservable"], false);
        assert_eq!(run_end["stats"]["already_satisfied_check_rejections"], 0);
    }

    #[tokio::test]
    async fn transcript_shows_a_moved_head_between_baseline_and_accept() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        std::fs::write(root_path.join("seed.txt"), "seed\n").expect("write");
        let ctx = git_ctx(&root_path);
        git_in(&root_path, &["add", "seed.txt"]);
        git_in(&root_path, &["commit", "-qm", "seed"]);

        let mut tools = ToolRegistry::new();
        tools.register("bash", Arc::new(crate::tools::bash::BashTool));
        tools.register(FINISH_TOOL_NAME, Arc::new(FinishTool::default()));

        let transcript = root_path.join("t.jsonl");
        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call(
                    "c-bash",
                    "bash",
                    serde_json::json!({
                        "command": "printf 'more\\n' >> seed.txt && \
                                    git -c user.name=t -c user.email=t@example.com \
                                    commit -qam agent",
                    }),
                )],
                StopReason::ToolUse,
            ),
            finish_call(
                "c-done",
                serde_json::json!({ "disposition": "done", "summary": "committed" }),
            ),
        ]);
        let config =
            RunConfig::new("commit the work", 5).with_transcript(transcript.clone(), "leg3-head");
        let _ = run(&backend, &tools, &ctx, &config).await;

        let lines = read_transcript_lines(&transcript);
        let baseline_head = lines[0]["tree_baseline"]["Observed"]["head"].clone();
        let finish_result = lines
            .iter()
            .rfind(|l| l["event"] == "tool_result")
            .expect("a finish tool_result");
        let current_head = finish_result["tree_current"]["Observed"]["head"].clone();
        assert!(baseline_head.is_string() && current_head.is_string());
        assert_ne!(baseline_head, current_head, "HEAD must have moved");
        assert_eq!(
            finish_result["finish_change"],
            serde_json::json!("TreeChanged")
        );
    }

    #[tokio::test]
    async fn unobservable_baseline_warns_and_still_records_the_baseline_in_run_start() {
        let dir = TempDir::new().expect("tempdir");
        let root_path = dir.path().canonicalize().expect("canonicalize");
        let workspace = Workspace::new(&root_path, None).expect("workspace");
        let ctx = ToolCtx::new(Arc::new(workspace), Arc::new(crate::tool::StubOffloadSink));
        let tools = registry_with_finish_and_echo();

        let transcript = root_path.join("t.jsonl");
        let backend = MockBackend::from_turns(vec![finish_call(
            "c-done",
            serde_json::json!({ "disposition": "done", "summary": "s" }),
        )]);
        let config = RunConfig::new("t", 1).with_transcript(transcript.clone(), "inert");
        let RunResult { stats, .. } = run(&backend, &tools, &ctx, &config).await;
        assert!(stats.tree_baseline_unobservable);

        let lines = read_transcript_lines(&transcript);
        assert!(
            lines[0]["tree_baseline"]["Unobservable"]["reason"]
                .as_str()
                .expect("a reason string")
                .contains("git status"),
            "run_start must carry the unobservable baseline reason: {}",
            lines[0]
        );
    }

    #[test]
    fn emit_run_end_audits_a_done_that_reached_the_terminal_with_an_unchanged_tree() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("audit.jsonl");
        let mut writer = TranscriptWriter::open(Some(&TranscriptConfig {
            path: path.clone(),
            label: "audit".to_string(),
        }));
        let result: Result<LoopOutcome, StoreError> =
            Ok(LoopOutcome::Finished(Disposition::Done {
                summary: "hand-built".to_string(),
                verification: Verification::NoChecksConfigured,
                change: ChangeEvidence::TreeUnchanged,
            }));
        emit_run_end(&mut writer, &result, &zero_stats(), Some("task:1"), None);
        drop(writer);

        let lines = read_transcript_lines(&path);
        let violation = lines
            .iter()
            .find(|l| l["event"] == "contract_violation")
            .expect("a contract_violation event");
        assert_eq!(violation["kind"], "done_with_unchanged_tree");
        assert_eq!(violation["run_id"], "task:1");
        assert_eq!(violation["change"], serde_json::json!("TreeUnchanged"));
    }

    #[test]
    fn emit_run_end_does_not_audit_an_honest_done() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("clean.jsonl");
        let mut writer = TranscriptWriter::open(Some(&TranscriptConfig {
            path: path.clone(),
            label: "clean".to_string(),
        }));
        let result: Result<LoopOutcome, StoreError> =
            Ok(LoopOutcome::Finished(Disposition::Done {
                summary: "real".to_string(),
                verification: Verification::NoChecksConfigured,
                change: ChangeEvidence::TreeChanged,
            }));
        emit_run_end(&mut writer, &result, &zero_stats(), None, None);
        drop(writer);

        let lines = read_transcript_lines(&path);
        assert!(lines.iter().all(|l| l["event"] != "contract_violation"));
    }

    // ---- answer mode ----------------------------------------------------

    /// A schema that accepts `{"verdict": "ok"|"bad"}` objects and nothing
    /// else — small enough to reason about, strict enough to produce a
    /// predictable error message.
    fn verdict_schema() -> AnswerSchema {
        AnswerSchema::compile(&serde_json::json!({
            "type": "object",
            "properties": { "verdict": { "enum": ["ok", "bad"] } },
            "required": ["verdict"],
        }))
        .expect("the verdict schema compiles")
    }

    #[test]
    fn answer_schema_compile_rejects_a_non_schema() {
        let err = AnswerSchema::compile(&serde_json::json!({ "type": 12345 }))
            .expect_err("`{\"type\": 12345}` is not a valid JSON Schema");
        assert!(
            !err.message.is_empty(),
            "the compile failure must carry a message"
        );
    }

    #[test]
    fn answer_schema_source_round_trips_the_value_it_compiled() {
        let schema = serde_json::json!({ "type": "object" });
        let compiled = AnswerSchema::compile(&schema).expect("compiles");
        assert_eq!(compiled.source(), &schema);
    }

    #[test]
    fn validation_errors_normalizes_a_root_path_to_slash() {
        let compiled =
            AnswerSchema::compile(&serde_json::json!({ "type": "object" })).expect("compiles");
        let errors = compiled.validation_errors(&serde_json::json!(1));
        assert_eq!(errors.len(), 1, "got {errors:?}");
        assert!(
            errors[0].starts_with("/: "),
            "a ROOT-level error renders its empty instance path as `/`; got {:?}",
            errors[0]
        );
    }

    #[test]
    fn validation_errors_names_the_failing_property_and_is_empty_when_valid() {
        let compiled = AnswerSchema::compile(&serde_json::json!({
            "type": "object",
            "properties": { "verdict": { "enum": ["ok"] } },
        }))
        .expect("compiles");
        let errors = compiled.validation_errors(&serde_json::json!({ "verdict": "nope" }));
        assert_eq!(errors.len(), 1, "got {errors:?}");
        assert!(errors[0].starts_with("/verdict: "), "got {:?}", errors[0]);
        assert!(
            compiled
                .validation_errors(&serde_json::json!({ "verdict": "ok" }))
                .is_empty()
        );
    }

    #[test]
    fn run_config_answer_schema_defaults_to_none_and_the_builder_sets_it() {
        let config = RunConfig::new("t", 1);
        assert!(config.answer_schema.is_none());
        let config = config.with_answer_schema(verdict_schema());
        assert!(config.answer_schema.is_some());
    }

    #[test]
    fn answer_schema_rejection_content_caps_lines_and_reports_the_remainder() {
        let errors: Vec<String> = (0..25).map(|i| format!("/f{i}: bad")).collect();
        let content = answer_schema_rejection_content(&errors);
        let lines: Vec<&str> = content.lines().collect();
        // header + MAX_LINES error lines + the "…and N more" line
        assert_eq!(lines.len(), ANSWER_SCHEMA_ERRORS_MAX_LINES + 2);
        assert_eq!(
            lines[0],
            "finish(answer) rejected: result does not conform to the configured schema:"
        );
        assert_eq!(
            lines[ANSWER_SCHEMA_ERRORS_MAX_LINES + 1],
            format!("…and {} more", 25 - ANSWER_SCHEMA_ERRORS_MAX_LINES)
        );
    }

    #[test]
    fn answer_schema_rejection_content_truncates_one_enormous_error() {
        let errors = vec!["x".repeat(10_000)];
        let content = answer_schema_rejection_content(&errors);
        let marker = format!("…[truncated at {ANSWER_SCHEMA_ERRORS_CAP} chars]");
        assert!(content.ends_with(&marker), "expected the truncation marker");
        let body = content
            .strip_suffix(&marker)
            .expect("the marker was just asserted");
        assert!(
            body.chars().count() <= ANSWER_SCHEMA_ERRORS_CAP,
            "pre-marker body was {} chars",
            body.chars().count()
        );
    }

    /// Build mode is byte-identical: the `answer` disposition is not
    /// advertised, so a `finish(answer)` is rejected by the pre-existing
    /// invalid-disposition path with the pre-existing wording and counters.
    #[tokio::test]
    async fn answer_without_a_configured_schema_is_the_inherited_invalid_rejection() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        let ctx = git_ctx(&root_path);
        let tools = registry_with_finish_and_edit();

        // A second, non-finish turn so the fed-back rejection reaches the
        // backend and can be inspected via `last_messages`.
        let backend = MockBackend::from_turns(vec![
            finish_call(
                "c-answer",
                serde_json::json!({ "disposition": "answer", "result": {} }),
            ),
            turn_with(
                vec![ContentBlock::Text("hm".to_string())],
                StopReason::EndTurn,
            ),
        ]);
        let config = RunConfig::new("answer me", 3);
        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert!(
            matches!(outcome, LoopOutcome::StoppedWithoutFinish),
            "an un-advertised answer must NOT terminate the loop as Finished; got {outcome:?}"
        );
        assert_eq!(stats.invalid_finish_calls, 1);
        assert_eq!(
            stats.first_invalid_finish_raw.as_deref(),
            Some("\"answer\"")
        );
        assert_eq!(stats.answer_schema_rejections, 0);

        let fed_back = backend.last_messages();
        assert!(
            fed_back.iter().any(|m| matches!(
                m,
                Message::User { content }
                    if content.iter().any(|b| matches!(
                        b,
                        UserBlock::ToolResult { content, is_error, .. }
                            if *is_error && content == "finish rejected: disposition must be \
                                one of: done, blocked, failed, already_satisfied; got \
                                \"answer\". Call finish again with one of those values."
                    ))
            )),
            "the inherited invalid-disposition wording must be reused byte-for-byte"
        );
    }

    #[tokio::test]
    async fn answer_without_a_result_is_a_malformed_finish_call() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        let ctx = git_ctx(&root_path);
        let tools = registry_with_finish_and_edit();

        let backend = MockBackend::from_turns(vec![
            finish_call("c-answer", serde_json::json!({ "disposition": "answer" })),
            turn_with(
                vec![ContentBlock::Text("hm".to_string())],
                StopReason::EndTurn,
            ),
        ]);
        let config = RunConfig::new("answer me", 3).with_answer_schema(verdict_schema());
        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert!(matches!(outcome, LoopOutcome::StoppedWithoutFinish));
        assert_eq!(stats.invalid_finish_calls, 1);
        assert_eq!(
            stats.first_invalid_finish_raw.as_deref(),
            Some("answer without result")
        );
        assert_eq!(stats.answer_schema_rejections, 0);

        let fed_back = backend.last_messages();
        assert!(fed_back.iter().any(|m| matches!(
            m,
            Message::User { content }
                if content.iter().any(|b| matches!(
                    b,
                    UserBlock::ToolResult { content, is_error, .. }
                        if *is_error && *content == missing_result_rejection_content()
                ))
        )));
    }

    #[tokio::test]
    async fn schema_invalid_answers_are_counted_then_a_valid_one_terminates() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        let ctx = git_ctx(&root_path);
        let tools = registry_with_finish_and_edit();
        let transcript = root_path.join("answer.jsonl");

        let backend = MockBackend::from_turns(vec![
            finish_call(
                "c-a1",
                serde_json::json!({ "disposition": "answer", "result": { "verdict": "nope" } }),
            ),
            finish_call(
                "c-a2",
                serde_json::json!({ "disposition": "answer", "result": 7 }),
            ),
            finish_call(
                "c-a3",
                serde_json::json!({ "disposition": "answer", "result": { "verdict": "ok" } }),
            ),
        ]);
        let config = RunConfig::new("answer me", 4)
            .with_answer_schema(verdict_schema())
            .with_transcript(transcript.clone(), "answer");
        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Answer { ref result, .. }) => {
                assert_eq!(result, &serde_json::json!({ "verdict": "ok" }));
            }
            other => panic!("expected Finished(Answer); got {other:?}"),
        }
        assert_eq!(stats.answer_schema_rejections, 2);
        assert_eq!(stats.invalid_finish_calls, 0);
        assert_eq!(stats.no_change_rejections, 0);
        assert_eq!(stats.already_satisfied_check_rejections, 0);

        // The counter must reach the DURABLE record, not just RunStats.
        let lines = read_transcript_lines(&transcript);
        let run_end = lines
            .iter()
            .rfind(|l| l["event"] == "run_end")
            .expect("a run_end line");
        assert_eq!(run_end["stats"]["answer_schema_rejections"], 2);
    }

    /// The transcript's `finish_answer` key discriminates the BRANCH, not
    /// merely validity — and is absent entirely on a non-finish call.
    #[tokio::test]
    async fn finish_answer_records_each_branch_on_the_transcript() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        let ctx = git_ctx(&root_path);
        // A NON-mutating first call: answer mode's inverted precondition
        // rejects a `finish(answer)` on a changed tree, so the accepted
        // branch this test needs is only reachable from a clean workspace.
        // `echo` gives the same thing the call was here for — a non-finish
        // `tool_result` to prove `finish_answer` is absent on one.
        let mut tools = registry_with_finish_and_edit();
        tools.register("echo", Arc::new(EchoTool));
        let transcript = root_path.join("branches.jsonl");

        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call("c-edit", "echo", serde_json::json!({ "i": 1 }))],
                StopReason::ToolUse,
            ),
            finish_call("c-miss", serde_json::json!({ "disposition": "answer" })),
            finish_call(
                "c-bad",
                serde_json::json!({ "disposition": "answer", "result": { "verdict": "nope" } }),
            ),
            finish_call(
                "c-ok",
                serde_json::json!({ "disposition": "answer", "result": { "verdict": "ok" } }),
            ),
        ]);
        let config = RunConfig::new("answer me", 5)
            .with_answer_schema(verdict_schema())
            .with_transcript(transcript.clone(), "branches");
        let RunResult { outcome, .. } = run(&backend, &tools, &ctx, &config).await;
        assert!(matches!(
            outcome,
            LoopOutcome::Finished(Disposition::Answer { .. })
        ));

        let lines = read_transcript_lines(&transcript);
        let by_call = |id: &str| -> serde_json::Value {
            lines
                .iter()
                .find(|l| l["event"] == "tool_result" && l["call_id"] == id)
                .unwrap_or_else(|| panic!("a tool_result for {id}"))
                .clone()
        };

        // A non-finish call carries NO `finish_answer` key at all.
        assert!(by_call("c-edit").get("finish_answer").is_none());

        let missing = by_call("c-miss");
        assert_eq!(missing["finish_answer"]["branch"], "missing_result");
        assert_eq!(
            missing["finish_answer"]["errors"],
            serde_json::json!([]),
            "the missing-result branch shows no schema errors"
        );
        assert_eq!(missing["finish_accepted"], serde_json::json!(false));

        let invalid = by_call("c-bad");
        assert_eq!(invalid["finish_answer"]["branch"], "invalid");
        assert!(
            !invalid["finish_answer"]["errors"]
                .as_array()
                .expect("an errors array")
                .is_empty()
        );
        assert_eq!(invalid["finish_accepted"], serde_json::json!(false));

        let valid = by_call("c-ok");
        assert_eq!(valid["finish_answer"]["branch"], "valid");
        assert_eq!(valid["finish_answer"]["errors"], serde_json::json!([]));
        assert_eq!(valid["finish_accepted"], serde_json::json!(true));
    }

    #[tokio::test]
    async fn finish_answer_is_null_for_a_done_claim() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        let ctx = git_ctx(&root_path);
        let tools = registry_with_finish_and_edit();
        let transcript = root_path.join("done.jsonl");

        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call(
                    "c-edit",
                    "edit_file",
                    serde_json::json!({
                        "path": "f.txt",
                        "old_string": "",
                        "new_string": "work\n",
                    }),
                )],
                StopReason::ToolUse,
            ),
            finish_call(
                "c-done",
                serde_json::json!({ "disposition": "done", "summary": "did it" }),
            ),
        ]);
        let config = RunConfig::new("work", 3)
            .with_answer_schema(verdict_schema())
            .with_transcript(transcript.clone(), "done");
        let _ = run(&backend, &tools, &ctx, &config).await;

        let lines = read_transcript_lines(&transcript);
        let finish = lines
            .iter()
            .find(|l| l["event"] == "tool_result" && l["call_id"] == "c-done")
            .expect("a finish tool_result");
        assert_eq!(finish["finish_answer"], serde_json::Value::Null);
    }

    /// The schema itself is recorded on `run_start` so a reviewer can
    /// re-verify any verdict off the record; build mode records `null`.
    #[tokio::test]
    async fn run_start_records_the_answer_schema_and_null_in_build_mode() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        let ctx = git_ctx(&root_path);
        let tools = registry_with_finish_and_edit();

        let schema_value = serde_json::json!({ "type": "object" });
        let answer_t = root_path.join("answer_start.jsonl");
        let backend = MockBackend::from_turns(vec![finish_call(
            "c-ok",
            serde_json::json!({ "disposition": "answer", "result": {} }),
        )]);
        let config = RunConfig::new("t", 1)
            .with_answer_schema(AnswerSchema::compile(&schema_value).expect("compiles"))
            .with_transcript(answer_t.clone(), "answer");
        let _ = run(&backend, &tools, &ctx, &config).await;
        let lines = read_transcript_lines(&answer_t);
        assert_eq!(lines[0]["config"]["answer_schema"], schema_value);

        let build_t = root_path.join("build_start.jsonl");
        let backend = MockBackend::from_turns(vec![finish_call(
            "c-done",
            serde_json::json!({ "disposition": "done", "summary": "s" }),
        )]);
        let config = RunConfig::new("t", 1).with_transcript(build_t.clone(), "build");
        let _ = run(&backend, &tools, &ctx, &config).await;
        let lines = read_transcript_lines(&build_t);
        assert_eq!(lines[0]["config"]["answer_schema"], serde_json::Value::Null);
    }

    /// Checks in answer mode RUN if configured but NEVER reject: answer
    /// mode's verifier is the schema, and a red gate describes a workspace
    /// the answering agent did not author.
    #[tokio::test]
    async fn a_red_gate_is_recorded_on_an_accepted_answer_not_rejected() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        let ctx = git_ctx(&root_path);
        let tools = registry_with_finish_and_edit();
        let runner = ChecksRunner::new(
            CheckCommand {
                program: "/bin/sh".to_string(),
                args: vec!["-c".to_string(), "exit 1".to_string()],
            },
            root_path.clone(),
            Duration::from_secs(10),
        );

        let backend = MockBackend::from_turns(vec![finish_call(
            "c-ok",
            serde_json::json!({ "disposition": "answer", "result": { "verdict": "ok" } }),
        )]);
        let config = RunConfig::new("answer me", 2)
            .with_answer_schema(verdict_schema())
            .with_checks(runner);
        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Answer { verification, .. }) => match verification {
                Verification::Checks(report) => assert!(
                    !report.passed,
                    "the RED report must be recorded as telemetry, not suppressed"
                ),
                other @ Verification::NoChecksConfigured => {
                    panic!("expected Verification::Checks; got {other:?}")
                }
            },
            other => panic!("a red gate must NOT reject an answer; got {other:?}"),
        }
        assert_eq!(stats.answer_schema_rejections, 0);
    }

    /// SUPERSEDED BY THE INVERTED PRECONDITION. When the library half
    /// (`bce155b`) landed, change evidence on an `answer` claim was RECORDED
    /// but not enforced, and this test pinned that — the enforcement was
    /// explicitly deferred to answer mode's CLI half. That half is now here:
    /// a `finish(answer)` on a mutated workspace is REJECTED. The test keeps
    /// its subject (an answer claim in a workspace the agent modified) and
    /// flips its expectation, which is the whole behavioural delta of this
    /// item; the rejection's wording and evidence are pinned separately by
    /// `answer_on_a_changed_tree_is_rejected_with_the_changed_paths`.
    #[tokio::test]
    async fn an_answer_in_a_mutated_workspace_is_rejected() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        let ctx = git_ctx(&root_path);
        let tools = registry_with_finish_and_edit();

        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call(
                    "c-edit",
                    "edit_file",
                    serde_json::json!({
                        "path": "scratch.txt",
                        "old_string": "",
                        "new_string": "incidental\n",
                    }),
                )],
                StopReason::ToolUse,
            ),
            finish_call(
                "c-ok",
                serde_json::json!({ "disposition": "answer", "result": { "verdict": "ok" } }),
            ),
        ]);
        let config = RunConfig::new("answer me", 2).with_answer_schema(verdict_schema());
        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;
        assert!(
            matches!(outcome, LoopOutcome::MaxIterations),
            "a mutated workspace must NOT terminate as Finished(Answer); got {outcome:?}"
        );
        assert_eq!(stats.modified_workspace_rejections, 1);
    }

    // ---- answer mode: the FinishTool schema gate -------------------------

    #[test]
    fn build_mode_finish_schema_disposition_enum_is_unchanged() {
        let schema = FinishTool::default().schema();
        assert_eq!(
            schema["input_schema"]["properties"]["disposition"]["enum"],
            serde_json::json!(["done", "blocked", "failed", "already_satisfied"]),
        );
    }

    #[test]
    fn build_mode_finish_schema_never_mentions_answer() {
        let rendered = serde_json::to_string(&FinishTool::default().schema())
            .expect("the finish schema serializes")
            .to_lowercase();
        assert!(
            !rendered.contains("answer"),
            "build mode must not advertise answer mode; got {rendered}"
        );
    }

    #[test]
    fn build_mode_system_prompt_never_mentions_answer() {
        let rendered = prompt::render_system_prompt(
            &prompt::tool_lines(&crate::tools::standard_registry(None)),
            None,
        )
        .to_lowercase();
        assert!(
            !rendered.contains("answer"),
            "the build-mode prompt must stay untouched by answer mode"
        );
    }

    #[test]
    fn answer_mode_finish_schema_advertises_answer_last_without_requiring_result() {
        let schema = FinishTool { answer_mode: true }.schema();
        assert_eq!(
            schema["input_schema"]["properties"]["disposition"]["enum"],
            serde_json::json!(["done", "blocked", "failed", "already_satisfied", "answer"]),
        );
        assert_eq!(
            schema["input_schema"]["required"],
            serde_json::json!(["disposition", "summary"]),
            "`result` is enforced by the harness's pinned rejection, not the schema"
        );
        assert!(!schema["input_schema"]["properties"]["result"].is_null());
    }

    // ---- answer mode: the runtime audit ---------------------------------

    #[test]
    fn emit_run_end_audits_an_answer_that_reached_the_terminal_with_no_schema() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("no_schema.jsonl");
        let mut writer = TranscriptWriter::open(Some(&TranscriptConfig {
            path: path.clone(),
            label: "audit".to_string(),
        }));
        let result: Result<LoopOutcome, StoreError> =
            Ok(LoopOutcome::Finished(Disposition::Answer {
                result: serde_json::json!({ "verdict": "ok" }),
                verification: Verification::NoChecksConfigured,
                change: ChangeEvidence::TreeUnchanged,
            }));
        emit_run_end(&mut writer, &result, &zero_stats(), Some("task:1"), None);
        drop(writer);

        let violation = read_transcript_lines(&path)
            .into_iter()
            .find(|l| l["event"] == "contract_violation")
            .expect("a contract_violation event");
        assert_eq!(violation["kind"], "answer_without_schema");
        assert_eq!(violation["run_id"], "task:1");
    }

    #[test]
    fn emit_run_end_audits_an_answer_whose_result_violates_the_schema() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("bad_result.jsonl");
        let mut writer = TranscriptWriter::open(Some(&TranscriptConfig {
            path: path.clone(),
            label: "audit".to_string(),
        }));
        let result: Result<LoopOutcome, StoreError> =
            Ok(LoopOutcome::Finished(Disposition::Answer {
                result: serde_json::json!({ "verdict": "nope" }),
                verification: Verification::NoChecksConfigured,
                change: ChangeEvidence::TreeUnchanged,
            }));
        let schema = verdict_schema();
        emit_run_end(
            &mut writer,
            &result,
            &zero_stats(),
            Some("task:1"),
            Some(&schema),
        );
        drop(writer);

        let violation = read_transcript_lines(&path)
            .into_iter()
            .find(|l| l["event"] == "contract_violation")
            .expect("a contract_violation event");
        assert_eq!(violation["kind"], "answer_result_fails_schema");
        assert!(
            !violation["errors"]
                .as_array()
                .expect("an errors array")
                .is_empty()
        );
    }

    #[test]
    fn emit_run_end_does_not_audit_an_honest_answer() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("honest.jsonl");
        let mut writer = TranscriptWriter::open(Some(&TranscriptConfig {
            path: path.clone(),
            label: "audit".to_string(),
        }));
        let result: Result<LoopOutcome, StoreError> =
            Ok(LoopOutcome::Finished(Disposition::Answer {
                result: serde_json::json!({ "verdict": "ok" }),
                verification: Verification::NoChecksConfigured,
                change: ChangeEvidence::TreeUnchanged,
            }));
        let schema = verdict_schema();
        emit_run_end(&mut writer, &result, &zero_stats(), None, Some(&schema));
        drop(writer);

        let lines = read_transcript_lines(&path);
        assert!(lines.iter().all(|l| l["event"] != "contract_violation"));
    }

    // =====================================================================
    // Answer mode — the INVERTED leg-3 precondition (item 2)
    // =====================================================================

    /// The fixture schema every answer-mode engine test in this item uses.
    fn answer_schema_fixture() -> AnswerSchema {
        AnswerSchema::compile(&serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false,
        }))
        .expect("the fixture schema compiles")
    }

    /// The fixture claim: schema-valid, so every test that bounces bounces on
    /// something OTHER than validation.
    fn valid_answer_claim() -> serde_json::Value {
        serde_json::json!({
            "disposition": "answer",
            "summary": "s",
            "result": { "answer": "42" },
        })
    }

    /// True when any fed-back tool result matched `pred`.
    fn any_tool_result(messages: &[Message], pred: impl Fn(&str, bool) -> bool) -> bool {
        messages.iter().any(|m| match m {
            Message::User { content } => content.iter().any(|b| match b {
                UserBlock::ToolResult {
                    content, is_error, ..
                } => pred(content, *is_error),
                UserBlock::Text(_) => false,
            }),
            Message::Assistant { .. } => false,
        })
    }

    /// AC-20. A `finish(answer)` whose `result` VALIDATES but whose working
    /// tree changed is REJECTED — answer mode's inverted precondition — and
    /// the loop continues to the iteration cap.
    ///
    /// Deliberately uses the EDIT-CAPABLE registry: registry composition is
    /// enforced at the CLI (`answer_registry` has no `edit_file`), and the
    /// precondition must hold regardless of which tool mutated the tree.
    #[tokio::test]
    async fn answer_on_a_changed_tree_is_rejected_with_the_changed_paths() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        let ctx = git_ctx(&root_path);
        let tools = registry_with_finish_and_edit();
        // The transcript lives OUTSIDE the workspace — writing it inside
        // would itself dirty the tree and confound the assertion.
        let out = TempDir::new().expect("tempdir");
        let transcript = out.path().join("t.jsonl");

        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call(
                    "c-edit",
                    "edit_file",
                    serde_json::json!({
                        "path": "scratch.txt",
                        "old_string": "",
                        "new_string": "oops\n",
                    }),
                )],
                StopReason::ToolUse,
            ),
            finish_call("c-answer", valid_answer_claim()),
            // A third draw so the rejection from the second is visible in the
            // messages the loop SENT — `last_messages` is the last request,
            // and a rejection on the final iteration is never sent.
            finish_call("c-answer-2", valid_answer_claim()),
        ]);
        let config = RunConfig::new("answer the question", 3)
            .with_answer_schema(answer_schema_fixture())
            .with_transcript(transcript.clone(), "answer");
        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert!(
            matches!(outcome, LoopOutcome::MaxIterations),
            "a modified tree must NOT terminate as Finished(Answer); got {outcome:?}"
        );
        assert_eq!(stats.modified_workspace_rejections, 2);
        assert_eq!(stats.no_change_rejections, 0);

        let fed_back = backend.last_messages();
        assert!(
            any_tool_result(&fed_back, |content, is_error| {
                is_error
                    && content.starts_with(
                        "finish(answer) rejected: the working tree changed since this run \
                         started — an answer run must not modify the workspace. Revert your \
                         edits (restore tracked files and delete files you created) and call \
                         finish(answer) again.\nchanged paths:",
                    )
                    && content.contains("scratch.txt")
            }),
            "the rejection must carry the pinned sentence and name the changed path; \
             got: {fed_back:?}"
        );

        let lines = read_transcript_lines(&transcript);
        let tool_result = lines
            .iter()
            .rfind(|l| l["event"] == "tool_result")
            .expect("a finish tool_result");
        assert_eq!(tool_result["finish_accepted"], false);
        assert_eq!(tool_result["finish_rejection"], "modified_workspace");
        assert_eq!(
            tool_result["finish_change"],
            serde_json::json!("TreeChanged")
        );
        assert!(
            tool_result["tree_current"]["Observed"].is_object(),
            "the rejection must record the observation it classified: {tool_result}"
        );
        let run_end = lines
            .iter()
            .rfind(|l| l["event"] == "run_end")
            .expect("a run_end event");
        assert_eq!(run_end["stats"]["modified_workspace_rejections"], 2);
    }

    /// AC-21. The mirror case: an untouched tree ACCEPTS the same claim.
    #[tokio::test]
    async fn answer_on_an_unchanged_tree_is_accepted_with_tree_unchanged_evidence() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        let ctx = git_ctx(&root_path);
        let tools = registry_with_finish_and_edit();

        let out = TempDir::new().expect("tempdir");
        let transcript = out.path().join("t.jsonl");

        let backend = MockBackend::from_turns(vec![finish_call("c-answer", valid_answer_claim())]);
        let config = RunConfig::new("answer the question", 2)
            .with_answer_schema(answer_schema_fixture())
            .with_transcript(transcript.clone(), "answer");
        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Answer {
                result,
                change,
                verification,
            }) => {
                assert_eq!(result, serde_json::json!({ "answer": "42" }));
                assert_eq!(change, ChangeEvidence::TreeUnchanged);
                assert_eq!(verification, Verification::NoChecksConfigured);
            }
            other => panic!("expected Finished(Answer{{TreeUnchanged}}); got {other:?}"),
        }
        assert_eq!(stats.modified_workspace_rejections, 0);
        assert_eq!(stats.edit_file_calls_ok, 0);

        // The fed-back result is the non-error ack. An accepted finish ENDS
        // the run, so that result is never part of a subsequent request —
        // the transcript is where it is observable.
        let lines = read_transcript_lines(&transcript);
        let tool_result = lines
            .iter()
            .rfind(|l| l["event"] == "tool_result")
            .expect("a finish tool_result");
        assert_eq!(tool_result["is_error"], false);
        assert_eq!(tool_result["content"], "finish acknowledged");
        assert_eq!(tool_result["finish_accepted"], true);
    }

    /// AC-22. Unobservable fails OPEN — accepted, with the unobservability
    /// RECORDED on both the disposition and the stats, so a caller can tell a
    /// verified read-only answer from an unverifiable one.
    #[tokio::test]
    async fn answer_in_a_non_git_workspace_fails_open_and_records_it() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        let probe = std::process::Command::new("git")
            .args(["rev-parse", "--is-inside-work-tree"])
            .current_dir(&root_path)
            .output()
            .expect("git runs");
        assert!(
            !probe.status.success(),
            "the fail-open assumption requires a non-git workspace root"
        );

        let workspace = Workspace::new(&root_path, None).expect("workspace");
        let ctx = ToolCtx::new(Arc::new(workspace), Arc::new(crate::tool::StubOffloadSink));
        let tools = registry_with_finish_and_edit();
        let backend = MockBackend::from_turns(vec![finish_call("c-answer", valid_answer_claim())]);
        let config =
            RunConfig::new("answer the question", 2).with_answer_schema(answer_schema_fixture());
        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Answer { change, .. }) => {
                assert!(
                    matches!(change, ChangeEvidence::Unobservable { .. }),
                    "an unobservable workspace must fail OPEN; got {change:?}"
                );
            }
            other => panic!("expected Finished(Answer{{Unobservable}}); got {other:?}"),
        }
        assert!(stats.tree_baseline_unobservable);
        assert_eq!(stats.modified_workspace_rejections, 0);
    }

    /// AC-22. The tripwire names the invariant that actually went inert —
    /// the two modes enforce OPPOSITE things off the same observation.
    #[test]
    fn inert_precondition_warning_names_the_mode_s_invariant() {
        assert_eq!(
            inert_precondition_warning(false),
            "the done-requires-change precondition is INERT for this run",
            "build mode's wording is load-bearing and must not drift"
        );
        assert!(
            inert_precondition_warning(true)
                .contains("the answer-requires-unchanged-tree precondition is INERT"),
            "answer mode's tripwire must name the INVERTED precondition; got {}",
            inert_precondition_warning(true)
        );
    }

    /// AC-24. The inverted rule is scoped to answer mode ONLY: a build-mode
    /// run that edits a file and claims `done` is still accepted.
    #[tokio::test]
    async fn build_mode_still_accepts_a_changed_tree_and_bumps_no_answer_counter() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        let ctx = git_ctx(&root_path);
        let tools = registry_with_finish_and_edit();

        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call(
                    "c-edit",
                    "edit_file",
                    serde_json::json!({
                        "path": "work.txt",
                        "old_string": "",
                        "new_string": "done\n",
                    }),
                )],
                StopReason::ToolUse,
            ),
            finish_call(
                "c-done",
                serde_json::json!({ "disposition": "done", "summary": "edited" }),
            ),
        ]);
        // No `with_answer_schema` — `config.answer_schema` is None, so this
        // is a build-mode run.
        let config = RunConfig::new("do the work", 5);
        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Done { change, .. }) => {
                assert_eq!(change, ChangeEvidence::TreeChanged);
            }
            other => panic!("expected Finished(Done{{TreeChanged}}); got {other:?}"),
        }
        assert_eq!(stats.modified_workspace_rejections, 0);
    }

    /// AC-25. `done` and `already_satisfied` are wrong-mode claims on an
    /// answer run: rejected with steering, before any checks run and before
    /// the tree is observed, bumping no counter.
    #[tokio::test]
    async fn answer_mode_rejects_the_build_mode_terminals() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        let ctx = git_ctx(&root_path);
        let tools = registry_with_finish_and_edit();
        let out = TempDir::new().expect("tempdir");
        let transcript = out.path().join("t.jsonl");

        let backend = MockBackend::from_turns(vec![
            finish_call(
                "c-done",
                serde_json::json!({ "disposition": "done", "summary": "ok" }),
            ),
            finish_call(
                "c-as",
                serde_json::json!({
                    "disposition": "already_satisfied",
                    "reason": "nothing to do",
                }),
            ),
            // A third draw so BOTH rejections appear in the messages the loop
            // sent — a rejection on the final iteration is never sent.
            finish_call(
                "c-done-2",
                serde_json::json!({ "disposition": "done", "summary": "ok" }),
            ),
        ]);
        let config = RunConfig::new("answer the question", 3)
            .with_answer_schema(answer_schema_fixture())
            .with_transcript(transcript.clone(), "answer");
        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert!(
            matches!(outcome, LoopOutcome::MaxIterations),
            "a build-mode terminal must not end an answer run; got {outcome:?}"
        );
        assert_eq!(
            stats.invalid_finish_calls, 0,
            "a wrong-mode claim is well-formed, not malformed"
        );
        assert_eq!(stats.modified_workspace_rejections, 0);
        assert_eq!(stats.no_change_rejections, 0);
        assert_eq!(stats.already_satisfied_check_rejections, 0);

        let fed_back = backend.last_messages();
        for claim in ["done", "already_satisfied"] {
            let expected = format!(
                "finish({claim}) rejected: this run is in answer mode — the only accepted \
                 terminal dispositions are answer, blocked, failed. End it with disposition \
                 `answer` and a `result` matching the schema in the task."
            );
            assert!(
                any_tool_result(&fed_back, |content, is_error| is_error
                    && content == expected),
                "the {claim} rejection must be the pinned wrong-mode text; got: {fed_back:?}"
            );
        }
        assert!(
            any_tool_result(&fed_back, |content, _| content.contains("answer mode")),
            "the rejection must name the mode; got: {fed_back:?}"
        );

        let lines = read_transcript_lines(&transcript);
        let finishes: Vec<_> = lines
            .iter()
            .filter(|l| l["event"] == "tool_result")
            .collect();
        assert_eq!(finishes.len(), 3, "every finish call is recorded");
        for line in finishes {
            assert_eq!(line["finish_accepted"], false);
            assert_eq!(line["finish_rejection"], "wrong_mode_disposition");
            assert!(
                line["finish_verification"].is_null(),
                "no checks ran for a wrong-mode claim: {line}"
            );
            assert!(
                line["finish_change"].is_null(),
                "the tree is not observed for a wrong-mode claim: {line}"
            );
        }
    }

    /// Ordering guard: schema validation runs BEFORE the tree precondition,
    /// so an invalid `result` on a CHANGED tree gets the schema rejection —
    /// the more actionable of the two — and never pays for a tree
    /// observation.
    #[tokio::test]
    async fn an_invalid_result_on_a_changed_tree_gets_the_schema_rejection() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        let ctx = git_ctx(&root_path);
        let tools = registry_with_finish_and_edit();

        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call(
                    "c-edit",
                    "edit_file",
                    serde_json::json!({
                        "path": "scratch.txt",
                        "old_string": "",
                        "new_string": "oops\n",
                    }),
                )],
                StopReason::ToolUse,
            ),
            finish_call(
                "c-answer",
                serde_json::json!({
                    "disposition": "answer",
                    "summary": "s",
                    "result": { "answer": 42 },
                }),
            ),
            finish_call(
                "c-answer-2",
                serde_json::json!({
                    "disposition": "answer",
                    "summary": "s",
                    "result": { "answer": 42 },
                }),
            ),
        ]);
        let config =
            RunConfig::new("answer the question", 3).with_answer_schema(answer_schema_fixture());
        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert!(matches!(outcome, LoopOutcome::MaxIterations));
        assert_eq!(stats.answer_schema_rejections, 2);
        assert_eq!(
            stats.modified_workspace_rejections, 0,
            "the schema verdict wins; the tree is never observed"
        );
        let fed_back = backend.last_messages();
        assert!(
            any_tool_result(&fed_back, |content, is_error| {
                is_error
                    && content.starts_with(
                        "finish(answer) rejected: result does not conform to the configured \
                         schema:",
                    )
            }),
            "expected the schema rejection; got: {fed_back:?}"
        );
    }

    /// AC-28(b). Every `FinishRejection` variant has a pinned wire label —
    /// the value jq audit queries grep for.
    #[test]
    fn finish_rejection_wire_labels_are_pinned() {
        assert_eq!(FinishRejection::NoChange.as_str(), "no_change");
        assert_eq!(
            FinishRejection::AlreadySatisfiedChecks.as_str(),
            "already_satisfied_checks"
        );
        assert_eq!(FinishRejection::AnswerSchema.as_str(), "schema_invalid");
        assert_eq!(
            FinishRejection::ModifiedWorkspace.as_str(),
            "modified_workspace"
        );
        assert_eq!(
            FinishRejection::WrongModeDisposition.as_str(),
            "wrong_mode_disposition"
        );
    }

    /// An accepted claim carries `finish_rejection: null` — the field is
    /// present and explicitly empty, not absent, so an audit query can treat
    /// it as a closed discriminator.
    #[tokio::test]
    async fn an_accepted_finish_records_a_null_finish_rejection() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        let ctx = git_ctx(&root_path);
        let tools = registry_with_finish_and_edit();
        let out = TempDir::new().expect("tempdir");
        let transcript = out.path().join("t.jsonl");

        let backend = MockBackend::from_turns(vec![finish_call("c-answer", valid_answer_claim())]);
        let config = RunConfig::new("answer the question", 2)
            .with_answer_schema(answer_schema_fixture())
            .with_transcript(transcript.clone(), "answer");
        let _ = run(&backend, &tools, &ctx, &config).await;

        let lines = read_transcript_lines(&transcript);
        let tool_result = lines
            .iter()
            .rfind(|l| l["event"] == "tool_result")
            .expect("a finish tool_result");
        assert_eq!(tool_result["finish_accepted"], true);
        assert!(
            tool_result.get("finish_rejection").is_some()
                && tool_result["finish_rejection"].is_null(),
            "an accepted claim carries an explicit null: {tool_result}"
        );
    }

    // =====================================================================
    // AC-19 / AC-28(a): the engine selects the mode's system prompt, and
    // `run_start.config` records which mode it is in.
    // =====================================================================

    /// AC-19 + AC-28(a), answer mode: the system string equals
    /// `render_answer_system_prompt`, and `run_start.config` says so.
    #[tokio::test]
    async fn answer_mode_run_renders_the_answer_system_prompt() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        let ctx = git_ctx(&root_path);
        let tools = registry_with_finish_and_edit();
        let out = TempDir::new().expect("tempdir");
        let transcript = out.path().join("t.jsonl");

        let backend = MockBackend::from_turns(vec![finish_call("c-answer", valid_answer_claim())]);
        let schema = answer_schema_fixture();
        let schema_source = schema.source().clone();
        let config = RunConfig::new("answer the question", 2)
            .with_answer_schema(schema)
            .with_transcript(transcript.clone(), "answer");
        let _ = run(&backend, &tools, &ctx, &config).await;

        let expected = prompt::render_answer_system_prompt(&prompt::tool_lines(&tools));
        let systems = backend.systems_seen();
        assert!(!systems.is_empty(), "at least one turn drawn");
        for (i, entry) in systems.iter().enumerate() {
            assert_eq!(
                entry.as_deref(),
                Some(expected.as_str()),
                "turn {i} must send the ANSWER system prompt"
            );
        }
        assert_ne!(
            expected,
            prompt::render_system_prompt(&prompt::tool_lines(&tools), None),
            "the answer system prompt must not be the build one"
        );

        let lines = read_transcript_lines(&transcript);
        assert_eq!(lines[0]["event"], "run_start");
        assert_eq!(lines[0]["config"]["mode"], "answer");
        assert_eq!(lines[0]["config"]["answer_schema"], schema_source);
        assert_eq!(lines[0]["system"], expected);
    }

    /// AC-19 + AC-28(a), build mode: unchanged — the system string equals
    /// `render_system_prompt`, `mode` is `"build"`, `answer_schema` is null.
    #[tokio::test]
    async fn build_mode_run_still_renders_the_build_system_prompt() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize");
        let ctx = git_ctx(&root_path);
        let tools = registry_with_finish_and_edit();
        let out = TempDir::new().expect("tempdir");
        let transcript = out.path().join("t.jsonl");

        let backend = MockBackend::from_turns(vec![finish_call(
            "c-done",
            serde_json::json!({ "disposition": "done", "summary": "ok" }),
        )]);
        let config = RunConfig::new("do the work", 1).with_transcript(transcript.clone(), "build");
        let _ = run(&backend, &tools, &ctx, &config).await;

        let expected = prompt::render_system_prompt(&prompt::tool_lines(&tools), None);
        let systems = backend.systems_seen();
        assert!(!systems.is_empty(), "at least one turn drawn");
        for (i, entry) in systems.iter().enumerate() {
            assert_eq!(
                entry.as_deref(),
                Some(expected.as_str()),
                "turn {i} must send the BUILD system prompt"
            );
        }

        let lines = read_transcript_lines(&transcript);
        assert_eq!(lines[0]["event"], "run_start");
        assert_eq!(lines[0]["config"]["mode"], "build");
        assert!(lines[0]["config"]["answer_schema"].is_null());
        assert_eq!(lines[0]["system"], expected);
    }
}
