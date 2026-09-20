//! `talos run` — accept a [`TaskSpec`] JSON (stdin or `--file`), execute
//! it with full persistence, and exit with a disposition-mapped code.
//!
//! `talos run --mode answer --schema <path>` — the READ-ONLY variant: the
//! input is a free-text question instead of a `TaskSpec`, the tool registry
//! omits `edit_file`, no gate and no nudges are wired, and the run terminates
//! with a `finish(answer)` whose `result` validated against `--schema` AND
//! whose working tree is UNCHANGED since the run started. See the `--mode`
//! answer section below.
//!
//! ## Exit code contract (locked)
//!
//! | Code | Meaning |
//! |------|---------|
//! | 0    | Task verified Done |
//! | 10   | Task Blocked |
//! | 20   | Task Failed, `StoppedWithoutFinish`, `MaxIterations`, or `BudgetExhausted` — including `FailureMode::Truncated` (a `max_tokens` cutoff) and `FailureMode::AnswerSchemaExhausted` (consecutive identical answer-schema rejections), which ride 20 under `Finished(Failed{..})` |
//! | 30   | Task was already satisfied — gates green, nothing changed; NOT pushable |
//! | 40   | Task produced a schema-validated Answer; NOT pushable |
//! | 1    | Harness/infra error (bad spec, `BackendError`, store error, clap error) |
//!
//! Why 40 and not 0/20/30 (the dispatch-worker contract, re-verified against
//! `agent-gtd-dispatch` at the cited lines): `talos.py::map_talos_result`
//! (talos.py:285-306) sets `push=True` only on exit 0 with parseable stdout,
//! and a validated answer has NOTHING to push — its deliverable is the JSON
//! payload on stdout — so it must not be 0. It is not an already-satisfied
//! build run either, so it must not be 30; keeping 40 distinct from 30 lets a
//! future mapper arm tell answer-with-payload from already-satisfied. Today
//! BOTH 30 and 40 fall to that mapper's unknown-exit-code catch-all
//! (talos.py:342-346), which fails safe (status `failed`, `push=False`) —
//! the right landing spot until the worker grows an arm for it.
//!
//! The code is read from [`harness::engine::LoopOutcome`], **not** from the
//! disposition — because `BackendError`'s `into_disposition` also yields
//! `Failed`, which would collapse engine-broke (must be 1) into task-Failed
//! (20). See [`exit_code`] for the full rationale.
//!
//! `talos run --transcript [path]` opts into a full JSONL run transcript
//! (see [`harness::transcript`]) — every model request/turn, tool call, and
//! tool result — default OFF, flag only (no env fallback); a bare
//! `--transcript` defaults to `transcript.jsonl` in the run's state dir.
//! `talos ralph` has no transcript support.
//!
//! ## `talos run --mode answer` flags
//!
//! - `--mode <build|answer>` (default `build`) — `build` is today's path,
//!   byte-for-byte. `answer` REQUIRES `--schema`; `build` REJECTS it. Both
//!   shape errors are checked BEFORE stdin is read, so a wrongly flagged
//!   invocation fails immediately instead of blocking on a pipe.
//! - `--schema <PathBuf>` (required with `--mode answer`, forbidden without
//!   it) — a JSON Schema file the answer's `result` must conform to. Used
//!   verbatim as supplied: resolved against the process CWD, NOT canonicalized
//!   and NOT confined to `--workspace`, exactly like `--file`. Its RAW bytes
//!   are shown to the model (key order and formatting survive) and separately
//!   compiled into the validator the harness enforces.
//! - `--file` / stdin — in answer mode this is the free-text QUESTION, never
//!   a `TaskSpec`; it is never handed to `serde_json`. A whitespace-only
//!   question is rejected before the store is opened.
//! - Everything else applies identically (`--task-id`, `--attempt`,
//!   `--max-iterations`, `--wall-clock-secs`, `--transcript`,
//!   `--state-retention-days`, `--offload-dir`, `--run-store`).
//!   `--gate-timeout-secs` is accepted and INERT — answer mode wires no gate
//!   this cut.
//!
//! CONCURRENCY: several answer agents on one host MUST each pass a distinct
//! `--task-id` (or distinct `--run-store` + `--offload-dir`) — the default
//! `talos-run` state dir and run id (`talos-run:1`) collide.
//!
//! ## Auditing an answer run's transcript
//!
//! Both queries must return ZERO rows on an answer run; any hit means the
//! read-only guard regressed:
//!
//! ```text
//! jq -c 'select(.event=="tool_result" and .finish_accepted==true and .finish_change=="TreeChanged")' t.jsonl
//! jq -c 'select(.event=="run_end" and .stats.edit_file_calls_ok>0)' t.jsonl
//! ```
//!
//! The first would mean the harness accepted an answer from a workspace the
//! agent modified; the second, that an `edit_file` succeeded in a registry
//! that does not register it. Note that `tree_dirty` and `mutating_iters` are
//! NOT the read-only oracle — in answer mode they count successful `bash`
//! calls, not observed tree changes. The authoritative evidence is
//! `Disposition::Answer.change`.
//!
//! `talos ralph` — a thin CLI over [`harness::ralph::run_ralph`]: drive the
//! Ralph outer loop toward a plain-objective `--stop-when` command oracle
//! with a fresh inner context per outer iteration. `ralph` is NOT run-record
//! persisted this cut — [`harness::ralph::run_ralph`] is invoked directly and
//! its [`harness::ralph::RalphReport`] is summarized to stdout; no store is
//! opened and no run id / record path is produced.
//!
//! ## `talos ralph` exit code contract
//!
//! | Code | Meaning |
//! |------|---------|
//! | 0    | `StopConditionMet` — objective met |
//! | 20   | `Stuck`, `MaxIterationsExhausted`, `TimeBudgetExhausted`, or `DoOversExhausted` — task-side failure terminals |
//! | 1    | `Error`, `BackendErrorsExhausted`, or a harness/infra error (clap error, workspace error) — sustained backend failure is an infra condition, NOT a task failure, so it must never collapse into code 20 |
//!
//! There is NO `10`/Blocked analog for ralph — the Ralph outer loop has no
//! Blocked terminal. See [`ralph_exit_code`] for the rationale.
//!
//! When the loop terminates with `Error`, the machine-readable summary on
//! stdout stays payload-free (`{"terminal":"Error"}`); the human-readable
//! detail — the failing git/spawn/revert command — is written to **stderr**
//! by [`write_ralph_error_detail`] (one `talos ralph: error: <msg>` line).
//! Non-Error terminals write nothing to stderr.
//!
//! ## `talos ralph` flags
//!
//! - `--workspace <PathBuf>` (required) — workspace root, MUST already be a
//!   git work tree ([`harness::ralph::run_ralph`] does NOT run `git init`; the
//!   harness owns per-iteration git commits and assumes an existing repo).
//! - `--objective <String>` (required) — the high-level objective.
//! - `--stop-when <String>` (required) — the outer stop-command oracle, run
//!   via `/bin/sh -c`; exit `0` = objective met. DISTINCT from `--gate`. A
//!   whitespace-only value is rejected with a JSON error before any
//!   filesystem/backend work (an empty oracle exits `0` every call and
//!   declares the objective met on iteration 1 — a false-done vector).
//! - `--gate <String>` (optional, default empty) — the inner per-iteration
//!   gate built via the existing [`build_checks_runner`]; whitespace-only =
//!   no inner gate.
//! - `--notes-file <String>` (default `PROGRESS.md`).
//! - `--max-ralph-iterations <u32>` (default `100`) — outer cap.
//! - `--inner-max-iterations <u32>` (default `500`) — mirrors `run`'s
//!   `max_iterations`.
//! - `--stuck-k <u32>` (default `3`) — matches
//!   [`harness::ralph::DEFAULT_STUCK_K`].
//! - `--max-do-overs <u32>` (default `3`) — consecutive do-over cap; matches
//!   [`harness::ralph::DEFAULT_MAX_DO_OVERS`]. A non-green inner outcome that
//!   left the tree dirty, or a green `Finished(Done)` whose per-iteration
//!   `git commit` exited non-zero (e.g. a rejecting pre-commit hook), is
//!   reverted to the last green commit and retried as a do-over; after this
//!   many CONSECUTIVE do-overs (a green commit resets the count) the loop
//!   terminates with [`harness::ralph::RalphTerminal::DoOversExhausted`]
//!   (exit 20). FIRST, a hook-rejected commit gets a bounded
//!   re-stage-and-retry ([`harness::ralph::run_ralph`] re-runs `git add -A`
//!   and the identical commit, at most 2 retries) — but ONLY when the hook
//!   MUTATED the work tree after the initial `git add -A`, so a pure checker
//!   hook still goes straight to the revert + do-over path.
//! - `--max-backend-errors <u32>` (default `5`) — consecutive-backend-error
//!   cap; matches [`harness::ralph::DEFAULT_MAX_BACKEND_ERRORS`]. This many
//!   CONSECUTIVE outer iterations whose inner outcome was
//!   `LoopOutcome::BackendError` terminate with
//!   [`harness::ralph::RalphTerminal::BackendErrorsExhausted`] (exit 1 —
//!   infra, NOT 20); any non-`BackendError` iteration resets the count.
//!   Independent of the do-over cap: a `BackendError` iteration stays exempt
//!   from do-overs, so without this breaker a SYSTEMATIC backend failure
//!   would churn the loop forever.
//! - `--stop-when-timeout-secs <u64>` (default `300`) — matches
//!   [`harness::ralph::DEFAULT_STOP_COMMAND_TIMEOUT`] of 5 min.
//! - `--gate-timeout-secs <u64>` (default `300`).
//! - `--ralph-wall-clock-secs <u64>` (optional; flag > env
//!   `TALOS_RALPH_WALL_CLOCK_SECS` > `0` = unbounded, resolved manually like
//!   the existing `wall_clock_secs` since the clap `env` feature is not
//!   enabled).
//! - `--offload-dir <PathBuf>` (optional; default
//!   `talos_state_dir("talos-ralph").join("offload")`).
//!
//! ## Environment variables
//!
//! - `TALOS_BEDROCK` — when set to a non-empty (after `.trim()`) value,
//!   selects the AWS Bedrock backend (Converse API) AHEAD of `TALOS_BACKEND`
//!   / `ANTHROPIC_*` / `OLLAMA_*`; unset/empty/whitespace-only falls through.
//!   Credentials AND region resolve via the standard AWS chain (env/profile
//!   /SSO/IMDS — no keys in source). Only `claude-haiku-4-5` /
//!   `claude-sonnet-5` / `claude-opus-4-8` (via `ANTHROPIC_MODEL`) are mapped.
//! - `TALOS_BACKEND` — `anthropic` (default when unset) | `ollama`
//! - `ANTHROPIC_API_KEY` — required for anthropic
//! - `ANTHROPIC_MODEL` — optional; default `claude-haiku-4-5`
//! - `OLLAMA_MODEL` — required for ollama
//! - `OLLAMA_BASE_URL` — optional; default `http://localhost:11434`
//! - `OLLAMA_API_KEY` — optional bearer token
//! - `OLLAMA_NUM_CTX` — optional. A non-empty `u32` is used verbatim (no
//!   probe); empty/whitespace is treated as unset. Unset with a
//!   `localhost`/`127.0.0.1` base URL probes `POST /api/show` and pins the
//!   model's advertised context length (a probe failure exits 1 — no
//!   fallback); unset with a non-local base URL leaves `num_ctx` unset
//!   (Ollama's own default).
//! - `OLLAMA_THINK` — `off|on|low|medium|high|max`
//! - `TALOS_WALL_CLOCK_SECS` — optional `u64` seconds; default `1500`
//!   seconds; `0` disables (unbounded). Precedence: the `--wall-clock-secs`
//!   flag, then the `TALOS_WALL_CLOCK_SECS` env, then the compiled default
//!   of `1500`. The
//!   harness self-terminates gracefully with recovery facts before the
//!   worker's hard kill when this budget is reached. The compiled default is
//!   a conservative floor that guarantees a clean `BudgetExhausted` terminal
//!   under the worst-case worker timeout (derived from the dispatch
//!   backstop's 1800 s); only the dispatch worker knows its own effective
//!   timeout for a given run, so a caller that knows its real timeout should
//!   pass a value derived from it (its effective timeout minus a slack
//!   margin) — the env var is the INTENDED PRODUCTION PATH, not an escape
//!   hatch. A non-numeric env value falls through silently to the default
//!   (an armed 1500 s, not unbounded). Note: the trade — a run that would
//!   legitimately finish after the budget self-terminates cleanly at exit 20
//!   instead of being hard-killed — is deliberate.
//! - `TALOS_STATE_RETENTION_DAYS` — optional `u64` days of age-based
//!   retention for talos's own XDG state dir (`run.sqlite`, `offload/`, and
//!   opt-in transcripts under `${XDG_STATE_HOME:-$HOME/.local/state}/talos/`).
//!   Precedence: `--state-retention-days` flag > `TALOS_STATE_RETENTION_DAYS`
//!   env > the compiled default of `30`. `0` disables pruning entirely. This
//!   is a `talos run` flag only — `talos ralph` does not prune (see
//!   [`RalphArgs`]). Like `--transcript`, the env fallback does NOT survive
//!   dispatch's sudo boundary: `templates/sudoers-dispatch-svc.tmpl`'s
//!   `env_keep` carries exactly one `TALOS_*` var (`TALOS_BACKEND`), so under
//!   `env_reset` this variable never reaches the process and the fleet always
//!   uses the compiled 30-day default until the worker passes the flag
//!   (kb-02979 shape). Each `talos run` overwrites
//!   `<state-root>/talos/prune-last.json` with a retention report (last
//!   writer wins under concurrency; the file's own mtime is the timestamp).
//! - `TALOS_COMPACT_THRESHOLD_PCT` — optional `u64` percent: the in-run
//!   compaction trigger threshold, the percent of the backend's advertised
//!   context window at which the previous turn's raw prompt triggers
//!   compaction (see [`harness::engine::should_compact`]). Precedence:
//!   `--compact-threshold-pct` flag > `TALOS_COMPACT_THRESHOLD_PCT` env >
//!   the compiled default of 90 ([`harness::engine::COMPACT_THRESHOLD_PCT`]).
//!   `0` DISABLES compaction entirely — no walk, no event, no counter.
//!   Values above 100 are accepted and simply never reachable. An empty or
//!   whitespace-only value is treated as unset; a NON-NUMERIC value is a
//!   hard construction error (JSON error + exit 1) — never a silent
//!   fallback, because a typo'd threshold would silently arm or disarm the
//!   very arm an A/B experiment is measuring. The resolved value is
//!   recorded on the transcript's `run_start.config.compact_threshold_pct`.
//!   Same sudo-boundary caveat as `TALOS_STATE_RETENTION_DAYS` (kb-02979
//!   shape): under `env_reset` the fleet runs the compiled default until
//!   the worker passes the flag.

use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use harness::anthropic::AnthropicBackend;
use harness::bedrock::BedrockBackend;
use harness::engine::{AnswerSchema, LoopOutcome, Persistence, RunConfig, run_id, run_persisted};
use harness::exec::{CheckCommand, ChecksRunner, shell_checks_runner};
use harness::model::{
    AssistantTurn, BackendError, MaxTokensSource, ModelBackend, OutputCapResolution, TurnRequest,
};
use harness::ollama::{OllamaBackend, ThinkLevel};
use harness::prompt::{render_answer_prompt, render_task_prompt_from_spec};
use harness::ralph::{
    DEFAULT_MAX_BACKEND_ERRORS, DEFAULT_MAX_DO_OVERS, DEFAULT_STUCK_K, RalphConfig, RalphReport,
    RalphTerminal, run_ralph,
};
use harness::run_record::{BackendKind, BackendSettings, Disposition};
use harness::store::{RunStore, SqliteRunStore};
use harness::task_spec::TaskSpec;
use harness::tool::{OffloadSink, ToolCtx};
use harness::tools::{answer_registry, standard_registry};
use harness::workspace::{DiskOffloadSink, Workspace};
use serde::Serialize;

/// Default age-based retention, in days, for talos's own XDG state dir.
/// Precedence: `--state-retention-days` flag > `TALOS_STATE_RETENTION_DAYS`
/// env > this default. See [`resolve_state_retention_days`].
const DEFAULT_STATE_RETENTION_DAYS: u64 = 30;

/// Default wall-clock budget, in seconds, for `talos run`. Precedence:
/// `--wall-clock-secs` flag > `TALOS_WALL_CLOCK_SECS` env > this default;
/// `0` disables (unbounded). See [`resolve_wall_clock_secs`].
///
/// The value is derived from the dispatch worker's 1800 s BACKSTOP timeout
/// (`agent-gtd-dispatch` `config.py` `TIMEOUT_SECONDS = 30 * 60`), leaving 5
/// minutes of slack under the smallest timeout any consumer of this binary
/// can have. That 1800 s figure is a backstop only — a consumer whose real
/// effective timeout is larger (e.g. the harness-design GTD project's
/// `dispatch_timeout_minutes = 60`, i.e. 3600 s) is expected to RAISE the
/// budget for its runs via the `TALOS_WALL_CLOCK_SECS` env var (its
/// effective timeout minus a slack margin) rather than by editing this
/// constant: a constant compiled into the binary applies to every consumer,
/// including those that never set a dispatch-side timeout. The compiled
/// default is therefore a conservative floor that guarantees a clean
/// `BudgetExhausted` terminal — recovery facts written, run resumable,
/// exit 20 — under the worst-case worker timeout; a hard kill from outside
/// loses the work entirely, which is strictly worse.
const DEFAULT_WALL_CLOCK_SECS: u64 = 1500;

/// Seconds per day, used to convert a retention day-count into a
/// [`Duration`] for [`prune_state_root`].
const SECS_PER_DAY: u64 = 86_400;

/// Immediate children of the talos state root whose own children — not the
/// aggregate directory itself — are subject to age-based pruning. See the
/// doc comment on [`prune_state_root`] for the rationale.
const AGGREGATE_DIRS: [&str; 2] = ["coding-eval", "mined-eval"];

/// Cap on how many removed directory names [`PruneReport::removed_names`]
/// records, to keep `prune-last.json` bounded.
const MAX_REPORT_NAMES: usize = 8;

/// Filename of the per-host retention report written under the talos state
/// root on every `talos run`.
const PRUNE_REPORT_FILENAME: &str = "prune-last.json";

// ============================================================================
// CLI types
// ============================================================================

/// Top-level CLI entry point.
#[derive(clap::Parser)]
#[command(name = "talos", about = "Talos agent runner", version = env!("TALOS_VERSION"))]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Talos subcommands.
#[derive(clap::Subcommand)]
enum Command {
    /// Execute a `TaskSpec` JSON and report outcome + exit code.
    Run(RunArgs),
    /// Run the Ralph outer loop over [`harness::ralph::run_ralph`] — a thin
    /// CLI that drives the inner engine toward an objective via a
    /// `--stop-when` command oracle. Not run-record persisted this cut.
    Ralph(RalphArgs),
}

/// What `talos run` is being asked to do.
///
/// The engine has no `RunMode` — answer mode there IS
/// `RunConfig::answer_schema.is_some()`. This flag's only jobs are CLI-side:
/// REQUIRE `--schema` (and forbid it in build mode), select the read-only
/// `answer_registry` over `standard_registry`, select `render_answer_prompt`
/// over the task-spec renderer, and — through the schema it forces you to
/// supply — turn on the engine's inverted tree precondition.
/// The two variants deliberately carry `//` comments rather than `///` doc
/// comments: clap turns a variant doc comment into per-variant long help,
/// which replaces the compact `[possible values: build, answer]` line that
/// `crates/talos/tests/cli.rs` pins. The prose lives here instead.
///
/// - `build` — execute a groomed [`TaskSpec`] and change the workspace. The
///   default, and what every existing caller gets.
/// - `answer` — answer a free-text question about the workspace WITHOUT
///   changing it. Requires `--schema`; the deliverable is the validated JSON
///   payload on stdout, and the exit code is 40.
#[derive(clap::ValueEnum, Clone, Copy, PartialEq, Eq, Debug)]
enum RunMode {
    // Execute a groomed `TaskSpec` and change the workspace.
    Build,
    // Answer a question about the workspace without changing it.
    Answer,
}

/// Arguments for `talos run`.
#[derive(clap::Args)]
struct RunArgs {
    /// Workspace root (confined working directory for the agent).
    #[arg(long)]
    workspace: PathBuf,

    /// Path to the run input — a `TaskSpec` JSON in build mode, the free-text
    /// prompt in answer mode (reads stdin when omitted).
    #[arg(long)]
    file: Option<PathBuf>,

    /// What this run is: `build` (execute a `TaskSpec`, the default) or
    /// `answer` (answer a question about the workspace without changing it).
    ///
    /// `answer` REQUIRES `--schema`; `build` REJECTS it. Both shape errors are
    /// checked before stdin is read, so a wrongly flagged invocation fails
    /// immediately instead of blocking on a pipe.
    #[arg(long, value_enum, default_value_t = RunMode::Build)]
    mode: RunMode,

    /// Path to a JSON Schema file the answer's `result` must conform to.
    /// Valid ONLY with `--mode answer`, where it is required.
    ///
    /// Used verbatim as supplied — resolved against the process CWD, NOT
    /// canonicalized and NOT confined to `--workspace`, exactly like
    /// `--file`. The raw file bytes are shown to the model (so key order and
    /// formatting survive) and separately compiled into the validator the
    /// harness checks `finish(answer)` against.
    #[arg(long)]
    schema: Option<PathBuf>,

    /// `SQLite` store path for the run record.
    /// Defaults to `${XDG_STATE_HOME:-~/.local/state}/talos/<task-id>/run.sqlite`.
    #[arg(long)]
    run_store: Option<PathBuf>,

    /// Offload directory for oversized tool output.
    /// Defaults to `${XDG_STATE_HOME:-~/.local/state}/talos/<task-id>/offload`.
    #[arg(long)]
    offload_dir: Option<PathBuf>,

    /// Task identifier — becomes `Persistence.task_id` and seeds the run id.
    /// `TaskSpec` has no `task_id` field; this must come from the CLI.
    ///
    /// CONCURRENCY (answer mode): several answer agents sharing one checkout
    /// MUST each pass a DISTINCT `--task-id` (or distinct `--run-store` +
    /// `--offload-dir`). The default `talos-run` resolves to one state dir and
    /// one run id (`talos-run:1`), so concurrent runs would collide on the
    /// same `run.sqlite` and the same run record.
    #[arg(long, default_value = "talos-run")]
    task_id: String,

    /// Attempt number — combined with `--task-id` to form the run id.
    #[arg(long, default_value_t = 1u32)]
    attempt: u32,

    /// Hard cap on agent-loop iterations.
    ///
    /// History: 12 → 24 after the first (small) dogfood run — a groomed item's
    /// AC list invites per-criterion re-verification, and haiku exhausted 12
    /// iterations one call short of finish(done). 24 → 500 after a 0.4.0 wave
    /// item (18 ACs, 5 files) exhausted 24 in ~48s without reaching a verify
    /// cycle: 24 is calibrated for single-crate dogfood work, far too low for
    /// multi-file changes. With the gate timeout (and the upcoming wall-clock
    /// budget) as the real bounds, a high iteration cap is a backstop, not the
    /// primary limit. 500 mirrors the GTD `max_turns` convention; the eventual
    /// fix is to plumb the dispatch `max_turns` through to this flag.
    #[arg(long, default_value_t = 500u32)]
    max_iterations: u32,

    /// Per-turn output-token cap handed to the backend. A turn that hits the
    /// cap without emitting a tool call terminates the run as
    /// `Failed { mode: Truncated }` (exit 20) instead of masquerading as an
    /// ordinary stop.
    ///
    /// UNSET means **resolve**: the harness asks the backend for its
    /// per-model cap each iteration (Ollama derives it from the pinned
    /// `num_ctx` and the previous turn's prompt size; Anthropic/Bedrock read
    /// their published per-model tables; anything else falls back to
    /// `harness::engine::DEFAULT_MAX_TOKENS`). SET means **override**: the
    /// flagged value wins verbatim on every iteration and no backend
    /// resolution happens — the same unset-vs-verbatim contract
    /// `OLLAMA_NUM_CTX` holds for the context window.
    ///
    /// The resolved (or overridden) value and its provenance are recorded on
    /// the run record's `backend_settings` (`max_tokens` +
    /// `max_tokens_source`), on the transcript's `run_start`
    /// `config.max_tokens` / `config.max_tokens_source`, and in the stdout
    /// summary.
    #[arg(long)]
    max_tokens: Option<u32>,

    /// Timeout for the gate command, in seconds.
    #[arg(long, default_value_t = 300u64)]
    gate_timeout_secs: u64,

    /// Wall-clock budget in seconds. Default `1500` seconds; `0` disables
    /// (unbounded).
    ///
    /// When non-zero, the harness self-terminates gracefully with recovery
    /// facts before the worker's hard timeout. Can also be set via the
    /// environment variable `TALOS_WALL_CLOCK_SECS` (flag takes precedence
    /// over env). The compiled default is a conservative floor sized against
    /// the worst-case worker timeout (the dispatch backstop's 1800 s); a
    /// caller that knows its real effective timeout should pass a value
    /// derived from it — the env var is the intended production path, since
    /// only the dispatch worker knows its own timeout for a given run.
    ///
    /// Note: the `env` feature is NOT enabled for this project's clap
    /// dependency, so `#[arg(env = ...)]` cannot be used — the env fallback
    /// is resolved manually in `main()` via the `env_accessor` closure.
    #[arg(long)]
    wall_clock_secs: Option<u64>,

    /// Opt-in JSONL transcript of every model request/turn, tool call, and
    /// tool result — see [`harness::transcript`]. Default off (no flag = no
    /// file, no transcript-related filesystem I/O). Bare `--transcript`
    /// writes `transcript.jsonl` into the run's own state dir, next to
    /// `run.sqlite`. `--transcript <path>` uses that path verbatim instead; a
    /// relative path resolves against the current working directory — point
    /// it OUTSIDE `--workspace` to keep it out of git status and out of the
    /// agent's own context.
    ///
    /// FLAG ONLY — there is deliberately no `TALOS_TRANSCRIPT` env fallback:
    /// `templates/sudoers-dispatch-svc.tmpl`'s `env_keep` omits `TALOS_*`
    /// runtime knobs, so an env-only toggle would be a silent no-op once
    /// dispatch's sudo boundary does its `env_reset` (kb-02979 shape).
    // `Option<Option<PathBuf>>` is the deliberate clap idiom for a flag whose
    // value is itself optional (flag absent / bare flag / flag with a value)
    // — see `resolve_transcript_path`'s doc for the three-branch resolution.
    #[allow(clippy::option_option)]
    #[arg(long, num_args = 0..=1, value_name = "PATH")]
    transcript: Option<Option<PathBuf>>,

    /// Age-based retention, in days, for talos's own XDG state dir
    /// (`${XDG_STATE_HOME:-$HOME/.local/state}/talos/`). Precedence: this
    /// flag > `TALOS_STATE_RETENTION_DAYS` env > the compiled default of
    /// `30`. `0` disables pruning entirely (a retention report is still
    /// written, with `"disabled":true`).
    ///
    /// The env fallback does NOT survive dispatch's sudo boundary — see
    /// `RunArgs::transcript` above and the module doc's `TALOS_STATE_RETENTION_DAYS`
    /// entry: `templates/sudoers-dispatch-svc.tmpl`'s `env_keep` carries only
    /// `TALOS_BACKEND`, so under `env_reset` the fleet always uses the
    /// compiled default until the worker passes this flag explicitly.
    #[arg(long)]
    state_retention_days: Option<u64>,

    /// In-run compaction trigger threshold, in PERCENT of the backend's
    /// advertised context window: the loop compacts at the top of a pass
    /// when the previous turn's raw prompt reaches this share of the window
    /// (see `harness::engine::should_compact`). Precedence: this flag >
    /// `TALOS_COMPACT_THRESHOLD_PCT` env > the compiled default of 90
    /// (`harness::engine::COMPACT_THRESHOLD_PCT`). `0` DISABLES compaction
    /// entirely — no walk, no event, no counter, byte-identical to a run on
    /// a backend with no advertised limit. Values above 100 are accepted
    /// and simply never reachable. This is the compaction A/B knob — lower
    /// it (e.g. `1`) to force compaction early, set `0` for the OFF control
    /// arm; do NOT simulate it by shrinking `OLLAMA_NUM_CTX`, which also
    /// moves the derived per-turn output cap and confounds two variables.
    /// The resolved value rides the transcript's
    /// `run_start.config.compact_threshold_pct` so an experiment can prove
    /// its arms really differ.
    ///
    /// The env fallback does NOT survive dispatch's sudo boundary — same
    /// `RunArgs::transcript` / `TALOS_STATE_RETENTION_DAYS` caveat: under
    /// `env_reset` the fleet runs the compiled default until the worker
    /// passes this flag explicitly.
    #[arg(long)]
    compact_threshold_pct: Option<u64>,
}

/// Arguments for `talos ralph`.
///
/// `ralph` is a thin CLI over [`harness::ralph::run_ralph`]: it selects a
/// backend, builds a [`Workspace`] + [`ToolCtx`] like `run`, assembles a
/// [`RalphConfig`] from these flags, calls [`run_ralph`], prints a
/// [`RalphSummary`], and exits with [`ralph_exit_code`]. It is NOT run-record
/// persisted this cut.
#[derive(clap::Args)]
struct RalphArgs {
    /// Workspace root (MUST already be a git work tree — `run_ralph` does NOT
    /// run `git init`; the harness owns per-iteration git commits).
    #[arg(long)]
    workspace: PathBuf,

    /// The high-level objective the ralph loop is working toward. Rendered
    /// verbatim into each per-iteration prompt and commit message.
    #[arg(long)]
    objective: String,

    /// The outer stop-command oracle, run via `/bin/sh -c`; exit `0` =
    /// objective met. DISTINCT from `--gate`. A whitespace-only value is
    /// rejected with a JSON error before any filesystem/backend work.
    #[arg(long)]
    stop_when: String,

    /// The inner per-iteration gate, built via [`build_checks_runner`];
    /// whitespace-only = no inner gate. DISTINCT from `--stop-when`.
    #[arg(long, default_value = "")]
    gate: String,

    /// Workspace-relative path of the notes/progress file the agent
    /// reads-then-appends each iteration.
    #[arg(long, default_value = "PROGRESS.md")]
    notes_file: String,

    /// Hard cap on outer iterations (the ralph outer loop).
    #[arg(long, default_value_t = 100u32)]
    max_ralph_iterations: u32,

    /// Inner-run iteration cap handed to each fresh inner [`RunConfig`]
    /// (mirrors `run`'s `max_iterations`).
    #[arg(long, default_value_t = 500u32)]
    inner_max_iterations: u32,

    /// Stuck-detection threshold K: this many consecutive iterations with no
    /// progress terminate with [`RalphTerminal::Stuck`]. Matches
    /// [`DEFAULT_STUCK_K`].
    #[arg(long, default_value_t = DEFAULT_STUCK_K)]
    stuck_k: u32,

    /// Consecutive-do-over cap: this many consecutive do-overs (a non-green
    /// inner outcome that left the tree dirty, or a green `Finished(Done)`
    /// whose per-iteration `git commit` exited non-zero — both reverted to
    /// the last green commit) terminate with
    /// [`RalphTerminal::DoOversExhausted`] (exit 20). A green commit resets
    /// the count; an inner `BackendError` is exempt. Before any revert, a
    /// hook-rejected commit gets a bounded re-stage-and-retry — at most 2
    /// retries, and only when the hook MUTATED the work tree after the
    /// initial `git add -A`. Matches [`DEFAULT_MAX_DO_OVERS`].
    #[arg(long, default_value_t = DEFAULT_MAX_DO_OVERS)]
    max_do_overs: u32,

    /// Consecutive-backend-error cap: this many CONSECUTIVE outer iterations
    /// whose inner outcome was `BackendError` terminate with
    /// [`RalphTerminal::BackendErrorsExhausted`] (exit 1). Any
    /// non-`BackendError` iteration resets the count. INDEPENDENT of the
    /// do-over cap — a `BackendError` iteration remains exempt from
    /// do-overs. Matches [`DEFAULT_MAX_BACKEND_ERRORS`].
    #[arg(long, default_value_t = DEFAULT_MAX_BACKEND_ERRORS)]
    max_backend_errors: u32,

    /// Timeout for the outer stop-command oracle, in seconds.
    #[arg(long, default_value_t = 300u64)]
    stop_when_timeout_secs: u64,

    /// Timeout for the inner gate command, in seconds.
    #[arg(long, default_value_t = 300u64)]
    gate_timeout_secs: u64,

    /// Wall-clock budget in seconds. `0` or absent = unbounded. Can also be
    /// set via the environment variable `TALOS_RALPH_WALL_CLOCK_SECS` (flag
    /// takes precedence over env). Resolved manually in the handler via the
    /// `env_accessor` closure (the clap `env` feature is NOT enabled).
    #[arg(long)]
    ralph_wall_clock_secs: Option<u64>,

    /// Offload directory for oversized tool output. Defaults to
    /// `talos_state_dir("talos-ralph").join("offload")`.
    #[arg(long)]
    offload_dir: Option<PathBuf>,
}

// ============================================================================
// Backend dispatch
// ============================================================================

/// Runtime backend: one variant per supported model provider.
///
/// No `Debug` derive — the api key must not appear in formatter chains.
enum Backend {
    Anthropic(AnthropicBackend),
    Bedrock(BedrockBackend),
    Ollama(OllamaBackend),
}

#[async_trait]
impl ModelBackend for Backend {
    async fn turn(&self, req: &TurnRequest<'_>) -> Result<AssistantTurn, BackendError> {
        match self {
            Self::Anthropic(b) => b.turn(req).await,
            Self::Bedrock(b) => b.turn(req).await,
            Self::Ollama(b) => b.turn(req).await,
        }
    }

    /// Forward the per-backend output-cap resolution — the dispatch enum
    /// must not silently flatten every lane to the trait fallback (the
    /// measured-vs-shipped drift design 07 exists to prevent).
    fn output_cap(&self, prompt_tokens: Option<u32>) -> OutputCapResolution {
        match self {
            Self::Anthropic(b) => b.output_cap(prompt_tokens),
            Self::Bedrock(b) => b.output_cap(prompt_tokens),
            Self::Ollama(b) => b.output_cap(prompt_tokens),
        }
    }

    /// Forward the advertised context limit — same lane-parity argument as
    /// [`Self::output_cap`]: without this forward, `Backend::Ollama` would
    /// silently advertise no limit and the engine's compaction path would be
    /// off on the only lane that supports it.
    fn context_limit(&self) -> Option<u32> {
        match self {
            Self::Anthropic(b) => b.context_limit(),
            Self::Bedrock(b) => b.context_limit(),
            Self::Ollama(b) => b.context_limit(),
        }
    }
}

// ============================================================================
// Pure, unit-testable helper functions
// ============================================================================

/// Apply the `--max-tokens` flag to a [`RunConfig`] — the ONE pinned
/// mechanism, shared by every call site: `Some(n)` overrides the per-turn cap
/// verbatim, `None` leaves the config alone so the engine loop resolves per
/// backend per iteration. Keeps [`RunConfig::with_max_tokens`]'s `u32`
/// signature intact (an unset flag is not a value).
fn with_flagged_max_tokens(config: RunConfig, v: Option<u32>) -> RunConfig {
    match v {
        Some(n) => config.with_max_tokens(n),
        None => config,
    }
}

/// Map a [`LoopOutcome`] to the locked exit-code contract.
///
/// ## Rationale for reading `LoopOutcome`, not `Disposition`
///
/// `BackendError`'s [`LoopOutcome::into_disposition`] also yields
/// `Disposition::Failed`, which would collapse engine-broke (code 1) into
/// task-Failed (code 20) if we read the disposition instead.
/// `BackendError` = transport/auth/rate-limit; it **must never** collide with
/// a task-originated `Failed` (code 20).
///
/// ## Locked map
///
/// | Outcome | Code |
/// |---------|------|
/// | `Finished(Done{..})` | 0 |
/// | `Finished(Blocked{..})` | 10 |
/// | `Finished(Failed{..})` (incl. `FailureMode::Truncated`, `FailureMode::AnswerSchemaExhausted`) | 20 |
/// | `StoppedWithoutFinish` | 20 |
/// | `MaxIterations` | 20 |
/// | `BudgetExhausted` | 20 |
/// | `Finished(AlreadySatisfied{..})` | 30 |
/// | `Finished(Answer{..})` | 40 |
/// | `BackendError(_)` | 1 |
///
/// 30 exists as its own code because at 20 the dispatch worker cannot tell an
/// already-satisfied run from a real failure, and at 0 it would be pushed as
/// a Done — and an already-satisfied run has nothing to push.
///
/// 40 exists for exactly the same reason, applied to answer mode. Exit 0 is
/// the ONE thing the dispatch worker (`agent-gtd-dispatch`
/// `talos.py::map_talos_result`) relies on to mean "push the branch", and an
/// answer run has nothing to push — its deliverable is the JSON payload on
/// stdout. 20 would misread a successful answer as a task failure. An unknown
/// code falls to that worker's engine-error catch-all, which fails safe —
/// which is the right landing spot until the worker grows an arm for it.
fn exit_code(outcome: &LoopOutcome) -> i32 {
    match outcome {
        LoopOutcome::Finished(Disposition::Done { .. }) => 0,
        LoopOutcome::Finished(Disposition::AlreadySatisfied { .. }) => 30,
        LoopOutcome::Finished(Disposition::Answer { .. }) => 40,
        LoopOutcome::Finished(Disposition::Blocked { .. }) => 10,
        LoopOutcome::Finished(Disposition::Failed { .. })
        | LoopOutcome::StoppedWithoutFinish
        | LoopOutcome::MaxIterations
        | LoopOutcome::BudgetExhausted { .. } => 20,
        LoopOutcome::BackendError(_) => 1,
    }
}

/// Closed [`LoopOutcome`] discriminant string for the stdout summary `outcome` field.
///
/// Uses a hand-written match over all variants. `format!("{:?}")` is
/// explicitly **forbidden** — it would leak the `BackendError` payload into
/// the summary and prevent consumers from reliably identifying infra-broke
/// runs by the `outcome` field alone.
fn outcome_str(outcome: &LoopOutcome) -> &'static str {
    match outcome {
        LoopOutcome::Finished(_) => "Finished",
        LoopOutcome::StoppedWithoutFinish => "StoppedWithoutFinish",
        LoopOutcome::MaxIterations => "MaxIterations",
        LoopOutcome::BudgetExhausted { .. } => "BudgetExhausted",
        LoopOutcome::BackendError(_) => "BackendError",
    }
}

/// Resolve `--transcript`'s effective output path from the three-state clap
/// flag: flag absent -> `None` (no transcript); bare `--transcript` ->
/// `Some(state_dir.join("transcript.jsonl"))`, alongside `run.sqlite`; an
/// explicit `--transcript <path>` -> that path verbatim, NOT joined against
/// `state_dir`. Pure — reads no process env, touches no filesystem, and never
/// resolves or canonicalizes a relative explicit path; it flows through
/// exactly as given, to be resolved against the process CWD downstream
/// exactly as it is today.
#[allow(clippy::option_option)]
fn resolve_transcript_path(flag: Option<Option<PathBuf>>, state_dir: &Path) -> Option<PathBuf> {
    match flag {
        None => None,
        Some(None) => Some(state_dir.join("transcript.jsonl")),
        Some(Some(path)) => Some(path),
    }
}

/// Compute the `--transcript` label from the resolved
/// [`BackendSettings`] — pure, no env accessor.
///
/// Non-Ollama kinds flow through `settings.model_label()` UNCHANGED (even
/// when the ollama-ish fields are populated). Ollama settings get
/// `think=<level|unset> num_ctx=<value|unset>` appended (sourced from the
/// RESOLVED construction values, never raw env — so the label can never
/// disagree with the record): those are prompt-surface knobs that make
/// otherwise-identical ollama transcripts incomparable if left unrecorded.
/// `num_ctx_source` is deliberately NOT in the label. `Persistence.model_label`
/// and the `Event::ModelCall` rows it feeds are UNCHANGED by this — only
/// the transcript's `run_start.label` is affected.
fn transcript_label(settings: &BackendSettings) -> String {
    if settings.kind != BackendKind::Ollama {
        return settings.model_label();
    }
    let think = settings.think.as_deref().unwrap_or("unset");
    let num_ctx = settings
        .num_ctx
        .map_or_else(|| "unset".to_string(), |n| n.to_string());
    format!("{} think={think} num_ctx={num_ctx}", settings.model_label())
}

/// Map a [`RalphTerminal`] to the ralph exit-code contract.
///
/// | Terminal | Code |
/// |---------|------|
/// | `StopConditionMet` | 0 |
/// | `Stuck` | 20 |
/// | `MaxIterationsExhausted` | 20 |
/// | `TimeBudgetExhausted` | 20 |
/// | `DoOversExhausted` | 20 |
/// | `BackendErrorsExhausted` | 1 |
/// | `Error(_)` | 1 |
///
/// Mirrors [`exit_code`]: `0` = objective met, `20` = task-side failure
/// terminals, `1` = harness/infra error. There is NO `10`/Blocked analog for
/// ralph — the Ralph outer loop has no Blocked terminal. A sustained
/// backend failure (`BackendErrorsExhausted`) is an INFRA condition, so it
/// maps to `1` — it must never collapse into the task-failure code 20.
fn ralph_exit_code(terminal: &RalphTerminal) -> i32 {
    match terminal {
        RalphTerminal::StopConditionMet => 0,
        RalphTerminal::Stuck
        | RalphTerminal::MaxIterationsExhausted
        | RalphTerminal::TimeBudgetExhausted
        | RalphTerminal::DoOversExhausted => 20,
        RalphTerminal::BackendErrorsExhausted | RalphTerminal::Error(_) => 1,
    }
}

/// Closed [`RalphTerminal`] discriminant string for the stdout
/// [`RalphSummary`] `terminal` field.
///
/// Uses a hand-written match over all variants. `format!("{:?}")` is
/// explicitly **forbidden** — it would leak the `Error(String)` payload into
/// the summary and prevent consumers from reliably identifying the terminal
/// by the `terminal` field alone.
fn ralph_terminal_str(terminal: &RalphTerminal) -> &'static str {
    match terminal {
        RalphTerminal::StopConditionMet => "StopConditionMet",
        RalphTerminal::Stuck => "Stuck",
        RalphTerminal::MaxIterationsExhausted => "MaxIterationsExhausted",
        RalphTerminal::TimeBudgetExhausted => "TimeBudgetExhausted",
        RalphTerminal::DoOversExhausted => "DoOversExhausted",
        RalphTerminal::BackendErrorsExhausted => "BackendErrorsExhausted",
        RalphTerminal::Error(_) => "Error",
    }
}

/// Write the human-readable detail for an [`RalphTerminal::Error`] payload to
/// stderr.
///
/// For `Error(msg)` writes exactly one line `talos ralph: error: <msg>` plus a
/// newline; for every other terminal writes NOTHING. The stdout
/// [`RalphSummary`] stays payload-free — this is the only channel that
/// surfaces the failing git/spawn/revert command. Matched on `Error(_)`
/// explicitly with a non-Error fallthrough so a future terminal variant
/// cannot break this contract.
///
/// Callers should ignore the [`std::io::Result`] — a stderr write failure
/// must never panic or change the exit code.
fn write_ralph_error_detail(
    terminal: &RalphTerminal,
    out: &mut impl std::io::Write,
) -> std::io::Result<()> {
    match terminal {
        RalphTerminal::Error(msg) => writeln!(out, "talos ralph: error: {msg}"),
        _ => Ok(()),
    }
}

/// Render the seed string passed to [`RunConfig::new`] for a task run.
///
/// Returns the [`render_task_prompt_from_spec`] output byte-for-byte — the
/// CLI never hand-formats task text. The engine re-wraps this under a
/// `# Task` heading via `render_task_prompt`; the renderer therefore must
/// NOT emit its own `# Task` heading.
fn make_run_seed(spec: &TaskSpec) -> String {
    render_task_prompt_from_spec(spec)
}

/// Build a [`ChecksRunner`] from a shell gate command string, or `None` if the
/// command is empty or whitespace-only.
///
/// A whitespace-only `gate_command` is treated as empty — closing the
/// `/bin/sh -c ' '` exits-0 rubber-stamp false-Done vector.
///
/// `/bin/sh -c` is used (NOT direct exec) so the gate string can contain
/// shell operators like `&&` and pipes.
fn build_checks_runner(
    gate_command: &str,
    workspace_root: PathBuf,
    gate_timeout_secs: u64,
) -> Option<ChecksRunner> {
    shell_checks_runner(
        gate_command,
        workspace_root,
        Duration::from_secs(gate_timeout_secs),
    )
}

/// Write a one-line `{"error": "<message>"}` JSON object to stderr.
fn stderr_json_error(message: &str) {
    let obj = serde_json::json!({"error": message});
    eprintln!("{obj}");
}

/// Build the one-line structured stderr object describing a successful
/// `num_ctx` resolution:
/// `{"num_ctx": {"value": <u32|null>, "source": "<explicit|probe|default>",
/// "desc": "<desc>", "warning": <string|null>}}`. Pure — the caller decides
/// when to print it (exactly once per successful resolution, and BEFORE any
/// `{"error": ...}` line, so the dispatch worker's last-stderr-line error
/// parse is unaffected). Mirrors the [`stderr_json_error`] idiom. This line is
/// DIAGNOSTIC only — nothing in-repo parses it; it is pinned so the dispatch
/// stderr log keeps a stable shape.
fn num_ctx_stderr_line(r: &harness::ollama::NumCtxResolution) -> String {
    let obj = serde_json::json!({
        "num_ctx": {
            "value": r.value,
            "source": r.source.as_str(),
            "desc": r.desc,
            "warning": r.warning,
        }
    });
    obj.to_string()
}

/// Stamp the resolved per-turn output cap (and its provenance) onto a
/// [`BackendSettings`] at construction time — the turn-1 rule. `flag` is the
/// `--max-tokens` override: `Some(v)` wins verbatim and stamps `"explicit"`;
/// `None` stamps the backend's `output_cap(None)` resolution, which the call
/// site passes as `resolved` (the backend is not consulted here, keeping the
/// helper pure and unit-testable).
///
/// Unlike `num_ctx`, a run ALWAYS has a cap, so both fields are `Some` on
/// every record this version writes — including the `"fallback"` source.
fn stamp_max_tokens(
    mut settings: BackendSettings,
    flag: Option<u32>,
    resolved: OutputCapResolution,
) -> BackendSettings {
    let (max_tokens, source) = match flag {
        Some(v) => (v, MaxTokensSource::Explicit),
        None => (resolved.max_tokens, resolved.source),
    };
    settings.max_tokens = Some(max_tokens);
    settings.max_tokens_source = Some(source.as_str().to_string());
    settings
}

/// The `num_ctx_source` a run RECORD carries for a resolved `num_ctx`:
/// `Some(as_str())` when a value was pinned, `None` when it was not —
/// `NumCtxSource::Default` pins no value, so the string `"default"` never
/// reaches a record (it appears only in the stderr provenance line).
fn num_ctx_source_for_record(r: &harness::ollama::NumCtxResolution) -> Option<String> {
    r.value.map(|_| r.source.as_str().to_string())
}

// ============================================================================
// stdout summary
// ============================================================================

/// Machine-readable JSON summary written to stdout after a successful
/// `run_persisted` call. Printed regardless of the task's disposition.
#[derive(Serialize)]
struct RunSummary {
    /// Closed [`LoopOutcome`] discriminant — one of
    /// `"Finished"`, `"StoppedWithoutFinish"`, `"MaxIterations"`,
    /// `"BudgetExhausted"`, `"BackendError"`.
    outcome: &'static str,
    /// The run's terminal [`Disposition`] (derived from the `LoopOutcome`).
    disposition: Disposition,
    /// `"{task_id}:{attempt_n}"` — the stable run identifier.
    run_id: String,
    /// Path to the `SQLite` store that holds the full run record.
    record_path: String,
    /// Number of model turns the loop drew.
    iterations: u32,
    /// Compactions that changed history — surfaced on the default path
    /// (no `--transcript`) because `RunStats` is never persisted and the
    /// run record's `compaction_facts` is only visible through the store.
    compactions: u32,
    /// The highest compaction tier reached: 0 = never compacted, 1 =
    /// reasoning tail-truncated only, 2 = tool-result payloads elided.
    highest_compaction_tier: u8,
    /// The resolved backend the run was CONSTRUCTED with (`kind`, `model`,
    /// `think`, `num_ctx`, `num_ctx_source`, `max_tokens`,
    /// `max_tokens_source`) — the SAME value stamped on the run record, so
    /// stdout and the store cannot disagree.
    backend_settings: BackendSettings,
}

/// Build the stdout [`RunSummary`] from a completed run.
#[allow(clippy::too_many_arguments)]
fn build_run_summary(
    outcome_s: &'static str,
    disposition: Disposition,
    run_id_str: String,
    record_path: String,
    iterations: u32,
    compactions: u32,
    highest_compaction_tier: u8,
    backend_settings: BackendSettings,
) -> RunSummary {
    RunSummary {
        outcome: outcome_s,
        disposition,
        run_id: run_id_str,
        record_path,
        iterations,
        compactions,
        highest_compaction_tier,
        backend_settings,
    }
}

/// Machine-readable JSON summary written to stdout after [`run_ralph`]
/// returns. `ralph` is NOT run-record persisted this cut — there is no
/// `run_id` or `record_path` field (unlike [`RunSummary`]).
#[derive(Serialize)]
struct RalphSummary {
    /// The objective the ralph loop worked toward (verbatim copy of
    /// [`RalphReport::objective`]).
    objective: String,
    /// Closed [`RalphTerminal`] discriminant — one of
    /// `"StopConditionMet"`, `"Stuck"`, `"MaxIterationsExhausted"`,
    /// `"TimeBudgetExhausted"`, `"DoOversExhausted"`,
    /// `"BackendErrorsExhausted"`, `"Error"` (the exact set
    /// [`ralph_terminal_str`] emits).
    terminal: &'static str,
    /// How many outer iterations ran ([`RalphReport::outer_iterations`]).
    outer_iterations: u32,
    /// Sum of every iteration's inner iterations
    /// ([`RalphReport::total_inner_iterations`]).
    total_inner_iterations: u64,
    /// Sum over [`RalphReport::iterations`] of each iteration's
    /// `commit_retries` — the re-stage-and-retry attempts after hook-rejected
    /// commits. A count, like [`Self::commit_rejects`]: the summary stays
    /// payload-free, so a reviewer can see the retry fired without the
    /// rejected commit's stderr ever reaching stdout.
    commit_retries_total: u32,
    /// How many iterations had at least one `git commit` exit non-zero
    /// (`commit_reject` is `Some`) — the count of hook-rejected iterations.
    commit_rejects: u32,
}

/// Build the stdout [`RalphSummary`] from a completed ralph run.
fn build_ralph_summary(
    objective: String,
    terminal_s: &'static str,
    outer_iterations: u32,
    total_inner_iterations: u64,
    commit_retries_total: u32,
    commit_rejects: u32,
) -> RalphSummary {
    RalphSummary {
        objective,
        terminal: terminal_s,
        outer_iterations,
        total_inner_iterations,
        commit_retries_total,
        commit_rejects,
    }
}

/// Resolve the `run` wall-clock budget: `--wall-clock-secs` flag >
/// `TALOS_WALL_CLOCK_SECS` env > [`DEFAULT_WALL_CLOCK_SECS`] (1500; `0` =
/// unbounded). The clap `env` feature is NOT enabled, so the env fallback is
/// resolved manually.
///
/// Unlike [`resolve_compact_threshold_pct`], a NON-NUMERIC env value here
/// falls through SILENTLY to the default instead of hard-erroring — with the
/// default armed, a typo'd env value lands on 1500 rather than unbounded
/// (previously it landed on the 0/unbounded sentinel), so the failure mode
/// is an armed-but-conservative budget, not an unbounded run. A wrong
/// threshold would poison an A/B experiment; a wrong wall-clock budget only
/// changes WHEN a graceful self-termination happens.
fn resolve_wall_clock_secs(flag: Option<u64>, env: &impl Fn(&str) -> Option<String>) -> u64 {
    flag.or_else(|| env("TALOS_WALL_CLOCK_SECS").and_then(|v| v.parse::<u64>().ok()))
        .unwrap_or(DEFAULT_WALL_CLOCK_SECS)
}

/// Resolve the ralph wall-clock budget: `flag > TALOS_RALPH_WALL_CLOCK_SECS
/// env > 0` (unbounded). Mirrors the [`resolve_wall_clock_secs`] pattern for
/// `run`'s `--wall-clock-secs` — the clap `env` feature is NOT enabled, so
/// the env fallback is resolved manually. Ralph keeps the `0` default
/// (unbounded): the ralph loop is a long-horizon outer driver, and arming a
/// default there is a separate decision from arming `run`. A non-`u64` env
/// value falls through to `0` (unbounded) without panicking.
fn resolve_ralph_wall_clock_secs(flag: Option<u64>, env: &impl Fn(&str) -> Option<String>) -> u64 {
    flag.or_else(|| env("TALOS_RALPH_WALL_CLOCK_SECS").and_then(|v| v.parse::<u64>().ok()))
        .unwrap_or(0)
}

/// Resolve the state-dir retention window, in days: `--state-retention-days`
/// flag > `TALOS_STATE_RETENTION_DAYS` env > [`DEFAULT_STATE_RETENTION_DAYS`].
/// Pure — reads no `std::env` directly, only the injected `env` accessor, and
/// never panics. Returns the resolved day-count AND the source literal
/// (`"flag"` / `"env"` / `"default"`) for the `prune-last.json` report. A
/// non-numeric, negative, or empty env value falls through to the default.
fn resolve_state_retention_days(
    flag: Option<u64>,
    env: &impl Fn(&str) -> Option<String>,
) -> (u64, &'static str) {
    if let Some(v) = flag {
        return (v, "flag");
    }
    if let Some(v) = env("TALOS_STATE_RETENTION_DAYS").and_then(|v| v.parse::<u64>().ok()) {
        return (v, "env");
    }
    (DEFAULT_STATE_RETENTION_DAYS, "default")
}

/// Resolve the in-run compaction trigger threshold, in percent:
/// `--compact-threshold-pct` flag > `TALOS_COMPACT_THRESHOLD_PCT` env >
/// [`harness::engine::COMPACT_THRESHOLD_PCT`] (the compiled 90). Pure —
/// reads no `std::env` directly, only the injected `env` accessor, and
/// never panics. `0` DISABLES compaction entirely; values above 100 are
/// accepted and simply never reachable.
///
/// An empty or whitespace-only env value is treated as unset. A
/// NON-NUMERIC env value is a hard construction error (`Err`) — never a
/// silent fallback, because a typo'd threshold would silently arm or
/// disarm the very arm an A/B experiment is measuring. (This deliberately
/// diverges from [`resolve_state_retention_days`], whose non-numeric env
/// falls through to the default: a wrong retention window only misprunes,
/// a wrong threshold poisons the experiment.)
fn resolve_compact_threshold_pct(
    flag: Option<u64>,
    env: &impl Fn(&str) -> Option<String>,
) -> Result<u64, String> {
    if let Some(v) = flag {
        return Ok(v);
    }
    if let Some(raw) = env("TALOS_COMPACT_THRESHOLD_PCT").filter(|v| !v.trim().is_empty()) {
        return raw.parse::<u64>().map_err(|_| {
            format!(
                "TALOS_COMPACT_THRESHOLD_PCT must be a number (a percent; 0 disables), got `{raw}`"
            )
        });
    }
    Ok(harness::engine::COMPACT_THRESHOLD_PCT)
}

// ============================================================================
// Backend selection from environment
// ============================================================================

/// Build an [`AnthropicBackend`] from the injected environment accessor.
///
/// Reads `ANTHROPIC_API_KEY` (required) and `ANTHROPIC_MODEL` (default
/// `claude-haiku-4-5`). Returns `Err` on any missing required var — never
/// panics. The recorded settings carry the resolved model and no
/// `think`/`num_ctx` fields (Anthropic has no such knobs this cut).
fn build_anthropic_backend(
    env: &impl Fn(&str) -> Option<String>,
) -> Result<(Backend, BackendSettings), String> {
    let api_key = env("ANTHROPIC_API_KEY").ok_or_else(|| {
        "ANTHROPIC_API_KEY must be set when using the anthropic backend".to_string()
    })?;
    let model = env("ANTHROPIC_MODEL").unwrap_or_else(|| "claude-haiku-4-5".to_string());
    Ok((
        Backend::Anthropic(AnthropicBackend::new(&model, api_key)),
        BackendSettings {
            kind: BackendKind::Anthropic,
            model,
            think: None,
            num_ctx: None,
            num_ctx_source: None,
            max_tokens: None,
            max_tokens_source: None,
        },
    ))
}

/// Build an [`OllamaBackend`] from the injected environment accessor.
///
/// Reads `OLLAMA_MODEL` (required), `OLLAMA_BASE_URL` (default
/// `http://localhost:11434`), `OLLAMA_API_KEY` (optional), `OLLAMA_THINK`
/// (`off|on|low|medium|high|max`), and resolves `OLLAMA_NUM_CTX` via the
/// shared `harness::ollama::resolve_num_ctx` — the five-branch precedence is
/// documented there: a non-empty `u32` wins verbatim (no probe);
/// empty/whitespace is treated as unset; unset with a `localhost`/`127.0.0.1`
/// base URL probes `POST /api/show` and pins the model's advertised context
/// length; unset with a non-local base URL leaves `num_ctx` unset (Ollama's
/// own default). A probe failure is returned as `Err` (fail-loud) — never a
/// fallback constant or a silent default.
///
/// `OLLAMA_THINK` is validated BEFORE any `num_ctx` resolution, so an
/// invalid value can never trigger an HTTP probe. On every successful
/// resolution a structured `{"num_ctx": {...}}` line is emitted to stderr
/// exactly once (via [`num_ctx_stderr_line`]), before any `{"error": ...}`
/// line.
///
/// The recorded settings carry the think level as the `OLLAMA_THINK` env
/// spelling (`ThinkLevel::as_str`) — the same `Option<ThinkLevel>` value
/// that is passed to `with_think` — and the `num_ctx` that is passed to
/// `with_num_ctx`, with `num_ctx_source` `"explicit"`/`"probe"` (or `None`
/// when no value was pinned): each value AND its source come out of the
/// SAME resolution, so the record can never disagree with the
/// construction.
///
/// Returns `Err` on any missing required var, unrecognised `OLLAMA_THINK`
/// value, non-`u32` `OLLAMA_NUM_CTX`, or probe failure — never panics.
async fn build_ollama_backend(
    env: &impl Fn(&str) -> Option<String>,
) -> Result<(Backend, BackendSettings), String> {
    let model = env("OLLAMA_MODEL")
        .ok_or_else(|| "OLLAMA_MODEL must be set when TALOS_BACKEND=ollama".to_string())?;
    let base_url = env("OLLAMA_BASE_URL").unwrap_or_else(|| "http://localhost:11434".to_string());

    // `OLLAMA_THINK` is validated BEFORE any num_ctx resolution so an
    // invalid value can never trigger a /api/show probe.
    let think: Option<ThinkLevel> = match env("OLLAMA_THINK").as_deref() {
        None => None,
        Some("off") => Some(ThinkLevel::Off),
        Some("on") => Some(ThinkLevel::On),
        Some("low") => Some(ThinkLevel::Low),
        Some("medium") => Some(ThinkLevel::Medium),
        Some("high") => Some(ThinkLevel::High),
        Some("max") => Some(ThinkLevel::Max),
        Some(other) => {
            return Err(format!(
                "OLLAMA_THINK must be off|on|low|medium|high|max, got \"{other}\""
            ));
        }
    };

    // `OLLAMA_API_KEY` is used FOR THE PROBE ONLY (an empty string has no
    // meaning as a bearer token); the `with_api_key` call below keeps today's
    // exact behaviour (applied whenever the var is `Some`, even empty).
    let api_key = env("OLLAMA_API_KEY").filter(|s| !s.is_empty());
    let num_ctx = harness::ollama::resolve_num_ctx(
        &base_url,
        &model,
        api_key.as_deref(),
        env("OLLAMA_NUM_CTX").as_deref(),
    )
    .await?;
    // Structured provenance line, exactly once per successful resolution
    // (all branches), before any `{"error": ...}` line.
    eprintln!("{}", num_ctx_stderr_line(&num_ctx));

    let mut ollama = OllamaBackend::new(&model, &base_url);
    if let Some(key) = env("OLLAMA_API_KEY") {
        ollama = ollama.with_api_key(key);
    }
    if let Some(n) = num_ctx.value {
        ollama = ollama.with_num_ctx(n);
    }
    if let Some(level) = think {
        ollama = ollama.with_think(level);
    }

    let settings = BackendSettings {
        kind: BackendKind::Ollama,
        model,
        think: think.map(ThinkLevel::as_str).map(str::to_string),
        num_ctx: num_ctx.value,
        num_ctx_source: num_ctx_source_for_record(&num_ctx),
        max_tokens: None,
        max_tokens_source: None,
    };
    Ok((Backend::Ollama(ollama), settings))
}

/// Build a [`BedrockBackend`] from the injected environment accessor.
///
/// Reads `ANTHROPIC_MODEL` (default `claude-haiku-4-5`, mirroring
/// [`build_anthropic_backend`]), maps it to the Bedrock inference-profile id
/// (returning `Err` on an unmapped model), constructs the backend, and
/// records the CANONICAL model name (e.g. `claude-haiku-4-5` — NOT the
/// inference-profile id) as `settings.model`; `settings.model_label()`
/// yields the historical `bedrock:{canonical}` label. `think`/`num_ctx`/
/// `num_ctx_source` are `None` (Bedrock has no such knobs this cut).
/// Credentials AND region
/// resolve via the standard AWS chain lazily on the first turn (no static
/// keys in source).
fn build_bedrock_backend(
    env: &impl Fn(&str) -> Option<String>,
) -> Result<(Backend, BackendSettings), String> {
    let canonical = env("ANTHROPIC_MODEL").unwrap_or_else(|| "claude-haiku-4-5".to_string());
    let backend = BedrockBackend::new(&canonical)?;
    Ok((
        Backend::Bedrock(backend),
        BackendSettings {
            kind: BackendKind::Bedrock,
            model: canonical,
            think: None,
            num_ctx: None,
            num_ctx_source: None,
            max_tokens: None,
            max_tokens_source: None,
        },
    ))
}

/// Select and construct the model backend from the injected environment.
///
/// Precedence: a `TALOS_BEDROCK` env var that is non-empty after `.trim()`
/// selects [`BedrockBackend`] FIRST — ahead of `TALOS_BACKEND` /
/// `ANTHROPIC_API_KEY` / `OLLAMA_*`. This lets talos run on a work machine
/// that cannot call the Anthropic API directly (AWS Bedrock Converse API,
/// standard AWS credential chain, no keys in source). An unset, empty, or
/// whitespace-only `TALOS_BEDROCK` falls through to the existing match.
///
/// Otherwise `TALOS_BACKEND`: `"anthropic"` (default when unset) |
/// `"ollama"`.
///
/// Returns `Err(message)` — never panics — for:
/// - `TALOS_BACKEND` set to anything other than `"anthropic"` / `"ollama"`
/// - Missing required provider vars (`ANTHROPIC_API_KEY`, `OLLAMA_MODEL`)
/// - `ANTHROPIC_MODEL` set to a model `TALOS_BEDROCK` cannot map (only
///   `claude-haiku-4-5` / `claude-sonnet-5` / `claude-opus-4-8`)
/// - `OLLAMA_THINK` outside the accepted set
/// - `OLLAMA_NUM_CTX` present (non-empty) but unparsable as `u32`
/// - a `POST /api/show` probe failure for a local (`localhost`/`127.0.0.1`)
///   base URL with `OLLAMA_NUM_CTX` unset (fail-loud; never a fallback)
async fn backend_from_env(
    env: &impl Fn(&str) -> Option<String>,
) -> Result<(Backend, BackendSettings), String> {
    if let Some(v) = env("TALOS_BEDROCK").as_deref()
        && !v.trim().is_empty()
    {
        return build_bedrock_backend(env);
    }
    match env("TALOS_BACKEND").as_deref() {
        Some("anthropic") | None => build_anthropic_backend(env),
        Some("ollama") => build_ollama_backend(env).await,
        Some(other) => Err(format!(
            "TALOS_BACKEND must be \"anthropic\" or \"ollama\", got \"{other}\""
        )),
    }
}

// ============================================================================
// Filesystem helpers
// ============================================================================

/// Compute talos's own XDG state root:
/// `${XDG_STATE_HOME:-$HOME/.local/state}/talos`. This is ALWAYS the prune
/// root passed to [`prune_state_root`] — never a `state_dir.parent()`, since
/// a `--task-id` containing a path separator would otherwise aim pruning at
/// the wrong directory.
fn talos_root_dir() -> PathBuf {
    let state_home = std::env::var("XDG_STATE_HOME").map_or_else(
        |_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            PathBuf::from(home).join(".local").join("state")
        },
        PathBuf::from,
    );
    state_home.join("talos")
}

/// Compute the default per-task state directory:
/// `${XDG_STATE_HOME:-$HOME/.local/state}/talos/<task-id>`.
fn talos_state_dir(task_id: &str) -> PathBuf {
    talos_root_dir().join(task_id)
}

/// Best-effort refresh of `dir`'s own mtime to "now", by opening it
/// read-only and calling `set_times`. Every error is swallowed — a run's
/// outcome and exit code must never depend on this succeeding — and it
/// writes no output. Used to protect a live run's state directory from a
/// LATER run's [`prune_state_root`] call even when nothing inside the
/// directory changed (rewriting a file inside a directory does not refresh
/// that directory's own mtime).
fn touch_dir_mtime(dir: &Path) {
    let _ = std::fs::File::open(dir)
        .and_then(|f| f.set_times(std::fs::FileTimes::new().set_modified(SystemTime::now())));
}

/// Counts from a single [`prune_state_root`] pass. `examined` is the bucket
/// total: `examined == removed + kept_keep + kept_young + skipped +
/// aggregates`.
#[derive(Debug, Default, PartialEq, Eq)]
struct PruneReport {
    /// Total candidates looked at (top-level entries plus aggregate
    /// children), including aggregate directories themselves.
    examined: usize,
    /// Directories removed via `remove_dir_all`.
    removed: usize,
    /// Expired directories that survived because they (or an ancestor of
    /// them) appear in the `keep` list.
    kept_keep: usize,
    /// Directories not yet older than `max_age`.
    kept_young: usize,
    /// Non-directories, symlinks, unreadable metadata, or a failed
    /// `remove_dir_all` call.
    skipped: usize,
    /// Immediate children of the root whose name is in [`AGGREGATE_DIRS`]
    /// (descended into, never removed themselves).
    aggregates: usize,
    /// Names of removed directories, capped at [`MAX_REPORT_NAMES`]. A
    /// top-level removal records the entry's own file name; an aggregate
    /// child records `"<aggregate>/<entry file name>"`.
    removed_names: Vec<String>,
    /// How many additional removals occurred past the [`MAX_REPORT_NAMES`]
    /// cap on `removed_names`.
    removed_truncated: usize,
}

/// True when `candidate` is `keep`-protected: `candidate` itself, or an
/// ancestor of some entry in `keep`, appears in `keep`. Component-wise via
/// [`Path::starts_with`], so `<root>/abc` does not spuriously match
/// `<root>/abc-def`.
fn is_kept(candidate: &Path, keep: &[PathBuf]) -> bool {
    keep.iter().any(|k| k.starts_with(candidate))
}

/// Classify one candidate directory by age against `max_age`, apply the
/// `keep` list, and remove it via `remove_dir_all` if both checks fall
/// through — updating `report` in place. `label` is what gets recorded into
/// `report.removed_names` on a successful removal.
///
/// A single combinator chain with one diverging `else` decides "is this a
/// directory we can read the mtime of": `std::fs::symlink_metadata` (NEVER
/// `std::fs::metadata`, which follows symlinks) is filtered to directories
/// and its modification time compared against `now`. Anything that fails
/// that chain — a plain file, a symlink (even one pointing at a directory),
/// a broken symlink, or unreadable metadata — is `skipped`. An `Err` from
/// `duration_since` (an mtime in the future relative to `now`) is also
/// `skipped` by this same chain.
fn classify_and_maybe_remove(
    candidate: &Path,
    now: SystemTime,
    max_age: Duration,
    keep: &[PathBuf],
    label: String,
    report: &mut PruneReport,
) {
    let Some(age) = std::fs::symlink_metadata(candidate)
        .ok()
        .filter(std::fs::Metadata::is_dir)
        .and_then(|m| m.modified().ok())
        .and_then(|mtime| now.duration_since(mtime).ok())
    else {
        report.skipped += 1;
        return;
    };
    if age <= max_age {
        report.kept_young += 1;
        return;
    }
    if is_kept(candidate, keep) {
        report.kept_keep += 1;
        return;
    }
    let ok = std::fs::remove_dir_all(candidate).is_ok();
    report.removed += usize::from(ok);
    report.skipped += usize::from(!ok);
    if ok {
        if report.removed_names.len() < MAX_REPORT_NAMES {
            report.removed_names.push(label);
        } else {
            report.removed_truncated += 1;
        }
    }
}

/// Age-based retention pass over talos's own XDG state root.
///
/// Every input is a parameter — this function reads no process environment
/// and calls no clock, so it is fully unit-testable against a synthetic
/// `tempdir`. Deletion is confined to the immediate children of `talos_root`
/// plus the immediate children of each [`AGGREGATE_DIRS`] entry, and nothing
/// else is ever passed to `remove_dir_all`. Every filesystem error is
/// swallowed (no `unwrap`, `expect`, or `?` anywhere in this function, and it
/// never returns an `Err`) so a run's outcome and exit code are never
/// affected by a pruning failure.
///
/// ## Aggregate directories
///
/// An immediate child of the root whose name is in [`AGGREGATE_DIRS`] is
/// NEVER removed regardless of its own mtime; instead pruning descends
/// exactly one level and applies the identical age/keep/remove rules to that
/// directory's immediate children. This is because a directory's own mtime
/// refreshes whenever a child is added, so treating an aggregator as a
/// single unit would either never expire it (while evals run) or delete
/// every historical capture at once the moment it finally did age out. Both
/// current producers of this shape write `<state>/talos/<aggregate>/<run-id>/...`:
/// [`harness::mined_eval`]'s `trial_state_dir` (not `pub`, hence not linked)
/// and the `coding_eval` example's `coding_eval_transcripts_root`. Any future
/// writer of a `<state>/talos/<aggregator>/<run-id>/...` tree must add its
/// directory name to [`AGGREGATE_DIRS`]. Note that an aggregate directory's
/// immediate children are a MIX of shapes (well-formed `<run-id>` dirs and
/// leaked `synth-task-<pid>-N` dirs) — the age rule treats them identically.
///
/// ## Concurrency
///
/// With the dispatch concurrency cap of 6, concurrent `talos run` processes
/// on one host cannot prune each other's live directories: each run
/// mtime-touches its own state dir at start (see the `touch_dir_mtime` call
/// in `run_cmd`) AND every path it writes to is in its own `keep` list — the
/// `keep` list is the load-bearing protection, the touch is belt-and-braces.
/// If both held false, the residual failure mode is a `SqliteRunStore` write
/// error surfacing as exit 1 (an engine error, re-dispatchable), never silent
/// corruption.
///
/// ## Post-mortem evidence
///
/// The run record under a pruned state dir is the ONLY post-mortem evidence
/// a run leaves — `RunSummary.record_path` is printed on stdout but never
/// persisted downstream (the dispatch worker's comment body reads only
/// `outcome`, `iterations`, and `disposition`), so anything needed for a
/// later post-mortem must be copied out of the state dir before it expires.
fn prune_state_root(
    talos_root: &Path,
    now: SystemTime,
    max_age: Duration,
    keep: &[PathBuf],
) -> PruneReport {
    let mut report = PruneReport::default();
    if max_age == Duration::ZERO {
        return report;
    }
    let root = std::fs::canonicalize(talos_root).unwrap_or_else(|_| talos_root.to_path_buf());
    let keep: Vec<PathBuf> = keep
        .iter()
        .map(|p| std::fs::canonicalize(p).unwrap_or_else(|_| p.clone()))
        .collect();

    let Ok(entries) = std::fs::read_dir(&root) else {
        return report;
    };
    for entry in entries.filter_map(Result::ok) {
        let name = entry.file_name();
        let name_str = name.to_string_lossy().into_owned();
        let candidate = root.join(&name);
        report.examined += 1;

        if AGGREGATE_DIRS.contains(&name_str.as_str()) {
            report.aggregates += 1;
            let Ok(children) = std::fs::read_dir(&candidate) else {
                continue;
            };
            for child in children.filter_map(Result::ok) {
                let child_name = child.file_name();
                let child_candidate = candidate.join(&child_name);
                report.examined += 1;
                let label = format!("{name_str}/{}", child_name.to_string_lossy());
                classify_and_maybe_remove(
                    &child_candidate,
                    now,
                    max_age,
                    &keep,
                    label,
                    &mut report,
                );
            }
            continue;
        }

        classify_and_maybe_remove(&candidate, now, max_age, &keep, name_str, &mut report);
    }
    report
}

/// Serialize a [`PruneReport`] as a single-line JSON object for
/// `<talos-root>/prune-last.json`. `disabled` is `true` exactly when
/// `retention_days == 0`.
fn prune_report_json(
    root: &Path,
    retention_days: u64,
    source: &str,
    report: &PruneReport,
) -> String {
    serde_json::json!({
        "root": root.display().to_string(),
        "retention_days": retention_days,
        "source": source,
        "disabled": retention_days == 0,
        "examined": report.examined,
        "removed": report.removed,
        "kept_keep": report.kept_keep,
        "kept_young": report.kept_young,
        "skipped": report.skipped,
        "aggregates": report.aggregates,
        "removed_names": report.removed_names,
        "removed_truncated": report.removed_truncated,
    })
    .to_string()
}

/// Read the raw run input from `--file <path>` or stdin.
///
/// Returns the bytes VERBATIM; how to interpret them depends on the mode. In
/// build mode they are parsed as a [`TaskSpec`] JSON; in answer mode they are
/// the free-text question and are NEVER handed to `serde_json`.
fn read_run_input(args: &RunArgs) -> Result<String, String> {
    if let Some(path) = &args.file {
        std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read spec file `{}`: {e}", path.display()))
    } else {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format!("failed to read spec from stdin: {e}"))?;
        Ok(buf)
    }
}

/// Validate the answer-mode flag SHAPE: `--schema` is required with
/// `--mode answer` and rejected without it.
///
/// Pure and separately unit-tested, and called BEFORE [`read_run_input`] —
/// mirroring the `--stop-when` guard in `run_ralph_cmd`. A wrongly flagged
/// invocation must fail immediately rather than block on a pipe that will
/// never be written.
fn validate_mode_flags(mode: RunMode, schema: Option<&Path>) -> Result<(), String> {
    match (mode, schema) {
        (RunMode::Answer, None) => Err("--schema is required with --mode answer".to_string()),
        (RunMode::Build, Some(_)) => Err("--schema is only valid with --mode answer".to_string()),
        (RunMode::Answer, Some(_)) | (RunMode::Build, None) => Ok(()),
    }
}

/// Read, parse and compile `--schema`, returning the RAW file text alongside
/// the compiled validator.
///
/// The raw text is what the model is shown (see
/// [`harness::prompt::render_answer_prompt`]) — re-serializing it through
/// `serde_json` first would silently reorder object keys, so the schema in the
/// prompt would stop matching the schema on disk. Runs BEFORE the run store is
/// opened, so a bad schema never touches the filesystem, exactly like a
/// malformed `TaskSpec`.
fn load_answer_schema(path: &Path) -> Result<(String, AnswerSchema), String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read --schema `{}`: {e}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("--schema `{}` is not valid JSON: {e}", path.display()))?;
    let compiled = AnswerSchema::compile(&value)
        .map_err(|e| format!("invalid --schema `{}`: {e}", path.display()))?;
    Ok((text, compiled))
}

// ============================================================================
// main
// ============================================================================

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // 1. Parse CLI — clap usage errors → exit 1 + JSON error (NOT clap's default exit 2).
    //    `--help`/`--version` also surface as `Err` from try_parse but are not
    //    usage errors: print them plainly and exit 0.
    let cli = match <Cli as clap::Parser>::try_parse() {
        Ok(c) => c,
        Err(e) => {
            if matches!(
                e.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) {
                print!("{e}");
                std::process::exit(0);
            }
            stderr_json_error(&e.to_string());
            std::process::exit(1);
        }
    };

    // A 2-variant `Command` enum can no longer use an irrefutable let-else
    // destructure — `Command::Run` and `Command::Ralph` are two distinct
    // entry shapes (TaskSpec/disposition vs objective/RalphTerminal), so
    // dispatch via match.
    match cli.command {
        Command::Run(args) => run_cmd(args).await,
        Command::Ralph(args) => run_ralph_cmd(args).await,
    }
}

/// Execute a run and report outcome + exit code (the `run` subcommand
/// handler). Full persistence: a `SQLite` store record is written on every
/// terminal path. See the module doc for the exit-code contract.
///
/// Two modes, selected by `--mode`. BUILD mode is the pre-existing path,
/// byte-for-byte: parse the input as a [`TaskSpec`], build a `ChecksRunner`
/// from its `gate_command`, wire `standard_registry`, seed with
/// `render_task_prompt_from_spec`. ANSWER mode never parses a `TaskSpec` at
/// all: the input IS the question, `--schema` supplies the result contract,
/// the registry is read-only, and no gate and no nudges are wired.
///
/// The answer-mode step ORDER is load-bearing and pinned by
/// `crates/talos/tests/cli.rs`: (1) flag-shape validation, (2) schema read +
/// parse + compile, (3) read the run input, (4) the non-empty guard, and only
/// then (5) today's backend/store/workspace sequence. (1) and (2) precede the
/// stdin read deliberately — a wrongly flagged or unreadable-schema invocation
/// must fail immediately rather than block on a pipe nobody will write.
#[allow(clippy::too_many_lines)]
async fn run_cmd(args: RunArgs) {
    // 1. Flag shape, BEFORE stdin — mirrors `run_ralph_cmd`'s `--stop-when`
    //    guard.
    if let Err(e) = validate_mode_flags(args.mode, args.schema.as_deref()) {
        stderr_json_error(&e);
        std::process::exit(1);
    }

    // 2. `--schema`, BEFORE stdin and before the store is opened. Keeps the
    //    raw text (what the model is shown) next to the compiled validator
    //    (what the harness enforces), both from the same bytes.
    let answer_schema = match args.schema.as_deref() {
        Some(path) => match load_answer_schema(path) {
            Ok(pair) => Some(pair),
            Err(e) => {
                stderr_json_error(&e);
                std::process::exit(1);
            }
        },
        None => None,
    };

    // 3. Read the run input — must happen BEFORE the store is opened, so a
    //    bad spec never touches the filesystem.
    let run_input = match read_run_input(&args) {
        Ok(s) => s,
        Err(e) => {
            stderr_json_error(&e);
            std::process::exit(1);
        }
    };

    // 4. Interpret the input per mode. In answer mode it is free text and is
    //    NEVER handed to `serde_json` — only emptiness is checked, because an
    //    empty question would send the agent off to answer nothing.
    let spec: Option<TaskSpec> = match args.mode {
        RunMode::Answer => {
            if run_input.trim().is_empty() {
                stderr_json_error("answer prompt must be non-empty");
                std::process::exit(1);
            }
            None
        }
        RunMode::Build => match serde_json::from_str(&run_input) {
            Ok(s) => Some(s),
            Err(e) => {
                stderr_json_error(&format!("invalid TaskSpec: {e}"));
                std::process::exit(1);
            }
        },
    };

    // 3. Select model backend from environment.
    let env_accessor = |k: &str| std::env::var(k).ok();
    let (backend, settings) = match backend_from_env(&env_accessor).await {
        Ok(pair) => pair,
        Err(e) => {
            stderr_json_error(&e);
            std::process::exit(1);
        }
    };
    // 3.5 Stamp the resolved per-turn output cap onto the settings — the ONE
    //     construction-time resolution, so the record, the transcript's
    //     `run_start.config`, and the stdout summary all carry the same cap
    //     and provenance. `--max-tokens` wins verbatim (`explicit`); unset,
    //     the backend's turn-1 rule resolves (Ollama with a pinned `num_ctx`
    //     derives `num_ctx - OUTPUT_TOKEN_MARGIN`).
    let settings = stamp_max_tokens(settings, args.max_tokens, backend.output_cap(None));

    // 4. Resolve run-artifact paths (default to XDG state dir outside the workspace).
    let state_dir = talos_state_dir(&args.task_id);
    let run_store_path = args
        .run_store
        .unwrap_or_else(|| state_dir.join("run.sqlite"));
    let offload_dir = args
        .offload_dir
        .unwrap_or_else(|| state_dir.join("offload"));

    // 5. Create run-store parent and offload dir (Workspace::new REQUIRES the
    //    offload dir to already exist).
    let store_parent = run_store_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    if let Err(e) = std::fs::create_dir_all(store_parent) {
        stderr_json_error(&format!(
            "failed to create store parent `{}`: {e}",
            store_parent.display()
        ));
        std::process::exit(1);
    }
    if let Err(e) = std::fs::create_dir_all(&offload_dir) {
        stderr_json_error(&format!(
            "failed to create offload dir `{}`: {e}",
            offload_dir.display()
        ));
        std::process::exit(1);
    }

    // 5.5. Prune talos's own XDG state root (age-based retention), silently.
    //      Touch this run's own state dir first so a later run's prune pass
    //      cannot treat a freshly-created-but-not-yet-modified directory as
    //      stale; then resolve the retention window and prune everything
    //      else under the root, protecting this run's own paths via `keep`.
    //      No stdout/stderr writer on any path — see `prune_state_root`'s and
    //      `touch_dir_mtime`'s doc comments.
    touch_dir_mtime(&state_dir);
    let talos_root = talos_root_dir();
    let (retention_days, retention_source) =
        resolve_state_retention_days(args.state_retention_days, &env_accessor);
    let report = if retention_days == 0 {
        PruneReport::default()
    } else {
        let keep = [
            state_dir.clone(),
            store_parent.to_path_buf(),
            offload_dir.clone(),
        ];
        prune_state_root(
            &talos_root,
            SystemTime::now(),
            Duration::from_secs(retention_days.saturating_mul(SECS_PER_DAY)),
            &keep,
        )
    };
    let _ = std::fs::write(
        talos_root.join(PRUNE_REPORT_FILENAME),
        prune_report_json(&talos_root, retention_days, retention_source, &report),
    );

    // 6. Build Workspace (canonicalizes and validates the roots).
    let workspace = match Workspace::new(args.workspace.clone(), Some(offload_dir.clone())) {
        Ok(w) => w,
        Err(e) => {
            stderr_json_error(&format!("workspace error: {e}"));
            std::process::exit(1);
        }
    };
    let workspace_root = workspace.root().to_path_buf();

    // 7. Build tool context with a disk-offload sink.
    let sink = DiskOffloadSink::new(offload_dir.clone());
    let ctx = ToolCtx::new(Arc::new(workspace), Arc::new(sink) as Arc<dyn OffloadSink>);

    // 8. Open the run store.
    let store = match SqliteRunStore::open(&run_store_path) {
        Ok(s) => s,
        Err(e) => {
            stderr_json_error(&format!("failed to open run store: {e}"));
            std::process::exit(1);
        }
    };
    let store: Arc<dyn RunStore> = Arc::new(store);

    // Resolve wall-clock budget: flag > TALOS_WALL_CLOCK_SECS env >
    // DEFAULT_WALL_CLOCK_SECS (1500; 0 = unbounded). The `env` clap feature
    // is NOT enabled (Cargo.toml features=['derive'] only), so the env
    // fallback is resolved here via the env_accessor closure.
    let wall_clock_secs = resolve_wall_clock_secs(args.wall_clock_secs, &env_accessor);

    // Resolve the compaction trigger threshold: flag > TALOS_COMPACT_THRESHOLD_PCT
    // env > the compiled default. A non-numeric env value is a hard construction
    // error — never a silent fallback (see `resolve_compact_threshold_pct`).
    let compact_threshold_pct =
        match resolve_compact_threshold_pct(args.compact_threshold_pct, &env_accessor) {
            Ok(v) => v,
            Err(e) => {
                stderr_json_error(&e);
                std::process::exit(1);
            }
        };

    // 9/10. Registry + seed prompt + RunConfig, per mode. The seed is always
    //       byte-for-byte from a renderer, never hand-formatted.
    let (tools, mut config) = match (answer_schema, spec) {
        (Some((schema_text, compiled)), _) => {
            // ANSWER mode wires NO gate and NO nudges this cut. There is no
            // TaskSpec, hence no `gate_command`, hence no `ChecksRunner` — so
            // `run_checks` is absent from the registry and an accepted
            // `Disposition::Answer` carries `Verification::NoChecksConfigured`.
            // `max_nudges(0)` structurally disables finish-recovery, whose
            // `nudge_prompt.md` steers toward `finish(done)` /
            // `finish(already_satisfied)` — both rejected in answer mode, which
            // would otherwise loop to `FailureMode::FinishDiscipline`.
            //
            // `--gate-timeout-secs` is accepted and inert here; there is
            // nothing for it to time out.
            let seed = render_answer_prompt(&run_input, &schema_text);
            let config = with_flagged_max_tokens(
                RunConfig::new(seed, args.max_iterations)
                    .with_answer_schema(compiled)
                    .with_wall_clock_secs(wall_clock_secs),
                args.max_tokens,
            )
            .with_max_nudges(0);
            (answer_registry(None), config)
        }
        (None, Some(spec)) => {
            // BUILD mode — unchanged. The optional ChecksRunner from
            // spec.gate_command wires to BOTH the tool registry (so the agent
            // can call `run_checks`) AND the RunConfig (so finish(done) is
            // harness-verified).
            let checks =
                build_checks_runner(&spec.gate_command, workspace_root, args.gate_timeout_secs);
            let tools = standard_registry(checks.clone());
            let seed = make_run_seed(&spec);
            let config = if let Some(runner) = checks {
                with_flagged_max_tokens(
                    RunConfig::new(seed, args.max_iterations)
                        .with_checks(runner)
                        .with_wall_clock_secs(wall_clock_secs),
                    args.max_tokens,
                )
            } else {
                with_flagged_max_tokens(
                    RunConfig::new(seed, args.max_iterations).with_wall_clock_secs(wall_clock_secs),
                    args.max_tokens,
                )
            };
            (tools, config)
        }
        // Unreachable: `validate_mode_flags` guarantees a schema iff
        // `--mode answer`, and the mode match above builds a `TaskSpec` for
        // exactly the other case. Expressed as an arm rather than an
        // `unwrap` so a future flag change degrades to a JSON error.
        (None, None) => {
            stderr_json_error("internal error: no TaskSpec and no --schema after validation");
            std::process::exit(1);
        }
    };
    // The resolved compaction threshold — flag > env > default — applied
    // once here so BOTH mode arms carry it, and `run_start.config` records
    // the value actually in force (the A/B experiment's proof its arms
    // really differ).
    config = config.with_compact_threshold_pct(compact_threshold_pct);
    // Label is computed from `settings` BEFORE it moves into `persistence`
    // below; `--transcript` is opt-in (`args.transcript` is `None` unless the
    // flag was passed) and has no env fallback — see `RunArgs::transcript`.
    if let Some(path) = resolve_transcript_path(args.transcript.clone(), &state_dir) {
        let label = transcript_label(&settings);
        config = config.with_transcript(path, label);
    }

    // 11. Assemble persistence bundle and run.
    let rid = run_id(&args.task_id, args.attempt);
    let persistence = Persistence {
        store,
        task_id: args.task_id,
        attempt_n: args.attempt,
        model_label: settings.model_label(),
        backend_settings: Some(settings.clone()),
    };

    let result = match run_persisted(&backend, &tools, &ctx, &config, &persistence).await {
        Ok(r) => r,
        Err(e) => {
            // StoreError — record may be partially written.
            stderr_json_error(&format!("store error during run: {e}"));
            std::process::exit(1);
        }
    };

    // 12. Print machine-readable summary and exit with the locked code.
    let outcome_s = outcome_str(&result.outcome);
    let exit_c = exit_code(&result.outcome);
    let iterations = result.stats.iterations;
    let compactions = result.stats.compactions;
    let highest_compaction_tier = result.stats.highest_compaction_tier;
    let disposition = result.outcome.into_disposition();
    let record_path = run_store_path.display().to_string();
    let summary = build_run_summary(
        outcome_s,
        disposition,
        rid,
        record_path,
        iterations,
        compactions,
        highest_compaction_tier,
        settings,
    );
    println!(
        "{}",
        serde_json::to_string(&summary)
            .expect("RunSummary serializes infallibly — all fields are owned serde types")
    );
    std::process::exit(exit_c);
}

/// Run the Ralph outer loop over [`run_ralph`] (the `ralph` subcommand
/// handler). A thin CLI: select the backend, build [`Workspace`] +
/// [`ToolCtx`] like `run`, assemble a [`RalphConfig`] from the flags, call
/// [`run_ralph`], print a [`RalphSummary`], and exit with
/// [`ralph_exit_code`]. NOT run-record persisted this cut — no store is
/// opened.
async fn run_ralph_cmd(args: RalphArgs) {
    // 1. Validate `--stop-when` BEFORE any filesystem/backend work: an empty
    //    oracle exits `0` every call and would declare the objective met on
    //    iteration 1 — a false-done vector. Whitespace-only is rejected.
    if args.stop_when.trim().is_empty() {
        stderr_json_error("--stop-when must be a non-empty command (whitespace-only rejected)");
        std::process::exit(1);
    }

    // 2. Select model backend from environment (same contract as `run`).
    let env_accessor = |k: &str| std::env::var(k).ok();
    let (backend, _settings) = match backend_from_env(&env_accessor).await {
        Ok(pair) => pair,
        Err(e) => {
            stderr_json_error(&e);
            std::process::exit(1);
        }
    };

    // 3. Resolve + create the offload dir (Workspace::new REQUIRES the
    //    offload dir to already exist). Defaults to
    //    `talos_state_dir("talos-ralph").join("offload")`.
    let offload_dir = args
        .offload_dir
        .unwrap_or_else(|| talos_state_dir("talos-ralph").join("offload"));
    if let Err(e) = std::fs::create_dir_all(&offload_dir) {
        stderr_json_error(&format!(
            "failed to create offload dir `{}`: {e}",
            offload_dir.display()
        ));
        std::process::exit(1);
    }

    // 3.5. Self-protect `talos-ralph`'s own state dir against a later `talos
    //      run`'s prune pass. `run_ralph` is not persisted and never adds
    //      `talos-ralph` (or `offload_dir`, unless overridden below the
    //      default) to any `keep` list, and this handler only ever
    //      `create_dir_all`s the `offload` CHILD — a no-op once it exists —
    //      so without this touch the `talos-ralph` directory's own mtime
    //      freezes at first-ever ralph run and a live loop's state dir could
    //      be deleted out from under it. No pruning happens here — ralph
    //      never prunes (see `RalphArgs`) — and this writes no output.
    let ralph_state_dir = talos_state_dir("talos-ralph");
    touch_dir_mtime(&ralph_state_dir);

    // 4. Build Workspace (canonicalizes and validates the roots) + ToolCtx
    //    with a disk-offload sink, exactly like the run handler.
    let workspace = match Workspace::new(args.workspace.clone(), Some(offload_dir.clone())) {
        Ok(w) => w,
        Err(e) => {
            stderr_json_error(&format!("workspace error: {e}"));
            std::process::exit(1);
        }
    };
    let workspace_root = workspace.root().to_path_buf();
    let sink = DiskOffloadSink::new(offload_dir.clone());
    let ctx = ToolCtx::new(Arc::new(workspace), Arc::new(sink) as Arc<dyn OffloadSink>);

    // 5. Build the stop-command oracle (outer) via `/bin/sh -c`.
    let stop_command = CheckCommand {
        program: "/bin/sh".to_string(),
        args: vec!["-c".to_string(), args.stop_when.clone()],
    };

    // 6. Build the optional inner per-iteration ChecksRunner from `--gate`.
    //    `build_checks_runner` returns `None` for whitespace-only/empty gate.
    let inner_runner = build_checks_runner(&args.gate, workspace_root, args.gate_timeout_secs);

    // 7. Resolve the ralph wall-clock: flag > TALOS_RALPH_WALL_CLOCK_SECS env
    //    > 0 (unbounded).
    let wall_clock_secs = resolve_ralph_wall_clock_secs(args.ralph_wall_clock_secs, &env_accessor);

    // 8. Assemble the RalphConfig.
    let mut config = RalphConfig::new(
        &args.objective,
        stop_command,
        args.max_ralph_iterations,
        args.inner_max_iterations,
    )
    .with_notes_file(&args.notes_file)
    .with_stuck_k(args.stuck_k)
    .with_max_do_overs(args.max_do_overs)
    .with_max_backend_errors(args.max_backend_errors)
    .with_wall_clock_secs(wall_clock_secs)
    .with_stop_command_timeout(Duration::from_secs(args.stop_when_timeout_secs));
    if let Some(runner) = inner_runner {
        config = config.with_inner_checks(runner);
    }

    // 9. Run the ralph outer loop. `run_ralph` does NOT run `git init` — the
    //    workspace must already be a git work tree (validated by the caller).
    let report: RalphReport = run_ralph(&backend, &ctx, &config).await;

    // 10. Print machine-readable summary and exit with the ralph code.
    let terminal_s = ralph_terminal_str(&report.terminal);
    let exit_c = ralph_exit_code(&report.terminal);
    // Retry telemetry (counts only — the summary stays payload-free):
    // `commit_retries_total` sums the per-iteration retry commit
    // invocations; `commit_rejects` counts iterations with at least one
    // rejected commit. Together they make the retry decision observable:
    // a run with zero rejects never fired the re-stage-and-retry.
    let commit_retries_total: u32 = report
        .iterations
        .iter()
        .map(|it| u32::from(it.commit_retries))
        .sum();
    let commit_rejects = u32::try_from(
        report
            .iterations
            .iter()
            .filter(|it| it.commit_reject.is_some())
            .count(),
    )
    .unwrap_or(u32::MAX);
    let summary = build_ralph_summary(
        report.objective.clone(),
        terminal_s,
        report.outer_iterations(),
        report.total_inner_iterations(),
        commit_retries_total,
        commit_rejects,
    );
    println!(
        "{}",
        serde_json::to_string(&summary)
            .expect("RalphSummary serializes infallibly — all fields are owned serde types")
    );
    // Surface the Error payload on stderr (never on stdout — the summary stays
    // payload-free). A stderr write failure is ignored: it must never panic
    // nor change the ralph exit code.
    let _ = write_ralph_error_detail(&report.terminal, &mut std::io::stderr());
    std::process::exit(exit_c);
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::{
        Backend, DEFAULT_WALL_CLOCK_SECS, MAX_REPORT_NAMES, PruneReport, RalphSummary, RunConfig,
        RunMode, RunSummary, SECS_PER_DAY, backend_from_env, build_checks_runner,
        build_ralph_summary, build_run_summary, exit_code, load_answer_schema, make_run_seed,
        num_ctx_source_for_record, num_ctx_stderr_line, outcome_str, prune_report_json,
        prune_state_root, ralph_exit_code, ralph_terminal_str, resolve_compact_threshold_pct,
        resolve_ralph_wall_clock_secs, resolve_state_retention_days, resolve_transcript_path,
        resolve_wall_clock_secs, stamp_max_tokens, touch_dir_mtime, transcript_label,
        validate_mode_flags, with_flagged_max_tokens, write_ralph_error_detail,
    };
    use harness::anthropic::AnthropicBackend;
    use harness::bedrock::BedrockBackend;
    use harness::engine::LoopOutcome;
    use harness::exec::ChangeEvidence;
    use harness::model::{
        BackendError, MaxTokensSource, ModelBackend, OutputCapResolution, TerminalKind,
        TransientKind,
    };
    use harness::ollama::{OllamaBackend, ThinkLevel};
    use harness::prompt::render_task_prompt_from_spec;
    use harness::ralph::RalphTerminal;
    use harness::run_record::{
        BackendKind, BackendSettings, Disposition, FailureMode, Verification,
    };
    use harness::task_spec::{FileToModify, TaskSpec};
    use std::path::PathBuf;
    use std::time::{Duration, SystemTime};

    /// A closed, no-knob Anthropic settings value for tests that don't care
    /// about the backend (only its presence in the summary).
    fn default_settings() -> BackendSettings {
        BackendSettings {
            kind: BackendKind::Anthropic,
            model: "claude-haiku-4-5".to_string(),
            think: None,
            num_ctx: None,
            num_ctx_source: None,
            max_tokens: None,
            max_tokens_source: None,
        }
    }

    // ---- output cap: dispatch forward + flag + construction stamping ------

    /// The `Backend` dispatch enum forwards `output_cap` to every variant —
    /// lane parity with the engine's per-iteration resolution (a wrapper that
    /// forgot the forward silently flattens its lane to the trait fallback).
    #[test]
    fn backend_dispatch_forwards_output_cap_to_each_variant() {
        assert_eq!(
            Backend::Anthropic(AnthropicBackend::new("claude-haiku-4-5", "k")).output_cap(None),
            OutputCapResolution {
                max_tokens: 64_000,
                source: MaxTokensSource::Table,
            }
        );
        assert_eq!(
            Backend::Ollama(OllamaBackend::new("m", "http://localhost:11434").with_num_ctx(32_768))
                .output_cap(None),
            OutputCapResolution {
                max_tokens: 16_384,
                source: MaxTokensSource::Derived,
            }
        );
        let bedrock = Backend::Bedrock(BedrockBackend::new("claude-sonnet-5").expect("mapped"));
        assert_eq!(
            bedrock.output_cap(None),
            OutputCapResolution {
                max_tokens: 128_000,
                source: MaxTokensSource::Table,
            }
        );
    }

    /// The `Backend` dispatch enum forwards `context_limit` to the boxed
    /// inner backend exactly as `output_cap` does — the wrapper must not
    /// flatten the Ollama lane to the trait's `None` default (which would
    /// silently disable compaction on the only backend that supports it),
    /// and must not clobber the existing `output_cap` forward while doing it.
    #[test]
    fn backend_dispatch_forwards_context_limit_and_keeps_output_cap() {
        // Unpinned inner: context_limit forwards `None`, output_cap forwards
        // the fallback resolution.
        let unpinned_inner = OllamaBackend::new("m", "http://localhost:11434");
        let unpinned = Backend::Ollama(OllamaBackend::new("m", "http://localhost:11434"));
        assert_eq!(
            unpinned.context_limit(),
            unpinned_inner.context_limit(),
            "the wrapper must forward the inner backend's (None) limit"
        );
        assert_eq!(unpinned.context_limit(), None);
        assert_eq!(
            unpinned.output_cap(None),
            unpinned_inner.output_cap(None),
            "the new forward must not clobber the output_cap forward"
        );
        assert_eq!(
            unpinned.output_cap(None),
            OutputCapResolution {
                max_tokens: harness::model::DEFAULT_MAX_TOKENS,
                source: MaxTokensSource::Fallback,
            }
        );

        // Pinned inner: context_limit forwards Some(8192), output_cap still
        // forwards the same inner derivation.
        let pinned_inner = OllamaBackend::new("m", "http://localhost:11434").with_num_ctx(8192);
        let pinned =
            Backend::Ollama(OllamaBackend::new("m", "http://localhost:11434").with_num_ctx(8192));
        assert_eq!(
            pinned.context_limit(),
            pinned_inner.context_limit(),
            "the wrapper must forward the inner backend's (Some) limit"
        );
        assert_eq!(pinned.context_limit(), Some(8192));
        assert_eq!(
            pinned.output_cap(None),
            pinned_inner.output_cap(None),
            "the new forward must not clobber the output_cap forward"
        );

        // The other two variants inherit the trait's `None` default (no
        // override exists), keeping compaction off by construction.
        assert_eq!(
            Backend::Anthropic(AnthropicBackend::new("claude-haiku-4-5", "k")).context_limit(),
            None
        );
        let bedrock = Backend::Bedrock(BedrockBackend::new("claude-sonnet-5").expect("mapped"));
        assert_eq!(bedrock.context_limit(), None);
    }

    /// `with_flagged_max_tokens` is the ONE pinned mechanism the three
    /// `RunConfig` call sites share: `Some` overrides verbatim, `None` leaves
    /// the config's `max_tokens` at `None` (resolve per backend).
    #[test]
    fn with_flagged_max_tokens_applies_the_flag_conditionally() {
        assert_eq!(
            with_flagged_max_tokens(RunConfig::new("t", 1), Some(4096)).max_tokens,
            Some(4096),
            "a flagged value overrides the cap verbatim"
        );
        assert_eq!(
            with_flagged_max_tokens(RunConfig::new("t", 1), None).max_tokens,
            None,
            "an unset flag must leave the config at resolve-per-backend"
        );
    }

    /// Construction-time stamping: the flag wins as `"explicit"`; unset, the
    /// backend's turn-1 resolution is recorded verbatim — the default
    /// Anthropic lane's published table, and the Ollama-with-pinned-`num_ctx`
    /// derivation.
    #[test]
    fn stamp_max_tokens_records_flag_and_backend_resolution() {
        // Flagged → verbatim + "explicit".
        let flagged = stamp_max_tokens(
            default_settings(),
            Some(4096),
            OutputCapResolution {
                max_tokens: 64_000,
                source: MaxTokensSource::Table,
            },
        );
        assert_eq!(flagged.max_tokens, Some(4096));
        assert_eq!(flagged.max_tokens_source.as_deref(), Some("explicit"));

        // Unset, default Anthropic (claude-haiku-4-5) → the published table.
        let anthropic = stamp_max_tokens(default_settings(), None, {
            let b = Backend::Anthropic(AnthropicBackend::new("claude-haiku-4-5", "k"));
            b.output_cap(None)
        });
        assert_eq!(anthropic.max_tokens, Some(64_000));
        assert_eq!(anthropic.max_tokens_source.as_deref(), Some("table"));

        // Unset, Ollama with OLLAMA_NUM_CTX=32768 pinned → the derivation.
        let ollama = stamp_max_tokens(default_settings(), None, {
            let b = Backend::Ollama(
                OllamaBackend::new("m", "http://localhost:11434").with_num_ctx(32_768),
            );
            b.output_cap(None)
        });
        assert_eq!(ollama.max_tokens, Some(16_384));
        assert_eq!(ollama.max_tokens_source.as_deref(), Some("derived"));
    }

    // ---- exit_code: all 6 arms ----------------------------------------

    #[test]
    fn exit_code_finished_done_is_0() {
        let outcome = LoopOutcome::Finished(Disposition::Done {
            summary: "ok".into(),
            verification: Verification::NoChecksConfigured,
            change: ChangeEvidence::default(),
        });
        assert_eq!(exit_code(&outcome), 0);
    }

    #[test]
    fn exit_code_already_satisfied_is_30() {
        let outcome = LoopOutcome::Finished(Disposition::AlreadySatisfied {
            reason: "nothing needed changing".into(),
            verification: Verification::NoChecksConfigured,
            change: ChangeEvidence::TreeUnchanged,
        });
        assert_eq!(
            exit_code(&outcome),
            30,
            "an already-satisfied run is neither a Done to push (0) nor a failure (20)"
        );
    }

    /// `RunSummary` embeds the `Disposition` and derives `Serialize`, so the
    /// new variant and its leg-3 evidence reach stdout with no new field.
    #[test]
    fn run_summary_serializes_already_satisfied_with_its_change_evidence() {
        let summary = build_run_summary(
            "Finished",
            Disposition::AlreadySatisfied {
                reason: "already complete".into(),
                verification: Verification::NoChecksConfigured,
                change: ChangeEvidence::TreeUnchanged,
            },
            "t:1".to_string(),
            "/tmp/run.sqlite".to_string(),
            1,
            0,
            0,
            default_settings(),
        );
        let json = serde_json::to_string(&summary).expect("serialize");
        assert!(json.contains("AlreadySatisfied"), "got {json}");
        assert!(json.contains("TreeUnchanged"), "got {json}");
    }

    #[test]
    fn exit_code_answer_is_40() {
        let outcome = LoopOutcome::Finished(Disposition::Answer {
            result: serde_json::json!({"verdict": "ok"}),
            verification: Verification::NoChecksConfigured,
            change: ChangeEvidence::TreeUnchanged,
        });
        assert_eq!(
            exit_code(&outcome),
            40,
            "an answer run is neither a pushable Done (0) nor a failure (20)"
        );
    }

    /// The validated payload reaches stdout transitively through the embedded
    /// `Disposition` — `RunSummary` gains no top-level `result` field.
    #[test]
    fn run_summary_serializes_answer_with_its_result() {
        let summary = build_run_summary(
            "Finished",
            Disposition::Answer {
                result: serde_json::json!({"verdict": "ok"}),
                verification: Verification::NoChecksConfigured,
                change: ChangeEvidence::TreeUnchanged,
            },
            "t:1".to_string(),
            "/tmp/run.sqlite".to_string(),
            1,
            0,
            0,
            default_settings(),
        );
        let json = serde_json::to_string(&summary).expect("serialize");
        assert!(json.contains("Answer"), "got {json}");
        assert!(json.contains("verdict"), "got {json}");
    }

    /// The ORCHESTRATOR contract `RunSummary` has to carry that item 1's
    /// positive test does not: the change evidence.
    ///
    /// `RunSummary` gains no field for it — the payload reaches stdout
    /// transitively through the embedded `Disposition`, which is externally
    /// tagged — so `["disposition"]["Answer"]["change"]` is the read path.
    /// A caller needs it to tell a VERIFIED read-only answer (`TreeUnchanged`,
    /// the precondition actually held) from an UNVERIFIABLE one
    /// (`Unobservable`, the precondition failed open), which is the only
    /// difference between an answer you can trust was read-only and one you
    /// cannot.
    #[test]
    fn run_summary_exposes_an_unobservable_answer_s_change_evidence() {
        let summary = build_run_summary(
            "Finished",
            Disposition::Answer {
                result: serde_json::json!({"verdict": "ok"}),
                verification: Verification::NoChecksConfigured,
                change: ChangeEvidence::Unobservable {
                    reason: "no git".to_string(),
                },
            },
            "t:1".to_string(),
            "/tmp/run.sqlite".to_string(),
            1,
            0,
            0,
            default_settings(),
        );
        let value = serde_json::to_value(&summary).expect("serialize");
        assert_eq!(
            value["disposition"]["Answer"]["change"]["Unobservable"]["reason"], "no git",
            "the fail-open reason must be visible to the orchestrator; got {value}"
        );
        // And the verified case is distinguishable from it.
        let verified = build_run_summary(
            "Finished",
            Disposition::Answer {
                result: serde_json::json!({"verdict": "ok"}),
                verification: Verification::NoChecksConfigured,
                change: ChangeEvidence::TreeUnchanged,
            },
            "t:1".to_string(),
            "/tmp/run.sqlite".to_string(),
            1,
            0,
            0,
            default_settings(),
        );
        let verified = serde_json::to_value(&verified).expect("serialize");
        assert_eq!(verified["disposition"]["Answer"]["change"], "TreeUnchanged");
    }

    // ---- answer-mode flag shape ----------------------------------------

    /// `--schema` is required with `--mode answer` and rejected without it;
    /// the two legal shapes pass. Pins the exact wording the CLI tests assert
    /// on stderr.
    #[test]
    fn validate_mode_flags_requires_schema_iff_answer_mode() {
        let path = PathBuf::from("/tmp/schema.json");

        assert_eq!(
            validate_mode_flags(RunMode::Answer, None),
            Err("--schema is required with --mode answer".to_string())
        );
        assert_eq!(
            validate_mode_flags(RunMode::Build, Some(&path)),
            Err("--schema is only valid with --mode answer".to_string())
        );
        assert_eq!(validate_mode_flags(RunMode::Answer, Some(&path)), Ok(()));
        assert_eq!(validate_mode_flags(RunMode::Build, None), Ok(()));
    }

    /// `load_answer_schema` returns the RAW file text next to the compiled
    /// validator — never a re-serialization, which would reorder keys and
    /// desynchronize the schema in the prompt from the schema on disk.
    #[test]
    fn load_answer_schema_returns_the_raw_text_and_a_working_validator() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("schema.json");
        // Deliberately NON-alphabetical key order and non-canonical spacing.
        let raw = "{ \"title\": \"t\", \"type\": \"object\",\n  \"required\": [\"a\"] }";
        std::fs::write(&path, raw).expect("write");

        let (text, compiled) = load_answer_schema(&path).expect("compiles");
        assert_eq!(text, raw, "the raw bytes must survive verbatim");
        assert!(
            compiled
                .validation_errors(&serde_json::json!({ "a": 1 }))
                .is_empty()
        );
        assert!(
            !compiled
                .validation_errors(&serde_json::json!({}))
                .is_empty(),
            "the compiled validator must actually enforce `required`"
        );
    }

    /// The three ways `--schema` can be bad, each with its own message.
    #[test]
    fn load_answer_schema_reports_each_failure_mode() {
        let dir = tempfile::tempdir().expect("tempdir");

        let missing = dir.path().join("missing.json");
        let err = load_answer_schema(&missing).expect_err("a missing file fails");
        assert!(err.contains("--schema"), "got {err}");
        assert!(err.contains("missing.json"), "got {err}");

        let not_json = dir.path().join("not-json.txt");
        std::fs::write(&not_json, "nope").expect("write");
        let err = load_answer_schema(&not_json).expect_err("non-JSON fails");
        assert!(err.contains("--schema"), "got {err}");
        assert!(err.contains("JSON"), "got {err}");

        let bad = dir.path().join("bad.json");
        std::fs::write(&bad, r#"{"type":"not-a-type"}"#).expect("write");
        let err = load_answer_schema(&bad).expect_err("an uncompilable schema fails");
        assert!(err.contains("invalid --schema"), "got {err}");
    }

    #[test]
    fn exit_code_finished_blocked_is_10() {
        let outcome = LoopOutcome::Finished(Disposition::Blocked {
            decision_needed: "needs input".into(),
        });
        assert_eq!(exit_code(&outcome), 10);
    }

    #[test]
    fn exit_code_finished_failed_is_20() {
        let outcome = LoopOutcome::Finished(Disposition::Failed {
            mode: FailureMode::Loop,
            summary: "looped".into(),
        });
        assert_eq!(exit_code(&outcome), 20);
    }

    /// AC7 — a `max_tokens` truncation rides exit 20 through the existing
    /// `Finished(Failed{..})` arm; NO new arm exists (the table is keyed by
    /// `LoopOutcome`, and `Truncated` is a `FailureMode`).
    #[test]
    fn exit_code_finished_failed_truncated_is_20() {
        let outcome = LoopOutcome::Finished(Disposition::Failed {
            mode: FailureMode::Truncated,
            summary: "x".into(),
        });
        assert_eq!(
            exit_code(&outcome),
            20,
            "Truncated must ride 20 under Finished(Failed{{..}})"
        );
    }

    /// `AnswerSchemaExhausted` rides exit 20 exactly as `Truncated` does —
    /// through the existing `Finished(Failed{..})` arm; NO new arm exists
    /// (the map is keyed by `LoopOutcome`, and `AnswerSchemaExhausted` is a
    /// `FailureMode`).
    #[test]
    fn exit_code_answer_schema_exhausted_is_20() {
        let outcome = LoopOutcome::Finished(Disposition::Failed {
            mode: FailureMode::AnswerSchemaExhausted,
            summary: "x".into(),
        });
        assert_eq!(
            exit_code(&outcome),
            20,
            "AnswerSchemaExhausted must ride 20 under Finished(Failed{{..}})"
        );
    }

    #[test]
    fn exit_code_stopped_without_finish_is_20() {
        assert_eq!(exit_code(&LoopOutcome::StoppedWithoutFinish), 20);
    }

    #[test]
    fn exit_code_max_iterations_is_20() {
        assert_eq!(exit_code(&LoopOutcome::MaxIterations), 20);
    }

    #[test]
    fn exit_code_budget_exhausted_is_20() {
        assert_eq!(
            exit_code(&LoopOutcome::BudgetExhausted {
                summary: "x".into()
            }),
            20,
            "BudgetExhausted must map to exit code 20"
        );
    }

    #[test]
    fn exit_code_backend_error_is_1() {
        let outcome = LoopOutcome::BackendError(BackendError::Transient {
            kind: TransientKind::Network,
            retry_after: None,
        });
        assert_eq!(exit_code(&outcome), 1, "BackendError must be 1, never 20");
    }

    // ---- outcome_str: all 5 literals -----------------------------------

    #[test]
    fn outcome_str_covers_all_five_literals() {
        assert_eq!(
            outcome_str(&LoopOutcome::Finished(Disposition::Done {
                summary: String::new(),
                verification: Verification::NoChecksConfigured,
                change: ChangeEvidence::default(),
            })),
            "Finished"
        );
        assert_eq!(
            outcome_str(&LoopOutcome::StoppedWithoutFinish),
            "StoppedWithoutFinish"
        );
        assert_eq!(outcome_str(&LoopOutcome::MaxIterations), "MaxIterations");
        assert_eq!(
            outcome_str(&LoopOutcome::BudgetExhausted {
                summary: "wall-clock budget exhausted".into()
            }),
            "BudgetExhausted"
        );
        assert_eq!(
            outcome_str(&LoopOutcome::BackendError(BackendError::Terminal {
                kind: TerminalKind::Auth,
                message: "bad key".into(),
            })),
            "BackendError"
        );
    }

    // ---- transcript_label ------------------------------------------------

    /// The label is a PURE function of the resolved `BackendSettings` (no
    /// env): non-Ollama kinds flow through `model_label()` unchanged — even
    /// when the ollama-ish fields are populated — and Ollama settings get
    /// the resolved `think`/`num_ctx` appended (`unset` when `None`).
    #[test]
    fn transcript_label_covers_the_pinned_tuples() {
        let anthropic = BackendSettings {
            kind: BackendKind::Anthropic,
            model: "claude-haiku-4-5".to_string(),
            think: None,
            num_ctx: None,
            num_ctx_source: None,
            max_tokens: None,
            max_tokens_source: None,
        };
        assert_eq!(transcript_label(&anthropic), "claude-haiku-4-5");

        let bedrock = |think: Option<&str>, num_ctx: Option<u32>, num_ctx_source: Option<&str>| {
            BackendSettings {
                kind: BackendKind::Bedrock,
                model: "claude-haiku-4-5".to_string(),
                think: think.map(str::to_string),
                num_ctx,
                num_ctx_source: num_ctx_source.map(str::to_string),
                max_tokens: None,
                max_tokens_source: None,
            }
        };
        assert_eq!(
            transcript_label(&bedrock(None, None, None)),
            "bedrock:claude-haiku-4-5",
            "a non-ollama label is unchanged"
        );
        assert_eq!(
            transcript_label(&bedrock(Some("on"), Some(65536), Some("explicit"))),
            "bedrock:claude-haiku-4-5",
            "a non-ollama label is unchanged regardless of populated ollama-ish fields"
        );

        let ollama = |think: Option<&str>, num_ctx: Option<u32>| BackendSettings {
            kind: BackendKind::Ollama,
            model: "x".to_string(),
            think: think.map(str::to_string),
            num_ctx,
            num_ctx_source: num_ctx.is_some().then(|| "explicit".to_string()),
            max_tokens: None,
            max_tokens_source: None,
        };
        assert_eq!(
            transcript_label(&ollama(None, None)),
            "ollama:x think=unset num_ctx=unset"
        );
        assert_eq!(
            transcript_label(&ollama(Some("on"), Some(65536))),
            "ollama:x think=on num_ctx=65536"
        );
    }

    // ---- resolve_transcript_path: flag absent / bare / explicit ----------

    #[test]
    fn resolve_transcript_path_absent_flag_yields_none() {
        let state_dir = PathBuf::from("/nonexistent/state-dir");
        assert_eq!(
            resolve_transcript_path(None, &state_dir),
            None,
            "no --transcript flag must resolve to no transcript at all"
        );
    }

    #[test]
    fn resolve_transcript_path_bare_flag_joins_state_dir() {
        let state_dir = PathBuf::from("/nonexistent/state-dir");
        assert_eq!(
            resolve_transcript_path(Some(None), &state_dir),
            Some(state_dir.join("transcript.jsonl")),
            "bare --transcript must default to transcript.jsonl in the state dir"
        );
    }

    #[test]
    fn resolve_transcript_path_explicit_path_is_used_verbatim() {
        let state_dir = PathBuf::from("/nonexistent/state-dir");
        let explicit = PathBuf::from("rel/x.jsonl");
        assert_eq!(
            resolve_transcript_path(Some(Some(explicit.clone())), &state_dir),
            Some(explicit),
            "an explicit --transcript path must flow through verbatim, not joined to state_dir"
        );
    }

    // ---- gate_command → ChecksRunner wiring ----------------------------

    fn dummy_root() -> PathBuf {
        PathBuf::from("/")
    }

    #[test]
    fn non_empty_gate_command_produces_sh_runner() {
        let runner = build_checks_runner("cargo nextest run", dummy_root(), 60)
            .expect("non-empty gate_command must yield Some(runner)");
        assert_eq!(runner.command().program, "/bin/sh");
        assert_eq!(runner.command().args, vec!["-c", "cargo nextest run"]);
    }

    #[test]
    fn non_empty_gate_command_registry_has_run_checks() {
        use harness::tools::standard_registry;
        let runner = build_checks_runner("cargo test", dummy_root(), 60).unwrap();
        let registry = standard_registry(Some(runner));
        assert!(
            registry.get("run_checks").is_some(),
            "registry must contain run_checks when gate_command is non-empty"
        );
    }

    #[test]
    fn empty_gate_command_produces_no_runner() {
        assert!(
            build_checks_runner("", dummy_root(), 60).is_none(),
            "empty gate_command must yield None"
        );
    }

    #[test]
    fn whitespace_only_gate_command_is_treated_as_empty() {
        assert!(
            build_checks_runner("   ", dummy_root(), 60).is_none(),
            "whitespace-only gate_command must be treated as empty (no ChecksRunner)"
        );
    }

    // ---- backend_from_env: error branches + model_label ---------------

    fn env_with<'a>(vars: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k: &str| {
            vars.iter()
                .find(|(key, _)| *key == k)
                .map(|(_, v)| (*v).to_string())
        }
    }

    #[tokio::test]
    async fn backend_from_env_defaults_to_anthropic_when_unset() {
        let env = env_with(&[("ANTHROPIC_API_KEY", "sk-test")]);
        let (backend, settings) = backend_from_env(&env).await.expect("must succeed");
        assert!(matches!(backend, Backend::Anthropic(_)));
        // model_label is the model id verbatim (default)
        assert_eq!(settings.model_label(), "claude-haiku-4-5");
        // No thinking knob for Anthropic this cut: the record stays closed.
        assert_eq!(
            settings,
            BackendSettings {
                kind: BackendKind::Anthropic,
                model: "claude-haiku-4-5".to_string(),
                think: None,
                num_ctx: None,
                num_ctx_source: None,
                max_tokens: None,
                max_tokens_source: None
            }
        );
    }

    // ---- TALOS_BEDROCK precedence ----------------------------------------

    #[tokio::test]
    async fn talos_bedrock_selects_bedrock_with_no_other_vars() {
        let env = env_with(&[("TALOS_BEDROCK", "1")]);
        let (backend, settings) = backend_from_env(&env).await.expect("must succeed");
        assert!(matches!(backend, Backend::Bedrock(_)));
        // The CANONICAL model name is recorded — NOT the inference-profile
        // id the backend was actually built with.
        assert_eq!(settings.kind, BackendKind::Bedrock);
        assert_eq!(settings.model, "claude-haiku-4-5");
        assert_eq!(settings.model_label(), "bedrock:claude-haiku-4-5");
        assert_eq!(settings.think, None);
        assert_eq!(settings.num_ctx, None);
        assert_eq!(settings.num_ctx_source, None);
    }

    #[tokio::test]
    async fn talos_bedrock_wins_over_ollama_when_both_set() {
        let env = env_with(&[
            ("TALOS_BEDROCK", "1"),
            ("TALOS_BACKEND", "ollama"),
            ("OLLAMA_MODEL", "qwen3:32b"),
        ]);
        let (backend, label_settings) = backend_from_env(&env).await.expect("must succeed");
        assert!(matches!(backend, Backend::Bedrock(_)));
        assert_eq!(label_settings.model_label(), "bedrock:claude-haiku-4-5");
    }

    #[tokio::test]
    async fn talos_bedrock_empty_falls_through_to_anthropic() {
        let env = env_with(&[("TALOS_BEDROCK", ""), ("ANTHROPIC_API_KEY", "sk-test")]);
        let (backend, _settings) = backend_from_env(&env).await.expect("must succeed");
        assert!(matches!(backend, Backend::Anthropic(_)));
    }

    #[tokio::test]
    async fn talos_bedrock_whitespace_only_falls_through_to_anthropic() {
        let env = env_with(&[("TALOS_BEDROCK", "   "), ("ANTHROPIC_API_KEY", "sk-test")]);
        let (backend, _settings) = backend_from_env(&env).await.expect("must succeed");
        assert!(
            matches!(backend, Backend::Anthropic(_)),
            "whitespace-only TALOS_BEDROCK must NOT select Bedrock"
        );
    }

    #[tokio::test]
    async fn talos_bedrock_unmapped_model_is_err() {
        let env = env_with(&[("TALOS_BEDROCK", "1"), ("ANTHROPIC_MODEL", "claude-3-opus")]);
        assert!(
            backend_from_env(&env).await.is_err(),
            "unmapped ANTHROPIC_MODEL under TALOS_BEDROCK must be Err"
        );
    }

    #[tokio::test]
    async fn talos_bedrock_explicit_model_label() {
        let env = env_with(&[
            ("TALOS_BEDROCK", "1"),
            ("ANTHROPIC_MODEL", "claude-sonnet-5"),
        ]);
        let (backend, settings) = backend_from_env(&env).await.expect("must succeed");
        assert!(matches!(backend, Backend::Bedrock(_)));
        assert_eq!(settings.model, "claude-sonnet-5");
        assert_eq!(settings.model_label(), "bedrock:claude-sonnet-5");
    }

    #[tokio::test]
    async fn backend_from_env_anthropic_explicit() {
        let env = env_with(&[
            ("TALOS_BACKEND", "anthropic"),
            ("ANTHROPIC_API_KEY", "sk-xyz"),
            ("ANTHROPIC_MODEL", "claude-sonnet-5"),
        ]);
        let (backend, settings) = backend_from_env(&env).await.expect("must succeed");
        assert!(matches!(backend, Backend::Anthropic(_)));
        assert_eq!(
            settings.model, "claude-sonnet-5",
            "model_label must be the model id verbatim"
        );
        assert_eq!(settings.model_label(), "claude-sonnet-5");
    }

    #[tokio::test]
    async fn backend_from_env_missing_anthropic_api_key_is_err() {
        let env = env_with(&[]); // ANTHROPIC_API_KEY absent
        assert!(
            backend_from_env(&env).await.is_err(),
            "missing ANTHROPIC_API_KEY must be Err"
        );
    }

    #[tokio::test]
    async fn backend_from_env_unknown_backend_is_err() {
        let env = env_with(&[("TALOS_BACKEND", "gemini")]);
        assert!(
            backend_from_env(&env).await.is_err(),
            "unknown TALOS_BACKEND must be Err"
        );
    }

    #[tokio::test]
    async fn backend_from_env_ollama_model_label_prefix() {
        let env = env_with(&[
            ("TALOS_BACKEND", "ollama"),
            ("OLLAMA_MODEL", "qwen3:32b"),
            ("OLLAMA_BASE_URL", "https://ollama.com"),
        ]);
        let (backend, settings) = backend_from_env(&env).await.expect("must succeed");
        assert!(matches!(backend, Backend::Ollama(_)));
        // The recorded model is VERBATIM (no prefix) — the label carries it.
        assert_eq!(settings.model, "qwen3:32b");
        assert_eq!(settings.model_label(), "ollama:qwen3:32b");
    }

    #[tokio::test]
    async fn backend_from_env_ollama_missing_model_is_err() {
        let env = env_with(&[("TALOS_BACKEND", "ollama")]);
        assert!(
            backend_from_env(&env).await.is_err(),
            "OLLAMA_MODEL must be required for ollama"
        );
    }

    #[tokio::test]
    async fn backend_from_env_bad_ollama_think_is_err() {
        let env = env_with(&[
            ("TALOS_BACKEND", "ollama"),
            ("OLLAMA_MODEL", "some-model"),
            ("OLLAMA_BASE_URL", "https://ollama.com"),
            ("OLLAMA_THINK", "turbo"),
        ]);
        assert!(
            backend_from_env(&env).await.is_err(),
            "invalid OLLAMA_THINK must be Err"
        );
    }

    #[tokio::test]
    async fn backend_from_env_bad_ollama_num_ctx_is_err() {
        let env = env_with(&[
            ("TALOS_BACKEND", "ollama"),
            ("OLLAMA_MODEL", "some-model"),
            ("OLLAMA_NUM_CTX", "not-a-number"),
        ]);
        let Err(e) = backend_from_env(&env).await else {
            panic!("non-u32 OLLAMA_NUM_CTX must be Err")
        };
        assert!(
            e.contains("OLLAMA_NUM_CTX"),
            "parse error must name OLLAMA_NUM_CTX: {e}"
        );
    }

    /// Round-trip that locks `ThinkLevel::as_str` and the production `OLLAMA_THINK`
    /// parser together: every level round-trips to the same spelling the
    /// record stores.
    #[tokio::test]
    async fn backend_from_env_all_ollama_think_values_accepted() {
        for level in [
            ThinkLevel::Off,
            ThinkLevel::On,
            ThinkLevel::Low,
            ThinkLevel::Medium,
            ThinkLevel::High,
            ThinkLevel::Max,
        ] {
            let vars = [
                ("TALOS_BACKEND", "ollama"),
                ("OLLAMA_MODEL", "m"),
                ("OLLAMA_BASE_URL", "https://ollama.com"),
                ("OLLAMA_THINK", level.as_str()),
            ];
            let env = env_with(&vars);
            let Ok((_backend, settings)) = backend_from_env(&env).await else {
                panic!("OLLAMA_THINK={level:?} must be accepted")
            };
            assert_eq!(
                settings.think,
                Some(level.as_str().to_string()),
                "the record must carry the env spelling of {level:?}"
            );
        }
    }

    // ---- backend_from_env: num_ctx resolution (hermetic) ----------------

    // ---- backend_from_env: num_ctx resolution via POST /api/show ---------
    //
    // `wiremock` is NOT a dev-dependency of this crate (and must not be
    // added), and the workspace tokio features omit `net` — so the fake
    // `/api/show` daemon below is a raw `std::net::TcpListener` thread.

    /// Bind `127.0.0.1:0` and serve ONE `POST /api/show` request with
    /// `model_info` as the response body. Returns the port and a channel
    /// that yields the FULL request head (request line + every header, up
    /// to the blank line) once the connection completes — the head is what
    /// the authorization-header assertions below read.
    fn fake_show_daemon(
        model_info: &serde_json::Value,
    ) -> (u16, std::sync::mpsc::Receiver<String>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("local addr").port();
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let body = format!("{{\"model_info\":{model_info}}}");
        std::thread::spawn(move || {
            use std::io::{Read as _, Write as _};
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buf: Vec<u8> = Vec::new();
            let mut one = [0u8; 1];
            while !buf.ends_with(b"\r\n\r\n") {
                match stream.read(&mut one) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => buf.push(one[0]),
                }
            }
            let head = String::from_utf8_lossy(&buf).trim_end().to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            stream
                .write_all(response.as_bytes())
                .expect("write response");
            let _ = tx.send(head);
        });
        (port, rx)
    }

    // (i) unset + local → the probe FIRES from talos and the advertised
    // 262144 is pinned, recorded with source `probe`.
    #[tokio::test]
    async fn backend_from_env_ollama_local_unset_probes_api_show() {
        let (port, head_rx) = fake_show_daemon(&serde_json::json!({
            "general.architecture": "qwen35",
            "qwen35.context_length": 262_144
        }));
        let base_url = format!("http://127.0.0.1:{port}");
        let vars = [
            ("TALOS_BACKEND", "ollama"),
            ("OLLAMA_MODEL", "m"),
            ("OLLAMA_BASE_URL", &base_url),
        ];
        let env = env_with(&vars);
        let (backend, settings) = backend_from_env(&env).await.expect("must succeed");
        assert!(matches!(backend, Backend::Ollama(_)));
        assert_eq!(settings.num_ctx, Some(262_144));
        assert_eq!(settings.num_ctx_source.as_deref(), Some("probe"));
        let head = head_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("server must have served the probe");
        assert_eq!(
            head.lines().next().unwrap_or_default(),
            "POST /api/show HTTP/1.1",
            "the probe must be a POST /api/show request: {head}"
        );
    }

    // (ii) an explicit `OLLAMA_NUM_CTX` wins verbatim — NO probe fires, so
    // the fake daemon (which would answer anything) must stay untouched.
    #[tokio::test]
    async fn backend_from_env_ollama_explicit_num_ctx_skips_the_probe() {
        let (port, head_rx) = fake_show_daemon(&serde_json::json!({
            "general.architecture": "qwen35",
            "qwen35.context_length": 262_144
        }));
        let base_url = format!("http://127.0.0.1:{port}");
        let vars = [
            ("TALOS_BACKEND", "ollama"),
            ("OLLAMA_MODEL", "m"),
            ("OLLAMA_BASE_URL", &base_url),
            ("OLLAMA_NUM_CTX", "65536"),
        ];
        let env = env_with(&vars);
        let (backend, settings) = backend_from_env(&env).await.expect("must succeed");
        assert!(matches!(backend, Backend::Ollama(_)));
        assert_eq!(settings.num_ctx, Some(65_536));
        assert_eq!(settings.num_ctx_source.as_deref(), Some("explicit"));
        assert!(
            head_rx.try_recv().is_err(),
            "an explicit value must not probe /api/show"
        );
    }

    // (iii) a below-floor advertised value is DATA, not a failure — the run
    // proceeds with the verbatim advertised value.
    #[tokio::test]
    async fn backend_from_env_ollama_local_probe_below_floor_still_ok() {
        let (port, _head_rx) = fake_show_daemon(&serde_json::json!({
            "general.architecture": "qwen35",
            "qwen35.context_length": 8_192
        }));
        let base_url = format!("http://127.0.0.1:{port}");
        let vars = [
            ("TALOS_BACKEND", "ollama"),
            ("OLLAMA_MODEL", "m"),
            ("OLLAMA_BASE_URL", &base_url),
        ];
        let env = env_with(&vars);
        let (backend, settings) = backend_from_env(&env).await.expect("below floor is Ok");
        assert!(matches!(backend, Backend::Ollama(_)));
        assert_eq!(settings.num_ctx, Some(8_192));
        assert_eq!(settings.num_ctx_source.as_deref(), Some("probe"));
    }

    // (iv) probe connection-refused → fail-loud Err naming model + host.
    #[tokio::test]
    async fn backend_from_env_ollama_local_probe_refused_is_err() {
        // Capture a port, then DROP the listener so the probe is refused.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("local addr").port();
        drop(listener);
        let base_url = format!("http://127.0.0.1:{port}");
        let vars = [
            ("TALOS_BACKEND", "ollama"),
            ("OLLAMA_MODEL", "m"),
            ("OLLAMA_BASE_URL", &base_url),
        ];
        let env = env_with(&vars);
        let Err(e) = backend_from_env(&env).await else {
            panic!("probe failure must be Err")
        };
        assert!(e.contains("127.0.0.1"), "error must name the host: {e}");
        assert!(e.contains("`m`"), "error must name the model: {e}");
    }

    // (v) a non-local base URL makes no request at all: no `num_ctx`, no
    // source. (There is nothing to observe a request against — the point
    // is that no local daemon is ever involved.)
    #[tokio::test]
    async fn backend_from_env_ollama_nonlocal_makes_no_request() {
        let env = env_with(&[
            ("TALOS_BACKEND", "ollama"),
            ("OLLAMA_MODEL", "m"),
            ("OLLAMA_BASE_URL", "https://ollama.com"),
        ]);
        let (backend, settings) = backend_from_env(&env).await.expect("must succeed");
        assert!(matches!(backend, Backend::Ollama(_)));
        assert_eq!(settings.num_ctx, None);
        assert_eq!(settings.num_ctx_source, None);
    }

    // An empty/whitespace `OLLAMA_NUM_CTX` is UNSET (resolver branch 0), so
    // a local base URL falls through to the probe — exactly one request.
    #[tokio::test]
    async fn backend_from_env_ollama_whitespace_num_ctx_is_unset_and_probes() {
        let (port, head_rx) = fake_show_daemon(&serde_json::json!({
            "general.architecture": "qwen35",
            "qwen35.context_length": 262_144
        }));
        let base_url = format!("http://127.0.0.1:{port}");
        let vars = [
            ("TALOS_BACKEND", "ollama"),
            ("OLLAMA_MODEL", "m"),
            ("OLLAMA_BASE_URL", &base_url),
            ("OLLAMA_NUM_CTX", "   "),
        ];
        let env = env_with(&vars);
        let (backend, settings) = backend_from_env(&env).await.expect("must succeed");
        assert!(matches!(backend, Backend::Ollama(_)));
        assert_eq!(settings.num_ctx, Some(262_144));
        assert_eq!(
            settings.num_ctx_source.as_deref(),
            Some("probe"),
            "whitespace-only OLLAMA_NUM_CTX must fall through to the probe"
        );
        let head = head_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the daemon must have served exactly one request");
        assert!(head.starts_with("POST /api/show HTTP/1.1"));
        assert!(
            head_rx.try_recv().is_err(),
            "the probe must fire exactly once"
        );
    }

    // `OLLAMA_THINK` is validated BEFORE the `num_ctx` resolution, so an
    // invalid value can never trigger a `/api/show` probe.
    #[tokio::test]
    async fn backend_from_env_ollama_bad_think_fails_before_any_probe() {
        let (port, head_rx) = fake_show_daemon(&serde_json::json!({
            "general.architecture": "qwen35",
            "qwen35.context_length": 262_144
        }));
        let base_url = format!("http://127.0.0.1:{port}");
        let vars = [
            ("TALOS_BACKEND", "ollama"),
            ("OLLAMA_MODEL", "m"),
            ("OLLAMA_BASE_URL", &base_url),
            ("OLLAMA_THINK", "bogus"),
        ];
        let env = env_with(&vars);
        let Err(e) = backend_from_env(&env).await else {
            panic!("invalid OLLAMA_THINK must be Err")
        };
        assert!(e.contains("OLLAMA_THINK"), "error must name the var: {e}");
        assert!(
            head_rx.try_recv().is_err(),
            "an invalid think level must not trigger a probe"
        );
    }

    // An EMPTY `OLLAMA_API_KEY` is not a bearer token: the probe request
    // carries no `authorization` header at all.
    #[tokio::test]
    async fn backend_from_env_ollama_empty_api_key_sends_no_authorization_header() {
        let (port, head_rx) = fake_show_daemon(&serde_json::json!({
            "general.architecture": "qwen35",
            "qwen35.context_length": 262_144
        }));
        let base_url = format!("http://127.0.0.1:{port}");
        let vars = [
            ("TALOS_BACKEND", "ollama"),
            ("OLLAMA_MODEL", "m"),
            ("OLLAMA_BASE_URL", &base_url),
            ("OLLAMA_API_KEY", ""),
        ];
        let env = env_with(&vars);
        let (backend, settings) = backend_from_env(&env).await.expect("must succeed");
        assert!(matches!(backend, Backend::Ollama(_)));
        assert_eq!(settings.num_ctx_source.as_deref(), Some("probe"));
        let head = head_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the daemon must have served the probe");
        assert!(
            !head
                .lines()
                .any(|l| l.to_ascii_lowercase().starts_with("authorization:")),
            "an empty key must not be sent as a bearer token: {head}"
        );
    }

    // A non-empty `OLLAMA_API_KEY` is sent as the probe credential.
    #[tokio::test]
    async fn backend_from_env_ollama_api_key_is_sent_as_bearer_on_the_probe() {
        let (port, head_rx) = fake_show_daemon(&serde_json::json!({
            "general.architecture": "qwen35",
            "qwen35.context_length": 262_144
        }));
        let base_url = format!("http://127.0.0.1:{port}");
        let vars = [
            ("TALOS_BACKEND", "ollama"),
            ("OLLAMA_MODEL", "m"),
            ("OLLAMA_BASE_URL", &base_url),
            ("OLLAMA_API_KEY", "k"),
        ];
        let env = env_with(&vars);
        let (_backend, _settings) = backend_from_env(&env).await.expect("must succeed");
        let head = head_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the daemon must have served the probe");
        assert!(
            head.lines().any(|l| l == "authorization: Bearer k"),
            "the probe must carry the filtered api key as a bearer token: {head}"
        );
    }

    /// Unset + a non-local base URL: no `num_ctx` at all (Ollama's own
    /// default applies) and `num_ctx_source` is exactly `None`.
    #[tokio::test]
    async fn backend_from_env_ollama_cloud_unset_num_ctx_is_none() {
        let env = env_with(&[
            ("TALOS_BACKEND", "ollama"),
            ("OLLAMA_MODEL", "x"),
            ("OLLAMA_BASE_URL", "https://ollama.com"),
        ]);
        let (backend, settings) = backend_from_env(&env).await.expect("must succeed");
        assert!(matches!(backend, Backend::Ollama(_)));
        assert_eq!(settings.think, None);
        assert_eq!(settings.num_ctx, None);
        assert_eq!(settings.num_ctx_source, None);
    }

    /// An explicit `OLLAMA_NUM_CTX` wins verbatim, recorded with source
    /// `explicit` — on a non-local base URL, so no other branch is involved.
    #[tokio::test]
    async fn backend_from_env_ollama_explicit_num_ctx_records_env_source() {
        let env = env_with(&[
            ("TALOS_BACKEND", "ollama"),
            ("OLLAMA_MODEL", "x"),
            ("OLLAMA_BASE_URL", "https://ollama.com"),
            ("OLLAMA_NUM_CTX", "65536"),
        ]);
        let (backend, settings) = backend_from_env(&env).await.expect("must succeed");
        assert!(matches!(backend, Backend::Ollama(_)));
        assert_eq!(settings.num_ctx, Some(65536));
        assert_eq!(settings.num_ctx_source.as_deref(), Some("explicit"));
    }

    /// `OLLAMA_THINK=high` records the env spelling alongside an explicit
    /// `num_ctx` — the SAME values the backend was constructed with.
    #[tokio::test]
    async fn backend_from_env_ollama_think_high_records_resolved_settings() {
        let env = env_with(&[
            ("TALOS_BACKEND", "ollama"),
            ("OLLAMA_MODEL", "x"),
            ("OLLAMA_BASE_URL", "https://ollama.com"),
            ("OLLAMA_THINK", "high"),
            ("OLLAMA_NUM_CTX", "32768"),
        ]);
        let (backend, settings) = backend_from_env(&env).await.expect("must succeed");
        assert!(matches!(backend, Backend::Ollama(_)));
        assert_eq!(settings.think.as_deref(), Some("high"));
        assert_eq!(settings.num_ctx, Some(32768));
        assert_eq!(settings.num_ctx_source.as_deref(), Some("explicit"));
    }

    /// An empty `OLLAMA_NUM_CTX` is UNSET: no `num_ctx`, no source.
    #[tokio::test]
    async fn backend_from_env_ollama_empty_num_ctx_is_unset() {
        let env = env_with(&[
            ("TALOS_BACKEND", "ollama"),
            ("OLLAMA_MODEL", "m"),
            ("OLLAMA_BASE_URL", "https://ollama.com"),
            ("OLLAMA_NUM_CTX", ""),
        ]);
        let (backend, settings) = backend_from_env(&env).await.expect("empty is unset");
        assert!(matches!(backend, Backend::Ollama(_)));
        assert_eq!(settings.model_label(), "ollama:m");
        assert_eq!(settings.num_ctx, None);
        assert_eq!(settings.num_ctx_source, None);
    }

    // ---- num_ctx_stderr_line: structured stderr shape --------------------

    // Assert on the PARSED value — serde_json sorts keys, so raw-string
    // assertions would be order-fragile.
    #[test]
    fn num_ctx_stderr_line_explicit_shape() {
        let r = harness::ollama::NumCtxResolution {
            value: Some(65536),
            source: harness::ollama::NumCtxSource::Explicit,
            desc: "num_ctx=65536 (explicit OLLAMA_NUM_CTX)".to_string(),
            warning: None,
        };
        let v: serde_json::Value =
            serde_json::from_str(&num_ctx_stderr_line(&r)).expect("must be JSON");
        assert_eq!(v["num_ctx"]["source"], "explicit");
        assert_eq!(v["num_ctx"]["value"], 65536);
        assert_eq!(
            v["num_ctx"]["desc"],
            "num_ctx=65536 (explicit OLLAMA_NUM_CTX)"
        );
        assert!(v["num_ctx"]["warning"].is_null());
    }

    #[test]
    fn num_ctx_stderr_line_probe_warning_shape() {
        let warning = format!(
            "WARNING: resolved num_ctx=8192 for `m` (arch=qwen35, key=qwen35.context_length) \
             is BELOW the {} sanity floor; the run may be truncation-invalid — \
             set OLLAMA_NUM_CTX to override",
            harness::ollama::MIN_EXPECTED_NUM_CTX
        );
        let r = harness::ollama::NumCtxResolution {
            value: Some(8192),
            source: harness::ollama::NumCtxSource::Probe,
            desc: "num_ctx=8192 (resolved, BELOW-FLOOR: arch=qwen35 key=qwen35.context_length)"
                .to_string(),
            warning: Some(warning.clone()),
        };
        let v: serde_json::Value =
            serde_json::from_str(&num_ctx_stderr_line(&r)).expect("must be JSON");
        assert_eq!(v["num_ctx"]["source"], "probe");
        assert_eq!(v["num_ctx"]["value"], 8192);
        assert_eq!(v["num_ctx"]["warning"], warning);
    }

    // The record's `num_ctx_source` follows the VALUE, never the source
    // variant: `Default` pins no value, so `"default"` can never reach a
    // run record (it lives only in the stderr provenance line).
    #[test]
    fn num_ctx_source_for_record_follows_value_not_variant() {
        let resolution = |value: Option<u32>, source: harness::ollama::NumCtxSource, desc: &str| {
            harness::ollama::NumCtxResolution {
                value,
                source,
                desc: desc.to_string(),
                warning: None,
            }
        };
        let explicit = resolution(
            Some(65_536),
            harness::ollama::NumCtxSource::Explicit,
            "num_ctx=65536 (explicit OLLAMA_NUM_CTX)",
        );
        assert_eq!(
            num_ctx_source_for_record(&explicit),
            Some("explicit".into())
        );
        let probe = resolution(
            Some(262_144),
            harness::ollama::NumCtxSource::Probe,
            "num_ctx=262144 (resolved: arch=qwen35 key=qwen35.context_length)",
        );
        assert_eq!(num_ctx_source_for_record(&probe), Some("probe".into()));
        let default = resolution(
            None,
            harness::ollama::NumCtxSource::Default,
            "num_ctx=default",
        );
        assert_eq!(
            num_ctx_source_for_record(&default),
            None,
            "the Default source pins no value, so the record carries no source either"
        );
    }

    // ---- seed: byte-identical to renderer, never raw description -------

    fn sample_spec() -> TaskSpec {
        TaskSpec {
            title: "Test task".into(),
            description: "Test description.".into(),
            acceptance_criteria: vec!["AC one".into(), "AC two".into()],
            files_to_modify: vec![FileToModify {
                path: "src/lib.rs".into(),
                change: "do something".into(),
            }],
            gate_command: "cargo nextest run".into(),
        }
    }

    #[test]
    fn seed_byte_identical_to_renderer_not_raw_description() {
        let spec = sample_spec();
        let seed = make_run_seed(&spec);
        let expected = render_task_prompt_from_spec(&spec);
        assert_eq!(
            seed.as_bytes(),
            expected.as_bytes(),
            "seed must be byte-identical to render_task_prompt_from_spec output"
        );
        // Guard: rendered output includes title/AC/files, so it cannot equal
        // the raw description field.
        assert_ne!(
            seed, spec.description,
            "seed must NOT be the raw description — the renderer must be used"
        );
    }

    // ---- summary: exact field set and outcome literals -----------------

    #[test]
    fn summary_exact_field_set() {
        let settings = BackendSettings {
            kind: BackendKind::Ollama,
            model: "x".to_string(),
            think: Some("high".to_string()),
            num_ctx: Some(32768),
            num_ctx_source: Some("explicit".to_string()),
            max_tokens: None,
            max_tokens_source: None,
        };
        let summary = build_run_summary(
            "BackendError",
            Disposition::Failed {
                mode: FailureMode::TransientInfra,
                summary: "conn refused".into(),
            },
            "my-task:1".into(),
            "/tmp/run.sqlite".into(),
            3,
            0,
            0,
            settings,
        );
        let json = serde_json::to_value(&summary).expect("summary must serialize");
        let obj = json.as_object().expect("must be object");
        let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec![
                "backend_settings",
                "compactions",
                "disposition",
                "highest_compaction_tier",
                "iterations",
                "outcome",
                "record_path",
                "run_id"
            ],
            "summary must have exactly the eight expected fields"
        );
        // The structured settings carry the kind through to stdout verbatim.
        assert_eq!(
            obj.get("backend_settings")
                .and_then(|s| s.get("kind"))
                .and_then(serde_json::Value::as_str),
            Some("Ollama"),
            "backend_settings.kind must be the bare externally-tagged string"
        );
        assert_eq!(
            obj.get("outcome").and_then(serde_json::Value::as_str),
            Some("BackendError")
        );
        assert_eq!(
            obj.get("run_id").and_then(serde_json::Value::as_str),
            Some("my-task:1")
        );
        assert_eq!(
            obj.get("iterations").and_then(serde_json::Value::as_u64),
            Some(3)
        );
    }

    #[test]
    fn summary_outcome_covers_all_five_literals() {
        // Verify all five LoopOutcome discriminants appear in outcome_str.
        let cases: &[(&'static str, LoopOutcome)] = &[
            (
                "Finished",
                LoopOutcome::Finished(Disposition::Done {
                    summary: String::new(),
                    verification: Verification::NoChecksConfigured,
                    change: ChangeEvidence::default(),
                }),
            ),
            ("StoppedWithoutFinish", LoopOutcome::StoppedWithoutFinish),
            ("MaxIterations", LoopOutcome::MaxIterations),
            (
                "BudgetExhausted",
                LoopOutcome::BudgetExhausted {
                    summary: "wall-clock budget exhausted".into(),
                },
            ),
            (
                "BackendError",
                LoopOutcome::BackendError(BackendError::Terminal {
                    kind: TerminalKind::Auth,
                    message: "x".into(),
                }),
            ),
        ];
        for (expected, outcome) in cases {
            assert_eq!(outcome_str(outcome), *expected);
        }
    }

    // ---- RunSummary is serializable ------------------------------------

    #[test]
    fn run_summary_serializes_disposition() {
        let summary: RunSummary = build_run_summary(
            "Finished",
            Disposition::Done {
                summary: "all green".into(),
                verification: Verification::NoChecksConfigured,
                change: ChangeEvidence::default(),
            },
            "task:1".into(),
            "/state/run.sqlite".into(),
            5,
            0,
            0,
            default_settings(),
        );
        let json = serde_json::to_value(&summary).expect("must serialize");
        assert!(
            json.get("disposition").is_some(),
            "disposition field must be present"
        );
    }

    /// AC8 — a Truncated disposition reaches stdout unchanged through the
    /// production `into_disposition()` pass-through (its `Finished` arm
    /// returns the disposition verbatim), so the JSON the dispatch worker
    /// parses names the truncation.
    #[test]
    fn run_summary_serializes_truncated_disposition() {
        let outcome = LoopOutcome::Finished(Disposition::Failed {
            mode: FailureMode::Truncated,
            summary: "turn truncated at max_tokens (produced 111 of 32768 \
                      output-token cap) before any tool call; raise --max-tokens"
                .into(),
        });
        // The production caller's verbatim pass-through (main.rs builds the
        // summary from `result.outcome.into_disposition()`).
        let summary: RunSummary = build_run_summary(
            outcome_str(&outcome),
            outcome.into_disposition(),
            "task:1".into(),
            "/state/run.sqlite".into(),
            1,
            0,
            0,
            default_settings(),
        );
        let json = serde_json::to_string(&summary).expect("must serialize");
        assert!(
            json.contains("Truncated"),
            "serialized summary must name the Truncated mode; got {json}"
        );
        assert!(
            json.contains("raise --max-tokens"),
            "serialized summary must carry the remedy; got {json}"
        );
    }

    // ---- ChecksRunner.clone() used for both registry and config --------
    #[test]
    fn checks_runner_can_be_cloned_for_dual_wiring() {
        let runner =
            build_checks_runner("cargo test", dummy_root(), 30).expect("non-empty yields Some");
        // clone() is required to wire the same runner to both
        // standard_registry(Some(..)) AND RunConfig::with_checks(..).
        let _clone = runner.clone();
        // If this compiles and runs, ChecksRunner is Clone. ✓
        assert_eq!(runner.command().program, "/bin/sh");
    }

    // ---- wall_clock_secs: flag > TALOS_WALL_CLOCK_SECS env >
    // DEFAULT_WALL_CLOCK_SECS (1500; 0 = unbounded) ----------------------

    #[test]
    fn wall_clock_secs_flag_beats_env() {
        let env = env_with(&[("TALOS_WALL_CLOCK_SECS", "999")]);
        assert_eq!(
            resolve_wall_clock_secs(Some(42), &env),
            42,
            "explicit flag must take precedence over env"
        );
    }

    #[test]
    fn wall_clock_secs_env_beats_default() {
        let env = env_with(&[("TALOS_WALL_CLOCK_SECS", "300")]);
        assert_eq!(
            resolve_wall_clock_secs(None, &env),
            300,
            "env must beat the default 1500"
        );
    }

    #[test]
    fn wall_clock_secs_both_unset_yields_default() {
        let env = env_with(&[]);
        assert_eq!(
            resolve_wall_clock_secs(None, &env),
            DEFAULT_WALL_CLOCK_SECS,
            "both-unset must yield the ARMED default 1500, not the unbounded \
             sentinel 0"
        );
    }

    #[test]
    fn wall_clock_secs_invalid_env_value_falls_back_to_default() {
        // A non-u64 env value must not panic — it falls through to the ARMED
        // default 1500 (a stated divergence from `resolve_compact_threshold_pct`'s
        // hard error: a typo'd env now lands on 1500 rather than unbounded).
        let env = env_with(&[("TALOS_WALL_CLOCK_SECS", "not-a-number")]);
        assert_eq!(
            resolve_wall_clock_secs(None, &env),
            DEFAULT_WALL_CLOCK_SECS,
            "invalid env value must fall back to the armed default 1500"
        );
    }

    #[test]
    fn wall_clock_secs_zero_env_is_the_disable_sentinel() {
        // `0` on the env path is the unbounded sentinel, passed through.
        let env = env_with(&[("TALOS_WALL_CLOCK_SECS", "0")]);
        assert_eq!(
            resolve_wall_clock_secs(None, &env),
            0,
            "env `0` must disable the budget (unbounded sentinel)"
        );
    }

    #[test]
    fn wall_clock_secs_explicit_zero_flag_beats_env() {
        let env = env_with(&[("TALOS_WALL_CLOCK_SECS", "300")]);
        assert_eq!(
            resolve_wall_clock_secs(Some(0), &env),
            0,
            "an explicit `--wall-clock-secs 0` disable must beat a numeric env"
        );
    }

    // ---- ralph_exit_code: all 7 arms -----------------------------------

    #[test]
    fn ralph_exit_code_stop_condition_met_is_0() {
        assert_eq!(
            ralph_exit_code(&RalphTerminal::StopConditionMet),
            0,
            "StopConditionMet must be 0 (objective met)"
        );
    }

    #[test]
    fn ralph_exit_code_stuck_is_20() {
        assert_eq!(
            ralph_exit_code(&RalphTerminal::Stuck),
            20,
            "Stuck is a task-side failure terminal → 20"
        );
    }

    #[test]
    fn ralph_exit_code_max_iterations_exhausted_is_20() {
        assert_eq!(
            ralph_exit_code(&RalphTerminal::MaxIterationsExhausted),
            20,
            "MaxIterationsExhausted is a task-side failure terminal → 20"
        );
    }

    #[test]
    fn ralph_exit_code_time_budget_exhausted_is_20() {
        assert_eq!(
            ralph_exit_code(&RalphTerminal::TimeBudgetExhausted),
            20,
            "TimeBudgetExhausted is a task-side failure terminal → 20"
        );
    }

    #[test]
    fn ralph_exit_code_do_overs_exhausted_is_20() {
        assert_eq!(
            ralph_exit_code(&RalphTerminal::DoOversExhausted),
            20,
            "DoOversExhausted is a task-side failure terminal → 20"
        );
    }

    #[test]
    fn ralph_exit_code_error_is_1() {
        assert_eq!(
            ralph_exit_code(&RalphTerminal::Error("git commit failed".into())),
            1,
            "Error is a harness/infra failure → 1, never 20"
        );
    }

    #[test]
    fn ralph_exit_code_backend_errors_exhausted_is_1() {
        assert_eq!(
            ralph_exit_code(&RalphTerminal::BackendErrorsExhausted),
            1,
            "BackendErrorsExhausted is a sustained infra failure → 1, never 20"
        );
    }

    // ---- ralph_terminal_str: all 7 literals, no payload leak -----------

    #[test]
    fn ralph_terminal_str_covers_all_seven_literals() {
        assert_eq!(
            ralph_terminal_str(&RalphTerminal::StopConditionMet),
            "StopConditionMet"
        );
        assert_eq!(ralph_terminal_str(&RalphTerminal::Stuck), "Stuck");
        assert_eq!(
            ralph_terminal_str(&RalphTerminal::MaxIterationsExhausted),
            "MaxIterationsExhausted"
        );
        assert_eq!(
            ralph_terminal_str(&RalphTerminal::TimeBudgetExhausted),
            "TimeBudgetExhausted"
        );
        assert_eq!(
            ralph_terminal_str(&RalphTerminal::DoOversExhausted),
            "DoOversExhausted"
        );
        assert_eq!(
            ralph_terminal_str(&RalphTerminal::BackendErrorsExhausted),
            "BackendErrorsExhausted"
        );
    }

    #[test]
    fn ralph_terminal_str_error_does_not_leak_payload() {
        // `format!("{:?}")` is forbidden — it would leak the String payload.
        // The closed discriminant must be the bare "Error" regardless of the
        // payload contents.
        assert_eq!(
            ralph_terminal_str(&RalphTerminal::Error("payload with spaces".into())),
            "Error"
        );
        assert_eq!(
            ralph_terminal_str(&RalphTerminal::Error(String::new())),
            "Error"
        );
    }

    // ---- write_ralph_error_detail: stderr payload surface ---------------

    #[test]
    fn write_ralph_error_detail_error_writes_one_stderr_line() {
        let mut out: Vec<u8> = Vec::new();
        write_ralph_error_detail(
            &RalphTerminal::Error(
                "git status exited Some(128): fatal: not a git repository".into(),
            ),
            &mut out,
        )
        .expect("Vec<u8> writes are infallible");
        assert_eq!(
            out,
            b"talos ralph: error: git status exited Some(128): fatal: not a git repository\n"
        );
    }

    #[test]
    fn write_ralph_error_detail_non_error_terminals_write_nothing() {
        for terminal in [
            RalphTerminal::StopConditionMet,
            RalphTerminal::Stuck,
            RalphTerminal::MaxIterationsExhausted,
            RalphTerminal::TimeBudgetExhausted,
            RalphTerminal::DoOversExhausted,
            RalphTerminal::BackendErrorsExhausted,
        ] {
            let mut out: Vec<u8> = Vec::new();
            write_ralph_error_detail(&terminal, &mut out).expect("Vec<u8> writes are infallible");
            assert!(
                out.is_empty(),
                "non-Error terminal {terminal:?} must write zero bytes to stderr"
            );
        }
    }

    // ---- RalphSummary: exact field set ---------------------------------

    #[test]
    fn ralph_summary_exact_field_set() {
        let summary = build_ralph_summary("build the thing".into(), "Stuck", 7, 42, 3, 2);
        let json = serde_json::to_value(&summary).expect("RalphSummary must serialize");
        let obj = json.as_object().expect("must be object");
        let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec![
                "commit_rejects",
                "commit_retries_total",
                "objective",
                "outer_iterations",
                "terminal",
                "total_inner_iterations"
            ],
            "RalphSummary must have exactly the six expected fields (no run_id/record_path)"
        );
        assert_eq!(
            obj.get("objective").and_then(serde_json::Value::as_str),
            Some("build the thing")
        );
        assert_eq!(
            obj.get("terminal").and_then(serde_json::Value::as_str),
            Some("Stuck")
        );
        assert_eq!(
            obj.get("outer_iterations")
                .and_then(serde_json::Value::as_u64),
            Some(7)
        );
        assert_eq!(
            obj.get("total_inner_iterations")
                .and_then(serde_json::Value::as_u64),
            Some(42)
        );
        assert_eq!(
            obj.get("commit_retries_total")
                .and_then(serde_json::Value::as_u64),
            Some(3)
        );
        assert_eq!(
            obj.get("commit_rejects")
                .and_then(serde_json::Value::as_u64),
            Some(2)
        );
    }

    #[test]
    fn ralph_summary_is_serializable() {
        let summary: RalphSummary =
            build_ralph_summary("obj".into(), "StopConditionMet", 1, 0, 0, 0);
        let json = serde_json::to_string(&summary).expect("must serialize");
        assert!(
            serde_json::from_str::<serde_json::Value>(&json).is_ok(),
            "round-trips through JSON"
        );
    }

    // ---- resolve_ralph_wall_clock_secs: flag > env > 0 -----------------

    #[test]
    fn ralph_wall_clock_secs_flag_beats_env() {
        let env = env_with(&[("TALOS_RALPH_WALL_CLOCK_SECS", "999")]);
        assert_eq!(
            resolve_ralph_wall_clock_secs(Some(42), &env),
            42,
            "explicit flag must take precedence over env"
        );
    }

    #[test]
    fn ralph_wall_clock_secs_env_beats_default() {
        let env = env_with(&[("TALOS_RALPH_WALL_CLOCK_SECS", "300")]);
        assert_eq!(
            resolve_ralph_wall_clock_secs(None, &env),
            300,
            "env must beat the default 0"
        );
    }

    #[test]
    fn ralph_wall_clock_secs_both_unset_yields_zero() {
        let env = env_with(&[]);
        assert_eq!(
            resolve_ralph_wall_clock_secs(None, &env),
            0,
            "both-unset must yield the sentinel 0 (unbounded)"
        );
    }

    #[test]
    fn ralph_wall_clock_secs_invalid_env_value_falls_back_to_zero() {
        // A non-u64 env value must not panic — it falls through to default 0.
        let env = env_with(&[("TALOS_RALPH_WALL_CLOCK_SECS", "not-a-number")]);
        assert_eq!(
            resolve_ralph_wall_clock_secs(None, &env),
            0,
            "invalid env value must fall back to 0 (unbounded)"
        );
    }

    // ---- resolve_state_retention_days: all six branches, with source -----

    #[test]
    fn state_retention_days_flag_beats_env() {
        let env = env_with(&[("TALOS_STATE_RETENTION_DAYS", "99")]);
        assert_eq!(resolve_state_retention_days(Some(7), &env), (7, "flag"));
    }

    #[test]
    fn state_retention_days_env_beats_default() {
        let env = env_with(&[("TALOS_STATE_RETENTION_DAYS", "99")]);
        assert_eq!(resolve_state_retention_days(None, &env), (99, "env"));
    }

    #[test]
    fn state_retention_days_default_when_both_absent() {
        let env = env_with(&[]);
        assert_eq!(resolve_state_retention_days(None, &env), (30, "default"));
    }

    #[test]
    fn state_retention_days_invalid_env_falls_back_to_default() {
        let env = env_with(&[("TALOS_STATE_RETENTION_DAYS", "abc")]);
        assert_eq!(resolve_state_retention_days(None, &env), (30, "default"));
    }

    #[test]
    fn state_retention_days_flag_zero_disables() {
        let env = env_with(&[]);
        assert_eq!(resolve_state_retention_days(Some(0), &env), (0, "flag"));
    }

    #[test]
    fn state_retention_days_env_zero_disables() {
        let env = env_with(&[("TALOS_STATE_RETENTION_DAYS", "0")]);
        assert_eq!(resolve_state_retention_days(None, &env), (0, "env"));
    }

    // ---- resolve_compact_threshold_pct: every branch, with the fatal-env
    // ---- rule that deliberately diverges from state-retention ----------

    #[test]
    fn compact_threshold_pct_flag_beats_env() {
        let env = env_with(&[("TALOS_COMPACT_THRESHOLD_PCT", "99")]);
        assert_eq!(resolve_compact_threshold_pct(Some(7), &env), Ok(7));
    }

    #[test]
    fn compact_threshold_pct_env_beats_default() {
        let env = env_with(&[("TALOS_COMPACT_THRESHOLD_PCT", "99")]);
        assert_eq!(resolve_compact_threshold_pct(None, &env), Ok(99));
    }

    #[test]
    fn compact_threshold_pct_default_when_both_absent() {
        let env = env_with(&[]);
        assert_eq!(
            resolve_compact_threshold_pct(None, &env),
            Ok(harness::engine::COMPACT_THRESHOLD_PCT),
            "the default must be the compiled engine constant"
        );
    }

    #[test]
    fn compact_threshold_pct_empty_and_whitespace_env_are_unset() {
        for raw in ["", "   "] {
            let vars = [("TALOS_COMPACT_THRESHOLD_PCT", raw)];
            let env = env_with(&vars);
            assert_eq!(
                resolve_compact_threshold_pct(None, &env),
                Ok(harness::engine::COMPACT_THRESHOLD_PCT),
                "an empty or whitespace-only value ({raw:?}) must be treated as unset"
            );
        }
    }

    #[test]
    fn compact_threshold_pct_non_numeric_env_is_a_hard_error() {
        let env = env_with(&[("TALOS_COMPACT_THRESHOLD_PCT", "abc")]);
        let err = resolve_compact_threshold_pct(None, &env)
            .expect_err("a non-numeric env value must never silently fall back");
        assert!(
            err.contains("TALOS_COMPACT_THRESHOLD_PCT") && err.contains("abc"),
            "the error must name the variable and the raw value; got {err:?}"
        );
    }

    #[test]
    fn compact_threshold_pct_flag_zero_disables() {
        let env = env_with(&[("TALOS_COMPACT_THRESHOLD_PCT", "99")]);
        assert_eq!(resolve_compact_threshold_pct(Some(0), &env), Ok(0));
    }

    #[test]
    fn compact_threshold_pct_env_zero_disables() {
        let env = env_with(&[("TALOS_COMPACT_THRESHOLD_PCT", "0")]);
        assert_eq!(resolve_compact_threshold_pct(None, &env), Ok(0));
    }

    #[test]
    fn compact_threshold_pct_flag_over_100_round_trips() {
        // Values above 100 are accepted verbatim — simply never reachable.
        let env = env_with(&[]);
        assert_eq!(resolve_compact_threshold_pct(Some(150), &env), Ok(150));
    }

    // ---- touch_dir_mtime --------------------------------------------------

    /// Age `path`'s own mtime by `days` days, per the pinned no-new-dependency
    /// mechanism: `File::open` (works on a directory opened read-only, on
    /// this platform) + `set_times`.
    fn age_dir(path: &std::path::Path, days: u64) {
        let mtime = SystemTime::now() - Duration::from_secs(days * SECS_PER_DAY);
        std::fs::File::open(path)
            .and_then(|f| f.set_times(std::fs::FileTimes::new().set_modified(mtime)))
            .expect("set_times must succeed on a directory opened read-only");
    }

    #[test]
    fn touch_dir_mtime_refreshes_an_aged_directory() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let target = dir.path().join("aged");
        std::fs::create_dir_all(&target).unwrap();
        age_dir(&target, 40);

        touch_dir_mtime(&target);

        let mtime = std::fs::metadata(&target)
            .expect("metadata")
            .modified()
            .expect("modified");
        let age = SystemTime::now().duration_since(mtime).unwrap_or_default();
        assert!(
            age < Duration::from_mins(1),
            "touched dir must be less than 60s old; age={age:?}"
        );
    }

    #[test]
    fn touch_dir_mtime_on_missing_path_is_a_silent_noop() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let missing = dir.path().join("does-not-exist");
        touch_dir_mtime(&missing);
        assert!(
            !missing.exists(),
            "touch_dir_mtime must never create the path"
        );
    }

    // ---- prune_state_root ---------------------------------------------

    #[test]
    fn prune_zero_max_age_disables_pruning_before_any_io() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let stale = dir.path().join("stale-task");
        std::fs::create_dir_all(&stale).unwrap();
        age_dir(&stale, 40);

        let report = prune_state_root(dir.path(), SystemTime::now(), Duration::ZERO, &[]);
        assert_eq!(report, PruneReport::default());
        assert!(stale.exists(), "disabled pruning must not touch anything");
    }

    #[test]
    fn prune_nonexistent_root_returns_default_report() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let missing = dir.path().join("does-not-exist");
        let report = prune_state_root(
            &missing,
            SystemTime::now(),
            Duration::from_secs(30 * SECS_PER_DAY),
            &[],
        );
        assert_eq!(report, PruneReport::default());
    }

    #[test]
    fn prune_removes_dir_older_than_max_age() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let stale = dir.path().join("stale-task");
        std::fs::create_dir_all(&stale).unwrap();
        age_dir(&stale, 40);

        let report = prune_state_root(
            dir.path(),
            SystemTime::now(),
            Duration::from_secs(30 * SECS_PER_DAY),
            &[],
        );
        assert_eq!(report.removed, 1);
        assert!(!stale.exists());
    }

    #[test]
    fn prune_keeps_dir_younger_than_max_age() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let stale = dir.path().join("stale-task");
        std::fs::create_dir_all(&stale).unwrap();
        age_dir(&stale, 40);

        let report = prune_state_root(
            dir.path(),
            SystemTime::now(),
            Duration::from_secs(60 * SECS_PER_DAY),
            &[],
        );
        assert_eq!(report.kept_young, 1);
        assert!(stale.exists());
    }

    #[test]
    fn prune_future_mtime_is_skipped_not_removed() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let future = dir.path().join("future-task");
        std::fs::create_dir_all(&future).unwrap();
        let now = SystemTime::now();
        let mtime = now + Duration::from_secs(SECS_PER_DAY);
        std::fs::File::open(&future)
            .and_then(|f| f.set_times(std::fs::FileTimes::new().set_modified(mtime)))
            .expect("set_times must succeed");

        let report = prune_state_root(dir.path(), now, Duration::from_secs(30 * SECS_PER_DAY), &[]);
        assert_eq!(report.skipped, 1);
        assert_eq!(report.removed, 0);
        assert!(future.exists());
    }

    #[test]
    fn prune_keep_list_direct_and_ancestor_survive_sibling_removed() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let root = dir.path();
        let direct_keep = root.join("direct-keep");
        let ancestor_keep = root.join("ancestor-keep");
        let sibling = root.join("sibling-task");
        for p in [&direct_keep, &ancestor_keep, &sibling] {
            std::fs::create_dir_all(p).unwrap();
            age_dir(p, 40);
        }
        // `ancestor_keep` is not itself in `keep` — a path NESTED under it is,
        // so `ancestor_keep` must survive as its ancestor.
        let nested = ancestor_keep.join("run.sqlite");
        let keep = vec![direct_keep.clone(), nested];

        let report = prune_state_root(
            root,
            SystemTime::now(),
            Duration::from_secs(30 * SECS_PER_DAY),
            &keep,
        );
        assert_eq!(report.kept_keep, 2);
        assert_eq!(report.removed, 1);
        assert!(direct_keep.exists());
        assert!(ancestor_keep.exists());
        assert!(!sibling.exists());
    }

    #[test]
    fn prune_keep_normalization_dotdot_and_cwd_relative() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let root = dir.path().join("talos");
        std::fs::create_dir_all(&root).unwrap();
        let keepme = root.join("keepme");
        std::fs::create_dir_all(&keepme).unwrap();
        age_dir(&keepme, 40);

        // `..`-segment normalization: `<root>/keepme/../keepme` must still
        // canonicalize to `<root>/keepme`.
        let keep_dotdot = root.join("keepme").join("..").join("keepme");
        let report = prune_state_root(
            &root,
            SystemTime::now(),
            Duration::from_secs(30 * SECS_PER_DAY),
            &[keep_dotdot],
        );
        assert_eq!(report.kept_keep, 1, "dotdot-normalized keep must protect");
        assert!(keepme.exists());

        // cwd-relative normalization: a bare relative path equal to `keepme`
        // when resolved against the test process's cwd.
        let old_cwd = std::env::current_dir().expect("current_dir");
        std::env::set_current_dir(&root).expect("set_current_dir");
        let keep_relative = PathBuf::from("keepme");
        let report2 = prune_state_root(
            &root,
            SystemTime::now(),
            Duration::from_secs(30 * SECS_PER_DAY),
            &[keep_relative],
        );
        std::env::set_current_dir(&old_cwd).expect("restore cwd");
        assert_eq!(report2.kept_keep, 1, "cwd-relative keep must protect");
        assert!(keepme.exists());
    }

    #[test]
    fn prune_skips_non_directories_files_symlinks_broken_symlinks() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let root = dir.path().join("talos");
        std::fs::create_dir_all(&root).unwrap();
        let outside = dir.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        let sentinel = outside.join("sentinel.txt");
        std::fs::write(&sentinel, "keep me").unwrap();

        let stale_file = root.join("stale.txt");
        std::fs::write(&stale_file, "x").unwrap();
        age_dir(&stale_file, 40);

        let link = root.join("link-to-dir");
        std::os::unix::fs::symlink(&outside, &link).expect("symlink");

        let broken_target = root.join("does-not-exist-target");
        let broken = root.join("broken");
        std::os::unix::fs::symlink(&broken_target, &broken).expect("symlink");

        let report = prune_state_root(
            &root,
            SystemTime::now(),
            Duration::from_secs(30 * SECS_PER_DAY),
            &[],
        );
        assert_eq!(report.removed, 0);
        assert_eq!(report.skipped, 3);
        assert!(stale_file.exists());
        assert!(std::fs::symlink_metadata(&link).is_ok());
        assert!(std::fs::symlink_metadata(&broken).is_ok());
        assert!(outside.exists());
        assert!(sentinel.exists());
    }

    #[test]
    fn prune_aggregate_dirs_descend_one_level_never_removed_themselves() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let root = dir.path();
        for agg in ["mined-eval", "coding-eval"] {
            let agg_dir = root.join(agg);
            std::fs::create_dir_all(&agg_dir).unwrap();
            // The aggregate's own mtime is aged too — it must not matter.
            age_dir(&agg_dir, 40);

            let old_child = agg_dir.join("old-run");
            std::fs::create_dir_all(&old_child).unwrap();
            age_dir(&old_child, 40);

            let fresh_child = agg_dir.join("fresh-run");
            std::fs::create_dir_all(&fresh_child).unwrap();
        }

        let report = prune_state_root(
            root,
            SystemTime::now(),
            Duration::from_secs(30 * SECS_PER_DAY),
            &[],
        );
        assert_eq!(report.aggregates, 2);
        assert_eq!(report.removed, 2);
        for agg in ["mined-eval", "coding-eval"] {
            let agg_dir = root.join(agg);
            assert!(agg_dir.exists(), "{agg} aggregate dir itself must survive");
            assert!(!agg_dir.join("old-run").exists());
            assert!(agg_dir.join("fresh-run").exists());
        }
    }

    #[test]
    fn prune_mixed_fixture_bucket_invariant_and_removed_names() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let root = dir.path();

        let expired_a = root.join("expired-a");
        let expired_b = root.join("expired-b");
        let fresh = root.join("fresh-task");
        let keep_dir = root.join("keep-task");
        for p in [&expired_a, &expired_b, &keep_dir] {
            std::fs::create_dir_all(p).unwrap();
            age_dir(p, 40);
        }
        std::fs::create_dir_all(&fresh).unwrap();

        let agg = root.join("mined-eval");
        std::fs::create_dir_all(&agg).unwrap();
        let agg_child = agg.join("old-trial");
        std::fs::create_dir_all(&agg_child).unwrap();
        age_dir(&agg_child, 40);

        let report = prune_state_root(
            root,
            SystemTime::now(),
            Duration::from_secs(30 * SECS_PER_DAY),
            std::slice::from_ref(&keep_dir),
        );

        assert_eq!(report.removed, 3);
        assert_eq!(report.kept_young, 1);
        assert_eq!(report.kept_keep, 1);
        assert_eq!(report.aggregates, 1);
        assert_eq!(
            report.examined,
            report.removed
                + report.kept_keep
                + report.kept_young
                + report.skipped
                + report.aggregates,
            "bucket invariant must hold"
        );
        assert!(report.removed_names.contains(&"expired-a".to_string()));
        assert!(report.removed_names.contains(&"expired-b".to_string()));
        assert!(
            report
                .removed_names
                .contains(&"mined-eval/old-trial".to_string())
        );
        assert!(fresh.exists());
        assert!(keep_dir.exists());
    }

    #[test]
    fn prune_removed_names_capped_at_max_report_names() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let root = dir.path();
        for i in 0..9 {
            let p = root.join(format!("expired-{i}"));
            std::fs::create_dir_all(&p).unwrap();
            age_dir(&p, 40);
        }
        let report = prune_state_root(
            root,
            SystemTime::now(),
            Duration::from_secs(30 * SECS_PER_DAY),
            &[],
        );
        assert_eq!(report.removed, 9);
        assert_eq!(report.removed_names.len(), MAX_REPORT_NAMES);
        assert_eq!(report.removed_truncated, 1);
    }

    #[test]
    fn prune_confines_removal_to_root_children_sibling_survives() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let root = dir.path().join("talos");
        std::fs::create_dir_all(&root).unwrap();
        let sibling = dir.path().join("sibling");
        std::fs::create_dir_all(&sibling).unwrap();
        age_dir(&sibling, 40);

        let stale = root.join("stale-task");
        std::fs::create_dir_all(&stale).unwrap();
        age_dir(&stale, 40);

        let report = prune_state_root(
            &root,
            SystemTime::now(),
            Duration::from_secs(30 * SECS_PER_DAY),
            &[],
        );
        assert_eq!(report.removed, 1);
        assert!(!stale.exists());
        assert!(
            sibling.exists(),
            "a sibling outside the prune root must never be touched"
        );
    }

    // ---- prune_report_json -------------------------------------------

    #[test]
    fn prune_report_json_shape_no_error_key_single_line() {
        let root = PathBuf::from("/tmp/talos");
        let report = PruneReport::default();
        let s = prune_report_json(&root, 30, "flag", &report);
        let v: serde_json::Value = serde_json::from_str(&s).expect("must parse as JSON");
        assert!(v.is_object());
        assert!(v.get("error").is_none(), "must not contain an `error` key");
        assert!(!s.contains('\n'), "must be single-line");
        assert_eq!(
            v.get("disabled").and_then(serde_json::Value::as_bool),
            Some(false)
        );
    }

    #[test]
    fn prune_report_json_disabled_true_when_retention_zero() {
        let root = PathBuf::from("/tmp/talos");
        let report = PruneReport::default();
        let s = prune_report_json(&root, 0, "flag", &report);
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(
            v.get("disabled").and_then(serde_json::Value::as_bool),
            Some(true)
        );
        assert_eq!(
            v.get("removed").and_then(serde_json::Value::as_u64),
            Some(0)
        );
        assert_eq!(
            v.get("examined").and_then(serde_json::Value::as_u64),
            Some(0)
        );
    }

    #[test]
    fn prune_report_json_round_trips_mixed_fixture_counts() {
        let root = PathBuf::from("/tmp/talos");
        let report = PruneReport {
            examined: 6,
            removed: 3,
            kept_young: 1,
            kept_keep: 1,
            skipped: 0,
            aggregates: 1,
            removed_names: vec![
                "expired-a".to_string(),
                "expired-b".to_string(),
                "mined-eval/old-trial".to_string(),
            ],
            removed_truncated: 0,
        };
        let s = prune_report_json(&root, 30, "default", &report);
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["examined"], 6);
        assert_eq!(v["removed"], 3);
        assert_eq!(v["kept_young"], 1);
        assert_eq!(v["kept_keep"], 1);
        assert_eq!(v["aggregates"], 1);
        assert_eq!(v["removed_names"].as_array().unwrap().len(), 3);
    }
}
