//! The `somnus` binary: a plain, timer-invoked subprocess — no listener, no
//! service, nothing for an installer to restart (the installer's only
//! `systemctl` string is a comment explaining its absence).
//!
//! The pinned argv/env/validation order (load-bearing for the CLI tests):
//!
//! 1. argv parse via [`Cli::try_parse`] — every clap usage error exits 1
//!    with [`CLI_USAGE_ERROR_MSG`] (clap's default exit 2 is overridden);
//!    `--version`/`--help` follow clap's default render and exit 0;
//! 2. the [`DISABLED_VAR`] kill switch (exactly `1` exits 0 with the pinned
//!    line) — BEFORE every other env read, the `--project` check, and any
//!    filesystem or network action;
//! 3. the empty-or-whitespace `--project` check ([`EMPTY_PROJECT_MSG`])
//!    BEFORE required-env validation;
//! 4. required-env validation;
//! 5. [`TOKEN_BUDGET_VAR`] parse;
//! 6. state-dir resolution + creation.
//!
//! Kill-switch, env-fault, and clap-usage exits occur before the state dir
//! is resolved and write nothing, by design.
//!
//! Every library construct lives in the `somnus` lib target (`src/lib.rs`);
//! this bin is thin on purpose.

use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::Parser;
use harness::anthropic::AnthropicBackend;
use harness::tool::ToolCtx;
use somnus::KB_API_KEY_VAR;
use somnus::ledger::HttpClusterLedger;
use somnus::loop_input::HttpLoopInputSource;
use somnus::map_op::HttpMapOpClient;
use somnus::unit::{UnitDeps, UnitOutcome, exit_code_for_outcome};
use somnus::worklist::{
    NOTHING_ELIGIBLE_MSG, NightlyUnitRecord, STOP_REASON_ALL_DONE, STOP_REASON_NOTHING_ELIGIBLE,
    WORKLIST_UNAUTHORIZED_MSG, budget_stopped_before, build_invocation, fault_stop_reason,
    fetch_worklist, render_nightly_budget_stop_line, render_nightly_unit_line,
    render_worklist_fault_line, select_first_projects, write_invocation,
};

/// Top-level CLI entry point.
#[derive(clap::Parser)]
#[command(name = "somnus", about = "KB nightly map-maintenance loop", version = env!("SOMNUS_VERSION"))]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Somnus subcommands.
#[derive(clap::Subcommand)]
enum Command {
    /// Run the nightly map-maintenance loop across the whole worklist.
    Nightly,
    /// Run the nightly map-maintenance loop for one project.
    Run(RunArgs),
    /// Run the loop for one project under the doubled backfill budget.
    Backfill(RunArgs),
}

/// What `somnus run` / `somnus backfill` is being asked to do.
#[derive(clap::Args)]
struct RunArgs {
    /// The project ref to run the loop for.
    #[arg(long)]
    project: String,
}

/// Rejection for an empty or whitespace-only `--project` value, checked
/// BEFORE required-env validation.
const EMPTY_PROJECT_MSG: &str = "somnus: --project must be a non-empty project ref";

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // (1) argv parse. clap's default usage-error exit 2 is overridden: any
    // usage error is ONE byte-pinned line and exit 1, so the exit codes are
    // somnus's, not clap's defaults. `--version`/`--help` follow clap's
    // default render and exit 0.
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(err) => {
            use clap::error::ErrorKind::{DisplayHelp, DisplayVersion};
            if matches!(err.kind(), DisplayHelp | DisplayVersion) {
                let _ = err.print();
                std::process::exit(0);
            }
            eprintln!("{}", somnus::CLI_USAGE_ERROR_MSG);
            std::process::exit(1);
        }
    };

    // (2) The kill switch: checked after argv parse and before every other
    // env read, the `--project` check, and any filesystem or network action.
    let disabled = std::env::var(somnus::DISABLED_VAR).ok();
    if somnus::is_disabled(disabled.as_deref()) {
        eprintln!("{}", somnus::DISABLED_MSG);
        std::process::exit(0);
    }

    // (3) The empty-or-whitespace `--project` check, BEFORE required-env
    // validation.
    let project = match &cli.command {
        Command::Run(args) | Command::Backfill(args) => {
            if args.project.trim().is_empty() {
                eprintln!("{EMPTY_PROJECT_MSG}");
                std::process::exit(1);
            }
            args.project.clone()
        }
        Command::Nightly => String::new(),
    };

    // (4) Required env.
    let base_raw = std::env::var(somnus::KB_BASE_URL_VAR).ok();
    let key_raw = std::env::var(KB_API_KEY_VAR).ok();
    let anthropic_raw = std::env::var(somnus::ANTHROPIC_API_KEY_VAR).ok();
    let machine = match somnus::parse_required_env(
        base_raw.as_deref(),
        key_raw.as_deref(),
        anthropic_raw.as_deref(),
    ) {
        Ok(machine) => machine,
        Err(line) => {
            eprintln!("{line}");
            std::process::exit(1);
        }
    };

    // (5) The token budget, armed by subcommand.
    let budget_raw = std::env::var(somnus::TOKEN_BUDGET_VAR).ok();
    let default = match &cli.command {
        Command::Nightly => somnus::NIGHTLY_TOKEN_BUDGET_DEFAULT,
        Command::Run(_) => somnus::RUN_TOKEN_BUDGET_DEFAULT,
        Command::Backfill(_) => somnus::BACKFILL_TOKEN_BUDGET_DEFAULT,
    };
    let armed = match somnus::parse_token_budget(default, budget_raw.as_deref()) {
        Ok(armed) => armed,
        Err(line) => {
            eprintln!("{line}");
            std::process::exit(1);
        }
    };

    // (6) State-dir resolution + creation (the state dir and its `offload`
    // child exist before `Workspace::new` canonicalizes both roots).
    let state_dir_var = std::env::var(somnus::STATE_DIR_VAR).ok();
    let xdg = std::env::var("XDG_STATE_HOME").ok();
    let home = std::env::var("HOME").ok();
    let state_dir =
        somnus::resolve_state_dir(state_dir_var.as_deref(), xdg.as_deref(), home.as_deref());
    if let Err(err) = std::fs::create_dir_all(&state_dir) {
        eprintln!(
            "somnus: could not create the state dir {}: {err}",
            state_dir.display()
        );
        std::process::exit(2);
    }
    let offload_dir = state_dir.join("offload");
    if let Err(err) = std::fs::create_dir_all(&offload_dir) {
        eprintln!(
            "somnus: could not create the offload dir {}: {err}",
            offload_dir.display()
        );
        std::process::exit(2);
    }

    // The token bridge for the gate curl: SOMNUS_KB_API_KEY verbatim into
    // `<state_dir>/kb-token` (mode 0600 on unix), written before the first
    // unit. The env var is the ONE token source; the state-dir file its
    // only derivative — the gate child can only read it by file, because
    // `exec::run`'s env_clear preserves only TERM/PATH/HOME and keeping the
    // path (never the token) in the script keeps it out of argv,
    // `command_display`, the transcript, and gitleaks.
    let token_file = state_dir.join("kb-token");
    if let Err(err) = write_token_file(&token_file, &machine.kb_api_key) {
        eprintln!(
            "somnus: could not write the kb-token file {}: {err}",
            token_file.display()
        );
        std::process::exit(2);
    }

    let exit_code = match &cli.command {
        Command::Nightly => {
            nightly(&machine, &state_dir, &token_file, armed, invoked_at_utc()).await
        }
        Command::Run(_) | Command::Backfill(_) => {
            run_single(&project, &machine, &state_dir, &token_file, armed).await
        }
    };
    std::process::exit(exit_code);
}

/// The current UNIX epoch in seconds (the invocation record's
/// `invoked_at_utc`).
fn invoked_at_utc() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |delta| delta.as_secs())
}

/// Write the KB bearer token to `path`, mode 0600 on unix.
///
/// # Errors
/// Any filesystem failure.
fn write_token_file(path: &Path, token: &str) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)?;
        file.write_all(token.as_bytes())
    }
    #[cfg(not(unix))]
    {
        std::fs::write(path, token)
    }
}

/// The shared production stack every subcommand drives: the production
/// [`AnthropicBackend`], [`HttpLoopInputSource`], [`HttpClusterLedger`],
/// [`HttpMapOpClient`], and the gate wiring (its runner reads the token from
/// the state dir's kb-token file).
struct Stack {
    base: String,
    backend: AnthropicBackend,
    ledger: HttpClusterLedger,
    source: Arc<dyn somnus::loop_input::LoopInputSource>,
    map_ops: Arc<dyn somnus::map_op::MapOpClient>,
}

impl Stack {
    /// Build the stack over `machine`.
    fn build(machine: &somnus::MachineEnv) -> Self {
        Self {
            base: machine.kb_base_url.clone(),
            backend: AnthropicBackend::new(
                somnus::rungs::SOMNUS_MODEL_ID,
                machine.anthropic_api_key.clone(),
            ),
            ledger: HttpClusterLedger::new(machine.kb_base_url.clone(), machine.kb_api_key.clone()),
            source: Arc::new(HttpLoopInputSource::new(
                machine.kb_base_url.clone(),
                machine.kb_api_key.clone(),
            )),
            map_ops: Arc::new(HttpMapOpClient::new(
                machine.kb_base_url.clone(),
                machine.kb_api_key.clone(),
            )),
        }
    }
}

/// Run ONE unit end to end against the production stack and return the
/// report. Never touches `/api/kb/map-worklist` — a single-project path has
/// no use for the worklist.
async fn run_one_unit(
    project_ref: &str,
    stack: &Stack,
    state_dir: &Path,
    token_file: &Path,
    armed: u64,
    billed_before: u64,
) -> Result<somnus::unit::UnitReport, String> {
    let workspace = harness::workspace::Workspace::new(state_dir, Some(state_dir.join("offload")))
        .map_err(|err| {
            format!(
                "somnus: could not build the workspace over {}: {err}",
                state_dir.display()
            )
        })?;
    let tool_ctx = ToolCtx::new(
        Arc::new(workspace),
        Arc::new(harness::workspace::DiskOffloadSink::new(
            state_dir.join("offload"),
        )),
    );
    let kb_base = stack.base.clone();
    let token_file = token_file.to_path_buf();
    let gate_for =
        move |body_path: &Path| somnus::gate::map_lint_runner(&kb_base, body_path, &token_file);
    let deps = UnitDeps {
        backend: &stack.backend,
        source: Arc::clone(&stack.source),
        ledger: &stack.ledger,
        gate_for: &gate_for,
        tool_ctx: &tool_ctx,
        map_ops: Arc::clone(&stack.map_ops),
        token_budget: armed,
        billed_before,
        body_root: state_dir.to_path_buf(),
    };
    Ok(somnus::unit::run_unit(&deps, project_ref).await)
}

/// The `nightly` orchestration: one worklist call, then at most
/// [`somnus::materialize::SOMNUS_MAX_PROJECTS_PER_NIGHT`] units run
/// SEQUENTIALLY in verbatim server order. A stopping unit stops the
/// invocation (burning more metered turns after a 401/403 is a defect); a
/// token-budget stop happens BEFORE the next unit's deps are constructed, so
/// the un-started project gets no unit and no report. Whatever the exit
/// code, the invocation record is written before returning.
#[allow(clippy::too_many_lines)] // one orchestration, three stop shapes
async fn nightly(
    machine: &somnus::MachineEnv,
    state_dir: &Path,
    token_file: &Path,
    armed: u64,
    invoked_at_utc: u64,
) -> i32 {
    let stack = Stack::build(machine);
    let worklist = fetch_worklist(&machine.kb_base_url, &machine.kb_api_key).await;
    let worklist_projects = match worklist {
        somnus::worklist::WorklistOutcome::Ready(projects) => projects,
        somnus::worklist::WorklistOutcome::Unauthorized => {
            eprintln!("{WORKLIST_UNAUTHORIZED_MSG}");
            let record = build_invocation(
                invoked_at_utc,
                "nightly",
                armed,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                fault_stop_reason(1),
                1,
            );
            let _ = write_invocation(state_dir, &record);
            return 1;
        }
        somnus::worklist::WorklistOutcome::Fault { reason } => {
            eprintln!("{}", render_worklist_fault_line(&reason));
            let record = build_invocation(
                invoked_at_utc,
                "nightly",
                armed,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                fault_stop_reason(2),
                2,
            );
            let _ = write_invocation(state_dir, &record);
            return 2;
        }
    };

    // The take-first-three slice: verbatim server order, no sorting, no
    // re-ranking. The obvious heuristic, most-unpointed-first, STARVES — a
    // project whose unpointed entries all sit in declined clusters tops the
    // list every night forever and blocks every other project, and
    // declining a cluster does not make its members pointed. Staleness-first
    // rotates whether or not a given night accomplished anything.
    let selected = select_first_projects(&worklist_projects);
    if selected.is_empty() {
        eprintln!("{NOTHING_ELIGIBLE_MSG}");
        let record = build_invocation(
            invoked_at_utc,
            "nightly",
            armed,
            worklist_projects,
            selected,
            Vec::new(),
            STOP_REASON_NOTHING_ELIGIBLE.to_string(),
            0,
        );
        let _ = write_invocation(state_dir, &record);
        return 0;
    }

    let mut units: Vec<NightlyUnitRecord> = Vec::new();
    let mut billed_total: u64 = 0;
    let mut stop_reason = STOP_REASON_ALL_DONE.to_string();
    let mut exit_code = 0i32;
    for project_ref in &selected {
        // The across-units budget stop: the sum of every COMPLETED unit's
        // billed usage is checked BEFORE the next unit's deps are
        // constructed. No second accumulator — this reads the accumulator
        // that already exists in [`somnus::unit::run_unit`]'s report usage.
        if harness::engine::token_budget_breached(billed_total, armed) {
            eprintln!(
                "{}",
                render_nightly_budget_stop_line(armed, billed_total, project_ref)
            );
            stop_reason = budget_stopped_before(project_ref);
            exit_code = 2;
            break;
        }
        let report = match run_one_unit(
            project_ref,
            &stack,
            state_dir,
            token_file,
            armed,
            billed_total,
        )
        .await
        {
            Ok(report) => report,
            Err(reason) => {
                eprintln!("{reason}");
                stop_reason = fault_stop_reason(2);
                exit_code = 2;
                break;
            }
        };
        eprintln!("{}", render_nightly_unit_line(project_ref, &report.outcome));
        let billed = report.usage_rung1.billed() + report.usage_rung2.billed();
        billed_total += billed;
        units.push(NightlyUnitRecord {
            project_ref: project_ref.clone(),
            outcome: report.outcome.clone(),
            billed_tokens: billed,
            report_path: report.report_path.clone(),
        });
        match &report.outcome {
            UnitOutcome::Ready | UnitOutcome::UnknownProject | UnitOutcome::NotEligible => {
                // Ordinary outcomes never stop the night.
            }
            // The stopping outcomes: Unauthorized/NotMachinePrincipal (1)
            // and Aborted (2). The stopping unit's code wins; within
            // stopping outcomes 1 takes precedence over 2, which holds
            // naturally because the loop STOPS here.
            other => {
                exit_code = exit_code_for_outcome(other);
                stop_reason = fault_stop_reason(exit_code);
                break;
            }
        }
    }
    let record = build_invocation(
        invoked_at_utc,
        "nightly",
        armed,
        worklist_projects,
        selected,
        units,
        stop_reason,
        exit_code,
    );
    if let Err(err) = write_invocation(state_dir, &record) {
        eprintln!("somnus: could not write the nightly invocation record: {err}");
    }
    exit_code
}

/// The single-project path (`run` and `backfill` — the ONLY difference
/// between them is the token-budget default armed by the caller).
async fn run_single(
    project_ref: &str,
    machine: &somnus::MachineEnv,
    state_dir: &Path,
    token_file: &Path,
    armed: u64,
) -> i32 {
    let stack = Stack::build(machine);
    let report = match run_one_unit(project_ref, &stack, state_dir, token_file, armed, 0).await {
        Ok(report) => report,
        Err(reason) => {
            eprintln!("{reason}");
            return 2;
        }
    };
    exit_code_for_outcome(&report.outcome)
}
