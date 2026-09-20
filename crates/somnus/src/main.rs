//! The `somnus` binary: a plain, timer-invoked subprocess — no listener, no
//! service, nothing for an installer to restart (the installer's only
//! `systemctl` string is a comment explaining its absence).
//!
//! This cut the run body is a LOUD stub, not a silent one: the functional
//! specification of the nightly loop is the KB session's to write, and its
//! two in-flight input contracts (`somnus-loop-input`,
//! `somnus-cluster-ledger`) are not landed, so an accidental nightly
//! invocation fails loudly with exit 1 instead of burning the metered
//! Anthropic lane or silently doing nothing.
//!
//! Every library construct lives in the `somnus` lib target (`src/lib.rs`);
//! this bin is thin on purpose.

use clap::Parser;

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
    /// Run the nightly map-maintenance loop for one project.
    Run(RunArgs),
}

/// What `somnus run` is being asked to do.
#[derive(clap::Args)]
struct RunArgs {
    /// The project ref to run the loop for.
    #[arg(long)]
    project: String,
}

/// Rejection for an empty or whitespace-only `--project` value, checked
/// BEFORE any other work.
const EMPTY_PROJECT_MSG: &str = "somnus: --project must be a non-empty project ref";

/// The not-wired run stub's stderr message: names the two in-flight GTD items
/// the run body is waiting on and the two deferred inference rungs, so a
/// reader of the log knows exactly why nothing happened.
const NOT_WIRED_MSG: &str = "somnus: run body not wired this cut — the run-body input contract is in flight as somnus-loop-input and the cluster/decline ledger as somnus-cluster-ledger; the rung-1 cluster extraction and rung-2 op inference calls are deferred until both land. Exiting without doing anything.";

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run(&args.project),
    }
}

/// The `run` body. The stub is the FIRST statement after argument validation —
/// before any count-source call, backend construction, or network I/O (none
/// of which exist in this cut).
///
// The vendored spec's line-103 startup assertion (working directory is NOT a
// git repo) is deliberately NOT implemented, per the lead's correction which
// overrides the spec: somnus supplies a custom `ChangeObserver` via
// `RunConfig::with_change_observer` (see `somnus::build_run_config`), so git
// is never consulted — the observer serves BOTH the run-start baseline and
// every finish-time observation, and `observe_tree` is not called on either.
// A stray `.git` is therefore inert, and an assertion refusing to start in a
// directory that happens to be a repo would block a legitimate deployment for
// no reason.
fn run(project: &str) {
    if project.trim().is_empty() {
        eprintln!("{EMPTY_PROJECT_MSG}");
        std::process::exit(1);
    }
    eprintln!("{NOT_WIRED_MSG}");
    std::process::exit(1);
}
