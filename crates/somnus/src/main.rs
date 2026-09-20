//! The `somnus` binary: a plain, timer-invoked subprocess — no listener, no
//! service, nothing for an installer to restart (the installer's only
//! `systemctl` string is a comment explaining its absence).
//!
//! This cut the run body is a LOUD stub, not a silent one: the KB
//! write-endpoint contract behind `MapOpSink` is not pinned anywhere in this
//! repo, so an accidental nightly invocation fails loudly with exit 1 instead
//! of burning the metered Anthropic lane or silently doing nothing. The
//! library code it waits on — loop-input fetch, the decline ledger filter,
//! and rungs 1-3 — is wired and exercised by the crate's own tests.
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

/// The not-wired run stub's stderr message: names the ONE remaining gap (the
/// KB write-endpoint contract behind `MapOpSink`) and what IS already wired,
/// so a reader of the log knows exactly why nothing happened.
const NOT_WIRED_MSG: &str = "somnus: run body not wired this cut — the KB write-endpoint contract behind MapOpSink is not pinned; loop-input fetch, the decline ledger filter, and rungs 1-3 are wired as library code and exercised by tests. Exiting without doing anything.";

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run(&args.project),
    }
}

/// The `run` body. The stub is the FIRST statement after argument validation —
/// before any count-source call, backend construction, or network I/O (none of
/// which exist in this cut's binary).
///
// The vendored spec's line-103 startup assertion (working directory is NOT a
// git repo) is deliberately NOT implemented, per the lead's correction which
// overrides the spec: somnus supplies a custom `ChangeObserver`
// (`observer::MapPointerObserver`, which the pipeline in `unit::run_unit`
// constructs over the loop-input source), so git is never consulted — the
// observer serves BOTH the run-start baseline and the final observation, and
// `observe_tree` is not called on either. A stray `.git` is therefore inert,
// and an assertion refusing to start in a directory that happens to be a repo
// would block a legitimate deployment for no reason.
fn run(project: &str) {
    if project.trim().is_empty() {
        eprintln!("{EMPTY_PROJECT_MSG}");
        std::process::exit(1);
    }
    eprintln!("{NOT_WIRED_MSG}");
    std::process::exit(1);
}
