//! Convert talos-evals tier-2 mined tasks into Harbor task directories.
//!
//! ```text
//! cargo run -p harness --example mined_to_harbor -- \
//!   --tasks-dir ~/git/talos-evals/tasks --out /tmp/harbor-tasks [--task <ID>] [--spec-level s1|s2|s3]
//! ```
//!
//! `--tasks-dir` is `<talos-evals>/tasks`: the directory whose immediate subdirectories are mined task dirs.
//!
//! The tool reads the private talos-evals checkout plus the source repos at each task's `repo_path` (and their siblings), so it is opt-in and never run by any hook, CI job or test; the gates only check that this file compiles.
//!
//! Its output must never be committed: every emitted task carries git bundles of private repos. The emitter refuses any `--out` inside a git work tree.
//!
//! Exit codes: 0 on success (or `--help`), 1 when emitting fails, 2 on a usage error. See `harness::mined_harbor` for the parity contract.

use std::process::ExitCode;

use harness::mined_harbor;

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let args = match mined_harbor::parse_emit_args(&raw) {
        Ok(None) => {
            print!("{}", mined_harbor::USAGE);
            return ExitCode::SUCCESS;
        }
        Ok(Some(args)) => args,
        Err(msg) => {
            eprintln!("error: {msg}");
            eprint!("{}", mined_harbor::USAGE);
            return ExitCode::from(2);
        }
    };
    let emitted = match mined_harbor::emit_all(&args) {
        Ok(emitted) => emitted,
        Err(msg) => {
            eprintln!("error: {msg}");
            return ExitCode::from(1);
        }
    };
    for (id, path) in &emitted {
        match mined_harbor::emitted_line(&args, id, path) {
            Ok(line) => println!("{line}"),
            Err(msg) => {
                eprintln!("error: {msg}");
                return ExitCode::from(1);
            }
        }
    }
    let out = args.out.display();
    println!("base image: {out}/_base (build with scripts/harbor-build-base.sh {out})");
    println!("{}", mined_harbor::run_hint(&args.out));
    ExitCode::SUCCESS
}
