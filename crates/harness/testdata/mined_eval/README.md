# Fixture files for mined_eval tests

These files are verbatim gate-output captures (stdout + stderr from the sealed re-gate) used as test input to pin specific scorer defects. They MUST NOT be hand-edited — a new case gets a new capture, never a mutation of an existing one.

Fixtures may only be sourced from a `<task-id>` directory that also exists under `~/git/talos-evals/tasks/`, NEVER from a `synth-task-*` directory (there are 120+ such test-residue dirs under the same state root, and they are exactly the headerless shape this change is defending against).

Verify a fixture with `sha256sum` against the hash recorded below before trusting a test that reads it. The captures live under `~/.local/state/talos/mined-eval/` on the workstation that ran the matrix — they are NOT present on a dispatch host, so a headless build agent cannot source them. Adding a fixture is therefore lead-side work: copy the bytes in locally, confirm the hash, and only then let the test assert against it.

## deadlock-trial-0.txt

Source: `~/.local/state/talos/mined-eval/agent-gtd-rollout-deadlock/trial-0/gate-output.txt`
Byte size: 26550
SHA-256: `bc6dc8598c3d2e1c0f99751436274cdfabb87b71979c3c2655e918edc97d6995`

Pins Defect 1: six caplog `ERROR    agent_gtd.event_bus:event_bus.py:130 Failed to fan out event <uuid> to project members` lines before the `short test summary info` header that the old parser incorrectly ingested as phantom test ids, inflating the parsed count above pytest's own total and tripping parse-mismatch.


## from-json-trial-1.txt

Source: `~/.local/state/talos/mined-eval/agent-gtd-from-json-contract/trial-1/gate-output.txt`
Byte size: 2689
SHA-256: `9f03ef22821f50079695c6825ee85c9b7d466e5f4c6c20ceadda64948a78d6e5`

Pins Defect 2: the agent introduced a `SyntaxError: '(' was never closed` at `src/agent_gtd/cli.py` line 136, so pytest reported a collection ERROR (`ERROR tests/test_cli.py`), the positive controls were never collected, and the old scorer returned `Invalid{positive-control-uncollected}` instead of `Unresolved`.


## attribution-trial-0.txt

Source: `~/.local/state/talos/mined-eval/agent-gtd-dispatch-attribution/trial-0/gate-output.txt`
Byte size: 5576
SHA-256: `e5c8435db7ca7942637c26bdb4bbd6eab6491a405887dc40cce003b2a3edb90c`

Pins the `SKIPPED [1] tests/test_dispatch_attribution.py:191: login tool not registered` bracketed-skip form (today keyed incorrectly as `[1]`) and the fact that 14 lines of stderr warning text follow the trailing summary banner (so no implementation may assume the summary line terminates the output).

