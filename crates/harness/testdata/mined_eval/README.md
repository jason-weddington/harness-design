# Gate-output captures for the tier-2 scorer tests

These files are verbatim gate-output captures (stdout + stderr from the sealed re-gate), read by `harbor/tests/test_mined_score.py` as test input to pin specific scorer defects. The test reads them by this path, which is why the directory keeps its name. They MUST NOT be hand-edited — a new case gets a new capture, never a mutation of an existing one. The one exception is redaction of the capturing machine's home directory to `/home/user/` (done 2026-10-13 for the two files below whose size and hash changed), which touches only traceback paths that no parser reads.

Fixtures may only be sourced from a `<task-id>` directory that also exists under `~/git/talos-evals/tasks/`, NEVER from a `synth-task-*` directory (there are 120+ such test-residue dirs under the same state root, and they are exactly the headerless shape this change is defending against).

Verify a fixture with `sha256sum` against the hash recorded below before trusting a test that reads it. The captures live under `~/.local/state/talos/mined-eval/` on the workstation that ran the matrix — they are NOT present on a dispatch host, so a headless build agent cannot source them. Adding a fixture is therefore lead-side work: copy the bytes in locally, confirm the hash, and only then let the test assert against it.

## deadlock-trial-0.txt

Source: `~/.local/state/talos/mined-eval/agent-gtd-rollout-deadlock/trial-0/gate-output.txt`
Byte size: 26550
SHA-256: `bc6dc8598c3d2e1c0f99751436274cdfabb87b71979c3c2655e918edc97d6995`

Pins Defect 1: six caplog `ERROR    agent_gtd.event_bus:event_bus.py:130 Failed to fan out event <uuid> to project members` lines before the `short test summary info` header that the old parser incorrectly ingested as phantom test ids, inflating the parsed count above pytest's own total and tripping parse-mismatch.


## from-json-trial-1.txt

Source: `~/.local/state/talos/mined-eval/agent-gtd-from-json-contract/trial-1/gate-output.txt`
Byte size: 2688
SHA-256: `7beb59bae3848fa04d848e7c03b651b1e7f7ef7138afeed66085633ebb9fb1c1`

Pins Defect 2: the agent introduced a `SyntaxError: '(' was never closed` at `src/agent_gtd/cli.py` line 136, so pytest reported a collection ERROR (`ERROR tests/test_cli.py`), the positive controls were never collected, and the old scorer returned `Invalid{positive-control-uncollected}` instead of `Unresolved`.


## attribution-trial-0.txt

Source: `~/.local/state/talos/mined-eval/agent-gtd-dispatch-attribution/trial-0/gate-output.txt`
Byte size: 5570
SHA-256: `3d96d8fd13672c6c7ebcf3fa9ecb329506c42a210267e91e7c47d4e15742868e`

Pins the `SKIPPED [1] tests/test_dispatch_attribution.py:191: login tool not registered` bracketed-skip form (today keyed incorrectly as `[1]`) and the fact that 14 lines of stderr warning text follow the trailing summary banner (so no implementation may assume the summary line terminates the output).

