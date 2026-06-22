# v1 Inner Loop & Tool Inventory

*Drafted 2026-06-22. Status: proposal for review.*

Scope: the **inner harness** — the per-task loop one dispatched build agent runs
to take a single groomed task to a pushed branch. The **outer harness** (Agent
GTD: grooming, dispatch, wave sequencing, merge/push, review-against-intent) is
out of scope here; it owns *which* work runs and *what happens to the branch
after*. See [The seam](#the-seam). Grounded in `docs/research/` (esp. 00, 01, 03,
05, 06) and the headless-dispatch-first stance.

## Design stance (carried from the research)

- **The tool registry is the boundary.** The agent can only call tools we compile
  in — there is no permission model, allowlist, or bypass concept.
- **"Done" is mechanical:** the project's own quality gates green against the
  task's acceptance criteria — never the model's self-report.
- **Bound everything outside the model** (caps + loop/no-progress detection), in
  deterministic Rust the model can't talk past.
- **Tools return signal, not firehose;** offload the full output and advertise a
  path; prefer precise pointers (`file:line`) over inlined content.
- **Deployment- and backend-agnostic:** a constrained, possibly-ephemeral host
  that may restart mid-run; Anthropic API and Ollama behind one backend trait.

## The loop shape — workflow-around-open-loop

The predictable outer sequence is hard-coded Rust (not model-driven); the open,
hard-bounded loop is reserved for the inner edit/fix phase, where unattended risk
is highest.

```
init run record (task: AC, files, scope, project config)
  -> orient (cheap repo facts)
  -> INNER LOOP (bounded, open): investigate / edit / fix
  -> run_checks (gates)  ── the mechanical done-oracle
       green            -> hand tree back to outer harness for commit/push
       not green, budget left -> back to INNER LOOP
       stuck / budget out     -> bail
  -> finish (disposition: done | blocked | failed + report)
```

Inner-loop iteration: assemble context → model call (backend trait) → model emits
tool calls or `submit` → execute via registry → append results → check budgets +
loop-detection → continue. Termination is mechanical (gates green) **and** bounded
(caps), with bail-with-report as the honest failure exit.

## Tool inventory (v1)

The smallest high-leverage set for a build engine. Consolidated, not granular API
coverage.

| Tool | Purpose | Input (shape) | Output contract |
|---|---|---|---|
| `read_file` | Read a file region | `path`, `offset?`, `limit?` | line-numbered contents; truncation note + offload path if large |
| `list_files` | Glob / directory listing | `glob` or `dir` | paths (capped) + total count |
| `search_code` | ripgrep content search | `pattern`, `path_glob?` | `file:line` + matched line, capped, "N more in `<path>`" |
| `edit_file` | Exact string-replace edit; creates file if absent | `path`, `old_string`, `new_string` (or `content` for new file) | ok + lines changed, **or** a steering error (no-unique-match → candidate locations) |
| `run_command` | Bounded shell workhorse (add a dep, run a script, `git diff`, `ls`) | `command` (argv) | exit code + bounded stdout/stderr extract + full-log offload path |
| `run_checks` | Run the project's declared quality gates | *(none — reads project config)* | structured per-gate pass/fail + failure extracts. **The done-oracle.** |
| `comment` | Progress / disposition note to the task tracker | `body` | ok |
| `finish` | Terminate with a structured disposition | `status: done\|blocked\|failed`, `report` | ends the run; `done` triggers final gate validation |

Notes: `write_file` is folded into `edit_file` (absent file ⇒ create). **Git
writes are not a tool** — the outer harness owns `commit`/`push` after gates pass;
the agent only mutates the working tree (and may *read* git via `run_command`).

## Tool shape & contract

The generalization that protects the context window on every call (see
`docs/research/03`):

- **Rust trait:** `trait Tool { fn name(&self) -> &str; fn schema(&self) ->
  serde_json::Value; async fn run(&self, input, ctx) -> ToolResult; }`.
- **`ToolResult { summary, detail: Option, offload_path: Option, is_error }`** —
  `summary` is always small; `detail` is bounded (~25K char cap); the **full**
  output is persisted to a run-scoped dir and its path advertised, so aggressive
  trimming stays safe (the agent can `read_file` the path for more).
- **Errors are a steering surface,** not loop-crashing exceptions: actionable
  messages, explicit SUCCESS/FAILED, poka-yoke schemas (e.g. `edit_file` returns
  the candidate locations when `old_string` isn't unique).
- **`ToolRegistry`** maps `name → Tool`; the agent's available tools are exactly
  what's registered. This *is* the safety boundary — no separate permission layer.
- **Deterministic JSON** (`BTreeMap` / ordered structs, never `HashMap`) so the
  byte-exact prompt cache actually hits.
- **Signal over firehose, pointers over dumps:** `run_checks` returns failures +
  `file:line`, not the green firehose; `search_code` returns locations, not file
  bodies. "Context for a failure" is a pointer the agent resolves on demand, not
  inlined source.

## The seam (inner ⇄ outer / GTD)

- **In:** the groomed task (acceptance criteria, files, scope) + project config
  (the toolchain commands `run_checks` runs; a model-routing hint). The inner
  harness stays project-agnostic and *reads* what GTD hands it — this is why the
  tool layer never hardcodes "cargo".
- **Out:** a branch (working tree; the outer harness commits/pushes) + a
  structured disposition (`done` / `blocked(needs-decision)` / `failed(retryable)`
  + report, via `comment`/`finish`).

## Should we bundle an LSP? — recommendation: **not in v1**

What an LSP would add over `search_code`+`read_file`: semantic navigation
(go-to-definition, find-references), real-time diagnostics, and safe rename. Real
value — but deferred, for four reasons:

1. **The SOTA coding agents don't rely on it.** Claude Code, OpenHands, and the
   Ralph pattern all run on grep/read/edit + compiler feedback, not LSP. Strong
   evidence it isn't v1-critical; frontier models navigate well with `search_code`.
2. **Diagnostics are already covered.** `run_checks` runs `cargo check`/clippy and
   returns compiler-grade diagnostics. The *unique* LSP value is find-references /
   go-to-def navigation — narrower than it first looks.
3. **Cost vs. our stance.** An LSP is a server process per language (lifecycle,
   JSON-RPC, indexing, edit-sync), heavy on a constrained host (rust-analyzer's
   RAM/indexing), and reintroduces a per-language matrix — fighting the
   toolchain-agnostic design where the project just *declares* its commands.
4. **Let the eval harness decide.** Ship grep/read/edit + `run_checks`; if the
   evals show navigation is the bottleneck, add it then — per language, on
   evidence. The registry boundary means an LSP-backed `find_references` /
   `diagnostics` tool slots in later **without reshaping the loop**. That's the
   whole point of the boundary.

If/when added, `find_references(symbol)` is the canonical high-signal pointer tool
(precise locations vs. grep's textual false positives) — a natural future add, not
a v1 need.

## Explicitly out of v1

- **Permission / allowlist model** — the registry is the boundary.
- **OS sandbox-against-malice** — v2; our threat model is mistakes, not adversaries
  (our groomed tasks, our code, our infra). v1 safety = blast-radius bounds +
  creds-hygiene, **plus the OS account itself**: headless agents already run in
  isolated accounts, so the account is the containment boundary — an in-harness
  sandbox would be redundant in v1.
- **LSP** — deferred; registry-pluggable later.
- **Code-mode / programmatic tool-calling / tool-search** — pays off at >10 tools
  and heavy multi-step orchestration; our ~8-tool set makes direct tool-calling
  simpler. Revisit when the tool count or step depth grows.
- **Git writes as agent tools** — the outer harness owns commit/push.
- **Subagents** — the isolated read-only verifier is high-value, but ships as a
  sequential second pass in v1, not a true subagent (research open-decision #3).

## Open questions

- ~~`run_command` vs `run_checks` boundary~~ — **resolved (2026-06-22):** ship
  both. General `run_command` stays in v1; headless agents run in isolated OS
  accounts (the containment boundary), so arbitrary shell + blast-radius caps +
  creds-hygiene is acceptable without an in-harness permission layer.
- `edit_file`: string-replace only for v1, or also a patch/diff mode?
- `search_code`: bundle a ripgrep binary or shell out to `rg`? Capability-detect
  and fall back to a slower in-process search if absent.
- `finish` disposition schema — this is research open-decision #8 (bail-with-report
  contract); pin it alongside the run-record schema (#4).
- Does `orient` build a small repo map, or stay minimal and JIT everything through
  tools? (Lean: minimal orient, just-in-time via tools.)
