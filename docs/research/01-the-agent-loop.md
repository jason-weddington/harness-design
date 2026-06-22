# The Agent Loop

*Researched 2026-06-20.*

The agent loop is the engine that turns a language model into an agent: a model
that can *act* on the world, observe the result, and decide what to do next —
repeatedly — until the task is done. This doc establishes the canonical loop,
the line between fixed workflows and open autonomous loops, the principle of
keeping control flow in *our* code, and — most importantly for an unattended
build engine — how a loop with no human watching knows whether it is **done**,
**stuck**, or **looping**.

---

## 1. What an agent actually is

The most useful working definition comes from Thorsten Ball: an agent is "an LLM
with *access to tools*, giving it the ability to modify something outside the
context window," and building one "comes down to" — in his words — "an LLM, a
loop, and enough tokens" ([How to Build an Agent, ampcode.com](https://ampcode.com/how-to-build-an-agent)).
There is no arcane machinery. The intelligence lives in the model; the harness
supplies the loop and the tools.

Anthropic frames the same idea structurally as the **augmented LLM**: a model
combined with "augmentations such as retrieval, tools, and memory," where modern
models "actively use these capabilities — generating their own search queries,
selecting appropriate tools, and determining what information to retain"
([Building Effective Agents, Anthropic](https://www.anthropic.com/engineering/building-effective-agents)).
The augmented LLM is the *building block*; the loop is what makes it agentic.

---

## 2. The canonical loop

Both primary sources describe the same cycle. Ball's `Run()` loop, stripped to
its essence ([ampcode.com](https://ampcode.com/how-to-build-an-agent)):

```
for {
  [get user input]
  conversation.append(userMessage)
  message = runInference(conversation)      // call the model with full history + tool defs
  conversation.append(message)
  [process response content]
  if toolResults found:
    conversation.append(toolResults)        // feed results back
    [loop WITHOUT asking the user]
  else:
    readUserInput = true                    // model produced text, not a tool call -> turn over
}
```

The canonical phases:

1. **Gather context** — assemble the conversation/state plus the tool
   definitions.
2. **Call the model** — one inference call. The model replies either with text
   (it's talking to the caller) or with one or more `tool_use` blocks (it wants
   to act).
3. **Execute tools** — deterministic code dispatches each requested tool by name
   and runs it.
4. **Feed results back** — tool outputs are appended to the conversation (in the
   Anthropic API, as a `tool_result` message keyed by the tool-use id).
5. **Repeat** — if the model called tools, loop again *without* yielding to the
   user; if it produced only text, the turn is over.

Anthropic describes the autonomous variant identically: agents are LLMs "using
tools based on environmental feedback in a loop," gaining "ground truth from the
environment at each step (such as tool call results or code execution) to assess
[their] progress" ([Anthropic](https://www.anthropic.com/engineering/building-effective-agents)).
That phrase — *ground truth from the environment at each step* — is the whole
game for an unattended coding agent: the build, the test suite, and the linter
*are* the environment, and they are what tells the agent (and the harness)
whether work is real.

The crucial structural insight from 12-factor agents: the model only ever
*decides* the next step; **deterministic code executes it, appends the result,
and loops** ([12-factor agents](https://github.com/humanlayer/12-factor-agents)).
The model proposes; the harness disposes.

---

## 3. Workflows vs. autonomous agents

Anthropic draws the canonical distinction ([Anthropic](https://www.anthropic.com/engineering/building-effective-agents)):

- **Workflows** — "systems where LLMs and tools are orchestrated through
  predefined code paths."
- **Agents** — "systems where LLMs dynamically direct their own processes and
  tool usage, maintaining control over how they accomplish tasks."

The decision rule: use a **fixed workflow** when the task "decompose[s] into
predictable steps and you need consistency"; reach for an **autonomous loop**
for open-ended problems where "it's difficult or impossible to predict the
required number of steps, and where you can't hardcode a fixed path." And the
overarching counsel: *start simple* — "optimizing single LLM calls with
retrieval and in-context examples is usually enough," and you should add agentic
complexity only when simpler approaches fall short, because agents trade latency
and cost for better task performance.

Anthropic catalogs five composable workflow patterns that sit *below* a full
autonomous agent ([Anthropic](https://www.anthropic.com/research/building-effective-agents)):

| Pattern | What it does | Relevance to a build engine |
|---|---|---|
| **Prompt chaining** | Sequence of steps, each consuming the prior output; trades latency for accuracy | The macro shape of a dispatch: groom -> implement -> verify -> ship |
| **Routing** | Classify input, dispatch to a specialized follow-up | Route by task type / by model backend (Haiku for cheap steps, Opus for hard ones) |
| **Parallelization** | LLMs work simultaneously, outputs aggregated in code | Parallel critics / parallel file edits |
| **Orchestrator-workers** | A central LLM decomposes a task and delegates to workers | A manage-agent driving sub-tasks |
| **Evaluator-optimizer** | One LLM generates, another evaluates and feeds back corrections in a loop | **The self-verification spine of an unattended agent** |

The evaluator-optimizer pattern is the most important one for us: it is the
in-harness substitute for the human who would otherwise eyeball the diff. More
on this in §6.

**The unattended lens.** With a human in the loop, a too-open agent loop is
*tolerable* — the human catches the wrong turn. Unattended, that safety net is
gone, so the bias should shift toward **more workflow, less open loop** wherever
the steps *are* predictable. A headless build engine's outer shape is a known
sequence (understand task -> change code -> run the project's own gates -> fix
-> commit -> push -> comment). Only the *inner* "change code / fix" phase
genuinely needs an open loop. Hardcode the predictable scaffolding; reserve
model-directed looping for the part that is irreducibly open-ended.

---

## 4. Own your loop — keep control flow in code

This is the load-bearing principle for a from-scratch harness, and 12-factor
agents states it most sharply.

**Factor 8 — Own Your Control Flow.** "Take direct responsibility for deciding
which steps execute next rather than delegating this entirely to framework
abstractions" ([12-factor agents, factor 8](https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-08-own-your-control-flow.md)).
The loop is a plain `while`/`for` in *your* code with a switch over the model's
chosen action, not a black box inside a framework:

```
while True:
    next_step = determine_next_step(thread)   # one model call
    match next_step.intent:
        case "run_tests":      append(result); continue   # sync: loop immediately
        case "request_review": save_state(); break         # async: yield, resume later
        case "done":           break                        # terminal
```

Sync actions (fetch data, run a tool) `continue` the loop with the new context;
async actions (anything needing an out-of-band event) `break` and resume on a
signal. Owning this switch is what lets *you* — not a framework — insert budget
checks, loop-detection, and verification gates at exactly the right seams.

**Factor 12 — Make Your Agent a Stateless Reducer.** Structure the agent as a
pure function: `(state, context_history) -> next_action`, with no hidden state
between calls ([12-factor agents](https://github.com/humanlayer/12-factor-agents)).
This is what makes a run **interruptible and resumable** — and for an agent on a
constrained, possibly-ephemeral host (a Pi, a container, a Fargate task — the harness
shouldn't assume which) that may be killed, restarted, evicted, or run out of budget
mid-task, resumability is not a nicety.

**Factor 5 — Unify Execution State and Business State.** Merge the agent's
execution tracking with the real work artifacts so "what the agent knows" and
"what the system records" cannot drift apart ([12-factor agents](https://github.com/humanlayer/12-factor-agents)).
The Ralph loop is the extreme expression of this: it uses *the filesystem and
git history as memory* rather than an in-process conversation buffer — green
builds, a mutable `fix_plan.md`, frozen `specs/*.md`, and structured commits all
serve as durable state across iterations ([Ralph Loop, Thomas Wiegold](https://thomas-wiegold.com/blog/ralph-loop-how-recursive-ai-agents-work/)).
For our harness, the equivalent is: the git working tree, the task-tracker item,
and a small structured run-log are the source of truth; the in-context
conversation is disposable scratch.

**Factor 9 — Compact Errors Into the Context Window.** When tools fail, distill
the error into a compact form before feeding it back, "preventing bloat while
maintaining necessary debugging signals" ([12-factor agents](https://github.com/humanlayer/12-factor-agents)).
See §7.

---

## 5. Stop / convergence conditions — the heart of unattended operation

Interactively, *the human* is the stop condition: they read the output and say
"that's it" or "you're going in circles." Headless, **the harness must own all
three of: done, stuck, and looping.** The field has converged on a
**multi-layered termination** strategy — no single signal is trusted; several
independent backstops run in parallel ([DEV/AWS: prevent reasoning loops](https://dev.to/aws/how-to-prevent-ai-agent-reasoning-loops-from-wasting-tokens-2652),
[BSWEN: stop infinite loops](https://docs.bswen.com/blog/2026-03-11-prevent-ai-agent-infinite-loops/)).

### 5a. "Done" — positive completion

The weakest-but-necessary signal is the model declaring completion. Anthropic's
loop ends when the model stops emitting tool calls (it produces a final text
answer). Ball's loop returns control to the user on the same condition. Ralph
implementations use an explicit sigil — e.g. emitting `<promise>COMPLETE</promise>`
to break the loop ([Ralph Loop](https://thomas-wiegold.com/blog/ralph-loop-how-recursive-ai-agents-work/)).

But a self-declared "done" is **not trustworthy on its own** — the documented
failure mode is exactly the agent that *thinks* it's making progress forever. So
"done" for a build engine must be **mechanical, not declared**: tests pass, build
is green, lint count is at/under target, the diff exists. Ralph's completion
criteria are precisely these "measurable criteria that trigger commits"
([Ralph Loop](https://thomas-wiegold.com/blog/ralph-loop-how-recursive-ai-agents-work/)).
Treat the model's "I'm done" as a *proposal to run the gates*, never as the
terminal state itself.

### 5b. Hard backstops — always-on, non-negotiable

These are dumb, cheap, and impossible for a confused model to talk its way past.
The consensus set ([DEV/AWS](https://dev.to/aws/how-to-prevent-ai-agent-reasoning-loops-from-wasting-tokens-2652),
[BSWEN](https://docs.bswen.com/blog/2026-03-11-prevent-ai-agent-infinite-loops/),
[Ralph Loop](https://thomas-wiegold.com/blog/ralph-loop-how-recursive-ai-agents-work/)):

- **Max-iteration cap** — Anthropic explicitly recommends "stopping conditions
  (such as a maximum number of iterations) to maintain control"
  ([Anthropic](https://www.anthropic.com/engineering/building-effective-agents)).
  Ralph runners use flags like `--max-iterations 50`.
- **Token / cost budget** — a hard spend ceiling that *kills the run* when
  exhausted, regardless of state. The cautionary tale: an agent that ran "847
  reasoning steps at \$47 per minute without delivering a final answer"
  ([DEV/AWS](https://dev.to/aws/how-to-prevent-ai-agent-reasoning-loops-from-wasting-tokens-2652)).
  CodiesHub's framing: "Iterations, tokens, time, spend are non-negotiable."
- **Per-tool call caps** — cap how many times a *single* tool may be invoked per
  task (e.g. `LimitToolCounts`), blocking with an explicit message at the
  ceiling ([DEV/AWS](https://dev.to/aws/how-to-prevent-ai-agent-reasoning-loops-from-wasting-tokens-2652)).
- **Wall-clock circuit breaker** — an absolute timeout independent of progress.
  Ralph practice: per-iteration timeouts (~15 min typical) plus hourly token
  caps ([Ralph Loop](https://thomas-wiegold.com/blog/ralph-loop-how-recursive-ai-agents-work/)).

### 5c. Convergence / loop detection — catching "stuck" before the budget burns

Hard caps stop runaway cost but waste it first. Loop detection catches a stuck
agent *early*. Techniques, cheapest first:

- **Structural repetition ("boredom") detection** — track the last N actions and
  flag exact repeats: same tool, same parameters, same output shape. A
  `DebounceHook` blocks a duplicate `(tool_name, input)` pair seen twice within a
  small window (e.g. 3) by cancelling the call with `BLOCKED: Duplicate call
  detected` ([DEV/AWS](https://dev.to/aws/how-to-prevent-ai-agent-reasoning-loops-from-wasting-tokens-2652)).
- **Domain-specific stuck detection** — e.g. "same test failing three iterations
  in a row -> stop" ([Ralph Loop](https://thomas-wiegold.com/blog/ralph-loop-how-recursive-ai-agents-work/)).
  For a build engine, "lint count not decreasing across K iterations" or "diff
  unchanged across K iterations" is a strong, cheap convergence signal.
- **Semantic similarity of consecutive states** — compute similarity between
  successive observation/thought/action triples and intervene above a threshold
  (one tool reports ~85%) ([Markaicode](https://markaicode.com/fix-ai-agent-looping-autonomous-coding/),
  search-aggregated). More expensive and fuzzier; useful as a backstop where
  structural detection misses paraphrased repetition.

### 5d. Intervention before termination

When a loop is *detected* but the budget isn't yet spent, the graded response is:
first inject a **reflection prompt** forcing the agent to reassess; if still
stuck, **suggest an alternative tool/approach**; only then terminate
([loop-detection writeups, search-aggregated](https://markaicode.com/fix-ai-agent-looping-autonomous-coding/)).
For us, an unattended escalation ladder ends not at "ask the human" but at a
clean **bail-with-report**: stop, write what was tried and why it's stuck, and
comment that back on the task item rather than burning the full budget.

---

## 6. Self-verification: the in-harness stand-in for the human reviewer

Because no human reviews mid-run, the **evaluator-optimizer** workflow becomes
load-bearing: a generator produces a change, a *separate* evaluator judges it
against criteria, and corrections feed back in a loop until it passes
([Anthropic](https://www.anthropic.com/research/building-effective-agents)).
Two kinds of evaluator matter for a build engine, and they are not
interchangeable:

- **Mechanical evaluators (primary).** The project's own gates — compile, test,
  lint, type-check — are deterministic ground truth. Ralph calls these "the
  measurable criteria that trigger commits" and treats type-checking, linting,
  and tests as "non-negotiable backpressure" ([Ralph Loop](https://thomas-wiegold.com/blog/ralph-loop-how-recursive-ai-agents-work/)).
  These should gate every commit; a self-declared "done" that hasn't passed them
  is meaningless.
- **Model-judge evaluators (secondary).** A second model pass that checks the
  diff against the task's acceptance criteria — "did this actually implement what
  was asked, and only that?" — catches the class of error gates miss: passing
  tests that verify the wrong thing, scope creep, a green build that solves a
  different problem. This is the headless equivalent of the human who reads the
  PR. It should *advise*, not silently auto-merge, when its verdict is negative.

---

## 7. Tool errors inside the loop

Tool failures are normal control flow, not exceptional events — the loop must
absorb them and let the model recover.

- **Feed errors back as observations, compacted.** A failed tool returns an
  error *result*, not a thrown exception that crashes the loop. 12-factor's
  Factor 9 says to compact the error into the context window — keep the signal,
  drop the noise — so a 4,000-line stack trace becomes the three lines the model
  needs ([12-factor agents](https://github.com/humanlayer/12-factor-agents)).
- **Make tools return explicit terminal states.** Ambiguous results ("more
  results may be available") cause the model to retry organically and loop;
  unambiguous `SUCCESS: ...` / `FAILED: ...` markers let the model recognize
  completion. One demo cut tool calls from 14 to 2 — a 7x reduction — purely by
  making terminal states explicit ([DEV/AWS](https://dev.to/aws/how-to-prevent-ai-agent-reasoning-loops-from-wasting-tokens-2652)).
- **Design tools to be hard to misuse (poka-yoke).** Anthropic urges treating the
  agent-computer interface (ACI) with the same care as a human UI: "change the
  arguments so that it is harder to make mistakes," avoid formatting overhead the
  model must track, and document tools with "example usage, edge cases, input
  format requirements, and clear boundaries from other tools"
  ([Anthropic](https://www.anthropic.com/engineering/building-effective-agents)).
  The cheapest tool error is the one the schema makes impossible.
- **Bound retries.** Tie repeated failures into loop-detection (§5c): the same
  tool failing identically K times is a stuck signal, not an invitation to a
  K+1th attempt.

---

## 8. Where sources agree and disagree

- **Agreement: the loop is simple.** Ball, Anthropic, and 12-factor all describe
  the same minimal cycle. There is broad consensus that the model supplies the
  intelligence and the harness supplies a thin, owned loop.
- **Agreement: bound everything.** Every source on unattended operation
  independently lands on multi-layered hard limits (iterations, tokens, time,
  per-tool) plus progress/loop detection. This is the strongest consensus in the
  current literature.
- **Tension: how open should the loop be?** Anthropic's headline advice is *start
  simple, prefer workflows, add autonomy only when needed* — a conservative,
  predictability-first stance. The Ralph community embraces a maximally open
  `while true` loop with a *fresh* context per iteration, leaning on filesystem
  state and mechanical gates rather than a tightly-scripted workflow. Both
  converge on the same safety mechanisms; they disagree on how much structure to
  impose *around* the model. For an unattended build engine the resolution is by
  phase: scripted workflow for the predictable outer shell, an open (but
  hard-bounded) loop only for the inner code-edit phase.
- **Tension: trust the model's "done"?** The Ralph pattern is comfortable with a
  model-emitted completion sigil; the loop-prevention literature warns that a
  self-declared "done" is exactly what fails silently. The reconciliation
  everyone implicitly reaches: a declared "done" may *trigger* verification but
  must never *be* the terminal state — mechanical gates are.

---

## Implications for our headless-dispatch harness

- **Own a thin, explicit loop in Rust with a `match` over the model's chosen
  action — never delegate control flow to a framework.** That switch is where we
  inject budget checks, loop-detection, and verification gates. Keep the model in
  the "decide next step" role only; deterministic Rust executes, appends, and
  decides whether to continue (12-factor Factor 8). Build the agent as a stateless
  reducer (`(state, history) -> action`) so a run survives a host restart (reboot, container eviction, spot reclaim, laptop sleep) and resumes.

- **Make termination multi-layered and non-negotiable, because there is no human
  stop condition.** Ship all of: max-iteration cap, hard token/cost ceiling that
  kills the run, per-tool call caps, and a wall-clock circuit breaker — on by
  default, configurable per dispatch. Add cheap structural loop-detection (same
  tool+args repeated; diff/lint-count unchanged across K iterations; same test
  failing K times) so we catch "stuck" before the budget is spent, not after.

- **"Done" is mechanical, never declared.** The terminal state is the project's
  own gates going green (build/test/lint/type-check), not the model saying it
  finished. Treat a model "I'm done" as a trigger to run the gates. A run that
  can't pass them ends in **bail-with-report** — comment back what was tried and
  why it's stuck — not in a false success.

- **Bake evaluator-optimizer in as the human-reviewer stand-in.** Mechanical
  gates are the primary evaluator and gate every commit; a secondary model-judge
  pass checks the diff against the task's acceptance criteria to catch
  green-but-wrong and scope-creep. On a negative judge verdict, flag the branch as
  needs-review rather than auto-shipping.

- **Structure the run as workflow-around-open-loop, not one big open loop.**
  Hardcode the predictable outer sequence (orient -> edit -> run gates -> fix ->
  commit -> push -> comment); reserve the bounded open loop for the inner
  code-edit/fix phase only. This honors Anthropic's "prefer workflows" guidance
  exactly where unattended risk is highest.

- **Treat tool errors as first-class, compacted observations, and design tools to
  be hard to misuse.** A failing tool returns a compact `FAILED: ...` result the
  model can recover from — never an exception that crashes the loop (Factor 9).
  Tools return explicit `SUCCESS`/`FAILED` terminal states (cuts pointless
  retries), use poka-yoke schemas, and carry rich descriptions. Use the
  filesystem + git + the task item as durable cross-iteration state (à la Ralph),
  keeping the in-context conversation disposable — which is also what enables
  fresh-context restarts before quality degrades past ~100-150k tokens.
