//! The prompt-authoring layer: versioned template files rendered into the
//! system + task prompts the loop feeds the model.
//!
//! # Design
//!
//! - **Own your prompts.** The prompt text lives in this repo as `.md` files
//!   under `crates/harness/templates/`. A prompt change is a text-file git
//!   diff — reviewable, blame-able, revertable.
//! - **Compile-time-checked.** Templates are embedded via [`askama`]'s derive
//!   macro. A template that references a missing field is a *build* error, in
//!   keeping with "rustc is a gate" — the templates cannot silently drift
//!   from the render context.
//! - **Deliberately decoupled.** The render API takes plain strings and small
//!   owned structs. It does NOT depend on `exec::CheckCommand`, on
//!   `tools/*`, or on any in-flight sibling item. The next item (engine
//!   wiring) is what glues the pieces together.
//! - **HTML escaping is off.** The templates are `.md`, and each template's
//!   derive attribute pins `escape = "none"` explicitly so a filename change
//!   cannot accidentally start escaping the prompt text. A dedicated test
//!   pins the behaviour with `"`, `<`, `>`, and `&` characters.
//! - **Determinism.** Rendering is a pure function of its inputs: same
//!   inputs, byte-identical output. The prompt cache depends on this. Tool
//!   ordering is the caller's responsibility — [`ToolRegistry`] is a
//!   [`std::collections::BTreeMap`], so [`tool_lines`] returns entries in a
//!   stable name order.
//!
//! # Public surface
//!
//! - [`ToolLine`] — the `(name, description)` pair the system prompt lists,
//!   extracted from a tool's advertised schema JSON.
//! - [`tool_lines`] — build a [`Vec<ToolLine>`] from a [`ToolRegistry`],
//!   tolerating schemas that omit `description`.
//! - [`render_system_prompt`] — render the system prompt.
//! - [`render_task_prompt`] — render the task-framing prompt.
//! - [`render_task_prompt_from_spec`] — render the task prompt from a groomed
//!   [`crate::task_spec::TaskSpec`], producing the `{{ task }}` slot content.
//! - [`render_answer_prompt`] — render ANSWER mode's task-framing content
//!   (the question plus the result schema, verbatim) into the `{{ task }}`
//!   slot.
//! - [`render_answer_system_prompt`] — render ANSWER mode's system prompt: a
//!   read-only analyst frame whose only terminal dispositions are `answer`,
//!   `blocked` and `failed`. A DISTINCT template from
//!   [`render_system_prompt`], deliberately — the build-mode prompt is the
//!   eval-parity surface and must stay byte-identical.

use askama::Template;
use serde_json::Value;

use crate::task_spec::{FileToModify, TaskSpec};
use crate::tool::ToolRegistry;

/// One entry in the system prompt's tool listing: the tool's registered name
/// plus its human-readable description, both extracted from the schema JSON
/// the tool advertises.
///
/// This is deliberately a small owned struct rather than a reference into
/// the registry — the render layer is decoupled from the tool layer's
/// concrete types, and the copy is cheap (one render per run).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolLine {
    /// The tool's stable registered name (used in `- name — description`).
    pub name: String,
    /// The tool's description. Empty string when the schema omits one —
    /// never a panic. An empty description still renders (`- name — `),
    /// which surfaces the omission to the model rather than hiding it.
    pub description: String,
}

/// Extract the `(name, description)` lines the system prompt lists from a
/// [`ToolRegistry`]'s advertised schemas.
///
/// The registry is a [`std::collections::BTreeMap`] internally, so
/// [`ToolRegistry::list`] returns schemas in a deterministic name order —
/// the returned [`Vec<ToolLine>`] preserves that order, which is what makes
/// [`render_system_prompt`] byte-deterministic across runs (a prompt-cache
/// requirement).
///
/// **Tolerant of a missing `description`.** The schema for a tool is JSON
/// authored by the tool itself; a tool that forgets to include a
/// `description` field will still show up in the listing with an empty
/// description, not blow the whole render up. A schema without a `name` is
/// skipped — a nameless tool cannot be advertised at all.
#[must_use]
pub fn tool_lines(registry: &ToolRegistry) -> Vec<ToolLine> {
    registry
        .list()
        .into_iter()
        .filter_map(|schema| {
            let name = schema.get("name").and_then(Value::as_str)?.to_string();
            let description = schema
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            Some(ToolLine { name, description })
        })
        .collect()
}

/// The system prompt template. `escape = "none"` is pinned explicitly so
/// prompt text with `"`, `<`, `>`, or `&` passes through untouched — the
/// model sees exactly what a reviewer sees in the `.md` file.
#[derive(Template)]
#[template(path = "system_prompt.md", escape = "none")]
struct SystemPromptTemplate<'a> {
    tools: &'a [ToolLine],
    check_command: Option<&'a str>,
}

/// The task-framing template. Kept near-passthrough — a heading line plus
/// the task verbatim. Structure grows later (AC lists, scope) when the GTD
/// adapter lands.
#[derive(Template)]
#[template(path = "task_prompt.md", escape = "none")]
struct TaskPromptTemplate<'a> {
    task: &'a str,
}

/// The groomed-item task-spec template. Renders a [`TaskSpec`] into the
/// content of the `{{ task }}` slot — the engine wraps it under `# Task`
/// via [`render_task_prompt`]. `escape = "none"` is pinned so prompt text
/// with `"`, `<`, `>`, or `&` passes through untouched.
#[derive(Template)]
#[template(path = "task_spec_prompt.md", escape = "none")]
struct TaskSpecPromptTemplate<'a> {
    title: &'a str,
    description: &'a str,
    acceptance_criteria: &'a [String],
    files_to_modify: &'a [FileToModify],
    gate_command: &'a str,
    include_test_first: bool,
}

/// The test-first ("red, green") approach guidance, factored into its own
/// template so the two surfaces that need it cannot drift apart:
/// [`TaskSpecPromptTemplate`] `{% include %}`s it for the groomed-item path
/// (talos dispatch, tier-1 coding eval), and [`render_test_first_approach`]
/// renders it standalone for the tier-2 mined-task path, whose agent prompt is
/// the raw mined statement and never passes through the task-spec template.
/// [`VerificationSectionTemplate`] is the sibling factoring for the
/// `## Verification` block, for the same reason.
#[derive(Template)]
#[template(path = "test_first_approach.md", escape = "none")]
struct TestFirstApproachTemplate;

/// The `## Verification` block, factored into its own template so the two
/// surfaces that need it cannot drift apart: [`TaskSpecPromptTemplate`]
/// `{% include %}`s it for the groomed-item path (talos dispatch, tier-1
/// coding eval), and [`render_verification_section`] renders it standalone
/// for the tier-2 mined-task path (agent-gate On mode), whose agent prompt is
/// the raw mined statement and never passes through the task-spec template.
/// `escape = "none"` is pinned so a `gate_command` containing `"`, `<`, `>`,
/// or `&` passes through untouched.
#[derive(Template)]
#[template(path = "verification_section.md", escape = "none")]
struct VerificationSectionTemplate<'a> {
    gate_command: &'a str,
}

/// The ralph outer-loop per-iteration prompt template. Renders the objective,
/// the injected progress notes, and the exact notes filename the agent must
/// append to next. `escape = "none"` is pinned so prompt text with `"`, `<`,
/// `>`, or `&` passes through untouched — the model sees exactly what a
/// reviewer sees in the `.md` file.
#[derive(Template)]
#[template(path = "ralph_prompt.md", escape = "none")]
struct RalphPromptTemplate<'a> {
    objective: &'a str,
    notes: &'a str,
    notes_file: &'a str,
}

/// The synthetic-nudge template — a variable-free string the finish-recovery
/// protocol appends onto the existing tool-results `Message::User` when it
/// detects a green-gates + static-tree spin. `escape = "none"` is pinned
/// explicitly so the backticks around `finish(done)` pass through untouched.
#[derive(Template)]
#[template(path = "nudge_prompt.md", escape = "none")]
struct NudgePromptTemplate;

/// Answer mode's task-framing template: the question, the result schema
/// verbatim inside a fenced `json` block, and the three rules the frame states.
/// Renders the content of the `{{ task }}` slot — the engine wraps it under
/// `# Task` via [`render_task_prompt`], so every heading here is `##` or
/// lower. `escape = "none"` is pinned so a schema containing `"`, `<`, `>`,
/// or `&` reaches the model byte-for-byte as supplied; a schema the model
/// sees in escaped form is a schema it cannot conform to.
#[derive(Template)]
#[template(path = "answer_prompt.md", escape = "none")]
struct AnswerPromptTemplate<'a> {
    question: &'a str,
    schema_text: &'a str,
}

/// Answer mode's system prompt template — a DISTINCT file from
/// `system_prompt.md`, not a branch inside it. Keeping them separate is the
/// point: `system_prompt.md` is the build-mode surface the tier-1/tier-2
/// evals measure, and answer mode must be able to change its own frame
/// without touching a byte of it. Takes no `check_command`: there is no
/// verification section in answer mode, because the mechanical verifier is
/// the result schema, not a gate.
#[derive(Template)]
#[template(path = "answer_system_prompt.md", escape = "none")]
struct AnswerSystemPromptTemplate<'a> {
    tools: &'a [ToolLine],
}

/// Render the system prompt.
///
/// - `tools` is the ordered listing (typically from [`tool_lines`]).
/// - `check_command` is a pre-rendered display string of the run's check
///   command(s). It is [`Option`]-typed because a run may have no checks
///   configured; the template renders a different verification section for
///   each case (an enforcement contract when checks are configured, a "no
///   checks" advisory when they are not).
///
/// The render is a pure function of its inputs and is byte-deterministic —
/// re-rendering with the same inputs produces the same bytes.
///
/// # Panics
/// Never in practice. Panics only if askama's own formatter fails, which
/// cannot happen for these templates (they only render owned data through
/// `Display`; there are no filters that can fail).
#[must_use]
pub fn render_system_prompt(tools: &[ToolLine], check_command: Option<&str>) -> String {
    SystemPromptTemplate {
        tools,
        check_command,
    }
    .render()
    .expect("system_prompt.md is a static template that renders infallibly for owned inputs")
}

/// Render the task-framing prompt: a heading plus the task text verbatim.
///
/// # Panics
/// Never in practice — see [`render_system_prompt`].
#[must_use]
pub fn render_task_prompt(task: &str) -> String {
    TaskPromptTemplate { task }
        .render()
        .expect("task_prompt.md is a static template that renders infallibly for owned inputs")
}

/// Render the task-spec prompt from a groomed GTD [`TaskSpec`].
///
/// Returns the content of the `{{ task }}` slot — the engine wraps it under
/// a `# Task` heading via [`render_task_prompt`] when it wires the spec into
/// a [`crate::engine::RunConfig`]. The output therefore begins with a
/// `##`-level heading (the title) rather than a top-level `# `, so the
/// two-level nesting composes correctly.
///
/// The rendered output covers all five [`TaskSpec`] fields: title (heading),
/// description (prose), acceptance criteria (bullet list), files to modify
/// (bullet list with navigation framing), and the gate command (verbatim
/// shell command). Both list sections loop over their entire collections, so
/// zero-element lists render without error.
///
/// The render is a pure function of its inputs and is byte-deterministic —
/// re-rendering with the same [`TaskSpec`] produces identical bytes.
///
/// # Panics
/// Never in practice — see [`render_system_prompt`].
#[must_use]
pub fn render_task_prompt_from_spec(spec: &TaskSpec) -> String {
    render_task_prompt_from_spec_with(spec, true)
}

/// [`render_task_prompt_from_spec`] with the test-first approach section
/// toggleable.
///
/// Exists so an eval can measure the guidance's effect by running the SAME
/// harness with and without it. Production (talos dispatch) always renders it —
/// `render_task_prompt_from_spec` hard-codes `true` — so a toggle can never
/// silently disable it on the shipping path; only a caller that explicitly asks
/// for `false` gets the stripped prompt.
///
/// The render is a pure function of its inputs and is byte-deterministic.
///
/// # Panics
/// Never in practice — see [`render_system_prompt`].
#[must_use]
pub fn render_task_prompt_from_spec_with(spec: &TaskSpec, include_test_first: bool) -> String {
    TaskSpecPromptTemplate {
        title: &spec.title,
        description: &spec.description,
        acceptance_criteria: &spec.acceptance_criteria,
        files_to_modify: &spec.files_to_modify,
        gate_command: &spec.gate_command,
        include_test_first,
    }
    .render()
    .expect("task_spec_prompt.md is a static template that renders infallibly for owned inputs")
}

/// Render the ralph per-iteration prompt: the objective, the injected
/// progress notes, and the exact notes filename the agent must append to.
///
/// The output is the content of the inner [`crate::engine::RunConfig::task`]
/// slot — the engine wraps it under `# Task` via [`render_task_prompt`], so it
/// must NOT itself begin with a top-level `# ` heading (all headings are `##`
/// or lower). The render is a pure function of its inputs and is
/// byte-deterministic.
///
/// # Panics
/// Never in practice — see [`render_system_prompt`].
#[must_use]
pub fn render_ralph_prompt(objective: &str, notes: &str, notes_file: &str) -> String {
    RalphPromptTemplate {
        objective,
        notes,
        notes_file,
    }
    .render()
    .expect("ralph_prompt.md is a static template that renders infallibly for owned inputs")
}

/// Render the synthetic-nudge steering text — the variable-free string the
/// finish-recovery protocol appends onto the existing tool-results
/// `Message::User` when it detects a green-gates + static-tree spin. The
/// exact wording is pinned by an engine test
/// (`green_static_spin_injects_exactly_one_nudge_with_pinned_text`).
///
/// The render is a pure function of its (empty) inputs and is
/// byte-deterministic — re-rendering produces identical bytes.
///
/// # Panics
/// Never in practice — see [`render_system_prompt`].
#[must_use]
pub fn render_nudge_prompt() -> String {
    NudgePromptTemplate
        .render()
        .expect("nudge_prompt.md is a static variable-free template that renders infallibly")
}

/// Render the test-first approach guidance on its own.
///
/// For callers that assemble an agent prompt WITHOUT the task-spec template —
/// today that is the tier-2 mined-task eval, whose prompt is the mined
/// statement verbatim. Sharing one template with
/// [`render_task_prompt_from_spec`] is the point: an experiment that changes
/// the guidance must change it for both surfaces at once, or the eval stops
/// measuring what dispatch actually ships.
///
/// The render is a pure function of its (empty) inputs and is
/// byte-deterministic — re-rendering produces identical bytes.
///
/// # Panics
/// Never in practice — see [`render_system_prompt`].
#[must_use]
pub fn render_test_first_approach() -> String {
    TestFirstApproachTemplate
        .render()
        .expect("test_first_approach.md is a static variable-free template that renders infallibly")
}

/// Render the `## Verification` block on its own.
///
/// For callers that assemble an agent prompt WITHOUT the task-spec template —
/// today that is the tier-2 mined-task eval in agent-gate On mode, whose
/// prompt is the mined statement verbatim plus this section. Sharing one
/// template with [`render_task_prompt_from_spec`] is the point: a wording
/// change must change it for both surfaces at once, or the eval stops
/// measuring the verification framing dispatch actually ships.
///
/// The render is a pure function of `gate_command` and is byte-deterministic —
/// re-rendering with the same input produces identical bytes.
///
/// # Panics
/// Never in practice — see [`render_system_prompt`].
#[must_use]
pub fn render_verification_section(gate_command: &str) -> String {
    VerificationSectionTemplate { gate_command }
        .render()
        .expect(
            "verification_section.md is a static template that renders infallibly for owned inputs",
        )
}

/// Render answer mode's task-framing content: the `question` verbatim, the
/// result schema verbatim inside a fenced `json` block, and the read-only rules.
///
/// `schema_text` is the RAW text of the schema as the operator supplied it —
/// the caller must NOT re-serialize it through `serde_json` first. Key order,
/// comments-as-whitespace and formatting are all part of what the model is
/// asked to conform to, and a round-trip through `Value` silently reorders
/// object keys.
///
/// The output is the content of the inner [`crate::engine::RunConfig::task`]
/// slot — the engine wraps it under `# Task` via [`render_task_prompt`], so
/// it must NOT itself begin with a top-level `# ` heading (all headings are
/// `##` or lower). The render is a pure function of its inputs and is
/// byte-deterministic.
///
/// # Panics
/// Never in practice — see [`render_system_prompt`].
#[must_use]
pub fn render_answer_prompt(question: &str, schema_text: &str) -> String {
    AnswerPromptTemplate {
        question,
        schema_text,
    }
    .render()
    .expect("answer_prompt.md is a static template that renders infallibly for owned inputs")
}

/// Render answer mode's system prompt: the read-only analyst frame plus the
/// tool listing.
///
/// Deliberately NOT a variant of [`render_system_prompt`] — it renders its
/// own template file, takes no `check_command`, and names only `answer` /
/// `blocked` / `failed` as terminals. `system_prompt.md` stays byte-identical
/// so the tier-1/tier-2 eval-parity rule gains no new surface.
///
/// The render is a pure function of `tools` and is byte-deterministic.
///
/// # Panics
/// Never in practice — see [`render_system_prompt`].
#[must_use]
pub fn render_answer_system_prompt(tools: &[ToolLine]) -> String {
    AnswerSystemPromptTemplate { tools }.render().expect(
        "answer_system_prompt.md is a static template that renders infallibly for owned inputs",
    )
}

#[cfg(test)]
mod tests {
    use super::{
        AnswerPromptTemplate, AnswerSystemPromptTemplate, NudgePromptTemplate, RalphPromptTemplate,
        ToolLine, render_answer_prompt, render_answer_system_prompt, render_nudge_prompt,
        render_ralph_prompt, render_system_prompt, render_task_prompt,
        render_task_prompt_from_spec, render_task_prompt_from_spec_with,
        render_test_first_approach, render_verification_section, tool_lines,
    };
    use crate::engine::{FINISH_TOOL_NAME, FinishTool};
    use crate::task_spec::{FileToModify, TaskSpec};
    use crate::tool::{EchoTool, Tool, ToolCtx, ToolRegistry, ToolResult};
    use askama::Template;
    use async_trait::async_trait;
    use serde_json::{Value, json};
    use std::sync::Arc;

    /// The golden bytes of `answer_system_prompt.md` rendered for
    /// [`two_tools`]. Pinned in full by
    /// `render_answer_system_prompt_production_byte_identity`.
    const ANSWER_SYSTEM_PROMPT_GOLDEN: &str = concat!(
        "# Role
",
        "
",
        "You are an autonomous read-only analyst operating inside a confined
",
        "workspace. Every path you emit or resolve is workspace-relative — the
",
        "workspace root is your world, and there is no filesystem outside it that you
",
        "should touch. Your deliverable is DATA, not a change: you investigate the
",
        "workspace and report a structured result.
",
        "
",
        "# Tools available
",
        "
",
        "- foo — does the foo thing
",
        "- bar — does the bar thing
",
        "
",
        "# Read-only contract
",
        "
",
        "This run must not modify the workspace. The harness observed the working tree
",
        "before your first turn and observes it again when you finish: an accepted
",
        "answer REQUIRES the tree to be unchanged since the run started. A modified
",
        "tree is rejected and fed back to you — revert your edits (restore tracked
",
        "files, delete files you created) and finish again.
",
        "
",
        "That constraint is on the WORKSPACE, not on your thinking. Read, list, search
",
        "and run read-only shell commands freely. Do not write, move, or delete files,
",
        "and do not run commands that mutate the tree or its git state.
",
        "
",
        "# Workflow
",
        "
",
        "Orient before you conclude. List and read enough of the workspace to ground
",
        "the answer in what is actually there, and prefer citing a file and line you
",
        "read over recalling something you did not. Then assemble the result payload
",
        "the task's result schema describes, and finish.
",
        "
",
        "# Steering semantics
",
        "
",
        "Tool results that come back with `is_error: true` are recoverable guidance,
",
        "not fatal errors. Read the message carefully, adjust your approach, and try
",
        "again. Do not give up because a single tool call returned an error.
",
        "
",
        "Long tool outputs are truncated in your view. When a result advertises a
",
        "full-output path (for example, \"full output at <path>\"), you can read the
",
        "untruncated contents via `read_file` on that path when the inline slice is
",
        "not enough.
",
        "
",
        "# Disposition guidance
",
        "
",
        "This run accepts exactly three terminal dispositions:
",
        "
",
        "- `answer` — you investigated the question and have the deliverable. Supply a
",
        "  `result` that conforms to the run's configured result schema; the harness
",
        "  validates it and feeds any validation errors back to you rather than
",
        "  terminating the run. The working tree must be unchanged since the run
",
        "  started.
",
        "- `blocked` — the question or the environment is the problem: retrying the
",
        "  same run unchanged is guaranteed not to produce an answer until a human
",
        "  makes a decision. State exactly what decision is needed.
",
        "- `failed` — the attempt is the problem: something in how *this* run went
",
        "  wrong, and a fresh attempt might succeed. Summarize what went wrong.",
    );

    fn two_tools() -> Vec<ToolLine> {
        vec![
            ToolLine {
                name: "foo".to_string(),
                description: "does the foo thing".to_string(),
            },
            ToolLine {
                name: "bar".to_string(),
                description: "does the bar thing".to_string(),
            },
        ]
    }

    #[test]
    fn system_prompt_lists_every_supplied_tool_name_and_description() {
        let tools = two_tools();
        let rendered = render_system_prompt(&tools, Some("cargo nextest run"));
        for tool in &tools {
            assert!(
                rendered.contains(&tool.name),
                "rendered prompt must contain tool name `{}`",
                tool.name
            );
            assert!(
                rendered.contains(&tool.description),
                "rendered prompt must contain description `{}`",
                tool.description
            );
        }
    }

    /// The verification contract is load-bearing: a wording pass must not be
    /// able to silently delete it. Pin the three key phrases the item
    /// specifies — `rejected`, `run_checks`, and `verified` — so a rename
    /// that loses any of them is a test failure the reviewer sees.
    #[test]
    fn system_prompt_pins_the_verification_contract_phrases() {
        let rendered = render_system_prompt(&[], Some("cargo nextest run"));
        assert!(
            rendered.contains("REJECTED") || rendered.contains("rejected"),
            "verification contract must state that a failed finish is rejected; got:\n{rendered}"
        );
        assert!(
            rendered.contains("run_checks"),
            "verification contract must reference the run_checks tool by name; got:\n{rendered}"
        );
        assert!(
            rendered.contains("verified"),
            "verification contract must speak of the fix being verified; got:\n{rendered}"
        );
    }

    #[test]
    fn system_prompt_none_check_renders_no_checks_wording_not_rejection() {
        let rendered = render_system_prompt(&[], None);
        assert!(
            rendered.to_lowercase().contains("no checks"),
            "no-check variant must announce that no checks are configured; got:\n{rendered}"
        );
        // Re-scoped to the claim this test was actually protecting: with no
        // checks configured there is no `run_checks` invocation to gate
        // `done` on, and no CHECKS-based rejection to threaten. The
        // check-INDEPENDENT rejection (an unchanged working tree) DOES apply
        // on this branch and is deliberately admitted here.
        assert!(
            !rendered.contains("run_checks"),
            "no-check variant must not reference the run_checks tool; got:\n{rendered}"
        );
        assert!(
            !rendered.contains("the checks pass"),
            "no-check variant must not promise a checks-based rejection; got:\n{rendered}"
        );
        assert!(
            rendered.contains("working tree is unchanged"),
            "no-check variant must still state the unchanged-tree rejection; got:\n{rendered}"
        );
    }

    /// The `already_satisfied` off-ramp must be advertised on BOTH branches of
    /// the check-command match — the guidance section sits outside it, and the
    /// leg-3 precondition applies with and without configured checks.
    #[test]
    fn system_prompt_advertises_already_satisfied_on_both_check_branches() {
        for check in [Some("cargo nextest run"), None] {
            let rendered = render_system_prompt(&[], check);
            assert!(
                rendered.contains("already_satisfied"),
                "already_satisfied must render for checks={check:?}; got:\n{rendered}"
            );
        }
    }

    /// The commit contract belongs to the harness, not the agent. The
    /// transcript study (docs/research/09, failure 2) recorded an agent that
    /// read its task, reasonably concluded a commit was expected, and ran
    /// `git add` + `git commit --no-verify` itself — bypassing repo hooks
    /// and leaving the dispatch worker with nothing left to commit, which
    /// failed the run after the work was already done. Pin the ownership
    /// paragraph mechanically: all five literals must co-occur in ONE
    /// blank-line-delimited chunk on BOTH check-command branches, so any
    /// wording pass that deletes or fragments the paragraph fails here.
    /// (`harness` and bare `commit` are deliberately NOT pinned — both
    /// already appear elsewhere in the template, which would make such a
    /// pin vacuous.) The length cap is a tripwire against a multi-paragraph
    /// paste: the system prompt is an anchor that is never compacted, so
    /// every character here is paid on every turn.
    #[test]
    fn system_prompt_pins_the_harness_owns_the_commit_contract() {
        for check in [Some("cargo nextest run"), None] {
            let rendered = render_system_prompt(&[], check);
            let flattened_chunks: Vec<String> = rendered
                .split("\n\n")
                .map(|chunk| chunk.split_whitespace().collect::<Vec<_>>().join(" "))
                .collect();
            let ownership_chunk = flattened_chunks
                .iter()
                .find(|chunk| chunk.contains("uncommitted"))
                .unwrap_or_else(|| {
                    panic!(
                        "system prompt must contain the git-ownership paragraph \
                         for checks={check:?}; got:\n{rendered}"
                    )
                });
            for literal in ["git add", "git commit", "git push", "owns", "uncommitted"] {
                assert!(
                    ownership_chunk.contains(literal),
                    "git-ownership paragraph must contain `{literal}` \
                     (checks={check:?}); got:\n{ownership_chunk}"
                );
            }
        }
        let rendered_none = render_system_prompt(&[], None);
        assert!(
            rendered_none.len() < 4000,
            "no-tools system prompt must stay under 4000 chars (anchor, never \
             compacted); got {}",
            rendered_none.len()
        );
    }

    #[test]
    fn system_prompt_renders_check_command_string_verbatim() {
        let rendered = render_system_prompt(&[], Some("cargo test && cargo clippy"));
        assert!(rendered.contains("cargo test && cargo clippy"));
    }

    /// HTML escaping must be off: `"`, `<`, `>`, `&` in tool descriptions
    /// pass through untouched. The `.md` extension plus the explicit
    /// `escape = "none"` on the template struct pin this.
    #[test]
    fn special_characters_pass_through_unescaped() {
        let tools = vec![ToolLine {
            name: "quoted".to_string(),
            description: r#"handles "quotes" & <angles> and &amp; itself"#.to_string(),
        }];
        let rendered = render_system_prompt(&tools, Some("echo 5 > 3 && true"));

        // The raw characters appear.
        assert!(rendered.contains(r#""quotes""#), "quotes preserved");
        assert!(rendered.contains("<angles>"), "angle brackets preserved");
        assert!(rendered.contains(" & "), "ampersand preserved");
        assert!(rendered.contains("5 > 3"), "> in check command preserved");

        // The HTML-escaped forms are NOT present — proving no escaping ran.
        // (`&amp;` is present as literal input; check its escaped form
        // `&amp;amp;` is absent instead.)
        assert!(
            !rendered.contains("&quot;"),
            "should not HTML-escape quotes"
        );
        assert!(!rendered.contains("&lt;"), "should not HTML-escape <");
        assert!(!rendered.contains("&gt;"), "should not HTML-escape >");
        assert!(
            !rendered.contains("&amp;amp;"),
            "the literal `&amp;` in the description should not be double-escaped"
        );
    }

    /// The prompt cache is byte-comparison sensitive. Rendering the same
    /// inputs twice must produce identical bytes — no timestamps, no hash
    /// randomness, no `HashMap` iteration order sneaking in.
    #[test]
    fn rendering_is_byte_deterministic() {
        let tools = two_tools();
        let first = render_system_prompt(&tools, Some("cargo test"));
        let second = render_system_prompt(&tools, Some("cargo test"));
        assert_eq!(
            first.as_bytes(),
            second.as_bytes(),
            "same inputs must produce byte-identical output"
        );

        // And the no-check variant is deterministic too.
        let a = render_system_prompt(&tools, None);
        let b = render_system_prompt(&tools, None);
        assert_eq!(a.as_bytes(), b.as_bytes());
    }

    #[test]
    fn task_prompt_contains_the_task_verbatim() {
        let task = "Refactor the widget to support 5 > 3 & \"quoted\" input";
        let rendered = render_task_prompt(task);
        assert!(
            rendered.contains(task),
            "task prompt must include the task verbatim (no escaping); got:\n{rendered}"
        );
    }

    #[test]
    fn task_prompt_rendering_is_byte_deterministic() {
        let first = render_task_prompt("do the thing");
        let second = render_task_prompt("do the thing");
        assert_eq!(first.as_bytes(), second.as_bytes());
    }

    #[test]
    fn tool_lines_extracts_from_the_registered_schemas() {
        let mut registry = ToolRegistry::new();
        registry.register("echo", Arc::new(EchoTool));
        registry.register(FINISH_TOOL_NAME, Arc::new(FinishTool::default()));

        let lines = tool_lines(&registry);
        // BTreeMap iteration order → alphabetical by registered name.
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].name, "echo");
        assert!(
            !lines[0].description.is_empty(),
            "EchoTool advertises a description"
        );
        assert_eq!(lines[1].name, FINISH_TOOL_NAME);
        assert!(
            lines[1].description.contains("End the run"),
            "FinishTool description should flow through unchanged; got `{}`",
            lines[1].description
        );
    }

    #[test]
    fn tool_lines_tolerates_schema_without_description() {
        /// A tool whose schema deliberately omits `description`. Registering
        /// one and asking [`tool_lines`] to consume it must yield an empty
        /// description, never a panic.
        #[derive(Debug, Default)]
        struct DescriptionlessTool;

        #[async_trait]
        impl Tool for DescriptionlessTool {
            // The trait fixes the return type as `&str`; a `&'static str` here
            // would diverge from the trait signature, so the lint doesn't apply.
            #[allow(clippy::unnecessary_literal_bound)]
            fn name(&self) -> &str {
                "bare"
            }
            fn schema(&self) -> Value {
                json!({ "name": "bare", "input_schema": { "type": "object" } })
            }
            async fn run(&self, _input: Value, _ctx: &ToolCtx) -> ToolResult {
                ToolResult::ok("bare")
            }
        }

        let mut registry = ToolRegistry::new();
        registry.register("bare", Arc::new(DescriptionlessTool));

        let lines = tool_lines(&registry);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].name, "bare");
        assert_eq!(
            lines[0].description, "",
            "missing description must degrade to empty string, not panic"
        );
    }

    #[test]
    fn tool_lines_skips_schema_without_name() {
        /// A schema without a `name` cannot be advertised meaningfully;
        /// [`tool_lines`] filters it out rather than emitting a nameless
        /// bullet in the prompt.
        #[derive(Debug, Default)]
        struct NamelessSchemaTool;

        #[async_trait]
        impl Tool for NamelessSchemaTool {
            // The trait fixes the return type as `&str`; a `&'static str` here
            // would diverge from the trait signature, so the lint doesn't apply.
            #[allow(clippy::unnecessary_literal_bound)]
            fn name(&self) -> &str {
                "nameless"
            }
            fn schema(&self) -> Value {
                // Note: no top-level `name` field — the registered name
                // exists, but the *schema* omits it.
                json!({ "description": "orphan" })
            }
            async fn run(&self, _input: Value, _ctx: &ToolCtx) -> ToolResult {
                ToolResult::ok("")
            }
        }

        let mut registry = ToolRegistry::new();
        registry.register("nameless", Arc::new(NamelessSchemaTool));

        let lines = tool_lines(&registry);
        assert!(
            lines.is_empty(),
            "a schema without a `name` field is filtered out; got {lines:?}"
        );
    }

    #[test]
    fn tool_lines_from_empty_registry_is_empty() {
        let registry = ToolRegistry::new();
        assert!(tool_lines(&registry).is_empty());
    }

    /// End-to-end: `tool_lines` feeds directly into `render_system_prompt`,
    /// which is what the engine loop will do at run start.
    #[test]
    fn tool_lines_feeds_render_system_prompt_end_to_end() {
        let mut registry = ToolRegistry::new();
        registry.register("echo", Arc::new(EchoTool));
        registry.register(FINISH_TOOL_NAME, Arc::new(FinishTool::default()));

        let lines = tool_lines(&registry);
        let rendered = render_system_prompt(&lines, Some("cargo nextest run"));

        assert!(rendered.contains("echo"));
        assert!(rendered.contains(FINISH_TOOL_NAME));
        assert!(rendered.contains("End the run"));
    }

    // --- render_task_prompt_from_spec tests ----------------------------------

    fn sentinel_spec() -> TaskSpec {
        TaskSpec {
            title: "SENTINEL_TITLE".to_string(),
            description: "SENTINEL_DESC".to_string(),
            acceptance_criteria: vec!["SENTINEL_AC_1".to_string(), "SENTINEL_AC_2".to_string()],
            files_to_modify: vec![
                FileToModify {
                    path: "SENTINEL_PATH_1".to_string(),
                    change: "SENTINEL_CHANGE_1".to_string(),
                },
                FileToModify {
                    path: "SENTINEL_PATH_2".to_string(),
                    change: "SENTINEL_CHANGE_2".to_string(),
                },
            ],
            gate_command: "SENTINEL_GATE".to_string(),
        }
    }

    /// Pinned "no field silently dropped" test (the 0.3.0 lesson): all nine
    /// sentinel substrings must appear in the rendered output. Askama's
    /// compile-time check catches "template references missing struct field"
    /// but NOT "struct field never referenced by template" — this runtime
    /// test closes that gap. It also catches list fields rendered by indexing
    /// `[0]` instead of a loop (only the second entry would be absent).
    #[test]
    fn render_task_prompt_from_spec_nine_sentinels() {
        let spec = sentinel_spec();
        let rendered = render_task_prompt_from_spec(&spec);

        assert!(rendered.contains("SENTINEL_TITLE"), "title must appear");
        assert!(
            rendered.contains("SENTINEL_DESC"),
            "description must appear"
        );
        assert!(rendered.contains("SENTINEL_AC_1"), "AC 1 must appear");
        assert!(rendered.contains("SENTINEL_AC_2"), "AC 2 must appear");
        assert!(rendered.contains("SENTINEL_PATH_1"), "path 1 must appear");
        assert!(
            rendered.contains("SENTINEL_CHANGE_1"),
            "change 1 must appear"
        );
        assert!(rendered.contains("SENTINEL_PATH_2"), "path 2 must appear");
        assert!(
            rendered.contains("SENTINEL_CHANGE_2"),
            "change 2 must appear"
        );
        assert!(
            rendered.contains("SENTINEL_GATE"),
            "gate_command must appear"
        );
    }

    /// Empty collections must not panic and must still render the scalar fields.
    #[test]
    fn render_task_prompt_from_spec_empty_collections_no_panic() {
        let spec = TaskSpec {
            title: "TITLE_SENTINEL".to_string(),
            description: "DESC_SENTINEL".to_string(),
            acceptance_criteria: vec![],
            files_to_modify: vec![],
            gate_command: "GATE_SENTINEL".to_string(),
        };
        let rendered = render_task_prompt_from_spec(&spec);
        assert!(
            rendered.contains("TITLE_SENTINEL"),
            "title still renders with empty collections"
        );
        assert!(
            rendered.contains("DESC_SENTINEL"),
            "description still renders with empty collections"
        );
        assert!(
            rendered.contains("GATE_SENTINEL"),
            "gate_command still renders with empty collections"
        );
    }

    /// The prompt cache is byte-comparison sensitive. Rendering the same
    /// `TaskSpec` twice must produce identical bytes.
    #[test]
    fn render_task_prompt_from_spec_byte_deterministic() {
        let spec = sentinel_spec();
        let first = render_task_prompt_from_spec(&spec);
        let second = render_task_prompt_from_spec(&spec);
        assert_eq!(
            first.as_bytes(),
            second.as_bytes(),
            "same TaskSpec must produce byte-identical output"
        );
    }

    /// HTML escaping must be off: `"`, `<`, `>`, `&` in any field pass
    /// through untouched. Mirrors `special_characters_pass_through_unescaped`.
    #[test]
    fn render_task_prompt_from_spec_unescaped_passthrough() {
        let spec = TaskSpec {
            title: "T".to_string(),
            description: r#"handles "quotes" & <angles>"#.to_string(),
            acceptance_criteria: vec![r#"AC with "quotes" & <angles>"#.to_string()],
            files_to_modify: vec![],
            gate_command: "cargo test".to_string(),
        };
        let rendered = render_task_prompt_from_spec(&spec);

        assert!(
            rendered.contains(r#""quotes""#),
            "double quotes pass through"
        );
        assert!(rendered.contains("<angles>"), "angle brackets pass through");
        assert!(rendered.contains(" & "), "ampersand passes through");
        assert!(
            !rendered.contains("&quot;"),
            "should not HTML-escape quotes"
        );
        assert!(!rendered.contains("&lt;"), "should not HTML-escape <");
        assert!(!rendered.contains("&gt;"), "should not HTML-escape >");
    }

    /// The `files_to_modify` section must carry affirmative navigation guidance
    /// directing the model to use the bash tool (grep/find) to locate code.
    /// The old denial phrase "No search tool" must be gone — replacing it was
    /// the whole point of this template change (post-mortem kb-02977). A wording
    /// pass that re-introduces the denial or removes the bash guidance fails here.
    #[test]
    fn render_task_prompt_from_spec_bash_navigation_guidance() {
        let spec = TaskSpec {
            title: "T".to_string(),
            description: "D".to_string(),
            acceptance_criteria: vec![],
            files_to_modify: vec![],
            gate_command: "cargo test".to_string(),
        };
        let rendered = render_task_prompt_from_spec(&spec);
        assert!(
            rendered.contains("bash tool"),
            "files_to_modify section must contain affirmative bash-tool navigation guidance; \
             got:\n{rendered}"
        );
        assert!(
            !rendered.to_lowercase().contains("no search tool"),
            "files_to_modify section must NOT contain the old denial phrase 'No search tool'; \
             got:\n{rendered}"
        );
    }

    /// The task prompt must carry test-first framing AND must not tell the
    /// model that a bare passing gate means done. Both are load-bearing: on a
    /// healthy repo the gate command is GREEN before the agent touches
    /// anything, so an unconditional "as soon as this passes, finish" reads as
    /// an instruction to finish at iteration 1. Red-first is what makes a green
    /// gate mean something — it turns the signal from a state into a
    /// transition. A wording pass must not silently drop either half.
    #[test]
    fn render_task_prompt_from_spec_carries_test_first_framing() {
        let spec = TaskSpec {
            title: "T".to_string(),
            description: "D".to_string(),
            acceptance_criteria: vec![],
            files_to_modify: vec![],
            gate_command: "cargo test".to_string(),
        };
        let rendered = render_task_prompt_from_spec(&spec);
        assert!(
            rendered.contains("FIRST"),
            "approach section must instruct the model to write a failing test \
             first; got:\n{rendered}"
        );
        assert!(
            rendered.contains("Confirm it fails"),
            "test-first framing must require observing the RED state, not just \
             writing a test; got:\n{rendered}"
        );
        assert!(
            !rendered.contains("As soon as this command passes"),
            "the unconditional passing-gate phrasing is a trap on a repo whose \
             gate is already green; got:\n{rendered}"
        );
    }

    /// The toggle must actually strip the section, and the production entry
    /// point must never be able to reach the stripped form: an eval knob that
    /// silently disabled guidance on the shipping path would be worse than no
    /// knob at all.
    #[test]
    fn test_first_toggle_strips_the_section_but_never_on_the_production_path() {
        let spec = TaskSpec {
            title: "T".to_string(),
            description: "D".to_string(),
            acceptance_criteria: vec![],
            files_to_modify: vec![],
            gate_command: "cargo test".to_string(),
        };
        let with = render_task_prompt_from_spec_with(&spec, true);
        let without = render_task_prompt_from_spec_with(&spec, false);

        assert!(with.contains("## Approach"), "got:\n{with}");
        assert!(!without.contains("## Approach"), "got:\n{without}");
        assert!(!without.contains("Confirm it fails"), "got:\n{without}");
        // Everything else is identical — the toggle changes ONE section, so an
        // A/B run differs only in the variable under test.
        assert_eq!(with.replace(&render_test_first_approach(), ""), without);
        // The default entry point is the `true` arm, byte for byte.
        assert_eq!(render_task_prompt_from_spec(&spec), with);
        // Verification framing survives in both arms.
        for r in [&with, &without] {
            assert!(r.contains("finish(done) immediately"), "got:\n{r}");
        }
    }

    /// The two surfaces that carry test-first guidance must carry the SAME
    /// bytes. `task_spec_prompt.md` `{% include %}`s the shared template for
    /// the groomed-item path; tier-2's mined-eval appends
    /// [`render_test_first_approach`] to the raw statement. If someone edits
    /// one and paraphrases the other, the eval stops measuring what dispatch
    /// ships — and that failure is invisible at runtime.
    #[test]
    fn task_spec_prompt_embeds_the_shared_test_first_template_verbatim() {
        let shared = render_test_first_approach();
        let spec = TaskSpec {
            title: "T".to_string(),
            description: "D".to_string(),
            acceptance_criteria: vec![],
            files_to_modify: vec![],
            gate_command: "cargo test".to_string(),
        };
        let rendered = render_task_prompt_from_spec(&spec);
        assert!(
            rendered.contains(shared.trim_end()),
            "task-spec prompt must embed the shared approach template verbatim;\n\
             shared:\n{shared}\nrendered:\n{rendered}"
        );
        // Byte-deterministic, like every other template render here.
        assert_eq!(shared, render_test_first_approach());
    }

    /// The verification section must carry finish-discipline framing containing
    /// the literal substring `finish(done) immediately` — added after the first
    /// dogfood run, where the model completed and verified the task by
    /// iteration 6 and then exhausted the remaining budget re-verifying
    /// individual acceptance criteria instead of finishing. A wording pass must
    /// not silently delete this instruction.
    #[test]
    fn render_task_prompt_from_spec_finish_discipline_framing() {
        let spec = TaskSpec {
            title: "T".to_string(),
            description: "D".to_string(),
            acceptance_criteria: vec![],
            files_to_modify: vec![],
            gate_command: "cargo test".to_string(),
        };
        let rendered = render_task_prompt_from_spec(&spec);
        assert!(
            rendered.contains("finish(done) immediately"),
            "verification section must instruct the model to call finish(done) \
             immediately after a passing check; got:\n{rendered}"
        );
    }

    /// SLOT-CONTENT contract: the rendered output must not contain any line
    /// that starts with `# ` (a top-level heading). The engine wraps the
    /// slot content under `# Task`; emitting another top-level heading would
    /// break the document structure. Title must use `##` or lower.
    #[test]
    fn render_task_prompt_from_spec_no_top_level_heading() {
        let spec = sentinel_spec();
        let rendered = render_task_prompt_from_spec(&spec);
        for line in rendered.lines() {
            assert!(
                line != "# Task",
                "rendered slot content must not contain a line exactly equal to '# Task'; \
                 got:\n{rendered}"
            );
            assert!(
                !line.starts_with("# "),
                "rendered slot content must not contain a top-level '# ' heading; \
                 offending line: {line:?}; full output:\n{rendered}"
            );
        }
    }

    // --- verification_section factoring tests -------------------------------

    /// Production byte-identity pin: the golden literal was captured from the
    /// UNMODIFIED templates (before `verification_section.md` was factored
    /// out) and must still match after the factoring.
    /// The anti-loop line exists because talos-haiku's first patrol (0.3.5) finished
    /// the work by iteration 6 and then re-verified acceptance criteria one at a time
    /// until it hit `MaxIterations`. It is also pinned indirectly by the production
    /// golden below; this test states the intent so a future edit fails loudly.
    #[test]
    fn verification_section_pins_no_reverify_line() {
        assert!(
            render_verification_section("cargo test").contains(
                "Do not re-verify individual acceptance criteria with extra reads or commands"
            ),
            "the anti-loop line must stay in verification_section.md"
        );
    }

    #[test]
    fn render_task_prompt_from_spec_production_byte_identity() {
        const GOLDEN: &str = "## T\n\nD\n\n## Acceptance Criteria\n\n- a1\n\n## Files to Modify\n\nThe paths listed below are starting points for navigation — they are a map, not\na complete inventory. Use the bash tool (e.g., `grep`, `find`) to locate code\nwithin and around these files. Edit exactly what the task requires and no more;\nprefer edit_file for mutations (its unique-match contract is safer than sed -i).\n\n- `p`: c\n\n## Approach\n\nWrite a test that captures the acceptance criteria FIRST, before you change any\nother code, and run it. Confirm it fails, and that it fails for the reason you\nexpect — a test that passes before you have implemented anything is testing\nnothing. Then implement until it passes.\n\nIf the task genuinely has no test-shaped outcome (a pure refactor, a dependency\nbump, a docs or config change), state that in one line and go straight to the\nchange.\n## Verification\n\nRun the following command to verify the task is complete:\n\n    cargo test\n\nOnce your new test passes and this command is green, call finish(done) immediately.\nNote that this command may already be green before you start — a green gate on\nuntouched code means you have not begun, not that you are done.\nDo not re-verify individual acceptance criteria with extra reads or commands\nafter a passing check — the passing check IS the verification, and every\nadditional step spends your iteration budget without adding evidence.";
        let spec = TaskSpec {
            title: "T".to_string(),
            description: "D".to_string(),
            acceptance_criteria: vec!["a1".to_string()],
            files_to_modify: vec![FileToModify {
                path: "p".to_string(),
                change: "c".to_string(),
            }],
            gate_command: "cargo test".to_string(),
        };
        assert_eq!(render_task_prompt_from_spec(&spec), GOLDEN);
    }

    /// The shared `verification_section.md` template is the single source for
    /// both surfaces: the task-spec-prompt tail and the standalone renderer.
    #[test]
    fn render_task_prompt_from_spec_ends_with_shared_verification_section() {
        let spec = TaskSpec {
            title: "T".to_string(),
            description: "D".to_string(),
            acceptance_criteria: vec!["a1".to_string()],
            files_to_modify: vec![FileToModify {
                path: "p".to_string(),
                change: "c".to_string(),
            }],
            gate_command: "cargo test".to_string(),
        };
        assert!(
            render_task_prompt_from_spec(&spec)
                .ends_with(&render_verification_section(&spec.gate_command))
        );
        assert!(render_verification_section("X").starts_with("## Verification\n"));
        assert!(render_verification_section("X").contains("\n    X\n"));
    }

    // --- render_nudge_prompt tests ------------------------------------------

    /// The nudge wording is load-bearing — an engine detection test pins it
    /// against `record.messages`. Pin the exact text at the prompt layer too
    /// so a template edit that drifts fails here before it reaches the
    /// engine.
    #[test]
    fn render_nudge_prompt_pins_exact_wording() {
        let rendered = render_nudge_prompt();
        let expected = "The quality gates are currently green. \
            If the acceptance criteria are met, call `finish(done)` now. \
            If nothing needed changing because the task was already complete, \
            call `finish(already_satisfied)` with a `reason`. \
            If they are not yet met, reply with a one-sentence status: \
            what remains, and why you are still working.";
        assert_eq!(
            rendered, expected,
            "nudge wording must match the pinned text exactly; got:\n{rendered}"
        );
    }

    /// The nudge is a variable-free template — rendering it twice must
    /// produce byte-identical output (prompt-cache correctness).
    #[test]
    fn render_nudge_prompt_is_byte_deterministic() {
        let a = render_nudge_prompt();
        let b = render_nudge_prompt();
        assert_eq!(
            a.as_bytes(),
            b.as_bytes(),
            "same (empty) inputs must produce byte-identical nudge output"
        );
    }

    /// `NudgePromptTemplate` is a unit struct with no fields — its `render`
    /// must succeed via the askama derive (a compile-time check) and return
    /// the same bytes as the free function.
    #[test]
    fn nudge_prompt_template_renders_via_derive() {
        let via_struct = NudgePromptTemplate
            .render()
            .expect("variable-free template renders infallibly");
        let via_fn = render_nudge_prompt();
        assert_eq!(via_struct, via_fn);
    }

    // --- render_ralph_prompt tests ------------------------------------------

    /// The ralph prompt must contain the load-bearing discipline phrase
    /// verbatim, an append instruction that names the injected notes file,
    /// both sentinels (objective + notes), and NO top-level `# ` heading.
    #[test]
    fn render_ralph_prompt_pins_phrases_and_injects_fields() {
        let objective = "SENTINEL_OBJECTIVE do the ralph thing";
        let notes = "SENTINEL_NOTES prior iteration journal";
        let notes_file = "SENTINEL_NOTES_FILE.md";
        let rendered = render_ralph_prompt(objective, notes, notes_file);

        // (a) the literal discipline phrase.
        assert!(
            rendered.contains("one thing — not one thing plus a quick refactor"),
            "ralph prompt must contain the literal discipline phrase; got:\n{rendered}"
        );

        // (b) an append instruction that includes the word `append` and the
        // injected notes_file path.
        assert!(
            rendered.contains("append"),
            "ralph prompt must instruct the agent to append; got:\n{rendered}"
        );
        assert!(
            rendered.contains(notes_file),
            "ralph prompt must name the injected notes file `{notes_file}`; got:\n{rendered}"
        );

        // (c) objective and notes sentinels appear verbatim.
        assert!(
            rendered.contains(objective),
            "ralph prompt must inject the objective verbatim; got:\n{rendered}"
        );
        assert!(
            rendered.contains(notes),
            "ralph prompt must inject the notes verbatim; got:\n{rendered}"
        );

        // (d) no top-level `# ` heading — the engine wraps this under `# Task`.
        for line in rendered.lines() {
            assert!(
                !line.starts_with("# "),
                "ralph prompt must not contain a top-level '# ' heading; \
                 offending line: {line:?}; full output:\n{rendered}"
            );
        }
    }

    /// The prompt cache is byte-comparison sensitive. Rendering the same
    /// ralph inputs twice must produce identical bytes.
    #[test]
    fn render_ralph_prompt_is_byte_deterministic() {
        let first = render_ralph_prompt("obj", "notes", "PROGRESS.md");
        let second = render_ralph_prompt("obj", "notes", "PROGRESS.md");
        assert_eq!(
            first.as_bytes(),
            second.as_bytes(),
            "same ralph inputs must produce byte-identical output"
        );
    }

    /// Empty notes (first iteration) must render without panic.
    #[test]
    fn render_ralph_prompt_empty_notes_renders() {
        let rendered = render_ralph_prompt("objective", "", "PROGRESS.md");
        assert!(rendered.contains("objective"));
        assert!(rendered.contains("PROGRESS.md"));
    }

    /// `RalphPromptTemplate`'s derive-render must agree with the free function.
    #[test]
    fn ralph_prompt_template_renders_via_derive() {
        let via_struct = RalphPromptTemplate {
            objective: "obj",
            notes: "notes",
            notes_file: "PROGRESS.md",
        }
        .render()
        .expect("ralph template renders infallibly");
        let via_fn = render_ralph_prompt("obj", "notes", "PROGRESS.md");
        assert_eq!(via_struct, via_fn);
    }

    // =====================================================================
    // Answer mode — `render_answer_prompt`
    // =====================================================================

    /// A schema text whose keys are in NON-alphabetical source order and
    /// which carries `"`, `<`, `>` and `&`. Used to prove both that the
    /// schema reaches the model byte-for-byte (AC-17) and that escaping is
    /// off (AC-14b) — a `serde_json` round-trip would reorder these keys.
    fn gnarly_schema_text() -> &'static str {
        "{\"title\":\"q & a <v1>\",\"type\":\"object\",\"required\":[\"answer\"],\
         \"properties\":{\"answer\":{\"type\":\"string\"}},\"additionalProperties\":false}"
    }

    /// AC-14(a): both arguments appear verbatim in the rendered output.
    #[test]
    fn render_answer_prompt_carries_question_and_schema_verbatim() {
        let question = "What does exit code 30 mean in this repo?";
        let rendered = render_answer_prompt(question, gnarly_schema_text());
        assert!(
            rendered.contains(question),
            "the question must appear verbatim; got:\n{rendered}"
        );
        assert!(
            rendered.contains(gnarly_schema_text()),
            "the schema text must appear verbatim; got:\n{rendered}"
        );
    }

    /// AC-14(b): `"`, `<`, `>`, `&` in EITHER argument pass through
    /// unescaped. Mirrors `render_task_prompt_from_spec_unescaped_passthrough`.
    #[test]
    fn render_answer_prompt_unescaped_passthrough() {
        let question = r#"why does "x" & <y> differ?"#;
        let rendered = render_answer_prompt(question, gnarly_schema_text());

        assert!(rendered.contains(r#""x""#), "double quotes pass through");
        assert!(rendered.contains("<y>"), "angle brackets pass through");
        assert!(rendered.contains(" & "), "ampersand passes through");
        assert!(!rendered.contains("&quot;"), "must not HTML-escape quotes");
        assert!(!rendered.contains("&lt;"), "must not HTML-escape <");
        assert!(!rendered.contains("&gt;"), "must not HTML-escape >");
        assert!(!rendered.contains("&amp;"), "must not HTML-escape &");
    }

    /// AC-14(c): the render is a pure function — same inputs, identical bytes
    /// (the prompt-cache correctness invariant).
    #[test]
    fn render_answer_prompt_is_byte_deterministic() {
        let a = render_answer_prompt("q?", gnarly_schema_text());
        let b = render_answer_prompt("q?", gnarly_schema_text());
        assert_eq!(a, b, "re-rendering identical inputs must be byte-identical");
    }

    /// AC-15: the skeleton is pinned — `## Question`, the question,
    /// `## Result schema`, a fenced `json` block wrapping the schema text,
    /// then `## Rules`, in that order.
    #[test]
    fn render_answer_prompt_skeleton_is_pinned() {
        let question = "SENTINEL-QUESTION";
        let schema = "SENTINEL-SCHEMA";
        let rendered = render_answer_prompt(question, schema);

        let idx = |needle: &str| {
            rendered
                .find(needle)
                .unwrap_or_else(|| panic!("missing {needle:?} in:\n{rendered}"))
        };
        let q_heading = idx("## Question");
        let q_text = idx(question);
        let s_heading = idx("## Result schema");
        let fence_open = idx("```json\n");
        let s_text = idx(schema);
        let rules = idx("## Rules");

        assert!(q_heading < q_text, "the question follows `## Question`");
        assert!(
            q_text < s_heading,
            "`## Result schema` follows the question"
        );
        assert!(
            s_heading < fence_open,
            "the fence follows `## Result schema`"
        );
        assert!(fence_open < s_text, "the schema sits inside the fence");
        assert!(
            rendered[s_text..].contains("\n```"),
            "the json fence must be closed after the schema; got:\n{rendered}"
        );
        assert!(s_text < rules, "`## Rules` follows the schema fence");
    }

    /// AC-15: the engine wraps this slot under `# Task`, so the rendered
    /// output must contain no top-level `# ` heading. Mirrors
    /// `render_task_prompt_from_spec_no_top_level_heading`.
    #[test]
    fn render_answer_prompt_no_top_level_heading() {
        let rendered = render_answer_prompt("q?", gnarly_schema_text());
        for line in rendered.lines() {
            assert!(
                !line.starts_with("# "),
                "rendered slot content must not contain a top-level '# ' heading; \
                 offending line: {line:?}; full output:\n{rendered}"
            );
        }
    }

    /// AC-16: the answer frame states three things, each pinned.
    #[test]
    fn render_answer_prompt_frame_states_the_three_rules() {
        let rendered = render_answer_prompt("q?", gnarly_schema_text());
        assert!(
            rendered.contains("answering a question about the workspace"),
            "(a) the agent is answering a question about the workspace; got:\n{rendered}"
        );
        assert!(
            rendered.contains("must not modify"),
            "(b) it must not modify the workspace; got:\n{rendered}"
        );
        assert!(
            rendered.contains("makes the answer unacceptable"),
            "(b) a modified tree makes the answer unacceptable; got:\n{rendered}"
        );
        assert!(
            rendered.contains("finish(answer)"),
            "(c) it must end the run with finish(answer); got:\n{rendered}"
        );
        assert!(
            rendered.contains("`result` that matches the schema"),
            "(c) supplying a result that matches the schema shown; got:\n{rendered}"
        );
    }

    /// AC-17: the schema is shown VERBATIM — non-alphabetical key order and
    /// special characters survive byte-for-byte. A `serde_json` round-trip
    /// before rendering would silently reorder the object keys and fail here.
    #[test]
    fn render_answer_prompt_shows_the_schema_byte_for_byte() {
        let rendered = render_answer_prompt("q?", gnarly_schema_text());
        assert!(
            rendered.contains(gnarly_schema_text()),
            "the schema must appear byte-for-byte as supplied; got:\n{rendered}"
        );
        // Key order is the discriminator: `title` precedes `type`, which a
        // re-serialization through `serde_json::Value` would invert.
        let t = rendered.find("\"title\"").expect("title key present");
        let ty = rendered.find("\"type\"").expect("type key present");
        assert!(t < ty, "source key order must survive; got:\n{rendered}");
    }

    /// `AnswerPromptTemplate`'s derive-render must agree with the free
    /// function — the wrapper adds nothing.
    #[test]
    fn answer_prompt_template_struct_matches_the_free_function() {
        let via_struct = AnswerPromptTemplate {
            question: "q?",
            schema_text: gnarly_schema_text(),
        }
        .render()
        .expect("renders");
        assert_eq!(via_struct, render_answer_prompt("q?", gnarly_schema_text()));
    }

    // =====================================================================
    // Answer mode — `render_answer_system_prompt`
    // =====================================================================

    /// AC-18(a): the FULL rendered bytes for a fixed `ToolLine` list, pinned
    /// against a golden. Mirrors
    /// `render_task_prompt_from_spec_production_byte_identity` — a wording
    /// pass has to update this deliberately.
    #[test]
    fn render_answer_system_prompt_production_byte_identity() {
        const GOLDEN: &str = ANSWER_SYSTEM_PROMPT_GOLDEN;
        assert_eq!(render_answer_system_prompt(&two_tools()), GOLDEN);
    }

    /// AC-18(b): every supplied tool name and description appears.
    #[test]
    fn answer_system_prompt_lists_every_supplied_tool_name_and_description() {
        let tools = two_tools();
        let rendered = render_answer_system_prompt(&tools);
        for tool in &tools {
            assert!(
                rendered.contains(&tool.name),
                "rendered prompt must contain tool name `{}`",
                tool.name
            );
            assert!(
                rendered.contains(&tool.description),
                "rendered prompt must contain description `{}`",
                tool.description
            );
        }
    }

    /// AC-18(c): the NEGATIVE test. With an EMPTY tool list no tool
    /// description can leak a disposition name in, so what remains is the
    /// frame's own wording: it names `answer`, `blocked` and `failed`, and
    /// names neither `already_satisfied` nor the backticked `` `done` ``.
    #[test]
    fn answer_system_prompt_names_only_the_three_answer_mode_terminals() {
        let rendered = render_answer_system_prompt(&[]);
        assert!(rendered.contains("answer"), "must name `answer`");
        assert!(rendered.contains("blocked"), "must name `blocked`");
        assert!(rendered.contains("failed"), "must name `failed`");
        assert!(
            !rendered.contains("already_satisfied"),
            "answer mode must not advertise already_satisfied; got:\n{rendered}"
        );
        assert!(
            !rendered.contains("`done`"),
            "answer mode must not advertise `done`; got:\n{rendered}"
        );
    }

    /// The build-mode system prompt is a DIFFERENT document — the two
    /// renderers must not collapse into one. `system_prompt.md` is the
    /// eval-parity surface; this is the guard that a refactor did not point
    /// both functions at the same template.
    #[test]
    fn answer_and_build_system_prompts_are_distinct_documents() {
        let tools = two_tools();
        assert_ne!(
            render_answer_system_prompt(&tools),
            render_system_prompt(&tools, None),
            "answer mode must render its own system prompt"
        );
    }

    /// `AnswerSystemPromptTemplate`'s derive-render must agree with the free
    /// function.
    #[test]
    fn answer_system_prompt_template_struct_matches_the_free_function() {
        let tools = two_tools();
        let via_struct = AnswerSystemPromptTemplate { tools: &tools }
            .render()
            .expect("renders");
        assert_eq!(via_struct, render_answer_system_prompt(&tools));
    }
}
