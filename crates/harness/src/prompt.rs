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
    TaskSpecPromptTemplate {
        title: &spec.title,
        description: &spec.description,
        acceptance_criteria: &spec.acceptance_criteria,
        files_to_modify: &spec.files_to_modify,
        gate_command: &spec.gate_command,
    }
    .render()
    .expect("task_spec_prompt.md is a static template that renders infallibly for owned inputs")
}

#[cfg(test)]
mod tests {
    use super::{
        ToolLine, render_system_prompt, render_task_prompt, render_task_prompt_from_spec,
        tool_lines,
    };
    use crate::engine::{FINISH_TOOL_NAME, FinishTool};
    use crate::task_spec::{FileToModify, TaskSpec};
    use crate::tool::{EchoTool, Tool, ToolCtx, ToolRegistry, ToolResult};
    use async_trait::async_trait;
    use serde_json::{Value, json};
    use std::sync::Arc;

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
        // With no checks configured, there is no rejection to threaten, and
        // no `run_checks` invocation to gate `done` on — those phrases must
        // NOT appear in the no-check variant.
        assert!(
            !rendered.contains("REJECTED") && !rendered.contains("rejected"),
            "no-check variant must not talk about rejection; got:\n{rendered}"
        );
        assert!(
            !rendered.contains("run_checks"),
            "no-check variant must not reference the run_checks tool; got:\n{rendered}"
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
        registry.register(FINISH_TOOL_NAME, Arc::new(FinishTool));

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
        registry.register(FINISH_TOOL_NAME, Arc::new(FinishTool));

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

    /// The `files_to_modify` section must carry navigation framing text
    /// containing the literal substring `no search tool` (case-insensitive)
    /// so a later wording pass cannot silently delete the framing.
    #[test]
    fn render_task_prompt_from_spec_no_search_tool_framing() {
        let spec = TaskSpec {
            title: "T".to_string(),
            description: "D".to_string(),
            acceptance_criteria: vec![],
            files_to_modify: vec![],
            gate_command: "cargo test".to_string(),
        };
        let rendered = render_task_prompt_from_spec(&spec);
        assert!(
            rendered.to_lowercase().contains("no search tool"),
            "files_to_modify section must contain framing text with 'no search tool'; \
             got:\n{rendered}"
        );
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
}
