//! The two inference rungs: rung 1 (one single-shot call per project that
//! groups entries into clusters) and rung 2 (one single-shot call per
//! cluster that emits ops from the closed vocabulary).
//!
//! The loop-input is INJECTED as synthetic tool-call/result events and the
//! read tools are deleted from the output union (the vendored spec's
//! factor-13 rule): the model cannot wander the KB, and the unit of work is
//! one inference rather than an agentic loop. There is no engine loop here
//! and no nudge/compaction machinery to configure — `engine::run` builds its
//! own `initial_messages` internally and exposes no history seam, so the
//! rungs call `ModelBackend::turn` directly, once each.
//!
//! Both rungs are SINGLE-SHOT: a parse failure is never retried (the
//! pipeline aborts the unit or skips the cluster and offloads the raw model
//! text), and the per-unit inference budget
//! ([`crate::SOMNUS_MAX_CLUSTERS_PER_UNIT`]) is checked by the pipeline before
//! every rung-2 turn.
//!
//! Layering (pinned): this module references [`crate::loop_input`] and
//! [`crate::ledger`] only.

use std::path::{Path, PathBuf};

use harness::model::{
    AssistantTurn, BackendError, ContentBlock, Message, ModelBackend, SamplingParams,
    ToolCallRequest, TurnRequest, UserBlock,
};
use serde_json::{Value, json};

use crate::loop_input::LoopInput;
use crate::ops::op_tool_schemas;

pub use crate::loop_input::Cluster;

/// The model the nightly loop runs on.
pub const SOMNUS_MODEL_ID: &str = "claude-sonnet-5";

/// The pinned output cap for [`SOMNUS_MODEL_ID`] — the published per-model
/// table value, asserted against `harness::anthropic::output_cap_for_model`.
pub const SOMNUS_MAX_TOKENS: u32 = 128_000;

/// The rung-1 instruction, pinned in full. The model returns clusters as
/// plain JSON text (the registry has NO emit-clusters tool, by the same
/// closed-vocabulary rule that forbids a model-written map body), and code
/// parses it strictly.
///
/// **The carving rules here came from an 18-agent adversarial review of the
/// first real run's 14 clusters against the actual entries**, and each one
/// names a failure that review found. They are ordered by the reviewer's own
/// merit order, and every one of them is a CONTEXT failure — a thing a
/// competent model cannot know from the payload alone:
///
/// - *One subject per label.* A label joining nouns with an ampersand, slash
///   or comma is the model reporting its own failed carve: it named the union
///   instead of re-splitting. Code refuses such a cluster mechanically (see
///   [`crate::admission::label_names_one_subject`]), so this sentence is the
///   cheap half of a rule that is enforced either way.
/// - *Merit order.* Rung 2 is one call per cluster and the write cap is one
///   new map per project, so SOMETHING has to decide which cluster becomes
///   tonight's map. Ranking is judgement no rule can compute; admission is
///   mechanical. This is where the judgement half is asked for.
/// - *A contiguous run of entry ids is authoring provenance, not a subject.*
///   The reviewer's sharpest finding: one cluster was exactly a thirty-id
///   range minus two, which is one planning wave sliced by id — and it is how
///   a GPU inference pipeline ended up filed under security. Ids correlate
///   with when things were written, never with what they are about.
/// - *Never silently dissolve an input pocket.* A mechanical pocket vanished
///   into a larger cluster with no account given.
/// - *A cited source is not a subject.* One cluster was named after a code
///   sample three different clusters cite.
/// - *Carve by consequence, not only similarity.* Privacy was the project's
///   only high-concern area by its own entries, with three documented
///   location-leak vectors, and all of it sat buried in a sixteen-entry
///   catch-all with no privacy node anywhere in the output.
pub const RUNG1_INSTRUCTION: &str = "You are given the project's entries, existing mental maps, and candidate pockets as a tool result. Group the entries into subject-area clusters. Reply with ONLY a JSON array of objects, each {\"label\": string, \"member_entry_ids\": [string], \"owning_map_id\": string | null, \"merit_reason\": string | null}. Do not wrap the array in a markdown code fence and do not write any prose around it. An entry may appear in no cluster or in several clusters; do not force an assignment. How to carve, in order of importance. (1) One subject per label: if you find yourself joining two nouns with an ampersand, a slash or a comma, you have found two clusters, so split them instead of naming the union. (2) Emit the clusters in merit order, the one most worth a new map first, and give that first cluster a merit_reason of one sentence saying why it leads; later clusters may leave merit_reason null. (3) A run of consecutive entry ids is authoring provenance, not a subject area: ids tell you when entries were written, never what they are about, so never let an id range hold a cluster together. (4) If a candidate pocket does not survive as a cluster, say so in the merit_reason of the cluster that absorbed it rather than letting it disappear without account. (5) Do not name a cluster after a source its entries merely cite, because several unrelated clusters may cite the same source. (6) Carve by consequence as well as similarity: a subject the entries treat as high risk or high concern deserves its own cluster even when its entries would otherwise scatter into larger ones.";

/// The synthetic tool-call id that pairs the injected loop-input with its
/// tool result.
pub const LOOP_INPUT_CALL_ID: &str = "somnus-loop-input-1";

/// The name of the synthetic tool call that injects the loop-input. No such
/// tool is registered anywhere — the call/result pair is synthetic history,
/// not a capability.
pub const LOOP_INPUT_TOOL_NAME: &str = "load_loop_input";

/// The sampling parameters every somnus turn runs under.
#[must_use]
pub fn somnus_sampling_params() -> SamplingParams {
    SamplingParams {
        max_tokens: SOMNUS_MAX_TOKENS,
        temperature: None,
        stop_sequences: vec![],
    }
}

/// The two synthetic messages that inject the (ledger-filtered) loop-input:
/// an assistant tool call under the pinned id/name, and the user tool result
/// carrying the filtered payload. Both rungs start from this pair.
///
/// # Panics
/// Panics if the payload cannot serialize — an internal invariant (the type
/// carries no serde renames and no non-string map keys), never model input.
#[must_use]
pub fn loop_input_messages(project_ref: &str, filtered: &LoopInput) -> Vec<Message> {
    let payload = serde_json::to_string(filtered)
        .expect("the loop-input payload is JSON by construction (no serde renames)");
    vec![
        Message::Assistant {
            content: vec![ContentBlock::ToolCall(ToolCallRequest {
                id: LOOP_INPUT_CALL_ID.to_string(),
                name: LOOP_INPUT_TOOL_NAME.to_string(),
                input: json!({ "project_ref": project_ref }),
            })],
        },
        Message::User {
            content: vec![UserBlock::ToolResult {
                call_id: LOOP_INPUT_CALL_ID.to_string(),
                content: payload,
                is_error: false,
            }],
        },
    ]
}

/// The rung-1 history: the loop-input pair plus the rung-1 instruction.
/// EXACTLY three messages, `tools = []`, `system = None`.
#[must_use]
pub fn rung1_messages(project_ref: &str, filtered: &LoopInput) -> Vec<Message> {
    let mut messages = loop_input_messages(project_ref, filtered);
    messages.push(Message::User {
        content: vec![UserBlock::Text(RUNG1_INSTRUCTION.to_string())],
    });
    messages
}

/// The `cluster_id` the model is TOLD to use, derived from the cluster's
/// index so code owns it.
///
/// It used to be a string the model invented, which left the fresh map's id
/// unpredictable to the very prompt that had to name it — the model had to
/// guess our id convention in order to point at its own map, and when it
/// guessed differently the create composed with no pointers and was refused.
#[must_use]
pub fn pinned_cluster_id(cluster_index: usize) -> String {
    format!("c{cluster_index}")
}

/// The rung-2 instruction for one cluster.
///
/// **A map is created together with its pointers, in one write.** There is no
/// incremental pointer API for an agent anywhere in the KB: a human agent
/// creates a map with a single store whose body already contains its pointer
/// lines, and the server enforces at least one at creation time. So an op set
/// for an unmapped cluster is one `create_map` plus one `add_pointer` per
/// member, where those `add_pointer` ops are LOCAL contributions to the
/// pending body — never HTTP calls. `add_pointer` over the wire keeps its own
/// separate meaning: adding a newly-written entry to a map that already
/// exists.
///
/// Three runs wrote nothing because this was left implicit. The model emitted
/// `create_map` and no pointers, compose refused for an empty pointer list,
/// and the remaining clusters degraded. Stating the exact
/// `map_id` to use is the fix, and it is only possible because
/// [`pinned_cluster_id`] made that id predictable.
///
/// **`propose_gap` is forbidden for a cluster with no map**, because a gap is
/// a line inside an existing map's body. Emitting one there is not a
/// disposition, it is an error — and it used to be recorded as a ledger
/// decline, which retired nine legitimate subject areas in a single run
/// before a human ever saw a map.
///
/// **The length sentence names a TARGET, not the limit, and that distinction
/// cost 8 of 11 bodies on run 5.** The map-lint's budget is
/// `900 + 175 × pointers`, which is roughly two and a half times what a
/// human actually writes — all 30 hand-authored maps in the corpus come in
/// UNDER it by 500 to 2,500 characters, at about `400 + 60 × pointers`. The
/// model was told nothing about length, optimised against the only number in
/// sight, and wrote 1,841 to 3,398 characters where a human writes 600 to
/// 900. Eight bodies were rejected `over_budget`.
///
/// So the numbers here are the measured human norm rather than the ceiling:
/// orientation prose under 600 characters, each gloss under 120. For a
/// six-pointer map that lands near 1,320 against a 1,950 budget. **Aiming at
/// a limit is how you get rejections; aim at half of it.** The budget is
/// advisory in name and fatal in effect, because the gate is mandatory.
///
/// The purity sentence is not style advice — it is the cheapest place to
/// prevent a gate rejection. The map-lint is grammar-agnostic (six purity
/// regexes plus a length budget; it never parses our section headings or
/// bullet shapes), so the ONLY thing this model can write that fails the gate
/// is the CONTENT of a gloss or the orientation prose.
///
/// `config_numeral` is the second trap of the same family, and it rejected
/// two bodies on run 7. A map exists so an agent knows WHERE to look; a
/// number that reads like configuration is precisely the retrievable value
/// the map is there to avoid duplicating, and duplicating it is how a map
/// goes stale. The model cannot infer that from "write orientation prose",
/// so it is stated: name the threshold, point at the entry, never carry the
/// value. Bare small integers used as counts still pass.
///
/// The non-obvious member of that list is `e.g.` — the dotted-identifier rule
/// is `\b[A-Za-z_]\w*\.[A-Za-z_]\w*`, which is meant to catch dotted code
/// identifiers and matches any `word.word`, so `e.g.` and `i.e.` trip it. A
/// model writing prose reaches for those constantly. Probed against the live
/// lint on 2026-09-20: bare integers pass, `Lives in packages/kb-core` passes
/// (the path rule requires a leading `/` or `~/`), and backticks, double
/// quotes, `SCREAMING_SNAKE` tokens and absolute paths all fail.
#[must_use]
pub fn render_rung2_instruction(cluster: &Cluster, cluster_index: usize) -> String {
    let cluster_id = pinned_cluster_id(cluster_index);
    let members = cluster.member_entry_ids.join(", ");
    let disposition = match &cluster.owning_map_id {
        Some(owning) => format!(
            "This cluster is already covered by map {owning}. Do not create a second map for it. Add pointers to {owning} for entries it does not yet point at, strike any gap in it that these entries now close, or propose a gap on it. If nothing applies, emit no_change with cluster_id {cluster_id}."
        ),
        None => format!(
            "This cluster has no map. If it is worth mapping, emit exactly one create_map with cluster_id set to the literal {cluster_id}, and one add_pointer for every entry the map should point at, each with map_id set to the literal {}. A map is created together with its pointers in a single write, so there is no way to add a pointer to this map afterwards, and a create_map carrying no add_pointer ops is refused outright. If it is not worth mapping, emit no_change with cluster_id {cluster_id} and nothing else. Do not emit propose_gap for this cluster: a gap is a line inside an existing map, and this cluster has none.",
            crate::materialize::new_map_id(&cluster_id)
        ),
    };
    format!(
        "You are given one cluster and the project's loop-input as tool results. The cluster is labelled {} and covers {members}. Emit ops from the closed vocabulary (add_pointer, create_map, strike_gap, propose_gap, no_change) as tool calls. Do not write a map body; code composes bodies. {disposition} Be brief: keep the orientation prose under 600 characters and each gloss under 120 characters. A map is a directory card, not a summary, and a body that runs long is rejected outright rather than trimmed. In every gloss and every line of orientation prose you write, use plain words only: no abbreviations containing a period such as e.g. or i.e., no backticks, no double quotes, no ALL_CAPS_UNDERSCORE tokens, no absolute paths beginning with / or ~/, and no decimal numbers. Never write a configuration value: no version numbers, thresholds, ports, sizes, timeouts or percentages. Say that a threshold exists and point at the entry that holds it; the number itself belongs in that entry, not in a map. Small whole numbers used as counts are fine, as in three stages. Spell out 'for example' and 'that is'.",
        cluster.label
    )
}

/// The rung-2 history for one cluster: the SAME two synthetic loop-input
/// messages plus the cluster's instruction. `tools` is exactly the four op
/// schemas, `system = None`.
#[must_use]
pub fn rung2_messages(
    project_ref: &str,
    filtered: &LoopInput,
    cluster: &Cluster,
    cluster_index: usize,
) -> Vec<Message> {
    let mut messages = loop_input_messages(project_ref, filtered);
    messages.push(Message::User {
        content: vec![UserBlock::Text(render_rung2_instruction(
            cluster,
            cluster_index,
        ))],
    });
    messages
}

/// One single-shot rung-1 inference: nothing callable (`tools = &[]`), no
/// system prompt, the pinned sampling parameters.
///
/// # Errors
/// Propagates the backend's failure verbatim; the pipeline decides the fate.
pub async fn rung1_turn(
    backend: &dyn ModelBackend,
    project_ref: &str,
    filtered: &LoopInput,
) -> Result<AssistantTurn, BackendError> {
    let messages = rung1_messages(project_ref, filtered);
    let tools: Vec<Value> = Vec::new();
    let params = somnus_sampling_params();
    let request = TurnRequest {
        system: None,
        messages: &messages,
        tools: &tools,
        params: &params,
    };
    backend.turn(&request).await
}

/// One single-shot rung-2 inference for one cluster: the four op schemas and
/// nothing else callable, no system prompt, the same sampling parameters.
///
/// # Errors
/// Propagates the backend's failure verbatim; the pipeline decides the fate.
pub async fn rung2_turn(
    backend: &dyn ModelBackend,
    project_ref: &str,
    filtered: &LoopInput,
    cluster: &Cluster,
    cluster_index: usize,
) -> Result<AssistantTurn, BackendError> {
    let messages = rung2_messages(project_ref, filtered, cluster, cluster_index);
    let tools = op_tool_schemas();
    let params = somnus_sampling_params();
    let request = TurnRequest {
        system: None,
        messages: &messages,
        tools: &tools,
        params: &params,
    };
    backend.turn(&request).await
}

/// Where a failed rung-1 parse offloads its raw model text.
#[must_use]
pub fn rung1_raw_path(root: &Path, project_ref: &str) -> PathBuf {
    root.join(project_ref).join("rung1_raw.txt")
}

/// Where a failed rung-2 parse for cluster `cluster_index` offloads its raw
/// model text.
#[must_use]
pub fn rung2_raw_path(root: &Path, project_ref: &str, cluster_index: usize) -> PathBuf {
    root.join(project_ref)
        .join(format!("rung2-{cluster_index}-raw.txt"))
}

/// Strip one markdown code fence wrapping `text`, info string and all.
/// `None` unless the text both opens and closes with a fence, so prose that
/// merely mentions a fence is left alone for the next candidate to handle.
fn strip_code_fence(text: &str) -> Option<&str> {
    let body = text.trim().strip_prefix("```")?;
    // The opening fence may carry an info string on the same line (```json).
    let (_info, body) = body.split_once('\n')?;
    Some(body.trim_end().strip_suffix("```")?.trim())
}

/// The slice from the first `[` to the last `]`, for a model that wrote the
/// array correctly and then framed it in prose. Both delimiters are ASCII, so
/// the byte range is always a char boundary.
fn bracketed_slice(text: &str) -> Option<&str> {
    let start = text.find('[')?;
    let end = text.rfind(']')?;
    (end > start).then(|| &text[start..=end])
}

/// Parse a rung-1 turn's text as a JSON array of clusters. Strict about
/// SHAPE — a non-array top level or a missing field is a named `Err` naming
/// the failure and the field, never a panic and never a guess — but tolerant
/// about FRAMING: a model that fences its JSON, or wraps it in a sentence,
/// still gets parsed.
///
/// That tolerance is not politeness. Rung 1 is single-shot against a metered
/// lane and the loop runs unattended, so a framing quirk costs a silently
/// wasted night plus one paid inference. [`RUNG1_INSTRUCTION`] asks for a
/// bare array, and a fence is the single most common thing a model does to
/// JSON anyway; instructing alone leaves the most likely failure the one we
/// did nothing about. Belt and braces — this fired on the first live run
/// against photoqueue on 2026-09-21, where line 1 of the offloaded raw text
/// was a fence and the 14 clusters inside it were perfectly good.
///
/// The candidates are tried in order of how much they assume, and each one
/// still has to parse as an array of well-formed cluster objects, so a wrong
/// guess fails exactly as loudly as no guess. The reported error is always
/// the STRICT one, since the later candidates' complaints are about text the
/// model never meant as JSON.
///
/// # Errors
/// `Err` naming why the text could not be parsed into clusters.
pub fn parse_clusters(turn: &AssistantTurn) -> Result<Vec<Cluster>, String> {
    let text = turn.text();
    let trimmed = text.trim();
    let value: Value = match serde_json::from_str(trimmed) {
        Ok(value) => value,
        Err(err) => [strip_code_fence(trimmed), bracketed_slice(trimmed)]
            .into_iter()
            .flatten()
            .find_map(|candidate| serde_json::from_str(candidate).ok())
            .ok_or_else(|| format!("somnus: rung-1 parse failed ({err})"))?,
    };
    let array = value
        .as_array()
        .ok_or_else(|| "somnus: rung-1 parse failed (expected a JSON array)".to_string())?;
    let mut clusters = Vec::with_capacity(array.len());
    for item in array {
        clusters.push(parse_cluster(item)?);
    }
    Ok(clusters)
}

/// Parse one cluster object; every field is required except
/// `owning_map_id`, which may be null.
fn parse_cluster(item: &Value) -> Result<Cluster, String> {
    let label = item
        .get("label")
        .and_then(Value::as_str)
        .ok_or_else(|| "somnus: rung-1 cluster is missing the string field `label`".to_string())?;
    let members = item.get("member_entry_ids").ok_or_else(|| {
        "somnus: rung-1 cluster is missing the field `member_entry_ids`".to_string()
    })?;
    let member_entry_ids: Vec<String> = members
        .as_array()
        .ok_or_else(|| {
            "somnus: rung-1 parse failed (`member_entry_ids` is not an array)".to_string()
        })?
        .iter()
        .map(|entry| {
            entry.as_str().map(str::to_string).ok_or_else(|| {
                "somnus: rung-1 parse failed (`member_entry_ids` holds a non-string)".to_string()
            })
        })
        .collect::<Result<_, _>>()?;
    let owning_map_id = match item.get("owning_map_id") {
        None | Some(Value::Null) => None,
        Some(value) => Some(
            value
                .as_str()
                .ok_or_else(|| {
                    "somnus: rung-1 parse failed (`owning_map_id` is not a string or null)"
                        .to_string()
                })?
                .to_string(),
        ),
    };
    let merit_reason = item
        .get("merit_reason")
        .and_then(Value::as_str)
        .map(str::to_string);
    Ok(Cluster {
        label: label.to_string(),
        member_entry_ids,
        owning_map_id,
        merit_reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use harness::model::{StopReason, Usage};
    use harness::test_support::MockBackend;

    fn usage() -> Usage {
        Usage {
            input_tokens: 100,
            output_tokens: 10,
            cache_read_tokens: Some(1),
            cache_write_tokens: None,
            reasoning_tokens: None,
        }
    }

    fn text_turn(text: &str) -> AssistantTurn {
        AssistantTurn {
            content: vec![ContentBlock::Text(text.to_string())],
            stop_reason: StopReason::EndTurn,
            usage: usage(),
        }
    }

    fn tool_call_turn(calls: &[(&str, Value)]) -> AssistantTurn {
        AssistantTurn {
            content: calls
                .iter()
                .map(|(name, input)| {
                    ContentBlock::ToolCall(ToolCallRequest {
                        id: format!("call-{name}"),
                        name: (*name).to_string(),
                        input: input.clone(),
                    })
                })
                .collect(),
            stop_reason: StopReason::ToolUse,
            usage: usage(),
        }
    }

    // --- the pinned constants --------------------------------------------

    #[test]
    fn the_model_and_output_cap_are_pinned_to_the_published_table() {
        assert_eq!(SOMNUS_MODEL_ID, "claude-sonnet-5");
        assert_eq!(SOMNUS_MAX_TOKENS, 128_000);
        assert_eq!(
            harness::anthropic::output_cap_for_model(SOMNUS_MODEL_ID),
            Some(SOMNUS_MAX_TOKENS)
        );
        assert_eq!(
            somnus_sampling_params(),
            SamplingParams {
                max_tokens: 128_000,
                temperature: None,
                stop_sequences: vec![],
            }
        );
    }

    // --- rung 1: the message shape ---------------------------------------

    #[test]
    fn rung1_messages_builds_exactly_the_three_pinned_messages() {
        let filtered = crate::loop_input::fixture();
        let messages = rung1_messages("demo-project", &filtered);
        assert_eq!(messages.len(), 3);

        // (1) the synthetic assistant tool call.
        let Message::Assistant { content } = &messages[0] else {
            panic!("message 1 must be the synthetic assistant tool call");
        };
        assert_eq!(content.len(), 1);
        let ContentBlock::ToolCall(call) = &content[0] else {
            panic!("message 1 must hold one tool call");
        };
        assert_eq!(call.id, "somnus-loop-input-1");
        assert_eq!(call.name, "load_loop_input");
        assert_eq!(call.input, json!({"project_ref": "demo-project"}));

        // (2) the tool result carrying the filtered payload.
        let Message::User { content } = &messages[1] else {
            panic!("message 2 must be the tool result");
        };
        assert_eq!(content.len(), 1);
        let UserBlock::ToolResult {
            call_id,
            content: payload,
            is_error,
        } = &content[0]
        else {
            panic!("message 2 must hold one tool result");
        };
        assert_eq!(call_id, "somnus-loop-input-1");
        assert!(!is_error, "the injected payload is not an error");
        let parsed: crate::loop_input::LoopInput =
            serde_json::from_str(payload).expect("the payload round-trips");
        assert_eq!(parsed, filtered);

        // (3) the pinned instruction.
        let Message::User { content } = &messages[2] else {
            panic!("message 3 must be the instruction");
        };
        assert_eq!(content, &[UserBlock::Text(RUNG1_INSTRUCTION.to_string())]);
    }

    #[test]
    fn the_rung1_instruction_is_pinned_in_full() {
        assert_eq!(
            RUNG1_INSTRUCTION,
            "You are given the project's entries, existing mental maps, and candidate pockets as a tool result. Group the entries into subject-area clusters. Reply with ONLY a JSON array of objects, each {\"label\": string, \"member_entry_ids\": [string], \"owning_map_id\": string | null, \"merit_reason\": string | null}. Do not wrap the array in a markdown code fence and do not write any prose around it. An entry may appear in no cluster or in several clusters; do not force an assignment. How to carve, in order of importance. (1) One subject per label: if you find yourself joining two nouns with an ampersand, a slash or a comma, you have found two clusters, so split them instead of naming the union. (2) Emit the clusters in merit order, the one most worth a new map first, and give that first cluster a merit_reason of one sentence saying why it leads; later clusters may leave merit_reason null. (3) A run of consecutive entry ids is authoring provenance, not a subject area: ids tell you when entries were written, never what they are about, so never let an id range hold a cluster together. (4) If a candidate pocket does not survive as a cluster, say so in the merit_reason of the cluster that absorbed it rather than letting it disappear without account. (5) Do not name a cluster after a source its entries merely cite, because several unrelated clusters may cite the same source. (6) Carve by consequence as well as similarity: a subject the entries treat as high risk or high concern deserves its own cluster even when its entries would otherwise scatter into larger ones."
        );
    }

    /// The exact shape that aborted the first live run: a fence with a `json`
    /// info string, the array inside it perfectly well-formed.
    #[test]
    fn parse_clusters_tolerates_a_json_info_string_fence() {
        let turn = text_turn(
            "```json\n[{\"label\":\"a\",\"member_entry_ids\":[\"kb-10001\"],\"owning_map_id\":null}]\n```",
        );
        let clusters = parse_clusters(&turn).expect("a fenced array is still an array");
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].label, "a");
    }

    #[test]
    fn parse_clusters_tolerates_a_bare_fence() {
        let turn = text_turn(
            "```\n[{\"label\":\"b\",\"member_entry_ids\":[],\"owning_map_id\":null}]\n```",
        );
        let clusters = parse_clusters(&turn).expect("an info string is not required");
        assert_eq!(clusters[0].label, "b");
    }

    #[test]
    fn parse_clusters_tolerates_prose_around_the_array() {
        let turn = text_turn(
            "Here are the clusters:\n[{\"label\":\"c\",\"member_entry_ids\":[],\"owning_map_id\":null}]\nLet me know if you want them merged.",
        );
        let clusters = parse_clusters(&turn).expect("a framed array is still an array");
        assert_eq!(clusters[0].label, "c");
    }

    /// Tolerance about framing must not become tolerance about shape: a fence
    /// around something that is not an array of clusters fails exactly as
    /// loudly as it did before, and reports the STRICT error rather than the
    /// candidate's complaint.
    #[test]
    fn parse_clusters_rejects_a_fence_around_garbage_with_the_strict_error() {
        let error = parse_clusters(&text_turn("```json\nnot json at all\n```"))
            .expect_err("a fence does not make garbage parseable");
        assert!(error.contains("rung-1 parse failed"), "error was {error}");
        assert!(
            error.contains("expected value at line 1 column 1"),
            "the strict error is the one reported, not the candidate's; got {error}"
        );
    }

    #[test]
    fn parse_clusters_rejects_a_fenced_non_array_top_level() {
        let error = parse_clusters(&text_turn("```json\n{\"label\":\"a\"}\n```"))
            .expect_err("a fenced object is still not an array");
        assert!(error.contains("expected a JSON array"), "error was {error}");
    }

    #[tokio::test]
    async fn rung1_turn_makes_one_call_with_no_tools_no_system_and_the_pinned_params() {
        let backend = MockBackend::from_turns(vec![text_turn("[]")]);
        let filtered = crate::loop_input::fixture();

        let turn = rung1_turn(&backend, "demo-project", &filtered)
            .await
            .expect("scripted turn");

        assert_eq!(backend.calls(), 1, "rung 1 is ONE inference");
        assert_eq!(turn.text(), "[]");
        let messages = backend.messages_seen();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0], rung1_messages("demo-project", &filtered));
        assert_eq!(backend.tools_seen(), vec![Vec::<Value>::new()]);
        assert_eq!(
            backend.systems_seen(),
            vec![None::<String>],
            "rung 1 sends NO system prompt"
        );
        assert_eq!(backend.params_seen(), vec![128_000]);
    }

    // --- rung-1 parse -----------------------------------------------------

    #[test]
    fn parse_clusters_accepts_an_entry_in_no_cluster() {
        // `kb-10004` appears in NEITHER cluster: forced assignment is a
        // defect, so the parse must succeed with both clusters intact.
        let turn = text_turn(
            "[{\"label\":\"a\",\"member_entry_ids\":[\"kb-10001\",\"kb-10002\",\"kb-10003\"],\"owning_map_id\":\"kb-20001\"},{\"label\":\"b\",\"member_entry_ids\":[\"kb-10005\",\"kb-10006\"],\"owning_map_id\":null}]",
        );
        let clusters = parse_clusters(&turn).expect("no forced assignment");
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].label, "a");
        assert_eq!(
            clusters[0].member_entry_ids,
            vec![
                "kb-10001".to_string(),
                "kb-10002".to_string(),
                "kb-10003".to_string(),
            ]
        );
        assert_eq!(clusters[0].owning_map_id.as_deref(), Some("kb-20001"));
        assert_eq!(clusters[1].owning_map_id, None);
        assert!(
            !clusters
                .iter()
                .flat_map(|cluster| cluster.member_entry_ids.iter())
                .any(|id| id == "kb-10004"),
            "kb-10004 belongs nowhere, and that is a legal answer"
        );
    }

    #[test]
    fn parse_clusters_accepts_overlapping_clusters() {
        // Two clusters sharing kb-10002: overlap is expected and correct,
        // never resolved — no dedup, no error.
        let turn = text_turn(
            "[{\"label\":\"a\",\"member_entry_ids\":[\"kb-10001\",\"kb-10002\"],\"owning_map_id\":null},{\"label\":\"b\",\"member_entry_ids\":[\"kb-10002\",\"kb-10003\"],\"owning_map_id\":null}]",
        );
        let clusters = parse_clusters(&turn).expect("overlap is not an error");
        assert_eq!(clusters.len(), 2);
        assert_eq!(
            clusters[0].member_entry_ids,
            vec!["kb-10001".to_string(), "kb-10002".to_string()]
        );
        assert_eq!(
            clusters[1].member_entry_ids,
            vec!["kb-10002".to_string(), "kb-10003".to_string()]
        );
    }

    #[test]
    fn parse_clusters_rejects_garbage_naming_the_parse_failure() {
        let error = parse_clusters(&text_turn("not json"))
            .expect_err("garbage must be a named error, never a panic");
        assert!(error.contains("rung-1 parse failed"), "error was {error}");
    }

    #[test]
    fn parse_clusters_rejects_a_non_array_top_level() {
        let error = parse_clusters(&text_turn("{\"label\":\"a\"}"))
            .expect_err("a non-array answer is a named error");
        assert!(error.contains("expected a JSON array"), "error was {error}");
    }

    #[test]
    fn parse_clusters_rejects_a_missing_field_by_name() {
        let error = parse_clusters(&text_turn("[{\"member_entry_ids\":[]}]"))
            .expect_err("a missing field is a named error");
        assert!(error.contains("`label`"), "error was {error}");
        let error = parse_clusters(&text_turn("[{\"label\":\"a\"}]"))
            .expect_err("a missing member list is a named error");
        assert!(error.contains("`member_entry_ids`"), "error was {error}");
    }

    #[test]
    fn parse_clusters_rejects_wrongly_typed_fields_by_name() {
        let error = parse_clusters(&text_turn(
            "[{\"label\":\"a\",\"member_entry_ids\":\"no\"}]",
        ))
        .expect_err("a non-array member list is a named error");
        assert!(error.contains("`member_entry_ids`"), "error was {error}");
        let error = parse_clusters(&text_turn(
            "[{\"label\":\"a\",\"member_entry_ids\":[3],\"owning_map_id\":null}]",
        ))
        .expect_err("a non-string member id is a named error");
        assert!(error.contains("non-string"), "error was {error}");
        let error = parse_clusters(&text_turn(
            "[{\"label\":\"a\",\"member_entry_ids\":[],\"owning_map_id\":5}]",
        ))
        .expect_err("a non-string owning map id is a named error");
        assert!(error.contains("`owning_map_id`"), "error was {error}");
    }

    // --- rung 2 -----------------------------------------------------------

    #[test]
    fn the_rung2_instruction_names_the_exact_map_id_for_an_unmapped_cluster() {
        let cluster = Cluster {
            label: "wireguard-and-dns".to_string(),
            member_entry_ids: vec!["kb-10001".to_string(), "kb-10002".to_string()],
            owning_map_id: None,
            merit_reason: None,
        };
        let rendered = render_rung2_instruction(&cluster, 3);
        // The exact id, because the model has to name it to point at its own
        // map and guessing our convention is what wrote nothing for 3 runs.
        assert!(
            rendered.contains("cluster_id set to the literal c3"),
            "{rendered}"
        );
        assert!(
            rendered.contains("map_id set to the literal somnus-new-c3"),
            "{rendered}"
        );
        // A create with no pointers is refused, so say so up front.
        assert!(
            rendered.contains("carrying no add_pointer ops is refused"),
            "{rendered}"
        );
        // And a gap has nothing to attach to here.
        assert!(
            rendered.contains("Do not emit propose_gap for this cluster"),
            "{rendered}"
        );
    }

    #[test]
    fn a_mapped_cluster_is_told_to_converge_rather_than_mint() {
        let cluster = Cluster {
            label: "wireguard-and-dns".to_string(),
            member_entry_ids: vec!["kb-10001".to_string()],
            owning_map_id: Some("kb-20001".to_string()),
            merit_reason: None,
        };
        let rendered = render_rung2_instruction(&cluster, 0);
        assert!(
            rendered.contains("already covered by map kb-20001"),
            "{rendered}"
        );
        assert!(
            rendered.contains("Do not create a second map"),
            "{rendered}"
        );
        assert!(
            !rendered.contains("Do not emit propose_gap"),
            "a mapped cluster may legitimately propose a gap: {rendered}"
        );
    }

    #[test]
    fn the_purity_warnings_survive_every_cluster_shape() {
        // The purity sentence is load-bearing, not decorative: it is the only
        // guard against the model writing a gloss the map-lint rejects, and
        // `e.g.` is the member of the list nobody predicts — the
        // dotted-identifier rule matches any `word.word`.
        for owning in [None, Some("kb-20001".to_string())] {
            let cluster = Cluster {
                label: "a".to_string(),
                member_entry_ids: vec!["kb-10001".to_string()],
                owning_map_id: owning,
                merit_reason: None,
            };
            let rendered = render_rung2_instruction(&cluster, 0);
            assert!(rendered.contains("e.g."), "the e.g. warning must survive");
            assert!(
                rendered.contains("no backticks"),
                "the backtick warning must survive"
            );
        }
    }

    #[test]
    fn rung2_messages_reuse_the_loop_input_pair_plus_the_cluster_instruction() {
        let filtered = crate::loop_input::fixture();
        let cluster = Cluster {
            label: "backup-drills".to_string(),
            member_entry_ids: vec!["kb-10006".to_string(), "kb-10007".to_string()],
            owning_map_id: None,
            merit_reason: None,
        };
        let messages = rung2_messages("demo-project", &filtered, &cluster, 0);
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0], rung1_messages("demo-project", &filtered)[0]);
        assert_eq!(messages[1], rung1_messages("demo-project", &filtered)[1]);
        assert_eq!(
            messages[2],
            Message::User {
                content: vec![UserBlock::Text(render_rung2_instruction(&cluster, 0))],
            }
        );
    }

    #[tokio::test]
    async fn rung2_turn_is_one_call_per_cluster_with_exactly_the_four_op_tools() {
        let clusters = [
            Cluster {
                label: "a".to_string(),
                member_entry_ids: vec!["kb-10001".to_string()],
                owning_map_id: None,
                merit_reason: None,
            },
            Cluster {
                label: "b".to_string(),
                member_entry_ids: vec!["kb-10002".to_string()],
                owning_map_id: Some("kb-20001".to_string()),
                merit_reason: None,
            },
        ];
        let backend = MockBackend::from_turns(vec![
            tool_call_turn(&[("no_change", json!({"cluster_id": "a"}))]),
            tool_call_turn(&[("no_change", json!({"cluster_id": "b"}))]),
        ]);
        let filtered = crate::loop_input::fixture();

        for (index, cluster) in clusters.iter().enumerate() {
            rung2_turn(&backend, "demo-project", &filtered, cluster, index)
                .await
                .expect("scripted turn");
        }

        // One inference PER cluster, single-shot.
        assert_eq!(backend.calls(), 2);
        let tools_seen = backend.tools_seen();
        assert_eq!(tools_seen.len(), 2);
        for tools in &tools_seen {
            assert_eq!(tools.len(), 4, "exactly the four op tools are callable");
            let mut names: Vec<&str> = tools
                .iter()
                .map(|schema| schema["name"].as_str().expect("a name"))
                .collect();
            names.sort_unstable();
            assert_eq!(
                names,
                ["add_pointer", "create_map", "propose_gap", "strike_gap"]
            );
        }
        assert_eq!(tools_seen[0], crate::ops::op_tool_schemas());
        assert_eq!(
            backend.systems_seen(),
            vec![None::<String>, None::<String>],
            "rung 2 sends NO system prompt"
        );
        assert_eq!(backend.params_seen(), vec![128_000, 128_000]);
        let messages = backend.messages_seen();
        assert_eq!(
            messages[0],
            rung2_messages("demo-project", &filtered, &clusters[0], 0)
        );
        assert_eq!(
            messages[1],
            rung2_messages("demo-project", &filtered, &clusters[1], 1)
        );
    }

    #[tokio::test]
    async fn a_backend_error_propagates_verbatim() {
        let backend = MockBackend::new(vec![Err(harness::model::BackendError::Terminal {
            kind: harness::model::TerminalKind::Auth,
            message: "bad key".to_string(),
        })]);
        let filtered = crate::loop_input::fixture();
        let error = rung1_turn(&backend, "demo-project", &filtered)
            .await
            .expect_err("the backend error propagates");
        assert!(matches!(
            error,
            harness::model::BackendError::Terminal { .. }
        ));
        assert_eq!(backend.calls(), 1);
    }

    // --- the raw-text offload paths ---------------------------------------

    #[test]
    fn the_raw_paths_are_pinned_per_project_and_per_cluster() {
        let root = Path::new("/tmp/somnus-root");
        assert_eq!(
            rung1_raw_path(root, "demo-project"),
            PathBuf::from("/tmp/somnus-root/demo-project/rung1_raw.txt")
        );
        assert_eq!(
            rung2_raw_path(root, "demo-project", 3),
            PathBuf::from("/tmp/somnus-root/demo-project/rung2-3-raw.txt")
        );
    }
}
