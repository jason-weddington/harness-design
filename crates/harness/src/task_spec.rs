//! `TaskSpec` — the wire type the agent-gtd-dispatch worker deserialises a
//! groomed GTD item into before handing it to the harness.
//!
//! # Field naming
//!
//! Field names are **verbatim** matches of the GTD groomed-item JSON shape so
//! that deserialisation from the dispatcher's payload is a projection (field
//! selection), not a mapping (rename). No `#[serde(rename)]` is used anywhere
//! in this module.
//!
//! # Required fields
//!
//! All fields are **required**: no `#[serde(default)]`. A missing field is a
//! parse error; the caller should treat that as a bad-spec condition (exit 1)
//! rather than silently constructing a half-formed spec. This is especially
//! important for `gate_command` — a silently-defaulted empty string would
//! leave the downstream exit-code map unable to distinguish "no gate
//! configured" from "spec was truncated".
//!
//! # Unknown fields
//!
//! Extra JSON keys at any level are **tolerated** (no
//! `#[serde(deny_unknown_fields)]`). The GTD item shape may carry additional
//! metadata (`"id"`, `"status"`, `"labels"`, …) that the harness does not
//! consume; ignoring unknown keys keeps the wire contract forward-compatible.

use serde::{Deserialize, Serialize};

/// The wire type for a groomed GTD task item, as serialised by the
/// agent-gtd-dispatch worker.
///
/// Field names mirror GTD's groomed-item JSON verbatim so deserialisation is
/// a projection (select the fields you need), not a mapping (rename on the
/// way in). All five fields are required — a missing field is a parse error.
///
/// Extra keys in the JSON payload are silently ignored; the GTD item shape
/// routinely carries metadata (e.g. `"id"`, `"status"`, `"labels"`) that the
/// harness does not consume.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskSpec {
    /// Short title of the task (rendered as a sub-level heading in the task
    /// prompt — the engine wraps it under `# Task`).
    pub title: String,
    /// Prose description of what the task is and why it matters.
    pub description: String,
    /// Ordered list of acceptance criteria; each entry is a prose string.
    pub acceptance_criteria: Vec<String>,
    /// Ordered list of files the agent must read and/or edit.
    pub files_to_modify: Vec<FileToModify>,
    /// Shell command the agent (and CI) run to verify the task is complete.
    pub gate_command: String,
}

/// One entry in [`TaskSpec::files_to_modify`]: the file path paired with a
/// description of the change to make there.
///
/// Both field names are verbatim GTD wire names. Both fields are required.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileToModify {
    /// Workspace-relative path to the file.
    pub path: String,
    /// Description of the change to make in that file.
    pub change: String,
}

#[cfg(test)]
mod tests {
    use super::{FileToModify, TaskSpec};

    fn sample_spec() -> TaskSpec {
        TaskSpec {
            title: "Add widget support".to_string(),
            description: "The widget subsystem needs support for 5 > 3 & \"quoted\".".to_string(),
            acceptance_criteria: vec![
                "Widget renders correctly".to_string(),
                "Widget handles empty input".to_string(),
            ],
            files_to_modify: vec![
                FileToModify {
                    path: "src/widget.rs".to_string(),
                    change: "Add render method".to_string(),
                },
                FileToModify {
                    path: "src/lib.rs".to_string(),
                    change: "Export widget module".to_string(),
                },
            ],
            gate_command: "cargo nextest run --workspace".to_string(),
        }
    }

    /// Serde round-trip: serialise then deserialise and assert equality.
    #[test]
    fn round_trip() {
        let original = sample_spec();
        let json = serde_json::to_string(&original).expect("serialisation must not fail");
        let recovered: TaskSpec =
            serde_json::from_str(&json).expect("deserialisation must not fail");
        assert_eq!(original, recovered);
    }

    /// Pin the verbatim wire field names by deserialising a hand-written JSON
    /// literal and asserting each field value. An accidental `#[serde(rename)]`
    /// would break this test.
    #[test]
    fn verbatim_field_names() {
        let json = r#"{
            "title": "My Task",
            "description": "Does things.",
            "acceptance_criteria": ["AC one", "AC two"],
            "files_to_modify": [
                {"path": "foo/bar.rs", "change": "add fn"},
                {"path": "baz/qux.rs", "change": "remove dead code"}
            ],
            "gate_command": "cargo nextest run"
        }"#;
        let spec: TaskSpec = serde_json::from_str(json).expect("must parse");
        assert_eq!(spec.title, "My Task");
        assert_eq!(spec.description, "Does things.");
        assert_eq!(spec.acceptance_criteria, vec!["AC one", "AC two"]);
        assert_eq!(spec.files_to_modify.len(), 2);
        assert_eq!(spec.files_to_modify[0].path, "foo/bar.rs");
        assert_eq!(spec.files_to_modify[0].change, "add fn");
        assert_eq!(spec.files_to_modify[1].path, "baz/qux.rs");
        assert_eq!(spec.files_to_modify[1].change, "remove dead code");
        assert_eq!(spec.gate_command, "cargo nextest run");
    }

    /// Extra top-level keys AND extra nested keys inside `files_to_modify`
    /// elements are silently ignored — no `deny_unknown_fields` means
    /// forward-compatibility with GTD shape growth.
    #[test]
    fn tolerates_extra_fields() {
        let json = r#"{
            "id": "abc",
            "title": "Task",
            "description": "Desc.",
            "acceptance_criteria": ["AC"],
            "files_to_modify": [
                {"path": "f.rs", "change": "c", "priority": 1}
            ],
            "gate_command": "cargo test",
            "status": "active"
        }"#;
        let spec: TaskSpec = serde_json::from_str(json).expect("extra fields must be tolerated");
        assert_eq!(spec.title, "Task");
        assert_eq!(spec.files_to_modify[0].path, "f.rs");
        assert_eq!(spec.files_to_modify[0].change, "c");
    }

    /// A JSON object missing `gate_command` must return `Err`, not a silently
    /// defaulted empty string — the downstream exit-code map depends on this
    /// being a hard parse failure.
    #[test]
    fn missing_gate_command_is_error() {
        let json = r#"{
            "title": "T",
            "description": "D",
            "acceptance_criteria": [],
            "files_to_modify": []
        }"#;
        let result = serde_json::from_str::<TaskSpec>(json);
        assert!(
            result.is_err(),
            "missing gate_command must be a parse error"
        );
    }

    /// A `files_to_modify` element missing the `change` field must return
    /// `Err` — the field is required, not optional.
    #[test]
    fn missing_nested_change_is_error() {
        let json = r#"{
            "title": "T",
            "description": "D",
            "acceptance_criteria": [],
            "files_to_modify": [{"path": "f.rs"}],
            "gate_command": "cargo test"
        }"#;
        let result = serde_json::from_str::<TaskSpec>(json);
        assert!(
            result.is_err(),
            "files_to_modify element missing `change` must be a parse error"
        );
    }
}
