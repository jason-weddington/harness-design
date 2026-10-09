"""Tests for talos_harbor.mined_score: parity ports of the Rust mined_eval.rs unit tests,
fixture-capture parity (read in place), and CLI end-to-end cases."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from talos_harbor import mined_score
from talos_harbor.mined_score import (
    Score,
    TaskScore,
    TestStatus,
    build_collection_errors,
    build_unexcluded_red,
    exit_code_fault_reason,
    find_status_for_task_id,
    is_test_path,
    match_task_id,
    matches_exclusion,
    normalize,
    parse_authored_tests,
    parse_pytest,
    parse_pytest_summary_totals,
    parse_short_summary_line,
    resolve,
    strip_param_suffix,
)

SCRIPT = Path(mined_score.__file__)
FIXTURES = Path(__file__).resolve().parents[2] / "crates/harness/testdata/mined_eval"

# ---- task fixtures (transcribed from mined_eval.rs) --------------------------

SAMPLE_TASK: dict[str, Any] = {
    "id": "cleanr-abcdef1",
    "gate_command": "uv run --frozen pytest tests/test_cleanr.py -q",
    "fail_to_pass": ["TestCommentFilters::test_owner_comments_skipped"],
    "positive_controls": ["TestCommentFilters::test_reader_comments_kept"],
    "pass_to_pass_exclusions": [
        "TestCalcBadScore::*",
        "TestOldWireFormat::test_legacy_still_parses",
    ],
    "sealed": [{"path": "tests/test_cleanr.py"}],
}

DEADLOCK_TASK: dict[str, Any] = {
    "id": "agent-gtd-rollout-deadlock",
    "gate_command": "uv run --frozen pytest tests/test_dispatch_service.py "
    "tests/test_rollout_service.py tests/test_rollout_executor.py -q",
    "fail_to_pass": [
        "tests/test_dispatch_service.py::test_create_run_manage_mode_rejected",
        "tests/test_rollout_service.py::test_complete_item_in_rollout_from_ready",
        "tests/test_rollout_service.py::test_complete_item_in_rollout_from_ready_unblocks_downstream",
        "tests/test_rollout_executor.py::test_manage_dispatch_does_not_flip_item_status",
    ],
    "positive_controls": [
        "tests/test_rollout_service.py::test_complete_item_in_rollout_rejects_pending_status",
        "tests/test_rollout_service.py::test_managed_rollout_happy_path",
        "tests/test_rollout_executor.py::test_manage_dispatch_flips_wave_to_running",
        "tests/test_rollout_executor.py::test_manage_dispatch_emits_wave_started_event",
        "tests/test_rollout_executor.py::test_happy_path_plan_rollout_to_complete_item_in_rollout",
    ],
    "pass_to_pass_exclusions": [],
    "sealed": [
        {"path": "tests/test_dispatch_service.py"},
        {"path": "tests/test_rollout_service.py"},
        {"path": "tests/test_rollout_executor.py"},
    ],
}

FROM_JSON_TASK: dict[str, Any] = {
    "id": "agent-gtd-from-json-contract",
    "gate_command": "uv run --frozen pytest tests/test_cli.py -q",
    "fail_to_pass": [
        "test_do_add_item_honors_project_id_and_status_from_payload",
        "test_do_add_item_persists_priority_due_date_assigned_to",
        "test_do_add_item_unknown_key_raises_and_creates_nothing",
        "test_cmd_add_item_unknown_key_exits_nonzero",
        "test_do_add_item_http_mode_threads_priority_due_date_assigned_to",
        "test_http_post_create_item_includes_priority_due_date_assigned_to",
    ],
    "positive_controls": [
        "test_do_add_item_flags_override_payload",
        "test_http_post_create_item_omits_priority_due_date_assigned_to_when_none",
    ],
    "pass_to_pass_exclusions": [],
    "sealed": [{"path": "tests/test_cli.py"}],
}


def task_of(d: dict[str, Any]) -> TaskScore:
    return TaskScore(
        id=d["id"],
        fail_to_pass=list(d["fail_to_pass"]),
        positive_controls=list(d["positive_controls"]),
        pass_to_pass_exclusions=list(d["pass_to_pass_exclusions"]),
        gate_command=d["gate_command"],
        sealed=[s["path"] for s in d["sealed"]],
    )


def sample_task() -> TaskScore:
    return task_of(SAMPLE_TASK)


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


OWNER = "tests/test_cleanr.py::TestCommentFilters::test_owner_comments_skipped"
READER = "tests/test_cleanr.py::TestCommentFilters::test_reader_comments_kept"

# ---- normalize / matching -----------------------------------------------------


def test_normalize_strips_leading_py_prefix() -> None:
    assert normalize(OWNER) == "TestCommentFilters::test_owner_comments_skipped"
    assert normalize("tests/test_x.py::test_fn") == "test_fn"
    assert normalize("Class::test") == "Class::test"


def test_normalize_preserves_param_suffix() -> None:
    assert normalize("tests/t.py::Class::test[case-a]") == "Class::test[case-a]"


def test_normalize_returns_input_when_first_segment_is_not_py() -> None:
    assert normalize("mymod::inner::test_fn") == "mymod::inner::test_fn"


def test_strip_param_suffix_trims_bracketed_suffix() -> None:
    assert strip_param_suffix("Class::test[a]") == "Class::test"
    assert strip_param_suffix("Class::test") == "Class::test"
    assert strip_param_suffix("Class::test[a]::more") == "Class::test[a]::more"


def test_match_task_id_handles_param_matching() -> None:
    assert match_task_id("tests/t.py::Class::test", "Class::test", "Class::test")
    assert match_task_id("tests/t.py::Class::test[a]", "Class::test[a]", "Class::test")
    assert match_task_id("tests/t.py::Class::test", "Class::test", "Class::test[a]")
    assert not match_task_id(
        "tests/t.py::OtherClass::test_owner_comments_skipped",
        "OtherClass::test_owner_comments_skipped",
        "TestCommentFilters::test_owner_comments_skipped",
    )


def test_file_qualified_task_ids_match_in_raw_space() -> None:
    assert match_task_id(
        "tests/test_rollout_service.py::test_complete_item_in_rollout_from_ready",
        "test_complete_item_in_rollout_from_ready",
        "tests/test_rollout_service.py::test_complete_item_in_rollout_from_ready",
    )
    assert not match_task_id(
        "tests/test_rollout_executor.py::test_complete_item_in_rollout_from_ready",
        "test_complete_item_in_rollout_from_ready",
        "tests/test_rollout_service.py::test_complete_item_in_rollout_from_ready",
    )
    assert match_task_id(
        "tests/test_a.py::test_fn[case-1]",
        "test_fn[case-1]",
        "tests/test_a.py::test_fn",
    )


def test_matches_exclusion_supports_literal_and_glob() -> None:
    assert matches_exclusion(
        "tests/t.py::TestCalcBadScore::test_low_score_flagged",
        "TestCalcBadScore::test_low_score_flagged",
        "TestCalcBadScore::*",
    )
    assert matches_exclusion(
        "tests/t.py::TestOldWireFormat::test_legacy_still_parses",
        "TestOldWireFormat::test_legacy_still_parses",
        "TestOldWireFormat::test_legacy_still_parses",
    )
    assert not matches_exclusion(
        "tests/t.py::OtherClass::test_other",
        "OtherClass::test_other",
        "TestCalcBadScore::*",
    )
    assert matches_exclusion(
        "tests/a.py::TestX::test_one", "TestX::test_one", "tests/a.py::TestX::*"
    )
    assert not matches_exclusion(
        "tests/b.py::TestX::test_one", "TestX::test_one", "tests/a.py::TestX::*"
    )


# ---- pytest parser --------------------------------------------------------------


def synthetic_pytest_output() -> str:
    return """
=== test session starts ===
platform linux -- Python 3.12.0
collected 5 items

tests/test_cleanr.py ..F..                                            [100%]

=== FAILURES ===
______________________ TestX::test_bad ______________________

    def test_bad():
>       assert False
E       assert False

tests/test_cleanr.py:5: AssertionError

=== short test summary info ===
PASSED tests/test_cleanr.py::TestX::test_ok
PASSED tests/test_cleanr.py::TestX::test_also_ok
FAILED tests/test_cleanr.py::TestX::test_bad
SKIPPED tests/test_cleanr.py::TestY::test_skipped
XFAIL tests/test_cleanr.py::TestY::test_expected_fail
=== 2 passed, 1 failed, 1 skipped, 1 xfailed in 0.12s ===
"""


def test_pytest_parser_reads_short_summary_lines_only() -> None:
    report = parse_pytest(synthetic_pytest_output())
    m = report.statuses
    assert report.summary_count == 5
    assert len(m) == 5
    assert m.get("tests/test_cleanr.py::TestX::test_ok") is TestStatus.PASSED
    assert m.get("tests/test_cleanr.py::TestX::test_bad") is TestStatus.FAILED
    assert m.get("tests/test_cleanr.py::TestY::test_skipped") is TestStatus.SKIPPED
    assert m.get("tests/test_cleanr.py::TestY::test_expected_fail") is TestStatus.XFAIL


def test_pytest_parser_handles_error_and_xpass() -> None:
    out = (
        "=========================== short test summary info ============================\n"
        "ERROR tests/test_x.py::TestX::test_broken\n"
        "XPASS tests/test_x.py::TestX::test_unex_pass\n"
        "=== 0 passed, 1 error, 1 xpassed in 0.01s ==="
    )
    report = parse_pytest(out)
    assert report.summary_count == 2
    assert report.statuses.get("tests/test_x.py::TestX::test_broken") is TestStatus.ERROR
    assert report.statuses.get("tests/test_x.py::TestX::test_unex_pass") is TestStatus.XPASS


def test_parse_short_summary_line_ignores_non_status_lines() -> None:
    assert parse_short_summary_line("=== FAILURES ===") is None
    assert parse_short_summary_line("  ") is None
    assert parse_short_summary_line("tests/x.py .") is None
    assert parse_short_summary_line("passed tests/x.py::t") is None


def test_pytest_parser_skips_status_line_with_empty_nodeid() -> None:
    out = (
        "=========================== short test summary info ============================\n"
        "PASSED \nPASSED tests/x.py::T::real\n=== 1 passed in 0.0s ==="
    )
    report = parse_pytest(out)
    assert len(report.statuses) == 1
    assert "tests/x.py::T::real" in report.statuses
    assert report.summary_count == 1


def test_parse_pytest_summary_totals_sums_all_buckets() -> None:
    assert (
        parse_pytest_summary_totals("=== 2 passed, 1 failed, 1 skipped, 1 xfailed in 0.1s ===") == 5
    )
    assert parse_pytest_summary_totals("=== 3 passed, 2 warnings in 0.5s ===") == 3
    assert parse_pytest_summary_totals("=== FAILURES ===") is None


def test_parse_pytest_summary_totals_strips_full_width_banner_padding() -> None:
    assert (
        parse_pytest_summary_totals(
            "============== 1 failed, 6 passed, 1 skipped, 8 warnings in 1.85s =============="
        )
        == 8
    )
    assert (
        parse_pytest_summary_totals(
            "============================== 68 passed in 0.30s =============================="
        )
        == 68
    )


def test_parse_pytest_summary_totals_edge_shapes() -> None:
    # Not '==='-prefixed; no closing '==='; usize parse rules.
    assert parse_pytest_summary_totals("2 passed in 0.01s") is None
    assert parse_pytest_summary_totals("=== 2 passed") is None
    assert parse_pytest_summary_totals("=== +2 passed, x failed, -1 error, 3 ===") == 2


def test_dots_only_output_yields_parse_empty_invalid() -> None:
    dots_only = "tests/x.py ...F.                                     [100%]\n"
    report = parse_pytest(dots_only)
    assert report.statuses == {}
    assert report.summary_count is None
    v = resolve(report.statuses, report.summary_count, sample_task())
    assert v.score is Score.INVALID
    assert v.detail == "parse-empty"


def test_status_lines_without_a_summary_header_parse_empty() -> None:
    inp = (
        "PASSED tests/test_x.py::test_a\nFAILED tests/test_x.py::test_b\n"
        "=== 1 passed, 1 failed in 0.01s ==="
    )
    report = parse_pytest(inp)
    assert report.statuses == {}
    assert report.summary_count == 2
    assert report.dropped_outside_section == 2
    assert report.sections_seen == 0
    v = resolve(report.statuses, report.summary_count, sample_task())
    assert v.score is Score.INVALID
    assert v.detail == "parse-empty"


QUIET_NO_BANNER = (
    "..                                                          [100%]\n"
    "==================================== PASSES ====================================\n"
    "=========================== short test summary info ============================\n"
    "PASSED tests/test_x.py::test_a\n"
    "PASSED tests/test_x.py::test_b\n"
    "2 passed in 0.01s"
)


def test_pytest_parser_parses_quiet_no_banner_capture() -> None:
    report = parse_pytest(QUIET_NO_BANNER)
    assert report.summary_count is None
    assert report.sections_seen == 1
    assert len(report.statuses) == 2
    assert report.dropped_outside_section == 0


def test_blank_line_terminates_the_summary_section() -> None:
    inp = (
        "=== short test summary info ===\nPASSED tests/test_x.py::test_a\n\n"
        "PASSED tests/test_y.py::test_b\n=== 1 passed in 0.01s ==="
    )
    report = parse_pytest(inp)
    assert len(report.statuses) == 1
    assert "tests/test_y.py::test_b" not in report.statuses
    assert report.dropped_outside_section == 1


def test_parser_line_split_and_whitespace_follow_rust() -> None:
    # CRLF lines, a \x1c (Python-only whitespace) is NOT stripped, and an
    # all-colon bracketed SKIPPED location loses every trailing ':'.
    out = (
        "=== short test summary info ===\r\n"
        "PASSED tests/a.py::t\r\n"
        "SKIPPED [2] tests/a.py:9::: reason\r\n"
        "SKIPPED [3]\r\n"
        "\x1cPASSED tests/b.py::t\r\n"
    )
    report = parse_pytest(out)
    assert report.statuses == {
        "tests/a.py::t": TestStatus.PASSED,
        "tests/a.py:9": TestStatus.SKIPPED,
    }
    # The bracketed SKIPPED with no location yields an empty key: not counted.
    assert report.dropped_outside_section == 0


# ---- resolve() ------------------------------------------------------------------


def test_resolve_skips_count_check_when_summary_count_is_none() -> None:
    parsed = {OWNER: TestStatus.PASSED, READER: TestStatus.PASSED}
    assert resolve(parsed, None, sample_task()).score is Score.RESOLVED


def test_resolve_returns_resolved_on_all_green() -> None:
    parsed = {OWNER: TestStatus.PASSED, READER: TestStatus.PASSED}
    v = resolve(parsed, 2, sample_task())
    assert v.score is Score.RESOLVED
    assert v.detail == ""
    assert v.fail_to_pass_status == [
        ("TestCommentFilters::test_owner_comments_skipped", TestStatus.PASSED)
    ]
    assert v.positive_control_status == [
        ("TestCommentFilters::test_reader_comments_kept", TestStatus.PASSED)
    ]


def test_resolve_unresolved_when_fail_to_pass_is_red() -> None:
    parsed = {OWNER: TestStatus.FAILED, READER: TestStatus.PASSED}
    v = resolve(parsed, 2, sample_task())
    assert v.score is Score.UNRESOLVED
    assert v.detail == "fail-to-pass-red"
    assert "TestCommentFilters::test_owner_comments_skipped" in v.missing_fail_to_pass


def test_resolve_unresolved_when_fail_to_pass_is_missing() -> None:
    v = resolve({READER: TestStatus.PASSED}, 1, sample_task())
    assert v.score is Score.UNRESOLVED
    assert v.missing_fail_to_pass


def test_resolve_unresolved_when_positive_control_red() -> None:
    parsed = {OWNER: TestStatus.PASSED, READER: TestStatus.FAILED}
    v = resolve(parsed, 2, sample_task())
    assert v.score is Score.UNRESOLVED
    assert v.detail == "positive-control-red"
    assert v.collection_errors == []


def test_resolve_invalid_when_positive_control_uncollected() -> None:
    v = resolve({OWNER: TestStatus.PASSED}, 1, sample_task())
    assert v.score is Score.INVALID
    assert v.detail.startswith("positive-control-uncollected")
    assert v.detail == "positive-control-uncollected: TestCommentFilters::test_reader_comments_kept"
    assert v.fail_to_pass_status == [
        ("TestCommentFilters::test_owner_comments_skipped", TestStatus.PASSED)
    ]


def test_resolve_unresolved_when_non_excluded_test_is_red() -> None:
    parsed = {
        OWNER: TestStatus.PASSED,
        READER: TestStatus.PASSED,
        "tests/test_cleanr.py::TestSomethingElse::test_broke": TestStatus.FAILED,
    }
    v = resolve(parsed, 3, sample_task())
    assert v.score is Score.UNRESOLVED
    assert v.detail == "unexcluded-red"
    assert v.uncovered_red == ["TestSomethingElse::test_broke"]
    assert v.missing_fail_to_pass == []


def test_resolve_ignores_class_glob_excluded_red() -> None:
    parsed = {
        OWNER: TestStatus.PASSED,
        READER: TestStatus.PASSED,
        "tests/test_cleanr.py::TestCalcBadScore::test_low_score": TestStatus.FAILED,
    }
    v = resolve(parsed, 3, sample_task())
    assert v.score is Score.RESOLVED
    assert v.excluded_red == [("TestCalcBadScore::test_low_score", "TestCalcBadScore::*")]


def test_resolve_ignores_literal_excluded_red() -> None:
    parsed = {
        OWNER: TestStatus.PASSED,
        READER: TestStatus.PASSED,
        "tests/test_cleanr.py::TestOldWireFormat::test_legacy_still_parses": TestStatus.FAILED,
    }
    v = resolve(parsed, 3, sample_task())
    assert v.score is Score.RESOLVED
    assert v.excluded_red == [
        (
            "TestOldWireFormat::test_legacy_still_parses",
            "TestOldWireFormat::test_legacy_still_parses",
        )
    ]


def test_resolve_flags_count_mismatch_as_invalid() -> None:
    parsed = {OWNER: TestStatus.PASSED, READER: TestStatus.PASSED}
    v = resolve(parsed, 99, sample_task())
    assert v.score is Score.INVALID
    assert v.detail.startswith("parse-mismatch")
    assert v.detail == "parse-mismatch: 2 ids vs summary 99"


def test_parametrized_instances_score_individually_and_any_red_breaks() -> None:
    task = TaskScore(
        id="cleanr-abcdef1",
        fail_to_pass=["TestParamCase::test_it"],
        positive_controls=[],
        pass_to_pass_exclusions=[],
    )
    parsed = {
        "tests/t.py::TestParamCase::test_it[a]": TestStatus.PASSED,
        "tests/t.py::TestParamCase::test_it[b]": TestStatus.FAILED,
    }
    assert resolve(parsed, 2, task).score is Score.UNRESOLVED


def test_find_status_for_task_id_promotes_red_across_param_instances() -> None:
    normalized = [
        ("tests/t.py::Cls::test[a]", "Cls::test[a]", TestStatus.PASSED),
        ("tests/t.py::Cls::test[b]", "Cls::test[b]", TestStatus.FAILED),
    ]
    assert find_status_for_task_id(normalized, "Cls::test") is TestStatus.FAILED
    normalized2 = [
        ("tests/t.py::Cls::test[a]", "Cls::test[a]", TestStatus.FAILED),
        ("tests/t.py::Cls::test[b]", "Cls::test[b]", TestStatus.PASSED),
    ]
    assert find_status_for_task_id(normalized2, "Cls::test") is TestStatus.FAILED


def test_build_unexcluded_red_dedups_duplicate_normalized_ids() -> None:
    task = sample_task()
    parsed = {
        OWNER: TestStatus.PASSED,
        READER: TestStatus.PASSED,
        "tests/test_cleanr.py::Extra::test_a": TestStatus.FAILED,
    }
    parsed_alt = {
        OWNER: TestStatus.PASSED,
        READER: TestStatus.PASSED,
        "tests/test_cleanr.py::Extra::test_a": TestStatus.FAILED,
        "other/test_dup.py::Extra::test_a": TestStatus.FAILED,
    }
    for p in (parsed, parsed_alt):
        v = resolve(p, len(p), task)
        assert v.score is Score.UNRESOLVED
        assert len(v.uncovered_red) == 1
    normalized = [(r, normalize(r), s) for r, s in sorted(parsed_alt.items())]
    assert build_unexcluded_red(normalized, task) == ["Extra::test_a"]


def test_build_collection_errors_discriminates_file_level_errors() -> None:
    assert build_collection_errors(
        [("tests/test_cli.py", "tests/test_cli.py", TestStatus.ERROR)]
    ) == ["tests/test_cli.py"]
    assert (
        build_collection_errors(
            [("tests/test_x.py::TestX::test_broken", "TestX::test_broken", TestStatus.ERROR)]
        )
        == []
    )
    assert (
        build_collection_errors(
            [
                (
                    "agent_gtd.event_bus:event_bus.py:130",
                    "agent_gtd.event_bus:event_bus.py:130",
                    TestStatus.ERROR,
                )
            ]
        )
        == []
    )
    assert (
        build_collection_errors(
            [(OWNER, "TestCommentFilters::test_owner_comments_skipped", TestStatus.PASSED)]
        )
        == []
    )


def test_collection_error_arm_precedes_positive_control_arm() -> None:
    v = resolve({"tests/test_cleanr.py": TestStatus.ERROR}, 1, sample_task())
    assert v.score is Score.UNRESOLVED
    assert v.detail == "collection-error"
    assert v.collection_errors == ["tests/test_cleanr.py"]


# ---- authored tests ---------------------------------------------------------------


def test_is_test_path_matches_pytest_discovery_conventions() -> None:
    assert is_test_path("tests/test_cli.py")
    assert is_test_path("src/pkg/tests/helpers.py")
    assert is_test_path("src/test_thing.py")
    assert is_test_path("src/thing_test.py")
    assert not is_test_path("src/agent_gtd/cli.py")
    assert not is_test_path("README.md")
    assert not is_test_path("data/test_input.json")
    assert not is_test_path("src/latest/thing.py")


def test_parse_authored_tests_splits_added_from_modified() -> None:
    porcelain = (
        "?? tests/test_agent_authored.py\n"
        "A  tests/test_staged_new.py\n"
        " M tests/test_existing.py\n"
        "M  tests/test_staged_edit.py\n"
        " D tests/test_deleted.py\n"
        "R  tests/test_old.py -> tests/test_renamed.py\n"
        "?? src/agent_gtd/cli.py\n"
        " M src/agent_gtd/service.py\n"
    )
    added, modified = parse_authored_tests(porcelain)
    assert added == ["tests/test_agent_authored.py", "tests/test_staged_new.py"]
    assert modified == [
        "tests/test_existing.py",
        "tests/test_staged_edit.py",
        "tests/test_deleted.py",
        "tests/test_renamed.py",
    ]


def test_parse_authored_tests_ignores_short_and_empty_lines() -> None:
    assert parse_authored_tests("") == ([], [])
    assert parse_authored_tests("??\n M\n\n") == ([], [])


def test_parse_authored_tests_unquotes_paths_with_spaces() -> None:
    added, _ = parse_authored_tests('?? "tests/test a b.py"\n')
    assert added == ["tests/test a b.py"]


# ---- fixture-capture parity (read in place) ------------------------------------------


def test_pytest_parser_ignores_caplog_error_lines_outside_summary_section() -> None:
    report = parse_pytest(fixture("deadlock-trial-0.txt"))
    assert len(report.statuses) == 121
    assert report.summary_count == 121
    assert report.dropped_outside_section == 6
    assert report.sections_seen == 1
    assert "agent_gtd.event_bus:event_bus.py:130" not in report.statuses
    for key in report.statuses:
        assert "::" in key, key


def test_pytest_parser_uses_location_for_bracketed_skipped_lines() -> None:
    report = parse_pytest(fixture("attribution-trial-0.txt"))
    assert report.statuses.get("tests/test_dispatch_attribution.py:191") is TestStatus.SKIPPED
    assert "[1]" not in report.statuses
    assert len(report.statuses) == 8
    assert report.summary_count == 8
    assert report.dropped_outside_section == 0
    assert report.sections_seen == 1


def test_deadlock_capture_no_longer_scores_parse_mismatch() -> None:
    report = parse_pytest(fixture("deadlock-trial-0.txt"))
    v = resolve(report.statuses, report.summary_count, task_of(DEADLOCK_TASK))
    assert v.score is Score.UNRESOLVED
    assert v.missing_fail_to_pass == [
        "tests/test_dispatch_service.py::test_create_run_manage_mode_rejected",
        "tests/test_rollout_executor.py::test_manage_dispatch_does_not_flip_item_status",
    ]
    assert v.uncovered_red == []
    assert v.collection_errors == []
    assert len(v.fail_to_pass_status) == 4


def test_collection_error_scores_unresolved_not_invalid() -> None:
    report = parse_pytest(fixture("from-json-trial-1.txt"))
    assert len(report.statuses) == 1
    assert report.summary_count == 1
    v = resolve(report.statuses, report.summary_count, task_of(FROM_JSON_TASK))
    assert v.score is not Score.INVALID
    assert v.score is Score.UNRESOLVED
    assert v.collection_errors == ["tests/test_cli.py"]
    assert len(v.missing_fail_to_pass) == 6
    assert len(v.fail_to_pass_status) == 6
    assert v.uncovered_red == ["tests/test_cli.py"]


def _rust_lines(s: str) -> list[str]:
    return mined_score._rust_lines(s)


def test_truncated_capture_still_scores_parse_mismatch() -> None:
    lines = _rust_lines(fixture("deadlock-trial-0.txt"))
    hdr = next(i for i, ln in enumerate(lines) if "short test summary info" in ln)
    truncated = "\n".join([*lines[hdr : hdr + 61], lines[-1]])
    report = parse_pytest(truncated)
    assert len(report.statuses) == 60
    assert report.summary_count == 121
    assert report.dropped_outside_section == 0
    v = resolve(report.statuses, report.summary_count, task_of(DEADLOCK_TASK))
    assert v.score is Score.INVALID
    assert v.detail == "parse-mismatch: 60 ids vs summary 121"


def test_extra_agent_authored_tests_do_not_change_the_verdict() -> None:
    text = fixture("deadlock-trial-0.txt")
    lines = _rust_lines(text)
    hdr = next(i for i, ln in enumerate(lines) if "short test summary info" in ln)
    mutated_lines = list(lines)
    mutated_lines.insert(hdr + 1, "PASSED tests/test_agent_authored.py::test_added")
    mutated = "\n".join(mutated_lines).replace("2 failed, 119 passed", "2 failed, 120 passed", 1)
    report = parse_pytest(mutated)
    assert len(report.statuses) == 122
    assert report.summary_count == 122
    task = task_of(DEADLOCK_TASK)
    original = parse_pytest(text)
    v1 = resolve(original.statuses, original.summary_count, task)
    v2 = resolve(report.statuses, report.summary_count, task)
    assert v2.score is Score.UNRESOLVED
    assert v1.score is Score.UNRESOLVED
    assert v1.missing_fail_to_pass == v2.missing_fail_to_pass


# ---- exit_code_fault_reason ----------------------------------------------------------


def test_exit_code_fault_reason_maps_every_arm() -> None:
    assert exit_code_fault_reason(124, "") == "gate-timeout: exit 124"
    assert exit_code_fault_reason(124, "PASSED x") == "gate-timeout: exit 124"
    assert exit_code_fault_reason(137, "") == "gate-no-exit-code: exit 137"
    assert exit_code_fault_reason(137, "  \n") == "gate-no-exit-code: exit 137"
    assert exit_code_fault_reason(137, "PASSED tests/t.py::a") is None
    assert exit_code_fault_reason(1, "") is None
    assert exit_code_fault_reason(0, "") is None
    assert exit_code_fault_reason(None, "") is None


# ---- module shape ----------------------------------------------------------------------


def test_mined_score_imports_are_stdlib_only() -> None:
    src = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top in sys.stdlib_module_names, top
                assert top not in {"talos_harbor", "harbor"}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert node.module is not None
            top = node.module.split(".")[0]
            assert top in sys.stdlib_module_names, top
            assert top not in {"talos_harbor", "harbor"}
    ast.parse(src, feature_version=(3, 12))
    doc = ast.get_docstring(tree)
    assert doc is not None
    assert "crates/harness/src/mined_eval.rs" in doc
    assert "BOTH" in doc
    assert src.endswith('if __name__ == "__main__":\n    sys.exit(main())\n')
    assert "type: ignore" not in src


def test_test_status_has_exactly_six_members() -> None:
    assert [m.name for m in TestStatus] == [
        "PASSED",
        "FAILED",
        "ERROR",
        "SKIPPED",
        "XFAIL",
        "XPASS",
    ]


# ---- CLI end-to-end ------------------------------------------------------------------------

RESOLVED_STDOUT = (
    "=========================== short test summary info ============================\n"
    f"PASSED {OWNER}\n"
    f"PASSED {READER}\n"
    "============================== 2 passed in 0.01s ===============================\n"
)

SCORE_KEYS = {
    "score",
    "detail",
    "passed",
    "failed",
    "missing_fail_to_pass",
    "uncovered_red",
    "agent_tests_added",
    "agent_tests_modified",
    "collection_errors",
    "fail_to_pass_status",
    "positive_control_status",
    "excluded_red",
    "summary_count",
    "sections_seen",
    "dropped_outside_section",
    "count_check",
    "authored_error",
    "task_id",
    "gate_exit_code",
    "gate_output_bytes",
    "scorer_sha256",
}


class Run:
    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.task = tmp / "task_score.json"
        self.gate = tmp / "gate-output.txt"
        self.authored = tmp / "authored.json"
        self.out = tmp / "out"
        self.task.write_text(json.dumps(SAMPLE_TASK))
        self.gate.write_text(RESOLVED_STDOUT)
        self.authored.write_text(json.dumps({"added": [], "modified": []}))

    def argv(self, *extra: str) -> list[str]:
        return [
            "--task",
            str(self.task),
            "--gate-output",
            str(self.gate),
            "--authored",
            str(self.authored),
            "--out-dir",
            str(self.out),
            *extra,
        ]

    def main(self, *extra: str) -> dict[str, Any]:
        assert mined_score.main(self.argv(*extra)) == 0
        score: dict[str, Any] = json.loads((self.out / "score.json").read_text())
        assert set(score) == SCORE_KEYS
        reward = (self.out / "reward.txt").read_bytes()
        assert reward == (b"1" if score["score"] == "resolved" else b"0")
        assert not (self.out / "reward.json").exists()
        return score

    def reward(self) -> bytes:
        return (self.out / "reward.txt").read_bytes()


@pytest.fixture
def run(tmp_path: Path) -> Run:
    return Run(tmp_path)


PRE_PARSE = {
    "passed": [],
    "failed": [],
    "collection_errors": [],
    "missing_fail_to_pass": [],
    "uncovered_red": [],
    "fail_to_pass_status": [],
    "positive_control_status": [],
    "excluded_red": [],
    "summary_count": None,
    "sections_seen": 0,
    "dropped_outside_section": 0,
    "count_check": "n/a",
}


def assert_pre_parse(score: dict[str, Any]) -> None:
    for k, v in PRE_PARSE.items():
        assert score[k] == v, k


def test_cli_resolved_synthetic(run: Run, capsys: pytest.CaptureFixture[str]) -> None:
    score = run.main()
    out = capsys.readouterr().out
    assert out == (
        "mined_score: task=cleanr-abcdef1 score=resolved detail= passed=2 failed=0 "
        "count_check=on sections=1\n"
    )
    assert run.reward() == b"1"
    assert score["score"] == "resolved"
    assert score["detail"] == ""
    assert score["passed"] == [OWNER, READER]
    assert score["task_id"] == "cleanr-abcdef1"
    assert score["gate_exit_code"] is None
    assert score["gate_output_bytes"] == len(RESOLVED_STDOUT.encode())
    assert score["authored_error"] is None
    assert score["fail_to_pass_status"] == [
        {"id": "TestCommentFilters::test_owner_comments_skipped", "status": "PASSED"}
    ]
    assert score["positive_control_status"] == [
        {"id": "TestCommentFilters::test_reader_comments_kept", "status": "PASSED"}
    ]
    raw = (run.out / "score.json").read_text()
    assert raw == json.dumps(score, indent=2, sort_keys=True) + "\n"


def test_cli_scorer_sha256(run: Run) -> None:
    score = run.main()
    assert score["scorer_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()


def test_cli_deadlock_capture(run: Run) -> None:
    run.task.write_text(json.dumps(DEADLOCK_TASK))
    shutil.copyfile(FIXTURES / "deadlock-trial-0.txt", run.gate)
    score = run.main()
    assert run.reward() == b"0"
    assert score["score"] == "unresolved"
    assert score["detail"] == "fail-to-pass-red"
    assert score["count_check"] == "on"
    assert score["sections_seen"] == 1
    assert score["dropped_outside_section"] == 6
    statuses = {e["id"]: e["status"] for e in score["fail_to_pass_status"]}
    for missing in score["missing_fail_to_pass"]:
        assert statuses[missing] is not None
        assert statuses[missing] != "PASSED"
    assert len(score["missing_fail_to_pass"]) == 2
    assert len(score["failed"]) == 2
    assert len(score["passed"]) == 119


def test_cli_from_json_collection_error(run: Run) -> None:
    run.task.write_text(json.dumps(FROM_JSON_TASK))
    shutil.copyfile(FIXTURES / "from-json-trial-1.txt", run.gate)
    score = run.main()
    assert score["score"] == "unresolved"
    assert score["detail"] == "collection-error"
    assert score["collection_errors"] == ["tests/test_cli.py"]
    assert score["failed"] == ["tests/test_cli.py"]


def test_cli_empty_gate_output_is_parse_empty(run: Run) -> None:
    run.gate.write_bytes(b"")
    score = run.main()
    assert run.reward() == b"0"
    assert score["score"] == "invalid"
    assert score["detail"] == "parse-empty"
    assert score["count_check"] == "n/a"
    assert score["gate_output_bytes"] == 0


def test_cli_missing_gate_output(run: Run, capsys: pytest.CaptureFixture[str]) -> None:
    run.gate.unlink()
    score = run.main()
    assert score["score"] == "invalid"
    assert score["detail"].startswith("gate-output-missing: ")
    assert score["gate_output_bytes"] is None
    assert score["task_id"] == "cleanr-abcdef1"
    assert_pre_parse(score)
    assert capsys.readouterr().out.startswith("mined_score: task=cleanr-abcdef1 score=invalid ")


def test_cli_unreadable_gate_stderr(run: Run) -> None:
    score = run.main("--gate-stderr", str(run.tmp / "nope.txt"))
    assert score["detail"].startswith("gate-output-missing: ")
    assert score["gate_output_bytes"] is None


def test_cli_malformed_task(run: Run, capsys: pytest.CaptureFixture[str]) -> None:
    run.task.write_text("{not json")
    score = run.main()
    assert score["score"] == "invalid"
    assert score["detail"].startswith("task-score-unreadable: ")
    assert score["task_id"] is None
    assert_pre_parse(score)
    assert capsys.readouterr().out.startswith("mined_score: task=- score=invalid ")


def test_cli_malformed_task_and_missing_gate_output(run: Run) -> None:
    run.task.write_text("[]")
    run.gate.unlink()
    score = run.main()
    assert score["detail"].startswith("task-score-unreadable: ")


@pytest.mark.parametrize(
    "key", ["id", "fail_to_pass", "positive_controls", "pass_to_pass_exclusions"]
)
def test_cli_task_missing_required_key(run: Run, key: str) -> None:
    task = dict(SAMPLE_TASK)
    del task[key]
    run.task.write_text(json.dumps(task))
    score = run.main()
    assert score["score"] == "invalid"
    assert score["detail"].startswith("task-score-unreadable: ")


@pytest.mark.parametrize(
    "patch",
    [
        {"fail_to_pass": "TestCommentFilters::test_owner_comments_skipped"},
        {"id": 7},
        {"positive_controls": [1]},
        {"gate_command": ["x"]},
        {"sealed": "tests/test_cleanr.py"},
        {"sealed": [{"path": 3}]},
        {"sealed": ["tests/test_cleanr.py"]},
    ],
)
def test_cli_task_wrong_types(run: Run, patch: dict[str, Any]) -> None:
    run.task.write_text(json.dumps({**SAMPLE_TASK, **patch}))
    score = run.main()
    assert score["score"] == "invalid"
    assert score["detail"].startswith("task-score-unreadable: ")


def test_cli_task_missing_file(run: Run) -> None:
    run.task.unlink()
    score = run.main()
    assert score["detail"].startswith("task-score-unreadable: ")


def test_cli_task_without_optional_keys_still_scores(run: Run) -> None:
    task = {k: v for k, v in SAMPLE_TASK.items() if k not in {"sealed", "gate_command"}}
    task["unknown_extra"] = {"ignored": True}
    run.task.write_text(json.dumps(task))
    score = run.main()
    assert score["score"] == "resolved"


def test_load_task_score_reads_optional_keys(tmp_path: Path) -> None:
    p = tmp_path / "t.json"
    p.write_text(json.dumps(SAMPLE_TASK))
    assert mined_score.load_task_score(str(p)) == sample_task()


def test_cli_gate_timeout(run: Run) -> None:
    score = run.main("--gate-exit-code", "124")
    assert score["score"] == "invalid"
    assert score["detail"] == "gate-timeout: exit 124"
    assert score["gate_exit_code"] == 124
    assert score["gate_output_bytes"] == len(RESOLVED_STDOUT.encode())
    assert_pre_parse(score)


def test_cli_gate_exit_code_one_still_scores(run: Run) -> None:
    score = run.main("--gate-exit-code", "1")
    assert score["score"] == "resolved"
    assert score["gate_exit_code"] == 1


def test_cli_xpass_fail_to_pass_is_red(run: Run) -> None:
    run.gate.write_text(
        "=========================== short test summary info ============================\n"
        f"XPASS {OWNER}\n"
        f"PASSED {READER}\n"
        "======================= 1 passed, 1 xpassed in 0.01s =======================\n"
    )
    score = run.main()
    assert score["score"] == "unresolved"
    assert score["detail"] == "fail-to-pass-red"
    assert OWNER not in score["passed"]
    assert OWNER not in score["failed"]
    assert score["fail_to_pass_status"] == [
        {"id": "TestCommentFilters::test_owner_comments_skipped", "status": "XPASS"}
    ]


def test_cli_authored_echo_and_missing_do_not_change_verdict(run: Run) -> None:
    run.authored.write_text(json.dumps({"added": ["tests/test_new.py"], "modified": []}))
    first = run.main()
    assert first["agent_tests_added"] == ["tests/test_new.py"]
    assert first["agent_tests_modified"] == []
    assert first["authored_error"] is None
    reward_first = run.reward()
    run.authored.unlink()
    second = run.main()
    assert second["authored_error"].startswith("authored-missing:")
    assert second["agent_tests_added"] == []
    assert (first["score"], first["detail"]) == (second["score"], second["detail"])
    assert reward_first == run.reward()


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("{nope", "authored-unparseable: "),
        ("[]", "authored-wrong-shape"),
        ('{"added": []}', "authored-wrong-shape"),
        ('{"added": [1], "modified": []}', "authored-wrong-shape"),
        ('{"added": "x", "modified": []}', "authored-wrong-shape"),
    ],
)
def test_cli_authored_errors(run: Run, body: str, expected: str) -> None:
    run.authored.write_text(body)
    score = run.main()
    assert score["authored_error"].startswith(expected)
    assert score["agent_tests_added"] == []
    assert score["agent_tests_modified"] == []
    assert score["score"] == "resolved"


def test_cli_authored_kept_on_invalid_paths(run: Run) -> None:
    run.authored.write_text(json.dumps({"added": [], "modified": ["tests/test_old.py"]}))
    run.task.write_text("nope")
    score = run.main()
    assert score["agent_tests_modified"] == ["tests/test_old.py"]


def test_cli_quiet_no_banner_count_check_off(run: Run) -> None:
    run.gate.write_text(QUIET_NO_BANNER)
    score = run.main()
    assert score["count_check"] == "off"
    assert score["summary_count"] is None
    assert score["sections_seen"] == 1


def test_cli_invalid_utf8_byte_in_non_status_line(run: Run) -> None:
    data = b"collected 2 items \xff garbage\n" + RESOLVED_STDOUT.encode()
    run.gate.write_bytes(data)
    score = run.main()
    assert score["score"] == "resolved"
    assert score["gate_output_bytes"] == len(data)
    assert score["gate_output_bytes"] == run.gate.stat().st_size


def test_cli_scorer_exception(
    run: Run, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(*_args: object) -> mined_score.Verdict:
        raise RuntimeError("boom")

    monkeypatch.setattr(mined_score, "resolve", boom)
    score = run.main()
    assert run.reward() == b"0"
    assert score["score"] == "invalid"
    assert score["detail"] == "scorer-exception: RuntimeError: boom"
    assert score["task_id"] == "cleanr-abcdef1"
    assert_pre_parse(score)
    captured = capsys.readouterr()
    assert "Traceback" in captured.err
    assert captured.out.count("\n") == 1


def test_cli_missing_required_flag_exits_2(tmp_path: Path) -> None:
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        mined_score.main(["--task", "t", "--gate-output", "g", "--out-dir", str(out)])
    assert exc.value.code == 2
    assert not out.exists()


def test_cli_non_integer_exit_code_exits_2(run: Run) -> None:
    with pytest.raises(SystemExit) as exc:
        mined_score.main(run.argv("--gate-exit-code", "abc"))
    assert exc.value.code == 2
    assert not run.out.exists()


def test_stderr_interleaved_mid_section_changes_verdict(run: Run) -> None:
    stderr = run.tmp / "stderr.txt"
    stderr.write_text("warning: x\n")
    score = run.main("--gate-stderr", str(stderr))
    assert score["score"] == "resolved"
    assert score["gate_output_bytes"] == len(RESOLVED_STDOUT.encode()) + len(b"warning: x\n")
    interleaved = RESOLVED_STDOUT.replace(f"PASSED {READER}\n", f"warning: x\nPASSED {READER}\n")
    run.gate.write_text(interleaved)
    score = run.main()
    assert score["score"] != "resolved"
    assert score["detail"] == "parse-mismatch: 1 ids vs summary 2"


# ---- subprocess: standalone ------------------------------------------------------------------


def test_script_runs_standalone_isolated(tmp_path: Path) -> None:
    lone = tmp_path / "lone"
    lone.mkdir()
    shutil.copyfile(SCRIPT, lone / "mined_score.py")
    r = Run(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-I", str(lone / "mined_score.py"), *r.argv()],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (r.out / "reward.txt").read_bytes() == b"1"
    assert (r.out / "score.json").exists()
    assert proc.stdout.startswith("mined_score: task=cleanr-abcdef1 score=resolved ")


def test_authored_one_liner_builds_json(tmp_path: Path) -> None:
    lone = tmp_path / "lone"
    lone.mkdir()
    shutil.copyfile(SCRIPT, lone / "mined_score.py")
    code = (
        "import sys, json; "
        f"sys.path.insert(0, {str(lone)!r}); "
        "import mined_score; "
        "a, m = mined_score.parse_authored_tests(sys.stdin.read()); "
        "print(json.dumps({'added': a, 'modified': m}))"
    )
    proc = subprocess.run(
        ["python3", "-I", "-c", code],
        input="?? tests/test_new.py\n M tests/test_old.py\n",
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(proc.stdout) == {
        "added": ["tests/test_new.py"],
        "modified": ["tests/test_old.py"],
    }


def test_rust_ws_is_unicode_white_space() -> None:
    # Rust char::is_whitespace = Python isspace() minus the \x1c-\x1f separators.
    expected = {chr(c) for c in range(0x110000) if chr(c).isspace()} - {
        "\x1c",
        "\x1d",
        "\x1e",
        "\x1f",
    }
    assert set(mined_score._RUST_WS) == expected
    assert len(mined_score._RUST_WS) == 25
