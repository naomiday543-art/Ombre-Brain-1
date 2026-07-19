"""WO-1 recall-miss telemetry: complaint detection + fail-soft logger.

Covers: pattern hit (simplified & traditional) / miss, enabled=False zero
behavior, and write-failure fail-soft (telemetry must never break recall).
"""

import json

import pytest

from recall_diagnostics import (
    RECALL_COMPLAINT_PATTERNS,
    RecallDiagnosticsLogger,
    is_recall_complaint,
    matched_complaint_terms,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("你忘了我上周说的话", "你忘了"),          # simplified/shared
        ("你怎么不记得我们的约定", "你怎么不记得"),  # simplified
        ("你說過要陪我去海边的", "你說過"),          # traditional
        ("我不是跟你說過了吗", "我不是跟你說過"),     # traditional
    ],
)
def test_complaint_pattern_hits(text, expected):
    terms = matched_complaint_terms(text)
    assert expected in terms
    assert is_recall_complaint(text) is True


def test_complaint_pattern_misses():
    text = "我们今天聊聊那本书吧，你觉得结局怎么样"
    assert matched_complaint_terms(text) == []
    assert is_recall_complaint(text) is False


def test_complaint_detection_is_fail_soft_on_bad_input():
    # Non-string / falsy input must degrade to empty, never raise.
    assert matched_complaint_terms(None) == []
    assert matched_complaint_terms(1234) == []
    assert is_recall_complaint("") is False


def test_pattern_table_carries_both_scripts():
    # Sanity: the constant is append-friendly and covers simp + trad.
    assert "你不记得" in RECALL_COMPLAINT_PATTERNS
    assert "你不記得" in RECALL_COMPLAINT_PATTERNS


def test_disabled_logger_writes_nothing(tmp_path):
    log_path = tmp_path / "state" / "recall_diagnostics.jsonl"
    logger = RecallDiagnosticsLogger(
        {
            "state_dir": str(tmp_path / "state"),
            "recall_diagnostics": {"enabled": False, "path": str(log_path)},
        }
    )
    assert logger.enabled is False
    logger.write({"source": "breath", "recall_failure": True})
    # Zero behavior: no file, no directory side effects.
    assert not log_path.exists()


def test_write_failure_is_swallowed(tmp_path):
    # Point the path under a regular file so os.makedirs raises: the writer
    # must swallow it and return None (recall path stays alive).
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    log_path = blocker / "nested" / "recall_diagnostics.jsonl"
    logger = RecallDiagnosticsLogger(
        {
            "recall_diagnostics": {"enabled": True, "path": str(log_path)},
        }
    )
    assert logger.enabled is True
    # Must not raise despite the un-creatable directory.
    assert logger.write({"source": "breath", "recall_failure": True}) is None
    assert not log_path.exists()


def test_enabled_logger_records_recall_failure_fields(tmp_path):
    log_path = tmp_path / "state" / "recall_diagnostics.jsonl"
    logger = RecallDiagnosticsLogger(
        {
            "state_dir": str(tmp_path / "state"),
            "recall_diagnostics": {"enabled": True, "path": str(log_path)},
        }
    )
    logger.write(
        {
            "source": "breath",
            "recall_failure": True,
            "recall_failure_terms": ["你说过"],
            "candidate_count": 3,
            "selected_candidates": [],
        }
    )
    event = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert event["schema"] == "ombre.recall_diagnostics.v1"
    assert event["recall_failure"] is True
    assert event["recall_failure_terms"] == ["你说过"]
    assert event["candidate_count"] == 3
