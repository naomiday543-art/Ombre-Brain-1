from __future__ import annotations

import json
import logging
import os
from typing import Any

from utils import now_iso

logger = logging.getLogger("ombre_brain.recall_diagnostics")


# User complaint patterns that flag a probable recall miss on the current turn.
# 用户抱怨模式：命中即认为「上一轮/近轮大概率漏撈」。
# 简繁双轨并列；日后要 append 新说法，直接往这个元组加字符串即可（纯子串匹配）。
RECALL_COMPLAINT_PATTERNS: tuple[str, ...] = (
    "你忘了",
    "你忘記了",
    "你不记得",
    "你不記得",
    "你说过",
    "你說過",
    "你怎么不记得",
    "你怎麼不記得",
    "我不是跟你说过",
    "我不是跟你說過",
)


def matched_complaint_terms(text: Any) -> list[str]:
    """Return the complaint patterns present in ``text`` (substring match).

    Fail-soft: any unexpected input coerces to empty — telemetry must never
    raise into the recall path.
    返回命中的抱怨模式；任何异常一律吞掉返回空表，绝不让遥测打掛召回。
    """
    try:
        haystack = text if isinstance(text, str) else str(text or "")
    except Exception:
        return []
    if not haystack:
        return []
    return [pattern for pattern in RECALL_COMPLAINT_PATTERNS if pattern in haystack]


def is_recall_complaint(text: Any) -> bool:
    """True when the user message looks like a recall-miss complaint."""
    return bool(matched_complaint_terms(text))


class RecallDiagnosticsLogger:
    """Append-only JSONL recall diagnostics for admin debugging."""

    def __init__(self, config: dict):
        config = config or {}
        cfg = config.get("recall_diagnostics", {}) or {}
        self.enabled = _bool_value(cfg.get("enabled", False))
        self.max_candidates = _int_between(cfg.get("max_candidates", 20), 20, 1, 100)
        self.max_text_chars = _int_between(cfg.get("max_text_chars", 220), 220, 0, 1000)
        self.path = str(cfg.get("path") or self._default_path(config))

    def write(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            payload = {
                "schema": "ombre.recall_diagnostics.v1",
                "timestamp": now_iso(),
                **(event or {}),
            }
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        except Exception as exc:
            logger.warning("Failed to write recall diagnostics: %s", exc)

    @staticmethod
    def _default_path(config: dict) -> str:
        state_dir = str(config.get("state_dir") or "").strip()
        if not state_dir:
            buckets_dir = str(config.get("buckets_dir") or "").strip()
            state_dir = os.path.join(os.path.dirname(buckets_dir), "state") if buckets_dir else "."
        return os.path.join(state_dir, "recall_diagnostics.jsonl")


def _bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _int_between(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(min_value, min(max_value, number))
