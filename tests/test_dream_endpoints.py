# ============================================================
# /dreams + /dream-surface endpoint tests
# 夢境管線消費側端點（工單 workorder-dream-pipeline-20260719）
# /dreams：列表含 metadata + 正文；/dream-surface：eligible 時回文本
# 並標 surfaced、ineligible 時回空。
# ============================================================

import json
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from dream_engine import DreamEngine

from tests.test_dream_engine import _dream_config


class DummyEmbeddingEngine:
    enabled = False

    def __init__(self):
        self.deleted = []

    async def get_embedding(self, bucket_id: str):
        return None

    def delete_embedding(self, bucket_id: str):
        self.deleted.append(bucket_id)


def _seed_dream(engine: DreamEngine, dream_id: str, *, hour: int, surfaced: bool,
                valence: float = 0.5, arousal: float = 0.4, body: str = "夢的正文。"):
    generated_at = (
        datetime(2026, 5, 25, hour, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
        .astimezone(ZoneInfo("UTC"))
        .isoformat(timespec="seconds")
    )
    return engine._write_record(
        {
            "dream_id": dream_id,
            "generated_at": generated_at,
            "local_date": "2026-05-25",
            "ai_name": "Haven",
            "dream_model": "deepseek-v4-flash",
            "core_affect": {"valence": valence, "arousal": arousal},
            "recall_cues": ["熟悉空间忽然陌生", "夜里想起未说完的话"],
            "source_bucket_ids": ["a", "b", "c", "d", "e"],
            "identity_anchor_id": "identity-anchor",
            "material_count": 5,
            "surfaced": surfaced,
            "surfaced_at": "2026-05-25T02:00:00+00:00" if surfaced else None,
            "surface_attempts": 0,
        },
        body,
    )


@pytest.mark.asyncio
async def test_dreams_endpoint_lists_all_records_newest_first_with_metadata_and_body(monkeypatch, test_config):
    import server

    engine = DreamEngine(_dream_config(test_config))
    _seed_dream(engine, "dream_old", hour=3, surfaced=True, valence=0.7, arousal=0.6,
                body="旧梦：我在走廊尽头等一扇不肯开的门。")
    _seed_dream(engine, "dream_new", hour=4, surfaced=False, valence=0.3, arousal=0.4,
                body="新梦：右手食指指尖有湿气。")
    monkeypatch.setattr(server, "dream_engine", engine)

    response = await server.dreams_endpoint(SimpleNamespace())
    data = json.loads(response.body)

    # 倒序：新夢在前；未浮現的也在列表裡（面板全看）
    assert [d["dream_id"] for d in data] == ["dream_new", "dream_old"]

    new, old = data
    assert new["surfaced"] is False
    assert new["surfaced_at"] == ""
    assert new["valence"] == 0.3
    assert new["arousal"] == 0.4
    assert new["local_date"] == "2026-05-25"
    assert new["ai_name"] == "Haven"
    assert new["recall_cues"] == ["熟悉空间忽然陌生", "夜里想起未说完的话"]
    assert "右手食指指尖有湿气" in new["body"]

    assert old["surfaced"] is True
    assert old["surfaced_at"] == "2026-05-25T02:00:00+00:00"
    assert "不肯开的门" in old["body"]


@pytest.mark.asyncio
async def test_dreams_endpoint_fails_soft_to_empty_list(monkeypatch):
    import server

    class BrokenEngine:
        identity = {"ai_name": "Haven"}

        def list_records(self):
            raise RuntimeError("disk on fire")

    monkeypatch.setattr(server, "dream_engine", BrokenEngine())

    response = await server.dreams_endpoint(SimpleNamespace())

    assert json.loads(response.body) == []


@pytest.mark.asyncio
async def test_dream_surface_endpoint_returns_text_and_marks_surfaced(monkeypatch, test_config):
    import server

    # valence=-1/arousal=-1 + 空 query → affect/cue 皆 0，唯一浮現通道是
    # spontaneous_surface_prob；設 1 使其確定性命中。
    cfg = _dream_config(test_config, min_surface_age_hours=0, spontaneous_surface_prob=1)
    engine = DreamEngine(cfg)
    record = _seed_dream(engine, "dream_surface_me", hour=3, surfaced=False,
                         body="我走进一条很窄的走廊，右手食指指尖有湿气。")
    embedding = DummyEmbeddingEngine()
    monkeypatch.setattr(server, "dream_engine", engine)
    monkeypatch.setattr(server, "embedding_engine", embedding)

    response = await server.dream_surface_endpoint(SimpleNamespace())
    text = response.body.decode("utf-8")

    assert text.startswith("===== 梦境 =====")
    assert "Haven的梦" in text
    assert "右手食指指尖有湿气" in text
    # 引擎現行為：surfaced 標記落 events 後一次性刪檔（surfaced_one_shot）
    assert not record.path.exists()
    events = engine._read_events()
    assert any(e.get("event") == "surfaced" and e.get("dream_id") == "dream_surface_me" for e in events)


@pytest.mark.asyncio
async def test_dream_surface_endpoint_returns_empty_when_no_resonance(monkeypatch, test_config):
    import server

    cfg = _dream_config(test_config, min_surface_age_hours=0, spontaneous_surface_prob=0)
    engine = DreamEngine(cfg)
    record = _seed_dream(engine, "dream_stays_latent", hour=3, surfaced=False)
    monkeypatch.setattr(server, "dream_engine", engine)
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())

    response = await server.dream_surface_endpoint(SimpleNamespace())

    assert response.body.decode("utf-8") == ""
    # 夢還在、沒被標 surfaced、也沒計入嘗試（top=0 < attempt_threshold）
    kept = engine._read_record(record.path)
    assert kept.surfaced is False
    assert int(kept.metadata.get("surface_attempts", 0)) == 0
    assert not any(e.get("event") == "surfaced" for e in engine._read_events())


@pytest.mark.asyncio
async def test_dream_surface_endpoint_returns_empty_when_no_pending_dream(monkeypatch, test_config):
    import server

    engine = DreamEngine(_dream_config(test_config, min_surface_age_hours=0))
    monkeypatch.setattr(server, "dream_engine", engine)
    monkeypatch.setattr(server, "embedding_engine", DummyEmbeddingEngine())

    response = await server.dream_surface_endpoint(SimpleNamespace())

    assert response.body.decode("utf-8") == ""


@pytest.mark.asyncio
async def test_dream_surface_endpoint_fails_soft_to_empty(monkeypatch):
    import server

    class BrokenEngine:
        async def surface_with_status(self, **kwargs):
            raise RuntimeError("engine down")

    monkeypatch.setattr(server, "dream_engine", BrokenEngine())

    response = await server.dream_surface_endpoint(SimpleNamespace())

    assert response.body.decode("utf-8") == ""
