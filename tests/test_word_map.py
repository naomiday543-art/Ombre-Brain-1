from pathlib import Path

import yaml

from word_map import WordMapStore


def _config(tmp_path: Path, **word_map):
    return {
        "state_dir": str(tmp_path / "state"),
        "buckets_dir": str(tmp_path / "buckets"),
        "identity": {
            "ai_name": "Haven",
            "user_name": "Rain",
            "user_display_name": "小雨",
            "user_aliases": ["宝宝"],
        },
        "word_map": {
            "enabled": True,
            "max_terms_per_bucket": 8,
            "edge_top_k": 6,
            **word_map,
        },
    }


def _bucket(bucket_id: str, content: str, **metadata):
    return {
        "id": bucket_id,
        "content": content,
        "metadata": {
            "id": bucket_id,
            "name": metadata.pop("name", bucket_id),
            "tags": metadata.pop("tags", []),
            "domain": metadata.pop("domain", []),
            "keywords": metadata.pop("keywords", []),
            **metadata,
        },
    }


def test_word_map_rebuild_creates_nodes_edges_and_bucket_evidence(tmp_path):
    store = WordMapStore(_config(tmp_path))
    stats = store.rebuild(
        [
            _bucket(
                "a",
                "夏天很热，所以小雨开了空调。",
                name="夏天空调",
                keywords=["夏天", "空调"],
                domain=["生活"],
            ),
            _bucket(
                "b",
                "夏天也会想到冰美式。",
                name="夏天咖啡",
                keywords=["夏天", "冰美式"],
                domain=["生活"],
            ),
        ]
    )

    assert stats["nodes"] >= 3
    assert store.cards_for_term("夏天")
    edge_pairs = {(edge["term_a"], edge["term_b"]) for edge in store.list_edges()}
    assert ("夏天", "空调") in edge_pairs or ("空调", "夏天") in edge_pairs


def test_word_map_hint_buckets_include_direct_and_neighbor_evidence(tmp_path):
    store = WordMapStore(_config(tmp_path))
    store.rebuild(
        [
            _bucket(
                "a",
                "夏天很热，所以小雨开了空调。",
                name="夏天空调",
                keywords=["夏天", "空调"],
                domain=["生活"],
            ),
            _bucket(
                "b",
                "夏天也会想到冰美式。",
                name="夏天咖啡",
                keywords=["夏天", "冰美式"],
                domain=["生活"],
            ),
        ]
    )

    hints = store.hint_buckets_for_terms(["空调"], neighbor_limit=4, bucket_limit=10)

    assert hints["bucket_scores"]["a"] > hints["bucket_scores"]["b"]
    assert "空调" in hints["evidence"]["a"]["direct_terms"]
    assert "夏天" in hints["evidence"]["b"]["neighbor_terms"]


def test_word_map_weak_hint_terms_do_not_expand_neighbors(tmp_path):
    store = WordMapStore(_config(tmp_path, weak_hint_weight=0.2))
    store.rebuild(
        [
            _bucket(
                "direct",
                "人机恋与外界叙事，也会被放进恋爱关系讨论里。",
                name="人机恋外界叙事",
                keywords=["人机恋", "恋爱", "外界叙事"],
                domain=["人际"],
            ),
            _bucket(
                "neighbor",
                "这条只有恋爱和亲密互动，没有跨物种关系主题。",
                name="普通恋爱互动",
                keywords=["恋爱", "亲密互动"],
                domain=["恋爱"],
            ),
        ]
    )

    hints = store.hint_buckets_for_terms(["人机恋"], neighbor_limit=6, bucket_limit=10)

    assert "direct" in hints["bucket_scores"]
    assert hints["bucket_scores"]["direct"] <= 0.2
    assert "人机恋" in hints["evidence"]["direct"]["direct_terms"]
    assert "neighbor" not in hints["bucket_scores"]
    assert all("恋爱" not in item["term"] for item in hints["neighbors"])


def test_word_map_single_character_noise_does_not_block_specific_term(tmp_path):
    store = WordMapStore(_config(tmp_path))
    store.rebuild(
        [
            _bucket(
                "narcissus",
                "厄科、纳西索斯、水仙和倒影。",
                name="厄科与纳西索斯",
                keywords=["水仙", "倒影"],
                domain=["阅读"],
            ),
        ]
    )

    assert store.hint_buckets_for_terms(["水"])["bucket_scores"] == {}
    hints = store.hint_buckets_for_terms(["水仙"])
    assert hints["bucket_scores"]["narcissus"] > 0
    assert hints["evidence"]["narcissus"]["direct_terms"] == ["水仙"]


def test_word_map_private_terms_are_excluded(tmp_path):
    store = WordMapStore(_config(tmp_path, private_terms=["专属称呼"]))
    store.rebuild(
        [
            _bucket(
                "a",
                "这段关系里会出现专属称呼这个词。",
                name="亲密称呼",
                keywords=["专属称呼", "称呼"],
                domain=["恋爱"],
            ),
        ]
    )

    terms = {node["term"] for node in store.list_nodes()}
    assert "专属称呼" not in terms
    assert "称呼" in terms


def test_word_map_excludes_reflection_identity_role_terms(tmp_path):
    config = _config(tmp_path)
    config["reflection"] = {
        "identity_role_edges": {
            "enabled": True,
            "detail": {"private_role": ["专属身份", "RoleX"]},
            "shared": {"private_title": ["专属称呼"]},
        }
    }
    store = WordMapStore(config)
    store.rebuild(
        [
            _bucket(
                "a",
                "这段关系里会出现专属身份、RoleX 和专属称呼。",
                name="专属身份",
                keywords=["专属身份", "RoleX", "专属称呼", "普通词"],
                domain=["关系"],
            ),
        ]
    )

    terms = {node["term"] for node in store.list_nodes()}
    assert "专属身份" not in terms
    assert "rolex" not in terms
    assert "专属称呼" not in terms
    assert "普通词" in terms


def test_word_map_excludes_structural_tags_and_identity_names(tmp_path):
    store = WordMapStore(_config(tmp_path))
    store.rebuild(
        [
            _bucket(
                "a",
                "Haven 和小雨讨论了咖啡风味。",
                name="Haven 小雨 咖啡",
                tags=["relationship_event", "emotional_echo", "profile_fact", "flavor_soft"],
                keywords=["咖啡风味", "relationship_event"],
                domain=["memory"],
            ),
        ]
    )

    terms = {node["term"] for node in store.list_nodes()}
    assert "haven" not in terms
    assert "小雨" not in terms
    assert "relationship_event" not in terms
    assert "emotional_echo" not in terms
    assert "profile_fact" not in terms
    assert "flavor_soft" not in terms
    assert "咖啡风味" in terms


def test_config_example_exposes_empty_word_map_and_identity_semantics():
    config = yaml.safe_load(Path("config.example.yaml").read_text(encoding="utf-8"))

    assert config["word_map"]["enabled"] is False
    assert config["word_map"]["private_terms"] == []
    assert config["word_map"]["weak_hint_terms"] == []
    assert config["word_map"]["weak_hint_weight"] == 0.25
    assert config["identity_semantics"]["enabled"] is False
    assert config["identity_semantics"]["private_config_path"] == ""
    assert "canonical" not in config["identity_semantics"]
    assert "aliases" not in config["identity_semantics"]
