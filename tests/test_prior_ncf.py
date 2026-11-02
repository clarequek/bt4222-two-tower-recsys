from jds import prior_ncf
from jds.config import load_json, paths


def test_prior_ncf_runs_and_reports_gaps(tiny_cfg):
    result = prior_ncf.run(tiny_cfg)
    g = load_json(paths(tiny_cfg).results / "prior_ncf_gaps.json")
    assert g["uses_test_query"] is False
    assert 0 <= g["displayed_items_without_item_embedding (cold, not in history >= min_item_count)"] <= 1
    assert g["train_items_per_list_sampled"] <= g["test_items_per_slate_full"]
    names = result["model"].to_list()
    for variant in ("fixed", "published"):
        assert any(f"({variant}) [train," in name and "in-sample" in name for name in names)
        assert any(f"({variant}) [test]" in name for name in names)
