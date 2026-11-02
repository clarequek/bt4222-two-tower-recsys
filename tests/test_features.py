import polars as pl
import pytest

from jds import baselines, features
from jds.config import paths
from jds.prepare import load


@pytest.fixture(scope="module")
def feats(tiny_cfg):
    names = features.build(tiny_cfg, force=True)
    return names, features.read(paths(tiny_cfg))


def test_feature_table_complete(tiny_cfg, feats):
    names, df = feats
    c = load(paths(tiny_cfg).data, "candidates")
    assert df.height == c.height
    assert df.select(names).null_count().sum_horizontal().item() == 0
    assert not any(n in names for n in ("label", "pos", "orig_pos", "split", "fold"))


def test_windowed_counts(tiny_cfg, feats):
    _, df = feats
    h = load(paths(tiny_cfg).data, "history")
    u1 = h.filter(pl.col("user_id") == 1)
    clicked = u1.filter(pl.col("type") == 1)["wid"].to_list()
    rows = df.filter((pl.col("user_id") == 1) & pl.col("wid").is_in(clicked))
    if rows.height:
        assert (rows["u_wid_click_7"] >= 1).all()
    ordered = u1.filter(pl.col("type") == 4)["wid"].item()        # 200 days old: outside order_90, inside order_365
    r = df.filter((pl.col("user_id") == 1) & (pl.col("wid") == ordered))
    if r.height:
        assert r["u_wid_order_90"].item() == 0 and r["u_wid_order_365"].item() == 1


def test_ctr_is_out_of_fold(tiny_cfg):
    """A training user's own labels must not change its CTR features."""
    p = paths(tiny_cfg)
    users = load(p.data, "users")
    attrs = load(p.data, "products", ["wid", *features.ATTR_KEYS])
    lab = (load(p.data, "candidates", ["user_id", "wid", "label"]).join(users.select("user_id", "qid"), on="user_id")
           .join(attrs, on="wid", how="left"))
    tables, g = features.oof_ctr_tables(lab, users, tiny_cfg)
    u = users.filter((pl.col("split") == "train") & pl.col("has_pos")).row(0, named=True)
    rows = lab.filter(pl.col("user_id") == u["user_id"]).with_columns(fold=pl.lit(u["fold"], pl.Int8))
    base = features.join_ctr(rows, tables, g).sort("wid")
    flipped = lab.with_columns(label=pl.when(pl.col("user_id") == u["user_id"]).then(3 - pl.col("label")).otherwise(pl.col("label")))
    tables2, _ = features.oof_ctr_tables(flipped, users, tiny_cfg)
    after = features.join_ctr(rows, tables2, g).sort("wid")
    cols = [c for c in base.columns if c.startswith(("ctr_", "gain_", "disp_"))]
    assert base.select(cols).equals(after.select(cols))


def test_baselines_report_train_and_test(tiny_cfg, feats):
    result = baselines.run(tiny_cfg)
    models = result["model"].to_list()
    for name in baselines.SINGLE:
        assert any(m.startswith(name + " [train,") for m in models)
        assert name + " [test]" in models
