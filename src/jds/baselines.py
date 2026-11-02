"""Stage 2b: non-learned baselines on training, validation and test slates from the feature table.

Each baseline ranks a slate by one signal; the ensemble mixes the within-slate percentiles of item popularity and
BM25 with a weight picked on validation users. All label-based signals (CTR) come from training users only.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from . import features
from .config import get_logger, paths, save_json
from .metrics import evaluate, results_table

SINGLE = {
    "Random": None,
    "Popularity: users who interacted (all history)": "ipop_users",
    "Popularity: clicks, last 7 days": "ipop_click_7",
    "Popularity: carts, last 90 days": "ipop_cart_90",
    "Popularity: orders, last 365 days": "ipop_order_365",
    "Impression CTR of item (training users)": "ctr_wid",
    "Query-item CTR (training users)": "ctr_q_wid",
    "Query-item history engagement (clicks, 7 days)": "qpop_wid_click_7",
    "BM25 (query vs item text)": "lex_bm25",
    "User affinity: same category clicked, 7 days": "u_cate_id_4_click_7",
}


def _pct(c: str) -> pl.Expr:
    return (pl.col(c).rank("average").over("user_id") - 1) / (pl.len().over("user_id") - 1).clip(1)


def run(cfg: dict) -> pl.DataFrame:
    p = paths(cfg)
    log = get_logger("baselines", p)
    cols = ["user_id", "wid", "label", "split"] + [c for c in SINGLE.values() if c]
    df = features.read(p, columns=cols, splits=["val", "test"])
    df = df.with_columns(rand=pl.lit(0.0), pop_pct=_pct("ipop_click_7"), bm25_pct=_pct("lex_bm25"))
    rows = []
    for split in ("val", "test"):
        d = df.filter(pl.col("split") == split)
        for name, col in SINGLE.items():
            rows.append(evaluate(d, col or "rand", seed=cfg["seed"], name=f"{name} [{split}]"))
    # ensemble weight chosen on validation users only
    val = df.filter(pl.col("split") == "val")
    grid = cfg["baselines"]["ensemble_grid"]
    scores = [evaluate(val.with_columns(s=pl.col("pop_pct") + w * pl.col("bm25_pct")), "s", seed=cfg["seed"])["gndcg10"] for w in grid]
    w = grid[int(np.argmax(scores))]
    log.info(f"ensemble weight on BM25 picked on validation: {w} (gNDCG@10 by weight: {dict(zip(grid, np.round(scores, 4)))})")
    for split in ("val", "test"):
        d = df.filter(pl.col("split") == split).with_columns(s=pl.col("pop_pct") + w * pl.col("bm25_pct"))
        rows.append(evaluate(d, "s", seed=cfg["seed"], name=f"Popularity + {w} x BM25 [{split}]"))
    # Match the ranker's deterministic training-user sample; CTR features are out-of-fold on this split.
    frac = cfg["ranker"].get("train_eval_user_frac", 0.25)
    train = features.read(p, columns=cols, splits=["train"], user_frac=frac, seed=cfg["seed"])
    train = train.with_columns(rand=pl.lit(0.0), pop_pct=_pct("ipop_click_7"), bm25_pct=_pct("lex_bm25"))
    tag = f"train, {frac:.0%} of users"
    for name, col in SINGLE.items():
        rows.append(evaluate(train, col or "rand", seed=cfg["seed"], name=f"{name} [{tag}]"))
    train = train.with_columns(s=pl.col("pop_pct") + w * pl.col("bm25_pct"))
    rows.append(evaluate(train, "s", seed=cfg["seed"], name=f"Popularity + {w} x BM25 [{tag}]"))
    save_json(rows, p.results / "baselines.json")
    tbl = results_table(rows)
    log.info("\n" + str(tbl.filter(pl.col("model").str.ends_with("[test]"))))
    return tbl
