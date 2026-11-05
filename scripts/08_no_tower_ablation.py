"""Stage 8: matched no-tower ablation.

Same training/validation/test users, same hand-crafted features and the same LambdaRank parameters as stage 5, but
the three tower-score columns (`tower_score`, `tower_score_pct`, `tower_score_gap`) are removed. This isolates the
tower's contribution from everything else.

Two no-tower variants are reported:
  * early-stopped: its own best iteration on validation;
  * matched rounds: the same number of trees as the final with-tower ranker, so capacity is held constant.

Outputs: <scope>/results/no_tower_ablation.csv, no_tower_ablation.json,
         <scope>/figures/no_tower_ablation.png, <scope>/ranker/ranker_no_tower.txt

    python scripts/08_no_tower_ablation.py --config configs/default.yaml --set paths.work_dir=outputs_v7
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jds.config import cli, get_logger, load_json, paths, save_json  # noqa: E402
from jds.metrics import evaluate, results_table  # noqa: E402
from jds.ranker import dataset  # noqa: E402


def _fit(cfg: dict, p, names: list[str], log):
    rc = cfg["ranker"]
    tr = dataset.load(p, ["train"], cfg, use_tower=False, only_with_pos=True)
    va = dataset.load(p, ["val"], cfg, use_tower=False, only_with_pos=True)
    log.info(f"[no-tower] train {tr['user_id'].n_unique():,} slates / {tr.height:,} rows, "
             f"val {va['user_id'].n_unique():,} slates, {len(names)} features")
    params = {**rc["params"], "label_gain": rc["label_gain"], "seed": cfg["seed"], "deterministic": True,
              "force_row_wise": True}
    dtr = lgb.Dataset(tr.select(names).to_numpy(), tr["label"].to_numpy(), group=dataset.groups(tr),
                      feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(va.select(names).to_numpy(), va["label"].to_numpy(), group=dataset.groups(va), reference=dtr)
    del tr, va
    booster = lgb.train(params, dtr, num_boost_round=rc["num_boost_round"], valid_sets=[dva],
                        valid_names=["val"],
                        callbacks=[lgb.early_stopping(rc["early_stopping_rounds"], verbose=False),
                                   lgb.log_evaluation(max(1, rc["num_boost_round"] // 10))])
    log.info(f"[no-tower] best iteration {booster.best_iteration}, val {dict(booster.best_score['val'])}")
    return booster


def _predict(cfg: dict, p, booster: lgb.Booster, names: list[str], split: str, num_iteration: int | None,
             user_frac: float = 1.0) -> pl.DataFrame:
    df = dataset.load(p, [split], cfg, use_tower=False, user_frac=user_frac)
    score = booster.predict(df.select(names).to_numpy(), num_threads=0, num_iteration=num_iteration)
    return df.select("user_id", "wid", "label").with_columns(score=pl.Series(score).cast(pl.Float32))


def _rows(cfg: dict, p, booster: lgb.Booster, names: list[str], label: str, num_iteration: int | None) -> list[dict]:
    rows = []
    frac = cfg["ranker"].get("train_eval_user_frac", 0.25)
    for split, frac_ in (("train", frac), ("val", 1.0), ("test", 1.0)):
        pred = _predict(cfg, p, booster, names, split, num_iteration, user_frac=frac_)
        name = f"{label} [{split}" + (f", {frac_:.0%} of users]" if split == "train" else "]")
        rows.append(evaluate(pred, "score", seed=cfg["seed"], name=name))
    return rows


def run(cfg: dict) -> None:
    p = paths(cfg)
    log = get_logger("no_tower_ablation", p)
    out = p.results / "no_tower_ablation.json"
    if out.exists():
        log.info(f"{out.name} exists (delete it or run with --force to redo)")
        return
    t0 = time.time()
    names = dataset.feature_names(p, use_tower=False)
    booster = _fit(cfg, p, names, log)
    booster.save_model(str(p.ranker / "ranker_no_tower.txt"), num_iteration=booster.best_iteration)

    with_tower = load_json(p.results / "ranker.json")
    with_tower_test = next(r for r in with_tower if "[test]" in r["model"])
    meta = load_json(p.ranker / "ranker_meta.json")
    matched = min(int(meta["best_iteration"]), booster.current_iteration())

    early_rows = _rows(cfg, p, booster, names, "LambdaRank (no tower, early-stopped)", booster.best_iteration)
    matched_rows = _rows(cfg, p, booster, names, "LambdaRank (no tower, matched rounds)", matched)

    # CSV: test rows for the with-tower reference and both no-tower variants.
    test_rows = [with_tower_test] + [r for r in early_rows + matched_rows if "[test]" in r["model"]]
    table = results_table(test_rows)
    table.write_csv(p.results / "no_tower_ablation.csv")

    payload = {
        "matched_rounds": matched,
        "no_tower_best_iteration": booster.best_iteration,
        "features": len(names),
        "with_tower": with_tower,
        "no_tower_early": early_rows,
        "no_tower_matched": matched_rows,
        "seconds": round(time.time() - t0),
    }
    save_json(payload, out)
    save_json({"features": names, "best_iteration": booster.best_iteration, "matched_rounds": matched},
              p.ranker / "ranker_no_tower_meta.json")

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["with tower", "no tower\n(early)", "no tower\n(matched)"]
    vals = [with_tower_test["mrr"], early_rows[2]["mrr"], matched_rows[2]["mrr"]]
    vals2 = [with_tower_test["ndcg10"], early_rows[2]["ndcg10"], matched_rows[2]["ndcg10"]]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, vals, 0.36, label="MRR")
    ax.bar(x + 0.18, vals2, 0.36, label="NDCG@10")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("test score")
    ax.set_title("Matched no-tower ablation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(p.figures / "no_tower_ablation.png", dpi=150)
    plt.close(fig)
    log.info("\n" + str(table))


if __name__ == "__main__":
    cfg, args = cli(__doc__)
    if args.force:
        (paths(cfg).results / "no_tower_ablation.json").unlink(missing_ok=True)
    run(cfg)
