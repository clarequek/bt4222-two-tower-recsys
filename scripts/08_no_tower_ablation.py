"""Matched no-tower ablation, paired bootstrap intervals and an attention breakdown for the final report.

Run from the repository root after the full pipeline (scripts/run_all.sh) has finished:

    python scripts/08_no_tower_ablation.py                 # no-tower ranker + bootstrap + attention
    python scripts/08_no_tower_ablation.py --window-ci     # also refit the windowed / naive rankers for a CI

What it does
1. Fits a LambdaRank ranker with exactly the final ranker's settings and training users (train_user_frac from the
   config, normally 100%), dropping only the three tower features. This is the matched comparison for the final
   model's test NDCG@10 of 0.4596.
2. Scores it on the 17,174 test users and on the same 25% training sample used in Table 4.
3. Paired bootstrap over test users (1,000 resamples) for the final model vs no-tower, the final model vs
   popularity and the no-tower ranker vs popularity. With --window-ci it also refits the windowed and naive
   window-ablation rankers (30% of training users) and bootstraps their difference.
4. Segment metrics of the no-tower ranker beside the final ranker, tower and popularity (same segments as Figure 8).
5. Attention of the tower by event type and by recency within the 7-day window.

Everything is written to outputs/<scope>/results/no_tower/. Nothing existing is overwritten.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

from jds import features  # noqa: E402
from jds.config import cli, get_logger, load_json, paths, save_json  # noqa: E402
from jds.metrics import evaluate, per_user_metrics, results_table  # noqa: E402
from jds.ranker import dataset  # noqa: E402
from jds.ranker import infer as rinfer  # noqa: E402
from jds.ranker import train as rtrain  # noqa: E402

METRICS = ["mrr", "p1", "ndcg10", "gndcg10"]
TYPES = {1: "click", 2: "follow", 3: "cart", 4: "order"}


def bootstrap(a: pl.DataFrame, b: pl.DataFrame, n_boot: int, seed: int) -> dict:
    """Paired bootstrap over users of mean(a) - mean(b) for each metric; a and b are per-user metric tables."""
    j = a.select("user_id", *METRICS).join(b.select("user_id", *METRICS), on="user_id", suffix="_b")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, j.height, size=(n_boot, j.height))
    out = {"n_users": j.height}
    for m in METRICS:
        d = (j[m] - j[f"{m}_b"]).to_numpy()
        boots = d[idx].mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[m] = {"diff": float(d.mean()), "ci95": [float(lo), float(hi)], "share_boot_le_0": float((boots <= 0).mean())}
    return out


def attention_breakdown(cfg, p, log) -> dict:
    """Mean attention per event type and recency rank in the tower's 7-day window, averaged over real candidates."""
    from jds.config import get_device
    from jds.twotower.train import ItemFeats, load_assets, load_model, make_stores, to_dev
    device = get_device(cfg)
    a = load_assets(p)
    st, users = make_stores(cfg, p, a)
    model = load_model(p.tower / "full.pt", a, device)
    model.eval()
    feats = ItemFeats(a["X"], device, cfg["tower"]["item_matrix_on_gpu"] and device == "cuda")
    ids = users.filter(pl.col("split") == "test")["user_id"].to_numpy()[: cfg["diagnostics"]["attention_users"]]
    rows = []
    with torch.no_grad():
        for i in range(0, len(ids), cfg["tower"]["eval_batch_users"]):
            ub = ids[i:i + cfg["tower"]["eval_batch_users"]]
            b, items, _ = st.slate_batch(ub)
            typ = np.asarray(b["ctx"]["short"]["type"])            # [B, L], right-aligned (most recent last)
            mask = np.asarray(b["ctx"]["short"]["mask"])
            b = to_dev(b, device)
            it = torch.as_tensor(items, device=device)
            r = feats(it)
            model.user_vec(b["q"], b["ctx"], feats, it, cand_rows=r, cand_v=model.item(r))
            att = model.user.attn.last_attn.float().cpu().numpy()  # [B, S, L+1], slot 0 = null
            keep = items > 0
            ev = (att[..., 1:] * keep[..., None]).sum(1) / keep.sum(1).clip(1)[:, None]   # [B, L]
            L = typ.shape[1]
            n_ev = mask.sum(1)
            for u in range(len(ub)):
                if n_ev[u] == 0:
                    continue
                pos = np.flatnonzero(mask[u])
                rank = (pos.max() - pos) + 1                       # 1 = most recent event
                for k, j in enumerate(pos):
                    rows.append((int(ub[u]), int(typ[u, j]), int(rank[k]), float(ev[u, j]), int(n_ev[u])))
    df = pl.DataFrame(rows, schema=["user_id", "type", "recency_rank", "attn", "n_events"], orient="row")
    per_user = df.group_by("user_id").agg(total=pl.col("attn").sum())
    by_type = (df.group_by("type").agg(events=pl.len(), attn_mass=pl.col("attn").sum(), mean_attn_per_event=pl.col("attn").mean())
               .with_columns(share_of_events=pl.col("events") / pl.col("events").sum(),
                             share_of_attention=pl.col("attn_mass") / pl.col("attn_mass").sum(),
                             type=pl.col("type").replace_strict(TYPES, default="other", return_dtype=pl.String))
               .sort("events", descending=True))
    rb = pl.when(pl.col("recency_rank") <= 1).then(pl.lit("1 (most recent)")).when(pl.col("recency_rank") <= 5) \
        .then(pl.lit("2-5")).when(pl.col("recency_rank") <= 20).then(pl.lit("6-20")).otherwise(pl.lit("21-50"))
    by_rec = (df.with_columns(bucket=rb).group_by("bucket")
              .agg(events=pl.len(), attn_mass=pl.col("attn").sum(), mean_attn_per_event=pl.col("attn").mean())
              .with_columns(share_of_events=pl.col("events") / pl.col("events").sum(),
                            share_of_attention=pl.col("attn_mass") / pl.col("attn_mass").sum()).sort("bucket"))
    by_type.write_csv(p.results / "no_tower" / "attention_by_type.csv")
    by_rec.write_csv(p.results / "no_tower" / "attention_by_recency.csv")
    out = {"users": int(per_user.height), "mean_event_attention": float(per_user["total"].mean())}
    log.info(f"attention by type:\n{by_type}\nattention by recency:\n{by_rec}\n{out}")
    return out


def window_ci(cfg, p, log, n_boot):
    from jds.window_ablation import _load, build, variants
    names = load_json(p.features / "feature_list.json")
    naive = build(cfg, False)
    frac = cfg["diagnostics"]["ablation_user_frac"]
    tr, va = _load(p, ["train"], cfg, frac), _load(p, ["val"], cfg, 1.0)
    te = _load(p, ["test"], cfg, 1.0, only_with_pos=False)
    v = variants(names, naive, cfg)
    pu = {}
    for name in ("windowed (pipeline)", "naive (all-history counts, decay, ratios)"):
        bst = rtrain.fit_frames(cfg, tr, va, v[name], tag=f"window CI: {name}", log=log)
        pred = te.select("user_id", "wid", "label").with_columns(score=pl.Series(bst.predict(te.select(v[name]).to_numpy())).cast(pl.Float32))
        pu[name] = per_user_metrics(pred, "score", seed=cfg["seed"])
        pu[name].write_parquet(p.results / "no_tower" / f"per_user_window_{name.split()[0]}.parquet")
    return bootstrap(pu["windowed (pipeline)"], pu["naive (all-history counts, decay, ratios)"], n_boot, cfg["seed"])


def main():
    cfg, args = cli(__doc__, extra=lambda ap: (ap.add_argument("--window-ci", action="store_true"),
                                               ap.add_argument("--n-boot", type=int, default=1000),
                                               ap.add_argument("--skip-attention", action="store_true")))
    p = paths(cfg)
    out_dir = p.results / "no_tower"
    out_dir.mkdir(exist_ok=True)
    log = get_logger("no_tower_ablation", p)
    t0 = time.time()

    # Attention first: PyTorch before LightGBM avoids two OpenMP runtimes fighting on macOS.
    att = None
    if not args.skip_attention:
        if sys.platform == "darwin":
            torch.set_num_threads(1)
        att = attention_breakdown(cfg, p, log)

    # 1. matched no-tower ranker: same users, settings and early stopping as the final ranker; only tower features dropped
    final_names = load_json(p.ranker / "ranker_meta.json")["features"]
    names = [n for n in final_names if n not in dataset.TOWER_FEATURES]
    assert len(final_names) - len(names) == len(dataset.TOWER_FEATURES), "final ranker does not carry the 3 tower features"
    f = out_dir / "ranker_no_tower.txt"
    if f.exists() and not args.force:
        import lightgbm as lgb
        bst = lgb.Booster(model_file=str(f))
        best_it = load_json(out_dir / "meta.json")["best_iteration"]
    else:
        bst = rtrain.fit(cfg, p, names, use_tower=False, tag="no-tower (matched)", log=log)
        best_it = bst.best_iteration
        bst.save_model(str(f), num_iteration=best_it)
        save_json({"best_iteration": best_it, "features": names}, out_dir / "meta.json")

    rows = []
    te = rinfer.predict(cfg, p, bst, names, "test", use_tower=False)
    rows.append(evaluate(te, "score", seed=cfg["seed"], name="no-tower LambdaRank (matched) [test]"))
    pu_nt = per_user_metrics(te, "score", seed=cfg["seed"])
    pu_nt.write_parquet(out_dir / "per_user_no_tower_test.parquet")
    frac = cfg["ranker"].get("train_eval_user_frac", 0.25)
    tr = rinfer.predict(cfg, p, bst, names, "train", use_tower=False, user_frac=frac)
    rows.insert(0, evaluate(tr, "score", seed=cfg["seed"], name=f"no-tower LambdaRank (matched) [train, {frac:.0%} of users]"))
    del tr
    va = rinfer.predict(cfg, p, bst, names, "val", use_tower=False)
    rows.insert(1, evaluate(va, "score", seed=cfg["seed"], name="no-tower LambdaRank (matched) [val]"))
    del va
    save_json(rows, out_dir / "no_tower.json")
    results_table(rows).write_csv(out_dir / "no_tower_summary.csv")

    # 2. paired bootstrap over test users
    pu_final = pl.read_parquet(p.results / "per_user_ranker_test.parquet")
    pop = features.read(p, columns=["user_id", "label", "ipop_users"], splits=["test"])
    pu_pop = per_user_metrics(pop, "ipop_users", seed=cfg["seed"])
    ts = pl.read_parquet(p.tower / "tower_scores.parquet").filter(pl.col("split") == "test")
    pu_tower = per_user_metrics(ts, "tower_score", seed=cfg["seed"])
    boot = {"final_minus_no_tower": bootstrap(pu_final, pu_nt, args.n_boot, cfg["seed"]),
            "final_minus_popularity": bootstrap(pu_final, pu_pop, args.n_boot, cfg["seed"]),
            "no_tower_minus_popularity": bootstrap(pu_nt, pu_pop, args.n_boot, cfg["seed"])}
    if args.window_ci:
        boot["windowed_minus_naive_30pct"] = window_ci(cfg, p, log, args.n_boot)
    save_json(boot, out_dir / "bootstrap.json")

    # 3. segments: same dimensions as Figure 8
    from jds.diagnostics import SEG_DIMS, segment_table
    seg = segment_table(cfg, p)
    models = {"final": pu_final, "no-tower": pu_nt, "two-tower": pu_tower, "popularity": pu_pop}
    srows = []
    for dim in SEG_DIMS:
        for name, pu in models.items():
            d = pu.join(seg, on="user_id").group_by(dim).agg(n=pl.len(), mrr=pl.col("mrr").mean(), gndcg10=pl.col("gndcg10").mean(),
                                                            ndcg10=pl.col("ndcg10").mean())
            srows.append(d.rename({dim: "segment"}).with_columns(dimension=pl.lit(dim), model=pl.lit(name)))
    segs = pl.concat(srows).select("dimension", "segment", "model", "n", "mrr", "ndcg10", "gndcg10").sort("dimension", "segment", "model")
    segs.write_csv(out_dir / "segments_with_no_tower.csv")

    summary = {"no_tower": rows, "bootstrap": boot, "attention": att, "best_iteration": best_it,
               "n_features": len(names), "minutes": round((time.time() - t0) / 60, 1)}
    save_json(summary, out_dir / "summary.json")
    log.info("\n" + json.dumps(summary, indent=1))
    print(f"\nDONE. Send back the folder {out_dir} (small CSV/JSON files; the .parquet and .txt files are optional).")


if __name__ == "__main__":
    main()
