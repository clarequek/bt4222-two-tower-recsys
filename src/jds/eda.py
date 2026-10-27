"""Stage 1: exploratory analysis that motivates the design -> <scope>/figures/*.png and results/eda.json.

Includes the history-timeline analysis requested in the interim feedback: event types by days before the test query
(retention windows), the click / cart / order funnel by elapsed days (eda_funnel_by_age.csv, funnel_by_age.png),
how much history can never have a recorded click, how the share of recent events drives naive user ratios, and the
structure of the displayed slates (positives per slate, label mix, cold items by label).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from .config import TYPE_NAME, get_logger, paths, save_json  # noqa: E402
from .prepare import load  # noqa: E402
from .windows import CLICK, mix_by_age, window_report  # noqa: E402

COLORS = {1: "#4C72B0", 2: "#8172B2", 3: "#DD8452", 4: "#55A868"}


def fig_timeline(history: pl.DataFrame, path) -> None:
    """Events by days before the test query, per type (log-spaced bins): the three retention windows."""
    bins = np.concatenate([[0], np.logspace(-3, np.log10(700), 60)])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for t in (1, 2, 3, 4):
        a = history.filter(pl.col("type") == t)["age_days"].to_numpy()
        ax[0].hist(a, bins=bins, histtype="step", lw=1.6, color=COLORS[t], label=f"{TYPE_NAME[t]} (n={len(a):,})")
    for d in (7, 90, 365):
        ax[0].axvline(d, color="grey", ls="--", lw=0.8)
        ax[0].text(d, ax[0].get_ylim()[1] * 0.9, f" {d} d", color="grey")
    ax[0].set(xscale="symlog", xlabel="days before the test query", ylabel="events", title="History events by age and type")
    ax[0].legend(fontsize=8)
    mix = mix_by_age(history)
    labels = mix["bucket"].cast(pl.String).to_list()
    bottom = np.zeros(mix.height)
    for t in (1, 2, 3, 4):
        v = mix[TYPE_NAME[t].lower()].to_numpy()
        ax[1].bar(labels, v, bottom=bottom, color=COLORS[t], label=TYPE_NAME[t])
        bottom += v
    ax[1].set(xlabel="days before the test query", ylabel="share of events", title="Event-type mix by age bucket")
    ax[1].tick_params(axis="x", rotation=45)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def recency_confound(history: pl.DataFrame, users: pl.DataFrame) -> dict:
    """How much of a naive 'click ratio' is just the share of the user's events inside the last 7 days."""
    u = (history.group_by("user_id").agg(click_ratio=(pl.col("type") == CLICK).mean(),
                                         recent_share=(pl.col("age_days") < 7).mean(),
                                         any_click=(pl.col("type") == CLICK).any()))
    corr = float(np.corrcoef(u["click_ratio"].to_numpy(), u["recent_share"].to_numpy())[0, 1])
    return {"corr_click_ratio_vs_recent_share": corr, "users_without_clicks": float(1 - u["any_click"].mean()),
            "n_users": users.height}


def signal_aucs(cands: pl.DataFrame, history: pl.DataFrame, users: pl.DataFrame, n: int, seed: int) -> dict:
    """Pooled AUC (positive vs not) of item popularity computed per type window, on a user sample."""
    ids = users.filter(pl.col("has_pos"))["user_id"]
    ids = ids.sample(min(n, ids.len()), seed=seed)
    c = cands.join(ids.to_frame(), on="user_id", how="semi")
    pops = history.group_by("wid").agg(all_events=pl.len(),
                                       click_7=((pl.col("type") == 1) & (pl.col("age_days") < 7)).sum(),
                                       cart_90=((pl.col("type") == 3) & (pl.col("age_days") < 90)).sum(),
                                       order_365=(pl.col("type") == 4).sum())
    c = c.join(pops, on="wid", how="left").fill_null(0)
    y = (c["label"] > 0).to_numpy()
    return {k: float(roc_auc_score(y, c[k].to_numpy())) for k in ("all_events", "click_7", "cart_90", "order_365")}


FUNNEL_EDGES = (0, 1, 3, 7, 14, 30, 90, 180, 365, 10_000)


def funnel_by_age(history: pl.DataFrame, edges=FUNNEL_EDGES) -> pl.DataFrame:
    """Interim feedback: elapsed days before the test query vs the click / cart / order funnel.

    The same 0-7-day window has about 6.22 clicks/cart and 6.58 carts/order.
    These are event-count ratios, not user-level conversion rates. After 7 days
    clicks fall almost to zero; after 90 days carts do too.
    """
    labels = [f"{a}-{b}" if b < 10_000 else f"{a}+" for a, b in zip(edges[:-1], edges[1:])]
    h = history.with_columns(bucket=pl.col("age_days").cut(list(edges[1:-1]), labels=labels, left_closed=True))
    return (h.group_by("bucket").agg(
                events=pl.len(), click=(pl.col("type") == 1).sum(), follow=(pl.col("type") == 2).sum(),
                cart=(pl.col("type") == 3).sum(), order=(pl.col("type") == 4).sum(), users=pl.col("user_id").n_unique())
            .sort("bucket")
            .with_columns(clicks_per_cart=pl.col("click") / pl.col("cart").clip(1), carts_per_order=pl.col("cart") / pl.col("order").clip(1),
                          cum_share_of_events=pl.col("events").cum_sum() / pl.col("events").sum())
            .with_columns(pl.col("bucket").cast(pl.String)))


def missing_clicks(history: pl.DataFrame, click_days: float = 7) -> dict:
    """How much of the history can never have a matching click, because it predates the click window."""
    old = pl.col("age_days") >= click_days
    s = history.select(
        carts_older_than_click_window=old.filter(pl.col("type") == 3).mean(),
        orders_older_than_click_window=old.filter(pl.col("type") == 4).mean(),
        events_older_than_click_window=old.mean(),
        clicks_older_than_click_window=old.filter(pl.col("type") == 1).mean()).row(0, named=True)
    u = history.group_by("user_id").agg(c=(pl.col("type") == 1).sum(), o=(pl.col("type") == 4).sum(), k=(pl.col("type") == 3).sum())
    s["users_with_orders_but_no_clicks"] = float(((u["c"] == 0) & (u["o"] > 0)).mean())
    s["overall_clicks_per_cart"] = float(u["c"].sum() / max(u["k"].sum(), 1))
    s["overall_carts_per_order"] = float(u["k"].sum() / max(u["o"].sum(), 1))
    return {k: float(v) if v is not None else None for k, v in s.items()}


def slate_stats(cands: pl.DataFrame, history: pl.DataFrame) -> dict:
    """What the supervised signal (impressions + graded labels) looks like, and how cold the displayed items are."""
    per = cands.group_by("user_id").agg(n=pl.len(), n_pos=(pl.col("label") > 0).sum(), max_label=pl.col("label").max())
    seen_any = history.select("wid").unique().with_columns(seen=pl.lit(True))
    seen_own = history.select("user_id", "wid").unique().with_columns(own=pl.lit(True))
    c = (cands.join(seen_any, on="wid", how="left").join(seen_own, on=["user_id", "wid"], how="left")
         .with_columns(pl.col("seen", "own").fill_null(False)))
    by_label = (c.group_by("label").agg(n=pl.len(), never_in_any_history=(~pl.col("seen")).mean(),
                                        in_users_own_history=pl.col("own").mean()).sort("label"))
    npos = per.with_columns(b=pl.col("n_pos").cut([0, 1, 2, 5], labels=["0", "1", "2", "3-5", "6+"], left_closed=False))
    return {
        "positives_per_slate": dict(npos.group_by("b").len().sort("b").with_columns(pl.col("b").cast(pl.String)).iter_rows()),
        "max_label_share": {str(k): v for k, v in per.group_by("max_label").agg(s=pl.len() / per.height).sort("max_label").iter_rows()},
        "mean_positive_rate_in_slate": float((per["n_pos"] / per["n"]).mean()),
        "by_label": by_label.to_dicts(),
    }


def fig_funnel(funnel: pl.DataFrame, slates: dict, path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    x = funnel["bucket"].to_list()
    ax[0].plot(x, funnel["clicks_per_cart"].to_numpy(), "o-", color=COLORS[1], label="clicks per cart")
    ax[0].plot(x, funnel["carts_per_order"].to_numpy(), "s-", color=COLORS[3], label="carts per order")
    ax[0].axhspan(5, 10, color=COLORS[1], alpha=0.08)
    ax[0].axhspan(3, 5, color=COLORS[3], alpha=0.08)
    ax[0].set(yscale="symlog", xlabel="days before the test query", ylabel="ratio",
              title="Funnel ratios by elapsed days (shaded: typical e-commerce ranges)")
    ax[0].tick_params(axis="x", rotation=45)
    ax[0].legend(fontsize=8)
    bl = pl.DataFrame(slates["by_label"])
    names = ["ignored", "click", "cart", "order"]
    lab = [names[i] for i in bl["label"].to_list()]
    xx = np.arange(len(lab))
    ax[1].bar(xx - 0.2, bl["never_in_any_history"].to_numpy(), 0.4, color="#8C8C8C", label="item never in any history")
    ax[1].bar(xx + 0.2, bl["in_users_own_history"].to_numpy(), 0.4, color=COLORS[4], label="item in the user's own history")
    ax[1].set_xticks(xx, lab)
    ax[1].set(ylabel="share of displayed items", title="Displayed items by label: cold vs previously seen")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(cfg: dict) -> dict:
    p = paths(cfg)
    log = get_logger("eda", p)
    users = load(p.data, "users")
    cands = load(p.data, "candidates", ["user_id", "wid", "label"])
    history = load(p.data, "history", ["user_id", "wid", "qid", "type", "age_days"])
    out = {
        "labels": dict(cands.group_by("label").len().sort("label").iter_rows()),
        "slate_len": cands.group_by("user_id").len()["len"].describe().to_dicts(),
        "history_len": users["hist_len"].describe().to_dicts(),
        "retention": window_report(history).to_dicts(),
        "type_mix_by_age": mix_by_age(history).with_columns(pl.col("bucket").cast(pl.String)).to_dicts(),
        "recency_confound": recency_confound(history, users),
        "missing_clicks": missing_clicks(history, cfg["windows"]["click"]),
        "slates": slate_stats(cands, history),
        "popularity_auc": signal_aucs(cands, history, users, cfg["eda"]["sample_users"], cfg["seed"]),
        "displayed_items_never_in_history": float(1 - cands.select("wid").unique().join(history.select("wid").unique(), on="wid",
                                                                                         how="semi").height
                                                  / cands["wid"].n_unique()),
    }
    fig_timeline(history, p.figures / "history_timeline.png")
    funnel = funnel_by_age(history)
    funnel.write_csv(p.results / "eda_funnel_by_age.csv")
    fig_funnel(funnel, out["slates"], p.figures / "funnel_by_age.png")
    log.info(f"funnel by elapsed days:\n{funnel}")
    log.info(f"missing clicks: {out['missing_clicks']}")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].hist(cands.group_by("user_id").len()["len"].to_numpy(), bins=50, color="#4C72B0")
    ax[0].set(title="Displayed items per test query", xlabel="slate length")
    ax[1].hist(np.log10(1 + users["hist_len"].to_numpy()), bins=50, color="#DD8452")
    ax[1].set(title="History length", xlabel="log10(1 + events)")
    fig.tight_layout()
    fig.savefig(p.figures / "slates_and_history.png", dpi=150)
    plt.close(fig)
    save_json(out, p.results / "eda.json")
    log.info(f"retention windows:\n{window_report(history)}")
    log.info(f"recency confound: {out['recency_confound']}; popularity AUC by window: {out['popularity_auc']}")
    return out
