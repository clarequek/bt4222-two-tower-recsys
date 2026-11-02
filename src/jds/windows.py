"""Retention windows of the JDsearch history log.

The history is not one continuous log: counted back from each user's test query, almost all clicks fall within
7 days, carts and follows within 90 days, and orders within 365 days. The full-file audit finds small exceptions:
107 clicks at least 7 days old and 1,204 carts at least 90 days old. These are observed horizons, not a documented
retention policy. Beyond 7 days, "no click" is mostly a coverage gap, not evidence the user did not click.

Consequences encoded here:
  * a count is only meaningful for an event type inside that type's own window -> `feature_windows`;
  * the history is split into segments whose content differs by construction -> `segment_bounds`.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .config import TYPE_CODE, TYPE_NAME

CLICK, FLW, CART, ORD = TYPE_CODE["CLICK"], TYPE_CODE["FLW"], TYPE_CODE["CART"], TYPE_CODE["ORD"]


def type_window_days(cfg: dict) -> dict[int, float]:
    w = cfg["windows"]
    return {CLICK: w["click"], FLW: w["follow"], CART: w["cart"], ORD: w["order"]}


def feature_windows(cfg: dict) -> list[tuple[str, int, float]]:
    """(name, type code, max age in days) for counts inside each observed history window.

    Shorter windows of the long-lived types (carts and orders in the last 7 days) are comparable with clicks; the
    longer ones carry the extra history that only those types have.
    """
    w = cfg["windows"]
    out = [(f"click_{w['click']}", CLICK, w["click"]),
           (f"cart_{w['click']}", CART, w["click"]), (f"cart_{w['cart']}", CART, w["cart"]),
           (f"order_{w['click']}", ORD, w["click"]), (f"order_{w['cart']}", ORD, w["cart"]), (f"order_{w['order']}", ORD, w["order"]),
           (f"follow_{w['follow']}", FLW, w["follow"])]
    return out


def window_expr(type_code: int, days: float) -> pl.Expr:
    """Boolean expression: event of `type_code` that happened less than `days` before the test query."""
    return (pl.col("type") == type_code) & (pl.col("age_days") < days)


def segment_bounds(h_user: np.ndarray, h_age: np.ndarray, n_users: int, short: float, mid: float):
    """Per-user index boundaries of the three history segments.

    History rows must be sorted by (user, step); age (days before the test query) is non-increasing within a user,
    so each segment is a contiguous block: long = [start, b_long), mid = [b_long, b_mid), short = [b_mid, end).
    """
    start = np.searchsorted(h_user, np.arange(n_users))
    end = np.searchsorted(h_user, np.arange(n_users), side="right")
    n_long = np.bincount(h_user[h_age >= mid], minlength=n_users)
    n_mid = np.bincount(h_user[(h_age >= short) & (h_age < mid)], minlength=n_users)
    b_long = start + n_long
    b_mid = b_long + n_mid
    return start, b_long, b_mid, end


def window_report(history: pl.DataFrame) -> pl.DataFrame:
    """Age quantiles (days before the test query) per event type: the evidence for the retention windows."""
    return (history.group_by("type")
            .agg(n=pl.len(),
                 p50=pl.col("age_days").quantile(0.5), p99=pl.col("age_days").quantile(0.99),
                 p999=pl.col("age_days").quantile(0.999), max=pl.col("age_days").max())
            .sort("type")
            .with_columns(type_name=pl.col("type").replace_strict(TYPE_NAME, return_dtype=pl.String)))


def mix_by_age(history: pl.DataFrame, edges=(0, 1, 3, 7, 14, 30, 60, 90, 180, 365, 10_000)) -> pl.DataFrame:
    """Share of each event type by days-before-query bucket (Figure: history type mix over time)."""
    labels = [f"{a}-{b}" for a, b in zip(edges[:-1], edges[1:])]
    h = history.with_columns(bucket=pl.col("age_days").cut(list(edges[1:-1]), labels=labels, left_closed=True))
    return (h.group_by("bucket").agg(
                n=pl.len(), **{TYPE_NAME[t].lower(): (pl.col("type") == t).mean() for t in (CLICK, FLW, CART, ORD)},
                with_query=pl.col("qid").is_not_null().mean())
            .sort("bucket"))
