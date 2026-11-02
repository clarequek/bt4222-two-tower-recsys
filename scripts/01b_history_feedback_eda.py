"""Audit JDsearch's historical action windows directly from the raw user file.

This is deliberately independent of the model pipeline. It streams the user file,
reconstructs each event's age from consecutive gaps in seconds, and writes small
tables that the companion notebook can inspect without a 27-million-row DataFrame.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from array import array
from collections import Counter
from pathlib import Path

import numpy as np


TYPES = ("CLICK", "FLW", "CART", "ORD")
EDGES = np.array([0, 1, 3, 7, 14, 30, 90, 180, 365, np.inf], dtype=float)
BUCKETS = [f"{a:g}–{b:g}" if np.isfinite(b) else f"{a:g}+" for a, b in zip(EDGES[:-1], EDGES[1:])]
SECONDS_PER_DAY = 86_400


def analyse(user_file: Path) -> tuple[dict, list[dict]]:
    counts = np.zeros((len(BUCKETS), len(TYPES)), dtype=np.int64)
    ages_by_type = {t: array("f") for t in TYPES}
    queries_by_type = Counter()
    labels = Counter()
    n_users = no_click = order_no_click = cart_no_click = positive_prefix = positive_slates = 0
    bad_rows = nonzero_first = nonmonotone = 0
    recent_shares, click_shares = array("f"), array("f")
    history_len, slate_len = array("I"), array("H")
    with user_file.open(encoding="utf-8", newline="") as stream:
        header = stream.readline().rstrip("\r\n").split("\t")
        expected = ["query", "candidate_wid_list", "candidate_label_list", "history_qry_list",
                    "history_wid_list", "history_type_list", "history_time_list"]
        if header != expected:
            raise ValueError(f"Unexpected header: {header}")
        for line in stream:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 7:
                bad_rows += 1
                continue
            _, raw_cands, raw_labels, raw_queries, raw_items, raw_types, raw_gaps = fields
            kinds = raw_types.split("_")
            gaps = np.fromstring(raw_gaps, sep="_", dtype=np.int64)
            queries = raw_queries.split("_")
            n = len(kinds)
            candidate_labels = np.fromstring(raw_labels, sep="_", dtype=np.float32).astype(np.int8)
            n_cands = raw_cands.count("_") + 1
            if (n == 0 or len(gaps) != n + 1 or len(queries) != n or
                    raw_items.count("_") + 1 != n or len(candidate_labels) != n_cands or
                    np.any(gaps < 0) or any(t not in TYPES for t in kinds) or
                    np.any((candidate_labels < 0) | (candidate_labels > 3))):
                bad_rows += 1
                continue
            n_users += 1
            nonzero_first += bool(gaps[0])
            # Example: [0, 12, 4, 5] -> event ages [21, 9, 5] seconds.
            # The last interval ends at the held-out query. The first zero is
            # an origin marker and must never be counted as a preceding gap.
            age_seconds = np.cumsum(gaps[:0:-1], dtype=np.int64)[::-1]
            nonmonotone += bool(np.any(np.diff(age_seconds) > 0))
            ages = age_seconds / SECONDS_PER_DAY
            code = np.fromiter((TYPES.index(t) for t in kinds), dtype=np.int8, count=n)
            bucket = np.searchsorted(EDGES[1:], ages, side="right")
            counts += np.bincount(bucket * len(TYPES) + code,
                                  minlength=len(BUCKETS) * len(TYPES)).reshape(counts.shape)
            for i, t in enumerate(TYPES):
                mask = code == i
                ages_by_type[t].frombytes(ages[mask].astype(np.float32).tobytes())
                queries_by_type[t] += sum(queries[j] not in ("-1", "") for j in np.flatnonzero(mask))
            c = int(np.count_nonzero(code == 0))
            k = int(np.count_nonzero(code == 2))
            o = int(np.count_nonzero(code == 3))
            no_click += c == 0
            order_no_click += c == 0 and o > 0
            cart_no_click += c == 0 and k > 0
            recent_shares.append(float(np.mean(ages < 7)))
            click_shares.append(c / n)
            history_len.append(n)
            slate_len.append(n_cands)
            labels.update(int(x) for x in candidate_labels)
            p = int(np.count_nonzero(candidate_labels))
            if p:
                positive_slates += 1
                positive_prefix += bool(np.all(candidate_labels[:p] > 0))
            if n_users % 50_000 == 0:
                print(f"Scanned {n_users:,} users", flush=True)

    if bad_rows:
        raise ValueError(f"Found {bad_rows:,} malformed user rows; results should not be used")
    type_stats = {}
    for i, t in enumerate(TYPES):
        a = np.frombuffer(ages_by_type[t], dtype=np.float32)
        type_stats[t] = {
            "events": int(len(a)), "query_tagged": int(queries_by_type[t]),
            "query_tagged_share": float(queries_by_type[t] / len(a)),
            "age_days_p50": float(np.quantile(a, .5)),
            "age_days_p90": float(np.quantile(a, .9)),
            "age_days_p99": float(np.quantile(a, .99)),
            "age_days_p999": float(np.quantile(a, .999)),
            "age_days_max": float(a.max()),
            "share_before_7d": float(np.mean(a < 7)),
            "share_before_90d": float(np.mean(a < 90)),
            "share_before_365d": float(np.mean(a < 365)),
        }
    rows = []
    for b, bucket in enumerate(BUCKETS):
        row = {"age_days": bucket, "lower_day": float(EDGES[b]), "events": int(counts[b].sum())}
        row.update({t.lower(): int(counts[b, i]) for i, t in enumerate(TYPES)})
        row["click_share"] = row["click"] / row["events"] if row["events"] else None
        row["cart_share"] = row["cart"] / row["events"] if row["events"] else None
        row["order_share"] = row["ord"] / row["events"] if row["events"] else None
        row["clicks_per_cart"] = row["click"] / row["cart"] if row["cart"] else None
        row["carts_per_order"] = row["cart"] / row["ord"] if row["ord"] else None
        rows.append(row)
    total = counts.sum(axis=0)
    same_window = counts[:3].sum(axis=0)  # [0, 7) days
    click_fraction = np.frombuffer(click_shares, dtype=np.float32)
    recent_fraction = np.frombuffer(recent_shares, dtype=np.float32)
    correlation = (float(np.corrcoef(click_fraction, recent_fraction)[0, 1])
                   if n_users > 1 and click_fraction.std() and recent_fraction.std() else None)
    recent_click_grid = np.histogram2d(recent_fraction, click_fraction,
                                       bins=20, range=((0, 1), (0, 1)))[0].astype(int)
    summary = {
        "source": str(user_file), "time_unit": "seconds", "users": n_users,
        "history_events": int(total.sum()), "candidate_impressions": int(sum(labels.values())),
        "candidate_labels": {str(k): int(labels[k]) for k in range(4)},
        "history_action_counts": dict(zip(TYPES, map(int, total))),
        "history_action_0_7d": dict(zip(TYPES, map(int, same_window))),
        "all_history_clicks_per_cart": float(total[0] / total[2]),
        "all_history_carts_per_order": float(total[2] / total[3]),
        "same_0_7d_clicks_per_cart": float(same_window[0] / same_window[2]),
        "same_0_7d_carts_per_order": float(same_window[2] / same_window[3]),
        "type_age": type_stats,
        "users_without_click": no_click, "users_with_cart_but_no_click": cart_no_click,
        "users_with_order_but_no_click": order_no_click,
        "click_share_recent_share_correlation": correlation,
        "recent_click_grid": recent_click_grid.tolist(),
        "users_with_positive_slate": positive_slates,
        "positive_slates_with_positive_prefix": positive_prefix,
        "first_gap_nonzero_users": nonzero_first, "age_sequence_violations": nonmonotone,
        "history_length_p50": float(np.median(np.frombuffer(history_len, dtype=np.uint32))),
        "history_length_p95": float(np.quantile(np.frombuffer(history_len, dtype=np.uint32), .95)),
        "slate_length_p50": float(np.median(np.frombuffer(slate_len, dtype=np.uint16))),
    }
    return summary, rows


def chart_svg(rows: list[dict], *, shares: bool) -> str:
    """Small standalone chart suitable for the notebook and GitHub preview."""
    colors = {"click": "#365fb5", "cart": "#e58b39", "ord": "#27856b", "flw": "#8564aa"}
    width, height = 1080, 510
    left, right, top, bottom = 95, 30, 70, 105
    plot_w, plot_h = width - left - right, height - top - bottom
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#253047} .small{font-size:14px} .tick{font-size:13px} .title{font-size:23px;font-weight:700}</style>',
             f'<text class="title" x="{left}" y="35">' +
             ('Recorded action mix changes with history age' if shares else 'Recorded actions vanish at different horizons') + '</text>']
    if not shares:
        parts.append(f'<text class="small" x="{width-right}" y="35" text-anchor="end">bar height = log10(count + 1)</text>')
    ymax = 100 if shares else 8
    for tick in ([0, 20, 40, 60, 80, 100] if shares else [0, 2, 4, 6, 8]):
        y = top + plot_h * (1 - tick / ymax)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e4e9ef"/>')
        label = f'{tick}%' if shares else str(tick)
        parts.append(f'<text class="tick" x="{left-12}" y="{y+5:.1f}" text-anchor="end">{label}</text>')
    slot = plot_w / len(rows)
    for i, row in enumerate(rows):
        x = left + i * slot + slot * .14
        bw = slot * .72
        base = top + plot_h
        if shares:
            for key in ("ord", "cart", "click", "flw"):
                v = 100 * row[key] / row["events"] if row["events"] else 0
                h = plot_h * v / 100
                base -= h
                parts.append(f'<rect x="{x:.1f}" y="{base:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{colors[key]}"/>')
        else:
            for j, key in enumerate(("click", "cart", "ord")):
                value = row[key]
                h = plot_h * np.log10(value + 1) / ymax
                parts.append(f'<rect x="{x+j*bw/3:.1f}" y="{top+plot_h-h:.1f}" width="{bw/3-2:.1f}" height="{h:.1f}" fill="{colors[key]}"/>')
        parts.append(f'<text class="tick" x="{left+(i+.5)*slot:.1f}" y="{top+plot_h+25}" text-anchor="middle">{html.escape(row["age_days"])}</text>')
        n_label = f'{row["events"]/1e6:.2f}m' if row["events"] >= 100_000 else f'{row["events"]:,}'
        parts.append(f'<text class="tick" x="{left+(i+.5)*slot:.1f}" y="{top+plot_h+46}" text-anchor="middle">n={n_label}</text>')
    lx = left + 15
    for key, name in (("click", "Click"), ("cart", "Cart"), ("ord", "Order"), ("flw", "Follow")):
        if not shares and key == "flw":
            continue
        parts.extend([f'<rect x="{lx}" y="{height-30}" width="15" height="15" fill="{colors[key]}"/>',
                      f'<text class="small" x="{lx+22}" y="{height-17}">{name}</text>'])
        lx += 110
    parts.append('</svg>')
    return '\n'.join(parts)


def ratio_svg(summary: dict) -> str:
    """Compare pooled history ratios with ratios inside the same seven-day window."""
    width, height = 960, 490
    left, top, plot_h = 95, 90, 295
    bottom = top + plot_h
    groups = [
        ("Clicks per cart", summary["all_history_clicks_per_cart"], summary["same_0_7d_clicks_per_cart"], (5, 10)),
        ("Carts per order", summary["all_history_carts_per_order"], summary["same_0_7d_carts_per_order"], (3, 5)),
    ]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#253047} .title{font-size:22px;font-weight:700} .body{font-size:15px} .small{font-size:13px}</style>',
             f'<text class="title" x="{left}" y="38">Same-window counts change the apparent funnel</text>',
             f'<text class="body" x="{left}" y="62">Ratios of recorded actions, not linked conversions</text>']
    for tick in (0, 2, 4, 6, 8, 10):
        y = bottom - plot_h * tick / 10
        parts += [f'<line x1="{left}" y1="{y:.1f}" x2="920" y2="{y:.1f}" stroke="#e3e9ef"/>',
                  f'<text class="small" x="{left-13}" y="{y+5:.1f}" text-anchor="end">{tick}</text>']
    for i, (name, pooled, recent, (low, high)) in enumerate(groups):
        x = 170 + i * 420
        band_top = bottom - plot_h * high / 10
        band_h = plot_h * (high - low) / 10
        parts.append(f'<rect x="{x-40}" y="{band_top:.1f}" width="280" height="{band_h:.1f}" fill="#ebf3ee"/>')
        for j, (value, color, label) in enumerate(((pooled, "#9aa8b9", "All history"),
                                                   (recent, "#27856b", "Same 0–7 days"))):
            bx = x + j * 130
            bh = plot_h * value / 10
            parts += [f'<rect x="{bx}" y="{bottom-bh:.1f}" width="74" height="{bh:.1f}" fill="{color}"/>',
                      f'<text class="body" x="{bx+37}" y="{bottom-bh-9:.1f}" text-anchor="middle">{value:.2f}</text>',
                      f'<text class="small" x="{bx+37}" y="{bottom+24}" text-anchor="middle">{label}</text>']
        parts.append(f'<text class="body" x="{x+75}" y="{height-43}" text-anchor="middle">{name}</text>')
        parts.append(f'<text class="small" x="{x+75}" y="{height-20}" text-anchor="middle">shaded: professor’s {low}:1–{high}:1 reference</text>')
    parts.append('</svg>')
    return '\n'.join(parts)


def confound_svg(summary: dict) -> str:
    """User-level density of click fraction against share of history in the last week."""
    grid = np.asarray(summary["recent_click_grid"], dtype=int)
    width, height = 760, 620
    left, top, size = 100, 95, 440
    step = size / 20
    peak = np.log1p(grid.max())
    corr = summary["click_share_recent_share_correlation"]
    corr_text = f"Pearson r = {corr:.3f}" if corr is not None else "Pearson r unavailable"
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#253047} .title{font-size:22px;font-weight:700} .body{font-size:15px} .small{font-size:13px}</style>',
             f'<text class="title" x="{left}" y="39">A user’s click fraction tracks recent-history share</text>',
             f'<text class="body" x="{left}" y="65">{summary["users"]:,} users · {corr_text}</text>']
    for xi in range(20):
        for yi in range(20):
            n = grid[xi, yi]
            intensity = np.log1p(n) / peak if peak else 0
            rgb = (int(238 - 195 * intensity), int(245 - 131 * intensity), int(251 - 71 * intensity))
            color = '#%02x%02x%02x' % rgb
            parts.append(f'<rect x="{left+xi*step:.1f}" y="{top+(19-yi)*step:.1f}" width="{step:.1f}" height="{step:.1f}" fill="{color}" stroke="#fff" stroke-width=".35"/>')
    for pct in (0, 25, 50, 75, 100):
        x = left + size * pct / 100
        y = top + size * (1 - pct / 100)
        parts += [f'<text class="small" x="{x:.1f}" y="{top+size+22}" text-anchor="middle">{pct}%</text>',
                  f'<text class="small" x="{left-12}" y="{y+5:.1f}" text-anchor="end">{pct}%</text>']
    parts += [f'<text class="body" x="{left+size/2}" y="{height-30}" text-anchor="middle">Share of user’s history in the last 7 days</text>',
              f'<text class="body" x="28" y="{top+size/2}" transform="rotate(-90 28 {top+size/2})" text-anchor="middle">Share of recorded actions that are clicks</text>',
              '<text class="small" x="575" y="145">Darker cells:</text>',
              '<text class="small" x="575" y="165">more users</text>',
              '<text class="small" x="575" y="210">Color scale is</text>',
              '<text class="small" x="575" y="230">logarithmic</text>',
              '</svg>']
    return '\n'.join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("user_file", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/history_feedback"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary, rows = analyse(args.user_file)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.out / "action_by_age.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "action_mix.svg").write_text(chart_svg(rows, shares=True))
    (args.out / "action_counts.svg").write_text(chart_svg(rows, shares=False))
    (args.out / "window_ratios.svg").write_text(ratio_svg(summary))
    (args.out / "recent_click_density.svg").write_text(confound_svg(summary))
    print(json.dumps({k: summary[k] for k in ("users", "history_events", "history_action_counts",
                                                    "same_0_7d_clicks_per_cart", "same_0_7d_carts_per_order")}, indent=2))
    print(f"Wrote {args.out / 'summary.json'} and {args.out / 'action_by_age.csv'}")


if __name__ == "__main__":
    main()
