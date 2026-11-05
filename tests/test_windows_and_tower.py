import numpy as np
import polars as pl
import torch

from jds.config import paths
from jds.prepare import load
from jds.twotower.data import N_FEAT, Stores, time_bucket
from jds.twotower.model import CATE1, QueryUserTower
from jds.twotower.train import inbatch_loss, listwise_loss
from jds.windows import feature_windows, segment_bounds


def test_feature_windows_respect_retention(tiny_cfg):
    w = {n: (t, d) for n, t, d in feature_windows(tiny_cfg)}
    assert w["click_7"] == (1, 7) and w["cart_90"] == (3, 90) and w["order_365"] == (4, 365)
    assert all(not n.startswith("click_") or d <= 7 for n, (t, d) in w.items())


def test_segment_bounds():
    h_user = np.array([0, 0, 0, 0, 1, 1])
    h_age = np.array([200.0, 30.0, 2.0, 0.1, 50.0, 1.0])
    start, b_long, b_mid, end = segment_bounds(h_user, h_age, 3, 7, 90)
    assert list(start) == [0, 4, 6] and list(end) == [4, 6, 6]
    assert list(b_long) == [1, 4, 6] and list(b_mid) == [2, 5, 6]


def test_stores_context_segments(tiny_cfg):
    p = paths(tiny_cfg)
    users = load(p.data, "users")
    hist = load(p.data, "history")
    cands = load(p.data, "candidates")
    q = load(p.data, "queries")
    Q = np.zeros((int(q["qid"].max()) + 2, 4), np.int32)
    wids = np.sort(pl.concat([cands["wid"], hist["wid"]]).unique().to_numpy())
    st = Stores(hist, users, cands, Q, wids, tiny_cfg)
    ctx = st.user_batch(np.array([1]))["ctx"]                  # ages 200, 30, 2, 1/24 days
    assert ctx["long"]["mask"].sum() == 1 and ctx["mid"]["mask"].sum() == 1 and ctx["short"]["mask"].sum() == 2
    assert list(ctx["short"]["type"][0][ctx["short"]["mask"][0]]) == [1, 1]
    # auxiliary anchor = the last click (1 hour old): only the three earlier events are its context
    ev = st.u_start[1] + 3
    b, item, w, _ = st.pair_batch(np.array([ev]))
    assert sum(b["ctx"][s]["mask"].sum() for s in ("short", "mid", "long")) == 3
    assert w[0] == 1.0
    # slates: 8 displayed items per user, labels preserved
    assert (st.s_label[1] >= 0).sum() == 8 and st.s_label[1].max() == users.filter(pl.col("user_id") == 1)["max_label"].item()
    assert time_bucket(np.array([0]))[0] == 1


def test_hard_negatives_only_from_own_training_slates(tiny_cfg):
    p = paths(tiny_cfg)
    users, hist, cands, q = (load(p.data, n) for n in ("users", "history", "candidates", "queries"))
    Q = np.zeros((int(q["qid"].max()) + 2, 4), np.int32)
    wids = np.sort(pl.concat([cands["wid"], hist["wid"]]).unique().to_numpy())
    st = Stores(hist, users, cands, Q, wids, tiny_cfg)
    u = np.array([1, 2])
    pos = st.s_item[u, 0]
    allowed = np.zeros(st.n_users, bool)
    allowed[1] = True
    neg, ok = st.neg_batch(u, pos, 8, allowed)
    assert ok[0].any() and not ok[1].any()                     # user 2 is not a training user of this model
    sl = dict(zip(st.s_item[1], st.s_label[1]))
    assert all(sl[i] == 0 for i in neg[0][ok[0]])              # only passed-over products, never a positive
    assert pos[0] not in neg[0][ok[0]]


def test_match_flags_ignore_padding_and_unknown_categories():
    B, S, L = 1, 3, 2
    segs = ("short", "mid", "long")
    cand = torch.tensor([[0, 1, 2]])                       # padding, real unknown cate, real known cate
    cand_rows = torch.zeros(B, S, N_FEAT, dtype=torch.long)
    cand_rows[0, 1, CATE1] = 0                             # unknown top category
    cand_rows[0, 2, CATE1] = 7
    hist_rows = {s: torch.zeros(B, L, N_FEAT, dtype=torch.long) for s in segs}
    hist_rows["short"][0, 0, CATE1] = 7                    # one real history event, one padding slot
    ctx = {s: {"item": torch.tensor([[3, 0]]), "mask": torch.tensor([[True, False]])} for s in segs}

    flags = QueryUserTower.match_flags(None, cand_rows, cand, hist_rows, ctx)
    assert flags[0, 0].sum() == 0                          # padding candidate never matches
    assert flags[0, 1].sum() == 0                          # unknown category never matches
    assert flags[0, 2, 0] == 0 and flags[0, 2, 1] == 1     # real product=2 != 3, category 7 matches short only
    assert flags[0, 2].sum() == 1

    empty = {s: {"item": torch.zeros(B, L, dtype=torch.long),
                 "mask": torch.zeros(B, L, dtype=torch.bool)} for s in segs}
    empty_rows = {s: torch.zeros(B, L, N_FEAT, dtype=torch.long) for s in segs}
    unknown = torch.zeros(B, S, N_FEAT, dtype=torch.long)  # every candidate category is 0
    flags = QueryUserTower.match_flags(None, unknown, torch.tensor([[1, 2, 3]]), empty_rows, empty)
    assert flags.abs().sum() == 0


def test_listwise_loss_ignores_padding_and_prefers_positive():
    labels = torch.tensor([[2, 0, 0, -1]])
    good = listwise_loss(torch.tensor([[5.0, 0.0, 0.0, 9.0]]), labels)
    bad = listwise_loss(torch.tensor([[0.0, 5.0, 0.0, 9.0]]), labels)
    assert good < bad
    assert listwise_loss(torch.tensor([[1.0, 2.0]]), torch.tensor([[0, 0]])) == 0      # no positive -> no loss


def test_inbatch_loss_masks_duplicates():
    u = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    items = torch.tensor([1, 1, 2, 3])
    loss = inbatch_loss(u, u.clone(), items, torch.ones(4), 10.0)
    assert torch.isfinite(loss)
