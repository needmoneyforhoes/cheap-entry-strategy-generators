#!/usr/bin/env python3
"""
newstrat_ev_threshold_logistic.py — EV-THRESHOLD LOGISTIC cheap-buy strategy.

IDEA (the breakeven math): at $1 stake, mean/fire = p_win/e - 1, so +$1/market
needs realized WR p >= 2*e. The only place this can live is CHEAP entries
(e <= ~0.25). So: fit (offline, leak-free) a logistic model p_win for the CHEAP
side from microstructure features, and FIRE the cheap side only when the fitted
p_win >= 2*entry + margin (i.e. modelled positive-EV by the p>=2e rule) AND that
condition has PERSISTED for X seconds. This directly targets the inequality.

LEAK-FREE EVALUATION
--------------------
Look-ahead has two faces here:
  (1) within-market: a decision at cd=K may use only ticks at cd>=K. Our features
      are all per-TICK snapshots (the tick's own ask/d10/vel/conv/ema/spread plus
      a trailing dip-depth that only looks BACKWARD in cd-time), so a feature row
      at cd=K is computed from data at cd>=K. OK.
  (2) cross-market: the model that scores market M must NOT have been trained on M.
      We use K-FOLD over markets (chronologically blocked into K contiguous folds).
      Each market is scored by a logistic trained ONLY on the OTHER folds.
      => the fitted p_win used to fire in market M never saw M's outcome.

The winner is used ONLY to (a) build TRAIN labels for OTHER markets and (b) score
PnL at settle in the evaluated market. No winner info enters a market's own fire
decision.

SAMPLE / LABELS
---------------
A "candidate" = a tick where a side is cheap (ask <= E_TRAIN_CAP) in the fillable
window (cd>CD_FILL_FLOOR). For TRAINING we take, per market per side, the FIRST
cheap candidate tick of that side (one row per side per market) with label =
(that side == winner). This mirrors how we actually fire (first-trigger on the
cheap side) so the model learns the conditional win-prob of "buy this cheap side
now". Features are computed for the CHEAP side (side-relative).

FEATURES (all side-relative; "this" = the cheap side, "opp" = the other):
  ask_this            : the cheap side's ask (entry price)
  dip_depth           : (running max of this-ask seen so far in window) - this-ask
                        (how far this side has fallen from its in-window high; only
                         looks backward => leak-free)
  d10_this            : d10 of this side  (10s mid delta)
  d10_asym            : d10_this - d10_opp
  vel_signed          : vel if crowd favors this side else -vel  (flow toward this)
  crowd_for_this      : 1 if crowd points to this side else 0
  conv                : conviction
  ema_this            : ema mid of this side
  ema_diff            : ema_this - ema_opp
  spread_this         : this side's spread
  mid_chg_signed      : mid_chg_20s oriented toward this side (+ = moving to this)

FIRE RULE (first-trigger, cd high->low):
  fire the cheap side at the first tick where, IN-WINDOW (cd_lo<cd<=cd_hi):
     this-ask <= E_TRAIN_CAP (fillable cheap)  AND
     p_win(this side) >= 2*this-ask + P_MARGIN  AND
     that p-condition has held (continuously, same side) for >= PERSIST seconds.
  If both sides qualify on a tick, take the one with the larger p - 2*ask edge.

SWEEP (exhaustive):
  E_TRAIN_CAP  in {0.15, 0.20, 0.25, 0.30}     (defines "cheap" universe + cap)
  P_MARGIN     in {0.00, 0.02, 0.05, 0.08, 0.12}
  PERSIST sec  in {1, 2, 3, 5, 8, 12, 20}
  CD_WINDOW    in {full, cd>=120, cd>=60, 60..240, 30..200, cd>=200}

Pick (1) BEST per-market mean among variants that BEAT breakeven (WR>2*avg_e,
fires>=MIN_FIRES) and (2) HIGHEST-COVERAGE variant that still beats breakeven.
Write best variant's {slug: pnl_per_dollar} (unfired=0.0) to
edge_pnl/ev_threshold_logistic.json.
"""

import json
import os
import math
from collections import Counter

import numpy as np

PANEL = "./data/market_panel.json"
OUT_COL = "./data/edge_pnl/ev_threshold_logistic.json"
CD_FILL_FLOOR = 15

# E_TRAIN_CAP defines the cheap universe used for BOTH training rows and the cap
# enforced at fire. We sweep it.
E_CAPS = [0.15, 0.20, 0.25, 0.30]
P_MARGINS = [0.00, 0.02, 0.05, 0.08, 0.12]
PERSISTS = [1, 2, 3, 5, 8, 12, 20]
CD_WINDOWS = {
    "full":    (CD_FILL_FLOOR, 10_000),
    "cd>=120": (120, 10_000),
    "cd>=60":  (60, 10_000),
    "60..240": (60, 240),
    "30..200": (30, 200),
    "cd>=200": (200, 10_000),
}
MIN_FIRES = 10
N_FOLDS = 5  # chronologically blocked, leak-free CV

# feature order
FEATS = [
    "ask_this", "dip_depth", "d10_this", "d10_asym", "vel_signed",
    "crowd_for_this", "conv", "ema_this", "ema_diff", "spread_this",
    "mid_chg_signed",
]


def pnl(entry, won):
    return (1.0 / entry - 1.0) if won else -1.0


def _f(x, default=0.0):
    return default if x is None else float(x)


def prep_market(m):
    """Flatten a market into parallel per-tick arrays, side-relative features
    precomputed for BOTH sides. Keeps only cd>CD_FILL_FLOOR (entry->cd15 window).

    For each tick we compute, for side s in {UP,DN}:
      feature row (FEATS) when side s is the 'this' side.
    dip_depth uses a running max of that side's ask over the window SO FAR (cd
    high->low order) => backward-looking only => leak-free.
    """
    cds = []
    up_ask = []
    dn_ask = []
    # per-side feature rows: list of np arrays (len FEATS)
    up_feat = []
    dn_feat = []

    run_max_up = -1.0  # running max ask seen so far (backward) for dip_depth
    run_max_dn = -1.0

    for t in m["ticks"]:
        cd = t["cd"]
        if cd <= CD_FILL_FLOOR:
            continue
        ua = t.get("up_ask")
        da = t.get("dn_ask")
        if ua is None or da is None:
            continue
        d10u = _f(t.get("d10_up"))
        d10d = _f(t.get("d10_dn"))
        vel = _f(t.get("vel"))
        crowd = t.get("crowd")  # 1=UP 0=DN or None
        conv = _f(t.get("conv"))
        emau = _f(t.get("ema_up"))
        emad = _f(t.get("ema_dn"))
        spru = _f(t.get("spread_up"))
        sprd = _f(t.get("spread_dn"))
        midc = _f(t.get("mid_chg_20s"))  # + means UP mid rose over 20s

        # update running maxima (backward-looking)
        if ua > run_max_up:
            run_max_up = ua
        if da > run_max_dn:
            run_max_dn = da

        cds.append(cd)
        up_ask.append(ua)
        dn_ask.append(da)

        # crowd-for-this: UP side -> crowd==1 ; DN side -> crowd==0
        crowd_up = 1.0 if crowd == 1 else 0.0
        crowd_dn = 1.0 if crowd == 0 else 0.0
        # vel_signed toward a side: vel is signed in the crowd's direction.
        # If crowd==1 (UP), +vel means flow to UP; for DN side flow-to-this = -vel.
        if crowd == 1:
            vel_up = vel
            vel_dn = -vel
        elif crowd == 0:
            vel_up = -vel
            vel_dn = vel
        else:
            vel_up = 0.0
            vel_dn = 0.0
        # mid_chg toward a side: +midc = UP mid rose => toward UP, away from DN
        midc_up = midc
        midc_dn = -midc

        dip_up = run_max_up - ua
        dip_dn = run_max_dn - da

        up_feat.append((
            ua, dip_up, d10u, (d10u - d10d), vel_up,
            crowd_up, conv, emau, (emau - emad), spru, midc_up,
        ))
        dn_feat.append((
            da, dip_dn, d10d, (d10d - d10u), vel_dn,
            crowd_dn, conv, emad, (emad - emau), sprd, midc_dn,
        ))

    return {
        "slug": m["slug"], "winner": m["winner"],
        "cd": np.asarray(cds, dtype=np.float64),
        "up_ask": np.asarray(up_ask, dtype=np.float64),
        "dn_ask": np.asarray(dn_ask, dtype=np.float64),
        "up_feat": np.asarray(up_feat, dtype=np.float64) if up_feat else np.zeros((0, len(FEATS))),
        "dn_feat": np.asarray(dn_feat, dtype=np.float64) if dn_feat else np.zeros((0, len(FEATS))),
    }


def build_train_rows(pm, e_cap):
    """One TRAIN row per side per market: the FIRST cheap (ask<=e_cap) tick of that
    side in-window. Returns list of (feat_vec, label) where label = (side==winner).
    First-cheap mirrors the firing logic (first-trigger on the cheap side)."""
    rows = []
    winner = pm["winner"]
    cd = pm["cd"]
    n = len(cd)
    # UP side first cheap
    ua = pm["up_ask"]
    for i in range(n):
        if ua[i] <= e_cap:
            rows.append((pm["up_feat"][i], 1 if winner == "UP" else 0))
            break
    da = pm["dn_ask"]
    for i in range(n):
        if da[i] <= e_cap:
            rows.append((pm["dn_feat"][i], 1 if winner == "DN" else 0))
            break
    return rows


class LogReg:
    """Tiny standardized logistic regression (L2) via sklearn, with a manual
    standardizer so prediction is a cheap dot-product (WS-portable)."""

    def __init__(self):
        self.mean = None
        self.std = None
        self.coef = None
        self.intercept = 0.0

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-9] = 1.0
        Xs = (X - self.mean) / self.std
        # class_weight balanced helps because cheap winners are the minority
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        clf.fit(Xs, y)
        self.coef = clf.coef_[0].copy()
        self.intercept = float(clf.intercept_[0])

    def predict_proba_matrix(self, Xfeat):
        """Xfeat: (n, F) raw feature matrix -> (n,) p_win."""
        if Xfeat.shape[0] == 0:
            return np.zeros(0)
        Xs = (Xfeat - self.mean) / self.std
        z = Xs @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))


def kfold_assign(n, k):
    """Chronologically blocked fold ids for n markets (panel is chrono order)."""
    fold = np.zeros(n, dtype=np.int64)
    sz = n / k
    for i in range(n):
        fold[i] = min(k - 1, int(i / sz))
    return fold


def fit_models_per_fold(pms, e_cap, folds, k):
    """For each fold f, train on all markets NOT in f. Return list of LogReg
    indexed by fold id. Also return calibration info."""
    models = [None] * k
    for f in range(k):
        X = []
        y = []
        for idx, pm in enumerate(pms):
            if folds[idx] == f:
                continue  # held out (will be scored by this model)
            for fv, lab in build_train_rows(pm, e_cap):
                X.append(fv)
                y.append(lab)
        if not X:
            continue
        lr = LogReg()
        lr.fit(X, y)
        models[f] = lr
    return models


def precompute_pwin(pm, model, e_cap):
    """For this market, using its held-out model, compute per-tick p_win for the
    cheap side at each tick (only where that side is cheap). Returns two arrays
    p_up, p_dn (NaN where not cheap / no model)."""
    n = len(pm["cd"])
    p_up = np.full(n, np.nan)
    p_dn = np.full(n, np.nan)
    if model is None:
        return p_up, p_dn
    ua = pm["up_ask"]
    da = pm["dn_ask"]
    up_mask = ua <= e_cap
    dn_mask = da <= e_cap
    if up_mask.any():
        p_up[up_mask] = model.predict_proba_matrix(pm["up_feat"][up_mask])
    if dn_mask.any():
        p_dn[dn_mask] = model.predict_proba_matrix(pm["dn_feat"][dn_mask])
    return p_up, p_dn


def held_seconds_vec(cd, ok):
    """Vectorized persistence. cd is descending (cd[0] highest). For boolean `ok`,
    held[i] = (cd at the start of the current contiguous True-run) - cd[i], else
    -1 where ok is False. Episode-start cd is forward-filled from each run's first
    True index. O(n) numpy, no Python tick loop."""
    n = cd.shape[0]
    held = np.full(n, -1.0)
    if n == 0:
        return held
    # run-start positions: ok True and (first index OR previous False)
    prev_false = np.empty(n, dtype=bool)
    prev_false[0] = True
    prev_false[1:] = ~ok[:-1]
    run_start = ok & prev_false
    # forward-fill the cd value at each run_start over the True run
    start_cd = np.where(run_start, cd, np.nan)
    # forward fill: carry last non-nan forward
    idx = np.where(~np.isnan(start_cd), np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    ff = start_cd[idx]
    held = np.where(ok, ff - cd, -1.0)
    return held


def precompute_edges(pm, p_up, p_dn, e_cap):
    """Per-side instantaneous EV-edge = p_win - 2*ask, NaN where not cheap/no p.
    edge>=margin <=> p_win>=2*ask+margin (the fire inequality)."""
    ua = pm["up_ask"]
    da = pm["dn_ask"]
    up_cheap = (ua <= e_cap) & ~np.isnan(p_up)
    dn_cheap = (da <= e_cap) & ~np.isnan(p_dn)
    edge_up = np.where(up_cheap, p_up - 2.0 * ua, np.nan)
    edge_dn = np.where(dn_cheap, p_dn - 2.0 * da, np.nan)
    return edge_up, edge_dn


def fire_variant_vec(pm, edge_up, edge_dn, p_margin, persist_s, cd_lo, cd_hi):
    """Vectorized first-trigger. Per side: ok = edge>=p_margin (NaN->False).
    Persistence: ok held continuously for >=persist_s (cd-seconds). Eligible only
    in (cd_lo,cd_hi]. First-trigger = smallest array index (cd descends) that fires
    on either side; if both fire same index, pick larger edge."""
    cd = pm["cd"]
    n = cd.shape[0]
    if n == 0:
        return None
    up_ok = edge_up >= p_margin  # NaN>=x is False
    dn_ok = edge_dn >= p_margin
    held_up = held_seconds_vec(cd, up_ok)
    held_dn = held_seconds_vec(cd, dn_ok)
    in_win = (cd > cd_lo) & (cd <= cd_hi)
    up_fire = up_ok & (held_up >= persist_s) & in_win
    dn_fire = dn_ok & (held_dn >= persist_s) & in_win

    # earliest (smallest index) firing tick on each side
    up_idxs = np.flatnonzero(up_fire)
    dn_idxs = np.flatnonzero(dn_fire)
    i_up = up_idxs[0] if up_idxs.size else None
    i_dn = dn_idxs[0] if dn_idxs.size else None
    if i_up is None and i_dn is None:
        return None
    winner = pm["winner"]
    if i_up is not None and i_dn is not None:
        if i_up < i_dn:
            i, side = i_up, "UP"
        elif i_dn < i_up:
            i, side = i_dn, "DN"
        else:  # same tick: pick larger edge
            if edge_up[i_up] >= edge_dn[i_dn]:
                i, side = i_up, "UP"
            else:
                i, side = i_dn, "DN"
    elif i_up is not None:
        i, side = i_up, "UP"
    else:
        i, side = i_dn, "DN"
    if side == "UP":
        return cd[i], "UP", pm["up_ask"][i], (winner == "UP")
    return cd[i], "DN", pm["dn_ask"][i], (winner == "DN")


def main():
    with open(PANEL) as f:
        markets = json.load(f)
    n = len(markets)
    wdist = Counter(m["winner"] for m in markets)
    print(f"Markets: {n}  UP={wdist.get('UP',0)} DN={wdist.get('DN',0)}")
    print(f"Window: entry -> cd{CD_FILL_FLOOR}; first-trigger, leak-free.")
    print(f"Leak-free CV: {N_FOLDS} chronologically-blocked folds "
          f"(each mkt scored by a model NOT trained on it).\n")

    pms = [prep_market(m) for m in markets]
    folds = kfold_assign(n, N_FOLDS)

    # Precompute, for each e_cap, the held-out p_win arrays per market.
    # (model depends on e_cap because the cheap universe / train rows depend on it)
    print("Fitting per-fold logistic models for each e_cap...")
    pwin_cache = {}  # e_cap -> list of (p_up,p_dn) per market
    for e_cap in E_CAPS:
        models = fit_models_per_fold(pms, e_cap, folds, N_FOLDS)
        # report a quick in/out sanity: AUC-ish via mean p on win vs loss rows
        per_mkt = []
        for idx, pm in enumerate(pms):
            mdl = models[folds[idx]]
            per_mkt.append(precompute_pwin(pm, mdl, e_cap))
        pwin_cache[e_cap] = per_mkt
        # quick calibration print: among first-cheap candidate rows (held-out)
        ps, ls, es = [], [], []
        for idx, pm in enumerate(pms):
            mdl = models[folds[idx]]
            if mdl is None:
                continue
            for fv, lab in build_train_rows(pm, e_cap):
                ps.append(float(mdl.predict_proba_matrix(np.asarray([fv]))[0]))
                ls.append(lab)
                es.append(fv[0])
        ps = np.asarray(ps); ls = np.asarray(ls); es = np.asarray(es)
        # bucketed reliability
        print(f"  e_cap={e_cap:.2f}: held-out first-cheap rows={len(ps)} "
              f"base_WR={ls.mean():.3f} avg_e={es.mean():.3f}")
        for lo, hi in [(0.0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.6), (0.6, 1.01)]:
            msk = (ps >= lo) & (ps < hi)
            if msk.sum() >= 5:
                print(f"      p in [{lo:.1f},{hi:.1f}): n={msk.sum():>4} "
                      f"pred_p={ps[msk].mean():.3f} actual_WR={ls[msk].mean():.3f}")
    print()

    # ---- exhaustive variant sweep (vectorized) ----
    # Precompute, per e_cap, the per-market per-side EV-edge arrays ONCE
    # (independent of margin/persist/window).
    print("Precomputing per-market EV-edge arrays...")
    edge_cache = {}  # e_cap -> list of (edge_up, edge_dn)
    for e_cap in E_CAPS:
        per_mkt_p = pwin_cache[e_cap]
        edges = []
        for k, pm in enumerate(pms):
            p_up, p_dn = per_mkt_p[k]
            edges.append(precompute_edges(pm, p_up, p_dn, e_cap))
        edge_cache[e_cap] = edges
    print("Sweeping variants...", flush=True)

    variants = {}
    n_done = 0
    total_variants = len(E_CAPS) * len(P_MARGINS) * len(PERSISTS) * len(CD_WINDOWS)
    for e_cap in E_CAPS:
        edges = edge_cache[e_cap]
        for p_margin in P_MARGINS:
            for persist_s in PERSISTS:
                for win_name, (lo, hi) in CD_WINDOWS.items():
                    fired = wins = 0
                    sum_pnl = sum_e = 0.0
                    col = {}
                    for k, pm in enumerate(pms):
                        col[pm["slug"]] = 0.0
                        edge_up, edge_dn = edges[k]
                        fr = fire_variant_vec(pm, edge_up, edge_dn, p_margin,
                                              persist_s, lo, hi)
                        if fr is None:
                            continue
                        c, side, ask, won = fr
                        fired += 1
                        sum_e += ask
                        if won:
                            wins += 1
                        p = pnl(ask, won)
                        sum_pnl += p
                        col[pm["slug"]] = p
                    cov = 100.0 * fired / n
                    wr = (100.0 * wins / fired) if fired else 0.0
                    avg_e = (sum_e / fired) if fired else 0.0
                    be_wr = 100.0 * 2.0 * avg_e
                    mean_fire = (sum_pnl / fired) if fired else 0.0
                    mean_mkt = sum_pnl / n
                    variants[(e_cap, p_margin, persist_s, win_name)] = {
                        "e_cap": e_cap, "p_margin": p_margin, "persist": persist_s,
                        "win_name": win_name, "cov": cov, "fires": fired,
                        "wins": wins, "wr": wr, "avg_e": avg_e, "be_wr": be_wr,
                        "wr_minus_be": wr - be_wr, "mean_fire": mean_fire,
                        "mean_mkt": mean_mkt, "col": col,
                    }
                    n_done += 1
        print(f"  e_cap={e_cap:.2f} done ({n_done}/{total_variants} variants)", flush=True)

    all_variants = list(variants.values())
    print(f"Swept {len(all_variants)} variants.\n")

    def label(v):
        return (f"logistic p_win>=2*ask+{v['p_margin']:.2f}, cheap e_cap<={v['e_cap']:.2f}, "
                f"persist>={v['persist']}s, cd-window={v['win_name']}, "
                f"first-trigger; both-fire->larger(p-2*ask); 5-fold leak-free CV")

    beats = [v for v in all_variants
             if v["wr"] > v["be_wr"] and v["fires"] >= MIN_FIRES]

    print("=== TOP 25 by mean/mkt (min_fires=%d) ===" % MIN_FIRES)
    print(f"{'cov%':>6} {'fires':>5} {'WR%':>5} {'beWR%':>6} {'WR-be':>6} "
          f"{'m/fire':>8} {'m/mkt':>8}  e_cap pmarg persist  window")
    for v in sorted((x for x in all_variants if x["fires"] >= MIN_FIRES),
                    key=lambda x: -x["mean_mkt"])[:25]:
        print(f"{v['cov']:>6.1f} {v['fires']:>5} {v['wr']:>5.1f} {v['be_wr']:>6.1f} "
              f"{v['wr_minus_be']:>+6.1f} {v['mean_fire']:>+8.3f} "
              f"{v['mean_mkt']:>+8.4f}  {v['e_cap']:.2f}  {v['p_margin']:.2f}  "
              f"{v['persist']:>3}s  {v['win_name']}")

    print(f"\n=== BEAT breakeven (WR>2e, fires>={MIN_FIRES}): {len(beats)} ===")
    if beats:
        for v in sorted(beats, key=lambda x: -x["mean_mkt"])[:25]:
            print(f"{v['cov']:>6.1f} {v['fires']:>5} {v['wr']:>5.1f} {v['be_wr']:>6.1f} "
                  f"{v['wr_minus_be']:>+6.1f} {v['mean_fire']:>+8.3f} "
                  f"{v['mean_mkt']:>+8.4f}  {v['e_cap']:.2f}  {v['p_margin']:.2f}  "
                  f"{v['persist']:>3}s  {v['win_name']}")
    else:
        print("NONE beat breakeven. Closest (largest WR-be):")
        for v in sorted((x for x in all_variants if x["fires"] >= MIN_FIRES),
                        key=lambda x: -x["wr_minus_be"])[:15]:
            print(f"{v['cov']:>6.1f} {v['fires']:>5} {v['wr']:>5.1f} {v['be_wr']:>6.1f} "
                  f"{v['wr_minus_be']:>+6.1f} {v['mean_fire']:>+8.3f} "
                  f"{v['mean_mkt']:>+8.4f}  {v['e_cap']:.2f}  {v['p_margin']:.2f}  "
                  f"{v['persist']:>3}s  {v['win_name']}")

    if beats:
        best = max(beats, key=lambda x: x["mean_mkt"])
        best_cov = max(beats, key=lambda x: x["cov"])
        pool = "beats_breakeven"
    else:
        best = max((v for v in all_variants if v["fires"] >= MIN_FIRES),
                   key=lambda x: x["mean_mkt"])
        best_cov = None
        pool = "none_beat_breakeven_best_meanmkt"

    print("\n=== CHOSEN BEST (pool=%s) ===" % pool)
    print(label(best))
    print(f"cov={best['cov']:.1f}% fires={best['fires']} WR={best['wr']:.2f}% "
          f"beWR={best['be_wr']:.2f}% WR-be={best['wr_minus_be']:+.2f} "
          f"avg_e={best['avg_e']:.4f} mean/fire={best['mean_fire']:+.4f} "
          f"mean/mkt={best['mean_mkt']:+.4f}")

    print("\n=== HIGHEST-COVERAGE that beats breakeven ===")
    if best_cov is not None:
        print(label(best_cov))
        print(f"cov={best_cov['cov']:.1f}% fires={best_cov['fires']} "
              f"WR={best_cov['wr']:.2f}% beWR={best_cov['be_wr']:.2f}% "
              f"WR-be={best_cov['wr_minus_be']:+.2f} "
              f"mean/fire={best_cov['mean_fire']:+.4f} "
              f"mean/mkt={best_cov['mean_mkt']:+.4f}")
    else:
        print("(none beat breakeven)")

    os.makedirs(os.path.dirname(OUT_COL), exist_ok=True)
    with open(OUT_COL, "w") as f:
        json.dump(best["col"], f)
    print(f"\nWrote best-variant per-market pnl column -> {OUT_COL} "
          f"({len(best['col'])} markets)")

    print("\nBEST_JSON=" + json.dumps({
        "family": "ev_threshold_logistic",
        "best_variant": label(best),
        "coverage_pct": best["cov"], "wr": best["wr"] / 100.0,
        "mean_entry": best["avg_e"], "breakeven_wr": 2 * best["avg_e"],
        "beats_breakeven": bool(best["wr"] > best["be_wr"]),
        "per_fire_mean": best["mean_fire"], "per_market_mean": best["mean_mkt"],
        "best_persistence_sec": best["persist"], "n_fires": best["fires"],
        "best_cov_variant": (label(best_cov) if best_cov else None),
        "best_cov_cov": (best_cov["cov"] if best_cov else None),
        "best_cov_meanmkt": (best_cov["mean_mkt"] if best_cov else None),
    }))


if __name__ == "__main__":
    main()
