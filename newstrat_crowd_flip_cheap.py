#!/usr/bin/env python3
"""
newstrat_crowd_flip_cheap.py — FAMILY: crowd_flip_cheap

CONCEPT
-------
A crowd FLIP is a reversal of the favored direction. Canonically "buy the
newly-favored side while its ask is still cheap" is structurally impossible for
cheap caps (when crowd flips to a side, that side's mid just crossed 0.50, so
its ask is ~0.50-0.52 — never <=0.25). So we test TWO interpretations:

 (A) CANONICAL flip:  crowd just flipped to side S (crowd value changed and HELD
     for X seconds) AND S.ask <= e_cap AND conv(at flip) >= conv_min.
     (Kept for completeness / honesty — measured coverage is ~0 for cheap caps.)

 (B) REVERSAL-IN-PROGRESS (the real cheap play): crowd is currently on the
     OTHER side, but the CHEAP side B (ask <= e_cap) is REVERSING toward favor:
     its own d10 (10s delta of its mid) >= d10_min AND the 20s mid move points
     toward B (mc_side >= mc_min). Require this to HOLD for X seconds
     (persistence) before firing — that is the "flip that holds X sec, buy the
     side becoming favored while still cheap".

REAL-TIME, FIRST-TRIGGER, LEAK-FREE
-----------------------------------
Scan ticks cd HIGH->LOW (excluding cd<=15). The first tick whose full condition
(including the persistence window) is satisfied is the FIRE. Persistence is
measured on the cd axis (cd counts down ~1/sec, used as the seconds clock): a
signal "holds X sec" iff it has been continuously true from the cd where it
first turned true (C0) down to the current cd <= C0 - X, with no intervening
tick where it was false.

PnL per $1 stake: win -> (1/entry - 1); loss -> -1. Mean over ALL markets
(unfired market contributes $0). Breakeven WR = 2*entry.

WS-SAFE: every check is O(1) over a small cached ring of recent ticks
(persistence look-back is bounded). No look-ahead: a decision at cd=K uses only
ticks at cd>=K.
"""

import json
import itertools
from collections import defaultdict

PANEL = "./data/market_panel.json"
OUT_COL = "./data/edge_pnl/crowd_flip_cheap.json"
CD_FILL_FLOOR = 15

# ---- sweep grids ----
E_CAPS = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25]
PERSIST_SEC = [1, 2, 3, 5, 8, 12, 20]
D10_MINS = [0.00, 0.02, 0.05, 0.08, 0.10]      # own-side 10s mid delta (reversal up)
MC_MINS = [0.00, 0.02, 0.05, 0.08, 0.10]       # 20s mid move toward the cheap side
CONV_MINS = [None, 0.55, 0.60]                  # for canonical flip variant


def pnl(entry, won):
    return (1.0 / entry - 1.0) if won else -1.0


def load():
    with open(PANEL) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Variant B: reversal-in-progress cheap buy with persistence
# ---------------------------------------------------------------------------
def cond_reversal(t, side, e_cap, d10_min, mc_min):
    """O(1) per-tick predicate: is cheap side `side` reversing toward favor?"""
    ask = t["up_ask"] if side == "UP" else t["dn_ask"]
    if ask is None or ask > e_cap:
        return False, None
    d10 = t.get("d10_up") if side == "UP" else t.get("d10_dn")
    if d10 is None or d10 < d10_min:
        return False, None
    mc = t.get("mid_chg_20s")
    if mc is None:
        return False, None
    mc_side = mc if side == "UP" else -mc
    if mc_side < mc_min:
        return False, None
    # also require crowd is currently NOT on this side (a genuine reversal, not
    # already-favored) — cheap side is by definition the underdog, this is implied
    return True, ask


def first_fire_reversal(ticks, e_cap, persist, d10_min, mc_min):
    """First-trigger with cd-axis persistence. Track, per side, the cd at which
    the condition turned continuously true (run_start_cd). Fire when the run has
    lasted >= persist seconds, i.e. run_start_cd - cd >= persist.
    Scan HIGH->LOW. Returns (cd, side, ask) or None."""
    run_start = {"UP": None, "DN": None}   # cd where current true-run began
    for t in ticks:
        cd = t["cd"]
        if cd <= CD_FILL_FLOOR:
            continue
        for side in ("UP", "DN"):
            ok, ask = cond_reversal(t, side, e_cap, d10_min, mc_min)
            if ok:
                if run_start[side] is None:
                    run_start[side] = cd
                # persistence on cd axis (cd decreasing): held >= persist sec
                if run_start[side] - cd >= persist:
                    return cd, side, ask
            else:
                run_start[side] = None
    return None


# ---------------------------------------------------------------------------
# Variant A: canonical flip (newly-favored side) — for honesty/coverage report
# ---------------------------------------------------------------------------
def first_fire_canonical(ticks, e_cap, persist, conv_min):
    """Crowd flips to side S, holds X sec, S.ask<=e_cap (and conv>=conv_min)."""
    prev = None
    flip_cd = None       # cd at which crowd flipped to current side
    flip_side = None
    for t in ticks:
        cd = t["cd"]
        if cd <= CD_FILL_FLOOR:
            continue
        c = t.get("crowd")
        if c is None:
            continue
        side = "UP" if c == 1 else "DN"
        if prev is not None and c != prev:
            flip_cd = cd
            flip_side = side
        prev = c
        if flip_cd is None or side != flip_side:
            continue
        # currently on the flipped side; check persistence + cheap + conv
        if flip_cd - cd < persist:
            continue
        ask = t["up_ask"] if side == "UP" else t["dn_ask"]
        cv = t.get("conv")
        if ask is None or ask > e_cap:
            continue
        if conv_min is not None and (cv is None or cv < conv_min):
            continue
        return cd, side, ask
    return None


def score(markets, fire_fn):
    """Run a fire fn over all markets; return metrics + per-slug pnl column."""
    n = len(markets)
    fired = wins = 0
    sum_pnl = 0.0
    sum_e = 0.0
    col = {}
    for m in markets:
        slug = m["slug"]
        w = m["winner"]
        fire = fire_fn(m["ticks"])
        if fire is None:
            col[slug] = 0.0
            continue
        cd, side, ask = fire
        fired += 1
        sum_e += ask
        won = (side == w)
        if won:
            wins += 1
        p = pnl(ask, won)
        sum_pnl += p
        col[slug] = p
    cov = 100.0 * fired / n
    wr = (100.0 * wins / fired) if fired else 0.0
    mean_fire = (sum_pnl / fired) if fired else 0.0
    mean_mkt = sum_pnl / n
    avg_e = (sum_e / fired) if fired else 0.0
    be_wr = 200.0 * avg_e  # 2*avg_entry in %
    return {
        "coverage_pct": cov, "fires": fired, "wr_pct": wr,
        "breakeven_wr_pct": be_wr, "wr_minus_be": wr - be_wr,
        "mean_fire": mean_fire, "mean_mkt": mean_mkt, "avg_entry": avg_e,
        "beats": wr > be_wr,
    }, col


def main():
    markets = load()
    n = len(markets)
    print(f"Panel: {PANEL}  markets={n}")
    print(f"Window: entry -> cd{CD_FILL_FLOOR} (fire excludes cd<={CD_FILL_FLOOR})")
    print()

    all_rows = []  # (label, params, metrics, col)

    # ----- Variant B sweep (the real cheap play) -----
    print("=== VARIANT B: reversal-in-progress cheap buy (EXHAUSTIVE) ===")
    print(f"{'e_cap':>6} {'pers':>4} {'d10':>5} {'mc':>5} {'cov%':>6} "
          f"{'fires':>5} {'WR%':>6} {'beWR%':>6} {'WR-be':>7} {'m/fire':>8} {'m/mkt':>8} {'beat':>5}")
    for e_cap, persist, d10m, mcm in itertools.product(E_CAPS, PERSIST_SEC, D10_MINS, MC_MINS):
        fn = lambda ticks, e=e_cap, p=persist, dm=d10m, mm=mcm: \
            first_fire_reversal(ticks, e, p, dm, mm)
        met, col = score(markets, fn)
        if met["fires"] == 0:
            continue
        label = f"B e_cap={e_cap} persist={persist}s d10>={d10m} mc>={mcm}"
        params = {"variant": "B_reversal", "e_cap": e_cap, "persist_sec": persist,
                  "d10_min": d10m, "mc_min": mcm}
        all_rows.append((label, params, met, col))
        # print only beats or notable coverage to keep output readable
        if met["beats"] or met["coverage_pct"] >= 20:
            print(f"{e_cap:>6.2f} {persist:>4d} {d10m:>5.2f} {mcm:>5.2f} "
                  f"{met['coverage_pct']:>6.1f} {met['fires']:>5d} {met['wr_pct']:>6.1f} "
                  f"{met['breakeven_wr_pct']:>6.1f} {met['wr_minus_be']:>+7.1f} "
                  f"{met['mean_fire']:>+8.3f} {met['mean_mkt']:>+8.4f} "
                  f"{'YES' if met['beats'] else 'no':>5}")

    # ----- Variant A sweep (canonical, for honesty) -----
    print()
    print("=== VARIANT A: canonical flip (newly-favored cheap) — coverage check ===")
    print(f"{'e_cap':>6} {'pers':>4} {'conv':>5} {'cov%':>6} {'fires':>5} "
          f"{'WR%':>6} {'beWR%':>6} {'m/fire':>8} {'m/mkt':>8} {'beat':>5}")
    for e_cap, persist, cmin in itertools.product(E_CAPS, PERSIST_SEC, CONV_MINS):
        fn = lambda ticks, e=e_cap, p=persist, c=cmin: first_fire_canonical(ticks, e, p, c)
        met, col = score(markets, fn)
        if met["fires"] == 0:
            continue
        label = f"A e_cap={e_cap} persist={persist}s conv>={cmin}"
        params = {"variant": "A_canonical", "e_cap": e_cap, "persist_sec": persist,
                  "conv_min": cmin}
        all_rows.append((label, params, met, col))
        cstr = f"{cmin}" if cmin is not None else "-"
        print(f"{e_cap:>6.2f} {persist:>4d} {cstr:>5} {met['coverage_pct']:>6.1f} "
              f"{met['fires']:>5d} {met['wr_pct']:>6.1f} {met['breakeven_wr_pct']:>6.1f} "
              f"{met['mean_fire']:>+8.3f} {met['mean_mkt']:>+8.4f} "
              f"{'YES' if met['beats'] else 'no':>5}")

    # ----- select winners -----
    beaters = [r for r in all_rows if r[2]["beats"] and r[2]["fires"] >= 3]
    print()
    print(f"Variants that BEAT breakeven (WR>2*avg_entry, fires>=3): {len(beaters)}")

    best_mean = None
    best_cov = None
    if beaters:
        best_mean = max(beaters, key=lambda r: r[2]["mean_mkt"])
        best_cov = max(beaters, key=lambda r: r[2]["coverage_pct"])
        print()
        print("BEST per-market-mean (beats be):")
        print(" ", best_mean[0])
        print("  ", best_mean[2])
        print("HIGHEST coverage (beats be):")
        print(" ", best_cov[0])
        print("  ", best_cov[2])
    else:
        # honesty: report the closest miss (smallest be gap) at decent coverage
        cand = [r for r in all_rows if r[2]["fires"] >= 10]
        if cand:
            closest = max(cand, key=lambda r: r[2]["wr_minus_be"])
            print("NOTHING beats breakeven. Closest miss (fires>=10):")
            print(" ", closest[0])
            print("  ", closest[2])
        # also the best per-market-mean overall (even if negative)
        cand2 = [r for r in all_rows if r[2]["fires"] >= 10]
        if cand2:
            bm = max(cand2, key=lambda r: r[2]["mean_mkt"])
            print("Best per-market mean overall (fires>=10, may be <=0):")
            print(" ", bm[0])
            print("  ", bm[2])

    # ----- write best variant's pnl column -----
    import os
    os.makedirs(os.path.dirname(OUT_COL), exist_ok=True)
    chosen = best_mean if best_mean is not None else (
        max([r for r in all_rows if r[2]["fires"] >= 10],
            key=lambda r: r[2]["mean_mkt"]) if all_rows else None)
    if chosen is not None:
        with open(OUT_COL, "w") as f:
            json.dump(chosen[3], f)
        print()
        print(f"Wrote per-market pnl column for: {chosen[0]}")
        print(f"  -> {OUT_COL}")
    return all_rows, best_mean, best_cov


if __name__ == "__main__":
    main()
