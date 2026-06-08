#!/usr/bin/env python3
"""
newstrat_ofi_confirmed_cheap.py — ORDER-FLOW-CONFIRMED cheap entry.

FAMILY: ofi_confirmed_cheap

Idea: only buy the cheap side S (ask <= e_cap) when ORDER-FLOW / DEPTH imbalance
confirms accumulation ON THAT SIDE, persisted X seconds.

The panel has NO raw depth/volume columns, so OFI is proxied from price-derived flow:
    - d10 dislocation: d10_S - d10_opp  (S's 10-tick mid momentum dominating opp)
    - crowd: 1=UP / 0=DN — which side order-flow currently favors
    - conv: conviction magnitude of the favored side
    - spread asymmetry: spread_S < spread_opp (tighter book on S = liquidity build)
    - mid_chg_20s: 20s trailing UP-mid move; aligned-with-S confirm
    - vel: signed velocity (UP-positive)
    - ema dislocation: side ema dominates opp ema

Per market (first-trigger, leak-free, full entry->cd15 window):
  Scan cd HIGH->LOW. A side is "armed" when its OFI-confirm predicate is TRUE;
  persistence requires the predicate to have stayed TRUE continuously for >= X sec
  (cd span). First tick where an armed side has persisted >=X sec AND ask<=e_cap
  AND cd>15 -> FIRE (cheaper side if both qualify). PnL/$1: win=1/e-1, loss=-1.

PERFORMANCE: per market, OFI proxy columns -> numpy arrays ONCE. Each variant's
per-side boolean truth array computed ONCE per market. The e_cap x persist fire
scan reuses those arrays, so the full 12k-config sweep runs in numpy/light python.

Picks: (1) BEST per-market-mean among variants BEATING breakeven (WR>2*avg_e),
        (2) HIGHEST-coverage variant still beating breakeven.
Writes best variant's per-market PnL column to edge_pnl/ofi_confirmed_cheap.json
"""

import json
import numpy as np
from collections import Counter

PANEL = "./data/market_panel.json"
OUT_COL = "./data/edge_pnl/ofi_confirmed_cheap.json"
CD_FLOOR = 15

E_CAPS = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25]
PERSIST = [0, 1, 2, 3, 5, 8, 12, 20]

NAN = np.nan


def load_markets():
    with open(PANEL) as f:
        raw = json.load(f)
    mk = []
    for m in raw:
        ticks = [t for t in m["ticks"] if t["cd"] > CD_FLOOR]
        if not ticks:
            continue
        n = len(ticks)
        def col(k):
            a = np.empty(n, dtype=np.float64)
            for i, t in enumerate(ticks):
                v = t.get(k)
                a[i] = NAN if v is None else v
            return a
        d = {
            "slug": m["slug"],
            "winner_up": 1 if m["winner"] == "UP" else 0,
            "cd": np.array([t["cd"] for t in ticks], dtype=np.float64),
            "up_ask": col("up_ask"), "dn_ask": col("dn_ask"),
            "spread_up": col("spread_up"), "spread_dn": col("spread_dn"),
            "crowd": col("crowd"), "conv": col("conv"), "vel": col("vel"),
            "ema_up": col("ema_up"), "ema_dn": col("ema_dn"),
            "d10_up": col("d10_up"), "d10_dn": col("d10_dn"),
            "mid_chg_20s": col("mid_chg_20s"),
        }
        mk.append(d)
    return mk


# ---------------------------------------------------------------------------
# Predicate truth arrays. Each returns (up_bool, dn_bool, up_valid, dn_valid).
# valid = inputs present. bool only meaningful where valid; where invalid the
# persistence chain is broken (treated as not-true).
# ---------------------------------------------------------------------------

def pred_d10(d, thr):
    du, dd = d["d10_up"], d["d10_dn"]
    valid = ~(np.isnan(du) | np.isnan(dd))
    up = (du - dd) >= thr
    dn = (dd - du) >= thr
    return up, dn, valid, valid


def pred_crowd(d):
    c = d["crowd"]
    valid = ~np.isnan(c)
    up = c == 1
    dn = c == 0
    return up, dn, valid, valid


def pred_conv(d, thr):
    c = d["crowd"]; cv = d["conv"]
    valid = ~(np.isnan(c) | np.isnan(cv))
    up = (c == 1) & (cv >= thr)
    dn = (c == 0) & (cv >= thr)
    return up, dn, valid, valid


def pred_spr(d, thr):
    su, sd = d["spread_up"], d["spread_dn"]
    valid = ~(np.isnan(su) | np.isnan(sd))
    up = (sd - su) >= thr
    dn = (su - sd) >= thr
    return up, dn, valid, valid


def pred_m20(d, thr):
    m = d["mid_chg_20s"]
    valid = ~np.isnan(m)
    up = m >= thr
    dn = m <= -thr
    return up, dn, valid, valid


def pred_vel(d, thr):
    v = d["vel"]
    valid = ~np.isnan(v)
    up = v >= thr
    dn = v <= -thr
    return up, dn, valid, valid


def pred_ema(d, thr):
    eu, ed = d["ema_up"], d["ema_dn"]
    valid = ~(np.isnan(eu) | np.isnan(ed))
    up = (eu - ed) >= thr
    dn = (ed - eu) >= thr
    return up, dn, valid, valid


def build_predicates():
    P = {}
    for thr in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
        P[f"d10>{thr}"] = (lambda thr: (lambda d: pred_d10(d, thr)))(thr)
    P["crowd"] = pred_crowd
    for thr in [0.5, 0.55, 0.6, 0.65, 0.7, 0.8]:
        P[f"conv>{thr}"] = (lambda thr: (lambda d: pred_conv(d, thr)))(thr)
    for thr in [0.0, 0.01, 0.02]:
        P[f"spr>{thr}"] = (lambda thr: (lambda d: pred_spr(d, thr)))(thr)
    for thr in [0.0, 0.05, 0.1, 0.2]:
        P[f"m20>{thr}"] = (lambda thr: (lambda d: pred_m20(d, thr)))(thr)
    for thr in [0.0, 0.05, 0.1, 0.2]:
        P[f"vel>{thr}"] = (lambda thr: (lambda d: pred_vel(d, thr)))(thr)
    for thr in [0.0, 0.02, 0.05, 0.1]:
        P[f"ema>{thr}"] = (lambda thr: (lambda d: pred_ema(d, thr)))(thr)
    return P


def combine(fns):
    def f(d):
        up = None; dn = None; uv = None; dv = None
        for fn in fns:
            u, n, uvi, dvi = fn(d)
            up = u if up is None else (up & u)
            dn = n if dn is None else (dn & n)
            uv = uvi if uv is None else (uv & uvi)
            dv = dvi if dv is None else (dv & dvi)
        return up, dn, uv, dv
    return f


# ---------------------------------------------------------------------------
# Given a market's arrays + a variant's (up_true, dn_true, up_valid, dn_valid),
# compute, for EVERY (e_cap, persist), the fire (cd, side, ask) via first-trigger.
# We precompute the "armed-since cd" per side using persistence, independent of
# e_cap. Then the e_cap scan just finds the first tick meeting ask<=e_cap & armed.
#
# armed_since[i] = cd at which the current TRUE-run started (continuously true,
# valid). A tick where pred is not (valid & true) resets the run. persistence
# satisfied at i for X iff (armed_since[i] - cd[i]) >= X.
# ---------------------------------------------------------------------------

def market_fire_table(d, up_t, dn_t, up_v, dn_v):
    cd = d["cd"]
    n = len(cd)
    out = {}
    for side, tr, va, ask in (("UP", up_t, up_v, d["up_ask"]),
                              ("DN", dn_t, dn_v, d["dn_ask"])):
        good = va & tr  # predicate truly TRUE here
        armed_since = np.empty(n, dtype=np.float64)
        run_start = NAN
        for i in range(n):
            if good[i]:
                if np.isnan(run_start):
                    run_start = cd[i]
                armed_since[i] = run_start
            else:
                run_start = NAN
                armed_since[i] = NAN
        out[side] = (armed_since, ask)
    return out, cd


def eval_all(markets, variants, e_caps, persists):
    """Return list of result dicts for every (variant,e_cap,persist) that fired."""
    n_mkt = len(markets)
    # accumulators keyed by (vname, e_cap, persist)
    acc = {}
    for vname, _ in variants:
        for e in e_caps:
            for X in persists:
                acc[(vname, e, X)] = {"fired": 0, "wins": 0, "spnl": 0.0, "sent": 0.0,
                                      "bf": Counter(), "bw": Counter()}

    for d in markets:
        win_up = d["winner_up"]
        cd = d["cd"]
        for vname, fn in variants:
            up_t, dn_t, up_v, dn_v = fn(d)
            tbl, _ = market_fire_table(d, up_t, dn_t, up_v, dn_v)
            up_arm, up_ask = tbl["UP"]
            dn_arm, dn_ask = tbl["DN"]
            # cd held since arm
            up_held = up_arm - cd  # NaN where unarmed
            dn_held = dn_arm - cd
            for e in e_caps:
                up_cap = (up_ask <= e)
                dn_cap = (dn_ask <= e)
                for X in persists:
                    up_ok = up_cap & (~np.isnan(up_held)) & (up_held >= X)
                    dn_ok = dn_cap & (~np.isnan(dn_held)) & (dn_held >= X)
                    any_ok = up_ok | dn_ok
                    idx = np.argmax(any_ok)
                    if not any_ok[idx]:
                        continue
                    # decide side at fire idx: cheaper qualifying side
                    uo = up_ok[idx]; do = dn_ok[idx]
                    if uo and do:
                        if up_ask[idx] <= dn_ask[idx]:
                            side, ask = 1, up_ask[idx]
                        else:
                            side, ask = 0, dn_ask[idx]
                    elif uo:
                        side, ask = 1, up_ask[idx]
                    else:
                        side, ask = 0, dn_ask[idx]
                    a = acc[(vname, e, X)]
                    a["fired"] += 1
                    a["sent"] += ask
                    won = (side == win_up)
                    if won:
                        a["wins"] += 1
                        a["spnl"] += (1.0 / ask - 1.0)
                    else:
                        a["spnl"] += -1.0
                    fcd = cd[idx]
                    b = (">180" if fcd > 180 else ("180-90" if fcd > 90 else "90-15"))
                    a["bf"][b] += 1
                    if won:
                        a["bw"][b] += 1

    results = []
    for (vname, e, X), a in acc.items():
        if a["fired"] == 0:
            continue
        f = a["fired"]
        avg_e = a["sent"] / f
        wr = 100.0 * a["wins"] / f
        be = 100.0 * 2.0 * avg_e
        results.append({
            "name": vname, "e_cap": e, "persist": X,
            "cov": 100.0 * f / n_mkt, "fires": f, "wr": wr, "be": be,
            "beats": wr > be, "avg_e": avg_e,
            "mean_fire": a["spnl"] / f, "mean_mkt": a["spnl"] / n_mkt,
            "bf": dict(a["bf"]), "bw": dict(a["bw"]),
        })
    return results


def compute_column(markets, fn, e, X):
    col = {}
    for d in markets:
        win_up = d["winner_up"]; cd = d["cd"]
        up_t, dn_t, up_v, dn_v = fn(d)
        tbl, _ = market_fire_table(d, up_t, dn_t, up_v, dn_v)
        up_arm, up_ask = tbl["UP"]; dn_arm, dn_ask = tbl["DN"]
        up_held = up_arm - cd; dn_held = dn_arm - cd
        up_ok = (up_ask <= e) & (~np.isnan(up_held)) & (up_held >= X)
        dn_ok = (dn_ask <= e) & (~np.isnan(dn_held)) & (dn_held >= X)
        any_ok = up_ok | dn_ok
        idx = np.argmax(any_ok)
        if not any_ok[idx]:
            col[d["slug"]] = 0.0
            continue
        uo = up_ok[idx]; do = dn_ok[idx]
        if uo and do:
            side, ask = (1, up_ask[idx]) if up_ask[idx] <= dn_ask[idx] else (0, dn_ask[idx])
        elif uo:
            side, ask = 1, up_ask[idx]
        else:
            side, ask = 0, dn_ask[idx]
        won = (side == win_up)
        col[d["slug"]] = (1.0 / ask - 1.0) if won else -1.0
    return col


def main():
    markets = load_markets()
    n = len(markets)
    wu = sum(m["winner_up"] for m in markets)
    print(f"Markets: {n}  UP={wu} DN={n-wu}")
    print(f"Window: entry->cd{CD_FLOOR}; OFI-confirm cheap entry; first-trigger, leak-free\n")

    P = build_predicates()
    variants = []
    for name, fn in P.items():
        variants.append((f"[{name}]", fn))
    d10_keys = [k for k in P if k.startswith("d10>")]
    confirm_keys = (["crowd"] + [k for k in P if k.startswith("conv>")]
                    + [k for k in P if k.startswith("m20>")]
                    + [k for k in P if k.startswith("vel>")]
                    + [k for k in P if k.startswith("spr>")]
                    + [k for k in P if k.startswith("ema>")])
    for dk in d10_keys:
        for ck in confirm_keys:
            variants.append((f"[{dk} & {ck}]", combine([P[dk], P[ck]])))
    for ck in [k for k in P if k.startswith(("conv>", "m20>", "vel>", "ema>"))]:
        variants.append((f"[crowd & {ck}]", combine([P["crowd"], P[ck]])))
    for dk in ["d10>0.02", "d10>0.03", "d10>0.05"]:
        for mk in ["m20>0.05", "m20>0.1", "vel>0.05", "ema>0.05", "conv>0.6"]:
            variants.append((f"[{dk} & crowd & {mk}]", combine([P[dk], P["crowd"], P[mk]])))

    print(f"Variants={len(variants)} e_caps={len(E_CAPS)} persist={len(PERSIST)} "
          f"-> {len(variants)*len(E_CAPS)*len(PERSIST)} configs\n")

    results = eval_all(markets, variants, E_CAPS, PERSIST)
    print(f"Configs that fired: {len(results)}\n")

    def show(r):
        return (f"{r['name']:42} e={r['e_cap']:.2f} X={r['persist']:>2}s | "
                f"cov={r['cov']:5.1f}% f={r['fires']:>3} WR={r['wr']:5.1f}% "
                f"be={r['be']:5.1f}% e_avg={r['avg_e']:.3f} "
                f"m/f={r['mean_fire']:+.3f} m/mkt={r['mean_mkt']:+.4f} "
                f"{'BEATS' if r['beats'] else 'below'}")

    beating = [r for r in results if r["beats"]]
    print(f"Configs BEATING breakeven (WR>2*avg_e): {len(beating)}\n")

    MIN_FIRES = 10
    best_mean = None; best_cov = None
    if beating:
        pool = [r for r in beating if r["fires"] >= MIN_FIRES] or beating
        best_mean = max(pool, key=lambda r: r["mean_mkt"])
        best_cov = max(beating, key=lambda r: (r["cov"], r["mean_mkt"]))
        print(f"=== BEST per-market-mean (beats be, fires>={MIN_FIRES}) ===")
        print(show(best_mean)); print()
        print("=== HIGHEST-coverage (beats be) ===")
        print(show(best_cov)); print()
        print(f"=== TOP 25 by per-market mean (beats be, fires>={MIN_FIRES}) ===")
        for r in sorted(pool, key=lambda r: r["mean_mkt"], reverse=True)[:25]:
            print(show(r))
        print()
        print("=== TOP 15 by coverage (beats be) ===")
        for r in sorted(beating, key=lambda r: r["cov"], reverse=True)[:15]:
            print(show(r))

    print(f"\n=== CLOSENESS: top 15 by WR-be gap (fires>=20) ===")
    for r in sorted([x for x in results if x["fires"] >= 20],
                    key=lambda r: r["wr"] - r["be"], reverse=True)[:15]:
        print(f"{show(r)}  GAP={r['wr']-r['be']:+.1f}pp")

    print(f"\n=== TOP 12 by per-market mean OVERALL (fires>=10) ===")
    for r in sorted([x for x in results if x["fires"] >= 10],
                    key=lambda r: r["mean_mkt"], reverse=True)[:12]:
        print(show(r))

    chosen = best_mean if best_mean is not None else \
        max([x for x in results if x["fires"] >= 10], key=lambda r: r["mean_mkt"], default=None)
    pmap = dict(variants)
    if chosen is not None:
        col = compute_column(markets, pmap[chosen["name"]], chosen["e_cap"], chosen["persist"])
        with open(OUT_COL, "w") as f:
            json.dump(col, f)
        # bucket WR for chosen
        print(f"\nCHOSEN -> {OUT_COL}")
        print(show(chosen))
        bf = chosen["bf"]; bw = chosen["bw"]
        print("  fire cd-buckets (fires/wins/WR%):", {b: f"{bf.get(b,0)}/{bw.get(b,0)}/"
              f"{100*bw.get(b,0)/bf[b]:.0f}%" for b in bf})

    # machine-readable summary line for the orchestrator
    def js(r):
        return None if r is None else {k: r[k] for k in
            ("name", "e_cap", "persist", "cov", "fires", "wr", "be", "beats",
             "avg_e", "mean_fire", "mean_mkt")}
    print("\nSUMMARY_JSON " + json.dumps({
        "best_mean": js(best_mean), "best_cov": js(best_cov),
        "chosen": js(chosen), "n_beating": len(beating), "n_markets": n,
    }))
    return results


if __name__ == "__main__":
    main()
