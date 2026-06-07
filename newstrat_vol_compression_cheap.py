#!/usr/bin/env python3
"""
newstrat_vol_compression_cheap.py
=================================

FAMILY: vol_compression_cheap

THESIS
------
Mean-reversion / dip-reversal is most reliable when SHORT-TERM NOISE DIES.
After a move, if realized volatility (std of recent per-second mid changes)
COMPRESSES and STABILIZES, the price has "found a level" — a cheap underdog
side bought at that level pays off above its implied probability more often
than the efficient-book baseline (where WR ~ entry price).

We buy the CHEAP side (ask <= e_cap) on the FIRST tick where:
    (1) short-term realized vol over `vol_win` seconds is COMPRESSED
        (vol <= vol_thr), and
    (2) [optional] a PRIOR move happened just before the compression
        (max-min of mid >= move_thr within the preceding `move_win` seconds),
        so we catch post-move stabilization not a dead-flat book, and
    (3) the compression has PERSISTED for >= `persist` seconds (a contiguous
        compressed run that has lasted that long), and
    (4) a side's ask <= e_cap (the cheap underdog). Both cheap -> cheaper; tie UP.

Scan cd HIGH->LOW (first-trigger, leak-free); winner used only to score PnL.
PnL/$1: win=(1/e - 1), loss=-1; mean over ALL markets (unfired=$0).

WS-SAFE: every quantity is O(1) from a cached per-second ring buffer of the
last `max_win` seconds of mid prices. No look-ahead: decision at cd=K uses
only ticks with cd>=K.

PERF: realized vol per vol_win is precomputed ONCE per market (depends only on
the mid series + window), then compression/persistence/fire is a cheap pass per
(e_cap, vol_thr, persist, move_req) variant.

EXHAUSTIVE SWEEP (7 e_cap x 3 vol_win x 6 vol_thr x 7 persist x 5 move_req
                  = 4410 variants over 456 markets).

Picks:
  * BEST   = highest per-market mean among variants that BEAT breakeven
             (WR > 2*entry) with fires >= MIN_FIRES.
  * MAXCOV = highest coverage among breakeven-beaters.
  * If none beat breakeven, report CLOSEST (smallest WR-be gap, fires>=MIN).
"""

import json
from collections import Counter

PANEL = "/home/polybot/polymarket-bot/data/market_panel.json"
OUT_COL = "/home/polybot/polymarket-bot/data/edge_pnl/vol_compression_cheap.json"
CD_FILL_FLOOR = 15
MIN_FIRES = 30  # significance floor for "best" selection

# ---- sweep grids ----
E_CAPS = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25]
VOL_WINS = [5, 10, 20]            # seconds for realized-vol window
PERSISTS = [1, 2, 3, 5, 8, 12, 20]  # seconds compression must hold
VOL_THRS = [0.004, 0.006, 0.008, 0.010, 0.014, 0.020]  # compression = vol<=thr
MOVE_REQS = [None, (20, 0.04), (20, 0.08), (40, 0.06), (40, 0.10)]
MAX_MOVE_WIN = 40  # largest move-lookback window used


def pnl(entry, won):
    return (1.0 / entry - 1.0) if won else -1.0


def build_second_series(ticks):
    """One row per distinct cd-second (LAST tick at that cd), sorted cd DESC
    (= chronological time order). up_mid carried forward if missing."""
    by_cd = {}
    for t in ticks:
        by_cd[t["cd"]] = t  # last wins
    out = []
    last_mid = None
    for cd in sorted(by_cd.keys(), reverse=True):
        t = by_cd[cd]
        um = t.get("up_mid")
        if um is None:
            ub = t.get("up_bid"); ua = t.get("up_ask")
            if ub is not None and ua is not None:
                um = 0.5 * (ub + ua)
        if um is None:
            um = last_mid
        last_mid = um
        out.append({
            "cd": cd,
            "mid": um,
            "up_ask": t.get("up_ask"),
            "dn_ask": t.get("dn_ask"),
        })
    return out


def precompute_vol(series, win_sec):
    """For each idx i (time order), realized vol = std of per-second mid changes
    over the window [cd_i, cd_i+win_sec] (i.e. the win_sec seconds up to & incl
    the current second). Returns list parallel to series; None if insufficient."""
    n = len(series)
    cds = [r["cd"] for r in series]
    mids = [r["mid"] for r in series]
    vols = [None] * n
    # window of indices is contiguous in the (cd-DESC) list: for idx i, the
    # window covers earlier-time entries with cd in [cd_i, cd_i+win]. Those are
    # indices j<=i with cds[j] <= cds[i]+win. Two-pointer.
    lo = 0
    for i in range(n):
        hi_cd = cds[i] + win_sec
        # shrink lo so cds[lo] <= hi_cd (cds DESC => cds[lo] largest)
        while lo < i and cds[lo] > hi_cd:
            lo += 1
        win_mids = [mids[j] for j in range(lo, i + 1) if mids[j] is not None]
        if len(win_mids) < 3:
            vols[i] = None
            continue
        diffs = [win_mids[k] - win_mids[k - 1] for k in range(1, len(win_mids))]
        m = sum(diffs) / len(diffs)
        var = sum((d - m) ** 2 for d in diffs) / len(diffs)
        vols[i] = var ** 0.5
    return vols


def precompute_move(series, move_win):
    """For each idx i, max-min of mid over the move_win seconds JUST BEFORE the
    current second (cd in [cd_i+1, cd_i+1+move_win]). Returns list of move
    magnitudes (0.0 if no data)."""
    n = len(series)
    cds = [r["cd"] for r in series]
    mids = [r["mid"] for r in series]
    res = [0.0] * n
    for i in range(n):
        lo_cd = cds[i] + 1
        hi_cd = cds[i] + 1 + move_win
        vals = [mids[j] for j in range(0, i + 1)
                if lo_cd <= cds[j] <= hi_cd and mids[j] is not None]
        if len(vals) >= 2:
            res[i] = max(vals) - min(vals)
    return res


def first_fire(series, vols, move_cache, e_cap, vol_thr, persist, move_req):
    """First leak-free fire (cd HIGH->LOW). Returns (cd, side, ask) or None."""
    n = len(series)
    cds = [r["cd"] for r in series]
    comp = [(v is not None and v <= vol_thr) for v in vols]
    # run-start cd of the contiguous compressed run ending at i
    run_start_cd = [None] * n
    for i in range(n):
        if not comp[i]:
            run_start_cd[i] = None
        elif i == 0 or not comp[i - 1]:
            run_start_cd[i] = cds[i]
        else:
            run_start_cd[i] = run_start_cd[i - 1]
    mv = move_cache.get(move_req) if move_req is not None else None
    for i in range(n):
        cd = cds[i]
        if cd <= CD_FILL_FLOOR:
            continue
        if not comp[i]:
            continue
        if persist > 1:
            # run must span >= persist seconds: run_start_cd - cd >= persist
            if (run_start_cd[i] - cd) < persist:
                continue
        if move_req is not None:
            if mv[i] < move_req[1]:
                continue
        ua = series[i]["up_ask"]
        da = series[i]["dn_ask"]
        up_ok = ua is not None and ua <= e_cap
        dn_ok = da is not None and da <= e_cap
        if up_ok and dn_ok:
            if ua <= da:
                return cd, "UP", ua
            return cd, "DN", da
        if up_ok:
            return cd, "UP", ua
        if dn_ok:
            return cd, "DN", da
    return None


def main():
    with open(PANEL) as f:
        markets = json.load(f)
    n = len(markets)
    print(f"Panel: {PANEL}  markets={n}")
    wdist = Counter(m["winner"] for m in markets)
    print(f"Winner dist: UP={wdist.get('UP',0)} DN={wdist.get('DN',0)}")

    print("Precomputing per-second series + vol(win) + move(win) caches ...", flush=True)
    move_wins = sorted({mr[0] for mr in MOVE_REQS if mr is not None})
    per_mkt = {}
    for m in markets:
        s = build_second_series(m["ticks"])
        vols = {w: precompute_vol(s, w) for w in VOL_WINS}
        moves = {w: precompute_move(s, w) for w in move_wins}
        # map move_req -> array
        move_cache = {mr: moves[mr[0]] for mr in MOVE_REQS if mr is not None}
        per_mkt[m["slug"]] = {"s": s, "vols": vols, "move_cache": move_cache,
                              "winner": m["winner"]}
    print("done.", flush=True)

    results = []
    total = len(E_CAPS) * len(VOL_WINS) * len(VOL_THRS) * len(PERSISTS) * len(MOVE_REQS)
    done = 0
    for vol_win in VOL_WINS:
        for vol_thr in VOL_THRS:
            for persist in PERSISTS:
                for move_req in MOVE_REQS:
                    for e_cap in E_CAPS:
                        fired = wins = 0
                        sum_pnl = sum_e = 0.0
                        col = {}
                        for slug, d in per_mkt.items():
                            fire = first_fire(d["s"], d["vols"][vol_win],
                                              d["move_cache"], e_cap, vol_thr,
                                              persist, move_req)
                            if fire is None:
                                col[slug] = 0.0
                                continue
                            cd, side, ask = fire
                            fired += 1
                            sum_e += ask
                            won = (side == d["winner"])
                            if won:
                                wins += 1
                            p = pnl(ask, won)
                            sum_pnl += p
                            col[slug] = p
                        cov = 100.0 * fired / n
                        wr = (100.0 * wins / fired) if fired else 0.0
                        avg_e = (sum_e / fired) if fired else 0.0
                        be_wr = 100.0 * 2.0 * avg_e
                        mean_fire = (sum_pnl / fired) if fired else 0.0
                        mean_mkt = sum_pnl / n
                        results.append({
                            "e_cap": e_cap, "vol_win": vol_win, "vol_thr": vol_thr,
                            "persist": persist, "move_req": move_req,
                            "fires": fired, "cov": cov, "wr": wr, "avg_e": avg_e,
                            "be_wr": be_wr, "wr_minus_be": wr - be_wr,
                            "mean_fire": mean_fire, "mean_mkt": mean_mkt, "col": col,
                        })
                        done += 1
                    if done % 350 == 0:
                        print(f"  {done}/{total} ...", flush=True)

    beaters = [r for r in results
               if r["wr_minus_be"] > 0 and r["fires"] >= MIN_FIRES]

    print("\n==== VARIANTS THAT BEAT BREAKEVEN (WR>2e), fires>=%d ====" % MIN_FIRES)
    print(f"total variants: {len(results)}   beaters: {len(beaters)}")
    hdr = (f"{'ecap':>5} {'vw':>3} {'vthr':>6} {'pst':>3} {'move':>9} "
           f"{'cov%':>5} {'fires':>5} {'WR%':>5} {'be%':>5} {'WR-be':>6} "
           f"{'m/fire':>7} {'m/mkt':>8}")
    print(hdr)
    print("-" * len(hdr))

    def fmt(r):
        mv = "off" if r["move_req"] is None else f"{r['move_req'][0]}/{r['move_req'][1]}"
        return (f"{r['e_cap']:>5.2f} {r['vol_win']:>3} {r['vol_thr']:>6.3f} "
                f"{r['persist']:>3} {mv:>9} {r['cov']:>5.1f} {r['fires']:>5} "
                f"{r['wr']:>5.1f} {r['be_wr']:>5.1f} {r['wr_minus_be']:>+6.1f} "
                f"{r['mean_fire']:>+7.3f} {r['mean_mkt']:>+8.4f}")

    for r in sorted(beaters, key=lambda x: -x["mean_mkt"])[:30]:
        print(fmt(r))

    best = max(beaters, key=lambda x: x["mean_mkt"]) if beaters else None
    maxcov = max(beaters, key=lambda x: x["cov"]) if beaters else None
    closest = None
    if not beaters:
        sig = [r for r in results if r["fires"] >= MIN_FIRES]
        closest = max(sig, key=lambda x: x["wr_minus_be"]) if sig else None

    print("\n==== BEST (max per-market mean among breakeven-beaters) ====")
    print(fmt(best) if best else "NONE beat breakeven.")
    if not best and closest:
        print("CLOSEST (smallest WR-be gap, fires>=%d):" % MIN_FIRES)
        print(fmt(closest))

    print("\n==== MAX-COVERAGE breakeven-beater ====")
    print(fmt(maxcov) if maxcov else "NONE.")

    chosen = best if best else closest
    if chosen is not None:
        with open(OUT_COL, "w") as f:
            json.dump(chosen["col"], f)
        nz = sum(1 for v in chosen["col"].values() if v != 0.0)
        print(f"\nWrote per-market pnl column -> {OUT_COL} "
              f"({len(chosen['col'])} mkts, {nz} fired)")

    summary = {
        "best": {k: v for k, v in (best or {}).items() if k != "col"} if best else None,
        "maxcov": {k: v for k, v in (maxcov or {}).items() if k != "col"} if maxcov else None,
        "closest": {k: v for k, v in (closest or {}).items() if k != "col"} if closest else None,
        "n_markets": n,
    }
    print("\nSUMMARY_JSON=" + json.dumps(summary))
    return results


if __name__ == "__main__":
    main()
