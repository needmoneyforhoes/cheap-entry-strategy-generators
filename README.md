# cheap-entry-strategy-generators

Cheap-entry (low-ask) strategy generators for Polymarket: crowd-flip, EV-threshold logistic, order-flow-confirmed, and vol-compression buys.

## Why it exists

On Polymarket 5-minute crypto up/down markets the order book is efficient, so for a $1 stake the only place a positive-EV edge can live is **cheap entries** (ask ≤ ~$0.25), where breakeven win-rate is `2 × entry`. These scripts are **offline backtest generators**: each one defines a family of cheap-side buy rules, runs an exhaustive parameter sweep over a market panel, and surfaces the variants whose realized win-rate actually beats `2 × avg_entry`. Each writes a per-market PnL column for downstream portfolio optimization. They are research/discovery tools — they do **not** place orders or touch a wallet.

## What's inside

Each script is a self-contained sweep over `data/market_panel.json`, evaluated **first-trigger and leak-free** (a decision at `cd=K` uses only ticks at `cd ≥ K`; the winner is used only to score PnL at settle). All share the same metric convention (`PnL/$1`: win = `1/entry − 1`, loss = `−1`; breakeven WR = `2 × entry`) and write `{slug: pnl}` of the chosen best variant to `data/edge_pnl/`.

| Script | Family | Core signal | Sweep axes |
|---|---|---|---|
| `newstrat_crowd_flip_cheap.py` | `crowd_flip_cheap` | Reversal-in-progress: cheap underdog side whose 10s/20s mid is turning toward favor, held N sec (plus a canonical-flip coverage check) | e_cap × persist × d10_min × mc_min × conv |
| `newstrat_ev_threshold_logistic.py` | `ev_threshold_logistic` | Fires the cheap side only when a **logistic** `p_win ≥ 2·ask + margin`, held N sec. Uses 5-fold chronologically-blocked CV so each market is scored by a model not trained on it | e_cap × p_margin × persist × cd-window |
| `newstrat_ofi_confirmed_cheap.py` | `ofi_confirmed_cheap` | Cheap side confirmed by order-flow/depth proxies (d10 dislocation, crowd, conviction, spread asym, mid-change, velocity, EMA), single + ANDed combos | predicate-combos × e_cap × persist |
| `newstrat_vol_compression_cheap.py` | `vol_compression_cheap` | Buy the cheap side after realized vol **compresses** post-move (price "found a level"), held N sec | e_cap × vol_win × vol_thr × persist × move_req |

Each `main()` prints a ranked table, the breakeven-beaters, the best per-market-mean and highest-coverage variants, and a machine-readable `SUMMARY_JSON` / `BEST_JSON` line for an orchestrator to parse.

## Requirements

- **Python 3.8+**
- `numpy` (all four)
- `scikit-learn` (`newstrat_ev_threshold_logistic.py` only — `LogisticRegression`)
- The private **`polymarket-data`** repo for the input panel (see Data). No wallet, private key, or network access is needed — these are offline backtests.

```bash
pip install numpy scikit-learn
```

## Usage

Run any generator directly; each is its own entry point and prints its sweep to stdout.

```bash
python3 newstrat_ev_threshold_logistic.py     # logistic p_win >= 2*ask (5-fold CV)
python3 newstrat_crowd_flip_cheap.py          # reversal-in-progress cheap buy
python3 newstrat_ofi_confirmed_cheap.py       # order-flow-confirmed cheap buy
python3 newstrat_vol_compression_cheap.py     # post-move vol-compression buy
```

## Data

The scripts read the market panel from a fixed path and write per-market PnL columns alongside it:

- **Input:** `data/market_panel.json` — per-market tick history (asks, mids, d10, crowd, conviction, EMA, spread, mid-change, winner). Sourced from the private **`polymarket-data`** repo.
- **Output:** `data/edge_pnl/<family>.json` — `{slug: pnl_per_dollar}` for the chosen best variant (unfired markets = `0.0`).

Paths are currently hardcoded as `/home/polybot/polymarket-bot/data/...` at the top of each file (`PANEL` / `OUT_COL`); point them at your local checkout of the data repo. JSON/PKL/CSV artifacts are git-ignored and never committed.

---

> Private research software. No warranty; trades/handles real funds at your own risk.
