# cheap-entry-strategy-generators

Offline backtest generators that discover cheap-entry (low-ask) buy rules for Polymarket 5-minute crypto up/down markets.

On a $1 stake the order book is efficient, so positive EV only exists at cheap entries (ask <= ~$0.25), where breakeven win-rate is `2 * entry`. Each script defines a family of cheap-side buy rules, runs an exhaustive parameter sweep over a market panel, and surfaces the variants whose realized win-rate beats `2 * avg_entry`. None place orders or touch a wallet.

Evaluation is first-trigger and leak-free: a decision at `cd=K` uses only ticks at `cd >= K`, and the winner is used only to score PnL at settle. Metric is `PnL/$1`: win = `1/entry - 1`, loss = `-1`; breakeven WR = `2 * entry`.

## Scripts

- `newstrat_crowd_flip_cheap.py`: cheap underdog side whose 10s/20s mid is turning toward favor, held N sec, with a canonical-flip coverage check. Sweeps e_cap x persist x d10_min x mc_min x conv.
- `newstrat_ev_threshold_logistic.py`: fires the cheap side when a logistic `p_win >= 2*ask + margin`, held N sec. 5-fold chronologically-blocked CV so each market is scored by a model not trained on it. Sweeps e_cap x p_margin x persist x cd-window.
- `newstrat_ofi_confirmed_cheap.py`: cheap side confirmed by order-flow/depth proxies (d10 dislocation, crowd, conviction, spread asym, mid-change, velocity, EMA), single and ANDed combos. Sweeps predicate-combos x e_cap x persist.
- `newstrat_vol_compression_cheap.py`: buys the cheap side after realized vol compresses post-move, held N sec. Sweeps e_cap x vol_win x vol_thr x persist x move_req.

Each `main()` prints a ranked table, the breakeven-beaters, the best per-market-mean and highest-coverage variants, and a machine-readable `SUMMARY_JSON` / `BEST_JSON` line for an orchestrator to parse.

## Usage

Each script is its own entry point and prints its sweep to stdout.

```bash
python3 newstrat_ev_threshold_logistic.py     # logistic p_win >= 2*ask (5-fold CV)
python3 newstrat_crowd_flip_cheap.py          # reversal-in-progress cheap buy
python3 newstrat_ofi_confirmed_cheap.py       # order-flow-confirmed cheap buy
python3 newstrat_vol_compression_cheap.py     # post-move vol-compression buy
```

Requires Python 3.8+, `numpy` (all four), and `scikit-learn` (logistic script only).

```bash
pip install numpy scikit-learn
```

## Data

Read-only; no credentials or network access.

- Input: `$DATA_DIR/market_panel.json`, per-market tick history (asks, mids, d10, crowd, conviction, EMA, spread, mid-change, winner). Sourced from the private `polymarket-data` repo.
- Output: `$DATA_DIR/edge_pnl/<family>.json`, `{slug: pnl_per_dollar}` for the chosen best variant (unfired markets = `0.0`).

`PANEL` and `OUT_COL` at the top of each file are hardcoded to a local path; point them at your data checkout. JSON/PKL/CSV artifacts are git-ignored.
