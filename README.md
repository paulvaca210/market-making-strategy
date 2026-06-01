# Home Exam — Simple Market Making Strategy

Reproducible pipeline for the home exam. Takes one parquet file of NASDAQ ITCH
top-of-book events (`data.parquet`) and produces every table and figure
referenced in the report, plus a consolidated performance summary.

## Contents

```
.
├── README.md                  this file
├── main.py                    full pipeline (Parts I → VI)
├── requirements.txt           Python dependencies
├── data.parquet               input data (not included in the zip — provide yours)
└── outputs/                   created by main.py
    ├── log.txt                console log of the run
    ├── figures/               PNGs for the report
    │   ├── part1_spread_seasonality.png
    │   ├── part1_event_intensity.png
    │   ├── part2_coefs_by_symbol.png
    │   ├── part5_pnl_paths_main.png
    │   ├── part5_inventory_main.png
    │   └── part5_pnl_comparison.png
    └── tables/                CSVs for the report
        ├── part1_summary_by_symbol.csv
        ├── part2_predictability.csv
        ├── part2_logit_coefs.csv
        ├── part5_strategy_main.csv
        ├── part5_strategy_no_skew.csv
        ├── part5_strategy_prob_fill.csv
        ├── part5_robust_no_edges.csv
        ├── part5_robust_wide.csv
        └── final_summary.csv
```

## How to run

1. **Install dependencies** (Python ≥ 3.10):

   ```bash
   pip install -r requirements.txt
   ```

2. **Place `data.parquet` in the project root** (same folder as `main.py`).

3. **Run**:

   ```bash
   python main.py
   ```

   Runtime is typically 5–15 minutes on a modern laptop. Everything is
   written to `./outputs/`. The script is fully deterministic
   (`numpy.random.default_rng(42)` is used for the probabilistic fill model).

## What the script does

### Part I — Data cleaning
Loads the parquet, sorts by `(symbol, ts_event)`, converts timestamps to
US/Eastern, filters to market hours (09:30–16:00 ET), drops events with
one side of the BBO missing, flags trade-price outliers (>20 ticks from
mid), and builds the core microstructure series: `mid`, `spread`,
`rel_spread`, `imbalance`, `microprice`, and signed trade size.

### Part II — Predictability
Builds a 1-second regular grid per symbol. Fits a logistic regression on
`sign(mid[t+1] − mid[t])` and a ridge regression on the 10-second mid
return. Features used: queue imbalance, `(microprice − mid)` in ticks,
5-second signed trade flow, recent event intensity. Train/test split is
chronological 70/30 per symbol; standardization uses train stats only.

### Part III — Quoting policy
Stylised market maker from the Avellaneda-Stoikov / Guéant-Lehalle-
Fernandez-Tapia family, simplified to discrete time:

```
σ        = rolling std of log-returns on mid (window=200 events)
half_sp  = max(k_vol · σ · mid, tick/2)
res      = fair − γ · σ² · mid² · q            (fair = microprice)
my_bid   = res − half_sp
my_ask   = res + half_sp
```

Inventory cap: no quoting on the aggravating side when `|q| ≥ inv_cap`.

### Part IV — Fill approximation
Two models:
- **trade_through (default).** When a trade prints, we are filled on the
  touched side if our posted quote was at or inside the trade price.
  Fill size is capped by an assumed passive size (100 shares) and the
  trade size.
- **probabilistic.** When a trade hits our side and our quote is at the
  BBO, we are filled with probability `fill_prob` at the BBO-clamped
  price.

### Part V — Backtest and metrics
Runs:
1. main strategy (skew + microprice + trade-through),
2. no-skew benchmark (γ = 0),
3. alternative probabilistic fill model,
4. robustness: drop first/last 30 min,
5. robustness: wider half-spread floor (1.5 ticks).

Reported metrics per symbol and aggregate: total PnL, PnL per day,
inventory distribution (mean, std, 95th percentile of |q|), average
holding time between sign flips, fill count and imbalance, mean spread
captured per fill, adverse selection per fill (mid 1 s after fill minus
fill price, times trade side), max drawdown.

### Part VI — Discussion
A short data-driven commentary is written to `log.txt` using the actual
aggregate numbers from the backtest.

## Assumptions (all documented in `main.py`)

- **Tick = 0.01 USD** for every symbol (all mid prices fall between 7 and
  16 USD — well above the sub-penny threshold).
- **Market hours = 09:30–16:00 US/Eastern**, no half-days handled
  separately (none in the 19-day window).
- **Trade sign** = +1 if `action='T'` and `side='A'` (ask hit by buyer),
  −1 if `side='B'`, 0 for `side='N'`.
- **Side 'N' trades are ignored** for signed flow and fill detection.
- **Trade outliers** (> 20 ticks from prevailing mid) are excluded from
  fill detection; they are visible in the summary stats.
- **No queue position.** We assume we are served when our quote is at or
  better than the traded price — this is an optimistic assumption and
  its effect is tested with the probabilistic fill model.

## License / Integrity

Code and analysis are entirely the author's work. External libraries
used: `pandas`, `numpy`, `scikit-learn`, `matplotlib`.
"# market--making-strategy" 
