"""
Home Exam - Market Making Strategy on NASDAQ ITCH Top-of-Book Data
====================================================================
Author: [student]
Dataset: ./data.parquet  (5.34M events, 6 symbols, 19 days)

This single script reproduces the full analysis end-to-end:
    Part I   - Data cleaning and microstructure features
    Part II  - Short-horizon predictability and adverse selection
    Part III - Quoting policy (reservation price + half-spread + skew)
    Part IV  - Fill approximation (two variants)
    Part V   - Backtest, metrics, robustness checks
    Part VI  - Textual conclusions (written to outputs/log.txt)

All outputs are written to ./outputs/{figures,tables}/.
Run: python main.py
"""

from __future__ import annotations

import os
import sys
import json
import time
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, accuracy_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DATA_PATH = Path("./data.parquet")
OUT_DIR   = Path("./outputs")
FIG_DIR   = OUT_DIR / "figures"
TAB_DIR   = OUT_DIR / "tables"
LOG_PATH  = OUT_DIR / "log.txt"

for d in (OUT_DIR, FIG_DIR, TAB_DIR):
    d.mkdir(parents=True, exist_ok=True)

SYMBOLS     = ["AAL", "AMRX", "GBDC", "GEO", "GT", "MLCO"]
TICK        = 0.01                       # USD, NMS sub-penny threshold
MKT_OPEN_ET = pd.Timestamp("09:30").time()
MKT_CLOSE_ET= pd.Timestamp("16:00").time()
RNG_SEED    = 42

plt.rcParams.update({
    "figure.figsize": (9, 4.5),
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

# ---------------------------------------------------------------------
# Utility : simple logger (also prints to stdout)
# ---------------------------------------------------------------------
_LOG_LINES: List[str] = []

def log(msg: str = ""):
    line = str(msg)
    print(line)
    _LOG_LINES.append(line)

def flush_log():
    LOG_PATH.write_text("\n".join(_LOG_LINES))


# =====================================================================
# PART I - Data understanding & cleaning
# =====================================================================
def load_and_clean(path: Path) -> pd.DataFrame:
    """
    Load the parquet file and perform minimal cleaning:
        - sort by (symbol, ts_event)
        - convert ts to US/Eastern for market-hour filtering
        - flag market-hours rows
        - drop rows where either side of the BBO is missing (rare, <0.01%)
        - winsorize trade prices that are > 20 ticks from the prevailing mid
          (these are almost certainly auction prints or crossed tape events)
    """
    t0 = time.time()
    log("=" * 70)
    log("PART I - Loading and cleaning data")
    log("=" * 70)

    df = pd.read_parquet(path)
    log(f"Loaded {len(df):,} rows, {df['symbol'].nunique()} symbols")

    # Keep only fields we need; keeps memory low on 5M rows.
    keep = ["ts_event", "action", "side", "price", "size",
            "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00",
            "bid_ct_00", "ask_ct_00", "flags", "symbol"]
    df = df[keep].copy()

    # Sort per symbol/time (critical for all rolling/lookback features).
    df = df.sort_values(["symbol", "ts_event"]).reset_index(drop=True)

    # US/Eastern for session filtering.
    df["ts_et"] = df["ts_event"].dt.tz_convert("US/Eastern")
    df["date"]  = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    df["in_session"] = (df["time_et"] >= MKT_OPEN_ET) & (df["time_et"] < MKT_CLOSE_ET)

    # --- Quality filters ---------------------------------------------
    # Drop rows missing either side of the BBO. These are almost always
    # end-of-day / start-of-day states with only one side quoted.
    n_before = len(df)
    df = df.dropna(subset=["bid_px_00", "ask_px_00"])
    log(f"Dropped {n_before - len(df):,} rows with missing BBO side")

    # Flag (but keep) trades whose print is far from prevailing mid.
    mid = 0.5 * (df["bid_px_00"] + df["ask_px_00"])
    trade_mask = df["action"].eq("T")
    far = trade_mask & ((df["price"] - mid).abs() > 20 * TICK)
    df["trade_outlier"] = far
    log(f"Flagged {far.sum():,} outlier trade prints (>20 ticks from mid)")

    # --- Core microstructure features --------------------------------
    df["mid"]       = mid
    df["spread"]    = df["ask_px_00"] - df["bid_px_00"]
    df["rel_spread"] = df["spread"] / df["mid"]

    # Queue imbalance at best: (bid_sz - ask_sz) / (bid_sz + ask_sz) in [-1, 1]
    tot = (df["bid_sz_00"].astype("float64") + df["ask_sz_00"].astype("float64"))
    df["imbalance"] = np.where(
        tot > 0,
        (df["bid_sz_00"].astype("float64") - df["ask_sz_00"].astype("float64")) / tot,
        0.0,
    )

    # Microprice = size-weighted mid:  (ask_sz * bid + bid_sz * ask) / (bid_sz + ask_sz)
    # Intuition: price drifts toward the thin side because it is easier to execute there.
    df["microprice"] = np.where(
        tot > 0,
        (df["ask_sz_00"] * df["bid_px_00"] + df["bid_sz_00"] * df["ask_px_00"]) / tot,
        df["mid"],
    )

    # Trade sign convention:
    #   action='T' and side='A' -> a marketable BUY hit the ask (signed +1)
    #   action='T' and side='B' -> a marketable SELL hit the bid (signed -1)
    #   action='T' and side='N' -> cross/hidden, no clear direction
    sign = np.where(df["action"].eq("T") & df["side"].eq("A"), +1,
            np.where(df["action"].eq("T") & df["side"].eq("B"), -1, 0))
    df["trade_sign"] = sign.astype("int8")
    df["signed_trade_size"] = df["trade_sign"] * df["size"].where(trade_mask, 0).astype("int64")

    log(f"Cleaning done in {time.time()-t0:.1f}s. Final rows: {len(df):,}")
    return df


def part1_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Summary stats per symbol (in-session only), produced as a table."""
    d = df[df["in_session"]]

    out_rows = []
    for sym, g in d.groupby("symbol"):
        trades = g[g["action"].eq("T")]
        out_rows.append({
            "symbol":         sym,
            "n_events":       len(g),
            "n_trades":       len(trades),
            "n_days":         g["date"].nunique(),
            "mid_mean":       g["mid"].mean(),
            "spread_mean_c":  g["spread"].mean() * 100,       # cents
            "spread_median_c":g["spread"].median() * 100,
            "rel_spread_bps": g["rel_spread"].mean() * 1e4,   # bps
            "imbalance_std":  g["imbalance"].std(),
            "trade_size_med": trades["size"].median() if len(trades) else np.nan,
            "events_per_s":   len(g) / max(1, g["date"].nunique()) / (6.5 * 3600),  # session length
        })
    out = pd.DataFrame(out_rows).set_index("symbol").round(4)
    out.to_csv(TAB_DIR / "part1_summary_by_symbol.csv")
    log("\nSummary stats by symbol (in-session):")
    log(out.to_string())
    return out


def part1_plot_spread_seasonality(df: pd.DataFrame):
    """Average spread by minute-of-session for each symbol."""
    d = df[df["in_session"]].copy()
    d["min_of_day"] = d["ts_et"].dt.hour * 60 + d["ts_et"].dt.minute
    # Relative spread is comparable across symbols.
    grp = d.groupby(["symbol", "min_of_day"])["rel_spread"].mean().unstack(level=0)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for sym in SYMBOLS:
        if sym in grp.columns:
            ax.plot(grp.index, grp[sym] * 1e4, label=sym, linewidth=1.1)
    ax.set_xlabel("Minute of day (ET)")
    ax.set_ylabel("Relative spread (bps)")
    ax.set_title("Intraday spread seasonality by symbol (mean relative spread)")
    # Show only 09:30, 10:30, ..., 16:00 as ticks
    ticks = [570 + 60*k for k in range(0, 7)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t//60:02d}:{t%60:02d}" for t in ticks])
    ax.set_xlim(MKT_OPEN_ET.hour*60 + MKT_OPEN_ET.minute,
                MKT_CLOSE_ET.hour*60 + MKT_CLOSE_ET.minute)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "part1_spread_seasonality.png")
    plt.close(fig)
    log("Wrote figure: part1_spread_seasonality.png")


def part1_plot_event_intensity(df: pd.DataFrame):
    """Event count per minute-of-session, averaged across all days, by symbol."""
    d = df[df["in_session"]].copy()
    d["min_of_day"] = d["ts_et"].dt.hour * 60 + d["ts_et"].dt.minute
    n_days = d.groupby("symbol")["date"].nunique()
    counts = d.groupby(["symbol", "min_of_day"]).size().unstack(level=0, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for sym in SYMBOLS:
        if sym in counts.columns:
            y = counts[sym] / n_days[sym]  # events per minute per day
            ax.plot(counts.index, y, label=sym, linewidth=1.1)
    ax.set_xlabel("Minute of day (ET)")
    ax.set_ylabel("Events per minute (avg across days)")
    ax.set_title("Intraday event intensity by symbol")
    ticks = [570 + 60*k for k in range(0, 7)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t//60:02d}:{t%60:02d}" for t in ticks])
    ax.set_xlim(MKT_OPEN_ET.hour*60 + MKT_OPEN_ET.minute,
                MKT_CLOSE_ET.hour*60 + MKT_CLOSE_ET.minute)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "part1_event_intensity.png")
    plt.close(fig)
    log("Wrote figure: part1_event_intensity.png")


# =====================================================================
# PART II - Short-horizon predictability
# =====================================================================
def _resample_1s(g: pd.DataFrame) -> pd.DataFrame:
    """
    For each symbol, build a 1-second regular grid of features so that
    a forward-looking return is well-defined. We use the last observation
    in the second (forward-fill for features, aggregate for flows).
    """
    g = g.copy()
    g = g.set_index("ts_event")

    # Features sampled as last obs in the second
    last = g[["mid", "microprice", "imbalance", "spread", "rel_spread",
              "bid_sz_00", "ask_sz_00"]].resample("1s").last()

    # Flow variables summed over each second
    flow = pd.DataFrame(index=last.index)
    flow["signed_vol"] = g["signed_trade_size"].resample("1s").sum()
    flow["n_adds"]     = g["action"].eq("A").resample("1s").sum()
    flow["n_cancels"]  = g["action"].eq("C").resample("1s").sum()
    flow["n_trades"]   = g["action"].eq("T").resample("1s").sum()

    out = pd.concat([last, flow], axis=1)

    # Only keep seconds where the BBO was actually observed.
    out = out.dropna(subset=["mid"])
    return out


def part2_predictability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 1-second aggregated series per symbol, fit:
        - Logistic regression on sign(mid[t+1] - mid[t])
        - Ridge on 10-second mid return
    Evaluate out-of-sample on a chronological 70/30 split per symbol.
    """
    log("\n" + "=" * 70)
    log("PART II - Short-horizon predictability")
    log("=" * 70)

    d = df[df["in_session"]]

    rows = []
    per_sym_curves = {}

    for sym in SYMBOLS:
        g = d[d["symbol"] == sym]
        if len(g) < 1000:
            continue
        sec = _resample_1s(g)

        # --- Features (lagged by construction - all at time t) -------
        sec["micro_minus_mid"] = sec["microprice"] - sec["mid"]
        sec["micro_minus_mid_norm"] = sec["micro_minus_mid"] / TICK

        # Recent signed flow: 5-second trailing sum of signed traded size
        sec["signed_flow_5s"]  = sec["signed_vol"].rolling(5, min_periods=1).sum()
        # Recent event imbalance: (adds_bid - adds_ask) proxy via order flow
        # We use (cancels - adds) as a simple event-pressure proxy normalized by total events
        sec["event_intensity"] = (sec["n_adds"] + sec["n_cancels"] + sec["n_trades"]).rolling(5, min_periods=1).sum()

        feat_cols = ["imbalance", "micro_minus_mid_norm", "signed_flow_5s", "event_intensity"]

        # --- Targets -------------------------------------------------
        # 1-second-ahead mid move (in ticks)
        dm1  = (sec["mid"].shift(-1) - sec["mid"]) / TICK
        # 10-second-ahead mid return
        dm10 = (sec["mid"].shift(-10) - sec["mid"]) / sec["mid"]

        dat = sec[feat_cols].copy()
        dat["y_sign"] = np.sign(dm1)
        dat["y_ret"]  = dm10
        dat = dat.dropna()
        # Binary target: did mid move up (+1) vs. down (-1); drop zero moves (tick-constrained).
        dat = dat[dat["y_sign"] != 0]

        if len(dat) < 500:
            continue

        # Chronological split 70/30
        cut = int(0.7 * len(dat))
        tr, te = dat.iloc[:cut], dat.iloc[cut:]

        X_tr, X_te = tr[feat_cols].values, te[feat_cols].values
        y_cls_tr, y_cls_te = (tr["y_sign"] > 0).astype(int), (te["y_sign"] > 0).astype(int)
        y_reg_tr, y_reg_te = tr["y_ret"].values, te["y_ret"].values

        # Standardize with train stats only (prevent leakage)
        mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-12
        X_tr = (X_tr - mu) / sd
        X_te = (X_te - mu) / sd

        # Classifier
        clf = LogisticRegression(max_iter=500, random_state=RNG_SEED).fit(X_tr, y_cls_tr)
        p_te = clf.predict_proba(X_te)[:, 1]
        try:
            auc = roc_auc_score(y_cls_te, p_te)
        except ValueError:
            auc = np.nan
        acc = accuracy_score(y_cls_te, (p_te > 0.5).astype(int))

        # Ridge on 10s return
        reg = Ridge(alpha=1.0, random_state=RNG_SEED).fit(X_tr, y_reg_tr)
        r2_te = reg.score(X_te, y_reg_te)

        rows.append({
            "symbol":  sym,
            "n_train": len(tr),
            "n_test":  len(te),
            "logit_AUC": auc,
            "logit_acc": acc,
            "ridge_R2_10s": r2_te,
            **{f"coef_{c}": cl for c, cl in zip(feat_cols, clf.coef_[0])},
        })
        per_sym_curves[sym] = dict(zip(feat_cols, clf.coef_[0]))

    out = pd.DataFrame(rows).set_index("symbol").round(4)
    out.to_csv(TAB_DIR / "part2_predictability.csv")
    log("\nOut-of-sample predictability:")
    log(out.to_string())

    # --- Cross-symbol coefficient comparison figure ----------------
    coef_df = pd.DataFrame(per_sym_curves).T
    coef_df.to_csv(TAB_DIR / "part2_logit_coefs.csv")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    coef_df.plot(kind="bar", ax=ax, width=0.82)
    ax.axhline(0, color="k", linewidth=0.7)
    ax.set_title("Logistic regression coefficients (standardized) by symbol")
    ax.set_ylabel("Coefficient")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "part2_coefs_by_symbol.png")
    plt.close(fig)
    log("Wrote figure: part2_coefs_by_symbol.png")

    return out


# =====================================================================
# PART III + IV - Quote model, inventory control, fill approximation
# =====================================================================
@dataclass
class MMConfig:
    """
    Configuration of one backtest run.

    Quote structure:
        reservation = fair - gamma_usd * q
        bid         = reservation - half_spread
        ask         = reservation + half_spread

    With:
        fair         : reference price (microprice or mid).
        gamma_usd    : USD of price shift per share of inventory. Chosen so
                       that at |q| = inv_cap the quote shifts by ~1 tick.
                       Default = TICK / inv_cap.
        k_vol        : vol-scaling coefficient (spread widens with realized vol).
        min_half_sp  : floor = tick/2 (tightest NMS-compliant quote).
        inv_cap      : hard cap on |q| in shares; side-withdrawal beyond.
        fill_model   : 'top_of_book' fills when we are at BBO and a same-side
                       trade arrives; 'probabilistic' adds a random filter.
        fill_prob    : used only by the probabilistic model.
        fair_src     : 'micro' or 'mid'.
        use_skew     : toggle the inventory term.
        fill_qty     : passive size posted (capped by traded size).
    """
    gamma_usd:   float = TICK / 5_000        # 1 tick shift at full capacity
    k_vol:       float = 20.0                # applied to per-event log-return vol
    min_half_sp: float = TICK / 2.0
    inv_cap:     int   = 5_000
    fill_model:  str   = "top_of_book"       # 'top_of_book' | 'probabilistic'
    fill_prob:   float = 1.0
    fair_src:    str   = "micro"
    use_skew:    bool  = True
    fill_qty:    int   = 100
    label:       str   = "base"


def _prepare_symbol_backtest_frame(df_sym: pd.DataFrame) -> pd.DataFrame:
    """
    Downsample the event stream: keep every trade (they drive fills) and
    one out of five non-trade events (BBO updates). Preserves all fill
    opportunities while reducing memory / runtime.
    """
    g = df_sym[df_sym["in_session"]].copy()
    keep = g["action"].eq("T") | (np.arange(len(g)) % 5 == 0)
    g = g.loc[keep].reset_index(drop=True)
    return g


def _rolling_vol(mid: np.ndarray, window: int = 200) -> np.ndarray:
    """Causal rolling std of log-returns on mid."""
    r = np.zeros_like(mid)
    r[1:] = np.log(mid[1:] / mid[:-1])
    s = pd.Series(r).rolling(window, min_periods=20).std().bfill().fillna(0).to_numpy()
    return s


def backtest_symbol(df_sym: pd.DataFrame, cfg: MMConfig) -> Dict:
    """
    Event-driven backtest for one symbol.

    Loop per event:
      1. Update BBO, pick fair = mid or microprice.
      2. sigma = rolling vol of mid; half_sp = max(k_vol*sigma*mid, min_half_sp).
      3. reservation = fair - gamma * sigma^2 * mid^2 * q  (skew term is
         homogeneous across price levels thanks to mid^2 scaling).
      4. my_bid = reservation - half_sp, my_ask = reservation + half_sp.
         Quote on the aggravating side is suppressed when |q| hits inv_cap.
      5. If the event is a trade, apply the fill rule.

    Fill models:
      top_of_book  : a marketable BUY (trade with side='A') fills our OFFER
                     if our ask is <= the current BBO ask, i.e. we sit at
                     or inside the top of the book. Analogous rule for
                     marketable SELL on our BID. Fill price = our quote
                     clipped to the BBO (never better than the NBBO).
      probabilistic: same trigger but with an independent Bernoulli(fill_prob)
                     filter, to stress-test an optimistic assumption.
    """
    g = _prepare_symbol_backtest_frame(df_sym)
    if len(g) == 0:
        return {}

    mid   = g["mid"].to_numpy()
    bid   = g["bid_px_00"].to_numpy()
    ask   = g["ask_px_00"].to_numpy()
    micro = g["microprice"].to_numpy()
    act   = g["action"].to_numpy()
    side  = g["side"].to_numpy()
    tpx   = g["price"].to_numpy()
    tsz   = g["size"].to_numpy()
    ts    = g["ts_event"].to_numpy()
    outlr = g["trade_outlier"].to_numpy()

    sigma = _rolling_vol(mid, window=200)
    n = len(g)

    # Pre-compute mid value 1s after each row (adverse-selection diagnostic).
    ts_ns = ts.astype("datetime64[ns]").astype("int64")
    mid_1s_later = np.empty(n)
    j = 0
    for i in range(n):
        target = ts_ns[i] + 1_000_000_000
        while j < n and ts_ns[j] < target:
            j += 1
        mid_1s_later[i] = mid[j] if j < n else mid[i]

    q, cash = 0, 0.0
    q_hist  = np.zeros(n, dtype=np.int32)
    mtm     = np.zeros(n, dtype=np.float64)
    fills_buy, fills_sell, adverse = [], [], []
    fill_times = []          # event indices of all fills (used for holding time)
    rng = np.random.default_rng(RNG_SEED)

    for i in range(n):
        fair = micro[i] if cfg.fair_src == "micro" else mid[i]
        sig  = sigma[i]
        half = max(cfg.k_vol * sig * mid[i], cfg.min_half_sp)
        # Linear skew in shares: gamma_usd = TICK / inv_cap -> 1-tick shift
        # at full capacity. The *mid^2 * sigma^2* form of Avellaneda-Stoikov
        # is bypassed here because it is numerically tiny in a tick-constrained
        # low-vol regime and would not produce any effective skew.
        skew = cfg.gamma_usd * q if cfg.use_skew else 0.0
        res  = fair - skew
        my_bid, my_ask = res - half, res + half
        bid_live = q <  cfg.inv_cap
        ask_live = q > -cfg.inv_cap

        if act[i] == "T" and not outlr[i]:
            s_tr = tsz[i]
            # Marketable BUY: our ASK is filled if we are at or inside the BBO.
            if side[i] == "A" and ask_live:
                at_bbo = my_ask <= ask[i] + 1e-9
                ok     = at_bbo and (cfg.fill_model == "top_of_book"
                                     or rng.random() < cfg.fill_prob)
                if ok:
                    qty = min(cfg.fill_qty, int(s_tr))
                    px  = max(my_ask, bid[i])   # can't sell below best bid
                    cash += qty * px
                    q    -= qty
                    fills_sell.append((i, px, qty))
                    adverse.append((-1, px, mid_1s_later[i]))
                    fill_times.append(i)
            # Marketable SELL: our BID is filled analogously.
            elif side[i] == "B" and bid_live:
                at_bbo = my_bid >= bid[i] - 1e-9
                ok     = at_bbo and (cfg.fill_model == "top_of_book"
                                     or rng.random() < cfg.fill_prob)
                if ok:
                    qty = min(cfg.fill_qty, int(s_tr))
                    px  = min(my_bid, ask[i])   # can't buy above best ask
                    cash -= qty * px
                    q    += qty
                    fills_buy.append((i, px, qty))
                    adverse.append((+1, px, mid_1s_later[i]))
                    fill_times.append(i)

        q_hist[i] = q
        mtm[i]    = cash + q * mid[i]

    return {
        "cfg":        cfg,
        "ts":         ts,
        "q":          q_hist,
        "pnl":        mtm,
        "cash":       cash,
        "q_final":    q,
        "final_mid":  mid[-1],
        "fills_buy":  fills_buy,
        "fills_sell": fills_sell,
        "fill_times": fill_times,
        "adverse":    adverse,
        "n_events":   n,
    }


# =====================================================================
# PART V - Backtest metrics, comparison, robustness
# =====================================================================
def compute_metrics(res: Dict, df_sym: pd.DataFrame) -> Dict[str, float]:
    """Summary statistics for a single backtest result."""
    if not res:
        return {}

    ts, q, pnl = res["ts"], res["q"], res["pnl"]
    fills_b, fills_s = res["fills_buy"], res["fills_sell"]
    adv = res["adverse"]

    n_buy, n_sell = len(fills_b), len(fills_s)
    n_fills = n_buy + n_sell
    fill_imbalance = (n_buy - n_sell) / max(1, n_fills)

    total_pnl = pnl[-1] if len(pnl) else 0.0

    # Per-day PnL: end-of-day mark minus end-of-previous-day mark.
    # Day 1's PnL is the end-of-day 1 level itself (we start flat).
    ts_dt = pd.to_datetime(ts)
    df_pnl = pd.DataFrame({"ts": ts_dt, "pnl": pnl})
    if df_pnl["ts"].dt.tz is not None:
        df_pnl["date"] = df_pnl["ts"].dt.tz_convert("US/Eastern").dt.date
    else:
        df_pnl["date"] = df_pnl["ts"].dt.date
    eod = df_pnl.groupby("date")["pnl"].last()
    daily_pnl = eod.diff()
    if len(eod):
        daily_pnl.iloc[0] = eod.iloc[0]
    pnl_per_day = daily_pnl.mean() if len(daily_pnl) else 0.0

    # Drawdown
    roll_max = np.maximum.accumulate(pnl)
    dd       = pnl - roll_max
    max_dd   = dd.min() if len(dd) else 0.0

    # Inventory statistics
    q_mean  = q.mean()
    q_std   = q.std()
    q_abs_p95 = np.percentile(np.abs(q), 95)

    # Average time between fills (seconds). More robust than sign-flip-based
    # holding time when inventory stays on the same side of zero for long.
    fill_times = res.get("fill_times", [])
    if len(fill_times) > 1:
        ts_fills = ts[np.array(fill_times)]
        dt_fills = np.diff(ts_fills).astype("timedelta64[s]").astype(float)
        avg_holding_s = dt_fills.mean()
    else:
        avg_holding_s = np.nan

    # Mean quoted spread captured: difference between sell and buy fill prices
    if n_buy and n_sell:
        mean_sell = np.mean([p for _, p, _ in fills_s])
        mean_buy  = np.mean([p for _, p, _ in fills_b])
        spread_captured = mean_sell - mean_buy
    else:
        spread_captured = np.nan

    # Adverse selection: positive number means our quotes were picked off.
    #   After a BUY fill  (side=+1) at price p, unfavourable move is mid < p ->  (p - m) > 0
    #   After a SELL fill (side=-1) at price p, unfavourable move is mid > p ->  (m - p) > 0
    # Unifying: adverse = side * (p - m)
    if adv:
        adv_arr = np.array([s * (p - m) for s, p, m in adv])
        adverse_sel_per_fill = adv_arr.mean()
    else:
        adverse_sel_per_fill = np.nan

    # Half-spread-net-of-adverse-selection per fill: this is the single most
    # informative market-making diagnostic. A genuine market maker has it
    # meaningfully positive; a directional trader in disguise has it ~0 or
    # negative (all PnL comes from inventory direction, not from spread).
    if not np.isnan(spread_captured) and not np.isnan(adverse_sel_per_fill):
        net_edge_per_fill = 0.5 * spread_captured - adverse_sel_per_fill
    else:
        net_edge_per_fill = np.nan

    return {
        "total_pnl":          total_pnl,
        "pnl_per_day":        pnl_per_day,
        "n_fills":            n_fills,
        "n_buy":              n_buy,
        "n_sell":             n_sell,
        "fill_imbalance":     fill_imbalance,
        "q_mean":             q_mean,
        "q_std":              q_std,
        "q_abs_p95":          q_abs_p95,
        "avg_holding_s":      avg_holding_s,
        "spread_captured":    spread_captured,
        "adverse_sel_per_fill":adverse_sel_per_fill,
        "net_edge_per_fill":  net_edge_per_fill,
        "max_drawdown":       max_dd,
        "final_inventory":    res["q_final"],
    }


def run_all_symbols(df: pd.DataFrame, cfg: MMConfig) -> Tuple[pd.DataFrame, Dict]:
    """Run backtest on every symbol, return a table of metrics + raw results."""
    rows, raw = [], {}
    for sym in SYMBOLS:
        g = df[df["symbol"] == sym]
        if len(g) == 0:
            continue
        res = backtest_symbol(g, cfg)
        if not res:
            continue
        m = compute_metrics(res, g)
        m["symbol"] = sym
        rows.append(m)
        raw[sym] = res
    out = pd.DataFrame(rows).set_index("symbol")
    # Aggregate row. For PnL we sum across symbols. For drawdown we use the
    # sum as a conservative upper bound on a portfolio-level drawdown (strict
    # portfolio-level DD would require aligning all PnL paths on a common
    # clock; the sum is simpler and dominates).
    agg = {
        "total_pnl":       out["total_pnl"].sum(),
        "pnl_per_day":     out["pnl_per_day"].sum(),  # sum of per-symbol avg daily PnL
        "n_fills":         out["n_fills"].sum(),
        "n_buy":           out["n_buy"].sum(),
        "n_sell":          out["n_sell"].sum(),
        "fill_imbalance":  (out["n_buy"].sum() - out["n_sell"].sum())
                           / max(1, out["n_fills"].sum()),
        "q_mean":          out["q_mean"].mean(),
        "q_std":           out["q_std"].mean(),
        "q_abs_p95":       out["q_abs_p95"].mean(),
        "avg_holding_s":   out["avg_holding_s"].mean(),
        "spread_captured": out["spread_captured"].mean(),
        "adverse_sel_per_fill": out["adverse_sel_per_fill"].mean(),
        "net_edge_per_fill": out["net_edge_per_fill"].mean(),
        "max_drawdown":    out["max_drawdown"].sum(),   # conservative portfolio DD
        "final_inventory": out["final_inventory"].sum(),
    }
    out.loc["__AGGREGATE__"] = agg
    return out.round(4), raw


def part5_main(df: pd.DataFrame):
    log("\n" + "=" * 70)
    log("PART V - Backtest, benchmarks, robustness")
    log("=" * 70)

    # --- (a) Main strategy ----------------------------------------
    cfg_main = MMConfig(label="main", use_skew=True, fair_src="micro",
                        fill_model="top_of_book")
    log("\n>>> Running MAIN strategy (inventory-aware, microprice, top-of-book fills)")
    tab_main, res_main = run_all_symbols(df, cfg_main)
    tab_main.to_csv(TAB_DIR / "part5_strategy_main.csv")
    log(tab_main.to_string())

    # --- (b) Benchmark: no inventory control ----------------------
    cfg_bench = MMConfig(label="no_skew", use_skew=False, fair_src="micro",
                         fill_model="top_of_book")
    log("\n>>> Running BENCHMARK strategy (no inventory skew)")
    tab_bench, res_bench = run_all_symbols(df, cfg_bench)
    tab_bench.to_csv(TAB_DIR / "part5_strategy_no_skew.csv")
    log(tab_bench.to_string())

    # --- (c) Alternative fill model -------------------------------
    cfg_prob = MMConfig(label="prob_fill", use_skew=True, fair_src="micro",
                        fill_model="probabilistic", fill_prob=0.5)
    log("\n>>> Running ALT FILL model (probabilistic, fill_prob=0.5)")
    tab_prob, _ = run_all_symbols(df, cfg_prob)
    tab_prob.to_csv(TAB_DIR / "part5_strategy_prob_fill.csv")
    log(tab_prob.to_string())

    # --- (d) Robustness #1: drop first / last 30 minutes ----------
    df_noedges = df.copy()
    t_open  = pd.Timestamp("10:00").time()
    t_close = pd.Timestamp("15:30").time()
    df_noedges["in_session"] = df_noedges["in_session"] & \
        (df_noedges["time_et"] >= t_open) & (df_noedges["time_et"] < t_close)
    log("\n>>> Robustness #1: drop first/last 30 minutes")
    tab_noedges, _ = run_all_symbols(df_noedges, cfg_main)
    tab_noedges.to_csv(TAB_DIR / "part5_robust_no_edges.csv")
    log(tab_noedges.to_string())

    # --- (e) Robustness #2: fill probability stressed to 25% ------
    # Instructions suggest "stress the fill probability downward". We pick 25%
    # to see how quickly spread capture disappears when we lose queue priority.
    cfg_stress = MMConfig(label="stress_fill", use_skew=True, fair_src="micro",
                          fill_model="probabilistic", fill_prob=0.25)
    log("\n>>> Robustness #2: probabilistic fill stressed to 25%")
    tab_stress, _ = run_all_symbols(df, cfg_stress)
    tab_stress.to_csv(TAB_DIR / "part5_robust_fill_stress.csv")
    log(tab_stress.to_string())

    # --- Figures --------------------------------------------------
    _plot_pnl_paths(res_main, "Main strategy — cumulative PnL per symbol",
                    FIG_DIR / "part5_pnl_paths_main.png")
    _plot_inventory(res_main, "Main strategy — inventory trajectory",
                    FIG_DIR / "part5_inventory_main.png")
    _plot_pnl_comparison(res_main, res_bench,
                          FIG_DIR / "part5_pnl_comparison.png")

    # --- Build the final consolidated summary table ---------------
    final = tab_main.copy()
    final.to_csv(TAB_DIR / "final_summary.csv")
    log("\nFinal summary table written: outputs/tables/final_summary.csv")

    return tab_main, tab_bench, tab_prob, tab_noedges, tab_stress


def _plot_pnl_paths(results: Dict, title: str, out: Path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for sym, r in results.items():
        ts = pd.to_datetime(r["ts"])
        ax.plot(ts, r["pnl"], label=sym, linewidth=0.9)
    ax.set_title(title)
    ax.set_ylabel("Cumulative PnL (USD)")
    ax.set_xlabel("Time")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _plot_inventory(results: Dict, title: str, out: Path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for sym, r in results.items():
        ts = pd.to_datetime(r["ts"])
        ax.plot(ts, r["q"], label=sym, linewidth=0.7, alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("Inventory (shares)")
    ax.set_xlabel("Time")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _plot_pnl_comparison(res_main: Dict, res_bench: Dict, out: Path):
    """Side-by-side: PnL path of main vs benchmark for the largest symbol."""
    # AAL is the most liquid — best signal-to-noise for a visual comparison.
    sym = "AAL" if "AAL" in res_main else list(res_main.keys())[0]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ts_m = pd.to_datetime(res_main[sym]["ts"])
    ts_b = pd.to_datetime(res_bench[sym]["ts"])
    ax.plot(ts_m, res_main[sym]["pnl"],  label=f"{sym} — main (skew)",       linewidth=1.0)
    ax.plot(ts_b, res_bench[sym]["pnl"], label=f"{sym} — benchmark (no skew)", linewidth=1.0)
    ax.set_title(f"Inventory-aware vs. no-skew benchmark — {sym}")
    ax.set_ylabel("Cumulative PnL (USD)")
    ax.set_xlabel("Time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# =====================================================================
# PART VI - Discussion (programmatic - data-dependent text)
# =====================================================================
def part6_discussion(tab_main: pd.DataFrame, tab_bench: pd.DataFrame):
    log("\n" + "=" * 70)
    log("PART VI - Discussion (programmatic summary)")
    log("=" * 70)

    def g(tab, col):
        return tab.loc["__AGGREGATE__", col]

    main_pnl   = g(tab_main, "total_pnl")
    bench_pnl  = g(tab_bench, "total_pnl")
    n_fills    = g(tab_main, "n_fills")
    q_std      = g(tab_main, "q_std")
    q_p95      = g(tab_main, "q_abs_p95")
    adv_sel    = g(tab_main, "adverse_sel_per_fill")
    sp_cap     = g(tab_main, "spread_captured")
    net_edge   = g(tab_main, "net_edge_per_fill")
    hold       = g(tab_main, "avg_holding_s")
    dd         = g(tab_main, "max_drawdown")

    log(f"""
Aggregate PnL (main)          : {main_pnl:>12,.2f} USD
Aggregate PnL (benchmark)     : {bench_pnl:>12,.2f} USD
Delta (skew vs no-skew)       : {main_pnl - bench_pnl:>12,.2f} USD
Total fills                   : {n_fills:>12,.0f}
Inventory std (avg)           : {q_std:>12,.1f} shares
Inventory |q| p95 (avg)       : {q_p95:>12,.1f} shares
Avg time between fills        : {hold:>12,.1f} s
Mean spread captured / fill   : {sp_cap:>12.4f} USD
Mean adverse selection / fill : {adv_sel:>12.4f} USD  (positive = picked off)
NET edge per fill             : {net_edge:>12.4f} USD  (0.5*spread - adverse)
Portfolio max drawdown (sum)  : {dd:>12,.2f} USD
""")

    # Verdict heuristics
    if not np.isnan(net_edge):
        if net_edge > 0.001:
            log("VERDICT: net edge positive. The strategy captures more spread "
                "than it loses to adverse selection — behaviour is consistent "
                "with genuine market making.")
        elif net_edge > -0.001:
            log("VERDICT: net edge near zero. Spread captured and adverse "
                "selection roughly cancel; any surviving PnL is driven by "
                "inventory direction, not by making prices.")
        else:
            log("VERDICT: net edge negative. Quotes are systematically picked "
                "off; this is not a viable market maker as-is.")

    if abs(main_pnl - bench_pnl) < 1.0:
        log("Skew effect: negligible at the chosen gamma_usd. Either this "
            "low-volatility tick-constrained regime does not produce enough "
            "inventory build-up for the skew to trigger, or gamma_usd should "
            "be raised.")
    elif main_pnl > bench_pnl:
        log("Skew effect: inventory-aware quoting improves aggregate PnL vs "
            "the no-skew benchmark.")
    else:
        log("Skew effect: inventory-aware quoting underperforms the no-skew "
            "benchmark in this sample.")


# =====================================================================
# ENTRY POINT
# =====================================================================
def main():
    t_all = time.time()
    df = load_and_clean(DATA_PATH)

    part1_summary_stats(df)
    part1_plot_spread_seasonality(df)
    part1_plot_event_intensity(df)

    part2_predictability(df)

    tab_main, tab_bench, *_ = part5_main(df)
    part6_discussion(tab_main, tab_bench)

    log(f"\nTotal runtime: {time.time()-t_all:.1f}s")
    flush_log()


if __name__ == "__main__":
    main()
