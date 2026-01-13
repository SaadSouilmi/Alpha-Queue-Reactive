#!/usr/bin/env python3
"""Generate a 2-page PDF fit report comparing empirical data with QR simulation."""

import argparse
import re
from functools import reduce
from pathlib import Path

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from lobib import DataLoader

sns.set_style("whitegrid")

# Constants
DAY_NS = int(5.5 * 3600 * 1e9)  # 5.5 hours in nanoseconds
HOUR_NS = int(3600 * 1e9)
HOURS_PER_DAY = 5.5
TICK_TO_USD = 0.01


def parse_sim_params(filename: str) -> dict:
    """Parse simulation parameters from filename."""
    params = {"impact": "none", "race": False, "k": None, "theta": None}

    if filename in ("result.parquet", "result"):
        return {"impact": "none", "race": False, "k": None, "theta": None, "label": "Basic QR"}

    if "ema_impact" in filename:
        params["impact"] = "ema"

    if "_race" in filename and "_norace" not in filename:
        params["race"] = True

    # Extract k value: _k1.0_ or _k0.5_
    if "_k" in filename:
        match = re.search(r"_k(\d+\.\d+)", filename)
        if match:
            params["k"] = float(match.group(1))

    # Extract theta: _theta0.05
    if "_theta" in filename:
        match = re.search(r"_theta(\d+\.\d+)", filename)
        if match:
            params["theta"] = float(match.group(1))

    # Build label
    parts = []
    if params["race"]:
        parts.append(f"Race (θ={params['theta']})" if params["theta"] is not None else "Race")
    else:
        parts.append("No Race")
    if params["impact"] == "ema":
        parts.append("EMA Impact")
    if params["k"] is not None:
        parts.append(f"k={params['k']}")
    params["label"] = ", ".join(parts)

    return params


def pl_select(condlist: list[pl.Expr], choicelist: list[pl.Expr]) -> pl.Expr:
    return reduce(
        lambda expr, cond_choice: expr.when(cond_choice[0]).then(cond_choice[1]),
        zip(condlist, choicelist),
        pl.when(condlist[0]).then(choicelist[0]),
    )


def imbalance_exp() -> tuple[pl.Expr, pl.Expr]:
    condlist = [pl.col("best_bid_nbr").eq(-i) for i in range(1, 11)]
    choicelist = [pl.col(f"Q_{-i}") for i in range(1, 11)]
    best_bid = pl_select(condlist, choicelist).alias("best_bid").truediv(500).ceil()

    condlist = [pl.col("best_ask_nbr").eq(i) for i in range(1, 11)]
    choicelist = [pl.col(f"Q_{i}") for i in range(1, 11)]
    best_ask = pl_select(condlist, choicelist).alias("best_ask").truediv(500).ceil()
    imb = ((best_bid - best_ask) / (best_bid + best_ask)).alias("imb")

    bins = np.arange(11, step=1) / 10
    condlist = [
        *[
            pl.col("imb").ge(left) & pl.col("imb").lt(right)
            for left, right in zip(-bins[1:][::-1], -bins[:-1][::-1])
        ],
        pl.col("imb").eq(0),
        *[
            pl.col("imb").gt(left) & pl.col("imb").le(right)
            for left, right in zip(bins[:-1], bins[1:])
        ],
    ]
    choicelist = [*(-bins[1:][::-1]), 0, *bins[1:]]
    imb_bin = pl_select(condlist, choicelist).alias("imb_bin")
    return imb, imb_bin


def imbalance_sim() -> pl.Expr:
    """Compute imbalance bin for simulation data."""
    imb = ((pl.col("Q_-1") - pl.col("Q_1")) / (pl.col("Q_-1") + pl.col("Q_1"))).alias("imb")
    bins = np.arange(11, step=1) / 10
    condlist = [
        *[
            pl.col("imb").ge(left) & pl.col("imb").lt(right)
            for left, right in zip(-bins[1:][::-1], -bins[:-1][::-1])
        ],
        pl.col("imb").eq(0),
        *[
            pl.col("imb").gt(left) & pl.col("imb").le(right)
            for left, right in zip(bins[:-1], bins[1:])
        ],
    ]
    choicelist = [*(-bins[1:][::-1]), 0, *bins[1:]]
    imb_bin = pl_select(condlist, choicelist).alias("imb_bin")
    return imb, imb_bin


def load_empirical(ticker: str) -> pl.LazyFrame:
    """Load and preprocess empirical data for a ticker (lazy)."""
    loader = DataLoader()
    info = loader.ticker_info(ticker)
    df = loader.load(
        ticker,
        start_date=info["date"].min(),
        end_date=info["date"].max(),
        schema="qr",
        eager=False,
    ).sort(["date", "ts_event"])

    # Filter rows with best_bid_nbr and best_ask_nbr in expected range
    df = df.filter(
        pl.col("best_bid_nbr").is_between(-10, -1)
        & pl.col("best_ask_nbr").is_between(1, 10)
    )

    df = df.filter(
        (
            pl.col("event_side")
            .replace({"A": 1, "B": -1})
            .cast(int)
            .mul(pl.col("event_queue_nbr"))
            >= 0
        )
    )
    df = df.with_columns(pl.col("event").replace({"Trd_All": "Trd"}))
    df = df.with_columns(
        pl.when(pl.col("event_queue_nbr").lt(0))
        .then(pl.col("event_queue_nbr").sub(pl.col("best_bid_nbr")).sub(1))
        .otherwise(pl.col("event_queue_nbr").sub(pl.col("best_ask_nbr")).add(1))
        .alias("event_q")
    )
    imb, imb_bin = imbalance_exp()
    df = df.with_columns(imb).with_columns(imb_bin)
    df = df.filter(pl.col("event_q").abs().le(2))
    df = df.with_columns(
        pl.when(pl.col("spread").ge(2)).then(2).otherwise(pl.col("spread")).alias("spread")
    )
    df = df.with_columns(pl.col("P_1").add(pl.col("P_-1")).truediv(2).alias("mid"))
    df = df.with_columns(pl.col("ts_event").diff().over("date").cast(int).alias("dt"))
    return df


def load_simulation(path: Path) -> pl.LazyFrame:
    """Load and preprocess simulation data (lazy)."""
    df = pl.scan_parquet(path)
    df = df.with_columns((pl.col("ts_event") // DAY_NS).alias("date"))
    df = df.filter(~pl.col("rejected"))
    imb, imb_bin = imbalance_sim()
    df = df.with_columns(imb).with_columns(imb_bin)
    df = df.with_columns(pl.col("ts_event").diff().over("date").alias("dt"))
    df = df.with_columns(pl.col("P_1").add(pl.col("P_-1")).truediv(2).alias("mid"))
    df = df.with_columns((pl.col("P_1") - pl.col("P_-1")).alias("spread"))
    return df


def compute_eta(lf: pl.LazyFrame, date_col: str = "date", mid_col: str = "mid") -> pl.DataFrame:
    """Compute eta per trading day (streaming)."""
    return (
        lf.select(date_col, mid_col)
        .with_columns(pl.col(mid_col).diff().over(date_col).alias("mid_diff"))
        .with_columns(
            pl.col("mid_diff").sign().alias("sign"),
            pl.col("mid_diff").sign().shift(1).over(date_col).alias("prev_sign"),
        )
        .filter(
            pl.col("mid_diff").ne(0)
            & pl.col("prev_sign").is_not_null()
            & pl.col("prev_sign").ne(0)
        )
        .with_columns(
            (pl.col("sign") == pl.col("prev_sign")).alias("is_continuation"),
            (pl.col("sign") != pl.col("prev_sign")).alias("is_alternation"),
        )
        .group_by(date_col)
        .agg(
            pl.col("is_continuation").sum().alias("n_continuations"),
            pl.col("is_alternation").sum().alias("n_alternations"),
        )
        .with_columns((pl.col("n_continuations") / (2 * pl.col("n_alternations"))).alias("eta"))
        .collect(engine="streaming")
    )


def compute_volatility(lf: pl.LazyFrame) -> pl.DataFrame:
    """Compute volatility per trading day in USD/sqrt(hour)."""
    return (
        lf.select("date", "mid")
        .with_columns((pl.col("mid") * TICK_TO_USD).diff().over("date").alias("mid_diff"))
        .filter(pl.col("mid_diff").is_not_null())
        .group_by("date")
        .agg((pl.col("mid_diff") ** 2).sum().alias("realized_var"))
        .with_columns(
            (pl.col("realized_var") / HOURS_PER_DAY).sqrt().alias("volatility_per_hour")
        )
        .collect(engine="streaming")
    )


def compute_event_probas(lf: pl.LazyFrame, event_q: int | None, is_empirical: bool = True) -> pl.DataFrame:
    """Compute event probabilities by imbalance bin.

    Args:
        lf: LazyFrame with event data
        event_q: Queue level (-1, -2) or None for spread>=2
        is_empirical: True for empirical data (different event names)
    """
    if event_q is not None:
        # Spread=1, specific queue level
        if is_empirical:
            # Empirical data has event_q column
            # Map various event names to canonical names
            lf = lf.with_columns(
                pl.col("event").replace({"Trd": "Trade", "Cxl": "Cancel", "Can": "Cancel"}).alias("event_mapped")
            )
            df = (
                lf.filter(pl.col("spread").eq(1) & pl.col("event_q").eq(event_q))
                .group_by(["imb_bin", "event_mapped"])
                .len()
                .collect(engine="streaming")
            )
            df = df.rename({"event_mapped": "event"})
        else:
            # Simulation: compute event_q from price and best bid/ask
            # event_side is numeric in simulation: -1 for bid, 1 for ask
            # For bid side: event_q = price - P_-1 - 1 (e.g., at P_-1: -1, at P_-2: -2)
            # For ask side: event_q = price - P_1 + 1 (e.g., at P_1: 1, at P_2: 2)
            lf = lf.with_columns(
                pl.when(pl.col("event_side").eq(-1))
                .then(pl.col("price") - pl.col("P_-1") - 1)
                .otherwise(pl.col("price") - pl.col("P_1") + 1)
                .alias("event_q_calc")
            )
            df = (
                lf.filter(pl.col("spread").eq(1) & pl.col("event_q_calc").eq(event_q))
                .group_by(["imb_bin", "event"])
                .len()
                .collect(engine="streaming")
            )
    else:
        # Spread>=2, create events only - simulation doesn't have these
        if is_empirical:
            df = (
                lf.filter(
                    pl.col("event").is_in(["Create_Bid", "Create_Ask"]) & pl.col("spread").ge(2)
                )
                .group_by(["imb_bin", "event"])
                .len()
                .collect(engine="streaming")
            )
        else:
            return None

    if df.is_empty():
        return None

    # Normalize to probabilities within each imb_bin
    df = df.with_columns(
        pl.col("len").truediv(pl.col("len").sum().over("imb_bin")).alias("proba")
    )
    return df


def plot_page1(lf_emp: pl.LazyFrame, lf_sim: pl.LazyFrame, sim_label: str):
    """Page 1: Microstructure statistics (3x2 grid)."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.5))  # Letter portrait
    fig.suptitle(f"Microstructure: Empirical vs {sim_label}", fontsize=12, fontweight="bold")

    # 1. Delta t distribution
    ax = axes[0, 0]
    dt_emp = (
        lf_emp.filter(pl.col("dt").gt(0))
        .select(pl.col("dt").log10().alias("dt_log"))
        .collect(engine="streaming")
    )
    dt_sim = (
        lf_sim.filter(pl.col("dt").gt(0))
        .select(pl.col("dt").log10().alias("dt_log"))
        .collect(engine="streaming")
    )
    ax.hist(dt_emp["dt_log"], bins=80, label="Empirical", density=True, alpha=0.7, color="tab:blue")
    ax.hist(dt_sim["dt_log"], bins=60, label="Simulation", density=True, alpha=0.7, color="tab:orange")
    ax.set_xlabel(r"$\log_{10}(\Delta t)$ [ns]")
    ax.set_ylabel("Density")
    ax.set_title(r"Inter-event time $\Delta t$")
    ax.legend()

    # 2. Inter-trade time
    ax = axes[0, 1]
    dt_trades_emp = (
        lf_emp.filter(pl.col("event").eq("Trd"))
        .with_columns(pl.col("ts_event").diff().over("date").cast(int).alias("dt_trade"))
        .filter(pl.col("dt_trade").gt(0))
        .select(pl.col("dt_trade").log10().alias("dt_log"))
        .collect(engine="streaming")
    )
    dt_trades_sim = (
        lf_sim.filter(pl.col("event").eq("Trade"))
        .with_columns(pl.col("ts_event").diff().over("date").cast(int).alias("dt_trade"))
        .filter(pl.col("dt_trade").gt(0))
        .select(pl.col("dt_trade").log10().alias("dt_log"))
        .collect(engine="streaming")
    )
    ax.hist(dt_trades_emp["dt_log"], bins=60, label="Empirical", density=True, alpha=0.7, color="tab:blue")
    ax.hist(dt_trades_sim["dt_log"], bins=40, label="Simulation", density=True, alpha=0.7, color="tab:orange")
    ax.set_xlabel(r"$\log_{10}(\Delta t_{\mathrm{trade}})$ [ns]")
    ax.set_ylabel("Density")
    ax.set_title("Inter-trade time")
    ax.legend()

    # 3. Event type distribution
    ax = axes[1, 0]
    evt_emp = lf_emp.select("event").collect(engine="streaming")["event"].value_counts(normalize=True).sort("event")
    evt_sim = lf_sim.select("event").collect(engine="streaming")["event"].value_counts(normalize=True).sort("event")

    # Map between empirical and simulation event names
    emp_to_sim = {"Trd": "Trade", "Add": "Add", "Cancel": "Cancel", "Cxl": "Cancel"}
    sim_to_display = {"Trade": "Trade", "Add": "Add", "Cancel": "Cancel"}

    # Use simulation event names as canonical labels
    sim_events = evt_sim["event"].to_list()
    labels = sim_events
    x = np.arange(len(labels))
    width = 0.35

    # Get simulation proportions
    sim_props = evt_sim["proportion"].to_list()

    # Match empirical events to simulation labels
    emp_props = []
    for sim_lbl in labels:
        # Find matching empirical label(s)
        emp_lbls = [k for k, v in emp_to_sim.items() if v == sim_lbl]
        prop = 0.0
        for emp_lbl in emp_lbls:
            row = evt_emp.filter(pl.col("event") == emp_lbl)
            if len(row) > 0:
                prop += row["proportion"][0]
        emp_props.append(prop)

    ax.bar(x - width/2, emp_props, width, label="Empirical", alpha=0.8, color="tab:blue")
    ax.bar(x + width/2, sim_props, width, label="Simulation", alpha=0.8, color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title("Event type distribution")
    ax.legend()

    # 4. Imbalance distribution
    ax = axes[1, 1]
    imb_emp = (
        lf_emp.select("imb_bin")
        .filter(pl.col("imb_bin").is_not_null())
        .collect(engine="streaming")["imb_bin"]
        .value_counts(normalize=True)
        .sort("imb_bin")
    )
    imb_sim = (
        lf_sim.select("imb_bin")
        .filter(pl.col("imb_bin").is_not_null())
        .collect(engine="streaming")["imb_bin"]
        .value_counts(normalize=True)
        .sort("imb_bin")
    )
    x = imb_emp["imb_bin"]
    width = 0.04
    ax.bar(x - width/2, imb_emp["proportion"], width, label="Empirical", alpha=0.8, color="tab:blue")
    ax.bar(imb_sim["imb_bin"] + width/2, imb_sim["proportion"], width, label="Simulation", alpha=0.8, color="tab:orange")
    ax.set_xlabel("Imbalance bin")
    ax.set_ylabel("Proportion")
    ax.set_title("Order book imbalance")
    ax.legend()

    # 5. Spread distribution
    ax = axes[2, 0]
    spread_emp = (
        lf_emp.select("spread")
        .collect(engine="streaming")["spread"]
        .value_counts(normalize=True)
        .sort("spread")
    )
    spread_sim = (
        lf_sim.select("spread")
        .collect(engine="streaming")["spread"]
        .value_counts(normalize=True)
        .sort("spread")
    )
    x = spread_emp["spread"]
    width = 0.35
    ax.bar(x - width/2, spread_emp["proportion"], width, label="Empirical", alpha=0.8, color="tab:blue")
    ax.bar(spread_sim["spread"] + width/2, spread_sim["proportion"], width, label="Simulation", alpha=0.8, color="tab:orange")
    ax.set_xlabel("Spread [ticks]")
    ax.set_ylabel("Proportion")
    ax.set_title("Spread distribution")
    ax.legend()

    # 6. Pre-trade spread
    ax = axes[2, 1]
    pretrade_emp = (
        lf_emp.filter(pl.col("event").eq("Trd"))
        .select("spread")
        .collect(engine="streaming")["spread"]
        .value_counts(normalize=True)
        .sort("spread")
    )
    pretrade_sim = (
        lf_sim.filter(pl.col("event").eq("Trade"))
        .select("spread")
        .collect(engine="streaming")["spread"]
        .value_counts(normalize=True)
        .sort("spread")
    )
    x = pretrade_emp["spread"]
    width = 0.35
    ax.bar(x - width/2, pretrade_emp["proportion"], width, label="Empirical", alpha=0.8, color="tab:blue")
    ax.bar(pretrade_sim["spread"] + width/2, pretrade_sim["proportion"], width, label="Simulation", alpha=0.8, color="tab:orange")
    ax.set_xlabel("Spread [ticks]")
    ax.set_ylabel("Proportion")
    ax.set_title("Pre-trade spread")
    ax.legend()

    plt.tight_layout()
    return fig


def plot_page2(lf_emp: pl.LazyFrame, lf_sim: pl.LazyFrame, sim_label: str):
    """Page 2: Aggregate statistics (3x2 grid)."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.5))  # Letter portrait
    fig.suptitle(f"Aggregate Statistics: Empirical vs {sim_label}", fontsize=12, fontweight="bold")

    # 1. Volatility distribution (twin y-axes)
    ax1 = axes[0, 0]
    vol_emp = compute_volatility(lf_emp)
    vol_sim = compute_volatility(lf_sim)
    ax1.hist(vol_emp["volatility_per_hour"], bins=30, label="Empirical", density=True, alpha=0.7, color="tab:blue")
    ax1.set_ylabel("Empirical density")
    ax2 = ax1.twinx()
    ax2.hist(vol_sim["volatility_per_hour"], bins=20, label="Simulation", density=True, alpha=0.7, color="tab:orange")
    ax2.set_ylabel("Simulation density")
    ax1.set_xlabel(r"Volatility (USD / $\sqrt{\mathrm{hour}}$)")
    ax1.set_title(r"Daily volatility: $\sigma = \sqrt{\sum_i (\Delta P_i)^2 / T}$")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # 2. Eta distribution (twin y-axes)
    ax1 = axes[0, 1]
    eta_emp = compute_eta(lf_emp)
    eta_sim = compute_eta(lf_sim)
    ax1.hist(eta_emp["eta"], bins=30, label=f"Emp (μ={eta_emp['eta'].mean():.3f})", density=True, alpha=0.7, color="tab:blue")
    ax1.set_ylabel("Empirical density")
    ax2 = ax1.twinx()
    ax2.hist(eta_sim["eta"], bins=20, label=f"Sim (μ={eta_sim['eta'].mean():.3f})", density=True, alpha=0.7, color="tab:orange")
    ax2.set_ylabel("Simulation density")
    ax1.set_xlabel(r"$\eta$")
    ax1.set_title(r"Mean reversion: $\eta = N_{\mathrm{cont}} / 2N_{\mathrm{alt}}$")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # 3. Daily event count (twin y-axes)
    ax1 = axes[1, 0]
    daily_emp = lf_emp.group_by("date").len().collect(engine="streaming")
    daily_sim = lf_sim.group_by("date").len().collect(engine="streaming")
    ax1.hist(daily_emp.with_columns(pl.col("len").log10())["len"], bins=50,
             label="Empirical", density=True, alpha=0.7, color="tab:blue")
    ax1.set_ylabel("Empirical density")
    ax2 = ax1.twinx()
    ax2.hist(daily_sim.with_columns(pl.col("len").log10())["len"], bins=20,
             label="Simulation", density=True, alpha=0.7, color="tab:orange")
    ax2.set_ylabel("Simulation density")
    ax1.set_xlabel(r"$\log_{10}(\text{events per day})$")
    ax1.set_title("Daily event count")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # 4. Hourly volume (twin y-axes)
    ax1 = axes[1, 1]
    vol_h_emp = (
        lf_emp.filter(pl.col("event").eq("Trd"))
        .group_by(pl.col("date"), pl.col("ts_event").dt.hour())
        .agg(pl.col("event_size").truediv(500).ceil().cast(int).sum())
        .collect(engine="streaming")["event_size"]
    )
    vol_h_sim = (
        lf_sim.filter(pl.col("event").eq("Trade"))
        .group_by(pl.col("date"), pl.col("ts_event") // HOUR_NS)
        .agg(pl.col("event_size").sum())
        .collect(engine="streaming")["event_size"]
    )
    ax1.hist(vol_h_emp, bins=50, label="Empirical", density=True, alpha=0.7, color="tab:blue")
    ax1.set_ylabel("Empirical density")
    ax2 = ax1.twinx()
    ax2.hist(vol_h_sim, bins=20, label="Simulation", density=True, alpha=0.7, color="tab:orange")
    ax2.set_ylabel("Simulation density")
    ax1.set_xlabel("Hourly volume [median event size]")
    ax1.set_title("Hourly traded volume")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # 5. Q_1 (best ask) queue size
    ax = axes[2, 0]
    q1_emp = (
        lf_emp.select(pl.col("Q_1").truediv(500).ceil().cast(int))
        .collect(engine="streaming")["Q_1"]
        .value_counts(normalize=True)
        .sort("Q_1")
        .filter(pl.col("Q_1").le(15))
    )
    q1_sim = (
        lf_sim.select("Q_1")
        .collect(engine="streaming")["Q_1"]
        .value_counts(normalize=True)
        .sort("Q_1")
        .filter(pl.col("Q_1").le(15))
    )
    width = 0.35
    ax.bar(q1_emp["Q_1"] - width/2, q1_emp["proportion"], width, label="Empirical", alpha=0.7, color="tab:blue")
    ax.bar(q1_sim["Q_1"] + width/2, q1_sim["proportion"], width, label="Simulation", alpha=0.7, color="tab:orange")
    ax.set_xlabel("Queue size")
    ax.set_ylabel("Proportion")
    ax.set_title("Best ask queue size ($Q_1$)")
    ax.legend()

    # 6. Q_-1 (best bid) queue size
    ax = axes[2, 1]
    qm1_emp = (
        lf_emp.select(pl.col("Q_-1").truediv(500).ceil().cast(int))
        .collect(engine="streaming")["Q_-1"]
        .value_counts(normalize=True)
        .sort("Q_-1")
        .filter(pl.col("Q_-1").le(15))
    )
    qm1_sim = (
        lf_sim.select("Q_-1")
        .collect(engine="streaming")["Q_-1"]
        .value_counts(normalize=True)
        .sort("Q_-1")
        .filter(pl.col("Q_-1").le(15))
    )
    width = 0.35
    ax.bar(qm1_emp["Q_-1"] - width/2, qm1_emp["proportion"], width, label="Empirical", alpha=0.7, color="tab:blue")
    ax.bar(qm1_sim["Q_-1"] + width/2, qm1_sim["proportion"], width, label="Simulation", alpha=0.7, color="tab:orange")
    ax.set_xlabel("Queue size")
    ax.set_ylabel("Proportion")
    ax.set_title("Best bid queue size ($Q_{-1}$)")
    ax.legend()

    plt.tight_layout()
    return fig


def plot_page3(lf_emp: pl.LazyFrame, lf_sim: pl.LazyFrame, ticker: str):
    """Page 3: Event probability differences (heatmaps)."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
    fig.suptitle(f"{ticker}: Simulated vs Empirical Event Probabilities (Bid Side)", fontsize=12, fontweight="bold")

    configs = [
        ("Best level (q=-1)", -1, ["Add", "Cancel", "Trade"]),
        ("Second level (q=-2)", -2, ["Add", "Cancel"]),
    ]

    for ax, (title, event_q, events) in zip(axes, configs):
        # Compute probabilities for both datasets
        probs_emp = compute_event_probas(lf_emp, event_q, is_empirical=True)
        probs_sim = compute_event_probas(lf_sim, event_q, is_empirical=False)

        if probs_emp is None or probs_sim is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue

        # Get all imbalance bins
        imb_bins = sorted(set(probs_emp["imb_bin"].to_list()) | set(probs_sim["imb_bin"].to_list()))

        # Build difference matrix: rows=events, cols=imb_bins
        diff_matrix = []
        for event in events:
            row = []
            for imb in imb_bins:
                emp_row = probs_emp.filter((pl.col("imb_bin") == imb) & (pl.col("event") == event))
                sim_row = probs_sim.filter((pl.col("imb_bin") == imb) & (pl.col("event") == event))
                p_emp = emp_row["proba"][0] if len(emp_row) > 0 else 0.0
                p_sim = sim_row["proba"][0] if len(sim_row) > 0 else 0.0
                row.append(p_sim - p_emp)
            diff_matrix.append(row)

        # Plot heatmap
        diff_array = np.array(diff_matrix)
        vmax = max(abs(diff_array.min()), abs(diff_array.max()), 0.01)
        im = ax.imshow(diff_array, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

        # Labels
        ax.set_xticks(range(len(imb_bins)))
        ax.set_xticklabels([f"{x:.1f}" for x in imb_bins], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(events)))
        ax.set_yticklabels(events)
        ax.set_xlabel("Imbalance bin")
        ax.set_ylabel("Event type")
        ax.set_title(title)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(r"$P_{sim} - P_{emp}$")

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate QR fit report for a ticker")
    parser.add_argument("ticker", type=str, help="Ticker symbol (e.g., PFE)")
    parser.add_argument("simulation_file", type=str, help="Path to simulation parquet file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PDF path (default: figures/fit_report_<ticker>.pdf)",
    )
    args = parser.parse_args()

    ticker = args.ticker
    sim_path = Path(args.simulation_file)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"figures/fit_report_{ticker}.pdf")

    # Parse simulation parameters from filename
    sim_params = parse_sim_params(sim_path.name)
    sim_label = sim_params["label"]

    print(f"Ticker: {ticker}")
    print(f"Simulation: {sim_path.name}")
    print(f"Parameters: {sim_label}")
    print()

    print("Loading empirical data (lazy)...")
    lf_emp = load_empirical(ticker)

    print(f"Loading simulation data (lazy)...")
    lf_sim = load_simulation(sim_path)

    print("Generating plots...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        print("  + Page 1: Microstructure")
        fig1 = plot_page1(lf_emp, lf_sim, sim_label)
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        print("  + Page 2: Aggregate Statistics")
        fig2 = plot_page2(lf_emp, lf_sim, sim_label)
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        print("  + Page 3: Event Probabilities")
        fig3 = plot_page3(lf_emp, lf_sim, sim_label)
        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)

    print(f"\nDone! Report saved to {output_path}")


if __name__ == "__main__":
    main()
