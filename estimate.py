#!/usr/bin/env python3
"""
Queue-Reactive Model Parameter Estimation

Estimates event probabilities, intensities, and size distributions
from order book data for a given ticker.
"""

import argparse
from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import polars as pl

from lobib import DataLoader

# Use science plots style if available
try:
    import scienceplots
    plt.style.use(["science", "no-latex", "grid"])
except ImportError:
    plt.style.use("seaborn-v0_8-whitegrid")


def pl_select(condlist: list[pl.Expr], choicelist: list[pl.Expr]) -> pl.Expr:
    """Polars equivalent of np.select - cascading when/then."""
    return reduce(
        lambda expr, cond_choice: expr.when(cond_choice[0]).then(cond_choice[1]),
        zip(condlist, choicelist),
        pl.when(condlist[0]).then(choicelist[0]),
    )


def load_raw_data(loader: DataLoader, ticker: str) -> pl.DataFrame:
    """Load raw order book data for a ticker."""
    info = loader.ticker_info(ticker)
    df = loader.load(
        ticker,
        start_date=info["date"].min(),
        end_date=info["date"].max(),
        schema="qr",
        eager=True,
    ).sort(["date", "ts_event"])
    return df


def compute_median_event_sizes(df: pl.DataFrame) -> dict[int, float]:
    """Compute median event sizes for Q_1 through Q_4 from raw data.

    For each queue level i, combines events from Q_i and Q_{-i} (symmetry)
    and computes the median event size.

    Returns:
        Dictionary mapping queue level (1-4) to median event size.
    """
    # Compute event_q (queue position relative to best bid/ask)
    df_with_q = df.with_columns(
        pl.when(pl.col("event_queue_nbr").lt(0))
        .then(pl.col("event_queue_nbr").sub(pl.col("best_bid_nbr")).sub(1))
        .otherwise(pl.col("event_queue_nbr").sub(pl.col("best_ask_nbr")).add(1))
        .alias("event_q")
    )

    # Compute median for each |event_q| level (combining Q_i and Q_{-i})
    median_sizes = {}
    for q in range(1, 5):
        sizes = df_with_q.filter(pl.col("event_q").abs().eq(q))["event_size"]
        if len(sizes) > 0:
            median_sizes[q] = float(sizes.median())
        else:
            median_sizes[q] = 100.0  # fallback default

    return median_sizes


def preprocess(df: pl.DataFrame, median_sizes: dict[int, float]) -> pl.DataFrame:
    """Preprocess raw order book data for estimation."""
    # Filter valid events (removed spread <= 4 filter)
    df = df.filter(
        pl.col("event_side")
        .replace({"A": 1, "B": -1})
        .cast(int)
        .mul(pl.col("event_queue_nbr"))
        >= 0
    )
    df = df.with_columns(pl.col("event").replace({"Trd_All": "Trd"}))

    # Compute event queue position relative to best bid/ask
    df = df.with_columns(
        pl.when(pl.col("event_queue_nbr").lt(0))
        .then(pl.col("event_queue_nbr").sub(pl.col("best_bid_nbr")).sub(1))
        .otherwise(pl.col("event_queue_nbr").sub(pl.col("best_ask_nbr")).add(1))
        .alias("event_q")
    )

    # Compute imbalance (normalize by median event size for Q_1)
    q1_median = median_sizes[1]
    condlist = [pl.col("best_bid_nbr").eq(-i) for i in range(1, 11)]
    choicelist = [pl.col(f"Q_{-i}") for i in range(1, 11)]
    best_bid = pl_select(condlist, choicelist).alias("best_bid").truediv(q1_median).ceil()

    condlist = [pl.col("best_ask_nbr").eq(i) for i in range(1, 11)]
    choicelist = [pl.col(f"Q_{i}") for i in range(1, 11)]
    best_ask = pl_select(condlist, choicelist).alias("best_ask").truediv(q1_median).ceil()
    imb = ((best_bid - best_ask) / (best_bid + best_ask)).alias("imb")

    df = df.with_columns(imb)

    # Bin imbalance
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
    df = df.with_columns(pl_select(condlist, choicelist).alias("imb_bin"))

    return df


def event_probas_spread1(df: pl.DataFrame) -> pl.DataFrame:
    """Compute event probabilities for spread=1."""
    stat = df.group_by(["imb_bin", "spread", "event", "event_side", "event_q"]).agg(
        pl.len()
    )
    stat = stat.filter(pl.col("spread").eq(1) & pl.col("event_q").is_between(-2, 2))
    stat = stat.with_columns(
        pl.col("len").truediv(pl.col("len").sum().over("imb_bin")).alias("proba")
    ).sort(["event", "event_side", "imb_bin"])

    stat1 = (
        stat.with_columns(
            pl.col("imb_bin")
            .sign()
            .mul(pl.col("event_side").replace({"A": 1, "B": -1}).cast(int))
            .alias("sign"),
            pl.col("imb_bin").abs(),
            pl.col("event_q").abs(),
        )
        .group_by(["imb_bin", "spread", "event", "event_q", "sign"])
        .agg(pl.col("len").sum())
        .with_columns(
            pl.col("sign")
            .cast(str)
            .replace({"1.0": "A", "0.0": "A", "-1.0": "B"})
            .alias("event_side")
        )
    )

    stat2 = stat1.with_columns(
        -pl.col("imb_bin"), pl.col("event_side").replace({"A": "B", "B": "A"})
    )

    stat = pl.concat((stat1, stat2)).with_columns(
        proba=pl.col("len").truediv(pl.col("len").sum().over("imb_bin"))
    )

    stat = stat.with_columns(
        pl.col("event_q").mul(pl.col("event_side").replace({"A": 1, "B": -1}).cast(int))
    )
    stat = stat.sort(["imb_bin", "spread", "event"])

    return stat.drop("sign").select(
        "imb_bin", "spread", "event", "event_q", "len", "event_side", "proba"
    )


def event_probas_spread2(df: pl.DataFrame) -> pl.DataFrame:
    """Compute event probabilities for spread>=2 (Create events only)."""
    stat = (
        df.filter(
            pl.col("event").is_in(["Create_Bid", "Create_Ask"]) & pl.col("spread").ge(2)
        )
        .with_columns(spread=2)
        .group_by(["imb_bin", "spread", "event", "event_side"])
        .len()
    )
    stat = stat.filter(
        pl.col("event")
        .replace({"Create_Ask": 1, "Create_Bid": -1})
        .cast(int)
        .mul(pl.col("event_side").replace({"A": 1, "B": -1}).cast(int))
        .gt(0)
    )
    stat1 = (
        stat.with_columns(
            pl.col("event_side")
            .replace({"A": 1, "B": -1})
            .cast(int)
            .mul(pl.col("imb_bin").sign())
            .alias("sign"),
            pl.col("imb_bin").abs(),
        )
        .group_by(["imb_bin", "spread", "sign"])
        .agg(pl.col("len").sum())
        .with_columns(
            pl.col("sign")
            .cast(str)
            .replace(
                {
                    "1.0": "Create_Ask",
                    "-0.0": "Create_Ask",
                    "0.0": "Create_Ask",
                    "-1.0": "Create_Bid",
                }
            )
            .alias("event")
        )
    )
    stat2 = stat1.with_columns(
        -pl.col("imb_bin"),
        pl.col("event").replace(
            {"Create_Ask": "Create_Bid", "Create_Bid": "Create_Ask"}
        ),
    )

    stat = pl.concat([stat1, stat2])
    stat = stat.with_columns(
        pl.col("len").truediv(pl.col("len").sum().over("imb_bin")).alias("proba")
    ).sort("imb_bin")

    return (
        stat.with_columns(
            event_q=pl.lit(0).cast(pl.Int64),
            event_side=pl.col("event").replace({"Create_Bid": "B", "Create_Ask": "A"}),
        )
        .drop("sign")
        .select("imb_bin", "spread", "event", "event_q", "len", "event_side", "proba")
    )


def estimate_event_probabilities(df: pl.DataFrame) -> pl.DataFrame:
    """Estimate event probabilities by state."""
    return pl.concat([event_probas_spread1(df), event_probas_spread2(df)]).sort(
        "imb_bin", "event", "event_side"
    )


def estimate_intensities(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Estimate inter-arrival times (lambda) by imbalance bin.

    Returns:
        Tuple of (aggregated intensities, raw dt data for plotting)
    """
    df_ = df.filter(pl.col("event_q").abs().le(2)).select(
        "date",
        "ts_event",
        "imb_bin",
        pl.when(pl.col("spread").eq(1)).then(1).otherwise(2).alias("spread"),
    )
    df_ = df_.with_columns(
        pl.col("ts_event").diff().over("date").alias("dt").cast(int)
    ).filter(pl.col("dt").gt(0))

    lamda = df_.group_by("imb_bin", "spread").agg(pl.col("dt").mean())

    # Symmetrize
    lamda = (
        lamda.with_columns(pl.col("imb_bin").abs())
        .group_by("imb_bin", "spread")
        .agg(pl.col("dt").mean())
    )
    lamda = pl.concat(
        (lamda, lamda.filter(pl.col("imb_bin").ne(0)).with_columns(-pl.col("imb_bin")))
    ).sort("imb_bin")

    return lamda, df_


def fit_geometric(sizes: np.ndarray, probs: np.ndarray) -> float:
    """Fit geometric distribution by matching mean. p = 1/mean."""
    mean = (sizes * probs).sum()
    return 1 / mean


def geometric_pmf(k: np.ndarray, p: float) -> np.ndarray:
    """P(X=k) for geometric starting at k=1."""
    return p * (1 - p) ** (k - 1)


def estimate_size_distributions(df: pl.DataFrame, median_sizes: dict[int, float]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Estimate geometric distribution parameters for event sizes.

    Returns:
        Tuple of (fitted parameters, raw size statistics for plotting)
    """
    q1_median = median_sizes[1]
    q2_median = median_sizes[2]
    stat = (
        df.filter(pl.col("spread").eq(1) & pl.col("event_q").abs().le(2))
        .select(
            "imb_bin",
            "event",
            pl.when(pl.col("event_q").abs().eq(1))
            .then(pl.col("event_size").truediv(q1_median).ceil())
            .otherwise(pl.col("event_size").truediv(q2_median).ceil())
            .cast(int)
            .alias("event_size"),
            "event_side",
            "event_q",
        )
        .filter(pl.col("event_size").le(50) & pl.col("event_size").ge(1))
    )

    # Group by state + size to get counts
    stat = stat.group_by(["imb_bin", "event", "event_size", "event_q"]).len()

    # Symmetrize
    stat = (
        stat.with_columns(
            pl.col("event_q").sign().mul(pl.col("imb_bin").sign()).alias("sign")
        )
        .group_by(
            pl.col("imb_bin").abs(), "event", "sign", "event_size", pl.col("event_q").abs()
        )
        .agg(pl.col("len").sum())
        .with_columns(
            pl.col("sign")
            .cast(str)
            .replace({"1.0": "A", "0.0": "A", "-1.0": "B"})
            .alias("event_side")
        )
        .drop("sign")
    )

    # Mirror to negative imbalances
    stat = pl.concat(
        (
            stat,
            stat.with_columns(
                -pl.col("imb_bin"), pl.col("event_side").replace({"A": "B", "B": "A"})
            ),
        )
    )

    # Compute probability: P(size | imb_bin, event, event_q)
    stat = stat.with_columns(
        pl.col("len")
        .truediv(pl.col("len").sum().over(["imb_bin", "event", "event_q"]))
        .alias("proba")
    ).sort(["imb_bin", "event", "event_q", "event_size"])

    # Fit geometric distribution for each state
    results = []
    for (imb_bin, event, event_q), group in stat.group_by(["imb_bin", "event", "event_q"]):
        group = group.sort("event_size")
        sizes = group["event_size"].to_numpy()
        probs = group["proba"].to_numpy()

        if len(sizes) > 0 and probs.sum() > 0:
            p = fit_geometric(sizes, probs)
            results.append({
                "imb_bin": imb_bin,
                "event": event,
                "event_q": event_q,
                "p": p,
                "spread": 1,
            })

    return pl.DataFrame(results).sort(["imb_bin", "event", "event_q"]), stat


def estimate_burst_sizes(raw_df: pl.DataFrame, median_sizes: dict[int, float]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Estimate burst size distributions for Create events (spread >= 2).

    Args:
        raw_df: Raw (unfiltered) dataframe for burst analysis
        median_sizes: Median event sizes by queue level

    Returns:
        Tuple of (fitted parameters, burst sizes data for plotting)
    """
    q1_median = median_sizes[1]

    # Compute imbalance (normalize by Q_1 median)
    condlist = [pl.col("best_bid_nbr").eq(-i) for i in range(1, 11)]
    choicelist = [pl.col(f"Q_{-i}") for i in range(1, 11)]
    best_bid = pl_select(condlist, choicelist).alias("best_bid").truediv(q1_median).ceil()

    condlist = [pl.col("best_ask_nbr").eq(i) for i in range(1, 11)]
    choicelist = [pl.col(f"Q_{i}") for i in range(1, 11)]
    best_ask = pl_select(condlist, choicelist).alias("best_ask").truediv(q1_median).ceil()
    imb = ((best_bid - best_ask) / (best_bid + best_ask)).alias("imb")

    df = raw_df.with_columns(imb)

    # Bin imbalance
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
    df = df.with_columns(pl_select(condlist, choicelist).alias("imb_bin"))

    # Compute event_q
    df = df.with_columns(
        pl.when(pl.col("event_queue_nbr").lt(0))
        .then(pl.col("event_queue_nbr").sub(pl.col("best_bid_nbr")).sub(1))
        .otherwise(pl.col("event_queue_nbr").sub(pl.col("best_ask_nbr")).add(1))
        .alias("event_q")
    )
    df = df.filter(pl.col("event_q").abs().le(2))

    # Group events into bursts
    df = df.with_columns([
        pl.col("event").is_in(["Create_Bid", "Create_Ask"]).alias("is_create"),
        (pl.col("price") != pl.col("price").shift(1)).alias("price_changed"),
        (~pl.col("event").is_in(["Add", "Create_Bid", "Create_Ask"])).alias("not_add"),
        (pl.col("date") != pl.col("date").shift(1)).alias("new_day"),
    ])

    df = df.with_columns(
        (pl.col("is_create") | pl.col("price_changed") | pl.col("not_add") | pl.col("new_day"))
        .cum_sum()
        .alias("group_id")
    )

    burst_sizes = (
        df.group_by("group_id")
        .agg([
            pl.col("imb_bin").first(),
            pl.col("event").first(),
            pl.col("is_create").first().alias("starts_with_create"),
            pl.col("event_side").first().alias("side"),
            pl.col("event_size").sum().alias("total_size"),
        ])
        .filter(pl.col("starts_with_create"))
    )
    burst_sizes = burst_sizes.with_columns(
        pl.col("total_size").truediv(q1_median).ceil().cast(int)
    )

    # Filter out null imb_bin values
    burst_sizes = burst_sizes.filter(pl.col("imb_bin").is_not_null())

    # Symmetrize: pool (imb=+x, Create_Ask) with (imb=-x, Create_Bid)
    # sign = sign(imb_bin) * sign(event) where Create_Ask=+1, Create_Bid=-1
    burst_sizes = burst_sizes.with_columns(
        pl.col("imb_bin")
        .sign()
        .mul(pl.col("event").replace({"Create_Ask": 1, "Create_Bid": -1}).cast(int))
        .alias("sign")
    )

    # Fit geometric for each (|imb_bin|, sign) then mirror
    fit_results = []
    imb_bins_abs = sorted(burst_sizes["imb_bin"].abs().unique().to_list())

    for imb_bin_abs in imb_bins_abs:
        for sign_val, event in [(1, "Create_Ask"), (-1, "Create_Bid")]:
            # Pool symmetric states: (imb=+x, Create_Ask) with (imb=-x, Create_Bid)
            data = burst_sizes.filter(
                (pl.col("imb_bin").abs() == imb_bin_abs) & (pl.col("sign") == sign_val)
            )["total_size"].to_numpy()

            if len(data) < 10:
                continue

            mean_size = data.mean()
            p_fit = 1 / mean_size

            # Add for positive imbalance
            fit_results.append({
                "imb_bin": imb_bin_abs,
                "event": event,
                "event_q": 0,
                "p": p_fit,
                "spread": 2,
            })
            # Mirror to negative imbalance (swap event)
            if imb_bin_abs != 0:
                mirror_event = "Create_Bid" if event == "Create_Ask" else "Create_Ask"
                fit_results.append({
                    "imb_bin": -imb_bin_abs,
                    "event": mirror_event,
                    "event_q": 0,
                    "p": p_fit,
                    "spread": 2,
                })

    return pl.DataFrame(fit_results).sort(["imb_bin", "event"]), burst_sizes


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_imbalance_distribution(df: pl.DataFrame, ticker: str) -> plt.Figure:
    """Plot the distribution of imbalance bins."""
    fig, ax = plt.subplots(figsize=(8, 5))
    stat = df.group_by("imb_bin").len().sort("imb_bin")

    ax.plot(stat["imb_bin"], stat["len"], alpha=0.8, ms=5, marker="o", mec="k", lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Imbalance Bin")
    ax.set_ylabel("Count")
    ax.set_title(f"{ticker} - Imbalance Bin Distribution")

    plt.tight_layout()
    return fig


def plot_spread_distribution(df: pl.DataFrame, ticker: str) -> plt.Figure:
    """Plot the distribution of spread (in ticks) - all events and trades only."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: All events
    spread_all = df.group_by("spread").len().sort("spread")
    spread_all = spread_all.filter(pl.col("spread").le(20))
    total_all = spread_all["len"].sum()
    proportions_all = spread_all["len"] / total_all
    ax1.bar(spread_all["spread"], proportions_all, edgecolor="k", alpha=0.7)
    ax1.set_xlabel("Spread (ticks)")
    ax1.set_ylabel("Proportion")
    ax1.set_title("Spread Distribution - All Events")

    # Right: Trades only
    trades_df = df.filter(pl.col("event").eq("Trd"))
    spread_trades = trades_df.group_by("spread").len().sort("spread")
    spread_trades = spread_trades.filter(pl.col("spread").le(20))
    total_trades = spread_trades["len"].sum()
    proportions_trades = spread_trades["len"] / total_trades
    ax2.bar(spread_trades["spread"], proportions_trades, edgecolor="k", alpha=0.7, color="tab:orange")
    ax2.set_xlabel("Spread (ticks)")
    ax2.set_ylabel("Proportion")
    ax2.set_title("Spread Distribution - Trades Only")

    fig.suptitle(f"{ticker} - Spread Distributions", fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def plot_median_sizes_table(median_sizes: dict[int, float], ticker: str) -> plt.Figure:
    """Plot a table showing median event sizes for Q_1 through Q_4."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")

    # Create table data
    table_data = [
        ["Queue Level", "Median Event Size"],
        ["Q_1 (|event_q| = 1)", f"{median_sizes[1]:,.0f}"],
        ["Q_2 (|event_q| = 2)", f"{median_sizes[2]:,.0f}"],
        ["Q_3 (|event_q| = 3)", f"{median_sizes[3]:,.0f}"],
        ["Q_4 (|event_q| = 4)", f"{median_sizes[4]:,.0f}"],
    ]

    table = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="center",
        colWidths=[0.5, 0.4],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header row
    for j in range(2):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(color="white", weight="bold")

    ax.set_title(f"{ticker} - Median Event Sizes by Queue Level", fontsize=12, pad=20)
    plt.tight_layout()
    return fig


def plot_event_probabilities(event_probs: pl.DataFrame, ticker: str) -> plt.Figure:
    """Plot event probabilities by imbalance for different queue positions."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # q=1 events (spread=1)
    for e in ["Add", "Can", "Trd"]:
        data = event_probs.filter(
            pl.col("event").eq(e) & pl.col("event_q").eq(-1) & pl.col("spread").eq(1)
        ).sort("imb_bin")
        if len(data) > 0:
            ax1.plot(data["imb_bin"], data["proba"], label=e, ms=4, mec="k", marker="o", lw=1.3)

    # q=2 events (spread=1)
    for e in ["Add", "Can"]:
        data = event_probs.filter(
            pl.col("event").eq(e) & pl.col("event_q").eq(-2) & pl.col("spread").eq(1)
        ).sort("imb_bin")
        if len(data) > 0:
            ax2.plot(data["imb_bin"], data["proba"], label=e, ms=4, mec="k", marker="o", lw=1.3)

    # Create events (spread>=2)
    for e in ["Create_Ask", "Create_Bid"]:
        data = event_probs.filter(
            pl.col("event").eq(e) & pl.col("spread").eq(2)
        ).sort("imb_bin")
        if len(data) > 0:
            ax3.plot(data["imb_bin"], data["proba"], label=e, ms=4, mec="k", marker="o", lw=1.3)

    ax1.set_ylabel("Probability")
    ax2.set_ylabel("Probability")
    ax3.set_ylabel("Probability")
    ax1.set_xlabel("Imbalance")
    ax2.set_xlabel("Imbalance")
    ax3.set_xlabel("Imbalance")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax1.set_title(r"P(event | Imb, spread=1, side=Bid, q=1)")
    ax2.set_title(r"P(event | Imb, spread=1, side=Bid, q=2)")
    ax3.set_title(r"P(event | Imb, spread>=2)")

    fig.suptitle(f"{ticker} - Event Probabilities", fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def plot_intensities(intensities: pl.DataFrame, dt_data: pl.DataFrame, ticker: str) -> plt.Figure:
    """Plot inter-arrival time intensities split by spread."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # Left: Mean intensity by imbalance for spread=1
    spread1 = intensities.filter(pl.col("spread").eq(1)).sort("imb_bin")
    if len(spread1) > 0:
        ax1.plot(spread1["imb_bin"], spread1["dt"], marker="o", ms=5, mec="k", lw=1.5)
    ax1.set_yscale("log")
    ax1.set_xlabel("Imbalance Bin")
    ax1.set_ylabel("Mean Inter-arrival Time (ns)")
    ax1.set_title("Spread = 1")

    # Middle: Mean intensity by imbalance for spread>=2
    spread2 = intensities.filter(pl.col("spread").eq(2)).sort("imb_bin")
    if len(spread2) > 0:
        ax2.plot(spread2["imb_bin"], spread2["dt"], marker="o", ms=5, mec="k", lw=1.5, color="tab:orange")
    ax2.set_yscale("log")
    ax2.set_xlabel("Imbalance Bin")
    ax2.set_ylabel("Mean Inter-arrival Time (ns)")
    ax2.set_title("Spread >= 2")

    # Right: Histogram of inter-arrival times (log10)
    dt_vals = dt_data["dt"].to_numpy()
    dt_vals = dt_vals[dt_vals > 0]
    ax3.hist(np.log10(dt_vals), bins=80, edgecolor="k", alpha=0.7)
    ax3.set_xlabel("log10(Inter-arrival Time)")
    ax3.set_ylabel("Count")
    ax3.set_title("Distribution of Inter-arrival Times")

    fig.suptitle(f"{ticker} - Intensities by Spread", fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def plot_size_distributions(size_stat: pl.DataFrame, ticker: str) -> plt.Figure:
    """Plot size distributions with geometric fits."""
    fig, axes = plt.subplots(3, 5, figsize=(16, 9))
    imbs = [-0.5, -0.2, 0, 0.2, 0.5]
    events = ["Add", "Can", "Trd"]

    for row, event in enumerate(events):
        for col, imb in enumerate(imbs):
            ax = axes[row, col]
            test = size_stat.filter(
                pl.col("imb_bin").eq(imb)
                & pl.col("event").eq(event)
                & pl.col("event_q").eq(1)
            ).sort("event_size")

            if test.height == 0:
                ax.set_title(f"{event}, imb={imb}\nNo data", fontsize=9)
                continue

            sizes = test["event_size"].to_numpy()
            probs = test["proba"].to_numpy()
            p_fit = fit_geometric(sizes, probs)
            mean = 1 / p_fit
            geom_probs = geometric_pmf(sizes, p_fit)

            ax.bar(sizes - 0.15, probs, width=0.3, label="Empirical", alpha=0.7)
            ax.bar(sizes + 0.15, geom_probs, width=0.3, label="Geometric", alpha=0.7)
            ax.set_title(f"{event}, imb={imb}\np={p_fit:.2f}, mean={mean:.1f}", fontsize=9)
            ax.set_xlim(0, 20)

            if row == 2:
                ax.set_xlabel("Size")
            if col == 0:
                ax.set_ylabel("P")

    axes[0, -1].legend(fontsize=8)
    fig.suptitle(f"{ticker} - Size Distributions (q=1, Geometric Fit)", fontsize=12)
    plt.tight_layout()
    return fig


def plot_burst_sizes(burst_sizes: pl.DataFrame, ticker: str) -> plt.Figure:
    """Plot burst size distributions for Create events."""
    imb_bins = sorted(burst_sizes["imb_bin"].unique().to_list())
    events = ["Create_Bid", "Create_Ask"]

    n_cols = 6
    n_rows = (len(imb_bins) + 2) // 3  # 3 imb_bins per row, 2 events each

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, imb_bin in enumerate(imb_bins):
        for j, event in enumerate(events):
            col = (i % 3) + (j * 3)
            row = i // 3

            if row >= n_rows:
                continue

            ax = axes[row, col]

            data = burst_sizes.filter(
                (pl.col("imb_bin") == imb_bin) & (pl.col("event") == event)
            )["total_size"].to_numpy()

            if len(data) < 10:
                ax.set_title(f"imb={imb_bin:.1f}, {event.split('_')[1]}\n(n={len(data)})", fontsize=8)
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
                continue

            # Compute empirical distribution
            max_size = min(50, int(data.max()))
            sizes = np.arange(1, max_size + 1)
            counts = np.bincount(np.clip(data.astype(int), 1, max_size), minlength=max_size + 1)[1:]
            probs = counts / counts.sum()

            # Fit geometric
            mean_size = data.mean()
            p_fit = 1 / mean_size
            geom_probs = geometric_pmf(sizes, p_fit)

            ax.bar(sizes - 0.2, probs, width=0.4, alpha=0.6, label="Empirical", edgecolor="k")
            ax.bar(sizes + 0.2, geom_probs, width=0.4, alpha=0.6, label=f"Geom(p={p_fit:.3f})", edgecolor="k")
            ax.set_xlim(0, 25)
            ax.set_title(f"imb={imb_bin:.1f}, {event.split('_')[1]}\n(n={len(data):,}, mean={mean_size:.1f})", fontsize=8)
            ax.tick_params(labelsize=6)

            if row == n_rows - 1:
                ax.set_xlabel("Burst size", fontsize=7)
            if col == 0 or col == 3:
                ax.set_ylabel("Probability", fontsize=7)

    # Hide unused axes
    for i in range(len(imb_bins), n_rows * 3):
        for j in range(2):
            col = (i % 3) + (j * 3)
            row = i // 3
            if row < n_rows:
                axes[row, col].axis("off")

    if len(imb_bins) > 0:
        axes[0, 0].legend(fontsize=7)

    fig.suptitle(f"{ticker} - Burst Size Distributions (Geometric Fit)", fontsize=12)
    plt.tight_layout()
    return fig


def generate_estimation_report(
    df: pl.DataFrame,
    raw_df: pl.DataFrame,
    median_sizes: dict[int, float],
    event_probs: pl.DataFrame,
    intensities: pl.DataFrame,
    dt_data: pl.DataFrame,
    size_stat: pl.DataFrame,
    burst_sizes: pl.DataFrame,
    ticker: str,
    output_path: Path,
) -> None:
    """Generate a PDF report with all estimation plots."""
    print(f"Generating estimation report: {output_path}")

    with PdfPages(output_path) as pdf:
        # Page 1: Spread distributions
        fig = plot_spread_distribution(raw_df, ticker)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: Median event sizes table
        fig = plot_median_sizes_table(median_sizes, ticker)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: Imbalance distribution
        fig = plot_imbalance_distribution(df, ticker)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 4: Event probabilities
        fig = plot_event_probabilities(event_probs, ticker)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 5: Intensities (split by spread)
        fig = plot_intensities(intensities, dt_data, ticker)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 6: Size distributions
        fig = plot_size_distributions(size_stat, ticker)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 7: Burst sizes
        if len(burst_sizes) > 0:
            fig = plot_burst_sizes(burst_sizes, ticker)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"  Saved estimation report to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate Queue-Reactive model parameters from order book data"
    )
    parser.add_argument(
        "-t", "--ticker",
        required=True,
        help="Ticker symbol to estimate parameters for"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory (default: data/<ticker>/)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating the PDF report"
    )
    args = parser.parse_args()

    ticker = args.ticker
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"data/{ticker}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Estimating QR model parameters for {ticker}")
    print(f"Output directory: {output_dir}")

    loader = DataLoader()

    # Load raw data
    print("Loading raw data...")
    raw_df = load_raw_data(loader, ticker)
    min_date = raw_df["date"].min()
    max_date = raw_df["date"].max()
    n_days = raw_df["date"].n_unique()
    avg_daily_events = len(raw_df) / n_days
    print(f"  Loaded {len(raw_df):,} raw events")
    print(f"  Date range: {min_date} to {max_date} ({n_days} days)")
    print(f"  Average daily events: {avg_daily_events:,.0f}")

    # Compute median event sizes
    print("Computing median event sizes...")
    median_sizes = compute_median_event_sizes(raw_df)
    for q in range(1, 5):
        print(f"  Q_{q}: {median_sizes[q]:,.0f}")

    # Save median sizes to CSV
    median_df = pl.DataFrame({
        "queue_level": [1, 2, 3, 4],
        "median_event_size": [median_sizes[q] for q in range(1, 5)]
    })
    median_df.write_csv(output_dir / "median_event_sizes.csv")

    # Preprocess data
    print("Preprocessing data...")
    df = preprocess(raw_df, median_sizes)
    print(f"  Preprocessed {len(df):,} events")

    # Estimate event probabilities
    print("Estimating event probabilities...")
    event_probs = estimate_event_probabilities(df)
    event_probs.write_csv(output_dir / "event_probabilities.csv")
    print(f"  Saved {len(event_probs)} rows to event_probabilities.csv")

    # Estimate intensities
    print("Estimating intensities...")
    intensities, dt_data = estimate_intensities(df)
    intensities.write_csv(output_dir / "intensities.csv")
    print(f"  Saved {len(intensities)} rows to intensities.csv")

    # Estimate size distributions (spread=1)
    print("Estimating size distributions (spread=1)...")
    size_dist, size_stat = estimate_size_distributions(df, median_sizes)

    # Estimate burst sizes (spread>=2)
    print("Estimating burst size distributions (spread>=2)...")
    burst_dist, burst_sizes = estimate_burst_sizes(raw_df, median_sizes)

    # Combine size distributions
    combined = pl.concat([size_dist, burst_dist]).sort(
        ["spread", "imb_bin", "event", "event_q"]
    )
    combined.write_csv(output_dir / "size_distrib.csv")
    print(f"  Saved {len(combined)} rows to size_distrib.csv")

    # Generate PDF report
    if not args.no_plots:
        report_path = output_dir / "estimation_report.pdf"
        generate_estimation_report(
            df, raw_df, median_sizes, event_probs, intensities, dt_data,
            size_stat, burst_sizes, ticker, report_path
        )

    print("Done!")


if __name__ == "__main__":
    main()
