#!/usr/bin/env python3
"""
Queue-Reactive Model Parameter Estimation

Estimates event probabilities, intensities, and size distributions
from order book data for a given ticker.
"""

import argparse
import gc
from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import polars as pl
from scipy.stats import gaussian_kde, norm as norm_dist, gamma as gamma_dist, weibull_min
from sklearn.mixture import GaussianMixture

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
    """Load raw order book data for a ticker.

    Args:
        loader: DataLoader instance
        ticker: Ticker symbol

    Returns:
        DataFrame with raw order book data
    """
    info = loader.ticker_info(ticker)
    df = loader.load(
        ticker,
        start_date=info["date"].min(),
        end_date=info["date"].max(),
        schema="qr",
        eager=True,
    )
    return df.sort(["date", "ts_event"])


def compute_total_lvl_quantiles(
    df: pl.DataFrame | pl.LazyFrame, median_sizes: dict[int, float], n_bins: int = 5,
    sample_size: int = 1_000_000
) -> tuple[np.ndarray, pl.DataFrame]:
    """Compute quantile edges for total best-level volume (normalized).

    Uses sampling for memory efficiency with large datasets.

    Args:
        df: Raw dataframe/lazyframe with Q_-1 and Q_1 columns
        median_sizes: Median event sizes by queue level (for normalization)
        n_bins: Number of quantile bins (default 5 = quintiles)
        sample_size: Maximum samples to use for quantile estimation (default 1M)

    Returns:
        Tuple of (quantile_edges array, stats DataFrame for plotting)
    """
    q1_median = median_sizes[1]

    # Use lazy evaluation for memory efficiency
    if isinstance(df, pl.LazyFrame):
        lazy_df = df
    else:
        lazy_df = df.lazy()

    # Compute total_lvl lazily, then sample and collect
    total_lvl_lazy = lazy_df.select(
        ((pl.col("Q_-1") + pl.col("Q_1")) / q1_median).ceil().alias("total_lvl")
    )

    # Get row count for sampling decision
    n_rows = total_lvl_lazy.select(pl.len()).collect().item()

    if n_rows > sample_size:
        # Sample for large datasets
        rng = np.random.default_rng(42)
        sample_frac = sample_size / n_rows
        total_lvl = (
            total_lvl_lazy
            .filter(pl.lit(rng.random(n_rows)) < sample_frac)
            .collect()["total_lvl"]
            .to_numpy()
        )
    else:
        total_lvl = total_lvl_lazy.collect()["total_lvl"].to_numpy()

    # Compute quantile edges
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(total_lvl, percentiles)

    # Create stats DataFrame for saving/plotting
    stats = pl.DataFrame({
        "bin": list(range(n_bins)),
        "lower": edges[:-1],
        "upper": edges[1:],
        "percentile_lower": percentiles[:-1],
        "percentile_upper": percentiles[1:],
    })

    return edges, stats


def bin_total_lvl(total_lvl: pl.Expr, edges: np.ndarray) -> pl.Expr:
    """Bin total_lvl values into quantile bins using edges.

    Args:
        total_lvl: Polars expression for total_lvl column
        edges: Array of quantile edges (n_bins + 1 values)

    Returns:
        Polars expression for total_lvl_bin (0 to n_bins-1)
    """
    n_bins = len(edges) - 1
    # Build cascading when/then for binning
    condlist = [
        total_lvl.le(edges[i + 1]) for i in range(n_bins - 1)
    ]
    choicelist = list(range(n_bins - 1))

    # Last bin catches everything above
    result = pl.lit(n_bins - 1)
    for cond, choice in zip(reversed(condlist), reversed(choicelist)):
        result = pl.when(cond).then(pl.lit(choice)).otherwise(result)

    return result


def compute_median_event_sizes(df: pl.DataFrame | pl.LazyFrame) -> dict[int, float]:
    """Compute median event sizes for Q_1 through Q_4 from raw data.

    For each queue level i, combines events from Q_i and Q_{-i} (symmetry)
    and computes the median event size.

    Uses lazy evaluation for memory efficiency.

    Returns:
        Dictionary mapping queue level (1-4) to median event size.
    """
    # Use lazy evaluation
    if isinstance(df, pl.LazyFrame):
        lazy_df = df
    else:
        lazy_df = df.lazy()

    # Compute event_q (queue position relative to best bid/ask)
    lazy_df = lazy_df.with_columns(
        pl.when(pl.col("event_queue_nbr").lt(0))
        .then(pl.col("event_queue_nbr").sub(pl.col("best_bid_nbr")).sub(1))
        .otherwise(pl.col("event_queue_nbr").sub(pl.col("best_ask_nbr")).add(1))
        .alias("event_q")
    )

    # Compute median for all queue levels in one query
    medians = (
        lazy_df
        .select("event_size", pl.col("event_q").abs().alias("abs_event_q"))
        .filter(pl.col("abs_event_q").is_between(1, 4))
        .group_by("abs_event_q")
        .agg(pl.col("event_size").median().alias("median_size"))
        .collect()
    )

    # Build result dictionary
    median_sizes = {}
    for row in medians.iter_rows(named=True):
        median_sizes[int(row["abs_event_q"])] = float(row["median_size"])

    # Fill in missing levels with fallback
    for q in range(1, 5):
        if q not in median_sizes:
            median_sizes[q] = 100.0

    return median_sizes


def preprocess(
    df: pl.DataFrame,
    median_sizes: dict[int, float],
    total_lvl_edges: np.ndarray | None = None,
) -> pl.DataFrame:
    """Preprocess raw order book data for estimation.

    Args:
        df: Raw dataframe
        median_sizes: Median event sizes by queue level
        total_lvl_edges: Optional quantile edges for total_lvl binning.
                         If provided, adds total_lvl and total_lvl_bin columns.
    """
    # Filter valid events (removed spread <= 4 filter)
    df = df.filter(
        pl.col("event_side")
        .replace({"A": 1, "B": -1})
        .cast(int)
        .mul(pl.col("event_queue_nbr"))
        >= 0
    )

    # Filter rows with best_bid_nbr and best_ask_nbr in expected range
    df = df.filter(
        pl.col("best_bid_nbr").is_between(-10, -1)
        & pl.col("best_ask_nbr").is_between(1, 10)
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
    best_bid = pl_select(condlist, choicelist).truediv(q1_median).ceil()

    condlist = [pl.col("best_ask_nbr").eq(i) for i in range(1, 11)]
    choicelist = [pl.col(f"Q_{i}") for i in range(1, 11)]
    best_ask = pl_select(condlist, choicelist).truediv(q1_median).ceil()
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

    # Add total_lvl and total_lvl_bin if edges provided
    if total_lvl_edges is not None:
        # total_lvl = best_bid + best_ask (already normalized)
        # Recompute from expressions to avoid storing intermediate columns
        total_lvl_expr = (best_bid + best_ask).alias("total_lvl")
        df = df.with_columns(total_lvl_expr)
        df = df.with_columns(
            bin_total_lvl(pl.col("total_lvl"), total_lvl_edges).alias("total_lvl_bin")
        )

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


def event_probas_spread1_3d(df: pl.DataFrame) -> pl.DataFrame:
    """Compute event probabilities for spread=1 with total_lvl_bin (3D state).

    Requires df to have total_lvl_bin column (from preprocess with edges).
    """
    # Group by imb_bin, spread, total_lvl_bin, event, event_side, event_q
    stat = df.group_by(["imb_bin", "spread", "total_lvl_bin", "event", "event_side", "event_q"]).agg(
        pl.len()
    )
    stat = stat.filter(pl.col("spread").eq(1) & pl.col("event_q").is_between(-2, 2))

    # Symmetrize: pool (imb=+x, side=A) with (imb=-x, side=B)
    # total_lvl_bin stays the same (symmetric by construction)
    stat1 = (
        stat.with_columns(
            pl.col("imb_bin")
            .sign()
            .mul(pl.col("event_side").replace({"A": 1, "B": -1}).cast(int))
            .alias("sign"),
            pl.col("imb_bin").abs(),
            pl.col("event_q").abs(),
        )
        .group_by(["imb_bin", "spread", "total_lvl_bin", "event", "event_q", "sign"])
        .agg(pl.col("len").sum())
        .with_columns(
            pl.col("sign")
            .cast(str)
            .replace({"1.0": "A", "0.0": "A", "-0.0": "A", "-1.0": "B"})
            .alias("event_side")
        )
    )

    # Mirror to negative imbalances (total_lvl_bin unchanged)
    stat2 = stat1.with_columns(
        -pl.col("imb_bin"), pl.col("event_side").replace({"A": "B", "B": "A"})
    )

    stat = pl.concat((stat1, stat2))

    # Compute probability within each (imb_bin, total_lvl_bin)
    stat = stat.with_columns(
        proba=pl.col("len").truediv(pl.col("len").sum().over(["imb_bin", "total_lvl_bin"]))
    )

    stat = stat.with_columns(
        pl.col("event_q").mul(pl.col("event_side").replace({"A": 1, "B": -1}).cast(int))
    )
    stat = stat.sort(["imb_bin", "total_lvl_bin", "spread", "event"])

    return stat.drop("sign").select(
        "imb_bin", "spread", "total_lvl_bin", "event", "event_q", "len", "event_side", "proba"
    )


def event_probas_spread2_3d(df: pl.DataFrame) -> pl.DataFrame:
    """Compute event probabilities for spread>=2 with total_lvl_bin (3D state).

    Create events only. Requires df to have total_lvl_bin column.
    """
    stat = (
        df.filter(
            pl.col("event").is_in(["Create_Bid", "Create_Ask"]) & pl.col("spread").ge(2)
        )
        .with_columns(spread=2)
        .group_by(["imb_bin", "spread", "total_lvl_bin", "event", "event_side"])
        .len()
    )
    stat = stat.filter(
        pl.col("event")
        .replace({"Create_Ask": 1, "Create_Bid": -1})
        .cast(int)
        .mul(pl.col("event_side").replace({"A": 1, "B": -1}).cast(int))
        .gt(0)
    )

    # Symmetrize
    stat1 = (
        stat.with_columns(
            pl.col("event_side")
            .replace({"A": 1, "B": -1})
            .cast(int)
            .mul(pl.col("imb_bin").sign())
            .alias("sign"),
            pl.col("imb_bin").abs(),
        )
        .group_by(["imb_bin", "spread", "total_lvl_bin", "sign"])
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

    # Mirror to negative imbalances
    stat2 = stat1.with_columns(
        -pl.col("imb_bin"),
        pl.col("event").replace(
            {"Create_Ask": "Create_Bid", "Create_Bid": "Create_Ask"}
        ),
    )

    stat = pl.concat([stat1, stat2])
    stat = stat.with_columns(
        pl.col("len").truediv(pl.col("len").sum().over(["imb_bin", "total_lvl_bin"])).alias("proba")
    ).sort(["imb_bin", "total_lvl_bin"])

    return (
        stat.with_columns(
            event_q=pl.lit(0).cast(pl.Int64),
            event_side=pl.col("event").replace({"Create_Bid": "B", "Create_Ask": "A"}),
        )
        .drop("sign")
        .select("imb_bin", "spread", "total_lvl_bin", "event", "event_q", "len", "event_side", "proba")
    )


def estimate_event_probabilities(df: pl.DataFrame) -> pl.DataFrame:
    """Estimate event probabilities by state (2D: imb_bin, spread)."""
    return pl.concat([event_probas_spread1(df), event_probas_spread2(df)]).sort(
        "imb_bin", "event", "event_side"
    )


def estimate_event_probabilities_3d(df: pl.DataFrame) -> pl.DataFrame:
    """Estimate event probabilities by state (3D: imb_bin, spread, total_lvl_bin).

    Requires df to have total_lvl_bin column (from preprocess with edges).
    """
    return pl.concat([event_probas_spread1_3d(df), event_probas_spread2_3d(df)]).sort(
        "imb_bin", "total_lvl_bin", "event", "event_side"
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


def estimate_burst_sizes(df: pl.DataFrame, median_sizes: dict[int, float]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Estimate burst size distributions for Create events (spread >= 2).

    Args:
        df: Preprocessed dataframe with imb, imb_bin, and event_q computed
        median_sizes: Median event sizes by queue level

    Returns:
        Tuple of (fitted parameters, burst sizes data for plotting)
    """
    q1_median = median_sizes[1]

    # Filter to events within Q_±2
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


def estimate_invariant_distributions(
    raw_df: pl.DataFrame | pl.LazyFrame, median_sizes: dict[int, float], q_max: int = 50
) -> pl.DataFrame:
    """Estimate invariant queue size distributions for Q_1 to Q_4.

    For each queue level i, combines Q_i and Q_{-i}, normalizes by median
    event size, and computes the empirical distribution truncated at q_max.

    Uses lazy evaluation for memory efficiency.

    Args:
        raw_df: Raw dataframe/lazyframe with Q_-10 to Q_10 columns
        median_sizes: Median event sizes by queue level
        q_max: Maximum queue size to include (truncate above this)

    Returns:
        DataFrame with columns: queue_level, q, probability
    """
    # Use lazy evaluation
    if isinstance(raw_df, pl.LazyFrame):
        lazy_df = raw_df
    else:
        lazy_df = raw_df.lazy()

    results = []

    for i in range(1, 5):
        median_i = median_sizes[i]

        # Compute both Q_i and Q_{-i} in a single lazy query
        counts = (
            lazy_df
            .select(
                (pl.col(f"Q_{i}") / median_i).ceil().cast(int).alias("q_pos"),
                (pl.col(f"Q_{-i}") / median_i).ceil().cast(int).alias("q_neg"),
            )
            .unpivot(on=["q_pos", "q_neg"], value_name="q")
            .select("q")
            .filter(pl.col("q").le(q_max))
            .group_by("q")
            .len()
            .collect()
            .sort("q")
        )

        total = counts["len"].sum()
        for row in counts.iter_rows():
            q_val, count = row
            results.append({
                "queue_level": i,
                "q": q_val,
                "probability": count / total,
            })

    return pl.DataFrame(results).sort(["queue_level", "q"])


def compute_dt_data(df: pl.DataFrame) -> pl.DataFrame:
    """Compute delta-t data lazily and return as DataFrame.

    Args:
        df: Preprocessed dataframe with ts_event column

    Returns:
        DataFrame with dt, dt_log, imb_bin, spread columns
    """
    return (
        df.lazy()
        .select(
            pl.col("ts_event").diff().over("date").cast(int).alias("dt"),
            "imb_bin", "spread"
        )
        .filter(pl.col("dt").gt(0))
        .with_columns(pl.col("dt").log10().alias("dt_log"))
        .collect()
    )


def estimate_delta_t_peak(dt_data: pl.DataFrame, sample_size: int = 500_000) -> pl.DataFrame:
    """Estimate the peak region of log10(dt) distribution using KDE and fit a normal.

    The peak region represents "normal" QR events (round-trip delta + reaction time).
    Events below this region are considered "race" events.

    Args:
        dt_data: DataFrame with dt_log column (from compute_dt_data)
        sample_size: Max samples for KDE (subsamples if larger)

    Returns:
        peak_params DataFrame with columns: mu, sigma, lower, upper
    """
    # Filter to exclude the QR tail (very slow events)
    cutoff = 5.5
    data = dt_data.lazy().filter(pl.col("dt_log").lt(cutoff)).select("dt_log").collect()["dt_log"].to_numpy()

    # Subsample for KDE if too large
    if len(data) > sample_size:
        rng = np.random.default_rng(42)
        data = rng.choice(data, size=sample_size, replace=False)

    # Use KDE to find peak and 80% threshold bounds
    kde = gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 500)
    y = kde(x)
    peak_idx = np.argmax(y)
    peak_val = y[peak_idx]

    # Find where density drops to 80% of peak
    threshold = 0.8 * peak_val
    left_idx = np.where(y[:peak_idx] < threshold)[0][-1] if any(y[:peak_idx] < threshold) else 0
    right_idx = peak_idx + np.where(y[peak_idx:] < threshold)[0][0] if any(y[peak_idx:] < threshold) else len(y) - 1

    lower = x[left_idx]
    upper = x[right_idx]

    # Fit normal to data in peak region
    peak_data = data[(data > lower) & (data < upper)]
    mu, sigma = norm_dist.fit(peak_data)

    return pl.DataFrame({
        "mu": [mu],
        "sigma": [sigma],
        "lower": [lower],
        "upper": [upper],
    })


def estimate_delta_t_mixtures(
    dt_data: pl.DataFrame, floor_threshold: float = 0.0
) -> pl.DataFrame:
    """Fit 3-component Gaussian Mixture to log10(dt) for each (imb_bin, spread) pair.

    Args:
        dt_data: DataFrame with dt_log, imb_bin, spread columns (from compute_dt_data)
        floor_threshold: Minimum log10(dt) value to include (0.0 for all, peak_upper for floored)

    Returns:
        DataFrame with GMM parameters: imb_bin, spread, w1-3, mu1-3, sigma1-3, n
    """
    # Apply floor threshold lazily
    dt_filtered = dt_data.lazy().filter(pl.col("dt_log").gt(floor_threshold)).collect()

    # Get unique imbalance bins and spreads
    imb_bins = sorted(dt_filtered["imb_bin"].unique().drop_nulls().to_list())
    spreads = sorted(dt_filtered["spread"].unique().to_list())

    gmm_params = []
    for imb_bin in imb_bins:
        for spread in spreads:
            # Get data for this (imb_bin, spread) pair
            data = dt_filtered.filter(
                (pl.col("imb_bin").eq(imb_bin)) & (pl.col("spread").eq(spread))
            )["dt_log"].to_numpy()

            if len(data) < 10:
                continue

            log_dt = data.reshape(-1, 1)

            # Fit 3-component Gaussian mixture
            gmm = GaussianMixture(n_components=3, random_state=42).fit(log_dt)

            # Extract parameters
            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_.flatten())
            weights = gmm.weights_

            # Store params (spread as 0 or 1 for C++ indexing)
            gmm_params.append({
                "imb_bin": imb_bin,
                "spread": spread - 1,  # 0 for spread=1, 1 for spread>=2
                "w1": weights[0], "mu1": means[0], "sigma1": stds[0],
                "w2": weights[1], "mu2": means[1], "sigma2": stds[1],
                "w3": weights[2], "mu3": means[2], "sigma3": stds[2],
                "n": len(data)
            })

    return pl.DataFrame(gmm_params)


def estimate_delta_t_left_tail(
    dt_data: pl.DataFrame, max_log10: float
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fit Gamma and Weibull distributions to the left tail (fast "race" events).

    Args:
        dt_data: DataFrame with dt_log column (from compute_dt_data)
        max_log10: Maximum log10(dt) value for left tail (peak_lower from peak estimation)

    Returns:
        Tuple of (gamma_params, weibull_params) DataFrames
    """
    # Filter to left tail lazily
    left_tail = (
        dt_data.lazy()
        .filter(pl.col("dt_log").lt(max_log10))
        .select("dt_log")
        .collect()["dt_log"]
        .to_numpy()
    )

    if len(left_tail) < 10:
        # Return empty DataFrames if not enough data
        gamma_params = pl.DataFrame({"k": [0.0], "scale": [0.0], "shift": [0.0], "max_log10": [max_log10]})
        weibull_params = pl.DataFrame({"k": [0.0], "scale": [0.0], "shift": [0.0], "max_log10": [max_log10]})
        return gamma_params, weibull_params

    # Shift data for fitting (both require x > 0)
    shift = left_tail.min()
    left_tail_shifted = left_tail - shift + 0.01

    # Fit Gamma
    k_gamma, _, scale_gamma = gamma_dist.fit(left_tail_shifted, floc=0)

    # Fit Weibull
    k_weibull, _, scale_weibull = weibull_min.fit(left_tail_shifted, floc=0)

    gamma_params = pl.DataFrame({
        "k": [k_gamma],
        "scale": [scale_gamma],
        "shift": [shift],
        "max_log10": [max_log10],
    })

    weibull_params = pl.DataFrame({
        "k": [k_weibull],
        "scale": [scale_weibull],
        "shift": [shift],
        "max_log10": [max_log10],
    })

    return gamma_params, weibull_params


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


def plot_spread_distribution_from_stats(
    spread_all: pl.DataFrame, spread_trades: pl.DataFrame, ticker: str
) -> plt.Figure:
    """Plot the distribution of spread using pre-computed stats."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: All events
    total_all = spread_all["len"].sum()
    proportions_all = spread_all["len"] / total_all
    ax1.bar(spread_all["spread"], proportions_all, edgecolor="k", alpha=0.7)
    ax1.set_xlabel("Spread (ticks)")
    ax1.set_ylabel("Proportion")
    ax1.set_title("Spread Distribution - All Events")

    # Right: Trades only
    total_trades = spread_trades["len"].sum()
    proportions_trades = spread_trades["len"] / total_trades
    ax2.bar(spread_trades["spread"], proportions_trades, edgecolor="k", alpha=0.7, color="tab:orange")
    ax2.set_xlabel("Spread (ticks)")
    ax2.set_ylabel("Proportion")
    ax2.set_title("Spread Distribution - Trades Only")

    fig.suptitle(f"{ticker} - Spread Distributions", fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def plot_invariant_distributions(
    invariant_dist: pl.DataFrame, ticker: str
) -> plt.Figure:
    """Plot invariant queue size distributions for Q_1 to Q_4."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes, start=1):
        data = invariant_dist.filter(pl.col("queue_level").eq(i)).sort("q")
        ax.bar(data["q"], data["probability"], edgecolor="k", alpha=0.7, width=0.8)
        ax.set_xlabel("Queue Size (normalized)")
        ax.set_ylabel("Probability")
        ax.set_title(f"Q_{i} Invariant Distribution")
        ax.set_xlim(0, 50)

    fig.suptitle(f"{ticker} - Invariant Queue Distributions", fontsize=12, y=1.02)
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


def plot_total_lvl_quantiles(
    raw_df: pl.DataFrame | pl.LazyFrame,
    quantile_stats: pl.DataFrame,
    median_sizes: dict[int, float],
    ticker: str,
    sample_size: int = 500_000,
) -> plt.Figure:
    """Plot total_lvl distribution with quantile edges marked.

    Uses sampling for memory efficiency with large datasets.

    Args:
        raw_df: Raw dataframe/lazyframe with Q_-1 and Q_1 columns
        quantile_stats: DataFrame with bin, lower, upper columns
        median_sizes: Median event sizes by queue level (for normalization)
        ticker: Ticker symbol for title
        sample_size: Maximum samples to use for histogram (default 500K)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    q1_median = median_sizes[1]

    # Use lazy evaluation and sampling for memory efficiency
    if isinstance(raw_df, pl.LazyFrame):
        lazy_df = raw_df
    else:
        lazy_df = raw_df.lazy()

    # Sample data if needed
    n_rows = lazy_df.select(pl.len()).collect().item()
    if n_rows > sample_size:
        sample_frac = sample_size / n_rows
        total_lvl = (
            lazy_df
            .select(((pl.col("Q_-1") + pl.col("Q_1")) / q1_median).ceil().alias("total_lvl"))
            .filter(pl.lit(np.random.default_rng(42).random(n_rows)) < sample_frac)
            .collect()["total_lvl"]
            .to_numpy()
        )
    else:
        total_lvl = (
            lazy_df
            .select(((pl.col("Q_-1") + pl.col("Q_1")) / q1_median).ceil().alias("total_lvl"))
            .collect()["total_lvl"]
            .to_numpy()
        )

    # Left: Histogram with quantile edges
    ax1.hist(total_lvl, bins=100, density=True, alpha=0.7, color="steelblue", edgecolor="black", linewidth=0.3)

    # Mark quantile edges
    edges = [quantile_stats["lower"][0]] + quantile_stats["upper"].to_list()
    colors = plt.cm.tab10(np.linspace(0, 1, len(edges) - 1))
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        ax1.axvline(left, color=colors[i], linestyle="--", lw=1.5, alpha=0.8)
        ax1.axvspan(left, right, alpha=0.15, color=colors[i], label=f"Bin {i}: [{left:.1f}, {right:.1f})")
    ax1.axvline(edges[-1], color=colors[-1], linestyle="--", lw=1.5, alpha=0.8)

    ax1.set_xlabel("Total Level (Q_bid + Q_ask, normalized)")
    ax1.set_ylabel("Density")
    ax1.set_title("Total Level Distribution with Quantile Bins")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_xlim(0, np.percentile(total_lvl, 99))

    # Right: Table showing quantile statistics
    ax2.axis("off")
    table_data = [["Bin", "Lower", "Upper", "Percentile Range"]]
    for row in quantile_stats.iter_rows(named=True):
        table_data.append([
            f"{row['bin']}",
            f"{row['lower']:.1f}",
            f"{row['upper']:.1f}",
            f"{row['percentile_lower']:.0f}% - {row['percentile_upper']:.0f}%",
        ])

    table = ax2.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="center",
        colWidths=[0.15, 0.25, 0.25, 0.35],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header row
    for j in range(4):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(color="white", weight="bold")

    ax2.set_title("Quantile Bin Edges", fontsize=12, pad=20)

    fig.suptitle(f"{ticker} - Total Level (Q_bid + Q_ask) Quantiles", fontsize=12, y=1.02)
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


def plot_delta_t_peak_fit(
    dt_data: pl.DataFrame, peak_params: pl.DataFrame, ticker: str
) -> plt.Figure:
    """Plot the peak region fit: Gaussian fit (left) and full distribution with 80% region (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    mu = peak_params["mu"][0]
    sigma = peak_params["sigma"][0]
    lower = peak_params["lower"][0]
    upper = peak_params["upper"][0]

    # Get data
    cutoff = 5.5
    data = dt_data.filter(pl.col("dt_log").lt(cutoff))["dt_log"].to_numpy()
    peak_data = data[(data > lower) & (data < upper)]

    # Left: Gaussian fit to peak region
    ax1.hist(peak_data, bins=50, density=True, alpha=0.5, color="gray", edgecolor="black", linewidth=0.3)
    x = np.linspace(lower - 0.1, upper + 0.1, 200)
    ax1.plot(x, norm_dist.pdf(x, mu, sigma), "b-", lw=2, label=f"N(μ={mu:.3f}, σ={sigma:.3f})")
    ax1.axvline(lower, color="gray", linestyle=":", alpha=0.5)
    ax1.axvline(upper, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("log₁₀(dt)")
    ax1.set_ylabel("Density")
    ax1.set_title("Peak Region: Gaussian Fit")
    ax1.legend()

    # Right: Full distribution with 80% density region shaded
    ax2.hist(data, bins=100, density=True, alpha=0.5, color="steelblue", edgecolor="black", linewidth=0.3)
    ax2.axvspan(lower, upper, alpha=0.3, color="coral", label=f"80% peak region [{lower:.2f}, {upper:.2f}]")
    ax2.set_xlabel("log₁₀(dt)")
    ax2.set_ylabel("Density")
    ax2.set_title("Full Distribution with Peak Region")
    ax2.legend()

    fig.suptitle(f"{ticker} - Delta-t Peak Distribution", fontsize=12)
    plt.tight_layout()
    return fig


def plot_delta_t_gmm_grid(
    dt_data: pl.DataFrame, gmm_params: pl.DataFrame, floor_threshold: float, ticker: str
) -> plt.Figure:
    """Plot 7x6 grid of GMM fits for floored delta_t (21 imb_bins × 2 spreads)."""
    # Get unique imbalance bins and spreads from gmm_params
    imb_bins = sorted(gmm_params["imb_bin"].unique().to_list())
    spreads = [0, 1]  # 0=spread1, 1=spread>=2

    fig, axes = plt.subplots(7, 6, figsize=(20, 20))

    for i, imb_bin in enumerate(imb_bins):
        for j, spread in enumerate(spreads):
            # Column: spread=0 uses cols 0,1,2; spread=1 uses cols 3,4,5
            col = (i % 3) + (j * 3)
            row = i // 3
            ax = axes[row, col]

            # Get GMM params for this (imb_bin, spread)
            params = gmm_params.filter(
                (pl.col("imb_bin").eq(imb_bin)) & (pl.col("spread").eq(spread))
            )

            if len(params) == 0:
                ax.set_title(f"imb={imb_bin:.1f}, s={spread+1}", fontsize=8)
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue

            # Get FLOORED data for this (imb_bin, spread) - same filter as GMM estimation
            data = dt_data.filter(
                (pl.col("imb_bin").eq(imb_bin)) &
                (pl.col("spread").eq(spread + 1)) &
                (pl.col("dt_log").gt(floor_threshold))
            )["dt_log"].to_numpy()

            if len(data) < 10:
                ax.set_title(f"imb={imb_bin:.1f}, s={spread+1}", fontsize=8)
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
                continue

            # Extract GMM parameters
            w = [params["w1"][0], params["w2"][0], params["w3"][0]]
            mu = [params["mu1"][0], params["mu2"][0], params["mu3"][0]]
            sigma = [params["sigma1"][0], params["sigma2"][0], params["sigma3"][0]]

            # Plot histogram of floored data
            ax.hist(data, bins=50, density=True, alpha=0.5, color="gray")

            # Plot GMM components
            x = np.linspace(data.min() - 0.5, data.max() + 0.5, 200)
            pdf_mix = np.zeros_like(x)
            for k in range(3):
                pdf_k = w[k] * norm_dist.pdf(x, mu[k], sigma[k])
                ax.plot(x, pdf_k, "--", lw=1, label=f"w={w[k]:.2f}")
                pdf_mix += pdf_k
            ax.plot(x, pdf_mix, "r-", lw=1.5)

            ax.set_title(f"imb={imb_bin:.1f}, s={spread+1} (n={params['n'][0]:,})", fontsize=8)
            ax.legend(fontsize=6, loc="upper right")
            ax.set_xlabel("log₁₀(dt)", fontsize=7)
            ax.set_ylabel("Density", fontsize=7)
            ax.tick_params(labelsize=6)

    fig.suptitle(f"{ticker} - GMM Fits for log₁₀(dt) (floored at {floor_threshold:.2f})", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    return fig


def plot_delta_t_left_tail(
    dt_data: pl.DataFrame, gamma_params: pl.DataFrame, weibull_params: pl.DataFrame, ticker: str
) -> plt.Figure:
    """Plot left tail fits: log10 space (left) and nanoseconds (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    max_log10 = gamma_params["max_log10"][0]
    left_tail = dt_data.filter(pl.col("dt_log").lt(max_log10))["dt_log"].to_numpy()

    if len(left_tail) < 10:
        ax1.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax1.transAxes)
        ax2.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax2.transAxes)
        return fig

    # Get params
    k_gamma = gamma_params["k"][0]
    scale_gamma = gamma_params["scale"][0]
    k_weibull = weibull_params["k"][0]
    scale_weibull = weibull_params["scale"][0]
    shift = gamma_params["shift"][0]

    # Left plot: log10 space
    ax1.hist(left_tail, bins=80, alpha=0.6, color="steelblue", edgecolor="black", linewidth=0.5, density=True)
    x_log = np.linspace(left_tail.min(), left_tail.max(), 200)
    x_shifted = x_log - shift + 0.01

    ax1.plot(x_log, gamma_dist.pdf(x_shifted, k_gamma, loc=0, scale=scale_gamma), "r-", lw=2,
             label=f"Gamma(k={k_gamma:.2f}, θ={scale_gamma:.2f})")
    ax1.plot(x_log, weibull_min.pdf(x_shifted, k_weibull, loc=0, scale=scale_weibull), "g-", lw=2,
             label=f"Weibull(k={k_weibull:.2f}, λ={scale_weibull:.2f})")
    ax1.axvline(max_log10, color="gray", linestyle="--", lw=1.5)
    ax1.set_xlabel("log₁₀(dt)")
    ax1.set_ylabel("Density")
    ax1.set_title(f"Left Tail (n={len(left_tail):,})")
    ax1.legend()

    # Right plot: nanoseconds
    ns_data = 10 ** left_tail
    ns_min, ns_max = ns_data.min(), ns_data.max()

    ax2.hist(ns_data, bins=80, alpha=0.6, color="coral", edgecolor="black", linewidth=0.5, density=True)

    x_ns = np.linspace(ns_min, ns_max, 200)
    x_log_t = np.log10(x_ns)
    x_shifted_t = x_log_t - shift + 0.01

    # Transform PDFs: f_Y(y) = f_X(log10(y)) / (y * ln(10))
    pdf_gamma_ns = gamma_dist.pdf(x_shifted_t, k_gamma, loc=0, scale=scale_gamma) / (x_ns * np.log(10))
    pdf_weibull_ns = weibull_min.pdf(x_shifted_t, k_weibull, loc=0, scale=scale_weibull) / (x_ns * np.log(10))

    ax2.plot(x_ns, pdf_gamma_ns, "r-", lw=2, label="Gamma")
    ax2.plot(x_ns, pdf_weibull_ns, "g-", lw=2, label="Weibull")
    ax2.set_xlabel("γ (ns)")
    ax2.set_ylabel("Density")
    ax2.set_title("Inter-racer Delay Distribution")
    ax2.legend()

    fig.suptitle(f"{ticker} - Left Tail (Race Events)", fontsize=12)
    plt.tight_layout()
    return fig


def save_figure(fig, path: Path, name: str) -> Path:
    """Save figure to path and return the full path."""
    full_path = path / f"{name}.png"
    fig.savefig(full_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {name}.png")
    return full_path


def concat_figures_to_pdf(figure_paths: list[Path], output_path: Path) -> None:
    """Concatenate PNG figures into a single PDF."""
    print(f"Concatenating {len(figure_paths)} figures into PDF...")
    with PdfPages(output_path) as pdf:
        for path in figure_paths:
            img = plt.imread(path)
            fig, ax = plt.subplots(figsize=(img.shape[1] / 150, img.shape[0] / 150), dpi=150)
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
    print(f"  Saved {output_path}")


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
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Use more aggressive memory management (slower but uses less RAM)"
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Fraction of data to sample (0.0-1.0, default: 1.0 = all data)"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum rows to load (default: all). Use for memory-constrained systems."
    )
    args = parser.parse_args()

    ticker = args.ticker
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"data/{ticker}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figures directory
    figures_dir = output_dir / "figures"
    if not args.no_plots:
        figures_dir.mkdir(exist_ok=True)
    figure_paths = []

    print(f"Estimating QR model parameters for {ticker}")
    print(f"Output directory: {output_dir}")

    loader = DataLoader()

    # Load raw data for initial computations
    print("Loading raw data...")
    raw_df = load_raw_data(loader, ticker)

    # Apply sampling/row limits if requested
    if args.max_rows is not None and len(raw_df) > args.max_rows:
        print(f"  Limiting to {args.max_rows:,} rows (from {len(raw_df):,})...")
        raw_df = raw_df.head(args.max_rows)
    if args.sample_frac < 1.0:
        print(f"  Sampling {args.sample_frac:.0%} of data...")
        raw_df = raw_df.sample(fraction=args.sample_frac, seed=42)

    min_date = raw_df["date"].min()
    max_date = raw_df["date"].max()
    n_days = raw_df["date"].n_unique()
    avg_daily_events = len(raw_df) / n_days
    print(f"  Loaded {len(raw_df):,} events")
    print(f"  Date range: {min_date} to {max_date} ({n_days} days)")
    print(f"  Average daily events: {avg_daily_events:,.0f}")

    # Compute median event sizes (requires raw Q columns)
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

    if not args.no_plots:
        fig = plot_median_sizes_table(median_sizes, ticker)
        figure_paths.append(save_figure(fig, figures_dir, "02_median_sizes"))

    # Compute total_lvl quantiles (requires raw Q columns)
    print("Computing total_lvl quantiles...")
    total_lvl_edges, total_lvl_stats = compute_total_lvl_quantiles(raw_df, median_sizes, n_bins=5)
    total_lvl_stats.write_csv(output_dir / "total_lvl_quantiles.csv")
    print(f"  Saved 5 quantile bins to total_lvl_quantiles.csv")
    print(f"  Edges: {total_lvl_edges}")

    if not args.no_plots:
        fig = plot_total_lvl_quantiles(raw_df, total_lvl_stats, median_sizes, ticker)
        figure_paths.append(save_figure(fig, figures_dir, "02b_total_lvl_quantiles"))

    # Estimate invariant distributions (requires raw Q columns)
    print("Estimating invariant queue distributions...")
    q_max = 50
    invariant_dist = estimate_invariant_distributions(raw_df, median_sizes, q_max=q_max)
    invariant_dist.write_csv(output_dir / f"invariant_distributions_qmax{q_max}.csv")
    print(f"  Saved {len(invariant_dist)} rows to invariant_distributions_qmax{q_max}.csv")

    if not args.no_plots:
        fig = plot_invariant_distributions(invariant_dist, ticker)
        figure_paths.append(save_figure(fig, figures_dir, "08_invariant_distributions"))

    # Pre-compute spread stats for plotting (before freeing raw_df)
    print("Computing spread statistics...")
    spread_all = raw_df.group_by("spread").len().sort("spread").filter(pl.col("spread").le(20))
    spread_trades = raw_df.filter(pl.col("event").is_in(["Trd", "Trd_All"])).group_by("spread").len().sort("spread").filter(pl.col("spread").le(20))

    if not args.no_plots:
        fig = plot_spread_distribution_from_stats(spread_all, spread_trades, ticker)
        figure_paths.append(save_figure(fig, figures_dir, "01_spread_distribution"))

    # Preprocess data (still needs raw_df)
    # Pass total_lvl_edges to add total_lvl and total_lvl_bin columns
    print("Preprocessing data...")
    df = preprocess(raw_df, median_sizes, total_lvl_edges=total_lvl_edges)
    print(f"  Preprocessed {len(df):,} events")

    # FREE raw_df to release memory
    del raw_df
    del spread_all, spread_trades
    gc.collect()
    print("  Freed raw data from memory")

    if not args.no_plots:
        fig = plot_imbalance_distribution(df, ticker)
        figure_paths.append(save_figure(fig, figures_dir, "03_imbalance_distribution"))

    # Estimate event probabilities (2D: imb_bin, spread)
    print("Estimating event probabilities (2D)...")
    event_probs = estimate_event_probabilities(df)
    event_probs.write_csv(output_dir / "event_probabilities.csv")
    print(f"  Saved {len(event_probs)} rows to event_probabilities.csv")

    if not args.no_plots:
        fig = plot_event_probabilities(event_probs, ticker)
        figure_paths.append(save_figure(fig, figures_dir, "04_event_probabilities"))

    # Estimate event probabilities (3D: imb_bin, spread, total_lvl_bin)
    print("Estimating event probabilities (3D with total_lvl)...")
    event_probs_3d = estimate_event_probabilities_3d(df)
    event_probs_3d.write_csv(output_dir / "event_probabilities_3d.csv")
    print(f"  Saved {len(event_probs_3d)} rows to event_probabilities_3d.csv")
    del event_probs_3d
    if args.low_memory:
        gc.collect()

    # Estimate intensities
    print("Estimating intensities...")
    intensities, dt_data = estimate_intensities(df)
    intensities.write_csv(output_dir / "intensities.csv")
    print(f"  Saved {len(intensities)} rows to intensities.csv")

    if not args.no_plots:
        fig = plot_intensities(intensities, dt_data, ticker)
        figure_paths.append(save_figure(fig, figures_dir, "05_intensities"))

    # Estimate size distributions (spread=1)
    print("Estimating size distributions (spread=1)...")
    size_dist, size_stat = estimate_size_distributions(df, median_sizes)

    if not args.no_plots:
        fig = plot_size_distributions(size_stat, ticker)
        figure_paths.append(save_figure(fig, figures_dir, "06_size_distributions"))

    # Estimate burst sizes (spread>=2)
    print("Estimating burst size distributions (spread>=2)...")
    burst_dist, burst_sizes = estimate_burst_sizes(df, median_sizes)

    if not args.no_plots and len(burst_sizes) > 0:
        fig = plot_burst_sizes(burst_sizes, ticker)
        figure_paths.append(save_figure(fig, figures_dir, "07_burst_sizes"))

    # Combine size distributions
    combined = pl.concat([size_dist, burst_dist]).sort(
        ["spread", "imb_bin", "event", "event_q"]
    )
    combined.write_csv(output_dir / "size_distrib.csv")
    print(f"  Saved {len(combined)} rows to size_distrib.csv")
    del size_dist, burst_dist, combined, size_stat, burst_sizes
    if args.low_memory:
        gc.collect()

    # Compute dt_data once for all delta_t estimations
    print("Computing delta_t data...")
    delta_t_data = compute_dt_data(df)
    print(f"  Computed {len(delta_t_data):,} inter-arrival times")

    # Free preprocessed dataframe - no longer needed
    del df
    gc.collect()
    print("  Freed preprocessed data from memory")

    # Step 1: Estimate peak region FIRST to get the floor threshold
    print("Estimating delta_t peak distribution...")
    delta_t_peak_params = estimate_delta_t_peak(delta_t_data)
    delta_t_peak_params.write_csv(output_dir / "delta_distrib.csv")
    peak_lower = delta_t_peak_params["lower"][0]
    peak_upper = delta_t_peak_params["upper"][0]
    print(f"  Peak region: [{peak_lower:.2f}, {peak_upper:.2f}]")

    if not args.no_plots:
        fig = plot_delta_t_peak_fit(delta_t_data, delta_t_peak_params, ticker)
        figure_paths.append(save_figure(fig, figures_dir, "09_delta_t_peak"))

    # Step 2: GMM on all events (floor_threshold=0) - saved but NOT plotted
    print("Estimating delta_t GMM (all events)...")
    delta_t_mixtures = estimate_delta_t_mixtures(delta_t_data, floor_threshold=0.0)
    delta_t_mixtures.write_csv(output_dir / "delta_t_mixtures.csv")
    print(f"  Saved {len(delta_t_mixtures)} rows to delta_t_mixtures.csv")

    # Step 3: GMM floored at peak_upper (excludes race events) - this one is PLOTTED
    print(f"Estimating delta_t GMM (floored at {peak_upper:.2f})...")
    delta_t_mixtures_floored = estimate_delta_t_mixtures(delta_t_data, floor_threshold=peak_upper)
    delta_t_mixtures_floored.write_csv(output_dir / "delta_t_mixtures_floored.csv")
    print(f"  Saved {len(delta_t_mixtures_floored)} rows to delta_t_mixtures_floored.csv")

    if not args.no_plots and len(delta_t_mixtures_floored) > 0:
        fig = plot_delta_t_gmm_grid(delta_t_data, delta_t_mixtures_floored, peak_upper, ticker)
        figure_paths.append(save_figure(fig, figures_dir, "10_delta_t_gmm_grid"))

    # Step 4: Left tail fits (gamma/weibull for race events)
    print("Estimating delta_t left tail (gamma/weibull)...")
    gamma_params, weibull_params = estimate_delta_t_left_tail(delta_t_data, max_log10=peak_lower)
    gamma_params.write_csv(output_dir / "gamma_distrib.csv")
    weibull_params.write_csv(output_dir / "weibull_distrib.csv")
    print("  Saved gamma_distrib.csv and weibull_distrib.csv")

    if not args.no_plots:
        fig = plot_delta_t_left_tail(delta_t_data, gamma_params, weibull_params, ticker)
        figure_paths.append(save_figure(fig, figures_dir, "11_delta_t_left_tail"))

    # Concatenate all figures into PDF
    if not args.no_plots:
        # Sort by filename to get correct order
        figure_paths.sort(key=lambda p: p.name)
        report_path = output_dir / "estimation_report.pdf"
        concat_figures_to_pdf(figure_paths, report_path)

    print("Done!")


if __name__ == "__main__":
    main()
