from functools import reduce

import polars as pl

def pl_select(condlist: list[pl.Expr], choicelist: list[pl.Expr]) -> pl.Expr:
    return reduce(
        lambda expr, cond_choice: expr.when(cond_choice[0]).then(cond_choice[1]),
        zip(condlist, choicelist),
        pl.when(condlist[0]).then(choicelist[0]),
    )

# === Recording LOB State ===

spread: pl.Expr = pl.col("ask_px_00").sub(pl.col("bid_px_00")).alias("spread")
imbalance: pl.Expr = (
    pl.col("bid_sz_00")
    .sub(pl.col("ask_sz_00"))
    .truediv(pl.col("bid_sz_00").add(pl.col("ask_sz_00")))
    .alias("imbalance")
)

bid_prices: dict[str, pl.Expr] = {
    f"P_{-i}": pl.when(spread.mod(2).eq(0))
    .then(pl.col("bid_px_00").add(spread.floordiv(2) - 1))
    .otherwise(pl.col("bid_px_00").add(spread.floordiv(2)))
    .sub(i - 1)
    for i in range(10, 0, -1)
}
ask_prices: dict[str, pl.Expr] = {
    f"P_{i}": pl.when(spread.mod(2).eq(0))
    .then(pl.col("ask_px_00").sub(spread.floordiv(2) - 1))
    .otherwise(pl.col("ask_px_00").sub(spread.floordiv(2)))
    .add(i - 1)
    for i in range(1, 11)
}
prices: dict[str, pl.Expr] = {
    **bid_prices,
    "P_0": pl.when(spread.mod(2).eq(0))
    .then(pl.col("bid_px_00").add(spread.floordiv(2)))
    .otherwise(None),
    **ask_prices,
}

bid_volumes: dict[str, pl.Expr] = {
    f"Q_{-i}": pl_select(
        condlist=[pl.col(f"bid_px_0{j}").eq(bid_prices[f"P_{-i}"]) for j in range(10)],
        choicelist=[pl.col(f"bid_sz_0{j}") for j in range(10)],
    ).fill_null(0)
    for i in range(10, 0, -1)
}
ask_volumes: dict[str, pl.Expr] = {
    f"Q_{i}": pl_select(
        condlist=[pl.col(f"ask_px_0{j}").eq(ask_prices[f"P_{i}"]) for j in range(10)],
        choicelist=[pl.col(f"ask_sz_0{j}") for j in range(10)],
    ).fill_null(0)
    for i in range(1, 11)
}
volumes: dict[str, pl.Expr] = {
    **bid_volumes,
    "Q_0": pl.lit(0),
    **ask_volumes,
}

best_bid_nbr: pl.Expr = pl.max_horizontal(
    [pl.when(volumes[f"Q_{-i}"].gt(0)).then(-i).otherwise(-11) for i in range(1, 11)]
).alias("best_bid_nbr")
best_ask_nbr: pl.Expr = pl.min_horizontal(
    [pl.when(volumes[f"Q_{i}"].gt(0)).then(i).otherwise(11) for i in range(1, 11)]
).alias("best_ask_nbr")

lob_state: dict[str, pl.Expr] = dict(
    best_bid_nbr=best_bid_nbr,
    **volumes,
    best_ask_nbr=best_ask_nbr,
    **prices,
    spread=spread,
    imbalance=imbalance,
)

# === Recording Events ===

event_queue_nbr: pl.Expr = pl_select(
    condlist=[pl.col("price").eq(prices[f"P_{i}"]) for i in range(-10, 11)],
    choicelist=[pl.lit(i) for i in range(-10, 11)],
)
event_queue_size: pl.Expr = pl_select(
    condlist=[event_queue_nbr.eq(i) for i in range(-10, 11)],
    choicelist=[volumes[f"Q_{i}"] for i in range(-10, 11)],
)

trd_all: pl.Expr = pl.col("action").eq("T") & pl.col("size").eq(event_queue_size)
create_new: pl.Expr = (
    pl.col("action").eq("A")
    & event_queue_nbr.lt(best_ask_nbr)
    & event_queue_nbr.gt(best_bid_nbr)
)
create_ask: pl.Expr = create_new & pl.col("side").eq("A")
create_bid: pl.Expr = create_new & pl.col("side").eq("B")

event: pl.Expr = (
    pl.when(trd_all)
    .then(pl.lit("Trd_All"))
    .when(create_ask)
    .then(pl.lit("Create_Ask"))
    .when(create_bid)
    .then(pl.lit("Create_Bid"))
    .otherwise(pl.col("action").replace({"A": "Add", "C": "Can", "T": "Trd"}))
).alias("event")

event_records: dict[str, pl.Expr] = dict(
    ts_event=pl.col("ts_event"),
    sequence=pl.col("sequence"),
    event=event,
    event_size=pl.col("size"),
    price=pl.col("price"),
    event_side=pl.col("side"),
    event_queue_nbr=event_queue_nbr,
    event_queue_size=event_queue_size,
)


def mbp_to_qr(df: pl.LazyFrame) -> pl.LazyFrame:
    """Transform LOB data into queue-reactive format.

    This function applies the queue-reactive framework to raw LOB data by:
    1. Loading and preprocessing the parquet file
    2. Converting prices to a normalized tick grid
    3. Computing queue-specific metrics
    4. Classifying and recording order events

    Args:
        file_path: Path to the parquet file containing raw LOB data

    Returns:
        A LazyFrame with columns:
        - symbol : str
            Instrument identifier
        - date : datetime.date
            Event date
        - ts_event : datetime
            Event timestamp
        - event : str
            Event type (Trd_All, Create_Ask, Create_Bid, Add, Can, Trd)
        - event_size : int
            Size of the event
        - price : float
            Price level where event occurred
        - event_side : str
            Side of the event (A/B)
        - event_queue_nbr : int
            Normalized queue index (-10 to 10)
        - event_queue_size : int
            Size of the affected queue
        - Q_-10 to Q_10 : int
            Volumes at each queue level
        - P_-10 to P_10 : float
            Prices at each queue level
        - spread : float
            Bid-ask spread
        - imbalance : float
            Order book imbalance

    Notes
    -----
    - Rows with null event_queue_nbr are dropped to ensure data quality
    - Only processes orders with valid sides (A/B) and actions (T/A/C)

    """
    df = df.filter(pl.col("ts_event").dt.time().ge(pl.time(10, 0)) & pl.col("ts_event").dt.time().le(pl.time(15, 30)))
    return df.select(
        symbol=pl.col("symbol"),
        date=pl.col("ts_event").dt.date(),
        **event_records,
        **lob_state,
    ).drop_nulls(subset="event_queue_nbr")


def main() -> None:
    from lobib import preprocessing as pr, DataLoader

    loader = DataLoader()
    tickers = ["PFE", "T", "VZ", "INTC"]
    info = loader.ticker_info(*tickers)
    files = info.filter(pl.col("schema").eq("mbp-10-raw"))["file"].to_list()

    def kernel(file: str) -> None:
        from pathlib import Path
        mbp_out = Path(file.replace("mbp-10-raw", "mbp-10"))
        qr_out = Path(file.replace("mbp-10-raw", "qr"))

        # Skip if already processed
        # if qr_out.exists():
        #     return

        lf = pl.scan_parquet(file)
        lf = pr.preprocess_mbp10(lf)
        lf = pr.truncate_time(lf)
        lf = pr.prices_to_ticks(lf)
        lf = pr.aggregate_trades(lf)
        lf = lf.filter(pl.col("side").ne("N") & pl.col("action").is_in(["T", "A", "C"]))
        lf = pr.pre_update_lob(lf)
        df = lf.collect()
        mbp_out.parent.mkdir(parents=True, exist_ok=True)
        qr_out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(mbp_out)
        df = mbp_to_qr(df)
        df.write_parquet(qr_out)

    errors = []
    for i, file in enumerate(files, 1):
        try:
            kernel(file)
            print(f"[{i}/{len(files)}] Processed {file}")
        except Exception as e:
            errors.append((file, str(e)))
            print(f"[{i}/{len(files)}] Error processing {file}: {e}")

    if errors:
        print(f"\n{len(errors)} files failed:")
        for f, e in errors:
            print(f"  {f}: {e}")

if __name__ == "__main__":
    main()

