from pathlib import Path

import pandas as pd


def create_sample_portfolio() -> pd.DataFrame:
    """
    Create a synthetic FX spot portfolio with 3 trades per FX pair.

    Conventions
    -----------
    - pair is BASEQUOTE, e.g. USDJPY or AUDUSD
    - notional_base is denominated in the base currency
    - LONG means long base / short quote
    - SHORT means short base / long quote

    Each FX pair deliberately contains both LONG and SHORT trades so the
    dashboard can test:
    - within-pair netting
    - pair-level aggregation
    - trade-level drilldown
    - pair-level Component / Incremental VaR
    """

    trades = [
        # ====================================================
        # USDJPY — net LONG
        # ====================================================
        {
            "trade_id": "FX001",
            "trade_date": "2026-09-03",
            "pair": "USDJPY",
            "side": "LONG",
            "notional_base": 2_000_000,
            "entry_price": 158.923004,
        },
        {
            "trade_id": "FX002",
            "trade_date": "2026-09-10",
            "pair": "USDJPY",
            "side": "SHORT",
            "notional_base": 750_000,
            "entry_price": 159.412500,
        },
        {
            "trade_id": "FX003",
            "trade_date": "2026-09-17",
            "pair": "USDJPY",
            "side": "LONG",
            "notional_base": 1_250_000,
            "entry_price": 159.087500,
        },

        # ====================================================
        # USDSGD — net SHORT
        # ====================================================
        {
            "trade_id": "FX004",
            "trade_date": "2026-09-04",
            "pair": "USDSGD",
            "side": "SHORT",
            "notional_base": 2_000_000,
            "entry_price": 1.266800,
        },
        {
            "trade_id": "FX005",
            "trade_date": "2026-09-11",
            "pair": "USDSGD",
            "side": "LONG",
            "notional_base": 750_000,
            "entry_price": 1.263900,
        },
        {
            "trade_id": "FX006",
            "trade_date": "2026-09-18",
            "pair": "USDSGD",
            "side": "SHORT",
            "notional_base": 1_250_000,
            "entry_price": 1.265450,
        },

        # ====================================================
        # USDKRW — net LONG
        # ====================================================
        {
            "trade_id": "FX007",
            "trade_date": "2026-09-07",
            "pair": "USDKRW",
            "side": "LONG",
            "notional_base": 1_500_000,
            "entry_price": 1345.500000,
        },
        {
            "trade_id": "FX008",
            "trade_date": "2026-09-14",
            "pair": "USDKRW",
            "side": "SHORT",
            "notional_base": 500_000,
            "entry_price": 1352.250000,
        },
        {
            "trade_id": "FX009",
            "trade_date": "2026-09-21",
            "pair": "USDKRW",
            "side": "LONG",
            "notional_base": 1_000_000,
            "entry_price": 1348.170044,
        },

        # ====================================================
        # USDINR — net SHORT
        # ====================================================
        {
            "trade_id": "FX010",
            "trade_date": "2026-09-08",
            "pair": "USDINR",
            "side": "SHORT",
            "notional_base": 1_500_000,
            "entry_price": 95.620000,
        },
        {
            "trade_id": "FX011",
            "trade_date": "2026-09-15",
            "pair": "USDINR",
            "side": "LONG",
            "notional_base": 500_000,
            "entry_price": 95.910000,
        },
        {
            "trade_id": "FX012",
            "trade_date": "2026-09-22",
            "pair": "USDINR",
            "side": "SHORT",
            "notional_base": 1_000_000,
            "entry_price": 95.835403,
        },

        # ====================================================
        # USDIDR — net LONG
        # ====================================================
        {
            "trade_id": "FX013",
            "trade_date": "2026-09-09",
            "pair": "USDIDR",
            "side": "LONG",
            "notional_base": 1_250_000,
            "entry_price": 17680.000000,
        },
        {
            "trade_id": "FX014",
            "trade_date": "2026-09-16",
            "pair": "USDIDR",
            "side": "SHORT",
            "notional_base": 500_000,
            "entry_price": 17820.000000,
        },
        {
            "trade_id": "FX015",
            "trade_date": "2026-09-23",
            "pair": "USDIDR",
            "side": "LONG",
            "notional_base": 750_000,
            "entry_price": 17757.000000,
        },

        # ====================================================
        # USDTWD — net SHORT
        # ====================================================
        {
            "trade_id": "FX016",
            "trade_date": "2026-09-10",
            "pair": "USDTWD",
            "side": "SHORT",
            "notional_base": 1_750_000,
            "entry_price": 31.580000,
        },
        {
            "trade_id": "FX017",
            "trade_date": "2026-09-17",
            "pair": "USDTWD",
            "side": "LONG",
            "notional_base": 750_000,
            "entry_price": 31.720000,
        },
        {
            "trade_id": "FX018",
            "trade_date": "2026-09-24",
            "pair": "USDTWD",
            "side": "SHORT",
            "notional_base": 1_250_000,
            "entry_price": 31.679501,
        },

        # ====================================================
        # AUDUSD — net LONG
        # ====================================================
        {
            "trade_id": "FX019",
            "trade_date": "2026-09-11",
            "pair": "AUDUSD",
            "side": "LONG",
            "notional_base": 1_500_000,
            "entry_price": 0.698500,
        },
        {
            "trade_id": "FX020",
            "trade_date": "2026-09-18",
            "pair": "AUDUSD",
            "side": "SHORT",
            "notional_base": 500_000,
            "entry_price": 0.704200,
        },
        {
            "trade_id": "FX021",
            "trade_date": "2026-09-25",
            "pair": "AUDUSD",
            "side": "LONG",
            "notional_base": 1_000_000,
            "entry_price": 0.700910,
        },
    ]

    return pd.DataFrame(trades)


def main() -> None:
    portfolio = create_sample_portfolio()

    # This file is intended to live under src/, so parent.parent is
    # the project root and the CSV is written to data/portfolio.csv.
    output_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "portfolio.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    portfolio.to_csv(
        output_path,
        index=False,
    )

    print(f"Created sample portfolio: {output_path}")
    print(f"Trades: {len(portfolio)}")
    print(f"Unique FX pairs: {portfolio['pair'].nunique()}")
    print()
    print("Trades per pair:")
    print(portfolio.groupby("pair").size().to_string())
    print()
    print(portfolio.to_string(index=False))


if __name__ == "__main__":
    main()
