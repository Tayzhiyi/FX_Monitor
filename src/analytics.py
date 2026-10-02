import numpy as np
import pandas as pd

from src.market_data import get_completed_bar_cutoff_date
from src.pnl import calculate_mtm_usd


# ============================================================
# HELPERS
# ============================================================

def _normalise_daily_date(value):
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is not None:
        timestamp = (
            timestamp
            .tz_convert("UTC")
            .tz_localize(None)
        )

    return timestamp.normalize()


# ============================================================
# HISTORICAL DAILY P&L
# ============================================================

def calculate_historical_daily_pnl(
    portfolio,
    historical_prices,
    valuation_timestamp,
):
    """
    Reconstruct historical close-to-close daily USD P&L.

    Completed UTC daily bars only.
    """

    cutoff_date = get_completed_bar_cutoff_date(
        valuation_timestamp
    )

    records = []

    for _, trade in portfolio.iterrows():

        trade_id = trade["trade_id"]
        pair = trade["pair"]

        trade_date = (
            pd.Timestamp(trade["trade_date"])
            .normalize()
        )

        if pair not in historical_prices.columns:
            continue

        prices = (
            historical_prices[pair]
            .dropna()
            .copy()
            .sort_index()
        )

        if prices.empty:
            continue

        prices.index = pd.DatetimeIndex(
            [
                _normalise_daily_date(index)
                for index in prices.index
            ]
        )

        prices = prices[
            [
                (
                    index.date() < cutoff_date
                    and index >= trade_date
                )
                for index in prices.index
            ]
        ]

        if prices.empty:
            continue

        mtm = prices.apply(
            lambda spot: calculate_mtm_usd(
                pair=pair,
                side=trade["side"],
                notional_base=trade["notional_base"],
                entry_price=trade["entry_price"],
                current_price=float(spot),
            )
        )

        daily_pnl = pd.Series(
            np.nan,
            index=prices.index,
            dtype="float64",
        )

        # First observation is the starting reference point.
        daily_pnl.iloc[0] = 0.0

        for i in range(1, len(prices)):

            previous_date = prices.index[i - 1]
            current_date = prices.index[i]

            expected_previous_date = (
                current_date
                - pd.offsets.BDay(1)
            )

            # Do not bridge a missing weekday.
            if previous_date == expected_previous_date:
                daily_pnl.iloc[i] = (
                    mtm.iloc[i]
                    - mtm.iloc[i - 1]
                )

        for date in prices.index:

            records.append(
                {
                    "date": date,
                    "trade_id": trade_id,
                    "pair": pair,
                    "daily_pnl_usd": daily_pnl.loc[date],
                    "mtm_usd": mtm.loc[date],
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "date",
                "trade_id",
                "pair",
                "daily_pnl_usd",
                "mtm_usd",
            ]
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            ["date", "trade_id"]
        )
        .reset_index(drop=True)
    )


# ============================================================
# P&L MATRIX
# ============================================================

def build_selected_pnl_matrix(
    pnl_history,
    portfolio,
    selected_trade_ids,
):
    """
    Create date x trade matrix of daily P&L.

    Before a trade exists:
        contribution = 0

    Missing data after trade inception:
        remains NaN
    """

    if (
        not selected_trade_ids
        or pnl_history.empty
    ):
        return pd.DataFrame()

    selected_trade_ids = list(
        selected_trade_ids
    )

    selected_portfolio = (
        portfolio[
            portfolio["trade_id"].isin(
                selected_trade_ids
            )
        ]
        .copy()
    )

    if selected_portfolio.empty:
        return pd.DataFrame()

    filtered_history = (
        pnl_history[
            pnl_history["trade_id"].isin(
                selected_trade_ids
            )
        ]
    )

    if filtered_history.empty:
        return pd.DataFrame()

    pivot = (
        filtered_history
        .pivot(
            index="date",
            columns="trade_id",
            values="daily_pnl_usd",
        )
        .sort_index()
    )

    earliest_trade_date = (
        selected_portfolio["trade_date"]
        .min()
    )

    latest_history_date = (
        filtered_history["date"]
        .max()
    )

    master_index = pd.bdate_range(
        start=earliest_trade_date,
        end=latest_history_date,
    )

    pivot = pivot.reindex(
        index=master_index,
        columns=selected_trade_ids,
    )

    trade_dates = (
        selected_portfolio
        .set_index("trade_id")[
            "trade_date"
        ]
    )

    for trade_id in selected_trade_ids:

        if trade_id not in trade_dates.index:
            continue

        trade_date = (
            pd.Timestamp(
                trade_dates.loc[trade_id]
            )
            .normalize()
        )

        # Before inception the trade contributes zero.
        pivot.loc[
            pivot.index < trade_date,
            trade_id,
        ] = 0.0

    pivot.index.name = "date"

    return pivot


# ============================================================
# PORTFOLIO DAILY P&L
# ============================================================

def get_portfolio_daily_pnl(
    pnl_matrix,
):
    if (
        pnl_matrix is None
        or pnl_matrix.empty
    ):
        return pd.Series(
            dtype="float64"
        )

    return (
        pnl_matrix
        .sum(
            axis=1,
            min_count=len(
                pnl_matrix.columns
            ),
        )
        .dropna()
    )


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_pnl_performance_metrics(
    pnl_matrix,
):
    empty_result = {
        "max_drawdown_usd": None,
        "best_day_usd": None,
        "worst_day_usd": None,
        "best_day_date": None,
        "worst_day_date": None,
    }

    portfolio_daily_pnl = (
        get_portfolio_daily_pnl(
            pnl_matrix
        )
    )

    if portfolio_daily_pnl.empty:
        return empty_result

    cumulative_pnl = (
        portfolio_daily_pnl
        .cumsum()
    )

    running_peak = (
        cumulative_pnl
        .cummax()
        .clip(lower=0.0)
    )

    drawdown = (
        cumulative_pnl
        - running_peak
    )

    best_date = (
        portfolio_daily_pnl.idxmax()
    )

    worst_date = (
        portfolio_daily_pnl.idxmin()
    )

    return {
        "max_drawdown_usd":
            float(drawdown.min()),

        "best_day_usd":
            float(
                portfolio_daily_pnl.max()
            ),

        "worst_day_usd":
            float(
                portfolio_daily_pnl.min()
            ),

        "best_day_date":
            best_date,

        "worst_day_date":
            worst_date,
    }


# ============================================================
# SHARPE / SORTINO
# ============================================================

def calculate_sharpe_sortino(
    pnl_matrix,
    annualisation_factor=252,
):
    """
    Annualised P&L-based Sharpe and Sortino.

    Uses completed daily P&L only.
    Benchmark / downside target = USD 0.
    """

    portfolio_daily_pnl = (
        get_portfolio_daily_pnl(
            pnl_matrix
        )
    )

    if len(portfolio_daily_pnl) < 2:
        return {
            "sharpe": None,
            "sortino": None,
            "observations":
                len(portfolio_daily_pnl),
        }

    mean_daily_pnl = (
        portfolio_daily_pnl.mean()
    )

    daily_pnl_vol = (
        portfolio_daily_pnl.std(
            ddof=1
        )
    )

    if (
        not np.isfinite(daily_pnl_vol)
        or daily_pnl_vol == 0
    ):
        sharpe = None

    else:
        sharpe = (
            mean_daily_pnl
            / daily_pnl_vol
            * np.sqrt(
                annualisation_factor
            )
        )

    downside_pnl = (
        portfolio_daily_pnl
        .clip(upper=0.0)
    )

    downside_deviation = np.sqrt(
        np.mean(
            downside_pnl ** 2
        )
    )

    if (
        not np.isfinite(
            downside_deviation
        )
        or downside_deviation == 0
    ):
        sortino = None

    else:
        sortino = (
            mean_daily_pnl
            / downside_deviation
            * np.sqrt(
                annualisation_factor
            )
        )

    return {
        "sharpe":
            (
                float(sharpe)
                if sharpe is not None
                else None
            ),

        "sortino":
            (
                float(sortino)
                if sortino is not None
                else None
            ),

        "observations":
            len(portfolio_daily_pnl),
    }


# ============================================================
# STRESS TEST
# ============================================================

def calculate_stress_scenarios(
    pnl,
    expected_trade_ids=None,
):
    """
    Exact revaluation under coherent USD shocks.

    The stress test is a full-book metric. If any expected
    position is missing or has an unavailable valuation, no
    partial stress number is returned.

    For USD-base:
        USD strength -> spot increases

    For USD-quote:
        USD strength -> spot decreases
    """

    output_columns = [
        "scenario",
        "stress_pnl_usd",
    ]

    if (
        pnl is None
        or pnl.empty
    ):
        return pd.DataFrame(
            columns=output_columns
        )

    working = pnl.copy()

    required_columns = {
        "trade_id",
        "valuation_status",
        "pair",
        "side",
        "notional_base",
        "entry_price",
        "current_spot",
    }

    if not required_columns.issubset(
        working.columns
    ):
        return pd.DataFrame(
            columns=output_columns
        )

    if expected_trade_ids is not None:

        expected_trade_ids = set(
            expected_trade_ids
        )

        observed_trade_ids = set(
            working["trade_id"].tolist()
        )

        if observed_trade_ids != expected_trade_ids:
            return pd.DataFrame(
                columns=output_columns
            )

    if not (
        working["valuation_status"]
        .eq("OK")
        .all()
    ):
        return pd.DataFrame(
            columns=output_columns
        )

    for column in [
        "notional_base",
        "entry_price",
        "current_spot",
    ]:
        working[column] = (
            pd.to_numeric(
                working[column],
                errors="coerce",
            )
        )

    if (
        working[
            [
                "notional_base",
                "entry_price",
                "current_spot",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        return pd.DataFrame(
            columns=output_columns
        )

    scenarios = [
        (
            "USD strengthens 1%",
            0.01,
        ),
        (
            "USD weakens 1%",
            -0.01,
        ),
        (
            "USD strengthens 2%",
            0.02,
        ),
        (
            "USD weakens 2%",
            -0.02,
        ),
    ]

    results = []

    for (
        scenario_name,
        usd_shock,
    ) in scenarios:

        portfolio_stress_pnl = 0.0

        for _, row in working.iterrows():

            pair = row["pair"]

            base = pair[:3]
            quote = pair[3:]

            current_spot = float(
                row["current_spot"]
            )

            # ------------------------------------------------
            # Convert USD macro shock into pair spot shock
            # ------------------------------------------------

            if base == "USD":

                pair_shock = usd_shock

            elif quote == "USD":

                pair_shock = -usd_shock

            else:

                pair_shock = 0.0

            shocked_spot = (
                current_spot
                * (
                    1.0
                    + pair_shock
                )
            )

            current_mtm = (
                calculate_mtm_usd(
                    pair=pair,
                    side=row["side"],
                    notional_base=row[
                        "notional_base"
                    ],
                    entry_price=row[
                        "entry_price"
                    ],
                    current_price=(
                        current_spot
                    ),
                )
            )

            shocked_mtm = (
                calculate_mtm_usd(
                    pair=pair,
                    side=row["side"],
                    notional_base=row[
                        "notional_base"
                    ],
                    entry_price=row[
                        "entry_price"
                    ],
                    current_price=(
                        shocked_spot
                    ),
                )
            )

            portfolio_stress_pnl += (
                shocked_mtm
                - current_mtm
            )

        results.append(
            {
                "scenario":
                    scenario_name,

                "stress_pnl_usd":
                    float(
                        portfolio_stress_pnl
                    ),
            }
        )

    return pd.DataFrame(results)