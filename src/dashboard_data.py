from src.market_data import (
    load_portfolio,
    get_all_historical_prices,
    get_completed_bar_cutoff_date,
)


# ============================================================
# PORTFOLIO
# ============================================================

PORTFOLIO = load_portfolio()

PAIRS = (
    PORTFOLIO["pair"]
    .unique()
)


# ============================================================
# HISTORICAL DATA CACHE
# ============================================================

HISTORICAL_PRICES = None
HISTORY_CUTOFF_DATE = None


def get_dashboard_history(
    valuation_timestamp,
):
    """
    Return cached historical FX prices.

    Historical daily data is refreshed only when:
    - no cache exists, or
    - the UTC daily-bar cutoff date changes.

    The manual dashboard refresh button intentionally does
    not redownload three years of daily history. Current
    spot prices are refreshed separately by the live P&L
    calculation on every dashboard refresh.
    """

    global HISTORICAL_PRICES
    global HISTORY_CUTOFF_DATE

    current_cutoff = (
        get_completed_bar_cutoff_date(
            valuation_timestamp
        )
    )

    needs_refresh = (
        HISTORICAL_PRICES is None
        or HISTORY_CUTOFF_DATE != current_cutoff
    )

    if needs_refresh:

        # Build a complete candidate first. Do not mutate the
        # shared cache while downloads are still in progress.
        refreshed_history = (
            get_all_historical_prices(
                PAIRS,
                period="3y",
            )
        )

        # If a refresh completely fails and an older cache
        # exists, retain the last known cache rather than
        # replacing it with an empty DataFrame.
        if (
            refreshed_history.empty
            and HISTORICAL_PRICES is not None
        ):
            return HISTORICAL_PRICES

        HISTORICAL_PRICES = (
            refreshed_history
        )

        HISTORY_CUTOFF_DATE = (
            current_cutoff
        )

    return HISTORICAL_PRICES


def get_risk_historical_prices(
    historical_prices,
    valid_pnl,
):
    """
    Restrict historical prices to risk factors
    corresponding to positions with valid valuations.

    Preserve any market-data error metadata.
    """

    risk_pairs = (
        valid_pnl["pair"]
        .unique()
        .tolist()
    )

    risk_history = (
        historical_prices[
            risk_pairs
        ].copy()
    )

    original_errors = (
        historical_prices
        .attrs
        .get(
            "errors",
            {},
        )
    )

    risk_history.attrs[
        "errors"
    ] = {
        pair: error
        for pair, error
        in original_errors.items()
        if pair in risk_pairs
    }

    return risk_history
