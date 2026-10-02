import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# CONFIGURATION
# ============================================================

# Singapore timezone is used for dashboard presentation.
# Daily P&L and Yahoo daily-bar cutoffs use UTC.
REPORTING_TIMEZONE = "Asia/Singapore"

# Hard safety limit for an intraday quote.
# Quotes can be labelled STALE much earlier in the dashboard,
# but are only rejected after 24 hours.
MAX_INTRADAY_QUOTE_AGE_MINUTES = 24 * 60

# Sanity check against the previous completed daily close.
# A move larger than this is treated as a likely bad tick for
# this demonstration portfolio rather than trusted automatically.
MAX_CURRENT_PRICE_MOVE_FROM_PREVIOUS_CLOSE = 0.20


REQUIRED_PORTFOLIO_COLUMNS = {
    "trade_id",
    "trade_date",
    "pair",
    "side",
    "notional_base",
    "entry_price",
}


# ============================================================
# FX TICKER HELPERS
# ============================================================

def pair_to_yahoo_ticker(pair):
    """
    Convert a standard six-character FX pair into
    a Yahoo Finance ticker.

    Examples:
        USDJPY -> JPY=X
        USDSGD -> SGD=X
        USDKRW -> KRW=X
        AUDUSD -> AUDUSD=X

    A valid ticker format does not guarantee that Yahoo
    provides usable data for the pair.
    """

    pair = str(pair).upper().strip()

    if len(pair) != 6 or not pair.isalpha():
        raise ValueError(
            f"Invalid FX pair format: {pair}"
        )

    base = pair[:3]
    quote = pair[3:]

    if base == "USD":
        return f"{quote}=X"

    return f"{pair}=X"


# ============================================================
# TIME HELPERS
# ============================================================

def to_utc_timestamp(timestamp):
    """
    Convert a timestamp into timezone-aware UTC.
    """

    timestamp = pd.Timestamp(timestamp)

    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")

    return timestamp.tz_convert("UTC")


def get_reporting_date(timestamp):
    """
    Return the Singapore calendar date.

    This is used for presentation only.
    """

    timestamp = to_utc_timestamp(
        timestamp
    )

    return (
        timestamp
        .tz_convert(REPORTING_TIMEZONE)
        .date()
    )


def get_completed_bar_cutoff_date(
    valuation_timestamp,
):
    """
    Return the UTC date used for daily-bar completion.

    Assumption:
    A Yahoo daily FX bar dated D is considered complete
    only once UTC has rolled into D + 1.

    Therefore only bars satisfying

        bar_date < current UTC date

    are considered completed.
    """

    valuation_timestamp = to_utc_timestamp(
        valuation_timestamp
    )

    return valuation_timestamp.date()


def get_expected_previous_fx_business_date(
    valuation_timestamp,
):
    """
    Return the expected previous FX business date using
    a generic Monday-Friday calendar.

    Pair-specific holidays are not modelled.
    """

    cutoff_date = (
        get_completed_bar_cutoff_date(
            valuation_timestamp
        )
    )

    cutoff_timestamp = pd.Timestamp(
        cutoff_date
    )

    previous_business_date = (
        cutoff_timestamp
        - pd.offsets.BDay(1)
    )

    return previous_business_date.date()


# ============================================================
# PORTFOLIO
# ============================================================

def load_portfolio(
    filepath="data/portfolio.csv",
):
    """
    Load and validate the portfolio CSV.
    """

    portfolio = pd.read_csv(
        filepath
    )

    missing_columns = (
        REQUIRED_PORTFOLIO_COLUMNS
        - set(portfolio.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required portfolio columns: "
            f"{sorted(missing_columns)}"
        )

    if portfolio.empty:
        return portfolio

    # --------------------------------------------------------
    # Normalise text
    # --------------------------------------------------------

    portfolio["trade_id"] = (
        portfolio["trade_id"]
        .astype(str)
        .str.strip()
    )

    portfolio["pair"] = (
        portfolio["pair"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    portfolio["side"] = (
        portfolio["side"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # --------------------------------------------------------
    # Trade dates
    # --------------------------------------------------------

    portfolio["trade_date"] = (
        pd.to_datetime(
            portfolio["trade_date"],
            errors="raise",
        )
    )

    # --------------------------------------------------------
    # Trade IDs
    # --------------------------------------------------------

    if (
        portfolio["trade_id"]
        .duplicated()
        .any()
    ):

        duplicates = portfolio.loc[
            portfolio["trade_id"].duplicated(
                keep=False
            ),
            "trade_id",
        ].tolist()

        raise ValueError(
            f"Duplicate trade IDs found: {duplicates}"
        )

    # --------------------------------------------------------
    # FX pairs
    # --------------------------------------------------------

    valid_pairs = (
        portfolio["pair"]
        .str.match(
            r"^[A-Z]{6}$"
        )
    )

    if not valid_pairs.all():

        bad_pairs = portfolio.loc[
            ~valid_pairs,
            "pair",
        ].tolist()

        raise ValueError(
            f"Invalid FX pair format: {bad_pairs}"
        )

    for pair in portfolio["pair"].unique():
        pair_to_yahoo_ticker(
            pair
        )

    # --------------------------------------------------------
    # Side
    # --------------------------------------------------------

    valid_sides = (
        portfolio["side"]
        .isin(
            ["LONG", "SHORT"]
        )
    )

    if not valid_sides.all():

        bad_sides = portfolio.loc[
            ~valid_sides,
            "side",
        ].tolist()

        raise ValueError(
            f"Invalid trade sides: {bad_sides}. "
            "Use LONG or SHORT."
        )

    # --------------------------------------------------------
    # Numeric fields
    # --------------------------------------------------------

    for column in [
        "notional_base",
        "entry_price",
    ]:

        portfolio[column] = (
            pd.to_numeric(
                portfolio[column],
                errors="coerce",
            )
        )

        if not np.isfinite(
            portfolio[column]
        ).all():

            raise ValueError(
                f"{column} contains missing "
                "or non-finite values."
            )

    if (
        portfolio["notional_base"]
        <= 0
    ).any():

        raise ValueError(
            "All notionals must be positive. "
            "Use side to represent direction."
        )

    if (
        portfolio["entry_price"]
        <= 0
    ).any():

        raise ValueError(
            "All entry prices must be positive."
        )

    return portfolio


# ============================================================
# HISTORICAL MARKET DATA
# ============================================================

def get_historical_prices(
    pair,
    period="2y",
):
    """
    Download historical daily closing prices
    for one FX pair.
    """

    ticker = pair_to_yahoo_ticker(
        pair
    )

    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            f"No historical data returned "
            f"for {pair} ({ticker})"
        )

    close = data["Close"]

    if isinstance(
        close,
        pd.DataFrame,
    ):
        close = close.iloc[:, 0]

    close = close.dropna()

    close = close[
        np.isfinite(close)
        & (close > 0)
    ]

    if close.empty:
        raise ValueError(
            f"No valid historical prices "
            f"returned for {pair}"
        )

    close.name = pair

    return close


def get_all_historical_prices(
    pairs,
    period="2y",
):
    """
    Download historical data for every unique pair.

    A failure for one pair does not abort the entire
    portfolio download.

    Failed pairs remain present as empty columns.

    Any download errors are stored in:

        prices.attrs["errors"]
    """

    unique_pairs = list(
        pd.unique(pairs)
    )

    if not unique_pairs:
        return pd.DataFrame()

    series_list = []
    errors = {}

    for pair in unique_pairs:

        try:

            series = (
                get_historical_prices(
                    pair=pair,
                    period=period,
                )
            )

        except Exception as exc:

            errors[pair] = str(
                exc
            )

            series = pd.Series(
                dtype="float64",
                name=pair,
            )

        series_list.append(
            series
        )

    prices = pd.concat(
        series_list,
        axis=1,
    )

    prices.attrs["errors"] = (
        errors
    )

    return prices


# ============================================================
# CURRENT MARKET DATA
# ============================================================

def get_current_spot(pair):
    """
    Get the latest available market price.

    Priority:
        1. Yahoo 1-minute intraday data
        2. Latest daily observation as fallback

    Returns:
        price
        quote timestamp in UTC
        source
    """

    ticker = pair_to_yahoo_ticker(
        pair
    )

    # --------------------------------------------------------
    # Intraday data
    # --------------------------------------------------------

    intraday = yf.download(
        ticker,
        period="1d",
        interval="1m",
        auto_adjust=False,
        progress=False,
    )

    if not intraday.empty:

        close = intraday["Close"]

        if isinstance(
            close,
            pd.DataFrame,
        ):
            close = close.iloc[:, 0]

        close = close.dropna()

        close = close[
            np.isfinite(close)
            & (close > 0)
        ]

        if not close.empty:

            price = float(
                close.iloc[-1]
            )

            timestamp = (
                to_utc_timestamp(
                    close.index[-1]
                )
            )

            return (
                price,
                timestamp,
                "intraday_1m",
            )

    # --------------------------------------------------------
    # Daily fallback
    # --------------------------------------------------------

    daily = yf.download(
        ticker,
        period="5d",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if daily.empty:
        raise ValueError(
            f"No current market data "
            f"returned for {pair}"
        )

    close = daily["Close"]

    if isinstance(
        close,
        pd.DataFrame,
    ):
        close = close.iloc[:, 0]

    close = close.dropna()

    close = close[
        np.isfinite(close)
        & (close > 0)
    ]

    if close.empty:
        raise ValueError(
            f"No valid current price "
            f"returned for {pair}"
        )

    price = float(
        close.iloc[-1]
    )

    timestamp = (
        to_utc_timestamp(
            close.index[-1]
        )
    )

    return (
        price,
        timestamp,
        "daily_fallback",
    )


def get_current_spots(pairs):
    """
    Fetch every pair once.

    A failed market-data request affects only that pair
    instead of aborting the entire portfolio refresh.
    """

    unique_pairs = list(
        pd.unique(pairs)
    )

    spots = {}

    for pair in unique_pairs:

        try:

            (
                price,
                timestamp,
                source,
            ) = get_current_spot(
                pair
            )

            spots[pair] = {
                "price": price,
                "timestamp": timestamp,
                "source": source,
                "error": None,
            }

        except Exception as exc:

            spots[pair] = {
                "price": np.nan,
                "timestamp": pd.NaT,
                "source": None,
                "error": str(exc),
            }

    return spots


# ============================================================
# PREVIOUS CLOSE
# ============================================================

def get_previous_close(
    pair,
    historical_prices,
    valuation_timestamp,
):
    """
    Return the latest completed daily close.

    The returned close must match the expected previous
    Monday-Friday business date.

    This prevents an older observation from silently being
    substituted when recent historical data is missing.

    Pair-specific holidays are not modelled.
    """

    if pair not in historical_prices.columns:

        raise ValueError(
            f"No historical prices available "
            f"for {pair}"
        )

    price_series = (
        historical_prices[pair]
        .dropna()
        .copy()
        .sort_index()
    )

    if price_series.empty:

        history_errors = (
            historical_prices.attrs.get(
                "errors",
                {},
            )
        )

        error_detail = (
            history_errors.get(pair)
        )

        if error_detail:

            raise ValueError(
                f"Historical market data unavailable "
                f"for {pair}: {error_detail}"
            )

        raise ValueError(
            f"No historical prices available "
            f"for {pair}"
        )

    cutoff_date = (
        get_completed_bar_cutoff_date(
            valuation_timestamp
        )
    )

    eligible_indices = [
        index
        for index in price_series.index
        if (
            pd.Timestamp(index).date()
            < cutoff_date
        )
    ]

    if not eligible_indices:

        raise ValueError(
            f"No completed daily close "
            f"available for {pair} "
            f"before UTC cutoff {cutoff_date}"
        )

    previous_index = (
        eligible_indices[-1]
    )

    previous_close_date = (
        pd.Timestamp(
            previous_index
        )
    )

    expected_date = (
        get_expected_previous_fx_business_date(
            valuation_timestamp
        )
    )

    actual_date = (
        previous_close_date.date()
    )

    if actual_date != expected_date:

        raise ValueError(
            f"{pair}: expected previous FX "
            f"business-day close for "
            f"{expected_date}, but latest "
            f"available completed close is "
            f"{actual_date}. "
            "Daily P&L baseline unavailable."
        )

    previous_close = float(
        price_series.loc[
            previous_index
        ]
    )

    return (
        previous_close,
        previous_close_date,
    )


# ============================================================
# QUOTE VALIDATION
# ============================================================

def quote_age_minutes(
    quote_timestamp,
    valuation_timestamp,
):
    """
    Measure quote age in minutes.
    """

    quote_timestamp = (
        to_utc_timestamp(
            quote_timestamp
        )
    )

    valuation_timestamp = (
        to_utc_timestamp(
            valuation_timestamp
        )
    )

    age = (
        valuation_timestamp
        - quote_timestamp
    )

    return (
        age.total_seconds()
        / 60
    )


def validate_current_quote(
    pair,
    current_price,
    previous_close,
    quote_timestamp,
    market_source,
    valuation_timestamp,
    previous_close_date,
):
    """
    Validate whether a market observation is suitable
    for a current daily valuation.

    Checks:
    - current and previous prices must be finite and positive
    - current price must not be an implausible jump from previous close
    - quote must not materially exceed valuation time
    - quote must be later than the previous-close baseline
    - intraday quote must be less than 24 hours old

    A failed check should make that position unavailable,
    not abort the entire portfolio.
    """

    current_price = float(current_price)
    previous_close = float(previous_close)

    if (
        not np.isfinite(current_price)
        or current_price <= 0
        or not np.isfinite(previous_close)
        or previous_close <= 0
    ):
        raise ValueError(
            f"{pair}: current or previous price is invalid."
        )

    price_move = (
        current_price
        / previous_close
        - 1.0
    )

    if (
        abs(price_move)
        > MAX_CURRENT_PRICE_MOVE_FROM_PREVIOUS_CLOSE
    ):
        raise ValueError(
            f"{pair}: latest market price implies a "
            f"{price_move:.1%} move from the previous "
            "completed close, exceeding the "
            f"{MAX_CURRENT_PRICE_MOVE_FROM_PREVIOUS_CLOSE:.0%} "
            "sanity threshold. Quote rejected as a "
            "possible bad tick."
        )

    quote_timestamp = (
        to_utc_timestamp(
            quote_timestamp
        )
    )

    valuation_timestamp = (
        to_utc_timestamp(
            valuation_timestamp
        )
    )

    future_tolerance = (
        pd.Timedelta(
            minutes=5
        )
    )

    if (
        quote_timestamp
        > valuation_timestamp
        + future_tolerance
    ):

        raise ValueError(
            f"{pair}: market quote timestamp "
            f"{quote_timestamp} is after "
            f"valuation timestamp "
            f"{valuation_timestamp}."
        )

    previous_close_date = (
        pd.Timestamp(
            previous_close_date
        ).date()
    )

    # Current price must be chronologically later
    # than the selected baseline.
    if (
        quote_timestamp.date()
        <= previous_close_date
    ):

        raise ValueError(
            f"{pair}: latest market quote "
            f"({quote_timestamp}) is not newer "
            f"than previous completed daily "
            f"baseline ({previous_close_date})."
        )

    if market_source == "intraday_1m":

        age_minutes = (
            quote_age_minutes(
                quote_timestamp,
                valuation_timestamp,
            )
        )

        if age_minutes < -5:

            raise ValueError(
                f"{pair}: invalid negative "
                "quote age."
            )

        if (
            age_minutes
            > MAX_INTRADAY_QUOTE_AGE_MINUTES
        ):

            raise ValueError(
                f"{pair}: latest intraday quote "
                f"is {age_minutes:.0f} minutes old, "
                "exceeding the hard freshness "
                f"limit of "
                f"{MAX_INTRADAY_QUOTE_AGE_MINUTES} "
                "minutes."
            )


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    portfolio = (
        load_portfolio()
    )

    pairs = (
        portfolio["pair"]
        .unique()
    )

    historical_prices = (
        get_all_historical_prices(
            pairs,
            period="2y",
        )
    )

    valuation_timestamp = (
        pd.Timestamp.now(
            tz="UTC"
        )
    )

    current_spots = (
        get_current_spots(
            pairs
        )
    )

    print(
        "\nPORTFOLIO"
    )

    print(
        portfolio
    )

    print(
        "\nVALUATION TIMESTAMP:",
        valuation_timestamp,
    )

    print(
        "\nCURRENT MARKET DATA"
    )

    for pair in pairs:

        market = (
            current_spots[pair]
        )

        if market["error"] is not None:

            print(
                f"{pair}: UNAVAILABLE - "
                f"{market['error']}"
            )

            continue

        try:

            (
                previous_close,
                previous_date,
            ) = get_previous_close(
                pair=pair,
                historical_prices=historical_prices,
                valuation_timestamp=valuation_timestamp,
            )

            validate_current_quote(
                pair=pair,
                current_price=market[
                    "price"
                ],
                previous_close=previous_close,
                quote_timestamp=market[
                    "timestamp"
                ],
                market_source=market[
                    "source"
                ],
                valuation_timestamp=valuation_timestamp,
                previous_close_date=previous_date,
            )

            if (
                market["source"]
                == "intraday_1m"
            ):

                age = (
                    quote_age_minutes(
                        market[
                            "timestamp"
                        ],
                        valuation_timestamp,
                    )
                )

                age_text = (
                    f"{age:.1f} minutes"
                )

            else:

                age_text = "N/A"

            print(
                f"{pair}: "
                f"spot={market['price']:.6f}, "
                f"previous_close="
                f"{previous_close:.6f}, "
                f"previous_close_date="
                f"{previous_date.date()}, "
                f"source="
                f"{market['source']}, "
                f"quote_age="
                f"{age_text}"
            )

        except ValueError as exc:

            print(
                f"{pair}: UNAVAILABLE - {exc}"
            )