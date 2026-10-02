import numpy as np
import pandas as pd

from src.market_data import (
    REPORTING_TIMEZONE,
    load_portfolio,
    get_all_historical_prices,
    get_current_spots,
    get_previous_close,
    quote_age_minutes,
    validate_current_quote,
)


OUTPUT_COLUMNS = [
    "trade_id",
    "trade_date",
    "pair",
    "side",
    "notional_base",
    "entry_price",
    "previous_close_date",
    "previous_close",
    "position_existed_at_baseline",
    "current_spot",
    "previous_mtm_usd",
    "daily_pnl_usd",
    "inception_pnl_usd",
    "quote_timestamp",
    "market_source",
    "quote_age_minutes",
    "valuation_status",
    "valuation_error",
]


# ============================================================
# BASIC HELPERS
# ============================================================

def side_multiplier(side):
    """
    Convert LONG / SHORT to +1 / -1.
    """

    side = str(side).upper()

    if side == "LONG":
        return 1

    if side == "SHORT":
        return -1

    raise ValueError(
        f"Invalid trade side: {side}"
    )


def normalize_valuation_timestamp(
    valuation_timestamp,
):
    """
    Convert valuation timestamp into timezone-aware UTC.
    """

    if valuation_timestamp is None:

        return pd.Timestamp.now(
            tz="UTC"
        )

    valuation_timestamp = (
        pd.Timestamp(
            valuation_timestamp
        )
    )

    if valuation_timestamp.tzinfo is None:

        return (
            valuation_timestamp
            .tz_localize("UTC")
        )

    return (
        valuation_timestamp
        .tz_convert("UTC")
    )


# ============================================================
# POSITION VALUATION
# ============================================================

def calculate_mtm_usd(
    pair,
    side,
    notional_base,
    entry_price,
    current_price,
):
    """
    Calculate inception mark-to-market P&L in USD.

    Assumptions:
    - notional is expressed in base currency
    - USD is either base or quote currency
    - original currency legs remain outstanding
    - no financing
    - no carry
    - no transaction costs
    - no intervening cash flows
    """

    sign = side_multiplier(
        side
    )

    base_currency = pair[:3]
    quote_currency = pair[3:]

    # --------------------------------------------------------
    # USD quote pair
    #
    # AUDUSD:
    #
    # V(S) = d * N * (S - K)
    # --------------------------------------------------------

    if quote_currency == "USD":

        pnl_usd = (
            sign
            * notional_base
            * (
                current_price
                - entry_price
            )
        )

    # --------------------------------------------------------
    # USD base pair
    #
    # USDJPY:
    #
    # V(S) = d * N * (1 - K / S)
    # --------------------------------------------------------

    elif base_currency == "USD":

        pnl_usd = (
            sign
            * notional_base
            * (
                1
                - entry_price
                / current_price
            )
        )

    else:

        raise ValueError(
            f"{pair} does not contain USD. "
            "Cross-currency P&L conversion "
            "is not supported."
        )

    return float(
        pnl_usd
    )


# ============================================================
# PORTFOLIO P&L
# ============================================================

def calculate_portfolio_pnl(
    portfolio,
    historical_prices,
    valuation_timestamp=None,
):
    """
    Calculate current and daily USD P&L.

    Intended for current/live valuation only.

    Daily P&L:

        current MTM
        -
        MTM at previous completed UTC daily bar

    Data failures are handled per position. One unavailable
    pair therefore does not prevent the remainder of the
    portfolio from being displayed.
    """

    valuation_timestamp = (
        normalize_valuation_timestamp(
            valuation_timestamp
        )
    )

    # --------------------------------------------------------
    # Current valuation only
    # --------------------------------------------------------

    actual_now = pd.Timestamp.now(
        tz="UTC"
    )

    valuation_difference_minutes = abs(
        (
            actual_now
            - valuation_timestamp
        ).total_seconds()
        / 60
    )

    if (
        valuation_difference_minutes
        > 60
    ):

        raise ValueError(
            "calculate_portfolio_pnl supports "
            "current/live valuations only. "
            "Historical as-of valuation is "
            "not implemented."
        )

    if portfolio.empty:

        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    portfolio_valuation_date = (
        valuation_timestamp.date()
    )

    # --------------------------------------------------------
    # Fetch each market once.
    #
    # Individual pair failures are stored rather than raised.
    # --------------------------------------------------------

    current_spots = (
        get_current_spots(
            portfolio[
                "pair"
            ].unique()
        )
    )

    results = []

    # ========================================================
    # POSITION LOOP
    # ========================================================

    for _, trade in (
        portfolio.iterrows()
    ):

        pair = trade["pair"]

        trade_date = (
            pd.Timestamp(
                trade[
                    "trade_date"
                ]
            ).date()
        )

        # ----------------------------------------------------
        # Future trade
        # ----------------------------------------------------

        if (
            trade_date
            > portfolio_valuation_date
        ):

            print(
                f"Skipping "
                f"{trade['trade_id']} "
                f"({pair}): trade date "
                f"{trade_date} is after "
                f"valuation date "
                f"{portfolio_valuation_date}"
            )

            continue

        market = (
            current_spots[pair]
        )

        # ----------------------------------------------------
        # Current quote download failed
        # ----------------------------------------------------

        if (
            market.get("error")
            is not None
        ):

            results.append(
                {
                    "trade_id":
                        trade[
                            "trade_id"
                        ],

                    "trade_date":
                        pd.Timestamp(
                            trade[
                                "trade_date"
                            ]
                        ),

                    "pair":
                        pair,

                    "side":
                        trade["side"],

                    "notional_base":
                        float(
                            trade[
                                "notional_base"
                            ]
                        ),

                    "entry_price":
                        float(
                            trade[
                                "entry_price"
                            ]
                        ),

                    "previous_close_date":
                        pd.NaT,

                    "previous_close":
                        np.nan,

                    "position_existed_at_baseline":
                        None,

                    "current_spot":
                        np.nan,

                    "previous_mtm_usd":
                        np.nan,

                    "daily_pnl_usd":
                        np.nan,

                    "inception_pnl_usd":
                        np.nan,

                    "quote_timestamp":
                        pd.NaT,

                    "market_source":
                        None,

                    "quote_age_minutes":
                        np.nan,

                    "valuation_status":
                        "UNAVAILABLE",

                    "valuation_error":
                        (
                            "Current market data "
                            "unavailable: "
                            + market["error"]
                        ),
                }
            )

            continue

        # ----------------------------------------------------
        # Current market observation
        # ----------------------------------------------------

        current_spot = float(
            market["price"]
        )

        quote_timestamp = (
            market["timestamp"]
        )

        market_source = (
            market["source"]
        )

        # ----------------------------------------------------
        # Quote age
        # ----------------------------------------------------

        if (
            market_source
            == "intraday_1m"
        ):

            quote_age = (
                quote_age_minutes(
                    quote_timestamp,
                    valuation_timestamp,
                )
            )

        else:

            quote_age = None

        # ----------------------------------------------------
        # Last-observed inception MTM
        #
        # Calculated before chronology validation so the
        # dashboard can still show the latest observation
        # even where daily P&L cannot safely be calculated.
        # ----------------------------------------------------

        current_mtm = (
            calculate_mtm_usd(
                pair=pair,
                side=trade["side"],
                notional_base=trade[
                    "notional_base"
                ],
                entry_price=trade[
                    "entry_price"
                ],
                current_price=current_spot,
            )
        )

        # Defaults if daily valuation cannot be completed
        previous_close = np.nan
        previous_close_date = pd.NaT
        previous_mtm = np.nan
        daily_pnl = np.nan

        position_existed_at_baseline = (
            None
        )

        valuation_status = (
            "OK"
        )

        valuation_error = (
            None
        )

        # ----------------------------------------------------
        # Baseline + quote validation
        # ----------------------------------------------------

        try:

            (
                previous_close,
                previous_close_date,
            ) = get_previous_close(
                pair=pair,
                historical_prices=historical_prices,
                valuation_timestamp=valuation_timestamp,
            )

            baseline_date = (
                pd.Timestamp(
                    previous_close_date
                ).date()
            )

            position_existed_at_baseline = (
                trade_date
                <= baseline_date
            )

            validate_current_quote(
                pair=pair,
                current_price=current_spot,
                previous_close=previous_close,
                quote_timestamp=quote_timestamp,
                market_source=market_source,
                valuation_timestamp=valuation_timestamp,
                previous_close_date=previous_close_date,
            )

            # ------------------------------------------------
            # Previous MTM
            # ------------------------------------------------

            if (
                position_existed_at_baseline
            ):

                previous_mtm = (
                    calculate_mtm_usd(
                        pair=pair,
                        side=trade[
                            "side"
                        ],
                        notional_base=trade[
                            "notional_base"
                        ],
                        entry_price=trade[
                            "entry_price"
                        ],
                        current_price=previous_close,
                    )
                )

            else:

                # Position did not exist at the baseline.
                previous_mtm = 0.0

            # ------------------------------------------------
            # Daily P&L
            # ------------------------------------------------

            daily_pnl = (
                current_mtm
                - previous_mtm
            )

        except ValueError as exc:

            # Keep the row visible, but do not produce
            # misleading daily P&L.
            valuation_status = (
                "UNAVAILABLE"
            )

            valuation_error = str(
                exc
            )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append(
            {
                "trade_id":
                    trade[
                        "trade_id"
                    ],

                "trade_date":
                    pd.Timestamp(
                        trade[
                            "trade_date"
                        ]
                    ),

                "pair":
                    pair,

                "side":
                    trade[
                        "side"
                    ],

                "notional_base":
                    float(
                        trade[
                            "notional_base"
                        ]
                    ),

                "entry_price":
                    float(
                        trade[
                            "entry_price"
                        ]
                    ),

                "previous_close_date":
                    previous_close_date,

                "previous_close":
                    previous_close,

                "position_existed_at_baseline":
                    position_existed_at_baseline,

                "current_spot":
                    current_spot,

                "previous_mtm_usd":
                    previous_mtm,

                "daily_pnl_usd":
                    daily_pnl,

                "inception_pnl_usd":
                    current_mtm,

                "quote_timestamp":
                    quote_timestamp,

                "market_source":
                    market_source,

                "quote_age_minutes":
                    quote_age,

                "valuation_status":
                    valuation_status,

                "valuation_error":
                    valuation_error,
            }
        )

    if not results:

        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    return pd.DataFrame(
        results,
        columns=OUTPUT_COLUMNS,
    )


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    portfolio = (
        load_portfolio()
    )

    pairs = (
        portfolio[
            "pair"
        ].unique()
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

    pnl = (
        calculate_portfolio_pnl(
            portfolio=portfolio,
            historical_prices=historical_prices,
            valuation_timestamp=valuation_timestamp,
        )
    )

    print(
        "\nVALUATION TIMESTAMP (UTC):",
        valuation_timestamp,
    )

    print(
        f"REPORTING TIMEZONE: "
        f"{REPORTING_TIMEZONE}"
    )

    print(
        "\nDAILY P&L BASIS: "
        "previous completed UTC daily bar"
    )

    print(
        "\nPOSITION P&L"
    )

    if pnl.empty:

        print(
            "No active positions."
        )

    else:

        print(
            pnl.to_string(
                index=False
            )
        )

        valid = pnl[
            pnl[
                "valuation_status"
            ] == "OK"
        ]

        unavailable_count = (
            len(pnl)
            - len(valid)
        )

        print(
            "\nPORTFOLIO TOTALS"
        )

        if valid.empty:

            print(
                "Previous MTM:   N/A"
            )

            print(
                "Daily P&L:      N/A"
            )

            print(
                "Inception P&L:  N/A"
            )

            print(
                "Reconciliation: N/A"
            )

        else:

            previous_total = (
                valid[
                    "previous_mtm_usd"
                ].sum()
            )

            daily_total = (
                valid[
                    "daily_pnl_usd"
                ].sum()
            )

            inception_total = (
                valid[
                    "inception_pnl_usd"
                ].sum()
            )

            print(
                f"Previous MTM:   "
                f"${previous_total:,.2f}"
            )

            print(
                f"Daily P&L:      "
                f"${daily_total:,.2f}"
            )

            print(
                f"Inception P&L:  "
                f"${inception_total:,.2f}"
            )

            reconciliation = (
                previous_total
                + daily_total
                - inception_total
            )

            print(
                f"Reconciliation: "
                f"${reconciliation:,.8f}"
            )

        if unavailable_count > 0:

            print(
                "\nWARNING: "
                f"{unavailable_count} position(s) "
                "have unavailable current daily "
                "valuations. Portfolio totals "
                "are incomplete."
            )