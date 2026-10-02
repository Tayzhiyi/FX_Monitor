from io import BytesIO

import pandas as pd

from dash import (
    Input,
    Output,
    State,
    dcc,
    ctx,
)

from dash.exceptions import (
    PreventUpdate,
)

from src.dashboard_data import (
    PORTFOLIO,
    get_dashboard_history,
    get_risk_historical_prices,
)

from src.pnl import (
    calculate_portfolio_pnl,
)

from src.risk import (
    prepare_risk_returns,
    calculate_var,
    calculate_normal_expected_shortfall,
)

from src.analytics import (
    calculate_historical_daily_pnl,
    build_selected_pnl_matrix,
    calculate_pnl_performance_metrics,
    calculate_sharpe_sortino,
    calculate_stress_scenarios,
)

from src.dashboard_ui import (
    format_usd,
    make_empty_risk_table,
    build_position_table,
    build_position_grid_records,
    build_daily_pnl_chart,
    build_stress_test_chart,
    build_component_var_chart,
    build_correlation_chart,
    build_methodology_text,
    build_data_quality_banner,
    table_to_records,
)


# ============================================================
# HELPERS
# ============================================================

def _safe_float(value):

    if (
        value is None
        or pd.isna(value)
    ):
        return None

    return float(value)


def _safe_timestamp(value):

    if (
        value is None
        or pd.isna(value)
    ):
        return None

    return pd.Timestamp(
        value
    ).isoformat()


def calculate_gross_initial_usd_notional(portfolio_rows):
    """
    Calculate gross initial USD-equivalent notional for the current book.

    For USD-base pairs (e.g. USDJPY), base notional is already USD.

    For USD-quote pairs (e.g. AUDUSD), convert the foreign-currency
    base notional into USD using that trade's entry price.

    Long and short trades are both included on a gross basis; notionals
    are not netted in the denominator.
    """

    if (
        portfolio_rows is None
        or portfolio_rows.empty
    ):
        return None

    gross_usd_notional = 0.0

    for _, row in portfolio_rows.iterrows():

        pair = str(
            row["pair"]
        ).upper()

        notional_base = float(
            row["notional_base"]
        )

        entry_price = float(
            row["entry_price"]
        )

        base_currency = pair[:3]
        quote_currency = pair[3:]

        if base_currency == "USD":

            initial_usd_notional = (
                notional_base
            )

        elif quote_currency == "USD":

            initial_usd_notional = (
                notional_base
                * entry_price
            )

        else:

            raise ValueError(
                f"{pair} does not contain USD. "
                "Gross initial USD notional "
                "conversion is not supported."
            )

        gross_usd_notional += abs(
            initial_usd_notional
        )

    return float(
        gross_usd_notional
    )


def format_pnl_usd_and_pct(
    pnl_usd,
    gross_initial_usd_notional,
    complete=True,
):
    """
    Display P&L in USD and as a percentage of gross initial
    USD-equivalent notional.

    This percentage is an exposure-normalised P&L measure,
    not a true portfolio/NAV return.
    """

    usd_text = format_usd(
        pnl_usd,
        complete=complete,
    )

    if (
        pnl_usd is None
        or pd.isna(pnl_usd)
        or gross_initial_usd_notional is None
        or pd.isna(
            gross_initial_usd_notional
        )
        or gross_initial_usd_notional <= 0
    ):
        return (
            f"{usd_text} / N/A"
        )

    pnl_pct = (
        float(pnl_usd)
        / float(
            gross_initial_usd_notional
        )
        * 100
    )

    pct_text = (
        f"{pnl_pct:.3f}%"
    )

    if not complete:
        pct_text += " *"

    return (
        f"{usd_text} / {pct_text}"
    )


def _build_current_pnl_store(pnl):
    """
    Convert the current portfolio valuation into a
    JSON-safe snapshot for downstream dashboard callbacks.

    This snapshot is the single source of truth for P&L,
    risk and stress analytics until the user explicitly
    refreshes market data again.
    """

    records = []

    for _, row in pnl.iterrows():

        market_source = (
            None
            if pd.isna(
                row["market_source"]
            )
            else str(
                row["market_source"]
            )
        )

        valuation_error = (
            None
            if pd.isna(
                row["valuation_error"]
            )
            else str(
                row["valuation_error"]
            )
        )

        records.append(
            {
                "trade_id":
                    str(
                        row["trade_id"]
                    ),

                "trade_date":
                    _safe_timestamp(
                        row["trade_date"]
                    ),

                "pair":
                    str(
                        row["pair"]
                    ),

                "side":
                    str(
                        row["side"]
                    ),

                "notional_base":
                    _safe_float(
                        row["notional_base"]
                    ),

                "entry_price":
                    _safe_float(
                        row["entry_price"]
                    ),

                "previous_close":
                    _safe_float(
                        row["previous_close"]
                    ),

                "current_spot":
                    _safe_float(
                        row["current_spot"]
                    ),

                "daily_pnl_usd":
                    _safe_float(
                        row["daily_pnl_usd"]
                    ),

                "inception_pnl_usd":
                    _safe_float(
                        row["inception_pnl_usd"]
                    ),

                "market_source":
                    market_source,

                "quote_age_minutes":
                    _safe_float(
                        row["quote_age_minutes"]
                    ),

                "valuation_status":
                    str(
                        row["valuation_status"]
                    ),

                "valuation_error":
                    valuation_error,
            }
        )

    return records


def append_live_daily_pnl(
    pnl_matrix,
    current_pnl_records,
    valuation_timestamp,
):
    """
    Append the current UTC day's live Daily P&L.

    Only appended if every portfolio position has
    a valid current valuation.
    """

    if (
        pnl_matrix is None
        or pnl_matrix.empty
    ):
        return pnl_matrix

    if not current_pnl_records:
        return pnl_matrix

    current = pd.DataFrame(
        current_pnl_records
    )

    required_columns = {
        "trade_id",
        "daily_pnl_usd",
        "valuation_status",
    }

    if not required_columns.issubset(
        current.columns
    ):
        return pnl_matrix

    all_trade_ids = (
        PORTFOLIO[
            "trade_id"
        ]
        .tolist()
    )

    current = (
        current[
            current["trade_id"].isin(
                all_trade_ids
            )
        ]
        .copy()
    )

    # --------------------------------------------------------
    # CRITICAL:
    # live bar requires all portfolio trades
    # --------------------------------------------------------

    if set(
        current["trade_id"]
        .tolist()
    ) != set(all_trade_ids):

        return pnl_matrix

    if not (
        current[
            "valuation_status"
        ]
        .eq("OK")
        .all()
    ):
        return pnl_matrix

    current[
        "daily_pnl_usd"
    ] = pd.to_numeric(
        current[
            "daily_pnl_usd"
        ],
        errors="coerce",
    )

    if (
        current[
            "daily_pnl_usd"
        ]
        .isna()
        .any()
    ):
        return pnl_matrix

    valuation_timestamp = (
        pd.Timestamp(
            valuation_timestamp
        )
    )

    if (
        valuation_timestamp.tzinfo
        is not None
    ):
        live_date = (
            valuation_timestamp
            .tz_convert("UTC")
            .tz_localize(None)
            .normalize()
        )

    else:
        live_date = (
            valuation_timestamp
            .normalize()
        )

    updated = pnl_matrix.copy()

    current_lookup = (
        current
        .set_index("trade_id")[
            "daily_pnl_usd"
        ]
    )

    for trade_id in all_trade_ids:

        updated.loc[
            live_date,
            trade_id,
        ] = float(
            current_lookup.loc[
                trade_id
            ]
        )

    return updated.sort_index()


# ============================================================
# CALLBACK REGISTRATION
# ============================================================

def register_callbacks(app):

    # ========================================================
    # MARKET SNAPSHOT / P&L
    # ========================================================

    @app.callback(
        Output(
            "daily-pnl-card",
            "children",
        ),
        Output(
            "inception-pnl-card",
            "children",
        ),
        Output(
            "last-updated",
            "children",
        ),
        Output(
            "valuation-timestamp-store",
            "data",
        ),
        Output(
            "current-pnl-store",
            "data",
        ),
        Input(
            "refresh-button",
            "n_clicks",
        ),
        running=[
            (
                Output(
                    "refresh-button",
                    "disabled",
                ),
                True,
                False,
            ),
            (
                Output(
                    "refresh-button",
                    "children",
                ),
                "Refreshing...",
                "Refresh Market Data",
            ),
        ],
    )
    def refresh_market_snapshot(
        refresh_clicks,
    ):
        """
        Fetch a fresh current-market snapshot and calculate
        P&L. This callback is triggered only on initial load
        and when Refresh Market Data is pressed.

        VaR confidence and lookback settings do not trigger
        this callback and therefore cannot refresh spot data
        or change P&L.
        """

        valuation_timestamp = (
            pd.Timestamp.now(
                tz="UTC"
            )
        )

        # Historical daily data are cached. This call only
        # downloads them on first use or when the UTC daily
        # cutoff date changes.
        historical_prices = (
            get_dashboard_history(
                valuation_timestamp
            )
        )

        pnl = (
            calculate_portfolio_pnl(
                portfolio=PORTFOLIO,
                historical_prices=(
                    historical_prices
                ),
                valuation_timestamp=(
                    valuation_timestamp
                ),
            )
        )

        valid_pnl = (
            pnl[
                pnl[
                    "valuation_status"
                ] == "OK"
            ]
            .copy()
        )

        unavailable_positions = (
            pnl[
                pnl[
                    "valuation_status"
                ] != "OK"
            ]
            .copy()
        )

        pnl_complete = (
            unavailable_positions.empty
        )

        if valid_pnl.empty:

            daily_pnl = None
            inception_pnl = None

        else:

            daily_pnl = (
                pd.to_numeric(
                    valid_pnl[
                        "daily_pnl_usd"
                    ],
                    errors="coerce",
                )
                .sum(
                    min_count=1
                )
            )

            inception_pnl = (
                pd.to_numeric(
                    valid_pnl[
                        "inception_pnl_usd"
                    ],
                    errors="coerce",
                )
                .sum(
                    min_count=1
                )
            )

        gross_initial_usd_notional = (
            calculate_gross_initial_usd_notional(
                pnl
            )
        )

        current_pnl_store = (
            _build_current_pnl_store(
                pnl
            )
        )

        singapore_time = (
            valuation_timestamp
            .tz_convert(
                "Asia/Singapore"
            )
        )

        last_updated = (
            "Last refreshed: "
            + singapore_time.strftime(
                "%d %b %Y %H:%M:%S SGT"
            )
        )

        return (
            format_pnl_usd_and_pct(
                pnl_usd=daily_pnl,
                gross_initial_usd_notional=(
                    gross_initial_usd_notional
                ),
                complete=pnl_complete,
            ),

            format_pnl_usd_and_pct(
                pnl_usd=inception_pnl,
                gross_initial_usd_notional=(
                    gross_initial_usd_notional
                ),
                complete=pnl_complete,
            ),

            last_updated,

            valuation_timestamp
            .isoformat(),

            current_pnl_store,
        )

    # ========================================================
    # RISK ANALYTICS
    # ========================================================

    @app.callback(
        Output(
            "portfolio-var-card",
            "children",
        ),
        Output(
            "expected-shortfall-card",
            "children",
        ),
        Output(
            "portfolio-vol-card",
            "children",
        ),
        Output(
            "netting-benefit-card",
            "children",
        ),
        Output(
            "cross-pair-diversification-card",
            "children",
        ),
        Output(
            "position-table-store",
            "data",
        ),
        Output(
            "risk-table-store",
            "data",
        ),
        Output(
            "correlation-chart",
            "figure",
        ),
        Output(
            "risk-methodology-text",
            "children",
        ),
        Output(
            "data-quality-banner",
            "children",
        ),
        Output(
            "data-quality-banner",
            "style",
        ),
        Input(
            "confidence-level",
            "value",
        ),
        Input(
            "lookback-days",
            "value",
        ),
        Input(
            "current-pnl-store",
            "data",
        ),
        Input(
            "valuation-timestamp-store",
            "data",
        ),
    )
    def update_risk_analytics(
        confidence_level,
        lookback_days,
        current_pnl_records,
        valuation_timestamp_iso,
    ):
        """
        Recalculate risk from the existing market snapshot.

        Changing confidence or lookback changes risk analytics
        only. It does not fetch new current spots and therefore
        does not change Daily or Inception P&L.
        """

        if (
            not current_pnl_records
            or valuation_timestamp_iso
            is None
        ):
            raise PreventUpdate

        valuation_timestamp = (
            pd.Timestamp(
                valuation_timestamp_iso
            )
        )

        pnl = pd.DataFrame(
            current_pnl_records
        )

        numeric_columns = [
            "notional_base",
            "entry_price",
            "previous_close",
            "current_spot",
            "daily_pnl_usd",
            "inception_pnl_usd",
            "quote_age_minutes",
        ]

        for column in numeric_columns:

            if column in pnl.columns:
                pnl[column] = (
                    pd.to_numeric(
                        pnl[column],
                        errors="coerce",
                    )
                )

        valid_pnl = (
            pnl[
                pnl[
                    "valuation_status"
                ] == "OK"
            ]
            .copy()
        )

        unavailable_positions = (
            pnl[
                pnl[
                    "valuation_status"
                ] != "OK"
            ]
            .copy()
        )

        pnl_complete = (
            unavailable_positions.empty
        )

        # Cached history only: confidence/lookback changes do
        # not cause a current-market refresh.
        historical_prices = (
            get_dashboard_history(
                valuation_timestamp
            )
        )

        risk = (
            make_empty_risk_table()
        )

        risk_summary = None

        correlation = (
            pd.DataFrame()
        )

        risk_error = None

        if not valid_pnl.empty:

            try:

                risk_history = (
                    get_risk_historical_prices(
                        historical_prices,
                        valid_pnl,
                    )
                )

                returns = (
                    prepare_risk_returns(
                        historical_prices=(
                            risk_history
                        ),
                        valuation_timestamp=(
                            valuation_timestamp
                        ),
                        lookback_days=(
                            lookback_days
                        ),
                    )
                )

                (
                    risk,
                    risk_summary,
                    _,
                    correlation,
                ) = calculate_var(
                    pnl=valid_pnl,
                    returns=returns,
                    confidence_level=(
                        confidence_level
                    ),
                )

            except Exception as exc:

                risk_error = str(exc)

                risk = (
                    make_empty_risk_table()
                )

                risk_summary = None

                correlation = (
                    pd.DataFrame()
                )

        if risk_summary is None:

            portfolio_var = None
            portfolio_vol = None
            expected_shortfall = None
            netting_benefit = None
            netting_benefit_pct = None
            cross_pair_diversification = None
            cross_pair_diversification_pct = None

        else:

            portfolio_var = (
                risk_summary[
                    "portfolio_var_usd"
                ]
            )

            portfolio_vol = (
                risk_summary[
                    "portfolio_volatility_usd"
                ]
            )

            netting_benefit = (
                risk_summary[
                    "netting_benefit_usd"
                ]
            )

            netting_benefit_pct = (
                risk_summary[
                    "netting_benefit_pct"
                ]
            )

            cross_pair_diversification = (
                risk_summary[
                    "cross_pair_diversification_benefit_usd"
                ]
            )

            cross_pair_diversification_pct = (
                risk_summary[
                    "cross_pair_diversification_benefit_pct"
                ]
            )

            expected_shortfall = (
                calculate_normal_expected_shortfall(
                    portfolio_volatility_usd=(
                        portfolio_vol
                    ),
                    confidence_level=(
                        confidence_level
                    ),
                )
            )

        risk_complete = (
            pnl_complete
            and risk_error is None
            and risk_summary is not None
        )

        def format_benefit(value, pct):
            if (
                value is None
                or pct is None
                or pd.isna(pct)
            ):
                return "N/A"

            text = (
                f"{format_usd(value)} "
                f"({pct:.1f}%)"
            )

            if not risk_complete:
                text += " *"

            return text

        netting_text = format_benefit(
            netting_benefit,
            netting_benefit_pct,
        )

        cross_pair_diversification_text = (
            format_benefit(
                cross_pair_diversification,
                cross_pair_diversification_pct,
            )
        )

        position_table = (
            build_position_table(
                pnl,
                risk,
            )
        )

        position_records = (
            table_to_records(
                position_table
            )
        )

        risk_records = (
            table_to_records(
                risk
            )
        )

        correlation_chart = (
            build_correlation_chart(
                correlation
            )
        )

        methodology_text = (
            build_methodology_text(
                confidence_level,
                lookback_days,
            )
        )

        (
            banner_text,
            banner_style,
        ) = (
            build_data_quality_banner(
                position_table,
                risk_error,
            )
        )

        return (
            format_usd(
                portfolio_var,
                complete=risk_complete,
            ),

            format_usd(
                expected_shortfall,
                complete=risk_complete,
            ),

            format_usd(
                portfolio_vol,
                complete=risk_complete,
            ),

            netting_text,
            cross_pair_diversification_text,

            position_records,
            risk_records,
            correlation_chart,
            methodology_text,
            banner_text,
            banner_style,
        )


    # ========================================================
    # CURRENT POSITIONS EXCEL EXPORT
    # ========================================================

    @app.callback(
        Output(
            "position-excel-download",
            "data",
        ),
        Input(
            "export-positions-excel-button",
            "n_clicks",
        ),
        State(
            "position-table-store",
            "data",
        ),
        State(
            "position-grid",
            "virtualRowData",
        ),
        State(
            "position-grid",
            "rowData",
        ),
        prevent_initial_call=True,
    )
    def export_positions_to_excel(
        export_clicks,
        position_records,
        visible_rows,
        grid_rows,
    ):
        """
        Export the current portfolio to an Excel workbook.

        Sheet 1: Pair Summary
            One row per FX pair using the same net pair-level risk
            measures displayed in the dashboard.

        Sheet 2: Trade Detail
            All underlying trades with trade-level P&L and risk.

        Pair ordering follows the current grid sort. Trade detail
        stays grouped by pair and ordered by trade date.
        """

        if (
            not export_clicks
            or not position_records
        ):
            raise PreventUpdate

        position_table = pd.DataFrame(
            position_records
        )

        # ----------------------------------------------------
        # Determine the pair order currently displayed.
        #
        # virtualRowData reflects client-side sort/filter state.
        # Fall back to rowData, then the original portfolio order.
        # ----------------------------------------------------

        current_grid_rows = (
            visible_rows
            or grid_rows
            or []
        )

        pair_order = []

        for row in current_grid_rows:

            if not isinstance(
                row,
                dict,
            ):
                continue

            if (
                row.get("row_type")
                == "trade"
            ):
                continue

            pair = row.get(
                "pair_key"
            )

            if (
                pair
                and pair not in pair_order
            ):
                pair_order.append(
                    pair
                )

        for pair in (
            position_table[
                "pair"
            ]
            .drop_duplicates()
            .tolist()
        ):
            if pair not in pair_order:
                pair_order.append(
                    pair
                )

        # ----------------------------------------------------
        # Build one pair row per risk factor.
        # Expand everything temporarily so the helper also gives
        # us a complete hierarchy; only parent/single rows are
        # retained for Pair Summary.
        # ----------------------------------------------------

        all_grid_records = (
            build_position_grid_records(
                position_table=position_table,
                expanded_pairs=(
                    position_table[
                        "pair"
                    ]
                    .drop_duplicates()
                    .tolist()
                ),
            )
        )

        hierarchy = pd.DataFrame(
            all_grid_records
        )

        pair_summary = hierarchy[
            hierarchy[
                "row_type"
            ].isin(
                [
                    "pair",
                    "single",
                ]
            )
        ].copy()

        pair_summary[
            "_pair_order"
        ] = pair_summary[
            "pair_key"
        ].map(
            {
                pair: index
                for index, pair
                in enumerate(
                    pair_order
                )
            }
        )

        pair_summary = (
            pair_summary
            .sort_values(
                "_pair_order",
                kind="stable",
            )
        )

        # ----------------------------------------------------
        # Trade detail: always export every underlying trade.
        # ----------------------------------------------------

        trade_detail = (
            position_table
            .copy()
        )

        trade_detail[
            "_pair_order"
        ] = trade_detail[
            "pair"
        ].map(
            {
                pair: index
                for index, pair
                in enumerate(
                    pair_order
                )
            }
        )

        trade_detail[
            "_trade_date_sort"
        ] = pd.to_datetime(
            trade_detail[
                "trade_date"
            ],
            errors="coerce",
        )

        trade_detail = (
            trade_detail
            .sort_values(
                [
                    "_pair_order",
                    "_trade_date_sort",
                    "trade_id",
                ],
                kind="stable",
            )
        )

        # ----------------------------------------------------
        # Friendly export columns.
        # ----------------------------------------------------

        pair_columns = {
            "pair":
                "Pair",
            "side":
                "Net Side",
            "notional_base":
                "Net Base Notional",
            "previous_close":
                "Prev Close",
            "current_spot":
                "Current Spot",
            "quote_status":
                "Status",
            "daily_pnl_usd":
                "Daily P&L (USD)",
            "inception_pnl_usd":
                "Inception P&L (USD)",
            "position_var_usd":
                "Standalone VaR (USD)",
            "component_var_usd":
                "Component VaR (USD)",
            "incremental_var_usd":
                "Incremental VaR (USD)",
            "component_var_pct":
                "VaR Contribution (%)",
            "marginal_var_per_usd":
                "Marginal VaR / $ Risk",
            "daily_volatility":
                "Daily Vol (%)",
            "pnl_per_1pct_move_usd":
                "1% FX Move P&L (USD)",
            "quote_age_minutes":
                "Quote Age (min)",
        }

        trade_columns = {
            "trade_id":
                "Trade ID",
            "pair":
                "Pair",
            "side":
                "Side",
            "trade_date":
                "Trade Date",
            "notional_base":
                "Base Notional",
            "entry_price":
                "Entry",
            "previous_close":
                "Prev Close",
            "current_spot":
                "Current Spot",
            "quote_status":
                "Status",
            "daily_pnl_usd":
                "Daily P&L (USD)",
            "inception_pnl_usd":
                "Inception P&L (USD)",
            "position_var_usd":
                "Standalone VaR (USD)",
            "component_var_usd":
                "Component VaR (USD)",
            "incremental_var_usd":
                "Incremental VaR (USD)",
            "component_var_pct":
                "VaR Contribution (%)",
            "marginal_var_per_usd":
                "Marginal VaR / $ Risk",
            "daily_volatility":
                "Daily Vol (%)",
            "pnl_per_1pct_move_usd":
                "1% FX Move P&L (USD)",
            "quote_age_minutes":
                "Quote Age (min)",
        }

        pair_export = (
            pair_summary[
                list(
                    pair_columns.keys()
                )
            ]
            .rename(
                columns=pair_columns
            )
            .reset_index(
                drop=True
            )
        )

        trade_export = (
            trade_detail[
                list(
                    trade_columns.keys()
                )
            ]
            .rename(
                columns=trade_columns
            )
            .reset_index(
                drop=True
            )
        )

        # Daily volatility is stored in percentage-point display
        # units in the position table (e.g. 0.49 means 0.49%).
        # Keep that convention in the Excel export.

        def write_workbook(
            bytes_io,
        ):

            with pd.ExcelWriter(
                bytes_io,
                engine="openpyxl",
            ) as writer:

                pair_export.to_excel(
                    writer,
                    sheet_name="Pair Summary",
                    index=False,
                )

                trade_export.to_excel(
                    writer,
                    sheet_name="Trade Detail",
                    index=False,
                )

                for worksheet in (
                    writer.book.worksheets
                ):

                    worksheet.freeze_panes = "A2"
                    worksheet.auto_filter.ref = (
                        worksheet.dimensions
                    )

                    for column_cells in (
                        worksheet.columns
                    ):

                        max_length = 0

                        for cell in column_cells:
                            value = (
                                ""
                                if cell.value is None
                                else str(
                                    cell.value
                                )
                            )
                            max_length = max(
                                max_length,
                                len(value),
                            )

                        width = min(
                            max(
                                max_length + 2,
                                11,
                            ),
                            28,
                        )

                        worksheet.column_dimensions[
                            column_cells[0].column_letter
                        ].width = width

        filename = (
            "fx_current_positions_"
            + pd.Timestamp.now(
                tz="Asia/Singapore"
            ).strftime(
                "%Y%m%d_%H%M%S"
            )
            + ".xlsx"
        )

        return dcc.send_bytes(
            write_workbook,
            filename,
        )

    # ========================================================
    # PAIR / TRADE POSITION DRILLDOWN
    # ========================================================

    @app.callback(
        Output(
            "position-grid",
            "rowData",
        ),
        Input(
            "position-table-store",
            "data",
        ),
        Input(
            "expanded-pairs-store",
            "data",
        ),
    )
    def render_position_grid(
        position_records,
        expanded_pairs,
    ):

        if not position_records:
            return []

        position_table = pd.DataFrame(
            position_records
        )

        return build_position_grid_records(
            position_table=position_table,
            expanded_pairs=expanded_pairs,
        )

    @app.callback(
        Output(
            "expanded-pairs-store",
            "data",
        ),
        Input(
            "position-grid",
            "cellClicked",
        ),
        State(
            "position-grid",
            "virtualRowData",
        ),
        State(
            "position-grid",
            "rowData",
        ),
        State(
            "expanded-pairs-store",
            "data",
        ),
        prevent_initial_call=True,
    )
    def toggle_position_pair(
        cell_clicked,
        visible_rows,
        current_rows,
        expanded_pairs,
    ):
        """
        Toggle only net pair rows that contain more than one trade.

        Dash AG Grid's cellClicked event does not reliably include
        the full row dictionary across versions, so the callback
        resolves the clicked row from rowIndex + the current rowData.
        """

        if not cell_clicked:
            raise PreventUpdate

        row_data = (
            cell_clicked.get("data")
            or {}
        )

        if not row_data:

            row_index = cell_clicked.get(
                "rowIndex"
            )

            try:
                row_index = int(row_index)
            except (
                TypeError,
                ValueError,
            ):
                raise PreventUpdate

            displayed_rows = (
                visible_rows
                or current_rows
                or []
            )

            if (
                row_index < 0
                or row_index >= len(
                    displayed_rows
                )
            ):
                raise PreventUpdate

            row_data = (
                displayed_rows[
                    row_index
                ]
                or {}
            )

        if (
            row_data.get("row_type")
            != "pair"
            or not row_data.get(
                "is_expandable",
                False,
            )
        ):
            raise PreventUpdate

        pair = row_data.get(
            "pair_key"
        )

        if pair is None:
            raise PreventUpdate

        updated = list(
            expanded_pairs or []
        )

        if pair in updated:
            updated.remove(pair)
        else:
            updated.append(pair)

        return updated

    # ========================================================
    # COMPONENT VAR DRILLDOWN
    # ========================================================

    @app.callback(
        Output(
            "component-var-selected-pair",
            "data",
        ),
        Input(
            "component-var-chart",
            "clickData",
        ),
        Input(
            "component-var-back-button",
            "n_clicks",
        ),
        State(
            "component-var-selected-pair",
            "data",
        ),
        prevent_initial_call=True,
    )
    def update_component_var_selection(
        click_data,
        back_clicks,
        selected_pair,
    ):

        triggered_id = ctx.triggered_id

        if (
            triggered_id
            == "component-var-back-button"
        ):
            return None

        if (
            triggered_id
            != "component-var-chart"
            or not click_data
        ):
            raise PreventUpdate

        # While already in trade drilldown, chart clicks are
        # intentionally ignored; use Pair View to return.
        if selected_pair is not None:
            raise PreventUpdate

        points = click_data.get(
            "points",
            []
        )

        if not points:
            raise PreventUpdate

        point = points[0]
        pair = (
            point.get("customdata")
            or point.get("y")
        )

        pair_trade_counts = (
            PORTFOLIO[
                "pair"
            ]
            .astype(str)
            .value_counts()
        )

        # Drilldown is only useful when a pair contains
        # multiple underlying trades.
        if (
            pair not in pair_trade_counts.index
            or int(
                pair_trade_counts.loc[
                    pair
                ]
            ) <= 1
        ):
            raise PreventUpdate

        return pair

    @app.callback(
        Output(
            "component-var-chart",
            "figure",
        ),
        Output(
            "component-var-back-button",
            "style",
        ),
        Input(
            "risk-table-store",
            "data",
        ),
        Input(
            "component-var-selected-pair",
            "data",
        ),
    )
    def render_component_var_chart(
        risk_records,
        selected_pair,
    ):

        risk = (
            pd.DataFrame(risk_records)
            if risk_records
            else pd.DataFrame()
        )

        figure = build_component_var_chart(
            risk=risk,
            selected_pair=selected_pair,
        )

        back_style = {
            "display": (
                "block"
                if selected_pair is not None
                else "none"
            ),
            "height": "30px",
            "padding": "0 12px",
            "borderRadius": "6px",
            "border": "1px solid #415078",
            "backgroundColor": "#25304d",
            "color": "white",
            "cursor": "pointer",
        }

        return (
            figure,
            back_style,
        )

    # ========================================================
    # STRESS TEST
    # ========================================================

    @app.callback(
        Output(
            "stress-test-chart",
            "figure",
        ),
        Input(
            "current-pnl-store",
            "data",
        ),
    )
    def update_stress_test(
        current_pnl_records,
    ):

        if not current_pnl_records:
            current_pnl = (
                pd.DataFrame()
            )
        else:
            current_pnl = (
                pd.DataFrame(
                    current_pnl_records
                )
            )

        stress_results = (
            calculate_stress_scenarios(
                pnl=current_pnl,
                expected_trade_ids=(
                    PORTFOLIO[
                        "trade_id"
                    ]
                    .tolist()
                ),
            )
        )

        return (
            build_stress_test_chart(
                stress_results
            )
        )

    # ========================================================
    # PAIR OVERLAY SELECTOR
    # ========================================================

    @app.callback(
        Output(
            "pnl-overlay-store",
            "data",
        ),
        Output(
            "pnl-position-selector",
            "value",
        ),
        Input(
            "pnl-position-selector",
            "value",
        ),
        Input(
            "clear-overlays-button",
            "n_clicks",
        ),
        State(
            "pnl-overlay-store",
            "data",
        ),
        prevent_initial_call=True,
    )
    def update_position_overlays(
        new_pair,
        clear_clicks,
        current_overlays,
    ):

        triggered_id = (
            ctx.triggered_id
        )

        if current_overlays is None:
            current_overlays = []

        if (
            triggered_id
            == "clear-overlays-button"
        ):
            return (
                [],
                None,
            )

        if (
            triggered_id
            == "pnl-position-selector"
            and new_pair
            is not None
        ):

            updated = list(
                current_overlays
            )

            if new_pair not in updated:
                updated.append(
                    new_pair
                )

            return (
                updated,
                None,
            )

        raise PreventUpdate

    # ========================================================
    # HISTORICAL PERFORMANCE
    # ========================================================

    @app.callback(
        Output(
            "daily-pnl-chart",
            "figure",
        ),
        Output(
            "max-drawdown-card",
            "children",
        ),
        Output(
            "best-day-card",
            "children",
        ),
        Output(
            "worst-day-card",
            "children",
        ),
        Output(
            "sharpe-card",
            "children",
        ),
        Output(
            "sortino-card",
            "children",
        ),
        Input(
            "pnl-overlay-store",
            "data",
        ),
        Input(
            "valuation-timestamp-store",
            "data",
        ),
        Input(
            "current-pnl-store",
            "data",
        ),
    )
    def update_historical_performance(
        selected_pairs,
        valuation_timestamp_iso,
        current_pnl_records,
    ):

        if (
            valuation_timestamp_iso
            is None
        ):
            raise PreventUpdate

        valuation_timestamp = (
            pd.Timestamp(
                valuation_timestamp_iso
            )
        )

        if selected_pairs is None:
            selected_pairs = []

        historical_prices = (
            get_dashboard_history(
                valuation_timestamp
            )
        )

        pnl_history = (
            calculate_historical_daily_pnl(
                portfolio=PORTFOLIO,
                historical_prices=(
                    historical_prices
                ),
                valuation_timestamp=(
                    valuation_timestamp
                ),
            )
        )

        all_trade_ids = (
            PORTFOLIO[
                "trade_id"
            ]
            .tolist()
        )

        completed_pnl_matrix = (
            build_selected_pnl_matrix(
                pnl_history=pnl_history,
                portfolio=PORTFOLIO,
                selected_trade_ids=(
                    all_trade_ids
                ),
            )
        )

        # ----------------------------------------------------
        # Sharpe / Sortino:
        # completed days only
        # ----------------------------------------------------

        ratio_metrics = (
            calculate_sharpe_sortino(
                completed_pnl_matrix
            )
        )

        sharpe = (
            ratio_metrics["sharpe"]
        )

        sortino = (
            ratio_metrics["sortino"]
        )

        sharpe_text = (
            f"{sharpe:.2f}"
            if sharpe is not None
            else "N/A"
        )

        sortino_text = (
            f"{sortino:.2f}"
            if sortino is not None
            else "N/A"
        )

        # ====================================================
        # APPEND CURRENT LIVE DAY
        # ====================================================

        display_pnl_matrix = (
            append_live_daily_pnl(
                pnl_matrix=(
                    completed_pnl_matrix
                ),
                current_pnl_records=(
                    current_pnl_records
                ),
                valuation_timestamp=(
                    valuation_timestamp
                ),
            )
        )

        figure = (
            build_daily_pnl_chart(
                pnl_matrix=(
                    display_pnl_matrix
                ),
                portfolio=PORTFOLIO,
                selected_pairs=(
                    selected_pairs
                ),
            )
        )

        metrics = (
            calculate_pnl_performance_metrics(
                display_pnl_matrix
            )
        )

        max_drawdown_text = (
            format_usd(
                metrics[
                    "max_drawdown_usd"
                ]
            )
        )

        if (
            metrics[
                "best_day_usd"
            ]
            is None
        ):
            best_day_text = "N/A"

        else:

            best_date = (
                pd.Timestamp(
                    metrics[
                        "best_day_date"
                    ]
                )
                .strftime(
                    "%d %b %Y"
                )
            )

            best_day_text = (
                f"{format_usd(metrics['best_day_usd'])}"
                f" · {best_date}"
            )

        if (
            metrics[
                "worst_day_usd"
            ]
            is None
        ):
            worst_day_text = "N/A"

        else:

            worst_date = (
                pd.Timestamp(
                    metrics[
                        "worst_day_date"
                    ]
                )
                .strftime(
                    "%d %b %Y"
                )
            )

            worst_day_text = (
                f"{format_usd(metrics['worst_day_usd'])}"
                f" · {worst_date}"
            )

        return (
            figure,
            max_drawdown_text,
            best_day_text,
            worst_day_text,
            sharpe_text,
            sortino_text,
        )