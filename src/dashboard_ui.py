import pandas as pd
import plotly.graph_objects as go

from dash import (
    dcc,
    html,
)

import dash_ag_grid as dag


# ============================================================
# STYLING
# ============================================================

PAGE_STYLE = {
    "backgroundColor": "#0b1020",
    "minHeight": "100vh",
    "padding": "28px",
    "fontFamily": "Arial, sans-serif",
    "color": "#f4f6fb",
}


CARD_STYLE = {
    "backgroundColor": "#151c2f",
    "border": "1px solid #26314d",
    "borderRadius": "10px",
    "padding": "18px",
    "flex": "1",
    "minWidth": "170px",
}


SMALL_CARD_STYLE = {
    "backgroundColor": "#101729",
    "border": "1px solid #26314d",
    "borderRadius": "8px",
    "padding": "14px 18px",
    "flex": "1",
    "minWidth": "170px",
}


PANEL_STYLE = {
    "backgroundColor": "#151c2f",
    "border": "1px solid #26314d",
    "borderRadius": "10px",
    "padding": "20px",
    "marginTop": "20px",
}


LABEL_STYLE = {
    "color": "#8994aa",
    "fontSize": "12px",
    "fontWeight": "600",
    "textTransform": "uppercase",
    "letterSpacing": "0.6px",
}


VALUE_STYLE = {
    "fontSize": "27px",
    "fontWeight": "700",
    "marginTop": "8px",
}


SMALL_VALUE_STYLE = {
    "fontSize": "22px",
    "fontWeight": "700",
    "marginTop": "6px",
}


SMALL_NOTE_STYLE = {
    "color": "#8994aa",
    "fontSize": "11px",
    "marginTop": "4px",
}


RISK_TABLE_COLUMNS = [
    "trade_id",
    "position_var_usd",
    "marginal_var_per_usd",
    "component_var_usd",
    "component_var_pct",
    "incremental_var_usd",
    "signed_usd_exposure",
    "daily_volatility",
    "pair_signed_usd_exposure",
    "pair_standalone_var_usd",
    "pair_component_var_usd",
    "pair_component_var_pct",
    "pair_incremental_var_usd",
]


# ============================================================
# HELPERS
# ============================================================

def format_usd(
    value,
    complete=True,
):
    if (
        value is None
        or pd.isna(value)
    ):
        return "N/A"

    if value < 0:
        text = (
            f"-${abs(value):,.0f}"
        )
    else:
        text = (
            f"${value:,.0f}"
        )

    if not complete:
        text += " *"

    return text


def get_quote_status(row):

    if (
        row["valuation_status"]
        != "OK"
    ):
        return "UNAVAILABLE"

    if (
        row["market_source"]
        != "intraday_1m"
    ):
        return "DAILY FALLBACK"

    age = row[
        "quote_age_minutes"
    ]

    if pd.isna(age):
        return "UNKNOWN"

    if age <= 30:
        return "LIVE"

    if age <= 120:
        return "DELAYED"

    return "STALE"


# ============================================================
# EMPTY RISK TABLE
# ============================================================

def make_empty_risk_table():

    float_columns = [
        "position_var_usd",
        "marginal_var_per_usd",
        "component_var_usd",
        "component_var_pct",
        "incremental_var_usd",
        "signed_usd_exposure",
        "daily_volatility",
        "pair_signed_usd_exposure",
        "pair_standalone_var_usd",
        "pair_component_var_usd",
        "pair_component_var_pct",
        "pair_incremental_var_usd",
    ]

    data = {
        "trade_id": pd.Series(dtype="object"),
    }

    for column in float_columns:
        data[column] = pd.Series(
            dtype="float64"
        )

    return pd.DataFrame(data)


# ============================================================
# POSITION TABLE
# ============================================================

def build_position_table(
    pnl,
    risk,
):

    if (
        risk is None
        or risk.empty
    ):
        risk_columns = (
            make_empty_risk_table()
        )
    else:
        risk_columns = (
            risk[
                RISK_TABLE_COLUMNS
            ]
            .copy()
        )

    table = pnl.merge(
        risk_columns,
        on="trade_id",
        how="left",
    )

    table["quote_status"] = (
        table.apply(
            get_quote_status,
            axis=1,
        )
    )

    numeric_risk_columns = [
        "position_var_usd",
        "marginal_var_per_usd",
        "component_var_usd",
        "component_var_pct",
        "incremental_var_usd",
        "signed_usd_exposure",
        "daily_volatility",
        "pair_signed_usd_exposure",
        "pair_standalone_var_usd",
        "pair_component_var_usd",
        "pair_component_var_pct",
        "pair_incremental_var_usd",
    ]

    for column in numeric_risk_columns:
        table[column] = (
            pd.to_numeric(
                table[column],
                errors="coerce",
            )
        )

    table[
        "pnl_per_1pct_move_usd"
    ] = (
        table[
            "signed_usd_exposure"
        ]
        * 0.01
    )

    table["trade_date"] = (
        pd.to_datetime(
            table["trade_date"]
        )
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    # Keep financial values at full precision internally.
    # The AG Grid value formatters handle display rounding,
    # which preserves exact pair/trade P&L reconciliation.
    for column in [
        "notional_base",
        "daily_pnl_usd",
        "inception_pnl_usd",
        "position_var_usd",
        "component_var_usd",
        "incremental_var_usd",
        "pnl_per_1pct_move_usd",
        "pair_signed_usd_exposure",
        "pair_standalone_var_usd",
        "pair_component_var_usd",
        "pair_incremental_var_usd",
    ]:
        table[column] = pd.to_numeric(
            table[column],
            errors="coerce",
        )

    for column in [
        "entry_price",
        "previous_close",
        "current_spot",
    ]:
        table[column] = (
            pd.to_numeric(
                table[column],
                errors="coerce",
            )
            .round(6)
        )

    table[
        "marginal_var_per_usd"
    ] = (
        table[
            "marginal_var_per_usd"
        ]
        .round(6)
    )

    for column in [
        "component_var_pct",
        "pair_component_var_pct",
    ]:
        table[column] = (
            table[column]
            .round(1)
        )

    table[
        "daily_volatility"
    ] = (
        table[
            "daily_volatility"
        ]
        * 100
    ).round(2)

    table[
        "quote_age_minutes"
    ] = (
        pd.to_numeric(
            table[
                "quote_age_minutes"
            ],
            errors="coerce",
        )
        .round(1)
    )

    return table


def build_position_grid_records(
    position_table,
    expanded_pairs=None,
):
    """
    Build the PM-facing current-position view.

    - FX pairs with ONE trade are shown directly as a normal
      position row using that trade's date, side, entry price
      and notional.
    - FX pairs with MULTIPLE trades are shown as a net pair row.
      Only those rows are expandable into their constituent trades.
    - Pair-level risk measures use the net FX-factor exposure;
      child rows retain the original trade-level measures.
    """

    if (
        position_table is None
        or position_table.empty
    ):
        return []

    expanded_pairs = set(
        expanded_pairs or []
    )

    def first_valid(series):
        values = series.dropna()
        if values.empty:
            return None
        return values.iloc[0]

    def complete_sum(series):
        values = pd.to_numeric(
            series,
            errors="coerce",
        )
        if values.isna().any():
            return None
        return float(values.sum())

    status_rank = {
        "LIVE": 0,
        "DAILY FALLBACK": 1,
        "DELAYED": 2,
        "STALE": 3,
        "UNKNOWN": 4,
        "UNAVAILABLE": 5,
    }

    output = []

    for pair, group in position_table.groupby(
        "pair",
        sort=False,
    ):
        group = group.copy()

        trade_count = len(group)

        # ----------------------------------------------------
        # Single-trade pair
        #
        # No drilldown is needed. Show the actual trade-level
        # information directly, including trade date and entry.
        # ----------------------------------------------------

        if trade_count == 1:

            row = group.iloc[0].to_dict()

            row.update(
                {
                    "row_type": "single",
                    "pair_key": pair,
                    "is_expandable": False,
                    "display_name": pair,
                }
            )

            output.append(row)
            continue

        # ----------------------------------------------------
        # Multi-trade pair
        #
        # Default to one net risk-factor row. Expanding reveals
        # the underlying trades.
        # ----------------------------------------------------

        is_expanded = pair in expanded_pairs

        signed_base_notional = 0.0

        for _, row in group.iterrows():

            multiplier = (
                1.0
                if str(row["side"]).upper() == "LONG"
                else -1.0
            )

            signed_base_notional += (
                multiplier
                * float(row["notional_base"])
            )

        if signed_base_notional > 0:
            net_side = "NET LONG"
        elif signed_base_notional < 0:
            net_side = "NET SHORT"
        else:
            net_side = "FLAT"

        statuses = (
            group["quote_status"]
            .dropna()
            .astype(str)
            .tolist()
        )

        if statuses:
            pair_status = max(
                statuses,
                key=lambda value: status_rank.get(
                    value,
                    99,
                ),
            )
        else:
            pair_status = "UNKNOWN"

        pair_exposure = first_valid(
            group["pair_signed_usd_exposure"]
        )

        pair_row = {
            "row_type": "pair",
            "pair_key": pair,
            "is_expandable": True,
            "display_name": (
                ("▼ " if is_expanded else "▶ ")
                + pair
            ),
            "trade_id": None,
            "pair": pair,
            "side": net_side,

            # There is no single economically meaningful
            # trade date or entry price for a netted pair.
            "trade_date": None,
            "entry_price": None,

            "notional_base": abs(
                signed_base_notional
            ),
            "previous_close": first_valid(
                group["previous_close"]
            ),
            "current_spot": first_valid(
                group["current_spot"]
            ),
            "quote_status": pair_status,
            "daily_pnl_usd": complete_sum(
                group["daily_pnl_usd"]
            ),
            "inception_pnl_usd": complete_sum(
                group["inception_pnl_usd"]
            ),
            "position_var_usd": first_valid(
                group["pair_standalone_var_usd"]
            ),
            "component_var_usd": first_valid(
                group["pair_component_var_usd"]
            ),
            "incremental_var_usd": first_valid(
                group["pair_incremental_var_usd"]
            ),
            "component_var_pct": first_valid(
                group["pair_component_var_pct"]
            ),
            "marginal_var_per_usd": first_valid(
                group["marginal_var_per_usd"]
            ),
            "daily_volatility": first_valid(
                group["daily_volatility"]
            ),
            "pnl_per_1pct_move_usd": (
                None
                if pair_exposure is None
                else float(pair_exposure) * 0.01
            ),
            "quote_age_minutes": (
                pd.to_numeric(
                    group["quote_age_minutes"],
                    errors="coerce",
                )
                .max()
            ),
        }

        output.append(pair_row)

        if not is_expanded:
            continue

        for _, row in group.iterrows():

            child = row.to_dict()

            child.update(
                {
                    "row_type": "trade",
                    "pair_key": pair,
                    "is_expandable": False,
                    "display_name": (
                        "↳ "
                        + str(row["trade_id"])
                    ),
                }
            )

            output.append(child)

    return output


# ============================================================
# HISTORICAL DAILY P&L CHART
# ============================================================

def build_daily_pnl_chart(
    pnl_matrix,
    portfolio,
    selected_pairs,
):

    figure = go.Figure()

    if (
        pnl_matrix is None
        or pnl_matrix.empty
    ):

        figure.add_annotation(
            text=(
                "Historical P&L unavailable."
            ),
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

        figure.update_layout(
            template="plotly_dark",
            paper_bgcolor="#151c2f",
            plot_bgcolor="#151c2f",
            height=420,
        )

        return figure

    portfolio_total = (
        pnl_matrix
        .sum(
            axis=1,
            min_count=len(
                pnl_matrix.columns
            ),
        )
    )

    positive_pnl = (
        portfolio_total.where(
            portfolio_total >= 0
        )
    )

    negative_pnl = (
        portfolio_total.where(
            portfolio_total < 0
        )
    )

    figure.add_trace(
        go.Bar(
            x=positive_pnl.index,
            y=positive_pnl,
            name=(
                "Portfolio Daily P&L — Gain"
            ),
            marker_color="#22c55e",
            opacity=0.78,
            hovertemplate=(
                "Portfolio Daily P&L"
                "<br>%{x|%d %b %Y}"
                "<br>P&L: $%{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Bar(
            x=negative_pnl.index,
            y=negative_pnl,
            name=(
                "Portfolio Daily P&L — Loss"
            ),
            marker_color="#ef4444",
            opacity=0.78,
            hovertemplate=(
                "Portfolio Daily P&L"
                "<br>%{x|%d %b %Y}"
                "<br>P&L: $%{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    pair_palette = [
        "#60a5fa",
        "#a78bfa",
        "#f59e0b",
        "#f472b6",
        "#818cf8",
        "#c084fc",
        "#94a3b8",
    ]

    child_palette = [
        "#93c5fd",
        "#c4b5fd",
        "#fcd34d",
        "#f9a8d4",
        "#a5b4fc",
        "#d8b4fe",
        "#cbd5e1",
    ]

    if selected_pairs is None:
        selected_pairs = []

    for pair_index, pair in enumerate(
        selected_pairs
    ):

        pair_portfolio = (
            portfolio[
                portfolio["pair"] == pair
            ]
            .copy()
        )

        if pair_portfolio.empty:
            continue

        trade_ids = [
            trade_id
            for trade_id in pair_portfolio[
                "trade_id"
            ].tolist()
            if trade_id in pnl_matrix.columns
        ]

        if not trade_ids:
            continue

        earliest_trade_date = (
            pd.to_datetime(
                pair_portfolio["trade_date"]
            )
            .min()
            .normalize()
        )

        pair_pnl = (
            pnl_matrix[trade_ids]
            .sum(
                axis=1,
                min_count=len(trade_ids),
            )
        )

        pair_pnl.loc[
            pair_pnl.index < earliest_trade_date
        ] = None

        pair_colour = pair_palette[
            pair_index % len(pair_palette)
        ]

        pair_name = (
            f"{pair} — Net"
            if len(trade_ids) > 1
            else pair
        )

        figure.add_trace(
            go.Scatter(
                x=pair_pnl.index,
                y=pair_pnl,
                mode="lines+markers",
                name=pair_name,
                line=dict(
                    width=3,
                    color=pair_colour,
                ),
                marker=dict(
                    size=6,
                    color=pair_colour,
                ),
                hovertemplate=(
                    f"{pair_name}"
                    "<br>%{x|%d %b %Y}"
                    "<br>Daily P&L: "
                    "$%{y:,.0f}"
                    "<extra></extra>"
                ),
            )
        )

        # For a multi-trade pair, show each constituent trade
        # beneath the net pair line. A one-trade pair is already
        # represented by the pair line and is not duplicated.
        if len(trade_ids) <= 1:
            continue

        trade_date_lookup = (
            pair_portfolio
            .set_index("trade_id")[
                "trade_date"
            ]
        )

        for child_index, trade_id in enumerate(
            trade_ids
        ):
            trade_pnl = (
                pnl_matrix[trade_id]
                .copy()
            )

            trade_date = (
                pd.Timestamp(
                    trade_date_lookup.loc[
                        trade_id
                    ]
                )
                .normalize()
            )

            trade_pnl.loc[
                trade_pnl.index < trade_date
            ] = None

            child_colour = child_palette[
                (pair_index + child_index)
                % len(child_palette)
            ]

            figure.add_trace(
                go.Scatter(
                    x=trade_pnl.index,
                    y=trade_pnl,
                    mode="lines+markers",
                    name=f"{pair} · {trade_id}",
                    line=dict(
                        width=1.6,
                        dash="dot",
                        color=child_colour,
                    ),
                    marker=dict(
                        size=4,
                        color=child_colour,
                    ),
                    hovertemplate=(
                        f"{pair} · {trade_id}"
                        "<br>%{x|%d %b %Y}"
                        "<br>Daily P&L: "
                        "$%{y:,.0f}"
                        "<extra></extra>"
                    ),
                )
            )

    figure.add_hline(
        y=0,
        line_width=1,
        line_dash="dot",
        line_color="#94a3b8",
        opacity=0.6,
    )

    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#151c2f",
        plot_bgcolor="#151c2f",

        title=dict(
            text="Portfolio Daily P&L",
            x=0.01,
            xanchor="left",
        ),

        xaxis_title="",
        yaxis_title="USD",

        hovermode="x unified",

        barmode="overlay",
        bargap=0.25,

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),

        margin=dict(
            l=60,
            r=20,
            t=85,
            b=40,
        ),

        height=420,
    )

    return figure


# ============================================================
# STRESS TEST CHART
# ============================================================

def build_stress_test_chart(
    stress_results,
):

    figure = go.Figure()

    if (
        stress_results is None
        or stress_results.empty
    ):
        figure.add_annotation(
            text=(
                "Stress test unavailable — incomplete portfolio"
            ),
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

    else:

        chart_data = (
            stress_results.copy()
        )

        colours = [
            (
                "#22c55e"
                if value >= 0
                else "#ef4444"
            )
            for value
            in chart_data[
                "stress_pnl_usd"
            ]
        ]

        labels = [
            (
                f"${value:,.0f}"
                if value >= 0
                else
                f"-${abs(value):,.0f}"
            )
            for value
            in chart_data[
                "stress_pnl_usd"
            ]
        ]

        figure.add_trace(
            go.Bar(
                x=chart_data[
                    "stress_pnl_usd"
                ],
                y=chart_data[
                    "scenario"
                ],
                orientation="h",
                marker_color=colours,
                text=labels,
                textposition="auto",
                hovertemplate=(
                    "%{y}"
                    "<br>Stress P&L: "
                    "$%{x:,.0f}"
                    "<extra></extra>"
                ),
            )
        )

    figure.add_vline(
        x=0,
        line_width=1,
        line_dash="dot",
        line_color="#94a3b8",
        opacity=0.6,
    )

    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#151c2f",
        plot_bgcolor="#151c2f",

        title=(
            "USD Spot Stress Test"
        ),

        xaxis_title="USD",
        yaxis_title="",

        yaxis=dict(
            autorange="reversed",
        ),

        margin=dict(
            l=80,
            r=20,
            t=55,
            b=40,
        ),

        height=400,
    )

    return figure


# ============================================================
# COMPONENT VAR
# ============================================================

def build_component_var_chart(
    risk,
    selected_pair=None,
):

    figure = go.Figure()

    if (
        risk is None
        or risk.empty
    ):
        figure.add_annotation(
            text="Risk unavailable",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

        title = "Component VaR by FX Pair"

    elif (
        selected_pair is not None
        and selected_pair in set(
            risk["pair"].astype(str)
        )
    ):

        chart_data = (
            risk[
                risk["pair"] == selected_pair
            ][
                [
                    "trade_id",
                    "side",
                    "component_var_usd",
                ]
            ]
            .copy()
            .sort_values(
                "component_var_usd",
                ascending=True,
            )
        )

        chart_data["label"] = (
            chart_data["trade_id"].astype(str)
            + " · "
            + chart_data["side"].astype(str)
        )

        colours = [
            (
                "#6366f1"
                if value >= 0
                else "#f59e0b"
            )
            for value in chart_data[
                "component_var_usd"
            ]
        ]

        figure.add_trace(
            go.Bar(
                x=chart_data[
                    "component_var_usd"
                ],
                y=chart_data["label"],
                orientation="h",
                marker_color=colours,
                customdata=chart_data[
                    "trade_id"
                ],
                text=[
                    (
                        f"${value:,.0f}"
                        if value >= 0
                        else f"-${abs(value):,.0f}"
                    )
                    for value in chart_data[
                        "component_var_usd"
                    ]
                ],
                textposition="auto",
                hovertemplate=(
                    "%{y}<br>"
                    "Component VaR: "
                    "$%{x:,.0f}"
                    "<extra></extra>"
                ),
            )
        )

        title = (
            f"{selected_pair} Component VaR — Trade Drilldown"
        )

    else:

        # Aggregate trade-level Component VaR to the FX
        # factor level so each currency pair appears once.
        chart_data = (
            risk[
                [
                    "pair",
                    "component_var_usd",
                ]
            ]
            .groupby(
                "pair",
                as_index=False,
            )[
                "component_var_usd"
            ]
            .sum()
            .sort_values(
                "component_var_usd",
                ascending=True,
            )
        )

        colours = [
            (
                "#6366f1"
                if value >= 0
                else "#f59e0b"
            )
            for value in chart_data[
                "component_var_usd"
            ]
        ]

        figure.add_trace(
            go.Bar(
                x=chart_data[
                    "component_var_usd"
                ],
                y=chart_data[
                    "pair"
                ],
                orientation="h",
                marker_color=colours,
                customdata=chart_data[
                    "pair"
                ],
                text=[
                    (
                        f"${value:,.0f}"
                        if value >= 0
                        else
                        f"-${abs(value):,.0f}"
                    )
                    for value
                    in chart_data[
                        "component_var_usd"
                    ]
                ],
                textposition="auto",
                hovertemplate=(
                    "%{y}<br>"
                    "Component VaR: "
                    "$%{x:,.0f}"
                    "<extra></extra>"
                ),
            )
        )

        title = (
            "Component VaR by FX Pair — multi-trade pairs can be drilled down"
        )

    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#151c2f",
        plot_bgcolor="#151c2f",
        title=title,
        xaxis_title="USD",
        yaxis_title="",
        margin=dict(
            l=40,
            r=20,
            t=50,
            b=40,
        ),
        height=380,
    )

    return figure


# ============================================================
# CORRELATION
# ============================================================

def build_correlation_chart(
    correlation,
):

    figure = go.Figure()

    if (
        correlation is None
        or correlation.empty
    ):
        figure.add_annotation(
            text=(
                "Correlation data unavailable"
            ),
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

    else:

        labels = (
            correlation.columns.tolist()
        )

        figure.add_trace(
            go.Heatmap(
                z=correlation.values,
                x=labels,
                y=labels,
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                reversescale=True,
                text=(
                    correlation
                    .round(2)
                    .values
                ),
                texttemplate="%{text}",
                hovertemplate=(
                    "%{y} vs %{x}"
                    "<br>Correlation: "
                    "%{z:.2f}"
                    "<extra></extra>"
                ),
                colorbar=dict(
                    title="ρ"
                ),
            )
        )

    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#151c2f",
        plot_bgcolor="#151c2f",
        title="FX Return Correlation",
        margin=dict(
            l=60,
            r=20,
            t=50,
            b=50,
        ),
        height=500,
    )

    return figure


# ============================================================
# GRID STYLES
# ============================================================

PNL_CELL_STYLE = {
    "styleConditions": [
        {
            "condition":
                "params.value != null "
                "&& params.value > 0",
            "style": {
                "color": "#22c55e",
                "fontWeight": "600",
            },
        },
        {
            "condition":
                "params.value != null "
                "&& params.value < 0",
            "style": {
                "color": "#f87171",
                "fontWeight": "600",
            },
        },
    ],
}


STATUS_CELL_STYLE = {
    "styleConditions": [
        {
            "condition":
                "params.value === 'LIVE'",
            "style": {
                "color": "#22c55e",
                "fontWeight": "700",
            },
        },
        {
            "condition":
                "params.value === 'DELAYED'",
            "style": {
                "color": "#facc15",
                "fontWeight": "700",
            },
        },
        {
            "condition":
                "params.value === 'STALE'",
            "style": {
                "color": "#fb923c",
                "fontWeight": "700",
            },
        },
        {
            "condition":
                "params.value === 'UNAVAILABLE'",
            "style": {
                "color": "#f87171",
                "fontWeight": "700",
            },
        },
    ],
}


COLUMN_DEFS = [
    {
        "field": "display_name",
        "headerName": "Pair / Trade",
        "pinned": "left",
        "minWidth": 130,
        "headerTooltip": (
            "Sorting is applied at FX-pair level. Expanded child trades "
            "remain attached to their pair and stay ordered by trade date."
        ),
        "comparator": {
            "function": "comparePairRows",
        },
    },
    {
        "field": "side",
        "headerName": "Side",
        "minWidth": 95,
        "sortable": False,
        "headerTooltip": (
            "Sorting disabled because multi-trade pair rows show a net direction "
            "while child rows show individual trade directions."
        ),
    },
    {
        "field": "trade_date",
        "headerName": "Trade Date",
        "minWidth": 115,
        "sortable": False,
        "headerTooltip": (
            "Sorting disabled because a netted multi-trade pair has no single trade date."
        ),
    },
    {
        "field": "notional_base",
        "headerName": "Base / Net Notional",
        "headerTooltip": (
            "Trade rows show base notional; pair rows show "
            "absolute net base notional."
        ),
        "type": "numericColumn",
        "valueFormatter": {
            "function":
                "params.value == null ? '' : "
                "d3.format(',.0f')(params.value)"
        },
        "minWidth": 145,
    },
    {
        "field": "entry_price",
        "headerName": "Entry",
        "type": "numericColumn",
        "minWidth": 115,
        "sortable": False,
        "headerTooltip": (
            "Sorting disabled because a netted multi-trade pair has no single entry price."
        ),
    },
    {
        "field": "previous_close",
        "headerName": "Prev Close",
        "type": "numericColumn",
        "minWidth": 115,
    },
    {
        "field": "current_spot",
        "headerName": "Current Spot",
        "type": "numericColumn",
        "minWidth": 115,
    },
    {
        "field": "quote_status",
        "headerName": "Status",
        "cellStyle":
            STATUS_CELL_STYLE,
        "minWidth": 105,
    },
    {
        "field": "daily_pnl_usd",
        "headerName":
            "Daily P&L (USD)",
        "type": "numericColumn",
        "valueFormatter": {
            "function":
                "params.value == null ? '' : "
                "d3.format('$,.0f')(params.value)"
        },
        "cellStyle":
            PNL_CELL_STYLE,
        "minWidth": 135,
    },
    {
        "field":
            "inception_pnl_usd",
        "headerName":
            "Inception P&L (USD)",
        "type": "numericColumn",
        "valueFormatter": {
            "function":
                "params.value == null ? '' : "
                "d3.format('$,.0f')(params.value)"
        },
        "cellStyle":
            PNL_CELL_STYLE,
        "minWidth": 150,
    },
    {
        "field":
            "position_var_usd",
        "headerName":
            "Standalone VaR (USD)",
        "headerTooltip": (
            "Pair rows use VaR on net pair exposure; "
            "trade rows use standalone trade VaR."
        ),
        "type": "numericColumn",
        "valueFormatter": {
            "function":
                "params.value == null ? '' : "
                "d3.format('$,.0f')(params.value)"
        },
        "minWidth": 150,
    },
    {
        "field":
            "component_var_usd",
        "headerName":
            "Component VaR (USD)",
        "type": "numericColumn",
        "valueFormatter": {
            "function":
                "params.value == null ? '' : "
                "d3.format('$,.0f')(params.value)"
        },
        "minWidth": 155,
    },
    {
        "field":
            "incremental_var_usd",
        "headerName":
            "Incremental VaR (USD)",
        "headerTooltip": (
            "Pair rows remove the whole FX pair; trade rows "
            "remove only that trade."
        ),
        "type": "numericColumn",
        "valueFormatter": {
            "function":
                "params.value == null ? '' : "
                "d3.format('$,.0f')(params.value)"
        },
        "minWidth": 160,
    },
    {
        "field":
            "component_var_pct",
        "headerName":
            "VaR Contribution",
        "type": "numericColumn",
        "valueFormatter": {
            "function":
                "params.value == null ? '' : "
                "params.value.toFixed(1) + '%'"
        },
        "minWidth": 135,
    },
    {
        "field":
            "marginal_var_per_usd",
        "headerName":
            "Marginal VaR / $ Risk",
        "type": "numericColumn",
        "minWidth": 155,
    },
    {
        "field":
            "daily_volatility",
        "headerName":
            "Daily Vol",
        "type": "numericColumn",
        "valueFormatter": {
            "function":
                "params.value == null ? '' : "
                "params.value.toFixed(2) + '%'"
        },
        "minWidth": 105,
    },
    {
        "field":
            "pnl_per_1pct_move_usd",
        "headerName":
            "1% FX Move P&L (USD)",
        "type": "numericColumn",
        "valueFormatter": {
            "function":
                "params.value == null ? '' : "
                "d3.format('$,.0f')(params.value)"
        },
        "minWidth": 155,
    },
    {
        "field":
            "quote_age_minutes",
        "headerName":
            "Quote Age",
        "type": "numericColumn",
        "valueFormatter": {
            "function":
                "params.value == null ? '' : "
                "params.value.toFixed(0) + ' min'"
        },
        "minWidth": 105,
    },
]



def table_to_records(table):

    output = (
        table
        .copy()
        .astype(object)
        .where(
            pd.notna(table),
            None,
        )
    )

    return output.to_dict(
        "records"
    )


# ============================================================
# METHODOLOGY / QUALITY
# ============================================================

def build_methodology_text(
    confidence_level,
    lookback_days,
):

    return (
        f"1-day parametric VaR and Expected Shortfall "
        f"at {confidence_level:.1%} confidence, estimated "
        f"from the latest {lookback_days} valid daily FX "
        f"returns and their covariance. The model assumes "
        f"zero expected daily return and approximates "
        f"position P&L using USD FX risk sensitivities. "
        f"Component VaR allocates total portfolio VaR across "
        f"positions; negative contributions indicate hedging. "
        f"Incremental VaR is the change in portfolio VaR if "
        f"a whole position is removed and is not additive. "
        f"Within-pair Netting Benefit measures the reduction from "
        f"offsetting trades sharing the same FX risk factor. "
        f"Cross-Pair Diversification Benefit compares the sum of "
        f"standalone VaRs on net pair exposures with portfolio VaR."
    )


def build_data_quality_banner(
    position_table,
    risk_error,
):

    status_counts = (
        position_table[
            "quote_status"
        ]
        .value_counts()
        .to_dict()
    )

    live_count = status_counts.get(
        "LIVE",
        0,
    )

    delayed_count = (
        status_counts.get(
            "DELAYED",
            0,
        )
    )

    stale_count = status_counts.get(
        "STALE",
        0,
    )

    fallback_count = (
        status_counts.get(
            "DAILY FALLBACK",
            0,
        )
    )

    unavailable_count = (
        status_counts.get(
            "UNAVAILABLE",
            0,
        )
    )

    total_count = len(
        position_table
    )

    status_text = (
        f"{live_count} LIVE, "
        f"{delayed_count} DELAYED, "
        f"{stale_count} STALE"
    )

    if fallback_count > 0:
        status_text += (
            f", {fallback_count} "
            f"DAILY FALLBACK"
        )

    warnings = []

    if unavailable_count > 0:
        warnings.append(
            f"{unavailable_count} of "
            f"{total_count} positions "
            f"are unavailable."
        )

    if risk_error is not None:
        warnings.append(
            "Risk calculation unavailable: "
            + risk_error
        )

    if warnings:
        return (
            (
                "DATA QUALITY — "
                + status_text
                + ". "
                + " ".join(warnings)
            ),
            {
                "display": "block",
                "marginTop": "20px",
                "padding": "14px 18px",
                "borderRadius": "8px",
                "backgroundColor":
                    "#362d19",
                "border":
                    "1px solid #6b5725",
                "color": "#f0d58c",
            },
        )

    if (
        stale_count > 0
        or delayed_count > 0
        or fallback_count > 0
    ):
        return (
            (
                f"All {total_count} positions "
                f"have usable valuations — "
                f"{status_text}."
            ),
            {
                "display": "block",
                "marginTop": "20px",
                "padding": "14px 18px",
                "borderRadius": "8px",
                "backgroundColor":
                    "#362d19",
                "border":
                    "1px solid #6b5725",
                "color": "#f0d58c",
            },
        )

    return (
        (
            f"All {total_count} positions "
            f"have usable valuations — "
            f"{status_text}."
        ),
        {
            "display": "block",
            "marginTop": "20px",
            "padding": "14px 18px",
            "borderRadius": "8px",
            "backgroundColor":
                "#173326",
            "border":
                "1px solid #285f43",
            "color": "#a8e0bd",
        },
    )


# ============================================================
# LAYOUT
# ============================================================

def build_layout(
    portfolio,
):

    pair_options = [
        {
            "label": pair,
            "value": pair,
        }
        for pair in portfolio[
            "pair"
        ].drop_duplicates().tolist()
    ]

    return html.Div(
        style=PAGE_STYLE,
        children=[

            dcc.Store(
                id=(
                    "valuation-timestamp-store"
                )
            ),

            dcc.Store(
                id="current-pnl-store",
                data=[],
            ),

            dcc.Store(
                id="pnl-overlay-store",
                data=[],
            ),

            dcc.Store(
                id="position-table-store",
                data=[],
            ),

            dcc.Store(
                id="risk-table-store",
                data=[],
            ),

            dcc.Store(
                id="expanded-pairs-store",
                data=[],
            ),

            dcc.Store(
                id="component-var-selected-pair",
                data=None,
            ),

            dcc.Download(
                id="position-excel-download",
            ),

            # =================================================
            # HEADER
            # =================================================

            html.H1(
                "FX Portfolio Management Dashboard",
                style={
                    "marginBottom": "5px"
                },
            ),

            html.Div(
                "Asia / APAC FX Portfolio",
                style={
                    "color": "#8994aa",
                    "fontSize": "14px",
                },
            ),

            html.Div(
                (
                    "Daily P&L uses the previous completed "
                    "UTC daily FX bar. Times are displayed "
                    "in Singapore time."
                ),
                style={
                    "color": "#6f7b92",
                    "fontSize": "12px",
                    "marginTop": "5px",
                },
            ),

            html.Div(
                id="last-updated",
                style={
                    "color": "#6f7b92",
                    "fontSize": "12px",
                    "marginTop": "5px",
                },
            ),

            html.Div(
                id="data-quality-banner"
            ),

            # =================================================
            # SETTINGS
            # =================================================

            html.Div(
                style={
                    **PANEL_STYLE,
                    "display": "flex",
                    "gap": "20px",
                    "alignItems": "end",
                    "flexWrap": "wrap",
                },
                children=[

                    html.Div(
                        [
                            html.Div(
                                "VaR Confidence",
                                style=LABEL_STYLE,
                            ),

                            dcc.Dropdown(
                                id=(
                                    "confidence-level"
                                ),
                                options=[
                                    {
                                        "label": "90%",
                                        "value": 0.90,
                                    },
                                    {
                                        "label": "95%",
                                        "value": 0.95,
                                    },
                                    {
                                        "label": "97.5%",
                                        "value": 0.975,
                                    },
                                    {
                                        "label": "99%",
                                        "value": 0.99,
                                    },
                                ],
                                value=0.95,
                                clearable=False,
                                style={
                                    "width": "160px",
                                    "color": "black",
                                },
                            ),
                        ]
                    ),

                    html.Div(
                        [
                            html.Div(
                                "Lookback",
                                style=LABEL_STYLE,
                            ),

                            dcc.Dropdown(
                                id="lookback-days",
                                options=[
                                    {
                                        "label":
                                            "126 days",
                                        "value": 126,
                                    },
                                    {
                                        "label":
                                            "252 days",
                                        "value": 252,
                                    },
                                    {
                                        "label":
                                            "504 days",
                                        "value": 504,
                                    },
                                ],
                                value=252,
                                clearable=False,
                                style={
                                    "width": "160px",
                                    "color": "black",
                                },
                            ),
                        ]
                    ),

                    html.Button(
                        "Refresh Market Data",
                        id="refresh-button",
                        n_clicks=0,
                        style={
                            "height": "38px",
                            "padding": "0 18px",
                            "borderRadius": "6px",
                            "border":
                                "1px solid #415078",
                            "backgroundColor":
                                "#25304d",
                            "color": "white",
                            "cursor": "pointer",
                            "fontWeight": "600",
                        },
                    ),
                ],
            ),

            # =================================================
            # HEADLINE METRICS
            # =================================================

            html.Div(
                style={
                    "display": "flex",
                    "gap": "14px",
                    "marginTop": "20px",
                    "flexWrap": "wrap",
                },
                children=[

                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div(
                                "Daily P&L (USD)",
                                style=LABEL_STYLE,
                            ),
                            html.Div(
                                id=(
                                    "daily-pnl-card"
                                ),
                                style=VALUE_STYLE,
                            ),
                            html.Div(
                                "% of gross initial USD notional",
                                style=SMALL_NOTE_STYLE,
                            ),
                        ],
                    ),

                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div(
                                "Inception P&L (USD)",
                                style=LABEL_STYLE,
                            ),
                            html.Div(
                                id=(
                                    "inception-pnl-card"
                                ),
                                style=VALUE_STYLE,
                            ),
                            html.Div(
                                "% of gross initial USD notional",
                                style=SMALL_NOTE_STYLE,
                            ),
                        ],
                    ),

                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div(
                                "1-Day VaR (USD)",
                                style=LABEL_STYLE,
                            ),
                            html.Div(
                                id=(
                                    "portfolio-var-card"
                                ),
                                style=VALUE_STYLE,
                            ),
                        ],
                    ),

                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div(
                                (
                                    "1-Day Expected "
                                    "Shortfall (USD)"
                                ),
                                style=LABEL_STYLE,
                            ),
                            html.Div(
                                id=(
                                    "expected-shortfall-card"
                                ),
                                style=VALUE_STYLE,
                            ),
                        ],
                    ),

                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div(
                                "1-Day Vol (USD)",
                                style=LABEL_STYLE,
                            ),
                            html.Div(
                                id=(
                                    "portfolio-vol-card"
                                ),
                                style=VALUE_STYLE,
                            ),
                        ],
                    ),

                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div(
                                "Netting Benefit",
                                style=LABEL_STYLE,
                            ),
                            html.Div(
                                id=(
                                    "netting-benefit-card"
                                ),
                                style=VALUE_STYLE,
                            ),
                            html.Div(
                                (
                                    "Trade VaR sum minus net-pair "
                                    "standalone VaR sum"
                                ),
                                style=SMALL_NOTE_STYLE,
                            ),
                        ],
                    ),

                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div(
                                "Cross-Pair Diversification",
                                style=LABEL_STYLE,
                            ),
                            html.Div(
                                id=(
                                    "cross-pair-diversification-card"
                                ),
                                style=VALUE_STYLE,
                            ),
                            html.Div(
                                (
                                    "Net-pair VaR sum minus "
                                    "portfolio VaR"
                                ),
                                style=SMALL_NOTE_STYLE,
                            ),
                        ],
                    ),
                ],
            ),

            # =================================================
            # HISTORICAL P&L
            # =================================================

            html.Div(
                style=PANEL_STYLE,
                children=[

                    html.Div(
                        style={
                            "display": "flex",
                            "justifyContent":
                                "space-between",
                            "alignItems": "end",
                            "gap": "20px",
                            "flexWrap": "wrap",
                        },
                        children=[

                            html.Div(
                                [
                                    html.H3(
                                        "Historical P&L",
                                        style={
                                            "margin":
                                                "0 0 4px 0"
                                        },
                                    ),

                                    html.Div(
                                        (
                                            "Portfolio Daily P&L "
                                            "reflects the full book. "
                                            "Select an FX pair to overlay. "
                                            "Multi-trade pairs show the net "
                                            "pair P&L and each constituent trade."
                                        ),
                                        style={
                                            "color":
                                                "#8994aa",
                                            "fontSize":
                                                "12px",
                                        },
                                    ),
                                ]
                            ),

                            html.Div(
                                style={
                                    "display": "flex",
                                    "alignItems": "end",
                                    "gap": "10px",
                                },
                                children=[

                                    html.Div(
                                        [
                                            html.Div(
                                                (
                                                    "Add Pair "
                                                    "Overlay"
                                                ),
                                                style=LABEL_STYLE,
                                            ),

                                            dcc.Dropdown(
                                                id=(
                                                    "pnl-position-selector"
                                                ),
                                                options=(
                                                    pair_options
                                                ),
                                                value=None,
                                                multi=False,
                                                clearable=True,
                                                placeholder=(
                                                    "Select a "
                                                    "pair..."
                                                ),
                                                style={
                                                    "width":
                                                        "260px",
                                                    "fontSize":
                                                        "13px",
                                                    "color":
                                                        "black",
                                                },
                                            ),
                                        ]
                                    ),

                                    html.Button(
                                        "Clear Overlays",
                                        id=(
                                            "clear-overlays-button"
                                        ),
                                        n_clicks=0,
                                        style={
                                            "height": "38px",
                                            "padding":
                                                "0 14px",
                                            "borderRadius":
                                                "6px",
                                            "border":
                                                (
                                                    "1px solid "
                                                    "#415078"
                                                ),
                                            "backgroundColor":
                                                "#25304d",
                                            "color":
                                                "white",
                                            "cursor":
                                                "pointer",
                                        },
                                    ),
                                ],
                            ),
                        ],
                    ),

                    dcc.Graph(
                        id="daily-pnl-chart",
                        config={
                            "displayModeBar":
                                False
                        },
                    ),

                    html.Div(
                        style={
                            "display": "flex",
                            "gap": "14px",
                            "flexWrap": "wrap",
                            "marginTop": "8px",
                        },
                        children=[

                            html.Div(
                                style=SMALL_CARD_STYLE,
                                children=[
                                    html.Div(
                                        (
                                            "Max P&L "
                                            "Drawdown (USD)"
                                        ),
                                        style=LABEL_STYLE,
                                    ),
                                    html.Div(
                                        id=(
                                            "max-drawdown-card"
                                        ),
                                        style=(
                                            SMALL_VALUE_STYLE
                                        ),
                                    ),
                                ],
                            ),

                            html.Div(
                                style=SMALL_CARD_STYLE,
                                children=[
                                    html.Div(
                                        "Best Day (USD)",
                                        style=LABEL_STYLE,
                                    ),
                                    html.Div(
                                        id=(
                                            "best-day-card"
                                        ),
                                        style=(
                                            SMALL_VALUE_STYLE
                                        ),
                                    ),
                                ],
                            ),

                            html.Div(
                                style=SMALL_CARD_STYLE,
                                children=[
                                    html.Div(
                                        "Worst Day (USD)",
                                        style=LABEL_STYLE,
                                    ),
                                    html.Div(
                                        id=(
                                            "worst-day-card"
                                        ),
                                        style=(
                                            SMALL_VALUE_STYLE
                                        ),
                                    ),
                                ],
                            ),

                            html.Div(
                                style=SMALL_CARD_STYLE,
                                children=[
                                    html.Div(
                                        (
                                            "Annualised "
                                            "P&L Sharpe"
                                        ),
                                        style=LABEL_STYLE,
                                    ),
                                    html.Div(
                                        id="sharpe-card",
                                        style=(
                                            SMALL_VALUE_STYLE
                                        ),
                                    ),
                                    html.Div(
                                        "0 P&L benchmark",
                                        style=(
                                            SMALL_NOTE_STYLE
                                        ),
                                    ),
                                ],
                            ),

                            html.Div(
                                style=SMALL_CARD_STYLE,
                                children=[
                                    html.Div(
                                        (
                                            "Annualised "
                                            "P&L Sortino"
                                        ),
                                        style=LABEL_STYLE,
                                    ),
                                    html.Div(
                                        id="sortino-card",
                                        style=(
                                            SMALL_VALUE_STYLE
                                        ),
                                    ),
                                    html.Div(
                                        (
                                            "0 P&L downside "
                                            "benchmark"
                                        ),
                                        style=(
                                            SMALL_NOTE_STYLE
                                        ),
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            # =================================================
            # STRESS TEST
            # =================================================

            html.Div(
                style=PANEL_STYLE,
                children=[

                    html.H3(
                        (
                            "Stress & Scenario "
                            "Analysis"
                        ),
                        style={
                            "margin":
                                "0 0 4px 0"
                        },
                    ),

                    html.Div(
                        (
                            "Exact spot revaluation under coherent "
                            "USD strengthening and weakening shocks. "
                            "Equal positive and negative shocks can "
                            "produce slightly different P&L because "
                            "USD-base position valuation is nonlinear "
                            "in spot."
                        ),
                        style={
                            "color":
                                "#8994aa",
                            "fontSize":
                                "12px",
                        },
                    ),

                    dcc.Graph(
                        id=(
                            "stress-test-chart"
                        ),
                        config={
                            "displayModeBar":
                                False
                        },
                    ),
                ],
            ),

            # =================================================
            # POSITIONS
            # =================================================

            html.Div(
                style=PANEL_STYLE,
                children=[

                    html.Div(
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "alignItems": "start",
                            "gap": "16px",
                            "marginBottom": "12px",
                        },
                        children=[
                            html.Div(
                                children=[
                                    html.H3(
                                        "Current Positions",
                                        style={
                                            "marginTop": "0",
                                            "marginBottom": "4px",
                                        },
                                    ),
                                    html.Div(
                                        (
                                            "Pairs with multiple trades are shown netted. "
                                            "Sorting is applied at pair level; expanded trades "
                                            "remain attached and ordered by trade date."
                                        ),
                                        style={
                                            "color": "#8994aa",
                                            "fontSize": "12px",
                                        },
                                    ),
                                ],
                            ),
                            html.Button(
                                "Export to Excel",
                                id="export-positions-excel-button",
                                n_clicks=0,
                                style={
                                    "height": "38px",
                                    "padding": "0 16px",
                                    "borderRadius": "6px",
                                    "border": "1px solid #415078",
                                    "backgroundColor": "#25304d",
                                    "color": "white",
                                    "cursor": "pointer",
                                    "fontWeight": "600",
                                    "whiteSpace": "nowrap",
                                },
                            ),
                        ],
                    ),

                    dag.AgGrid(
                        id="position-grid",
                        columnDefs=COLUMN_DEFS,
                        rowData=[],

                        defaultColDef={
                            "sortable": True,
                            "filter": False,
                            "resizable": True,
                            "minWidth": 105,
                        },

                        dashGridOptions={
                            "domLayout":
                                "autoHeight",

                            "rowHeight":
                                40,

                            "headerHeight":
                                42,

                            "suppressCellFocus":
                                True,

                            "tooltipShowDelay":
                                250,

                            # Native AG Grid sorting can otherwise separate
                            # child trades from their pair parent. Rebuild the
                            # displayed order after every sort so parent rows
                            # define block order and child trades remain attached.
                            "postSortRows": {
                                "function": "keepTradesWithPair(params)",
                            },

                            "theme": {
                                "function": (
                                    "themeQuartz."
                                    "withParams({"
                                    "backgroundColor:"
                                    "'#101729',"
                                    "foregroundColor:"
                                    "'#e5e7eb',"
                                    "headerBackgroundColor:"
                                    "'#0f172a',"
                                    "headerTextColor:"
                                    "'#cbd5e1',"
                                    "oddRowBackgroundColor:"
                                    "'#121a2d',"
                                    "accentColor:"
                                    "'#60a5fa',"
                                    "borderColor:"
                                    "'#26314d',"
                                    "rowBorderColor:"
                                    "'#26314d',"
                                    "headerColumnResize"
                                    "HandleColor:"
                                    "'#475569'"
                                    "})"
                                )
                            },
                        },

                        style={
                            "width": "100%"
                        },
                    ),
                ],
            ),

            # =================================================
            # RISK ANALYTICS
            # =================================================

            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns":
                        (
                            "repeat(auto-fit, "
                            "minmax(450px, 1fr))"
                        ),
                    "gap": "20px",
                    "marginTop": "20px",
                },
                children=[

                    html.Div(
                        style={
                            **PANEL_STYLE,
                            "marginTop": "0",
                        },
                        children=[
                            html.Div(
                                style={
                                    "display": "flex",
                                    "justifyContent": "flex-end",
                                    "minHeight": "34px",
                                },
                                children=[
                                    html.Button(
                                        "← Pair View",
                                        id=(
                                            "component-var-back-button"
                                        ),
                                        n_clicks=0,
                                        style={
                                            "display": "none",
                                            "height": "30px",
                                            "padding": "0 12px",
                                            "borderRadius": "6px",
                                            "border": (
                                                "1px solid #415078"
                                            ),
                                            "backgroundColor": "#25304d",
                                            "color": "white",
                                            "cursor": "pointer",
                                        },
                                    ),
                                ],
                            ),
                            dcc.Graph(
                                id=(
                                    "component-var-chart"
                                ),
                                config={
                                    "displayModeBar":
                                        False
                                },
                            ),
                        ],
                    ),

                    html.Div(
                        style={
                            **PANEL_STYLE,
                            "marginTop": "0",
                        },
                        children=[
                            dcc.Graph(
                                id=(
                                    "correlation-chart"
                                ),
                                config={
                                    "displayModeBar":
                                        False
                                },
                            ),
                        ],
                    ),
                ],
            ),

            # =================================================
            # METHODOLOGY
            # =================================================

            html.Div(
                style=PANEL_STYLE,
                children=[

                    html.Div(
                        "Risk Methodology",
                        style=LABEL_STYLE,
                    ),

                    html.P(
                        id=(
                            "risk-methodology-text"
                        ),
                        style={
                            "marginBottom": "0",
                            "color": "#b5bdd0",
                            "lineHeight": "1.6",
                        },
                    ),
                ],
            ),
        ],
    )
