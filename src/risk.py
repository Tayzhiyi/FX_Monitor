from statistics import NormalDist
from math import (
    exp,
    pi,
    sqrt,
)

import numpy as np
import pandas as pd


from src.market_data import (
    load_portfolio,
    get_all_historical_prices,
    get_completed_bar_cutoff_date,
)

from src.pnl import (
    calculate_portfolio_pnl,
    side_multiplier,
)


# ============================================================
# RETURN HISTORY
# ============================================================

def prepare_risk_returns(
    historical_prices,
    valuation_timestamp,
    lookback_days=252,
):
    """
    Prepare aligned one-day FX returns for VaR.

    Assumptions:
    - Yahoo daily FX bars are treated as UTC-day bars.
    - Only completed bars are used.
    - Missing observations are not forward-filled.
    - Returns spanning a missing Monday-Friday business
      date are removed.
    - Pair-specific holidays are not modelled.
    """

    if historical_prices.empty:

        raise ValueError(
            "Historical price data is empty."
        )

    cutoff_date = (
        get_completed_bar_cutoff_date(
            valuation_timestamp
        )
    )

    prices = (
        historical_prices
        .copy()
        .sort_index()
    )

    # --------------------------------------------------------
    # Completed UTC daily bars only
    # --------------------------------------------------------

    completed_mask = [
        pd.Timestamp(index).date()
        < cutoff_date
        for index in prices.index
    ]

    prices = (
        prices.loc[
            completed_mask
        ]
    )

    if prices.empty:

        raise ValueError(
            "No completed historical "
            "observations before UTC cutoff "
            f"{cutoff_date}."
        )

    if len(prices) < 2:

        raise ValueError(
            "Not enough historical prices "
            "to calculate returns."
        )

    # --------------------------------------------------------
    # Calculate returns BEFORE dropping missing values
    #
    # This prevents missing values within an existing date
    # from being bridged.
    # --------------------------------------------------------

    returns = (
        prices
        .pct_change(
            fill_method=None
        )
    )

    # --------------------------------------------------------
    # Generic FX weekday continuity check
    #
    # Friday -> Monday is valid.
    # Monday -> Wednesday is not, because Tuesday is absent.
    #
    # Pair-specific holidays are intentionally not modelled.
    # --------------------------------------------------------

    valid_daily_gap = pd.Series(
        False,
        index=prices.index,
        dtype=bool,
    )

    for i in range(
        1,
        len(prices.index),
    ):

        previous_date = (
            pd.Timestamp(
                prices.index[i - 1]
            ).normalize()
        )

        current_date = (
            pd.Timestamp(
                prices.index[i]
            ).normalize()
        )

        expected_previous_date = (
            current_date
            - pd.offsets.BDay(1)
        )

        valid_daily_gap.iloc[i] = (
            previous_date
            == expected_previous_date
        )

    returns = (
        returns.loc[
            valid_daily_gap
        ]
    )

    # --------------------------------------------------------
    # Finite aligned returns only
    # --------------------------------------------------------

    returns = (
        returns
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            how="any"
        )
    )

    if len(returns) < lookback_days:

        raise ValueError(
            f"Requested {lookback_days} "
            f"one-day return observations, "
            f"but only {len(returns)} valid "
            "aligned observations are available."
        )

    return returns.tail(
        lookback_days
    )


# ============================================================
# USD RISK EXPOSURE
# ============================================================

def calculate_trade_risk_exposure(
    pair,
    side,
    notional_base,
    entry_price,
    current_spot,
):
    """
    Calculate signed USD exposure to the FX pair's
    proportional return.

    Approximation:

        dP&L ≈ exposure_usd * FX_return

    Therefore a 1% FX move produces approximately:

        exposure_usd * 0.01

    of USD P&L.
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
    # x = d * N * S
    # --------------------------------------------------------

    if quote_currency == "USD":

        exposure_usd = (
            sign
            * notional_base
            * current_spot
        )

    # --------------------------------------------------------
    # USD base pair
    #
    # USDJPY:
    #
    # x = d * N * K / S
    # --------------------------------------------------------

    elif base_currency == "USD":

        exposure_usd = (
            sign
            * notional_base
            * entry_price
            / current_spot
        )

    else:

        raise ValueError(
            f"{pair} does not contain USD. "
            "Cross-currency risk is "
            "not supported."
        )

    return float(
        exposure_usd
    )


def calculate_normal_expected_shortfall(
    portfolio_volatility_usd,
    confidence_level=0.95,
):
    """
    Calculate one-day Expected Shortfall under the
    same zero-mean normal model used for parametric VaR.

    Expected Shortfall is the expected loss conditional
    on exceeding the VaR threshold.
    """

    if not (
        0.5
        < confidence_level
        < 1.0
    ):

        raise ValueError(
            "confidence_level must be "
            "between 0.5 and 1.0."
        )

    if (
        portfolio_volatility_usd
        is None
    ):

        return None

    if (
        portfolio_volatility_usd
        < 0
    ):

        raise ValueError(
            "Portfolio volatility "
            "cannot be negative."
        )

    if (
        portfolio_volatility_usd
        == 0
    ):

        return 0.0

    z_score = (
        NormalDist()
        .inv_cdf(
            confidence_level
        )
    )

    normal_pdf = (
        exp(
            -0.5
            * z_score ** 2
        )
        / sqrt(
            2
            * pi
        )
    )

    expected_shortfall = (
        portfolio_volatility_usd
        * normal_pdf
        / (
            1
            - confidence_level
        )
    )

    return float(
        expected_shortfall
    )

# ============================================================
# PARAMETRIC VAR
# ============================================================

def calculate_var(
    pnl,
    returns,
    confidence_level=0.95,
):
    """
    Calculate one-day normal parametric
    variance-covariance VaR.

    Assumptions:
    - zero expected return
    - linearized USD P&L
    - covariance estimated from aligned daily returns

    Returns:
        risk table
        summary
        covariance matrix
        correlation matrix
    """

    if pnl.empty:

        raise ValueError(
            "Cannot calculate VaR "
            "for an empty portfolio."
        )

    if not (
        0.5
        < confidence_level
        < 1.0
    ):

        raise ValueError(
            "confidence_level must be "
            "between 0.5 and 1.0."
        )

    if returns.empty:

        raise ValueError(
            "Return history is empty."
        )

    z_score = (
        NormalDist()
        .inv_cdf(
            confidence_level
        )
    )

    # --------------------------------------------------------
    # Trade-level exposures
    # --------------------------------------------------------

    risk = pnl[
        [
            "trade_id",
            "pair",
            "side",
            "notional_base",
            "entry_price",
            "current_spot",
        ]
    ].copy()

    risk[
        "signed_usd_exposure"
    ] = risk.apply(
        lambda row:
        calculate_trade_risk_exposure(
            pair=row["pair"],
            side=row["side"],
            notional_base=row[
                "notional_base"
            ],
            entry_price=row[
                "entry_price"
            ],
            current_spot=row[
                "current_spot"
            ],
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # Required risk factors
    # --------------------------------------------------------

    required_pairs = set(
        risk["pair"]
    )

    missing_pairs = (
        required_pairs
        - set(
            returns.columns
        )
    )

    if missing_pairs:

        raise ValueError(
            "Missing historical return "
            "data for: "
            f"{sorted(missing_pairs)}"
        )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    volatility = (
        returns.std(
            ddof=1
        )
    )

    if not np.isfinite(
        volatility.loc[
            list(required_pairs)
        ].to_numpy()
    ).all():

        raise ValueError(
            "Non-finite FX volatility "
            "estimate detected."
        )

    risk[
        "daily_volatility"
    ] = (
        risk["pair"]
        .map(
            volatility
        )
    )

    # --------------------------------------------------------
    # Standalone position VaR
    # --------------------------------------------------------

    risk[
        "position_var_usd"
    ] = (
        z_score
        * risk[
            "signed_usd_exposure"
        ].abs()
        * risk[
            "daily_volatility"
        ]
    )

    # --------------------------------------------------------
    # Aggregate exposure by FX factor
    # --------------------------------------------------------

    factor_exposure = (
        risk
        .groupby(
            "pair"
        )[
            "signed_usd_exposure"
        ]
        .sum()
    )

    factor_pairs = list(
        factor_exposure.index
    )

    # --------------------------------------------------------
    # Covariance / correlation
    # --------------------------------------------------------

    covariance = (
        returns[
            factor_pairs
        ]
        .cov()
    )

    correlation = (
        returns[
            factor_pairs
        ]
        .corr()
    )

    if not np.isfinite(
        covariance.to_numpy()
    ).all():

        raise ValueError(
            "Covariance matrix contains "
            "non-finite values."
        )

    if not np.isfinite(
        correlation.to_numpy()
    ).all():

        raise ValueError(
            "Correlation matrix contains "
            "non-finite values."
        )

    exposure_vector = (
        factor_exposure[
            factor_pairs
        ]
        .to_numpy(
            dtype=float
        )
    )

    covariance_matrix = (
        covariance
        .loc[
            factor_pairs,
            factor_pairs,
        ]
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # Portfolio variance
    #
    # sigma^2 = x' Sigma x
    # --------------------------------------------------------

    portfolio_variance = float(
        exposure_vector.T
        @ covariance_matrix
        @ exposure_vector
    )

    # Protect against tiny negative floating-point values.
    portfolio_variance = max(
        portfolio_variance,
        0.0,
    )

    portfolio_volatility_usd = (
        np.sqrt(
            portfolio_variance
        )
    )

    # --------------------------------------------------------
    # Portfolio VaR
    #
    # VaR = z * sqrt(x' Sigma x)
    # --------------------------------------------------------

    portfolio_var_usd = (
        z_score
        * portfolio_volatility_usd
    )

    # --------------------------------------------------------
    # Incremental VaR
    #
    # Incremental VaR_i = VaR(portfolio) - VaR(portfolio
    # without position i)
    #
    # Positive: removing the position reduces portfolio VaR.
    # Negative: the position is acting as a hedge/diversifier,
    # because removing it increases portfolio VaR.
    #
    # Incremental VaR is not additive across positions.
    # --------------------------------------------------------

    pair_index = {
        pair: index
        for index, pair
        in enumerate(factor_pairs)
    }

    incremental_var_values = []

    for _, position in risk.iterrows():

        reduced_exposure_vector = (
            exposure_vector.copy()
        )

        reduced_exposure_vector[
            pair_index[
                position["pair"]
            ]
        ] -= float(
            position[
                "signed_usd_exposure"
            ]
        )

        reduced_variance = float(
            reduced_exposure_vector.T
            @ covariance_matrix
            @ reduced_exposure_vector
        )

        reduced_variance = max(
            reduced_variance,
            0.0,
        )

        reduced_var_usd = (
            z_score
            * np.sqrt(
                reduced_variance
            )
        )

        incremental_var_values.append(
            portfolio_var_usd
            - reduced_var_usd
        )

    risk[
        "incremental_var_usd"
    ] = incremental_var_values

    # --------------------------------------------------------
    # Netting and cross-pair diversification
    #
    # Trade standalone VaR sum treats every trade separately.
    # Pair standalone VaR first nets signed exposures within
    # each FX pair, then applies that pair's volatility.
    #
    # Netting Benefit
    #   = sum(trade standalone VaR)
    #     - sum(net-pair standalone VaR)
    #
    # Cross-Pair Diversification Benefit
    #   = sum(net-pair standalone VaR)
    #     - portfolio VaR
    # --------------------------------------------------------

    standalone_var_total_usd = float(
        risk[
            "position_var_usd"
        ].sum()
    )

    pair_standalone_var = (
        z_score
        * factor_exposure.abs()
        * volatility.reindex(
            factor_pairs
        )
    )

    # Pair-level metrics are mapped back to every constituent
    # trade so the dashboard can render a pair parent row while
    # preserving trade-level drilldown rows.
    risk[
        "pair_signed_usd_exposure"
    ] = risk["pair"].map(
        factor_exposure
    )

    risk[
        "pair_standalone_var_usd"
    ] = risk["pair"].map(
        pair_standalone_var
    )

    pair_incremental_var = {}

    for pair in factor_pairs:

        reduced_exposure_vector = (
            exposure_vector.copy()
        )

        reduced_exposure_vector[
            pair_index[pair]
        ] = 0.0

        reduced_variance = float(
            reduced_exposure_vector.T
            @ covariance_matrix
            @ reduced_exposure_vector
        )

        reduced_variance = max(
            reduced_variance,
            0.0,
        )

        reduced_pair_var_usd = (
            z_score
            * np.sqrt(
                reduced_variance
            )
        )

        pair_incremental_var[pair] = (
            portfolio_var_usd
            - reduced_pair_var_usd
        )

    risk[
        "pair_incremental_var_usd"
    ] = risk["pair"].map(
        pair_incremental_var
    )

    pair_standalone_var_total_usd = float(
        pair_standalone_var.sum()
    )

    netting_benefit_usd = (
        standalone_var_total_usd
        - pair_standalone_var_total_usd
    )

    if np.isclose(
        standalone_var_total_usd,
        0.0,
    ):
        netting_benefit_pct = np.nan
    else:
        netting_benefit_pct = (
            netting_benefit_usd
            / standalone_var_total_usd
            * 100
        )

    cross_pair_diversification_benefit_usd = (
        pair_standalone_var_total_usd
        - portfolio_var_usd
    )

    if np.isclose(
        pair_standalone_var_total_usd,
        0.0,
    ):
        cross_pair_diversification_benefit_pct = np.nan
    else:
        cross_pair_diversification_benefit_pct = (
            cross_pair_diversification_benefit_usd
            / pair_standalone_var_total_usd
            * 100
        )

    # Backward-compatible total benefit. This combines
    # within-pair netting and cross-pair diversification.
    diversification_benefit_usd = (
        standalone_var_total_usd
        - portfolio_var_usd
    )

    if np.isclose(
        standalone_var_total_usd,
        0.0,
    ):
        diversification_benefit_pct = np.nan
    else:
        diversification_benefit_pct = (
            diversification_benefit_usd
            / standalone_var_total_usd
            * 100
        )

    # --------------------------------------------------------
    # Fully hedged / zero-risk portfolio
    # --------------------------------------------------------

    if np.isclose(
        portfolio_volatility_usd,
        0.0,
    ):

        risk[
            "marginal_var_per_usd"
        ] = np.nan

        risk[
            "component_var_usd"
        ] = 0.0

        risk[
            "component_var_pct"
        ] = np.nan

        risk[
            "pair_component_var_usd"
        ] = 0.0

        risk[
            "pair_component_var_pct"
        ] = np.nan

        summary = {
            "confidence_level":
                confidence_level,

            "z_score":
                z_score,

            "observations":
                len(returns),

            "portfolio_volatility_usd":
                0.0,

            "portfolio_var_usd":
                0.0,

            "standalone_var_total_usd":
                standalone_var_total_usd,

            "pair_standalone_var_total_usd":
                pair_standalone_var_total_usd,

            "netting_benefit_usd":
                netting_benefit_usd,

            "netting_benefit_pct":
                netting_benefit_pct,

            "cross_pair_diversification_benefit_usd":
                cross_pair_diversification_benefit_usd,

            "cross_pair_diversification_benefit_pct":
                cross_pair_diversification_benefit_pct,

            "diversification_benefit_usd":
                diversification_benefit_usd,

            "diversification_benefit_pct":
                diversification_benefit_pct,

            "component_var_total_usd":
                0.0,

            "reconciliation":
                0.0,
        }

        return (
            risk,
            summary,
            covariance,
            correlation,
        )

    # --------------------------------------------------------
    # Marginal VaR per USD signed factor exposure
    #
    # MVaR = z * Sigma x / sigma_P
    # --------------------------------------------------------

    covariance_times_exposure = (
        covariance_matrix
        @ exposure_vector
    )

    marginal_var = (
        z_score
        * covariance_times_exposure
        / portfolio_volatility_usd
    )

    marginal_var_by_pair = (
        pd.Series(
            marginal_var,
            index=factor_pairs,
        )
    )

    risk[
        "marginal_var_per_usd"
    ] = (
        risk["pair"]
        .map(
            marginal_var_by_pair
        )
    )

    # --------------------------------------------------------
    # Component VaR
    #
    # CVaR_i = x_i * MVaR_pair(i)
    #
    # Negative values are legitimate and indicate
    # diversification / hedging contributions.
    # --------------------------------------------------------

    risk[
        "component_var_usd"
    ] = (
        risk[
            "signed_usd_exposure"
        ]
        * risk[
            "marginal_var_per_usd"
        ]
    )

    risk[
        "component_var_pct"
    ] = (
        risk[
            "component_var_usd"
        ]
        / portfolio_var_usd
        * 100
    )

    pair_component_var = (
        factor_exposure
        * marginal_var_by_pair
    )

    risk[
        "pair_component_var_usd"
    ] = risk["pair"].map(
        pair_component_var
    )

    risk[
        "pair_component_var_pct"
    ] = (
        risk[
            "pair_component_var_usd"
        ]
        / portfolio_var_usd
        * 100
    )

    component_var_total = (
        risk[
            "component_var_usd"
        ].sum()
    )

    reconciliation = (
        component_var_total
        - portfolio_var_usd
    )

    summary = {
        "confidence_level":
            confidence_level,

        "z_score":
            z_score,

        "observations":
            len(returns),

        "portfolio_volatility_usd":
            portfolio_volatility_usd,

        "portfolio_var_usd":
            portfolio_var_usd,

        "standalone_var_total_usd":
            standalone_var_total_usd,

        "pair_standalone_var_total_usd":
            pair_standalone_var_total_usd,

        "netting_benefit_usd":
            netting_benefit_usd,

        "netting_benefit_pct":
            netting_benefit_pct,

        "cross_pair_diversification_benefit_usd":
            cross_pair_diversification_benefit_usd,

        "cross_pair_diversification_benefit_pct":
            cross_pair_diversification_benefit_pct,

        "diversification_benefit_usd":
            diversification_benefit_usd,

        "diversification_benefit_pct":
            diversification_benefit_pct,

        "component_var_total_usd":
            component_var_total,

        "reconciliation":
            reconciliation,
    }

    return (
        risk,
        summary,
        covariance,
        correlation,
    )


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    CONFIDENCE_LEVEL = 0.95
    LOOKBACK_DAYS = 252

    valuation_timestamp = (
        pd.Timestamp.now(
            tz="UTC"
        )
    )

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
            period="3y",
        )
    )

    pnl = (
        calculate_portfolio_pnl(
            portfolio=portfolio,
            historical_prices=historical_prices,
            valuation_timestamp=valuation_timestamp,
        )
    )

    returns = (
        prepare_risk_returns(
            historical_prices=historical_prices,
            valuation_timestamp=valuation_timestamp,
            lookback_days=LOOKBACK_DAYS,
        )
    )

    (
        risk,
        summary,
        covariance,
        correlation,
    ) = calculate_var(
        pnl=pnl,
        returns=returns,
        confidence_level=CONFIDENCE_LEVEL,
    )

    print(
        "\nRISK SETTINGS"
    )

    print(
        f"Confidence level: "
        f"{summary['confidence_level']:.1%}"
    )

    print(
        f"Lookback observations: "
        f"{summary['observations']}"
    )

    print(
        f"Z-score: "
        f"{summary['z_score']:.4f}"
    )

    print(
        "\nPOSITION RISK"
    )

    display_columns = [
        "trade_id",
        "pair",
        "side",
        "signed_usd_exposure",
        "daily_volatility",
        "position_var_usd",
        "marginal_var_per_usd",
        "component_var_usd",
        "component_var_pct",
        "incremental_var_usd",
    ]

    print(
        risk[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nPORTFOLIO RISK"
    )

    print(
        f"1-day portfolio volatility: "
        f"${summary['portfolio_volatility_usd']:,.2f}"
    )

    print(
        f"1-day VaR: "
        f"${summary['portfolio_var_usd']:,.2f}"
    )

    print(
        f"Within-pair netting benefit: "
        f"${summary['netting_benefit_usd']:,.2f} "
        f"({summary['netting_benefit_pct']:.1f}%)"
    )

    print(
        f"Cross-pair diversification benefit: "
        f"${summary['cross_pair_diversification_benefit_usd']:,.2f} "
        f"({summary['cross_pair_diversification_benefit_pct']:.1f}%)"
    )

    print(
        f"Component VaR total: "
        f"${summary['component_var_total_usd']:,.2f}"
    )

    print(
        f"VaR reconciliation: "
        f"${summary['reconciliation']:,.8f}"
    )

    print(
        "\nCOVARIANCE MATRIX"
    )

    print(
        covariance.round(8)
    )

    print(
        "\nCORRELATION MATRIX"
    )

    print(
        correlation.round(3)
    )