import math
import sys
import types
import unittest
from statistics import NormalDist

import numpy as np
import pandas as pd

# The test suite does not make web calls. Some project modules import
# yfinance at module load time, so provide a tiny stub if the package is
# unavailable in the test environment. Your normal project environment
# can use the real yfinance package unchanged.
try:
    import yfinance  # noqa: F401
except ModuleNotFoundError:
    sys.modules["yfinance"] = types.SimpleNamespace()

from src.analytics import calculate_stress_scenarios
from src.market_data import validate_current_quote
from src.pnl import calculate_mtm_usd
from src.risk import (
    calculate_normal_expected_shortfall,
    calculate_trade_risk_exposure,
    calculate_var,
)


TOL = 1e-8


def make_returns(n=504):
    """Deterministic, non-stationary synthetic FX returns for repeatable tests."""
    t = np.arange(n, dtype=float)

    # Deliberately change volatility through time so 126/252/504-day
    # lookbacks do not produce identical risk estimates.
    regime = np.where(
        t >= n - 126,
        1.70,
        np.where(t >= n - 252, 1.20, 0.80),
    )

    common = regime * (
        0.0028 * np.sin(0.23 * t)
        + 0.0012 * np.cos(0.071 * t)
    )

    usdjpy = common + regime * 0.0010 * np.sin(0.41 * t)
    usdsgd = -0.25 * common + regime * 0.0007 * np.cos(0.29 * t)
    audusd = 0.55 * common + regime * 0.0015 * np.sin(0.17 * t + 0.4)

    index = pd.bdate_range("2024-01-02", periods=n)

    return pd.DataFrame(
        {
            "USDJPY": usdjpy,
            "USDSGD": usdsgd,
            "AUDUSD": audusd,
        },
        index=index,
    )


def make_pnl_snapshot():
    """Synthetic current portfolio snapshot; no web calls required."""
    return pd.DataFrame(
        [
            {
                "trade_id": "T1",
                "pair": "USDJPY",
                "side": "LONG",
                "notional_base": 2_000_000.0,
                "entry_price": 158.0,
                "current_spot": 160.0,
            },
            {
                "trade_id": "T2",
                "pair": "USDSGD",
                "side": "SHORT",
                "notional_base": 2_500_000.0,
                "entry_price": 1.27,
                "current_spot": 1.26,
            },
            {
                "trade_id": "T3",
                "pair": "AUDUSD",
                "side": "LONG",
                "notional_base": 2_000_000.0,
                "entry_price": 0.70,
                "current_spot": 0.71,
            },
        ]
    )


def make_multi_trade_same_pair_snapshot():
    """Portfolio with offsetting trades in the same USDJPY risk factor."""
    base = make_pnl_snapshot()
    extra = pd.DataFrame(
        [
            {
                "trade_id": "T4",
                "pair": "USDJPY",
                "side": "SHORT",
                "notional_base": 750_000.0,
                "entry_price": 159.0,
                "current_spot": 160.0,
            },
            {
                "trade_id": "T5",
                "pair": "USDJPY",
                "side": "LONG",
                "notional_base": 1_250_000.0,
                "entry_price": 157.5,
                "current_spot": 160.0,
            },
        ]
    )
    return pd.concat([base, extra], ignore_index=True)


class TestFXDashboardCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.returns = make_returns()
        cls.pnl = make_pnl_snapshot()

    # ------------------------------------------------------------------
    # P&L formulas
    # ------------------------------------------------------------------

    def test_usd_quote_pnl_formula(self):
        # AUDUSD: V(S) = d * N * (S - K)
        pnl = calculate_mtm_usd(
            pair="AUDUSD",
            side="LONG",
            notional_base=2_000_000.0,
            entry_price=0.70,
            current_price=0.71,
        )
        self.assertAlmostEqual(pnl, 20_000.0, places=8)

        short_pnl = calculate_mtm_usd(
            pair="AUDUSD",
            side="SHORT",
            notional_base=2_000_000.0,
            entry_price=0.70,
            current_price=0.71,
        )
        self.assertAlmostEqual(short_pnl, -20_000.0, places=8)

    def test_usd_base_pnl_formula(self):
        # USDJPY: V(S) = d * N * (1 - K / S)
        expected = 2_000_000.0 * (1.0 - 158.0 / 160.0)

        pnl = calculate_mtm_usd(
            pair="USDJPY",
            side="LONG",
            notional_base=2_000_000.0,
            entry_price=158.0,
            current_price=160.0,
        )
        self.assertAlmostEqual(pnl, expected, places=8)

        short_pnl = calculate_mtm_usd(
            pair="USDJPY",
            side="SHORT",
            notional_base=2_000_000.0,
            entry_price=158.0,
            current_price=160.0,
        )
        self.assertAlmostEqual(short_pnl, -expected, places=8)

    # ------------------------------------------------------------------
    # Risk exposures
    # ------------------------------------------------------------------

    def test_trade_risk_exposure_formulas(self):
        audusd = calculate_trade_risk_exposure(
            pair="AUDUSD",
            side="LONG",
            notional_base=2_000_000.0,
            entry_price=0.70,
            current_spot=0.71,
        )
        self.assertAlmostEqual(audusd, 2_000_000.0 * 0.71, places=8)

        usdjpy = calculate_trade_risk_exposure(
            pair="USDJPY",
            side="LONG",
            notional_base=2_000_000.0,
            entry_price=158.0,
            current_spot=160.0,
        )
        self.assertAlmostEqual(
            usdjpy,
            2_000_000.0 * 158.0 / 160.0,
            places=8,
        )

    # ------------------------------------------------------------------
    # VaR reconciliation and new metrics
    # ------------------------------------------------------------------

    def test_component_var_reconciles_to_portfolio_var(self):
        risk, summary, _, _ = calculate_var(
            self.pnl,
            self.returns.tail(252),
            confidence_level=0.95,
        )

        self.assertAlmostEqual(
            risk["component_var_usd"].sum(),
            summary["portfolio_var_usd"],
            places=7,
        )
        self.assertAlmostEqual(summary["reconciliation"], 0.0, places=7)

    def test_diversification_benefit_identity(self):
        risk, summary, _, _ = calculate_var(
            self.pnl,
            self.returns.tail(252),
            confidence_level=0.95,
        )

        expected_usd = (
            risk["position_var_usd"].sum()
            - summary["portfolio_var_usd"]
        )

        self.assertAlmostEqual(
            summary["diversification_benefit_usd"],
            expected_usd,
            places=7,
        )

        expected_pct = (
            expected_usd
            / risk["position_var_usd"].sum()
            * 100.0
        )
        self.assertAlmostEqual(
            summary["diversification_benefit_pct"],
            expected_pct,
            places=7,
        )

    def test_netting_and_cross_pair_diversification_decomposition(self):
        pnl = make_multi_trade_same_pair_snapshot()
        risk, summary, _, _ = calculate_var(
            pnl,
            self.returns.tail(252),
            confidence_level=0.95,
        )

        trade_var_sum = risk["position_var_usd"].sum()
        pair_exposure = risk.groupby("pair")["signed_usd_exposure"].sum()
        pair_vol = risk.groupby("pair")["daily_volatility"].first()
        z = NormalDist().inv_cdf(0.95)
        pair_var_sum = (z * pair_exposure.abs() * pair_vol).sum()

        self.assertAlmostEqual(
            summary["pair_standalone_var_total_usd"],
            pair_var_sum,
            places=7,
        )
        self.assertAlmostEqual(
            summary["netting_benefit_usd"],
            trade_var_sum - pair_var_sum,
            places=7,
        )
        self.assertGreater(summary["netting_benefit_usd"], 0.0)
        self.assertAlmostEqual(
            summary["cross_pair_diversification_benefit_usd"],
            pair_var_sum - summary["portfolio_var_usd"],
            places=7,
        )
        self.assertAlmostEqual(
            summary["netting_benefit_usd"]
            + summary["cross_pair_diversification_benefit_usd"],
            summary["diversification_benefit_usd"],
            places=7,
        )

    def test_incremental_var_equals_remove_position_recalculation(self):
        risk, summary, _, _ = calculate_var(
            self.pnl,
            self.returns.tail(252),
            confidence_level=0.95,
        )

        full_var = summary["portfolio_var_usd"]

        for _, row in risk.iterrows():
            reduced_pnl = self.pnl[
                self.pnl["trade_id"] != row["trade_id"]
            ].copy()

            _, reduced_summary, _, _ = calculate_var(
                reduced_pnl,
                self.returns.tail(252),
                confidence_level=0.95,
            )

            expected_incremental = (
                full_var
                - reduced_summary["portfolio_var_usd"]
            )

            self.assertAlmostEqual(
                row["incremental_var_usd"],
                expected_incremental,
                places=7,
            )

    # ------------------------------------------------------------------
    # Confidence / lookback behaviour
    # ------------------------------------------------------------------

    def test_higher_confidence_increases_var_and_es_but_not_vol(self):
        _, s90, _, _ = calculate_var(
            self.pnl,
            self.returns.tail(252),
            confidence_level=0.90,
        )
        _, s95, _, _ = calculate_var(
            self.pnl,
            self.returns.tail(252),
            confidence_level=0.95,
        )
        _, s99, _, _ = calculate_var(
            self.pnl,
            self.returns.tail(252),
            confidence_level=0.99,
        )

        self.assertAlmostEqual(
            s90["portfolio_volatility_usd"],
            s95["portfolio_volatility_usd"],
            places=10,
        )
        self.assertAlmostEqual(
            s95["portfolio_volatility_usd"],
            s99["portfolio_volatility_usd"],
            places=10,
        )

        self.assertLess(s90["portfolio_var_usd"], s95["portfolio_var_usd"])
        self.assertLess(s95["portfolio_var_usd"], s99["portfolio_var_usd"])

        es90 = calculate_normal_expected_shortfall(
            s90["portfolio_volatility_usd"], 0.90
        )
        es95 = calculate_normal_expected_shortfall(
            s95["portfolio_volatility_usd"], 0.95
        )
        es99 = calculate_normal_expected_shortfall(
            s99["portfolio_volatility_usd"], 0.99
        )

        self.assertLess(es90, es95)
        self.assertLess(es95, es99)
        self.assertGreater(es95, s95["portfolio_var_usd"])

    def test_diversification_pct_is_confidence_invariant(self):
        _, s90, _, _ = calculate_var(
            self.pnl,
            self.returns.tail(252),
            confidence_level=0.90,
        )
        _, s95, _, _ = calculate_var(
            self.pnl,
            self.returns.tail(252),
            confidence_level=0.95,
        )
        _, s99, _, _ = calculate_var(
            self.pnl,
            self.returns.tail(252),
            confidence_level=0.99,
        )

        self.assertAlmostEqual(
            s90["diversification_benefit_pct"],
            s95["diversification_benefit_pct"],
            places=10,
        )
        self.assertAlmostEqual(
            s95["diversification_benefit_pct"],
            s99["diversification_benefit_pct"],
            places=10,
        )

        # Dollar diversification benefit should scale with z-score.
        self.assertLess(
            s90["diversification_benefit_usd"],
            s95["diversification_benefit_usd"],
        )
        self.assertLess(
            s95["diversification_benefit_usd"],
            s99["diversification_benefit_usd"],
        )

    def test_lookback_can_change_vol_and_diversification(self):
        _, s126, _, c126 = calculate_var(
            self.pnl,
            self.returns.tail(126),
            confidence_level=0.95,
        )
        _, s252, _, c252 = calculate_var(
            self.pnl,
            self.returns.tail(252),
            confidence_level=0.95,
        )
        _, s504, _, c504 = calculate_var(
            self.pnl,
            self.returns.tail(504),
            confidence_level=0.95,
        )

        # The synthetic series deliberately has changing volatility regimes.
        self.assertFalse(
            math.isclose(
                s126["portfolio_volatility_usd"],
                s504["portfolio_volatility_usd"],
                rel_tol=1e-6,
            )
        )
        self.assertFalse(c126.equals(c504))

        # VaR remains internally consistent at every lookback.
        z95 = NormalDist().inv_cdf(0.95)
        for summary in (s126, s252, s504):
            self.assertAlmostEqual(
                summary["portfolio_var_usd"],
                z95 * summary["portfolio_volatility_usd"],
                places=7,
            )

    # ------------------------------------------------------------------
    # Exact stress revaluation
    # ------------------------------------------------------------------

    def test_stress_uses_exact_revaluation_and_is_slightly_asymmetric(self):
        one_trade = pd.DataFrame(
            [
                {
                    "trade_id": "J1",
                    "pair": "USDJPY",
                    "side": "LONG",
                    "notional_base": 2_000_000.0,
                    "entry_price": 158.0,
                    "current_spot": 160.0,
                    "valuation_status": "OK",
                }
            ]
        )

        stress = calculate_stress_scenarios(
            one_trade,
            expected_trade_ids=["J1"],
        ).set_index("scenario")

        plus_1 = stress.loc[
            "USD strengthens 1%", "stress_pnl_usd"
        ]
        minus_1 = stress.loc[
            "USD weakens 1%", "stress_pnl_usd"
        ]

        current = calculate_mtm_usd(
            "USDJPY", "LONG", 2_000_000.0, 158.0, 160.0
        )
        exact_plus = calculate_mtm_usd(
            "USDJPY", "LONG", 2_000_000.0, 158.0, 160.0 * 1.01
        ) - current
        exact_minus = calculate_mtm_usd(
            "USDJPY", "LONG", 2_000_000.0, 158.0, 160.0 * 0.99
        ) - current

        self.assertAlmostEqual(plus_1, exact_plus, places=8)
        self.assertAlmostEqual(minus_1, exact_minus, places=8)

        # Nonlinear USD-base valuation means equal +/- shocks need not
        # produce exactly equal P&L magnitudes.
        self.assertFalse(
            math.isclose(abs(plus_1), abs(minus_1), rel_tol=1e-10)
        )

    def test_stress_rejects_partial_book(self):
        snapshot = self.pnl.copy()
        snapshot["valuation_status"] = "OK"
        snapshot.loc[
            snapshot["trade_id"] == "T2",
            "valuation_status",
        ] = "UNAVAILABLE"

        stress = calculate_stress_scenarios(
            snapshot,
            expected_trade_ids=["T1", "T2", "T3"],
        )

        self.assertTrue(stress.empty)

    # ------------------------------------------------------------------
    # Market-data sanity check
    # ------------------------------------------------------------------

    def test_bad_tick_filter_accepts_normal_move_and_rejects_extreme_move(self):
        valuation_timestamp = pd.Timestamp(
            "2026-10-02 10:00:00", tz="UTC"
        )
        quote_timestamp = pd.Timestamp(
            "2026-10-02 09:55:00", tz="UTC"
        )

        # Normal move: should not raise.
        validate_current_quote(
            pair="USDJPY",
            current_price=161.0,
            previous_close=160.0,
            quote_timestamp=quote_timestamp,
            market_source="intraday_1m",
            valuation_timestamp=valuation_timestamp,
            previous_close_date=pd.Timestamp("2026-10-01"),
        )

        # >20% move: should be rejected as a possible bad tick.
        with self.assertRaises(ValueError):
            validate_current_quote(
                pair="USDJPY",
                current_price=200.0,
                previous_close=160.0,
                quote_timestamp=quote_timestamp,
                market_source="intraday_1m",
                valuation_timestamp=valuation_timestamp,
                previous_close_date=pd.Timestamp("2026-10-01"),
            )

    # ------------------------------------------------------------------
    # Multiple positions in the same currency pair
    # ------------------------------------------------------------------

    def test_multiple_positions_same_pair_are_netted_at_factor_level(self):
        duplicate_pair_book = pd.DataFrame(
            [
                {
                    "trade_id": "J_LONG",
                    "pair": "USDJPY",
                    "side": "LONG",
                    "notional_base": 2_000_000.0,
                    "entry_price": 158.0,
                    "current_spot": 160.0,
                },
                {
                    "trade_id": "J_SHORT",
                    "pair": "USDJPY",
                    "side": "SHORT",
                    "notional_base": 2_000_000.0,
                    "entry_price": 158.0,
                    "current_spot": 160.0,
                },
                {
                    "trade_id": "A_LONG",
                    "pair": "AUDUSD",
                    "side": "LONG",
                    "notional_base": 2_000_000.0,
                    "entry_price": 0.70,
                    "current_spot": 0.71,
                },
            ]
        )

        risk, summary, _, _ = calculate_var(
            duplicate_pair_book,
            self.returns.tail(252),
            confidence_level=0.95,
        )

        # Equal/opposite USDJPY sensitivities should net at the factor level.
        usdjpy_net = risk.loc[
            risk["pair"] == "USDJPY",
            "signed_usd_exposure",
        ].sum()
        self.assertAlmostEqual(usdjpy_net, 0.0, places=8)

        # Component VaR still reconciles after factor-level netting.
        self.assertAlmostEqual(
            risk["component_var_usd"].sum(),
            summary["portfolio_var_usd"],
            places=7,
        )

        # Current diversification benefit includes within-pair netting.
        z = summary["z_score"]
        pair_exposure = (
            risk.groupby("pair")["signed_usd_exposure"].sum()
        )
        vols = self.returns.tail(252).std(ddof=1)
        pair_net_standalone_total = sum(
            z * abs(pair_exposure[pair]) * vols[pair]
            for pair in pair_exposure.index
        )

        within_pair_netting_benefit = (
            summary["standalone_var_total_usd"]
            - pair_net_standalone_total
        )

        self.assertGreater(within_pair_netting_benefit, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
