import unittest
from statistics import NormalDist

import numpy as np
import pandas as pd

from src.risk import calculate_var
from src.dashboard_ui import (
    build_position_table,
    build_position_grid_records,
    build_daily_pnl_chart,
    build_component_var_chart,
)


class PairDrilldownTests(unittest.TestCase):
    def setUp(self):
        self.pnl = pd.DataFrame(
            [
                {
                    "trade_id": "FX001",
                    "trade_date": "2026-09-03",
                    "pair": "USDJPY",
                    "side": "LONG",
                    "notional_base": 2_000_000.0,
                    "entry_price": 158.0,
                    "previous_close": 157.0,
                    "current_spot": 157.5,
                    "daily_pnl_usd": 1000.0,
                    "inception_pnl_usd": 2000.0,
                    "market_source": "intraday_1m",
                    "quote_age_minutes": 1.0,
                    "valuation_status": "OK",
                },
                {
                    "trade_id": "FX002",
                    "trade_date": "2026-09-10",
                    "pair": "USDJPY",
                    "side": "SHORT",
                    "notional_base": 750_000.0,
                    "entry_price": 159.0,
                    "previous_close": 157.0,
                    "current_spot": 157.5,
                    "daily_pnl_usd": -400.0,
                    "inception_pnl_usd": 700.0,
                    "market_source": "intraday_1m",
                    "quote_age_minutes": 1.0,
                    "valuation_status": "OK",
                },
                {
                    "trade_id": "FX003",
                    "trade_date": "2026-09-17",
                    "pair": "USDJPY",
                    "side": "LONG",
                    "notional_base": 1_250_000.0,
                    "entry_price": 158.5,
                    "previous_close": 157.0,
                    "current_spot": 157.5,
                    "daily_pnl_usd": 600.0,
                    "inception_pnl_usd": -500.0,
                    "market_source": "intraday_1m",
                    "quote_age_minutes": 1.0,
                    "valuation_status": "OK",
                },
                {
                    "trade_id": "FX004",
                    "trade_date": "2026-09-08",
                    "pair": "USDSGD",
                    "side": "SHORT",
                    "notional_base": 2_500_000.0,
                    "entry_price": 1.26,
                    "previous_close": 1.27,
                    "current_spot": 1.28,
                    "daily_pnl_usd": -800.0,
                    "inception_pnl_usd": -1200.0,
                    "market_source": "intraday_1m",
                    "quote_age_minutes": 2.0,
                    "valuation_status": "OK",
                },
            ]
        )

        # Deterministic aligned return history with non-zero covariance.
        rng = np.random.default_rng(42)
        base = rng.normal(0, 0.004, 300)
        self.returns = pd.DataFrame(
            {
                "USDJPY": base,
                "USDSGD": 0.5 * base + rng.normal(0, 0.002, 300),
            }
        )

        self.risk, self.summary, _, _ = calculate_var(
            pnl=self.pnl,
            returns=self.returns,
            confidence_level=0.95,
        )

        self.position_table = build_position_table(
            self.pnl,
            self.risk,
        )

    def test_pair_net_exposure_equals_constituent_sum(self):
        usd_jpy = self.risk[
            self.risk["pair"] == "USDJPY"
        ]
        expected = usd_jpy["signed_usd_exposure"].sum()
        actual = usd_jpy["pair_signed_usd_exposure"].iloc[0]
        self.assertAlmostEqual(actual, expected, places=8)

    def test_pair_standalone_var_uses_net_exposure(self):
        usd_jpy = self.risk[
            self.risk["pair"] == "USDJPY"
        ]
        exposure = abs(usd_jpy["signed_usd_exposure"].sum())
        vol = self.returns["USDJPY"].std(ddof=1)
        expected = NormalDist().inv_cdf(0.95) * exposure * vol
        actual = usd_jpy["pair_standalone_var_usd"].iloc[0]
        self.assertAlmostEqual(actual, expected, places=8)

    def test_default_grid_is_pair_level(self):
        rows = build_position_grid_records(
            self.position_table,
            expanded_pairs=[],
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["row_type"] == "pair" for row in rows))
        self.assertEqual(rows[0]["display_name"], "▶ USDJPY")

    def test_expanding_usdjpy_reveals_three_trades(self):
        rows = build_position_grid_records(
            self.position_table,
            expanded_pairs=["USDJPY"],
        )
        usd_jpy_rows = [
            row for row in rows
            if row["pair_key"] == "USDJPY"
        ]
        self.assertEqual(len(usd_jpy_rows), 4)
        self.assertEqual(usd_jpy_rows[0]["display_name"], "▼ USDJPY")
        self.assertEqual(
            [row["display_name"] for row in usd_jpy_rows[1:]],
            ["↳ FX001", "↳ FX002", "↳ FX003"],
        )

    def test_pair_parent_pnl_is_sum_of_trades(self):
        rows = build_position_grid_records(
            self.position_table,
            expanded_pairs=[],
        )
        usd_jpy = next(row for row in rows if row["pair_key"] == "USDJPY")
        self.assertEqual(usd_jpy["daily_pnl_usd"], 1200.0)
        self.assertEqual(usd_jpy["inception_pnl_usd"], 2200.0)

    def test_component_pair_equals_trade_components(self):
        usd_jpy = self.risk[
            self.risk["pair"] == "USDJPY"
        ]
        expected = usd_jpy["component_var_usd"].sum()
        actual = usd_jpy["pair_component_var_usd"].iloc[0]
        self.assertAlmostEqual(actual, expected, places=8)

    def test_component_chart_drills_to_trades(self):
        pair_figure = build_component_var_chart(self.risk)
        pair_labels = list(pair_figure.data[0].y)
        self.assertIn("USDJPY", pair_labels)

        trade_figure = build_component_var_chart(
            self.risk,
            selected_pair="USDJPY",
        )
        trade_labels = list(trade_figure.data[0].y)
        self.assertEqual(len(trade_labels), 3)
        self.assertTrue(any("FX001" in label for label in trade_labels))

    def test_historical_pair_overlay_adds_net_and_trades(self):
        dates = pd.bdate_range("2026-09-03", periods=20)
        matrix = pd.DataFrame(
            {
                "FX001": np.arange(20, dtype=float),
                "FX002": np.arange(20, dtype=float) * -0.4,
                "FX003": np.arange(20, dtype=float) * 0.6,
                "FX004": np.arange(20, dtype=float) * 0.2,
            },
            index=dates,
        )
        portfolio = self.pnl[
            ["trade_id", "trade_date", "pair"]
        ].copy()

        figure = build_daily_pnl_chart(
            pnl_matrix=matrix,
            portfolio=portfolio,
            selected_pairs=["USDJPY"],
        )
        names = [trace.name for trace in figure.data]
        self.assertIn("USDJPY — Net", names)
        self.assertIn("USDJPY · FX001", names)
        self.assertIn("USDJPY · FX002", names)
        self.assertIn("USDJPY · FX003", names)


if __name__ == "__main__":
    unittest.main()
