"""Tests for the core cost analysis logic."""
import unittest
import pandas as pd
import numpy as np
from src.cost_analyzer import (
    get_week_monday,
    compute_weekly_costs,
    add_trailing_avg,
    add_peer_avg,
    add_baselines,
    find_anomalies,
)


def _make_shipments(rows):
    """Helper to build a small shipment DataFrame for testing."""
    df = pd.DataFrame(rows, columns=[
        "shipment_id", "origin", "destination", "route_type",
        "material", "quantity_tonnes", "distance_km",
        "freight_cost_inr", "shipment_date", "transporter",
    ])
    df["shipment_date"] = pd.to_datetime(df["shipment_date"])
    df["route"] = df["origin"] + "-" + df["destination"]
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]
    return df


class TestWeekMonday(unittest.TestCase):
    def test_monday(self):
        self.assertEqual(
            get_week_monday(pd.Timestamp("2024-01-01")),  # Monday
            pd.Timestamp("2024-01-01"),
        )

    def test_wednesday(self):
        self.assertEqual(
            get_week_monday(pd.Timestamp("2024-01-03")),  # Wednesday
            pd.Timestamp("2024-01-01"),
        )

    def test_sunday(self):
        self.assertEqual(
            get_week_monday(pd.Timestamp("2024-01-07")),  # Sunday
            pd.Timestamp("2024-01-01"),
        )


class TestWeeklyCosts(unittest.TestCase):
    def setUp(self):
        # Two shipments on Mumbai-Pune in the same week
        self.shipments = _make_shipments([
            ["S1", "Mumbai", "Pune", "Short", "Steel", 10, 150, 4500,
             "2024-01-01", "T1"],
            ["S2", "Mumbai", "Pune", "Short", "Cement", 20, 150, 12000,
             "2024-01-03", "T2"],
        ])

    def test_aggregation(self):
        weekly = compute_weekly_costs(self.shipments)
        self.assertEqual(len(weekly), 1)  # Both in same week
        row = weekly.iloc[0]
        self.assertEqual(row["route"], "Mumbai-Pune")
        # cost_per_tonne_km = (4500 + 12000) / (10*150 + 20*150)
        # = 16500 / 4500 = 3.6667
        self.assertAlmostEqual(row["cost_per_tonne_km"], 16500 / 4500, places=4)

    def test_two_weeks(self):
        shipments = _make_shipments([
            ["S1", "Mumbai", "Pune", "Short", "Steel", 10, 150, 4500,
             "2024-01-01", "T1"],
            ["S2", "Mumbai", "Pune", "Short", "Steel", 10, 150, 6000,
             "2024-01-08", "T1"],  # Next week
        ])
        weekly = compute_weekly_costs(shipments)
        self.assertEqual(len(weekly), 2)


class TestTrailingAvg(unittest.TestCase):
    def test_first_week_is_nan(self):
        """First week has no history — trailing avg should be NaN."""
        shipments = _make_shipments([
            ["S1", "A", "B", "Short", "X", 10, 100, 3000, "2024-01-01", "T"],
            ["S2", "A", "B", "Short", "X", 10, 100, 4000, "2024-01-08", "T"],
        ])
        weekly = compute_weekly_costs(shipments)
        weekly = add_trailing_avg(weekly, n_weeks=8)
        self.assertTrue(np.isnan(weekly.iloc[0]["trailing_avg"]))
        self.assertFalse(np.isnan(weekly.iloc[1]["trailing_avg"]))

    def test_uses_prior_weeks_only(self):
        """Trailing avg must not include the current week (no look-ahead)."""
        rows = []
        for i in range(10):
            date = (pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=i)).strftime("%Y-%m-%d")
            cost = 3000 + i * 100  # Steadily increasing
            rows.append([f"S{i}", "A", "B", "Short", "X", 10, 100, cost, date, "T"])
        shipments = _make_shipments(rows)
        weekly = compute_weekly_costs(shipments)
        weekly = add_trailing_avg(weekly, n_weeks=8)

        # Week index 8 (9th week): should average weeks 0-7 (indices 0-7)
        row = weekly.iloc[8]
        expected_costs = [(3000 + j * 100) / (10 * 100) for j in range(8)]
        expected_avg = np.mean(expected_costs)
        self.assertAlmostEqual(row["trailing_avg"], expected_avg, places=4)

    def test_fewer_than_8_weeks(self):
        """With fewer prior weeks, use all available."""
        rows = []
        for i in range(4):
            date = (pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=i)).strftime("%Y-%m-%d")
            rows.append([f"S{i}", "A", "B", "Short", "X", 10, 100, 3000, date, "T"])
        shipments = _make_shipments(rows)
        weekly = compute_weekly_costs(shipments)
        weekly = add_trailing_avg(weekly, n_weeks=8)

        # Week 3 (4th week, index 3): should use 3 prior weeks
        self.assertEqual(weekly.iloc[3]["trailing_weeks_used"], 3)


class TestPeerAvg(unittest.TestCase):
    def test_excludes_self(self):
        """Peer average must exclude the route itself."""
        shipments = _make_shipments([
            ["S1", "A", "B", "Short", "X", 10, 100, 3000, "2024-01-01", "T"],
            ["S2", "C", "D", "Short", "X", 10, 100, 5000, "2024-01-01", "T"],
        ])
        weekly = compute_weekly_costs(shipments)
        weekly = add_peer_avg(weekly)

        ab = weekly[weekly["route"] == "A-B"].iloc[0]
        cd = weekly[weekly["route"] == "C-D"].iloc[0]
        # A-B's peer avg should be C-D's cost, and vice versa
        self.assertAlmostEqual(ab["peer_avg"], cd["cost_per_tonne_km"], places=4)
        self.assertAlmostEqual(cd["peer_avg"], ab["cost_per_tonne_km"], places=4)

    def test_different_route_types(self):
        """Routes with different route_type are not peers."""
        shipments = _make_shipments([
            ["S1", "A", "B", "Short", "X", 10, 100, 3000, "2024-01-01", "T"],
            ["S2", "C", "D", "Long", "X", 10, 100, 5000, "2024-01-01", "T"],
        ])
        weekly = compute_weekly_costs(shipments)
        weekly = add_peer_avg(weekly)

        # No peers of same type → peer_avg should be NaN
        self.assertTrue(np.isnan(weekly.iloc[0]["peer_avg"]))
        self.assertTrue(np.isnan(weekly.iloc[1]["peer_avg"]))


class TestFindAnomalies(unittest.TestCase):
    def test_flags_above_threshold(self):
        """Route-weeks above threshold should be flagged."""
        weekly = pd.DataFrame({
            "route": ["A-B", "A-B", "A-B"],
            "route_type": ["Short", "Short", "Short"],
            "week_of": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]),
            "cost_per_tonne_km": [3.0, 3.0, 5.0],
            "vs_own_history_pct": [np.nan, 0.0, 0.30],  # 30% increase
            "vs_similar_routes_pct": [0.0, 0.0, 0.10],
            "trailing_avg": [np.nan, 3.0, 3.0],
            "peer_avg": [3.0, 3.0, 3.0],
        })
        anomalies = find_anomalies(weekly, history_threshold=0.15, peer_threshold=0.15)
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies.iloc[0]["week_of"], pd.Timestamp("2024-01-15"))

    def test_no_anomalies_below_threshold(self):
        """Route-weeks below threshold should not be flagged."""
        weekly = pd.DataFrame({
            "route": ["A-B"],
            "route_type": ["Short"],
            "week_of": pd.to_datetime(["2024-01-01"]),
            "cost_per_tonne_km": [3.0],
            "vs_own_history_pct": [0.05],  # Only 5%
            "vs_similar_routes_pct": [0.05],
            "trailing_avg": [2.85],
            "peer_avg": [2.85],
        })
        anomalies = find_anomalies(weekly, history_threshold=0.15, peer_threshold=0.15)
        self.assertEqual(len(anomalies), 0)


if __name__ == "__main__":
    unittest.main()
