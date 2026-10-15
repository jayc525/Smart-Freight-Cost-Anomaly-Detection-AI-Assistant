"""
Weekly cost aggregation, baseline computation, and anomaly detection.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_week_monday(date):
    """Return the Monday of the ISO week containing `date`."""
    return date - pd.Timedelta(days=date.weekday())


def compute_weekly_costs(shipments):
    """
    Aggregate shipments into weekly cost_per_tonne_km per route.

    Uses weighted average: sum(freight_cost) / sum(quantity * distance),
    which gives each tonne-km equal weight regardless of shipment size.
    """
    df = shipments.copy()
    df["week_of"] = df["shipment_date"].apply(get_week_monday)

    weekly = (
        df.groupby(["route", "route_type", "week_of"])
        .agg(
            total_cost=("freight_cost_inr", "sum"),
            total_tonne_km=("tonne_km", "sum"),
            shipment_count=("shipment_id", "count"),
        )
        .reset_index()
    )
    weekly["cost_per_tonne_km"] = weekly["total_cost"] / weekly["total_tonne_km"]
    weekly = weekly.sort_values(["route", "week_of"]).reset_index(drop=True)

    logger.info(f"Aggregated into {len(weekly)} route-week entries")
    return weekly


def add_trailing_avg(weekly, n_weeks=8):
    """
    Add trailing N-week rolling average for each route.

    Uses only weeks strictly before the current week (no look-ahead).
    If fewer than `n_weeks` prior weeks exist, uses all available.
    """
    results = []
    for route in weekly["route"].unique():
        route_data = weekly[weekly["route"] == route].sort_values("week_of")
        costs = route_data["cost_per_tonne_km"].values

        trailing_avgs = []
        weeks_used_list = []
        for i in range(len(costs)):
            if i == 0:
                trailing_avgs.append(np.nan)
                weeks_used_list.append(0)
            else:
                start = max(0, i - n_weeks)
                prior = costs[start:i]
                trailing_avgs.append(prior.mean())
                weeks_used_list.append(len(prior))

        route_data = route_data.copy()
        route_data["trailing_avg"] = trailing_avgs
        route_data["trailing_weeks_used"] = weeks_used_list
        results.append(route_data)

    weekly = pd.concat(results, ignore_index=True)
    weekly["vs_own_history_pct"] = (
        (weekly["cost_per_tonne_km"] - weekly["trailing_avg"]) / weekly["trailing_avg"]
    )
    return weekly


def add_peer_avg(weekly):
    """
    Add same-week peer average: mean cost_per_tonne_km across other routes
    sharing the same route_type, excluding the route itself.
    """
    type_week = (
        weekly.groupby(["route_type", "week_of"])
        .agg(
            type_sum=("cost_per_tonne_km", "sum"),
            type_count=("cost_per_tonne_km", "count"),
        )
        .reset_index()
    )

    weekly = weekly.merge(type_week, on=["route_type", "week_of"], how="left")

    # Exclude self from the peer average
    peer_count = weekly["type_count"] - 1
    peer_sum = weekly["type_sum"] - weekly["cost_per_tonne_km"]

    weekly["peer_avg"] = np.where(peer_count > 0, peer_sum / peer_count, np.nan)
    weekly["vs_similar_routes_pct"] = (
        (weekly["cost_per_tonne_km"] - weekly["peer_avg"]) / weekly["peer_avg"]
    )

    weekly = weekly.drop(columns=["type_sum", "type_count"])
    return weekly


def add_baselines(weekly, trailing_weeks=8):
    """Compute both baselines in one call."""
    weekly = add_trailing_avg(weekly, n_weeks=trailing_weeks)
    weekly = add_peer_avg(weekly)
    return weekly


def find_anomalies(weekly, history_threshold=0.15, peer_threshold=0.15):
    """
    Flag route-weeks where cost significantly exceeds either baseline.

    A row is included if:
      - vs_own_history_pct > history_threshold, OR
      - vs_similar_routes_pct > peer_threshold
    and at least one comparison is available (not NaN).
    """
    has_history = weekly["vs_own_history_pct"].notna()
    has_peers = weekly["vs_similar_routes_pct"].notna()

    above_history = has_history & (weekly["vs_own_history_pct"] > history_threshold)
    above_peers = has_peers & (weekly["vs_similar_routes_pct"] > peer_threshold)

    anomalies = weekly[above_history | above_peers].copy()
    anomalies = anomalies.sort_values(["route", "week_of"]).reset_index(drop=True)

    logger.info(
        f"Found {len(anomalies)} anomalies: "
        f"{above_history.sum()} above history threshold, "
        f"{above_peers.sum()} above peer threshold"
    )
    return anomalies
