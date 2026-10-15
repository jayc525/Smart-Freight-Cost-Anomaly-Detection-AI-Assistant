import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_shipments(filepath):
    """Load shipment records and add derived columns for analysis."""
    df = pd.read_csv(filepath, parse_dates=["shipment_date"])
    df["route"] = df["origin"] + "-" + df["destination"]
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]

    expected_cols = {"shipment_id", "origin", "destination", "route_type",
                     "material", "quantity_tonnes", "distance_km",
                     "freight_cost_inr", "shipment_date", "transporter"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Shipment file missing columns: {missing}")

    logger.info(f"Loaded {len(df)} shipments across {df['route'].nunique()} routes, "
                f"date range {df['shipment_date'].min().date()} to {df['shipment_date'].max().date()}")
    return df


def load_notes(filepath):
    """Load context notes that explain real-world cost changes."""
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    # Drop any empty rows
    df = df.dropna(subset=["note_id"]).reset_index(drop=True)

    logger.info(f"Loaded {len(df)} context notes")
    return df
