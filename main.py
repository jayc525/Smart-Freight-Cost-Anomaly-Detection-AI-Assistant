"""
FreightTiger Smart Freight Cost Anomaly Detection Assistant
============================================================
Main entry point — orchestrates the full analysis pipeline:
  1. Load shipment records and context notes
  2. Compute weekly cost metrics per route
  3. Compare against trailing average and peer routes
  4. Flag anomalies and match context notes (RAG)
  5. Generate plain-English explanations
  6. Output CSV in the required format

Usage:
    python main.py                # Run analysis
    python main.py --qa           # Run analysis then start Q&A mode
"""
import time
import json
import logging
import argparse

import pandas as pd

from src.config import (
    SHIPMENT_FILE, NOTES_FILE, OUTPUT_FILE, TOKEN_LOG_FILE,
    OUTPUT_DIR, FLAG_THRESHOLD_HISTORY, FLAG_THRESHOLD_PEERS,
    TRAILING_WEEKS, NOTE_WINDOW_DAYS, GROQ_API_KEY, GROQ_MODEL,
    LLM_TEMPERATURE,
)
from src.data_loader import load_shipments, load_notes
from src.cost_analyzer import compute_weekly_costs, add_baselines, find_anomalies
from src.note_matcher import NoteIndex
from src.explainer import Explainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def format_output(anomalies):
    """Format anomalies DataFrame into the exact CSV structure required."""
    out = anomalies[["route", "week_of", "cost_per_tonne_km",
                      "vs_own_history_pct", "vs_similar_routes_pct",
                      "flagged", "matched_note_id", "reason"]].copy()

    # Format week_of as date string
    out["week_of"] = out["week_of"].dt.strftime("%Y-%m-%d")

    # Round cost to 2 decimal places
    out["cost_per_tonne_km"] = out["cost_per_tonne_km"].round(2)

    # Format percentage columns as human-readable strings
    out["vs_own_history"] = out["vs_own_history_pct"].apply(
        lambda x: f"{x:+.1%} vs this route's past average"
        if pd.notna(x) else "N/A (insufficient history)"
    )
    out["vs_similar_routes"] = out["vs_similar_routes_pct"].apply(
        lambda x: f"{x:+.1%} vs similar-length routes this week"
        if pd.notna(x) else "N/A (no peer data)"
    )

    # Select and order final columns to match sample_output_format_v2.csv
    return out[["route", "week_of", "cost_per_tonne_km",
                "vs_own_history", "vs_similar_routes",
                "flagged", "matched_note_id", "reason"]]


def run_pipeline():
    """Execute the full analysis pipeline."""
    start_time = time.time()

    logger.info("Loading data")
    shipments = load_shipments(SHIPMENT_FILE)
    notes = load_notes(NOTES_FILE)

    logger.info("Computing weekly cost metrics")
    weekly = compute_weekly_costs(shipments)

    logger.info("Computing baselines (trailing avg + peer avg)")
    weekly = add_baselines(weekly, trailing_weeks=TRAILING_WEEKS)

    logger.info("Identifying cost anomalies")
    anomalies = find_anomalies(
        weekly,
        history_threshold=FLAG_THRESHOLD_HISTORY,
        peer_threshold=FLAG_THRESHOLD_PEERS,
    )

    if len(anomalies) == 0:
        logger.info("No anomalies found. Output will be empty.")
        output = pd.DataFrame(columns=[
            "route", "week_of", "cost_per_tonne_km",
            "vs_own_history", "vs_similar_routes",
            "flagged", "matched_note_id", "reason",
        ])
        output.to_csv(OUTPUT_FILE, index=False)
        return weekly, output

    logger.info("Matching notes and generating explanations")
    note_index = NoteIndex(notes)
    explainer = Explainer(
        notes,
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=LLM_TEMPERATURE,
    )

    matched_note_ids = []
    flags = []
    reasons = []

    for i, (_, row) in enumerate(anomalies.iterrows()):
        match_result = note_index.match(
            row["route"], row["week_of"], window_days=NOTE_WINDOW_DAYS
        )

        explanation, _ = explainer.explain(
            route=row["route"],
            week_of=row["week_of"],
            cost=row["cost_per_tonne_km"],
            vs_history_pct=row["vs_own_history_pct"],
            vs_peers_pct=row["vs_similar_routes_pct"],
            match_result=match_result,
        )

        matched_note_ids.append(match_result["note_id"] or "")
        flags.append("No (justified)" if match_result["justified"] else "Yes")
        reasons.append(explanation)

        if (i + 1) % 25 == 0:
            logger.info(f"  Processed {i + 1}/{len(anomalies)} anomalies")

    anomalies["matched_note_id"] = matched_note_ids
    anomalies["flagged"] = flags
    anomalies["reason"] = reasons

    logger.info("Formatting and saving output")
    output = format_output(anomalies)
    OUTPUT_DIR.mkdir(exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Output saved to {OUTPUT_FILE}")

    elapsed = time.time() - start_time
    token_log = explainer.get_token_totals()
    token_log["elapsed_seconds"] = round(elapsed, 1)
    token_log["model"] = GROQ_MODEL if GROQ_API_KEY else "template (no LLM)"
    token_log["temperature"] = LLM_TEMPERATURE

    input_cost = token_log["input_tokens"] * 0.59 / 1_000_000
    output_cost = token_log["output_tokens"] * 0.79 / 1_000_000
    token_log["estimated_cost_usd"] = round(input_cost + output_cost, 6)

    with open(TOKEN_LOG_FILE, "w") as f:
        json.dump(token_log, f, indent=2)
    logger.info(f"Token log saved to {TOKEN_LOG_FILE}")

    justified = flags.count("No (justified)")
    unexplained = flags.count("Yes")
    logger.info(
        f"Pipeline complete in {elapsed:.1f}s — "
        f"{len(anomalies)} anomalies: {justified} justified, {unexplained} unexplained"
    )
    logger.info(
        f"LLM usage: {token_log['llm_calls']} calls, "
        f"{token_log['input_tokens']} input tokens, "
        f"{token_log['output_tokens']} output tokens, "
        f"~${token_log['estimated_cost_usd']:.4f}"
    )

    return weekly, output


def main():
    parser = argparse.ArgumentParser(
        description="FreightTiger Smart Freight Cost Anomaly Detection"
    )
    parser.add_argument("--qa", action="store_true",
                        help="Start interactive Q&A after analysis")
    args = parser.parse_args()

    weekly, output = run_pipeline()

    print(f"\n{'='*60}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Rows:   {len(output)}")
    print(f"{'='*60}")

    if args.qa:
        from src.qa_assistant import QAAssistant
        # Reload anomalies with flag info for Q&A context
        anomalies_df = pd.read_csv(OUTPUT_FILE, parse_dates=["week_of"])
        notes = load_notes(NOTES_FILE)
        weekly_for_qa = weekly.copy()

        groq_client = None
        if GROQ_API_KEY:
            try:
                from groq import Groq
                groq_client = Groq(api_key=GROQ_API_KEY)
            except Exception:
                pass

        qa = QAAssistant(weekly_for_qa, anomalies_df, notes,
                         client=groq_client, model_name=GROQ_MODEL)
        qa.interactive()


if __name__ == "__main__":
    main()
