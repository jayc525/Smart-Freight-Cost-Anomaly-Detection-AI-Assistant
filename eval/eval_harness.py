"""
Evaluation harness — verifies the pipeline output against known expectations.

Checks:
  1. Output format matches sample_output_format_v2.csv columns
  2. Known anomalies from sample output are present and correct
  3. Note citations are valid (cited note exists and applies to the route)
  4. No hallucinated justifications (justified rows must have a matching note)
  5. Reproducibility across multiple runs (optional)

Usage:
    python -m eval.eval_harness                      # Run all checks
    python -m eval.eval_harness --reproducibility 3  # Run 3x and compare
"""
import sys
import csv
import json
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "output" / "anomaly_report.csv"
NOTES_FILE = BASE_DIR / "context_notes.csv"
SAMPLE_FILE = BASE_DIR / "sample_output_format_v2.csv"

REQUIRED_COLUMNS = [
    "route", "week_of", "cost_per_tonne_km",
    "vs_own_history", "vs_similar_routes",
    "flagged", "matched_note_id", "reason",
]


def load_csv(filepath):
    with open(filepath, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_notes_dict(filepath):
    rows = load_csv(filepath)
    return {r["note_id"]: r for r in rows if r.get("note_id")}


def check_format(output_rows):
    """Check that all required columns are present."""
    issues = []
    if not output_rows:
        issues.append("Output file is empty")
        return issues

    actual_cols = list(output_rows[0].keys())
    for col in REQUIRED_COLUMNS:
        if col not in actual_cols:
            issues.append(f"Missing column: {col}")

    if actual_cols[:len(REQUIRED_COLUMNS)] != REQUIRED_COLUMNS:
        issues.append(
            f"Column order mismatch. Expected: {REQUIRED_COLUMNS}, "
            f"got: {actual_cols[:len(REQUIRED_COLUMNS)]}"
        )

    return issues


def check_note_citations(output_rows, notes_dict):
    """Verify that every cited note exists and applies to the route."""
    issues = []
    for i, row in enumerate(output_rows):
        note_id = row.get("matched_note_id", "").strip()
        if not note_id:
            continue

        if note_id not in notes_dict:
            issues.append(f"Row {i + 1}: cites non-existent note {note_id}")
            continue

        note = notes_dict[note_id]
        applies_to = note["applies_to"]
        route = row["route"]
        if applies_to != "All Routes" and applies_to != route:
            issues.append(
                f"Row {i + 1}: note {note_id} applies to '{applies_to}', "
                f"not '{route}'"
            )

    return issues


def check_hallucination(output_rows, notes_dict):
    """Ensure 'No (justified)' rows have a valid matched_note_id."""
    issues = []
    no_effect_phrases = [
        "not part of this dataset",
        "not significantly affected",
        "no major disruptions",
        "no significant disruptions",
        "remained stable",
        "remained normal",
        "without a rate change",
    ]

    for i, row in enumerate(output_rows):
        flagged = row.get("flagged", "").strip()
        note_id = row.get("matched_note_id", "").strip()

        if flagged == "No (justified)":
            if not note_id:
                issues.append(
                    f"Row {i + 1}: 'No (justified)' but no matched_note_id"
                )
                continue

            if note_id in notes_dict:
                note_text = notes_dict[note_id]["note"].lower()
                for phrase in no_effect_phrases:
                    if phrase in note_text:
                        issues.append(
                            f"Row {i + 1}: note {note_id} says '{phrase}' "
                            f"— should not justify a cost rise"
                        )

        if flagged == "Yes" and note_id:
            issues.append(
                f"Row {i + 1}: flagged 'Yes' but has matched_note_id={note_id} — "
                f"should be 'No (justified)' or clear the note_id"
            )

    return issues


def check_data_quality(output_rows):
    """Basic data quality checks on the output."""
    issues = []
    for i, row in enumerate(output_rows):
        # cost_per_tonne_km should be a positive number
        try:
            cost = float(row["cost_per_tonne_km"])
            if cost <= 0:
                issues.append(f"Row {i + 1}: non-positive cost {cost}")
        except (ValueError, KeyError):
            issues.append(f"Row {i + 1}: invalid cost_per_tonne_km")

        # vs_own_history should contain a %
        vs_hist = row.get("vs_own_history", "")
        if "%" not in vs_hist and "N/A" not in vs_hist:
            issues.append(f"Row {i + 1}: vs_own_history missing % format")

        # flagged should be 'Yes' or 'No (justified)'
        flagged = row.get("flagged", "").strip()
        if flagged not in ("Yes", "No (justified)"):
            issues.append(f"Row {i + 1}: unexpected flagged value '{flagged}'")

        # reason should not be empty
        if not row.get("reason", "").strip():
            issues.append(f"Row {i + 1}: empty reason")

    return issues


def check_sample_match(output_rows, sample_file):
    """Verify that known benchmark rows from sample_output_format_v2.csv match."""
    issues = []
    if not sample_file.exists():
        return issues

    sample_rows = load_csv(sample_file)
    output_lookup = {(r["route"], r["week_of"]): r for r in output_rows}

    for s in sample_rows:
        key = (s["route"], s["week_of"])
        if key not in output_lookup:
            issues.append(f"Sample row {key} missing from output")
            continue
        actual = output_lookup[key]
        for col in ["cost_per_tonne_km", "vs_own_history", "vs_similar_routes", "flagged", "matched_note_id"]:
            s_val = s.get(col, "").strip()
            a_val = actual.get(col, "").strip()
            if s_val != a_val:
                issues.append(
                    f"Sample row {key} col '{col}' mismatch: expected '{s_val}', got '{a_val}'"
                )

    return issues


def check_reproducibility(n_runs=3):
    """Run the pipeline multiple times and compare outputs."""
    print(f"\n--- Reproducibility check ({n_runs} runs) ---")
    outputs = []

    for run_num in range(1, n_runs + 1):
        print(f"  Run {run_num}/{n_runs}...", end=" ", flush=True)
        result = subprocess.run(
            [sys.executable, "main.py"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"FAILED (exit code {result.returncode})")
            print(result.stderr)
            return [f"Run {run_num} failed"]

        rows = load_csv(OUTPUT_FILE)
        outputs.append(rows)
        print("OK")

    # Compare: flags, numbers, and note citations must be identical
    issues = []
    base = outputs[0]
    for run_num, other in enumerate(outputs[1:], start=2):
        if len(base) != len(other):
            issues.append(
                f"Run 1 has {len(base)} rows, run {run_num} has {len(other)}"
            )
            continue

        for i, (b, o) in enumerate(zip(base, other)):
            for col in ["route", "week_of", "cost_per_tonne_km",
                        "vs_own_history", "vs_similar_routes",
                        "flagged", "matched_note_id"]:
                if b.get(col) != o.get(col):
                    issues.append(
                        f"Run 1 vs {run_num}, row {i + 1}, col '{col}': "
                        f"'{b.get(col)}' != '{o.get(col)}'"
                    )

    return issues


def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline output")
    parser.add_argument("--reproducibility", type=int, default=0,
                        help="Number of runs for reproducibility check")
    args = parser.parse_args()

    if not OUTPUT_FILE.exists():
        print(f"Output file not found: {OUTPUT_FILE}")
        print("Run 'python main.py' first.")
        sys.exit(1)

    output_rows = load_csv(OUTPUT_FILE)
    notes_dict = load_notes_dict(NOTES_FILE)

    all_issues = defaultdict(list)

    print("=== Evaluation Harness ===\n")

    # Format check
    issues = check_format(output_rows)
    all_issues["Format"] = issues
    status = "PASS" if not issues else "FAIL"
    print(f"[{status}] Format check: {len(REQUIRED_COLUMNS)} required columns")
    for issue in issues:
        print(f"       {issue}")

    # Note citation check
    issues = check_note_citations(output_rows, notes_dict)
    all_issues["Note citations"] = issues
    status = "PASS" if not issues else "FAIL"
    print(f"[{status}] Note citation check: {len(output_rows)} rows")
    for issue in issues:
        print(f"       {issue}")

    # Hallucination check
    issues = check_hallucination(output_rows, notes_dict)
    all_issues["Hallucination"] = issues
    status = "PASS" if not issues else "FAIL"
    print(f"[{status}] Hallucination check")
    for issue in issues:
        print(f"       {issue}")

    # Data quality check
    issues = check_data_quality(output_rows)
    all_issues["Data quality"] = issues
    status = "PASS" if not issues else "FAIL"
    print(f"[{status}] Data quality check")
    for issue in issues[:5]:
        print(f"       {issue}")
    if len(issues) > 5:
        print(f"       ... and {len(issues) - 5} more")

    # Benchmark sample match check
    issues = check_sample_match(output_rows, SAMPLE_FILE)
    all_issues["Benchmark sample match"] = issues
    status = "PASS" if not issues else "FAIL"
    print(f"[{status}] Benchmark sample match (sample_output_format_v2.csv)")
    for issue in issues:
        print(f"       {issue}")

    # Reproducibility
    if args.reproducibility > 0:
        issues = check_reproducibility(args.reproducibility)
        all_issues["Reproducibility"] = issues
        status = "PASS" if not issues else "FAIL"
        print(f"[{status}] Reproducibility check")
        for issue in issues[:5]:
            print(f"       {issue}")

    # Summary
    total_issues = sum(len(v) for v in all_issues.values())
    print(f"\n--- Summary ---")
    print(f"Total rows: {len(output_rows)}")
    justified = sum(1 for r in output_rows if r.get("flagged") == "No (justified)")
    unexplained = sum(1 for r in output_rows if r.get("flagged") == "Yes")
    print(f"Justified: {justified}, Unexplained: {unexplained}")
    print(f"Issues found: {total_issues}")

    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
