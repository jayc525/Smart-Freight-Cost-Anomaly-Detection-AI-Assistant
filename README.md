# FreightTiger — Smart Freight Cost Anomaly Detection Assistant

> **Software Engineering Intern (AI) Case Study Round**  
> An end-to-end intelligent pipeline that monitors shipping costs across routes over time, flags abnormal cost surges against historical baselines and peer routes, performs grounded RAG retrieval on real-world context notes, and generates plain-English, hallucination-free explanations.

---

## 1. System Overview & Problem Statement

In freight logistics, shipping costs depend on cargo weight (tonnes) and travel distance (km). Over time, costs can surge due to genuine market disruptions (diesel price hikes, highway flooding, festival surcharges) or unexplained operational inefficiencies.

This system solves this challenge by:
1. **Aggregating shipments** into weekly cost metrics per route (`cost_per_tonne_km`).
2. **Computing dual comparison baselines**:
   - **Own history**: Trailing 8-week rolling average (strictly prior weeks, no look-ahead).
   - **Similar routes**: Same-week average of other routes within the same `route_type` bucket (excluding the route itself).
3. **Detecting cost anomalies**: Identifies routes where costs rise significantly (>15%) above historical or peer baselines.
4. **Retrieving real-world context (RAG)**: Searches `context_notes.csv` with strict route and temporal constraints to verify whether a genuine justification exists.
5. **Generating grounded explanations**: Produces plain-English reasons for each flagged anomaly without hallucination.
6. **Formatting output**: Emits CSV matching `sample_output_format_v2.csv` with exact schema, types, and rounding.
7. **Interactive Q&A (Stretch Goal)**: Allows users to query anomalies and cost patterns in natural language.

---

## 2. Pipeline Architecture

```
shipment_records.csv ──► [ Data Loader ]
                              │
                              ▼
                      [ Cost Analyzer ] ◄── Group by Route + Route Type + ISO Week (Mon-Sun)
                              │           ├── Weighted cost_per_tonne_km
                              │           ├── Trailing 8-week history (prior weeks only)
                              │           └── Peer route average (same route_type, excluding self)
                              ▼
                      [ Anomaly Detector ] ──► Filter: vs_history > 15% OR vs_peers > 15%
                              │
context_notes.csv ──────► [ Note Matcher ] ◄── RAG: Route matching + Temporal proximity + Semantic search
                              │                └── Filters out non-disruption notes
                              ▼
                      [ Explainer Engine ] ──► Grounded reasons (LLM via Groq / Deterministic template)
                              │
                              ▼
                      [ Output Generator ] ──► output/anomaly_report.csv
                              │
                              ▼
                      [ Eval Harness ] ──► 5 Automated Checks (Schema, Citations, Anti-Hallucination, Quality, Reproducibility)
```

---

## 3. Mathematical Definitions & Edge Cases

### A. Cost Metric
$$\text{Cost per Tonne-Km} = \frac{\sum \text{freight\_cost\_inr}}{\sum (\text{quantity\_tonnes} \times \text{distance\_km})}$$
*Weighted average aggregation* gives every tonne-km equal weight, preventing small high-cost parcels from distorting the route metric.

### B. Week Alignment
- Weeks strictly run **Monday to Sunday**.
- `week_of` is mapped to that week's Monday date (`date - Timedelta(days=date.weekday())`).

### C. vs. Own History (Trailing 8-Week Baseline)
- Rolling average of the route's `cost_per_tonne_km` over the prior 8 weeks.
- **Zero look-ahead**: Only weeks strictly before the current week are used.
- **<8 Weeks History Handling**: If fewer than 8 prior weeks exist, all available prior weeks are used. The first week has zero prior weeks, returning `NaN` / `N/A (insufficient history)`. No artificial extrapolation or padding is performed.

### D. vs. Similar Routes (Peer Baseline)
- Average `cost_per_tonne_km` during the **same week** across all other routes sharing the same `route_type` (`Short`, `Medium`, or `Long`).
- **Self-exclusion**: The route itself is strictly excluded from its own peer average:
$$\text{Peer Avg} = \frac{\sum_{\text{peers}} \text{cost} - \text{cost}_{\text{self}}}{N_{\text{peers}} - 1}$$

### E. Flagging Thresholds
- A route-week is flagged if:
  - `vs_own_history_pct > +15%`, OR
  - `vs_similar_routes_pct > +15%`

---

## 4. Guardrails & Anti-Hallucination

The assignment brief emphasizes: *"A working demo isn't the bar. We will probe whether your assistant can be trusted, not just whether it runs."*

### Anti-Hallucination Principles Implemented:
1. **Strict Route Applicability**: A note applying to `Ahmedabad-Mumbai` (e.g. N002) will **never** be cited to justify or explain a rise on `Mumbai-Pune` or `Delhi-Jaipur`.
2. **Strict Date Windows**:
   - Notes with explicit event dates (e.g. N001: *"from Feb 24 to Mar 8... roads passable by Mar 9"*) only justify weeks within that window.
   - Week 2025-03-17 on Chennai-Bangalore is not falsely attributed to the flood because Note N009 explicitly confirms normal conditions returned on March 17.
3. **Exclusion of Non-Disruptive Notes**:
   - Notes that say costs were not affected (N005), routes were not in the dataset (N004), demand was stable (N006), road conditions improved (N007), or costs were absorbed without a rate change (N010) are **never used as justifications**.
4. **Honest "Unexplained" Fallbacks**:
   - If no note is in the date range: `"No matching note found for this route or date range. Cost rise looks unexplained and worth a human review."` (matches benchmark row 1).
   - If a nearby note exists but doesn't explain the rise: cites the closest note and explains why it does not justify the rise (matches benchmark row 3).
   - `matched_note_id` is left strictly blank (`""`) for all unexplained rows.

---

## 5. Verification & Benchmark Alignment

### A. Exact Match Against Company Sample Output (`sample_output_format_v2.csv`)
| Route | Week Of | Cost | vs History | vs Similar Routes | Flagged | Matched Note ID | Reason |
|---|---|---|---|---|---|---|---|
| **Delhi-Jaipur** | 2024-11-11 | 4.17 | +35.5% | +21.0% | Yes | *(blank)* | Exact match: No note found for route/date |
| **Ahmedabad-Mumbai** | 2025-01-20 | 3.29 | +29.5% | +22.5% | No (justified) | N002 | Exact match: Festival surcharge cited |
| **Mumbai-Pune** | 2025-09-15 | 3.98 | +9.2% | +23.6% | Yes | *(blank)* | Exact match: Closest note N006 cited as non-justifying |

### B. Automated Test Suite (20 Unit Tests)
- `tests/test_cost_analyzer.py`: 12 tests verifying Monday alignment, weighted aggregation, zero look-ahead rolling history, peer route grouping, self-exclusion, and anomaly thresholding.
- `tests/test_note_matcher.py`: 8 tests verifying note justification, rejection of wrong routes/dates, exclusion of non-cost-increasing notes, and date proximity limits.
- **Pass rate**: 20/20 (100% passing).

### C. Evaluation Harness (`eval/eval_harness.py`)
Run with `python -m eval.eval_harness --reproducibility 3`:
- **[PASS] Format check**: 8 required columns, correct names, order, and rounding.
- **[PASS] Note citation check**: All cited notes exist in `context_notes.csv` and apply to the route.
- **[PASS] Hallucination check**: No "No (justified)" row lacks a note; no non-justifying note used as justification; no unexplained row has a note ID.
- **[PASS] Data quality check**: Positive costs, correctly formatted percentage strings, valid enum flags.
- **[PASS] Benchmark sample match**: Verified against `sample_output_format_v2.csv`.
- **[PASS] Reproducibility check**: 3 untouched sequential runs produce bit-for-bit identical outputs.
- **Total issues found**: 0.

---

## 6. Token Usage & Execution Cost

| Metric | Deterministic Template Mode | Groq LLM Mode |
|---|---|---|
| **Model** | Built-in deterministic template engine | Configurable via `GROQ_MODEL` env var |
| **Temperature** | Deterministic (N/A) | `0.0` (deterministic) |
| **LLM Calls** | 0 | 27 (one per anomaly) |
| **Input Tokens** | 0 | ~5,400 tokens (~200 per prompt) |
| **Output Tokens** | 0 | ~1,350 tokens (~50 per explanation) |
| **Estimated Run Cost** | **$0.0000** | **< $0.01 USD** |
| **Execution Time** | ~0.15 seconds | ~4.5 seconds |

*Detailed token log is automatically saved to `output/token_log.json` after every run.*

---

## 7. How to Install and Run

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Dependencies in `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

### 1. Run the Pipeline
```bash
python main.py
```
Outputs generated:
- `output/anomaly_report.csv`: Complete formatted anomaly report matching the sample CSV contract.
- `output/token_log.json`: Run metadata, timing, token counts, and cost estimate.

### 2. Run the Evaluation Harness & Reproducibility Check
```bash
# Run all automated validation checks:
python -m eval.eval_harness

# Run 3-run untouched reproducibility check:
python -m eval.eval_harness --reproducibility 3
```

### 3. Run Unit Tests
```bash
python -m unittest discover tests
```

### 4. Interactive Q&A Assistant (Stretch Goal)
```bash
python main.py --qa
```
Example interactive session:
```text
=== FreightTiger Cost Analysis Q&A ===
Ask questions about freight cost anomalies.
Examples:
  - "Why did Chennai-Bangalore get pricier in March?"
  - "What happened on Ahmedabad-Mumbai in January?"
  - "Tell me about Delhi-Jaipur"
Type "quit" to exit.

Q: Why did Chennai-Bangalore get pricier in March?
A: Here's what I found:
   Route: Chennai-Bangalore
   Flagged anomalies (3 found):
     Week 2025-03-03: cost=3.48, flagged=No (justified)
       Reason: Matches note N001 dated 2025-02-24: heavy flooding on the highway...
     ...
```

### 5. (Optional) Enabling LLM Mode
Copy `.env.example` to `.env` and set your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```
When an API key is present, `main.py` uses the configured Groq model at `temperature=0.0`. When absent, it runs in deterministic template mode without requiring network access.

---

## 8. Interview Demo Outline

### Problem & Architecture (2 min)
- Core challenge: detecting abnormal shipping cost surges and explaining them with real-world context.
- Key difficulties: correct weighted aggregation, baselines without data leakage, anti-hallucination guardrails.
- Show modular architecture: `data_loader` → `cost_analyzer` → `note_matcher` → `explainer`.

### Core Logic & Edge Cases (3 min)
- **Aggregation**: Weighted cost (`sum(cost) / sum(tonne-km)`) vs naive mean.
- **Own History**: `add_trailing_avg` — strictly prior weeks, no look-ahead. Graceful handling when <8 weeks exist.
- **Peer History**: `add_peer_avg` — self-exclusion via `(S - x) / (N - 1)` so a route's own spike doesn't inflate its benchmark.
- **Thresholding**: 15% threshold → 27 anomalies found.

### Anti-Hallucination & Note Matching (2 min)
- Walk through `note_matcher.py`: route filtering, temporal windows, `NO_EFFECT_PHRASES` rejecting non-disruptive notes.
- Show the 3 benchmark rows matching `sample_output_format_v2.csv`.

### Verification (2 min)
- Run `python -m eval.eval_harness` and `python -m unittest discover tests` live.
- Show `output/token_log.json` for cost tracking.

### Q&A Demo (1 min)
- Launch `python main.py --qa` and ask: *"Why did Chennai-Bangalore get pricier in March?"*
