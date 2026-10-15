"""
Explanation generator: produces plain-English reasons for flagged anomalies.

Uses Groq LLM when an API key is available, falls back to deterministic
template-based explanations otherwise.
"""
import logging
import time

import pandas as pd

logger = logging.getLogger(__name__)


NOTE_SUMMARIES = {
    "N001": "heavy flooding on the Chennai-Bangalore highway disrupted normal truck movement from Feb 24 to Mar 8, forcing longer detours and higher trip costs",
    "N002": "a regional festival week drove a temporary surcharge on this corridor",
    "N003": "diesel prices rose nationwide by roughly 5-7%",
    "N004": "a new toll plaza was commissioned on a national highway stretch, but the affected routes are not part of this dataset",
    "N005": "scheduled highway maintenance work caused minor delays, but costs were not significantly affected",
    "N006": "stable demand with no major disruptions",
    "N007": "improved road conditions on the stretch after resurfacing work was completed",
    "N008": "no significant disruptions were reported and freight movement remained normal",
    "N009": "highway authorities confirmed the route returned to normal conditions after flood-related repairs were completed",
    "N010": "a new vehicle tracking mandate was introduced and compliance costs were absorbed without a rate change",
}


def _init_groq(api_key, model_name):
    """Initialize Groq client. Returns (client, True) or (None, False)."""
    if not api_key:
        return None, False
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        logger.info(f"Groq initialized: model={model_name}")
        return client, True
    except Exception as e:
        logger.warning(f"Failed to initialize Groq: {e}. Using template mode.")
        return None, False


class Explainer:
    """Generate plain-English explanations for cost anomalies."""

    def __init__(self, notes_df, api_key="", model_name="llama-3.3-70b-versatile",
                 temperature=0.0):
        self.notes = {row["note_id"]: row for _, row in notes_df.iterrows()}
        self.model_name = model_name
        self.temperature = temperature
        self.client, self.use_llm = _init_groq(api_key, model_name)
        self.token_totals = {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0}

        mode = f"LLM (Groq: {self.model_name})" if self.use_llm else "template"
        logger.info(f"Explainer using {mode} mode")

    def explain(self, route, week_of, cost, vs_history_pct, vs_peers_pct,
                match_result):
        """
        Generate an explanation for a flagged anomaly.

        Returns (explanation_text, token_usage_dict).
        """
        if self.use_llm:
            return self._explain_llm(route, week_of, cost, vs_history_pct,
                                     vs_peers_pct, match_result)
        return self._explain_template(route, week_of, cost, vs_history_pct,
                                      vs_peers_pct, match_result)

    def _explain_template(self, route, week_of, cost, vs_history_pct,
                          vs_peers_pct, match_result):
        """Deterministic template-based explanation (no LLM needed)."""
        if match_result["justified"]:
            note = match_result["matched_note"]
            note_id = match_result["note_id"]
            note_date = note["date"].strftime("%Y-%m-%d")
            summary = NOTE_SUMMARIES.get(note_id, self._summarize_note(note["note"]))
            explanation = (
                f"Matches note {note_id} dated {note_date}: "
                f"{summary}. "
                f"The cost rise has a clear explanation."
            )
        else:
            # Mention the closest note and why it doesn't apply
            closest = match_result.get("closest_note")
            if closest is not None:
                closest_id = match_result["closest_note_id"]
                closest_date = closest["date"].strftime("%Y-%m-%d")
                summary = NOTE_SUMMARIES.get(closest_id, self._summarize_note(closest["note"]))
                explanation = (
                    f"The closest note ({closest_id}, {closest_date}) "
                    f"mentions {summary} -- it does not describe "
                    f"a reason for a cost rise on this route. "
                    f"No genuine justification found; flagged for review."
                )
            else:
                explanation = (
                    "No matching note found for this route or date range. "
                    "Cost rise looks unexplained and worth a human review."
                )

        return explanation, {"input": 0, "output": 0, "calls": 0}

    def _explain_llm(self, route, week_of, cost, vs_history_pct,
                     vs_peers_pct, match_result):
        """LLM-powered explanation grounded in the matched note."""
        prompt = self._build_prompt(route, week_of, cost, vs_history_pct,
                                    vs_peers_pct, match_result)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a freight cost analysis assistant. "
                            "Generate concise, grounded, factual explanations for cost anomalies. "
                            "Do not hallucinate or speculate."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )
            text = response.choices[0].message.content.strip()

            usage = response.usage
            tokens = {
                "input": usage.prompt_tokens if usage else 0,
                "output": usage.completion_tokens if usage else 0,
                "calls": 1,
            }
            self.token_totals["input_tokens"] += tokens["input"]
            self.token_totals["output_tokens"] += tokens["output"]
            self.token_totals["llm_calls"] += 1

            # Groq inference is fast; light pacing
            time.sleep(0.1)

            return text, tokens

        except Exception as e:
            logger.warning(f"LLM call failed for {route} {week_of}: {e}. "
                          f"Falling back to template.")
            return self._explain_template(route, week_of, cost, vs_history_pct,
                                          vs_peers_pct, match_result)

    def _build_prompt(self, route, week_of, cost, vs_history_pct,
                      vs_peers_pct, match_result):
        """Build a focused prompt for the LLM explanation."""
        week_str = week_of.strftime("%Y-%m-%d") if hasattr(week_of, "strftime") else str(week_of)
        hist_str = f"{vs_history_pct:+.1%}" if pd.notna(vs_history_pct) else "N/A"
        peer_str = f"{vs_peers_pct:+.1%}" if pd.notna(vs_peers_pct) else "N/A"

        context = (
            f"Route: {route}\n"
            f"Week of: {week_str}\n"
            f"Cost per tonne-km: {cost:.2f} INR\n"
            f"vs own history: {hist_str}\n"
            f"vs similar routes: {peer_str}\n"
        )

        if match_result["justified"]:
            note = match_result["matched_note"]
            context += (
                f"\nMatching note: {match_result['note_id']}\n"
                f"Note date: {note['date'].strftime('%Y-%m-%d')}\n"
                f"Note content: {note['note']}\n"
            )
            instruction = (
                "This cost anomaly has a matching context note. Write a 1-2 sentence "
                "explanation stating the verdict (justified), citing the note ID and "
                "reflecting its content and dates. Be factual and concise."
            )
        else:
            closest = match_result.get("closest_note")
            if closest is not None:
                context += (
                    f"\nClosest note found: {match_result['closest_note_id']}\n"
                    f"Note date: {closest['date'].strftime('%Y-%m-%d')}\n"
                    f"Note content: {closest['note']}\n"
                )
            instruction = (
                "No note was found that genuinely explains this cost rise. "
                "If a nearby note was found, briefly explain why it does not apply. "
                "State that the cost rise is unexplained and flagged for review. "
                "Do NOT invent a reason. Be factual and concise, 1-2 sentences."
            )

        return f"{instruction}\n\n{context}"

    def _summarize_note(self, note_text):
        """Extract the first clause from a note as a short summary."""
        for sep in [",", "."]:
            idx = note_text.find(sep)
            if idx > 0 and idx < 120:
                return note_text[:idx].strip()
        return note_text[:120].strip()

    def get_token_totals(self):
        return self.token_totals.copy()
