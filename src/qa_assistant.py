"""
Interactive Q&A assistant (stretch goal).

Lets a user ask plain-English questions about the analysis results,
e.g. "Why did Chennai-Bangalore get pricier in March?"

Uses the pre-computed weekly costs and anomaly data as context,
with optional LLM for natural language understanding.
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class QAAssistant:
    """Answer questions about freight cost analysis results."""

    def __init__(self, weekly_costs, anomalies, notes, client=None,
                 model_name="llama-3.3-70b-versatile"):
        self.weekly = weekly_costs.copy()
        self.anomalies = anomalies.copy()
        if "week_of" in self.anomalies.columns and not pd.api.types.is_datetime64_any_dtype(self.anomalies["week_of"]):
            self.anomalies["week_of"] = pd.to_datetime(self.anomalies["week_of"])
        if "week_of" in self.weekly.columns and not pd.api.types.is_datetime64_any_dtype(self.weekly["week_of"]):
            self.weekly["week_of"] = pd.to_datetime(self.weekly["week_of"])
        self.notes = notes
        self.client = client
        self.model_name = model_name
        self.token_totals = {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0}

    def answer(self, question):
        """Answer a question about freight cost analysis."""
        route, period = self._parse_question(question)

        if route is None:
            return ("I couldn't identify which route you're asking about. "
                    "Please mention a route like 'Mumbai-Delhi' or 'Chennai-Bangalore'.")

        context = self._build_context(route, period)

        if self.client:
            return self._answer_llm(question, context)
        return self._answer_keyword(question, context)

    def _parse_question(self, question):
        """Extract route name and time period from a question."""
        q_lower = question.lower()

        routes = self.weekly["route"].unique()
        found_route = None
        for route in routes:
            if route.lower() in q_lower or route.lower().replace("-", " ") in q_lower:
                found_route = route
                break

        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4,
            "jun": 6, "jul": 7, "aug": 8, "sep": 9,
            "oct": 10, "nov": 11, "dec": 12,
        }
        found_month = None
        for name, num in months.items():
            if name in q_lower:
                found_month = num
                break

        return found_route, found_month

    def _build_context(self, route, month=None):
        """Build a context string with relevant data for the route."""
        route_anomalies = self.anomalies[self.anomalies["route"] == route]
        if month:
            route_anomalies = route_anomalies[
                route_anomalies["week_of"].dt.month == month
            ]

        route_weekly = self.weekly[self.weekly["route"] == route]
        if month:
            route_weekly = route_weekly[route_weekly["week_of"].dt.month == month]

        parts = [f"Route: {route}"]

        if len(route_anomalies) > 0:
            parts.append(f"\nFlagged anomalies ({len(route_anomalies)} found):")
            for _, row in route_anomalies.iterrows():
                week = row["week_of"].strftime("%Y-%m-%d") if hasattr(row["week_of"], "strftime") else str(row["week_of"])
                cost = row["cost_per_tonne_km"]
                flag = row.get("flagged", "")
                reason = row.get("reason", "")
                parts.append(f"  Week {week}: cost={cost:.2f}, flagged={flag}")
                if reason:
                    parts.append(f"    Reason: {reason}")
        else:
            parts.append("\nNo anomalies flagged for this route in this period.")

        if len(route_weekly) > 0:
            avg_cost = route_weekly["cost_per_tonne_km"].mean()
            min_cost = route_weekly["cost_per_tonne_km"].min()
            max_cost = route_weekly["cost_per_tonne_km"].max()
            parts.append(f"\nCost summary: avg={avg_cost:.2f}, "
                        f"min={min_cost:.2f}, max={max_cost:.2f}")

        route_notes = self.notes[
            (self.notes["applies_to"] == route) |
            (self.notes["applies_to"] == "All Routes")
        ]
        if len(route_notes) > 0:
            parts.append(f"\nRelevant notes ({len(route_notes)}):")
            for _, note in route_notes.iterrows():
                parts.append(f"  {note['note_id']} ({note['date'].strftime('%Y-%m-%d')}): "
                           f"{note['note'][:150]}")

        return "\n".join(parts)

    def _answer_keyword(self, question, context):
        """Simple keyword-based answer without LLM."""
        return (
            f"Here's what I found:\n\n{context}\n\n"
            f"(For more natural answers, configure a GROQ_API_KEY.)"
        )

    def _answer_llm(self, question, context):
        """LLM-powered answer grounded in the analysis data."""
        prompt = (
            "You are a freight cost analysis assistant. Answer the user's question "
            "based ONLY on the data provided below. Do not make up information. "
            "If the data doesn't cover the question, say so.\n\n"
            f"DATA:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "Give a concise, factual answer."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a freight cost analysis assistant. Answer questions concisely and factually based only on the provided data.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            usage = response.usage
            if usage:
                self.token_totals["input_tokens"] += usage.prompt_tokens
                self.token_totals["output_tokens"] += usage.completion_tokens
            self.token_totals["llm_calls"] += 1
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return self._answer_keyword(question, context)

    def interactive(self):
        """Run interactive Q&A loop in the terminal."""
        print("\n=== FreightTiger Cost Analysis Q&A ===")
        print("Ask questions about freight cost anomalies.")
        print("Examples:")
        print('  - "Why did Chennai-Bangalore get pricier in March?"')
        print('  - "What happened on the Mumbai-Delhi route?"')
        print('  - "Show anomalies for Ahmedabad-Mumbai"')
        print('Type "quit" to exit.\n')

        while True:
            try:
                question = input("Q: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not question or question.lower() in ("quit", "exit", "q"):
                break

            answer = self.answer(question)
            print(f"\nA: {answer}\n")

    def get_token_totals(self):
        return self.token_totals.copy()
