"""
Note matching: determines whether a context note justifies a flagged anomaly.

Uses rule-based filtering (route applicability, date proximity, content analysis)
to ensure only genuine cost-increasing events are accepted as justifications.
"""
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# Phrases that indicate a note does NOT explain a cost increase
NO_EFFECT_PHRASES = [
    "not part of this dataset",
    "not significantly affected",
    "no major disruptions",
    "no significant disruptions",
    "improved road conditions",
    "returned to normal",
    "without a rate change",
    "remained stable",
    "remained normal",
    "compliance costs were absorbed",
]

# Phrases that indicate a note DOES describe a cost-increasing event
COST_INCREASE_PHRASES = [
    "surcharge",
    "higher",
    "rose",
    "disrupted",
    "flooding",
    "pricier",
    "pushing up",
    "higher trip costs",
    "forcing longer",
    "detour",
]


class NoteIndex:
    """Index of context notes supporting rule-based matching and semantic search."""

    def __init__(self, notes_df):
        self.notes = notes_df.copy()
        self._build_tfidf_index()

    def _build_tfidf_index(self):
        """Build TF-IDF vectors for semantic similarity search."""
        # Combine note metadata and text for richer matching
        self.note_texts = [
            f"{row['applies_to']} {row['date'].strftime('%Y-%m-%d')} {row['note']}"
            for _, row in self.notes.iterrows()
        ]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.note_texts)
        logger.info(f"Built TF-IDF index over {len(self.note_texts)} notes")

    def _route_matches(self, note_applies_to, route):
        """Check if a note's applies_to field covers the given route."""
        return note_applies_to == "All Routes" or note_applies_to == route

    def _date_in_window(self, note_date, week_start, window_days=21):
        """Check if note date is within the matching window for a flagged week."""
        week_end = week_start + pd.Timedelta(days=6)
        return (note_date >= week_start - pd.Timedelta(days=window_days) and
                note_date <= week_end)

    def _explains_cost_increase(self, note_text):
        """Determine if a note describes an event that would increase costs."""
        text_lower = note_text.lower()

        if any(phrase in text_lower for phrase in NO_EFFECT_PHRASES):
            return False

        return any(phrase in text_lower for phrase in COST_INCREASE_PHRASES)

    def match(self, route, week_start, window_days=21, max_proximity_days=45):
        """
        Find the best matching note for a flagged route-week.

        Returns a dict with:
          - note_id: matched note ID (str or None)
          - justified: whether a valid justification was found (bool)
          - matched_note: the full note row (Series or None)
          - closest_note: the closest note applicable to this route within proximity (or None)
          - closest_note_id: ID of the closest note (or None)
        """
        week_start = pd.Timestamp(week_start)

        justifying_note = None
        for _, note in self.notes.iterrows():
            if not self._route_matches(note["applies_to"], route):
                continue
            if not self._date_in_window(note["date"], week_start, window_days):
                continue
            if self._explains_cost_increase(note["note"]):
                justifying_note = note
                break

        # Find closest applicable note within proximity window (for explanation text)
        applicable_notes = []
        for _, note in self.notes.iterrows():
            if not self._route_matches(note["applies_to"], route):
                continue
            days_diff = abs((note["date"] - week_start).days)
            if days_diff <= max_proximity_days:
                applicable_notes.append((note, days_diff))

        closest_note = None
        if applicable_notes:
            applicable_notes.sort(key=lambda x: x[1])
            closest_note = applicable_notes[0][0]

        if justifying_note is not None:
            return {
                "note_id": justifying_note["note_id"],
                "justified": True,
                "matched_note": justifying_note,
                "closest_note": justifying_note,
                "closest_note_id": justifying_note["note_id"],
            }

        return {
            "note_id": None,
            "justified": False,
            "matched_note": None,
            "closest_note": closest_note,
            "closest_note_id": closest_note["note_id"] if closest_note is not None else None,
        }
