"""Tests for note matching logic."""
import unittest
import pandas as pd
from src.note_matcher import NoteIndex


def _make_notes():
    """Create a small set of notes for testing."""
    data = [
        {
            "note_id": "N001",
            "date": "2025-02-24",
            "applies_to": "Chennai-Bangalore",
            "note": "Heavy flooding disrupted normal truck movement, forcing longer detours and higher trip costs.",
        },
        {
            "note_id": "N002",
            "date": "2025-01-20",
            "applies_to": "Ahmedabad-Mumbai",
            "note": "A regional festival week saw a temporary surcharge applied by transporters.",
        },
        {
            "note_id": "N004",
            "date": "2024-03-11",
            "applies_to": "All Routes",
            "note": "A new toll plaza was commissioned, but the affected routes are not part of this dataset.",
        },
        {
            "note_id": "N005",
            "date": "2024-07-29",
            "applies_to": "Mumbai-Delhi",
            "note": "Highway maintenance caused minor delays; costs were not significantly affected.",
        },
        {
            "note_id": "N006",
            "date": "2025-09-22",
            "applies_to": "All Routes",
            "note": "A logistics industry report noted demand remained stable with no major disruptions reported.",
        },
    ]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df


class TestNoteMatching(unittest.TestCase):
    def setUp(self):
        self.notes = _make_notes()
        self.index = NoteIndex(self.notes)

    def test_flooding_matches_chennai_bangalore(self):
        """N001 flooding should match Chennai-Bangalore in the right week."""
        result = self.index.match("Chennai-Bangalore", pd.Timestamp("2025-02-24"))
        self.assertTrue(result["justified"])
        self.assertEqual(result["note_id"], "N001")

    def test_festival_matches_ahmedabad_mumbai(self):
        """N002 festival surcharge should match Ahmedabad-Mumbai."""
        result = self.index.match("Ahmedabad-Mumbai", pd.Timestamp("2025-01-20"))
        self.assertTrue(result["justified"])
        self.assertEqual(result["note_id"], "N002")

    def test_wrong_route_no_match(self):
        """N001 flooding should NOT match Mumbai-Pune (wrong route)."""
        result = self.index.match("Mumbai-Pune", pd.Timestamp("2025-02-24"))
        self.assertFalse(result["justified"])

    def test_wrong_date_no_match(self):
        """N002 festival should NOT match weeks far from Jan 20."""
        result = self.index.match("Ahmedabad-Mumbai", pd.Timestamp("2025-06-01"))
        self.assertFalse(result["justified"])

    def test_no_effect_note_excluded(self):
        """N004 and N005 mention 'not part of dataset' / 'not significantly affected'."""
        # N004 is All Routes, date 2024-03-11 — but says routes not in dataset
        result = self.index.match("Mumbai-Pune", pd.Timestamp("2024-03-11"))
        self.assertFalse(result["justified"])

    def test_stable_demand_not_justification(self):
        """N006 'remained stable, no major disruptions' should not justify a flag."""
        result = self.index.match("Mumbai-Pune", pd.Timestamp("2025-09-22"))
        self.assertFalse(result["justified"])

    def test_closest_note_when_nearby(self):
        """When a note is nearby for the route, closest_note is populated."""
        result = self.index.match("Mumbai-Pune", pd.Timestamp("2025-09-15"))
        self.assertFalse(result["justified"])
        self.assertIsNotNone(result["closest_note"])
        self.assertEqual(result["closest_note_id"], "N006")

    def test_no_note_when_out_of_range(self):
        """When no note is within range for the route, closest_note is None."""
        result = self.index.match("Delhi-Jaipur", pd.Timestamp("2024-11-11"))
        self.assertFalse(result["justified"])
        self.assertIsNone(result["closest_note"])
        self.assertIsNone(result["closest_note_id"])


if __name__ == "__main__":
    unittest.main()
