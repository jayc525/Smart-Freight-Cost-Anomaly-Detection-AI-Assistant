import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

SHIPMENT_FILE = BASE_DIR / "shipment_records.csv"
NOTES_FILE = BASE_DIR / "context_notes.csv"
OUTPUT_FILE = OUTPUT_DIR / "anomaly_report.csv"
TOKEN_LOG_FILE = OUTPUT_DIR / "token_log.json"

# Cost analysis thresholds
FLAG_THRESHOLD_HISTORY = 0.15   # 15% above trailing avg
FLAG_THRESHOLD_PEERS = 0.15     # 15% above peer avg
TRAILING_WEEKS = 8

# Note matching
NOTE_WINDOW_DAYS = 21           # max days between note date and flagged week

# LLM configuration (Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE = 0.0

OUTPUT_DIR.mkdir(exist_ok=True)
