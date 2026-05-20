"""
config.py — Shared application configuration and singleton instances.

Import from here instead of main.py to avoid circular imports across routers
and service modules.
"""

import os
import json
import logging

import google.generativeai as genai
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONCURRENCY_LIMIT = 8
BATCH_SIZE = 5

# ---------------------------------------------------------------------------
# AI clients
# ---------------------------------------------------------------------------
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
print(f"Gemini SDK version: {genai.__version__}")

openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Scoring rules (loaded once at startup)
# ---------------------------------------------------------------------------
with open("scoring_rules.json", "r") as f:
    scoring_rules = json.load(f)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("gemini_prompts.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("gemini_prompts")
