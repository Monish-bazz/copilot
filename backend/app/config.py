import os
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

# Snapshot time for all business logic — NEVER use datetime.now() for business rules
SNAPSHOT_TS = datetime(2026, 8, 16, 11, 0, tzinfo=ZoneInfo("Asia/Kolkata"))

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

# Database
DATABASE_URL = os.environ.get("supabase") or os.environ.get("DATABASE_URL")

# NVIDIA NIM / LLM
NVIDIA_API_KEY = os.environ.get("NVIDIA_NIM_API_KEY") or os.environ.get("NVIDIA_API_KEY")
NIM_MODEL = os.environ.get("NEMOTRON_MODEL", "meta/llama-3.1-8b-instruct")
NIM_BASE_URL = os.environ.get("NEMOTRON_BASE_URL", "https://integrate.api.nvidia.com/v1")
