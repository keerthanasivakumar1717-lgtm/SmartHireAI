"""
SmartHireAI - Centralized Configuration Module
Provides typed configuration, directory paths, and environment variable loaders.
"""

from pathlib import Path
import os
from typing import List
from dotenv import load_dotenv

# Base Project Root Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Attempt to load .env from project root
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE)
else:
    load_dotenv()  # Fallback to standard environment loading

# API Keys (Never hardcode secrets!)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Model Settings
DEFAULT_LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# Directory Paths
DATA_DIR = BASE_DIR / "data"
JOBS_DATA_DIR = DATA_DIR / "jobs"
RESUMES_DATA_DIR = DATA_DIR / "resumes"
CAREER_NOTES_DIR = DATA_DIR / "career_notes"

VECTORSTORE_DIR = BASE_DIR / "vectorstore"
FAISS_INDEX_FILE = VECTORSTORE_DIR / "job_index.faiss"
FAISS_METADATA_FILE = VECTORSTORE_DIR / "job_metadata.pkl"

REPORTS_DIR = BASE_DIR / "reports"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Default File Paths
DEFAULT_JOBS_CSV = JOBS_DATA_DIR / "sample_jobs.csv"

# Search Parameters
DEFAULT_TOP_K = int(os.getenv("TOP_K_JOBS", "5"))
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

# Required Schema for Jobs Dataset
REQUIRED_JOB_COLUMNS: List[str] = ["title", "company", "skills", "job_description"]
COLUMN_ALIASES = {
    "job_title": "title",
    "role": "title",
    "position": "title",
    "organization": "company",
    "employer": "company",
    "key_skills": "skills",
    "skill_requirements": "skills",
    "description": "job_description",
    "details": "job_description",
    "summary": "job_description",
}

# Ensure directories exist
for path in [
    DATA_DIR,
    JOBS_DATA_DIR,
    RESUMES_DATA_DIR,
    CAREER_NOTES_DIR,
    VECTORSTORE_DIR,
    REPORTS_DIR,
    NOTEBOOKS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)


def is_gemini_configured() -> bool:
    """Check if a non-empty Gemini API key is available."""
    return bool(GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here")

