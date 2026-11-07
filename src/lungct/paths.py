"""Repository folders."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

DATA_RAW = REPO / "data" / "raw"              # Zenodo downloads
DATA_PROCESSED = REPO / "data" / "processed"  # slices written by scripts/prepare_data.py

RESULTS = REPO / "results"
RUNS = RESULTS / "runs"                       # one CSV row per training run, histories
TABLES = RESULTS / "tables"                   # statistics (notebooks/analysis.ipynb)
FIGURES = RESULTS / "figures"                 # article figures (notebooks/analysis.ipynb)
LOGS = RESULTS / "logs"
