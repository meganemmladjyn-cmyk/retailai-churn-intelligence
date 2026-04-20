"""CLI script: engineer and select features, then save to data/processed/features.csv."""

import logging
import time
from pathlib import Path

import pandas as pd

from src.features.engineering import create_features, select_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parents[2] / "data"
INPUT_PATH = DATA_DIR / "processed" / "customers_clean.csv"
OUTPUT_PATH = DATA_DIR / "features.csv"


def run(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> None:
    """Load, engineer, select, and save features."""
    t0 = time.time()

    raw = pd.read_csv(input_path, parse_dates=["signup_date", "last_purchase_date"])
    print(f"Loaded:              {raw.shape[0]:>7,} rows × {raw.shape[1]} columns")

    engineered = create_features(raw)
    print(f"After engineering:   {engineered.shape[0]:>7,} rows × {engineered.shape[1]} columns")

    selected_names, reduced = select_features(engineered)
    print(f"After selection:     {reduced.shape[0]:>7,} rows × {reduced.shape[1]} columns")
    print(f"\nKept features ({len(selected_names)}):")
    for name in selected_names:
        print(f"  - {name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reduced.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"Elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    run()
