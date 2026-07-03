"""Download and preprocess the freMTPL2 dataset for the real data examples.

Follows the data cleaning procedure described in Chapter 13.1 of
Wüthrich & Merz (2023) "Statistical Foundations of Actuarial Learning
and its Applications", Springer.

The raw data is downloaded from OpenML (dataset IDs 41214 and 41215),
which hosts the FreMTPL2freq and FreMTPL2sev tables from the R package
CASdatasets (version 1.0-8).

Produces two CSV files in data/:
  - freMTPL2_counts.csv   (claim frequency model: y=ClaimNb, w=Exposure)
  - freMTPL2_severity.csv (claim severity model:  y=ClaimAmount, w=ClaimNb)

Usage:
    python -m numerical_illustration.prepare_real_data
    python numerical_illustration/prepare_real_data.py

Requires: openml  (pip install openml)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENML_FREQ_ID = 41214
OPENML_SEV_ID = 41215

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"

VEHBRAND_GROUPS = {
    "B1": "B1",
    "B2": "B2",
    "B3": "B3456",
    "B4": "B3456",
    "B5": "B3456",
    "B6": "B3456",
    "B10": "B1011",
    "B11": "B1011",
    "B12": "B12",
    "B13": "B1314",
    "B14": "B1314",
}

CATEGORICAL_FEATURES = ["VehBrand", "VehGas", "Area", "Region"]

CONTINUOUS_FEATURES = [
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "log_Density",
]


def download_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download FreMTPL2freq and FreMTPL2sev from OpenML."""
    import openml

    logger.info("Downloading FreMTPL2freq (OpenML ID %d)...", OPENML_FREQ_ID)
    freq_dataset = openml.datasets.get_dataset(
        OPENML_FREQ_ID, download_data=True, download_qualities=False
    )
    freq_df, *_ = freq_dataset.get_data()

    logger.info("Downloading FreMTPL2sev (OpenML ID %d)...", OPENML_SEV_ID)
    sev_dataset = openml.datasets.get_dataset(
        OPENML_SEV_ID, download_data=True, download_qualities=False
    )
    sev_df, *_ = sev_dataset.get_data()

    logger.info(
        "Downloaded %d freq rows, %d sev rows", len(freq_df), len(sev_df)
    )
    return freq_df, sev_df


def clean_data(freq: pd.DataFrame, sev: pd.DataFrame) -> pd.DataFrame:
    """Apply the Wüthrich Ch. 13.1 cleaning procedure.

    Steps:
      1. Aggregate sev per IDpol to get corrected ClaimNb and ClaimAmount.
      2. Left-join freq onto aggregated sev.
      3. Fill NA claim info with 0 (policies without claims).
      4. Drop rows with ClaimNb > 5 (suspected data errors).
      5. Cap Exposure at 1.0.
      6. Re-level VehBrand into 7 groups.
    """
    # Step 1: aggregate severities
    sev_agg = (
        sev.groupby("IDpol")
        .agg(ClaimNb=("ClaimAmount", "count"), ClaimAmount=("ClaimAmount", "sum"))
        .reset_index()
    )
    sev_agg["IDpol"] = sev_agg["IDpol"].astype(freq["IDpol"].dtype)

    # Step 2: merge — drop original ClaimNb from freq, use corrected version
    freq = freq.drop(columns=["ClaimNb"], errors="ignore")
    df = freq.merge(sev_agg, on="IDpol", how="left")

    # Step 3: fill NAs
    df["ClaimNb"] = df["ClaimNb"].fillna(0).astype(int)
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0.0)

    n_before = len(df)
    # Step 4: drop policies with > 5 claims
    df = df[df["ClaimNb"] <= 5].copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info("Dropped %d rows with ClaimNb > 5", n_dropped)

    # Step 5: cap Exposure at 1
    df["Exposure"] = df["Exposure"].clip(upper=1.0)

    # Step 6: re-level VehBrand
    df["VehBrand"] = df["VehBrand"].astype(str).map(VEHBRAND_GROUPS).fillna("other")

    logger.info(
        "After cleaning: %d rows, %d with claims",
        len(df),
        (df["ClaimNb"] > 0).sum(),
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare model-ready features.

    - log-transform Density
    - Map Area to ordinal integer
    - Cap VehPower at 12
    - Categorical columns stored as strings in CSV;
      categorical dtype is applied at load time via the YAML config.
    """
    out = pd.DataFrame()

    # Continuous features
    out["VehPower"] = df["VehPower"].clip(upper=12).astype(int)
    out["VehAge"] = df["VehAge"].astype(float)
    out["DrivAge"] = df["DrivAge"].astype(float)
    out["BonusMalus"] = df["BonusMalus"].astype(float)
    out["log_Density"] = np.log(df["Density"].astype(float))

    # Categorical features (stored as strings; config marks them categorical)
    area_map = {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5", "F": "6"}
    out["Area"] = df["Area"].astype(str).map(area_map).fillna(df["Area"].astype(str))
    out["VehBrand"] = df["VehBrand"].astype(str)
    out["VehGas"] = df["VehGas"].astype(str)
    out["Region"] = df["Region"].astype(str)

    # Targets and weights
    out["ClaimNb"] = df["ClaimNb"]
    out["ClaimAmount"] = df["ClaimAmount"]
    out["Exposure"] = df["Exposure"]

    return out


def write_counts_csv(df: pd.DataFrame, path: Path) -> None:
    """Write the claim count dataset (all policies)."""
    features = CATEGORICAL_FEATURES + CONTINUOUS_FEATURES
    out = df[features].copy()
    out["y"] = df["ClaimNb"]
    out["w"] = df["Exposure"]
    out.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(out))


def write_severity_csv(df: pd.DataFrame, path: Path) -> None:
    """Write the claim severity dataset (policies with claims only)."""
    has_claims = df["ClaimNb"] > 0
    sub = df[has_claims].copy()

    features = CATEGORICAL_FEATURES + CONTINUOUS_FEATURES
    out = sub[features].copy()
    out["y"] = sub["ClaimAmount"]
    out["w"] = sub["ClaimNb"]
    out.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(out))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    freq, sev = download_data()
    df = clean_data(freq, sev)
    df = engineer_features(df)

    write_counts_csv(df, OUTPUT_DIR / "freMTPL2_counts.csv")
    write_severity_csv(df, OUTPUT_DIR / "freMTPL2_severity.csv")

    logger.info("Done!")


if __name__ == "__main__":
    main()
