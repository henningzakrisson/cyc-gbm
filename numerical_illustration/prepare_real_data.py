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

from pathlib import Path

import numpy as np
import openml
import pandas as pd

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

AREA_MAP = {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5", "F": "6"}

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
    freq_df, *_ = openml.datasets.get_dataset(
        OPENML_FREQ_ID, download_data=True, download_qualities=False
    ).get_data()

    sev_df, *_ = openml.datasets.get_dataset(
        OPENML_SEV_ID, download_data=True, download_qualities=False
    ).get_data()

    return freq_df, sev_df


def _aggregate_severities(sev: pd.DataFrame, id_dtype: np.dtype) -> pd.DataFrame:
    """Aggregate claim severities per policy.

    Returns one row per IDpol with corrected ClaimNb (count of individual
    claim records) and total ClaimAmount.
    """
    return (
        sev.groupby("IDpol")
        .agg(ClaimNb=("ClaimAmount", "count"), ClaimAmount=("ClaimAmount", "sum"))
        .reset_index()
        .assign(IDpol=lambda d: d["IDpol"].astype(id_dtype))
    )


def _merge_and_fill(
    freq: pd.DataFrame, sev_agg: pd.DataFrame
) -> pd.DataFrame:
    """Left-join frequency table onto aggregated severities.

    Policies without any claim record get ClaimNb=0 and ClaimAmount=0.
    """
    return (
        freq.drop(columns=["ClaimNb"], errors="ignore")
        .merge(sev_agg, on="IDpol", how="left")
        .fillna({"ClaimNb": 0, "ClaimAmount": 0.0})
        .assign(ClaimNb=lambda d: d["ClaimNb"].astype(int))
    )


def _remove_suspicious_policies(df: pd.DataFrame) -> pd.DataFrame:
    """Drop policies with more than 5 claims (suspected data errors)."""
    return df.query("ClaimNb <= 5")


def _cap_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """Cap policy exposure at 1 year."""
    return df.assign(Exposure=lambda d: d["Exposure"].clip(upper=1.0))


def _relevel_vehbrand(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse VehBrand into 7 broader groups per Wüthrich Ch. 13.1."""
    return df.assign(
        VehBrand=lambda d: d["VehBrand"].astype(str).map(VEHBRAND_GROUPS).fillna("other")
    )


def clean_data(freq: pd.DataFrame, sev: pd.DataFrame) -> pd.DataFrame:
    """Apply the Wüthrich Ch. 13.1 cleaning procedure.

    1. Aggregate severities per policy to obtain corrected claim counts
       and total claim amounts.
    2. Left-join onto the frequency table and fill missing claim info.
    3. Remove policies with >5 claims (suspected data errors).
    4. Cap exposure at 1 year.
    5. Re-level VehBrand into 7 groups.
    """
    sev_agg = _aggregate_severities(sev, freq["IDpol"].dtype)
    return (
        _merge_and_fill(freq, sev_agg)
        .pipe(_remove_suspicious_policies)
        .pipe(_cap_exposure)
        .pipe(_relevel_vehbrand)
    )


def _prepare_continuous(df: pd.DataFrame) -> pd.DataFrame:
    """Derive continuous model features.

    - Cap VehPower at 12.
    - Log-transform population density.
    - Cast remaining continuous columns to float.
    """
    return df.assign(
        VehPower=lambda d: d["VehPower"].clip(upper=12).astype(int),
        VehAge=lambda d: d["VehAge"].astype(float),
        DrivAge=lambda d: d["DrivAge"].astype(float),
        BonusMalus=lambda d: d["BonusMalus"].astype(float),
        log_Density=lambda d: np.log(d["Density"].astype(float)),
    )


def _prepare_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Derive categorical model features.

    - Map Area letters to ordinal integers.
    - Ensure all categorical columns are stored as strings
      (categorical dtype is applied at load time via the YAML config).
    """
    return df.assign(
        Area=lambda d: d["Area"].astype(str).map(AREA_MAP).fillna(d["Area"].astype(str)),
        VehBrand=lambda d: d["VehBrand"].astype(str),
        VehGas=lambda d: d["VehGas"].astype(str),
        Region=lambda d: d["Region"].astype(str),
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare model-ready features from the cleaned dataset.

    Applies continuous and categorical transformations without mutating
    the input DataFrame.
    """
    return df.pipe(_prepare_continuous).pipe(_prepare_categorical)


def _select_and_label(
    df: pd.DataFrame, y_col: str, w_col: str
) -> pd.DataFrame:
    """Select model features and append target (y) and weight (w) columns."""
    features = CATEGORICAL_FEATURES + CONTINUOUS_FEATURES
    return df[features].assign(y=df[y_col], w=df[w_col])


def write_counts_csv(df: pd.DataFrame, path: Path) -> None:
    """Write the claim-count dataset (all policies).

    Columns: categorical + continuous features, y=ClaimNb, w=Exposure.
    """
    _select_and_label(df, y_col="ClaimNb", w_col="Exposure").to_csv(
        path, index=False
    )


def write_severity_csv(df: pd.DataFrame, path: Path) -> None:
    """Write the claim-severity dataset (policies with claims only).

    Columns: categorical + continuous features, y=ClaimAmount, w=ClaimNb.
    """
    (
        df.query("ClaimNb > 0")
        .pipe(_select_and_label, y_col="ClaimAmount", w_col="ClaimNb")
        .to_csv(path, index=False)
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    freq, sev = download_data()
    df = clean_data(freq, sev).pipe(engineer_features)

    write_counts_csv(df, OUTPUT_DIR / "freMTPL2_counts.csv")
    write_severity_csv(df, OUTPUT_DIR / "freMTPL2_severity.csv")


if __name__ == "__main__":
    main()
