"""Export the columns the PyTorch pipeline needs, in a format any environment can read.

Run with the environment that holds the EDA cache (the `sc` kernel):

    /home/mhanyu/.conda/envs/sc/bin/python export_model_input.py

The harmonized pickle is written by numpy 2.x and cannot be unpickled by the older numpy in the
torch environment, so the model input travels as a flat .npz instead. Rows stay in patient /
hour order, and `patient_start` marks where each patient's rows begin.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "datasets"        # relative, so a clone of the repo works unchanged
SOURCE = DATA_DIR / "physionet2019_harmonized.pkl"   # site value grids already equalised (EDA 6.4)
TARGET = DATA_DIR / "physionet2019_model_input.npz"
VITALS = ["MAP", "SBP", "DBP", "HR", "Resp", "O2Sat"]
STATIC = ["Age", "Gender", "HospAdmTime"]
# HospAdmTime runs to -5,367 h (7 months in hospital before the ICU). Standardised, that tail
# would dominate the feature, so it is clipped to 30 days before / 1 day after ICU admission.
# Affects well under 1% of patients and is a modelling choice, not part of the site alignment.
HOSP_ADM_RANGE = (-720.0, 24.0)

df = pd.read_pickle(SOURCE).sort_values(["patient_id", "hour"]).reset_index(drop=True)
clipped = (df.groupby("patient_id")["HospAdmTime"].first().pipe(
    lambda x: ~x.between(*HOSP_ADM_RANGE) & x.notna()).sum())
df["HospAdmTime"] = df["HospAdmTime"].clip(*HOSP_ADM_RANGE)
codes, patient_ids = pd.factorize(df["patient_id"], sort=True)
start = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]]).astype(np.int64)

np.savez_compressed(
    TARGET,
    patient_ids=patient_ids.to_numpy().astype(str),       # let numpy size the strings:
    patient_start=start,
    site=df.groupby("patient_id", sort=True)["set"].first().to_numpy().astype("U1"),
    iculos=df["ICULOS"].to_numpy(np.float32),
    vitals=df[VITALS].to_numpy(np.float32),           # NaN where not measured
    static=df.groupby("patient_id", sort=True)[STATIC].first().to_numpy(np.float32),
    vital_names=np.array(VITALS),                         # a fixed width silently truncates
    static_names=np.array(STATIC),                        # names like "HospAdmTime"
)
print(f"{TARGET}  rows={len(df):,}  patients={len(patient_ids):,}  "
      f"size={TARGET.stat().st_size / 1e6:.0f} MB")
print(f"HospAdmTime clipped to {HOSP_ADM_RANGE} h for {clipped:,} patients")
print("missing per vital (% of rows):",
      {v: round(100 * df[v].isna().mean(), 1) for v in VITALS})
