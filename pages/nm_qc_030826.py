#!/usr/bin/env python3
"""
Streamlit version of:
Advanced Temperature QC with Silver Labels

Run with:
    streamlit run streamlit_advanced_temperature_qc.py
"""

from __future__ import annotations

import os
import io
import math
import zipfile
import warnings
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from geopy.distance import geodesic

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    IsolationForest,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Advanced Temperature QC",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Advanced Temperature QC with Silver Labels")
st.write(
    """
This Streamlit app performs climatology-aware, spatially-informed QA/QC on NOAA
hourly station temperature data, generates silver labels, and optionally trains
a supervised ML model to reproduce anomaly labels.
"""
)

# =============================================================================
# CONSTANTS
# =============================================================================
ISD_METADATA_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
ACCESS_BASE_URL = "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{filename}"

SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall"
}

SEASON_CODE_MAP = {
    "Winter": 0,
    "Spring": 1,
    "Summer": 2,
    "Fall": 3
}

# =============================================================================
# PARAMS
# =============================================================================
@dataclass
class AppArgs:
    output_dir: str
    state: str = "NM"
    primary_filename: Optional[str] = None
    primary_name: Optional[str] = None
    start_year: int = 2018
    end_year: int = 2023
    max_neighbors: int = 3
    max_distance_km: float = 30.0
    max_elev_diff: float = 300.0
    min_corr: float = 0.40
    aux_contamination: float = 0.02
    metadata_events_csv: Optional[str] = None
    expert_labels_csv: Optional[str] = None
    expert_label_col: str = "expert_label"
    train_ml_on: str = "expert_else_silver"
    ml_model: str = "random_forest"
    split_method: str = "latest_years"
    n_test_years: Optional[int] = None
    ml_prob_threshold: float = 0.80


# =============================================================================
# HELPERS
# =============================================================================
def safe_percentile(x: pd.Series, q: float, min_n: int = 24) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if len(vals) < min_n:
        return np.nan
    return float(np.nanpercentile(vals, q))


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    return out.sort_index()


def detect_event_type(row: pd.Series) -> str:
    if bool(row.get("wind_low_flag", False)) and bool(row.get("diurnal_range_low_flag", False)):
        return "Calm Night"
    if bool(row.get("front_like_flag", False)):
        return "Frontal Passage"
    if bool(row.get("cold_surge_flag", False)):
        return "Cold Surge"
    return "Other"


def make_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def zip_directory(directory: Path) -> io.BytesIO:
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(directory))
    mem_zip.seek(0)
    return mem_zip


# =============================================================================
# TEMPLATE FILES
# =============================================================================
def write_template_csvs(output_dir: Path) -> None:
    metadata_template = pd.DataFrame({
        "DATE": [
            "2020-05-10 00:00:00",
            "2021-09-18 12:00:00"
        ],
        "event_type": [
            "instrument_change",
            "station_relocation"
        ],
        "notes": [
            "Sensor replaced after maintenance",
            "Station moved to new site"
        ]
    })

    expert_labels_template = pd.DataFrame({
        "DATE": [
            "2020-01-05 03:00:00",
            "2020-01-05 04:00:00",
            "2020-02-10 12:00:00",
            "2020-03-14 07:00:00"
        ],
        "expert_label": [
            1,
            1,
            0,
            1
        ],
        "label_source": [
            "manual_review",
            "manual_review",
            "manual_review",
            "manual_review"
        ],
        "notes": [
            "Flatline and disagrees with neighbors",
            "Persistence continues",
            "Sharp cooling but matched by neighbors",
            "Sustained spatial disagreement"
        ]
    })

    metadata_template.to_csv(output_dir / "metadata_events_template.csv", index=False)
    expert_labels_template.to_csv(output_dir / "expert_true_labels_template.csv", index=False)


# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data(show_spinner=False)
def load_station_metadata() -> pd.DataFrame:
    stations = pd.read_csv(ISD_METADATA_URL)
    stations = stations.rename(columns=str.upper).copy()

    if "ELEV(M)" in stations.columns and "ELEV_M" not in stations.columns:
        stations["ELEV_M"] = pd.to_numeric(stations["ELEV(M)"], errors="coerce")
    elif "ELEV_M" not in stations.columns:
        stations["ELEV_M"] = np.nan
    else:
        stations["ELEV_M"] = pd.to_numeric(stations["ELEV_M"], errors="coerce")

    stations["LAT"] = pd.to_numeric(stations["LAT"], errors="coerce")
    stations["LON"] = pd.to_numeric(stations["LON"], errors="coerce")

    stations = stations[
        stations["LAT"].notna() &
        stations["LON"].notna()
    ].copy()

    stations["USAF"] = stations["USAF"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
    stations["WBAN"] = stations["WBAN"].astype(str).str.replace(".0", "", regex=False).str.zfill(5)
    stations["FILENAME"] = stations["USAF"] + stations["WBAN"] + ".csv"

    if "STATE" not in stations.columns:
        stations["STATE"] = ""

    if "STATION NAME" not in stations.columns:
        stations["STATION NAME"] = stations["USAF"] + "-" + stations["WBAN"]

    return stations


@st.cache_data(show_spinner=False)
def load_access_csv(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url, low_memory=False)
        if "DATE" not in df.columns or "TMP" not in df.columns:
            return pd.DataFrame()

        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.dropna(subset=["DATE"]).set_index("DATE").sort_index()

        tmp_split = df["TMP"].astype(str).str.split(",", expand=True)
        df["TMP_raw"] = pd.to_numeric(tmp_split[0], errors="coerce")
        df["TMP_raw"] = df["TMP_raw"].replace(9999, np.nan)
        df["T_air"] = df["TMP_raw"] / 10.0

        if "WND" in df.columns:
            wnd_split = df["WND"].astype(str).str.split(",", expand=True)
            if wnd_split.shape[1] >= 4:
                df["WND_speed_raw"] = pd.to_numeric(wnd_split[3], errors="coerce")
                df["wind_speed_ms"] = df["WND_speed_raw"].replace(9999, np.nan) / 10.0
            else:
                df["wind_speed_ms"] = np.nan
        else:
            df["wind_speed_ms"] = np.nan

        return df
    except Exception:
        return pd.DataFrame()


def load_station_years(filename: str, years: Tuple[int, ...]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for year in years:
        url = ACCESS_BASE_URL.format(year=year, filename=filename)
        df = load_access_csv(url)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


# =============================================================================
# CLIMATOLOGY
# =============================================================================
def compute_station_climatology(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_datetime_index(df)
    temp = pd.to_numeric(out["T_air"], errors="coerce")

    clim_df = pd.DataFrame(index=out.index)
    clim_df["T_air"] = temp
    clim_df["hour"] = clim_df.index.hour
    clim_df["month"] = clim_df.index.month
    clim_df["season"] = clim_df["month"].map(SEASON_MAP)

    grouped = (
        clim_df.groupby(["season", "hour"])["T_air"]
        .agg(
            clim_mean="mean",
            clim_std="std",
            clim_p01=lambda x: safe_percentile(x, 1),
            clim_p05=lambda x: safe_percentile(x, 5),
            clim_p10=lambda x: safe_percentile(x, 10),
            clim_p90=lambda x: safe_percentile(x, 90),
            clim_p95=lambda x: safe_percentile(x, 95),
            clim_p99=lambda x: safe_percentile(x, 99),
        )
        .reset_index()
    )
    return grouped


def attach_climatology(df: pd.DataFrame, clim_df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_datetime_index(df).copy()
    out["hour"] = out.index.hour
    out["month"] = out.index.month
    out["season"] = out["month"].map(SEASON_MAP)

    out = (
        out.reset_index()
        .merge(clim_df, on=["season", "hour"], how="left")
        .set_index("DATE")
        .sort_index()
    )
    return out


# =============================================================================
# NEIGHBORS
# =============================================================================
def find_candidate_neighbors(
    primary_row: pd.Series,
    station_df: pd.DataFrame,
    max_candidates: int = 10,
    max_distance_km: float = 200.0,
    max_elev_diff: float = 300.0
) -> pd.DataFrame:
    records = []

    plat = float(primary_row["LAT"])
    plon = float(primary_row["LON"])
    pelev = pd.to_numeric(primary_row.get("ELEV_M", np.nan), errors="coerce")
    primary_filename = primary_row["FILENAME"]

    for _, row in station_df.iterrows():
        if row["FILENAME"] == primary_filename:
            continue

        try:
            dist = geodesic((plat, plon), (float(row["LAT"]), float(row["LON"]))).km
        except Exception:
            continue

        elev = pd.to_numeric(row.get("ELEV_M", np.nan), errors="coerce")
        elev_diff = np.nan if pd.isna(pelev) or pd.isna(elev) else abs(pelev - elev)

        if dist <= max_distance_km:
            if pd.isna(elev_diff) or elev_diff <= max_elev_diff:
                records.append({
                    "STATION NAME": row["STATION NAME"],
                    "FILENAME": row["FILENAME"],
                    "LAT": row["LAT"],
                    "LON": row["LON"],
                    "ELEV_M": elev,
                    "distance_km": dist,
                    "elev_diff_m": elev_diff
                })

    recs = pd.DataFrame(records)
    if recs.empty:
        return recs

    recs = recs.sort_values(["distance_km", "elev_diff_m"], na_position="last").head(max_candidates)
    return recs


def compute_pairwise_climatology_correlation(primary_df: pd.DataFrame, neighbor_df: pd.DataFrame) -> Tuple[float, int]:
    common = primary_df.index.intersection(neighbor_df.index)
    if len(common) < 200:
        return np.nan, len(common)

    p = pd.to_numeric(primary_df.loc[common, "T_air"], errors="coerce")
    n = pd.to_numeric(neighbor_df.loc[common, "T_air"], errors="coerce")
    mask = p.notna() & n.notna()
    valid_overlap = int(mask.sum())

    if valid_overlap < 200:
        return np.nan, valid_overlap

    try:
        r, _ = pearsonr(p[mask], n[mask])
        return float(r), valid_overlap
    except Exception:
        return np.nan, valid_overlap


def merge_neighbor_series(primary_df: pd.DataFrame, neighbor_dict: Dict[str, dict]) -> Tuple[pd.DataFrame, List[Tuple[str, dict]]]:
    out = primary_df.copy()
    neighbor_cols: List[Tuple[str, dict]] = []

    for i, (fname, info) in enumerate(neighbor_dict.items(), start=1):
        ndf = info["data"][["T_air"]].rename(columns={"T_air": f"neighbor_{i}_T"})
        out = out.join(ndf, how="left")
        neighbor_cols.append((f"neighbor_{i}_T", info))

    return out, neighbor_cols


# =============================================================================
# RULE FEATURES
# =============================================================================
def build_dynamic_rule_features(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_datetime_index(df).copy()

    out["T_air"] = pd.to_numeric(out["T_air"], errors="coerce")
    out["month"] = out.index.month
    out["season"] = out["month"].map(SEASON_MAP)
    out["hour"] = out.index.hour
    out["dayofyear"] = out.index.dayofyear

    out["dT_1h"] = out["T_air"].diff(1)
    out["dT_3h"] = out["T_air"].diff(3)
    out["dT_6h"] = out["T_air"].diff(6)

    out["rolling_mean_6h"] = out["T_air"].rolling(6, min_periods=3).mean()
    out["rolling_std_6h"] = out["T_air"].rolling(6, min_periods=3).std()
    out["rolling_mean_24h"] = out["T_air"].rolling(24, min_periods=12).mean()
    out["rolling_std_24h"] = out["T_air"].rolling(24, min_periods=12).std()
    out["rolling_min_24h"] = out["T_air"].rolling(24, min_periods=12).min()
    out["rolling_max_24h"] = out["T_air"].rolling(24, min_periods=12).max()

    out["zscore_clim"] = (out["T_air"] - out["clim_mean"]) / out["clim_std"]

    out["within_seasonal_bounds"] = (
        (out["T_air"] >= out["clim_p05"]) &
        (out["T_air"] <= out["clim_p95"])
    )

    out["far_outside_seasonal_bounds"] = (
        (out["T_air"] < out["clim_p01"] - 2.0) |
        (out["T_air"] > out["clim_p99"] + 2.0)
    )

    seasonal_spike = {
        "Winter": 10.0,
        "Spring": 8.0,
        "Summer": 6.5,
        "Fall": 8.0
    }
    out["spike_thresh_1h"] = out["season"].map(seasonal_spike).astype(float)
    out["spike_dynamic_flag"] = out["dT_1h"].abs() > out["spike_thresh_1h"]

    out["flatline_flag"] = (
        (out["rolling_std_6h"] < 0.15) &
        (out["rolling_std_24h"] < 0.25)
    )

    out["cold_surge_flag"] = out["dT_3h"] <= -8
    out["front_like_flag"] = out["dT_3h"].abs() >= 7
    out["wind_low_flag"] = pd.to_numeric(out.get("wind_speed_ms", np.nan), errors="coerce").fillna(99) < 1.5
    out["diurnal_range_low_flag"] = (out["rolling_max_24h"] - out["rolling_min_24h"]) < 2.0
    out["event_type"] = out.apply(detect_event_type, axis=1)

    return out


# =============================================================================
# SPATIAL QC
# =============================================================================
def build_spatial_qc(
    primary_df: pd.DataFrame,
    neighbor_cols: List[Tuple[str, dict]],
    min_neighbors_required: int = 2
) -> pd.DataFrame:
    out = primary_df.copy()

    if len(neighbor_cols) == 0:
        out["available_neighbors"] = 0
        out["neighbor_weighted_mean"] = np.nan
        out["neighbor_median"] = np.nan
        out["neighbor_neighbor_spread"] = np.nan
        out["spatial_diff_to_median"] = np.nan
        out["primary_anom"] = out["T_air"] - out["rolling_mean_24h"]
        out["neighbor_median_anom"] = np.nan
        out["spatial_anom_diff"] = np.nan
        out["spatial_thresh"] = np.nan
        out["spatial_instant_flag"] = False
        out["spatial_sustained_flag"] = False
        out["neighbor_median_dT1"] = np.nan
        out["implausible_jump_no_neighbor_support"] = False
        return out

    robust_vals = []
    robust_weights = []
    corr_vals = []
    temp_cols = []

    for col, info in neighbor_cols:
        corr_val = info.get("correlation", np.nan)
        dist = info.get("distance_km", np.nan)
        elev_diff = info.get("elev_diff_m", np.nan)

        weight = 1.0
        if pd.notna(corr_val):
            weight *= max(float(corr_val), 0.05)
        if pd.notna(dist):
            weight *= 1.0 / max(float(dist), 1.0)
        if pd.notna(elev_diff):
            weight *= 1.0 / (1.0 + float(elev_diff) / 100.0)

        robust_vals.append(out[col])
        robust_weights.append(weight)
        corr_vals.append(corr_val)
        temp_cols.append(col)

        out[f"{col}_anom"] = out[col] - out[col].rolling(24, min_periods=12).mean()

    stacked = pd.concat(robust_vals, axis=1)
    stacked.columns = temp_cols

    out["available_neighbors"] = stacked.notna().sum(axis=1)

    weights = np.array(robust_weights, dtype=float)
    weighted_num = stacked.mul(weights, axis=1).sum(axis=1, skipna=True)
    weighted_den = stacked.notna().mul(weights, axis=1).sum(axis=1, skipna=True)
    out["neighbor_weighted_mean"] = weighted_num / weighted_den.replace(0, np.nan)

    out["neighbor_median"] = stacked.median(axis=1)
    out["neighbor_neighbor_spread"] = stacked.std(axis=1)

    out["primary_anom"] = out["T_air"] - out["rolling_mean_24h"]
    out["neighbor_median_anom"] = out["neighbor_median"] - out["neighbor_median"].rolling(24, min_periods=12).mean()
    out["spatial_diff_to_median"] = out["T_air"] - out["neighbor_median"]
    out["spatial_anom_diff"] = out["primary_anom"] - out["neighbor_median_anom"]

    mean_corr = np.nanmean(corr_vals) if len(corr_vals) > 0 else np.nan
    if np.isnan(mean_corr):
        mean_corr = 0.5

    base_thresh = np.interp(mean_corr, [0.3, 0.95], [8.0, 3.5])
    out["spatial_thresh"] = base_thresh

    out["spatial_instant_flag"] = (
        (out["available_neighbors"] >= min_neighbors_required) &
        (
            out["neighbor_neighbor_spread"].isna() |
            (out["neighbor_neighbor_spread"] <= 3.0)
        ) &
        (out["spatial_anom_diff"].abs() > out["spatial_thresh"])
    )

    out["spatial_sustained_flag"] = (
        out["spatial_instant_flag"].rolling(4, min_periods=3).sum() >= 3
    ).fillna(False)

    neighbor_diff_cols = []
    for col in temp_cols:
        out[f"{col}_dT1"] = out[col].diff(1)
        neighbor_diff_cols.append(f"{col}_dT1")

    if len(neighbor_diff_cols) > 0:
        out["neighbor_median_dT1"] = out[neighbor_diff_cols].median(axis=1)
        out["implausible_jump_no_neighbor_support"] = (
            (out["dT_1h"].abs() > out["spike_thresh_1h"]) &
            (out["available_neighbors"] >= min_neighbors_required) &
            (
                out["neighbor_neighbor_spread"].isna() |
                (out["neighbor_neighbor_spread"] <= 3.0)
            ) &
            (out["neighbor_median_dT1"].abs() < 2.0)
        )
    else:
        out["neighbor_median_dT1"] = np.nan
        out["implausible_jump_no_neighbor_support"] = False

    return out


# =============================================================================
# METADATA EVENTS
# =============================================================================
def prepare_metadata_events(metadata_events: Optional[pd.DataFrame]) -> pd.DataFrame:
    if metadata_events is None or metadata_events.empty:
        return pd.DataFrame(columns=["DATE", "event_type", "notes"])

    out = metadata_events.copy()
    out["DATE"] = pd.to_datetime(out["DATE"], errors="coerce")
    out = out.dropna(subset=["DATE"]).sort_values("DATE")

    if "event_type" not in out.columns:
        out["event_type"] = "metadata_event"
    if "notes" not in out.columns:
        out["notes"] = ""

    return out


def attach_metadata_event_flag(df: pd.DataFrame, metadata_events: Optional[pd.DataFrame], window_days: int = 3) -> pd.DataFrame:
    out = df.copy()
    out["metadata_event_flag"] = 0

    meta = prepare_metadata_events(metadata_events)
    if meta.empty:
        return out

    for event_date in meta["DATE"]:
        mask = (out.index >= event_date - pd.Timedelta(days=window_days)) & (out.index <= event_date + pd.Timedelta(days=window_days))
        out.loc[mask, "metadata_event_flag"] = 1

    return out


# =============================================================================
# RULE-BASED QC
# =============================================================================
def apply_advanced_rule_qc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["rule_range_flag"] = out["far_outside_seasonal_bounds"].fillna(False)
    out["rule_spike_flag"] = out["spike_dynamic_flag"].fillna(False)
    out["rule_flatline_flag"] = out["flatline_flag"].fillna(False)
    out["rule_spatial_flag"] = out["spatial_sustained_flag"].fillna(False)
    out["rule_jump_nosupport_flag"] = out["implausible_jump_no_neighbor_support"].fillna(False)

    out["rule_flag"] = (
        out["rule_range_flag"] |
        out["rule_spike_flag"] |
        out["rule_flatline_flag"] |
        out["rule_spatial_flag"] |
        out["rule_jump_nosupport_flag"]
    )

    def _reason(row):
        reasons = []
        if row["rule_range_flag"]:
            reasons.append("Dynamic range")
        if row["rule_spike_flag"]:
            reasons.append("Seasonal spike")
        if row["rule_flatline_flag"]:
            reasons.append("Flatline")
        if row["rule_spatial_flag"]:
            reasons.append("Sustained spatial disagreement")
        if row["rule_jump_nosupport_flag"]:
            reasons.append("Jump unsupported by neighbors")
        return "; ".join(reasons) if reasons else "None"

    out["rule_reason"] = out.apply(_reason, axis=1)
    return out


# =============================================================================
# AUXILIARY UNSUPERVISED MODEL
# =============================================================================
def prepare_unsupervised_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out["T_air"] = pd.to_numeric(df["T_air"], errors="coerce")
    out["hour"] = df.index.hour
    out["month"] = df.index.month
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    out["dT_1h"] = df["dT_1h"]
    out["dT_3h"] = df["dT_3h"]
    out["rolling_std_6h"] = df["rolling_std_6h"]
    out["rolling_std_24h"] = df["rolling_std_24h"]
    out["zscore_clim"] = df["zscore_clim"]
    out["spatial_anom_diff"] = df["spatial_anom_diff"]
    out["available_neighbors"] = df["available_neighbors"]

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def run_auxiliary_isolation_forest(df: pd.DataFrame, contamination: float = 0.02) -> pd.Series:
    feats = prepare_unsupervised_features(df)
    valid = feats.dropna()

    flags = pd.Series(False, index=df.index)
    if len(valid) < 100:
        return flags

    scaler = StandardScaler()
    X = scaler.fit_transform(valid)

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
        bootstrap=True
    )
    pred = model.fit_predict(X)
    flags.loc[valid.index] = pred == -1
    return flags


# =============================================================================
# SILVER LABELS
# =============================================================================
def generate_silver_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required_defaults = {
        "far_outside_seasonal_bounds": False,
        "flatline_flag": False,
        "spatial_sustained_flag": False,
        "implausible_jump_no_neighbor_support": False,
        "metadata_event_flag": 0,
        "available_neighbors": 0,
        "spatial_anom_diff": np.nan,
        "spatial_thresh": np.nan,
        "neighbor_neighbor_spread": np.nan,
        "rule_flag": False,
        "aux_iforest_flag": False,
        "spatial_instant_flag": False,
        "within_seasonal_bounds": False,
    }
    for col, default_val in required_defaults.items():
        if col not in out.columns:
            out[col] = default_val

    high_conf_bad = (
        out["far_outside_seasonal_bounds"].fillna(False) |
        out["flatline_flag"].fillna(False) |
        out["spatial_sustained_flag"].fillna(False) |
        out["implausible_jump_no_neighbor_support"].fillna(False) |
        (
            (out["metadata_event_flag"] == 1) &
            (out["available_neighbors"] >= 2) &
            (out["spatial_anom_diff"].abs() > out["spatial_thresh"].fillna(np.inf)) &
            (
                out["neighbor_neighbor_spread"].isna() |
                (out["neighbor_neighbor_spread"] <= 4.5)
            )
        )
    )

    out["anomaly_vote_count"] = (
        out["rule_flag"].fillna(False).astype(int) +
        out["aux_iforest_flag"].fillna(False).astype(int) +
        out["spatial_instant_flag"].fillna(False).astype(int)
    )

    spatial_acceptable = (
        out["spatial_anom_diff"].isna() |
        out["spatial_thresh"].isna() |
        (out["spatial_anom_diff"].abs() <= out["spatial_thresh"].fillna(np.inf) * 1.25)
    )

    spread_acceptable = (
        out["neighbor_neighbor_spread"].isna() |
        (out["neighbor_neighbor_spread"] <= 4.5)
    )

    neighbor_ok = (
        (out["available_neighbors"] >= 2) |
        out["spatial_anom_diff"].isna()
    )

    high_conf_good = (
        out["within_seasonal_bounds"].fillna(False) &
        (~out["rule_flag"].fillna(False)) &
        neighbor_ok &
        spatial_acceptable &
        spread_acceptable &
        (out["metadata_event_flag"] == 0) &
        (out["anomaly_vote_count"] <= 1)
    )

    out["silver_label"] = np.nan
    out.loc[high_conf_bad, "silver_label"] = 1.0
    out.loc[high_conf_good & ~high_conf_bad, "silver_label"] = 0.0

    n_good = int((out["silver_label"] == 0).sum())
    if n_good == 0:
        fallback_good_mask = (
            out["within_seasonal_bounds"].fillna(False) &
            (~out["rule_flag"].fillna(False)) &
            (out["metadata_event_flag"] == 0)
        )

        fallback_candidates = out[fallback_good_mask].copy()
        if len(fallback_candidates) > 0:
            n_bad = int((out["silver_label"] == 1).sum())
            fallback_n = min(
                len(fallback_candidates),
                max(200, min(1000, n_bad if n_bad > 0 else 500))
            )
            fallback_idx = fallback_candidates.sample(n=fallback_n, random_state=42).index
            out.loc[fallback_idx, "silver_label"] = 0.0
            out.loc[fallback_idx, "silver_reason"] = "Fallback high-confidence good"

    def _silver_reason(row):
        if pd.isna(row["silver_label"]):
            return "Uncertain"
        if row["silver_label"] == 1:
            reasons = []
            if bool(row["far_outside_seasonal_bounds"]):
                reasons.append("Far outside seasonal bounds")
            if bool(row["flatline_flag"]):
                reasons.append("Sustained flatline")
            if bool(row["spatial_sustained_flag"]):
                reasons.append("Sustained spatial disagreement")
            if bool(row["implausible_jump_no_neighbor_support"]):
                reasons.append("Implausible jump unsupported by neighbors")
            if int(row["metadata_event_flag"]) == 1:
                reasons.append("Near metadata event")
            return "; ".join(reasons) if reasons else "High-confidence bad"

        existing_reason = row.get("silver_reason", None)
        if isinstance(existing_reason, str) and existing_reason.strip():
            return existing_reason
        return "High-confidence good"

    if "silver_reason" not in out.columns:
        out["silver_reason"] = ""

    needs_reason = out["silver_reason"].isna() | (out["silver_reason"].astype(str).str.strip() == "")
    out.loc[needs_reason, "silver_reason"] = out.loc[needs_reason].apply(_silver_reason, axis=1)

    out["review_priority"] = "Low"
    out.loc[out["silver_label"].isna(), "review_priority"] = "Medium"
    out.loc[
        out["silver_label"].isna() &
        (
            out["rule_flag"].fillna(False) |
            out["aux_iforest_flag"].fillna(False) |
            out["spatial_instant_flag"].fillna(False)
        ),
        "review_priority"
    ] = "High"

    out.attrs["silver_debug_counts"] = {
        "within_seasonal_bounds": int(out["within_seasonal_bounds"].fillna(False).sum()),
        "not_rule_flag": int((~out["rule_flag"].fillna(False)).sum()),
        "available_neighbors_ge_2": int((out["available_neighbors"] >= 2).sum()),
        "spatial_acceptable": int(spatial_acceptable.fillna(False).sum()),
        "neighbor_spread_acceptable": int(spread_acceptable.fillna(False).sum()),
        "metadata_event_flag_eq_0": int((out["metadata_event_flag"] == 0).sum()),
        "anomaly_vote_count_le_1": int((out["anomaly_vote_count"] <= 1).sum()),
        "high_conf_bad": int(high_conf_bad.sum()),
        "high_conf_good": int(high_conf_good.sum()),
        "silver_bad_final": int((out["silver_label"] == 1).sum()),
        "silver_good_final": int((out["silver_label"] == 0).sum()),
        "silver_uncertain_final": int(out["silver_label"].isna().sum()),
    }

    return out


# =============================================================================
# ML FEATURES
# =============================================================================
def prepare_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out["T_air"] = pd.to_numeric(df["T_air"], errors="coerce")
    out["hour"] = df.index.hour
    out["month"] = df.index.month
    out["dayofyear"] = df.index.dayofyear
    out["season_code"] = df["season"].map(SEASON_CODE_MAP).fillna(-1)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    for lag in [1, 3, 6, 12, 24]:
        out[f"lag_{lag}"] = df["T_air"].shift(lag)

    for window in [6, 12, 24]:
        out[f"roll_mean_{window}"] = df["T_air"].rolling(window, min_periods=3).mean()
        out[f"roll_std_{window}"] = df["T_air"].rolling(window, min_periods=3).std()
        out[f"roll_min_{window}"] = df["T_air"].rolling(window, min_periods=3).min()
        out[f"roll_max_{window}"] = df["T_air"].rolling(window, min_periods=3).max()

    out["dT_1h"] = df["dT_1h"]
    out["dT_3h"] = df["dT_3h"]
    out["dT_6h"] = df["dT_6h"]
    out["zscore_clim"] = df["zscore_clim"]
    out["spatial_anom_diff"] = df["spatial_anom_diff"]
    out["available_neighbors"] = df["available_neighbors"]
    out["neighbor_neighbor_spread"] = df["neighbor_neighbor_spread"]
    out["metadata_event_flag"] = df["metadata_event_flag"]
    out["aux_iforest_flag"] = df["aux_iforest_flag"].astype(int)
    out["event_type_code"] = df["event_type"].map({
        "Other": 0,
        "Cold Surge": 1,
        "Frontal Passage": 2,
        "Calm Night": 3
    }).fillna(0)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# =============================================================================
# METRICS TABLES
# =============================================================================
def seasonal_metrics(df: pd.DataFrame, truth_col: str, pred_col: str) -> pd.DataFrame:
    rows = []
    for season, g in df.groupby("season"):
        if g[truth_col].nunique() < 2:
            continue

        y_true = g[truth_col].astype(int)
        y_pred = g[pred_col].astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        rows.append({
            "season": season,
            "n": len(g),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mcc": matthews_corrcoef(y_true, y_pred)
        })
    return pd.DataFrame(rows)


def event_metrics(df: pd.DataFrame, truth_col: str, pred_col: str) -> pd.DataFrame:
    rows = []
    for event_type, g in df.groupby("event_type"):
        if g[truth_col].nunique() < 2:
            continue

        y_true = g[truth_col].astype(int)
        y_pred = g[pred_col].astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        rows.append({
            "event_type": event_type,
            "n": len(g),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mcc": matthews_corrcoef(y_true, y_pred)
        })
    return pd.DataFrame(rows)


# =============================================================================
# ML MODEL FACTORY
# =============================================================================
def make_classifier(model_name: str):
    model_name = model_name.lower()

    if model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=14,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    elif model_name == "extra_trees":
        clf = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=14,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    elif model_name == "gradient_boosting":
        clf = GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    elif model_name == "logistic_regression":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    else:
        raise ValueError(
            "Unsupported model. Choose from: "
            "random_forest, extra_trees, gradient_boosting, logistic_regression"
        )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])
    return pipe


# =============================================================================
# FIGURE HELPERS
# =============================================================================
def select_best_zoom_window(
    df: pd.DataFrame,
    months: int = 6,
    flag_cols: List[str] = None
) -> pd.DataFrame:
    if flag_cols is None:
        flag_cols = ["rule_flag", "ml_flag"]

    plot_df = df.copy().sort_index()

    combined_flag = pd.Series(0.0, index=plot_df.index)

    for col in flag_cols:
        if col in plot_df.columns:
            combined_flag += pd.to_numeric(plot_df[col], errors="coerce").fillna(0).astype(float)

    if combined_flag.sum() == 0:
        start = plot_df.index.min()
        end = start + pd.DateOffset(months=months)
        return plot_df.loc[start:end].copy()

    daily_flag_count = combined_flag.resample("D").sum()
    window_days = max(30, int(months * 30))

    rolling_sum = daily_flag_count.rolling(
        window_days,
        min_periods=max(7, window_days // 4)
    ).sum()

    if rolling_sum.dropna().empty:
        start = plot_df.index.min()
        end = start + pd.DateOffset(months=months)
        return plot_df.loc[start:end].copy()

    best_end_day = rolling_sum.idxmax()
    best_start_day = best_end_day - pd.Timedelta(days=window_days)

    zoom_df = plot_df.loc[best_start_day:best_end_day].copy()

    if zoom_df.empty:
        start = plot_df.index.min()
        end = start + pd.DateOffset(months=months)
        zoom_df = plot_df.loc[start:end].copy()

    return zoom_df


def save_zoom_time_series_figures(
    df: pd.DataFrame,
    output_dir: Path,
    base_title: str = "Temperature and Flag Comparison"
) -> None:
    zoom_6m = select_best_zoom_window(df, months=6, flag_cols=["original_label", "ml_flag"])
    save_time_series_comparison_png(
        zoom_6m,
        output_dir / "time_series_zoom_6month.png",
        f"{base_title} (Automatic 6-Month Zoom)"
    )

    zoom_3m = select_best_zoom_window(df, months=3, flag_cols=["original_label", "ml_flag"])
    save_time_series_comparison_png(
        zoom_3m,
        output_dir / "time_series_zoom_3month.png",
        f"{base_title} (Automatic 3-Month Zoom)"
    )


def save_daily_flag_counts_png(df: pd.DataFrame, output_path: Path, title: str) -> None:
    plot_df = df.copy()
    daily_counts = plot_df["rule_flag"].fillna(False).astype(int).resample("D").sum()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(daily_counts.index, daily_counts.values, width=1.0, color="slateblue")
    ax.set_title(title)
    ax.set_ylabel("Flag count per day")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_spatial_consistency_png(df: pd.DataFrame, output_path: Path, title: str) -> None:
    if "neighbor_median_anom" not in df.columns or df["neighbor_median_anom"].isna().all():
        return

    plot_df = df.copy()
    spatial_mask = plot_df["spatial_sustained_flag"].fillna(False)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.plot(plot_df.index, plot_df["primary_anom"], color="steelblue", linewidth=1.1, label="Primary anomaly")
    ax1.plot(plot_df.index, plot_df["neighbor_median_anom"], color="darkorange", linewidth=1.1, label="Neighbor median anomaly")
    ax1.scatter(
        plot_df.index[spatial_mask],
        plot_df.loc[spatial_mask, "primary_anom"],
        color="red", s=18, marker="o", label="Spatial sustained flags"
    )
    ax1.set_ylabel("Anomaly (°C)")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.plot(plot_df.index, plot_df["spatial_anom_diff"], color="black", linewidth=1.0, label="Spatial anomaly difference")

    thresh = pd.to_numeric(plot_df["spatial_thresh"], errors="coerce").median()
    if pd.notna(thresh):
        ax2.axhline(thresh, color="red", linestyle="--", linewidth=1.2, label=f"+threshold ({thresh:.1f})")
        ax2.axhline(-thresh, color="red", linestyle="--", linewidth=1.2, label=f"-threshold ({thresh:.1f})")

    ax2.set_ylabel("Primary - Neighbor")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_climatology_zscore_png(df: pd.DataFrame, output_path: Path, title: str) -> None:
    plot_df = df.copy()
    bad_mask = plot_df["rule_flag"].fillna(False)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.plot(plot_df.index, plot_df["T_air"], color="steelblue", linewidth=1.0, alpha=0.8, label="Observed temperature")
    ax1.plot(plot_df.index, plot_df["clim_mean"], color="darkgreen", linewidth=1.0, alpha=0.8, label="Climatological mean")
    ax1.scatter(
        plot_df.index[bad_mask],
        plot_df.loc[bad_mask, "T_air"],
        color="red", s=16, marker="o", label="Rule-based flags"
    )
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.plot(plot_df.index, plot_df["zscore_clim"], color="black", linewidth=1.0, alpha=0.8, label="Climatological z-score")
    ax2.axhline(3, color="red", linestyle="--", linewidth=1.2, label="+3")
    ax2.axhline(-3, color="red", linestyle="--", linewidth=1.2, label="-3")
    ax2.set_ylabel("Z-score")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_spike_detector_png(df: pd.DataFrame, output_path: Path, title: str) -> None:
    plot_df = df.copy()
    spike_mask = plot_df["spike_dynamic_flag"].fillna(False)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.plot(plot_df.index, plot_df["T_air"], color="steelblue", linewidth=1.2, alpha=0.8, label="Temperature")
    ax1.scatter(
        plot_df.index[spike_mask],
        plot_df.loc[spike_mask, "T_air"],
        color="crimson", s=20, marker="o", label="Spike flag"
    )
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    if "spike_thresh_1h" in plot_df.columns:
        thresh = pd.to_numeric(plot_df["spike_thresh_1h"], errors="coerce").median()
    else:
        thresh = 8.0

    ax2.plot(plot_df.index, plot_df["dT_1h"], color="black", linewidth=1.0, alpha=0.8, label="1h temperature change")
    ax2.axhline(thresh, color="red", linestyle="--", linewidth=1.2, label=f"+threshold ({thresh:.1f}°C)")
    ax2.axhline(-thresh, color="red", linestyle="--", linewidth=1.2, label=f"-threshold ({thresh:.1f}°C)")
    ax2.set_ylabel("dT / 1h (°C)")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_png(cm: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred Good", "Pred Bad"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True Good", "True Bad"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_probability_plot(eval_plot_df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(eval_plot_df.index, eval_plot_df["pred_prob_bad"], label="Predicted probability of bad")
    bad_mask = eval_plot_df["true_label"] == 1
    ax.scatter(eval_plot_df.index[bad_mask], eval_plot_df.loc[bad_mask, "pred_prob_bad"], s=18, label="True bad in holdout")
    ax.set_title(title)
    ax.set_ylabel("Probability")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_yearly_comparison_png(yearly_summary: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(yearly_summary))
    w = 0.38

    ax.bar(x - w/2, yearly_summary["original_flag_rate_pct"], width=w, label="Original bad rate (%)")
    ax.bar(x + w/2, yearly_summary["ml_flag_rate_pct"], width=w, label="ML bad rate (%)")

    ax.set_xticks(x)
    ax.set_xticklabels(yearly_summary["year"].astype(str).tolist())
    ax.set_ylabel("Bad rate (%)")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_flatline_detector_png(df: pd.DataFrame, output_path: Path, title: str) -> None:
    plot_df = df.copy()
    flat_mask = plot_df["flatline_flag"].fillna(False)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.plot(plot_df.index, plot_df["T_air"], color="steelblue", linewidth=1.2, alpha=0.8, label="Temperature")
    ax1.scatter(
        plot_df.index[flat_mask],
        plot_df.loc[flat_mask, "T_air"],
        color="darkorange", s=20, marker="s", label="Flatline flag"
    )
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.plot(plot_df.index, plot_df["rolling_std_6h"], color="black", linewidth=1.0, alpha=0.8, label="6h rolling std")
    ax2.axhline(0.15, color="red", linestyle="--", linewidth=1.2, label="6h flatline threshold")
    ax2.set_ylabel("6h rolling std")
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_time_series_comparison_png(ts_plot_df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(ts_plot_df.index, ts_plot_df["T_air"], color="steelblue", linewidth=1.3, alpha=0.6, label="Temperature")

    orig_bad_mask = (
        pd.to_numeric(ts_plot_df["original_label"], errors="coerce")
        .fillna(0)
        .astype(int) == 1
    )

    ml_bad_mask = (
        pd.to_numeric(ts_plot_df["ml_flag"], errors="coerce")
        .fillna(0)
        .astype(int) == 1
    )

    ax.scatter(
        ts_plot_df.index[orig_bad_mask],
        ts_plot_df.loc[orig_bad_mask, "T_air"],
        s=35, marker="x", color="black", linewidths=1.5, label="Original QC flags"
    )

    ax.scatter(
        ts_plot_df.index[ml_bad_mask],
        ts_plot_df.loc[ml_bad_mask, "T_air"],
        s=40, facecolors="none", edgecolors="red", linewidths=1.5, label="ML detected anomalies"
    )

    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Temperature (°C)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance_png(importances: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_df = importances.sort_values("importance", ascending=True)
    ax.barh(plot_df["feature"], plot_df["importance"])
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# PIPELINE
# =============================================================================
def run_pipeline(args: AppArgs, status_box=None) -> Dict[str, object]:
    def log(msg: str):
        if status_box is not None:
            status_box.write(msg)

    output_dir = make_output_dir(args.output_dir)
    write_template_csvs(output_dir)

    log("Loading station metadata...")
    stations = load_station_metadata()

    if args.state:
        stations = stations[stations["STATE"] == args.state.upper()].copy()

    if stations.empty:
        raise ValueError("No stations found after applying state filter.")

    if args.primary_filename:
        matches = stations[stations["FILENAME"] == args.primary_filename].copy()
    elif args.primary_name:
        mask = stations["STATION NAME"].str.contains(args.primary_name, case=False, na=False)
        matches = stations[mask].copy()
    else:
        raise ValueError("Provide either a primary station filename or a station-name substring.")

    if matches.empty:
        raise ValueError("No matching primary station found.")

    primary_station = matches.iloc[0]
    years = tuple(range(args.start_year, args.end_year + 1))

    log(f"Loading primary station data: {primary_station['STATION NAME']}")
    df_primary = load_station_years(primary_station["FILENAME"], years)
    if df_primary.empty:
        raise ValueError("No primary station data found for selected years.")

    raw_year_counts = (
        df_primary.groupby(df_primary.index.year)
        .size()
        .reset_index(name="n_raw_samples")
        .rename(columns={"DATE": "year"})
    )
    raw_year_counts.to_csv(output_dir / "raw_loaded_year_counts.csv", index=False)

    log("Building climatology and dynamic rule features...")
    primary_clim = compute_station_climatology(df_primary)
    df_primary = attach_climatology(df_primary, primary_clim)
    df_primary = build_dynamic_rule_features(df_primary)

    metadata_events = None
    if args.metadata_events_csv and Path(args.metadata_events_csv).exists():
        metadata_events = pd.read_csv(args.metadata_events_csv)
    df_primary = attach_metadata_event_flag(df_primary, metadata_events, window_days=3)

    log("Finding and evaluating neighbors...")
    candidate_neighbors = find_candidate_neighbors(
        primary_station,
        stations,
        max_candidates=10,
        max_distance_km=args.max_distance_km,
        max_elev_diff=args.max_elev_diff
    )

    neighbor_dict: Dict[str, dict] = {}
    neighbor_rows = []

    if not candidate_neighbors.empty:
        for _, nbr in candidate_neighbors.iterrows():
            ndf = load_station_years(nbr["FILENAME"], years)
            if ndf.empty:
                continue

            r, overlap = compute_pairwise_climatology_correlation(df_primary, ndf)
            neighbor_rows.append({
                "name": nbr["STATION NAME"],
                "filename": nbr["FILENAME"],
                "distance_km": nbr["distance_km"],
                "elev_diff_m": nbr["elev_diff_m"],
                "corr": r,
                "valid_overlap": overlap,
                "n_points": len(ndf)
            })

            neighbor_dict[nbr["FILENAME"]] = {
                "name": nbr["STATION NAME"],
                "distance_km": nbr["distance_km"],
                "elev_diff_m": nbr["elev_diff_m"],
                "correlation": r,
                "valid_overlap": overlap,
                "data": ndf
            }

    neighbor_summary = pd.DataFrame(neighbor_rows)
    if not neighbor_summary.empty:
        neighbor_summary["corr_filled"] = neighbor_summary["corr"].fillna(-999)

        selected_files = (
            neighbor_summary[neighbor_summary["corr_filled"] >= args.min_corr]
            .sort_values(["corr_filled", "distance_km"], ascending=[False, True])
            .head(args.max_neighbors)["filename"]
            .tolist()
        )

        if len(selected_files) < 2:
            fallback_candidates = neighbor_summary.copy()
            fallback_candidates["corr_rank"] = fallback_candidates["corr"].fillna(-1)
            fallback_candidates = fallback_candidates.sort_values(
                ["corr_rank", "distance_km"],
                ascending=[False, True],
                na_position="last"
            )
            selected_files = fallback_candidates.head(
                min(args.max_neighbors, len(fallback_candidates))
            )["filename"].tolist()
            log("Too few neighbors met correlation threshold. Using best available fallback neighbors.")
        elif len(selected_files) < 3:
            log("Fewer than 3 suitable neighbors met threshold. Spatial QC will be weaker.")

        neighbor_dict = {k: v for k, v in neighbor_dict.items() if k in selected_files}
    else:
        log("No usable neighbor summary available.")

    neighbor_summary.to_csv(output_dir / "neighbor_summary.csv", index=False)

    if len(neighbor_dict) < 2:
        log("Limited spatial neighbors available. QC relies more heavily on climatology and rule-based diagnostics.")
    else:
        log(f"Spatial QC using {len(neighbor_dict)} neighboring stations.")

    log("Building spatial QC features...")
    if len(neighbor_dict) > 0:
        df_primary, neighbor_cols = merge_neighbor_series(df_primary, neighbor_dict)
        df_primary = build_spatial_qc(
            df_primary,
            neighbor_cols,
            min_neighbors_required=min(2, len(neighbor_dict))
        )
    else:
        df_primary = build_spatial_qc(df_primary, [], min_neighbors_required=2)

    log("Applying advanced rule-based QC...")
    df_primary = apply_advanced_rule_qc(df_primary)

    log("Running auxiliary anomaly model...")
    df_primary["aux_iforest_flag"] = run_auxiliary_isolation_forest(
        df_primary,
        contamination=args.aux_contamination
    )

    log("Generating silver labels...")
    df_primary = generate_silver_labels(df_primary)

    debug_counts = df_primary.attrs.get("silver_debug_counts", {})
    pd.DataFrame([debug_counts]).to_csv(output_dir / "silver_label_debug_counts.csv", index=False)

    if args.expert_labels_csv and Path(args.expert_labels_csv).exists():
        expert_df = pd.read_csv(args.expert_labels_csv)
        expert_df["DATE"] = pd.to_datetime(expert_df["DATE"], errors="coerce")
        expert_df = expert_df.dropna(subset=["DATE"]).set_index("DATE").sort_index()

        if args.expert_label_col not in expert_df.columns:
            raise ValueError(f"Column '{args.expert_label_col}' not found in expert labels CSV")

        df_primary = df_primary.join(expert_df[[args.expert_label_col]], how="left")
    else:
        df_primary[args.expert_label_col] = np.nan

    if args.train_ml_on == "silver":
        training_label_col = "silver_label"
    elif args.train_ml_on == "expert":
        training_label_col = args.expert_label_col
    else:
        df_primary["training_label"] = df_primary[args.expert_label_col]
        fill_mask = df_primary["training_label"].isna()
        df_primary.loc[fill_mask, "training_label"] = df_primary.loc[fill_mask, "silver_label"]
        training_label_col = "training_label"

    comparison_df = pd.DataFrame(index=df_primary.index)
    comparison_df["Temperature"] = df_primary["T_air"]
    comparison_df["season"] = df_primary["season"]
    comparison_df["event_type"] = df_primary["event_type"]
    comparison_df["rule_flag"] = df_primary["rule_flag"].astype(int)
    comparison_df["rule_reason"] = df_primary["rule_reason"]
    comparison_df["aux_iforest_flag"] = df_primary["aux_iforest_flag"].astype(int)
    comparison_df["silver_label"] = df_primary["silver_label"]
    comparison_df["silver_reason"] = df_primary["silver_reason"]
    comparison_df["review_priority"] = df_primary["review_priority"]
    comparison_df["metadata_event_flag"] = df_primary["metadata_event_flag"]
    comparison_df["available_neighbors"] = df_primary["available_neighbors"]
    comparison_df["neighbor_neighbor_spread"] = df_primary["neighbor_neighbor_spread"]
    comparison_df["spatial_anom_diff"] = df_primary["spatial_anom_diff"]
    comparison_df["spatial_thresh"] = df_primary["spatial_thresh"]
    comparison_df["far_outside_seasonal_bounds"] = df_primary["far_outside_seasonal_bounds"].astype(int)
    comparison_df["flatline_flag"] = df_primary["flatline_flag"].astype(int)
    comparison_df["spatial_sustained_flag"] = df_primary["spatial_sustained_flag"].astype(int)
    comparison_df["implausible_jump_no_neighbor_support"] = df_primary["implausible_jump_no_neighbor_support"].astype(int)
    comparison_df["expert_label"] = df_primary[args.expert_label_col]
    comparison_df["training_label"] = df_primary[training_label_col]
    comparison_df["ml_prob_bad"] = np.nan
    comparison_df["ml_flag"] = np.nan

    log("Preparing ML data...")
    ml_features = prepare_ml_features(df_primary)
    train_df = ml_features.join(df_primary[[training_label_col]], how="left")
    train_df = train_df.dropna(subset=[training_label_col]).copy()

    ml_trained = False
    holdout_metrics_df = pd.DataFrame()
    yearly_summary = pd.DataFrame()
    importances = pd.DataFrame()

    if train_df.empty:
        log("No labels available for supervised ML training.")
    else:
        train_df[training_label_col] = train_df[training_label_col].astype(int)
        train_df = train_df.sort_index()
        train_df["year"] = train_df.index.year

        unique_years = sorted(train_df["year"].dropna().unique().tolist())
        year_counts = train_df.groupby("year").size().reset_index(name="n_labeled_samples")
        year_counts.to_csv(output_dir / "usable_labeled_year_counts.csv", index=False)

        log(f"Requested years: {args.start_year}-{args.end_year}")
        log(f"Years with usable labeled samples: {unique_years}")

        if len(unique_years) < 2:
            log("Need at least 2 years with labels for whole-year train/test split.")
        elif train_df[training_label_col].nunique() < 2:
            log("Training labels contain only one class. ML training skipped.")
        else:
            if args.split_method == "latest_years":
                n_test_years = args.n_test_years
                if n_test_years is None:
                    n_test_years = max(1, int(round(len(unique_years) * 0.2)))
                n_test_years = min(n_test_years, len(unique_years) - 1)
                test_years = unique_years[-n_test_years:]
                train_years = unique_years[:-n_test_years]
            else:
                n_test_years = max(1, int(math.ceil(len(unique_years) * 0.2)))
                test_years = unique_years[-n_test_years:]
                train_years = unique_years[:-n_test_years]

            log(f"Training years: {train_years}")
            log(f"Testing years: {test_years}")

            X = train_df.drop(columns=[training_label_col, "year"]).copy()
            y = train_df[training_label_col].copy()

            train_mask = train_df["year"].isin(train_years)
            test_mask = train_df["year"].isin(test_years)

            X_train = X.loc[train_mask].copy()
            X_test = X.loc[test_mask].copy()
            y_train = y.loc[train_mask].copy()
            y_test = y.loc[test_mask].copy()

            if len(X_train) == 0 or len(X_test) == 0:
                log("Empty train or test split after year filtering.")
            elif y_train.nunique() < 2 or y_test.nunique() < 2:
                log("Year-based split produced only one class in train or test.")
            else:
                ml_prob_threshold = args.ml_prob_threshold
                pipe = make_classifier(args.ml_model)

                log(f"Training model: {args.ml_model}")
                pipe.fit(X_train, y_train)

                if hasattr(pipe.named_steps["clf"], "predict_proba"):
                    y_pred_proba = pipe.predict_proba(X_test)[:, 1]
                else:
                    if hasattr(pipe.named_steps["clf"], "decision_function"):
                        dec = pipe.decision_function(X_test)
                        y_pred_proba = (dec - dec.min()) / (dec.max() - dec.min() + 1e-12)
                    else:
                        y_pred_proba = pipe.predict(X_test).astype(float)

                y_pred = (y_pred_proba >= ml_prob_threshold).astype(int)

                if hasattr(pipe.named_steps["clf"], "predict_proba"):
                    full_pred_proba = pipe.predict_proba(ml_features)[:, 1]
                else:
                    if hasattr(pipe.named_steps["clf"], "decision_function"):
                        full_dec = pipe.decision_function(ml_features)
                        full_pred_proba = (full_dec - full_dec.min()) / (full_dec.max() - full_dec.min() + 1e-12)
                    else:
                        full_pred_proba = pipe.predict(ml_features).astype(float)

                df_primary["ml_prob_bad"] = full_pred_proba
                df_primary["ml_flag"] = (df_primary["ml_prob_bad"] >= ml_prob_threshold).astype(int)

                comparison_df["ml_prob_bad"] = df_primary["ml_prob_bad"]
                comparison_df["ml_flag"] = df_primary["ml_flag"]

                ml_trained = True

                cm = confusion_matrix(y_test, y_pred)
                kappa = cohen_kappa_score(y_test, y_pred)
                mcc = matthews_corrcoef(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average="binary", zero_division=0
                )

                holdout_metrics_df = pd.DataFrame([{
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "mcc": mcc,
                    "kappa": kappa,
                    "holdout_samples": len(y_test),
                    "holdout_bad_rate_pct": y_test.mean() * 100,
                    "holdout_start": X_test.index.min(),
                    "holdout_end": X_test.index.max(),
                    "train_years": ",".join(map(str, train_years)),
                    "test_years": ",".join(map(str, test_years)),
                    "ml_model": args.ml_model,
                    "ml_prob_threshold": ml_prob_threshold,
                    "training_label_source": training_label_col
                }])
                holdout_metrics_df.to_csv(output_dir / "ml_holdout_metrics.csv", index=False)

                classif_text = classification_report(y_test, y_pred, zero_division=0)
                with open(output_dir / "ml_classification_report.txt", "w", encoding="utf-8") as f:
                    f.write(classif_text)

                save_confusion_matrix_png(
                    cm,
                    output_dir / "ml_confusion_matrix.png",
                    "Supervised ML Confusion Matrix (Year-Based Holdout)"
                )

                eval_plot_df = pd.DataFrame({
                    "time": X_test.index,
                    "true_label": y_test.values,
                    "pred_label": y_pred,
                    "pred_prob_bad": y_pred_proba,
                    "ml_prob_threshold": ml_prob_threshold
                }).set_index("time")
                eval_plot_df.to_csv(output_dir / "holdout_predictions.csv")

                save_probability_plot(
                    eval_plot_df,
                    output_dir / "holdout_probability_plot.png",
                    "Holdout Years: Predicted Probability of Bad Observations"
                )

                plot_df = df_primary.copy()
                plot_df["original_label"] = df_primary[training_label_col]

                yearly_summary = pd.DataFrame(index=sorted(plot_df.index.year.unique()))
                yearly_summary["n_obs"] = plot_df.groupby(plot_df.index.year).size()
                yearly_summary["original_flag_rate_pct"] = (
                    plot_df.groupby(plot_df.index.year)["original_label"].mean() * 100
                )
                yearly_summary["ml_flag_rate_pct"] = (
                    plot_df.groupby(plot_df.index.year)["ml_flag"].mean() * 100
                )
                yearly_summary = yearly_summary.reset_index().rename(columns={"index": "year"})
                yearly_summary.to_csv(output_dir / "yearly_bad_rate_comparison.csv", index=False)

                save_yearly_comparison_png(
                    yearly_summary,
                    output_dir / "yearly_bad_rate_comparison.png",
                    "Yearly Bad-Rate Comparison"
                )

                ts_plot_df = df_primary.copy()
                ts_plot_df["original_label"] = df_primary[training_label_col]

                save_time_series_comparison_png(
                    ts_plot_df,
                    output_dir / "time_series_original_vs_ml.png",
                    "Temperature and Flag Comparison"
                )

                save_zoom_time_series_figures(
                    ts_plot_df,
                    output_dir,
                    base_title="Temperature and Flag Comparison"
                )

                save_flatline_detector_png(
                    df_primary,
                    output_dir / "flatline_detector.png",
                    "Flatline Detector: Temperature and Rolling Variability"
                )

                save_spike_detector_png(
                    df_primary,
                    output_dir / "spike_detector.png",
                    "Spike Detector: Temperature and Hourly Change"
                )

                save_climatology_zscore_png(
                    df_primary,
                    output_dir / "climatology_zscore.png",
                    "Observed Temperature vs Climatology and Z-Score"
                )

                save_spatial_consistency_png(
                    df_primary,
                    output_dir / "spatial_consistency.png",
                    "Spatial Consistency Check: Primary vs Neighbor Anomalies"
                )

                save_daily_flag_counts_png(
                    df_primary,
                    output_dir / "daily_flag_counts.png",
                    "Daily Count of Rule-Based QC Flags"
                )

                clf = pipe.named_steps["clf"]
                feature_names = X_train.columns.tolist()

                if hasattr(clf, "feature_importances_"):
                    importances_vals = clf.feature_importances_
                    min_len = min(len(feature_names), len(importances_vals))
                    importances = pd.DataFrame({
                        "feature": feature_names[:min_len],
                        "importance": importances_vals[:min_len]
                    }).sort_values("importance", ascending=False).head(20)

                    importances.to_csv(output_dir / "feature_importance_top20.csv", index=False)
                    save_feature_importance_png(
                        importances,
                        output_dir / "feature_importance_top20.png",
                        f"Top 20 Feature Importances ({args.ml_model})"
                    )

                elif hasattr(clf, "coef_"):
                    coef = np.ravel(clf.coef_)
                    min_len = min(len(feature_names), len(coef))
                    importances = pd.DataFrame({
                        "feature": feature_names[:min_len],
                        "importance": np.abs(coef[:min_len])
                    }).sort_values("importance", ascending=False).head(20)

                    importances.to_csv(output_dir / "feature_importance_top20.csv", index=False)
                    save_feature_importance_png(
                        importances,
                        output_dir / "feature_importance_top20.png",
                        f"Top 20 Absolute Coefficients ({args.ml_model})"
                    )

    if ml_trained:
        comparison_df["comparison_type"] = "Neither"
        comparison_df.loc[(comparison_df["rule_flag"] == 1) & (comparison_df["ml_flag"] == 1), "comparison_type"] = "Both"
        comparison_df.loc[(comparison_df["rule_flag"] == 1) & (comparison_df["ml_flag"] == 0), "comparison_type"] = "Rule Only"
        comparison_df.loc[(comparison_df["rule_flag"] == 0) & (comparison_df["ml_flag"] == 1), "comparison_type"] = "ML Only"
    else:
        comparison_df["comparison_type"] = np.where(comparison_df["rule_flag"] == 1, "Rule Flagged", "Rule Clean")

    comparison_df.to_csv(output_dir / "qc_output_table.csv")

    silver_eval = comparison_df.dropna(subset=["silver_label"]).copy()
    if not silver_eval.empty and silver_eval["silver_label"].nunique() >= 2:
        y_true = silver_eval["silver_label"].astype(int)
        y_rule = silver_eval["rule_flag"].astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(
            y_true, y_rule, average="binary", zero_division=0
        )

        silver_rule_metrics = pd.DataFrame([{
            "precision": pr,
            "recall": rc,
            "f1": f1
        }])
        silver_rule_metrics.to_csv(output_dir / "rule_vs_silver_metrics.csv", index=False)

        if ml_trained:
            y_ml = silver_eval["ml_flag"].astype(int)
            pr, rc, f1, _ = precision_recall_fscore_support(
                y_true, y_ml, average="binary", zero_division=0
            )
            silver_ml_metrics = pd.DataFrame([{
                "precision": pr,
                "recall": rc,
                "f1": f1
            }])
            silver_ml_metrics.to_csv(output_dir / "ml_vs_silver_metrics.csv", index=False)

        season_rule = seasonal_metrics(silver_eval, "silver_label", "rule_flag")
        season_rule.to_csv(output_dir / "seasonal_metrics_rule_vs_silver.csv", index=False)

        if ml_trained:
            season_ml = seasonal_metrics(silver_eval, "silver_label", "ml_flag")
            season_ml.to_csv(output_dir / "seasonal_metrics_ml_vs_silver.csv", index=False)

        event_rule = event_metrics(silver_eval, "silver_label", "rule_flag")
        event_rule.to_csv(output_dir / "event_metrics_rule_vs_silver.csv", index=False)

        if ml_trained:
            event_ml = event_metrics(silver_eval, "silver_label", "ml_flag")
            event_ml.to_csv(output_dir / "event_metrics_ml_vs_silver.csv", index=False)

    log("Done.")

    return {
        "output_dir": output_dir,
        "primary_station": primary_station,
        "comparison_df": comparison_df,
        "df_primary": df_primary,
        "neighbor_summary": neighbor_summary,
        "holdout_metrics_df": holdout_metrics_df,
        "yearly_summary": yearly_summary,
        "importances": importances,
        "ml_trained": ml_trained
    }


# =============================================================================
# SIDEBAR UI
# =============================================================================
with st.sidebar:
    st.header("Inputs")

    state = st.text_input("State code", value="NM").strip().upper()
    station_mode = st.radio("Primary station selection mode", ["Filename", "Name contains"], index=0)

    primary_filename = None
    primary_name = None
    if station_mode == "Filename":
        primary_filename = st.text_input("Primary filename", value="72253923044.csv").strip() or None
    else:
        primary_name = st.text_input("Primary station name contains", value="HOBBS").strip() or None

    c1, c2 = st.columns(2)
    with c1:
        start_year = st.number_input("Start year", min_value=1900, max_value=2100, value=2007, step=1)
    with c2:
        end_year = st.number_input("End year", min_value=1900, max_value=2100, value=2012, step=1)

    st.subheader("Neighbor settings")
    max_neighbors = st.number_input("Max neighbors", min_value=1, max_value=10, value=3, step=1)
    max_distance_km = st.number_input("Max distance (km)", min_value=1.0, max_value=500.0, value=30.0, step=1.0)
    max_elev_diff = st.number_input("Max elevation diff (m)", min_value=0.0, max_value=5000.0, value=300.0, step=10.0)
    min_corr = st.slider("Minimum neighbor correlation", min_value=0.0, max_value=1.0, value=0.40, step=0.01)

    st.subheader("Anomaly / ML settings")
    aux_contamination = st.slider("Isolation Forest contamination", min_value=0.001, max_value=0.20, value=0.02, step=0.001)
    train_ml_on = st.selectbox("Train ML on", ["silver", "expert", "expert_else_silver"], index=2)
    ml_model = st.selectbox(
        "ML model",
        ["random_forest", "extra_trees", "gradient_boosting", "logistic_regression"],
        index=0
    )
    split_method = st.selectbox("Split method", ["latest_years", "eighty_twenty_years"], index=0)
    n_test_years = st.number_input("Number of test years (optional for latest_years)", min_value=0, max_value=20, value=0, step=1)
    ml_prob_threshold = st.slider("ML bad-probability threshold", min_value=0.0, max_value=1.0, value=0.80, step=0.01)

    st.subheader("Optional uploads")
    metadata_upload = st.file_uploader("Metadata events CSV", type=["csv"])
    expert_upload = st.file_uploader("Expert labels CSV", type=["csv"])
    expert_label_col = st.text_input("Expert label column", value="expert_label")

    run_button = st.button("Run QC Pipeline", type="primary", use_container_width=True)

# =============================================================================
# APP EXECUTION
# =============================================================================
if run_button:
    if end_year < start_year:
        st.error("End year must be greater than or equal to start year.")
        st.stop()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        metadata_path = None
        expert_path = None

        if metadata_upload is not None:
            metadata_path = str(tmpdir_path / "metadata_events.csv")
            with open(metadata_path, "wb") as f:
                f.write(metadata_upload.getbuffer())

        if expert_upload is not None:
            expert_path = str(tmpdir_path / "expert_labels.csv")
            with open(expert_path, "wb") as f:
                f.write(expert_upload.getbuffer())

        output_dir = str(tmpdir_path / "results")

        args = AppArgs(
            output_dir=output_dir,
            state=state,
            primary_filename=primary_filename,
            primary_name=primary_name,
            start_year=int(start_year),
            end_year=int(end_year),
            max_neighbors=int(max_neighbors),
            max_distance_km=float(max_distance_km),
            max_elev_diff=float(max_elev_diff),
            min_corr=float(min_corr),
            aux_contamination=float(aux_contamination),
            metadata_events_csv=metadata_path,
            expert_labels_csv=expert_path,
            expert_label_col=expert_label_col,
            train_ml_on=train_ml_on,
            ml_model=ml_model,
            split_method=split_method,
            n_test_years=None if int(n_test_years) == 0 else int(n_test_years),
            ml_prob_threshold=float(ml_prob_threshold),
        )

        st.info("Running pipeline... this can take some time depending on years and neighbor availability.")

        status_box = st.container()

        try:
            results = run_pipeline(args, status_box=status_box)
            outdir = results["output_dir"]

            st.success(f"Done. Outputs written to: {outdir}")

            primary_station = results["primary_station"]
            comparison_df = results["comparison_df"]
            neighbor_summary = results["neighbor_summary"]
            holdout_metrics_df = results["holdout_metrics_df"]
            yearly_summary = results["yearly_summary"]
            importances = results["importances"]
            ml_trained = results["ml_trained"]

            st.subheader("Primary station")
            st.write({
                "station_name": primary_station["STATION NAME"],
                "filename": primary_station["FILENAME"],
                "state": primary_station.get("STATE", ""),
                "lat": primary_station["LAT"],
                "lon": primary_station["LON"],
                "elev_m": primary_station.get("ELEV_M", np.nan),
            })

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total observations", f"{len(comparison_df):,}")
            with col2:
                st.metric("Rule flags", f"{int(comparison_df['rule_flag'].fillna(0).sum()):,}")
            with col3:
                if ml_trained and "ml_flag" in comparison_df.columns:
                    st.metric("ML flags", f"{int(pd.to_numeric(comparison_df['ml_flag'], errors='coerce').fillna(0).sum()):,}")
                else:
                    st.metric("ML trained", "No")

            st.subheader("Neighbor summary")
            if neighbor_summary is not None and not neighbor_summary.empty:
                st.dataframe(neighbor_summary, use_container_width=True)
            else:
                st.warning("No usable neighbors found.")

            st.subheader("QC output table preview")
            st.dataframe(comparison_df.head(200), use_container_width=True)

            if holdout_metrics_df is not None and not holdout_metrics_df.empty:
                st.subheader("ML holdout metrics")
                st.dataframe(holdout_metrics_df, use_container_width=True)

            if yearly_summary is not None and not yearly_summary.empty:
                st.subheader("Yearly bad-rate comparison")
                st.dataframe(yearly_summary, use_container_width=True)

            if importances is not None and not importances.empty:
                st.subheader("Top feature importances")
                st.dataframe(importances, use_container_width=True)

            st.subheader("Figures")
            png_files = sorted(outdir.glob("*.png"))
            if png_files:
                for png in png_files:
                    st.markdown(f"**{png.name}**")
                    st.image(str(png), use_container_width=True)
            else:
                st.info("No PNG figures were generated.")

            st.subheader("Downloads")

            qc_csv = outdir / "qc_output_table.csv"
            if qc_csv.exists():
                with open(qc_csv, "rb") as f:
                    st.download_button(
                        "Download qc_output_table.csv",
                        data=f.read(),
                        file_name="qc_output_table.csv",
                        mime="text/csv"
                    )

            zip_buffer = zip_directory(outdir)
            st.download_button(
                "Download all outputs as ZIP",
                data=zip_buffer,
                file_name="advanced_temperature_qc_outputs.zip",
                mime="application/zip"
            )

        except Exception as e:
            st.error(f"Pipeline failed: {e}")
else:
    st.caption("Set parameters in the sidebar, then click **Run QC Pipeline**.")
