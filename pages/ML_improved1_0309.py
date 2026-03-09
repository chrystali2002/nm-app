import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr
from geopy.distance import geodesic
from typing import Dict, Optional, Tuple, List
import requests
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Advanced Temperature QC with Silver Labels",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Advanced Temperature QC with Silver Labels")
st.markdown("""
This dashboard performs **advanced rule-based quality control**, **automatic silver-label generation**, and **optional supervised machine-learning (ML) quality control** for hourly station temperature data.

---

## Why silver labels?

Expert-reviewed **true quality-control labels are not yet available** for this dataset.

To enable model development and anomaly screening, the system generates **silver labels** — high-confidence **proxy labels** derived from physical rules, climatology, and spatial consistency checks.

Silver labels are **not absolute truth**.  
They are intended for:

• anomaly screening  
• model bootstrapping  
• identifying cases for expert review  

---

## Silver label logic

### High-confidence bad (1)

An observation is labeled **bad** when strong evidence suggests it is physically or instrumentally inconsistent.

Examples include:

• temperature far outside **station-season climatological percentiles**  
• **sustained flatline behavior** indicating sensor persistence  
• **large sustained disagreement with neighboring stations** when neighbors are available  
• **implausible jumps** not supported by neighboring stations  
• observations near **known metadata events** (e.g., station relocation or sensor change) that are unsupported by surrounding stations  

These represent **high-confidence anomalies likely related to measurement issues or sensor behavior**.

---

### High-confidence good (0)

An observation is labeled **good** when it satisfies conservative physical and climatological checks.

Typical conditions include:

• within **seasonal climatological bounds**  
• **no rule-based QC flags**  
• consistent with **available spatial context** when neighboring stations exist  
• **not near metadata events**  
• **not flagged by multiple anomaly detection methods**

If spatial neighbors are unavailable or insufficient, the system can still assign good labels using **climatology consistency and rule-based stability checks**.

---

### Uncertain (unlabeled)

Observations that do not meet strict criteria for either class remain **unlabeled**.

These cases are ideal candidates for:

• expert inspection  
• manual quality control  
• future training-label refinement  

---

## Spatial consistency checks

When sufficient neighboring stations are available, the system performs **multi-station spatial consistency analysis**.

This includes:

• comparison against **weighted neighbor median anomalies**  
• verification that neighboring stations agree with each other  
• detection of **sustained disagreement events**

If fewer suitable neighbors are available, the system automatically **reduces reliance on spatial checks** and uses climatology and rule-based diagnostics instead.

---

## Bootstrapping ML models

Silver labels allow the system to **train supervised ML models** that learn patterns associated with high-confidence good and bad observations.

However:

**ML models are evaluated against the silver labels themselves**, not against expert truth labels.

Therefore extremely high metrics (e.g., near-perfect precision or Kappa) indicate that the ML model successfully reproduces the silver labeling logic — **not that the ML model has been fully validated against real-world errors**.

---

## Scientific purpose

This workflow provides a **practical first step toward operational quality control**:

1. Apply physical and statistical QC rules  
2. Generate **silver proxy labels**  
3. Train ML models to assist detection  
4. Prioritize uncertain observations for expert review  

As expert-reviewed labels become available, they can replace silver labels to produce **fully validated ML-based quality-control systems**.
""")


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
# UTILITY HELPERS
# =============================================================================
def safe_percentile(x: pd.Series, q: float, min_n: int = 24) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if len(vals) < min_n:
        return np.nan
    return float(np.nanpercentile(vals, q))


def season_from_month(month: int) -> str:
    return SEASON_MAP.get(int(month), "Unknown")


def detect_event_type(row: pd.Series) -> str:
    """
    Simple event tagging for evaluation.
    This is still heuristic and should be refined if wind/pressure/humidity are available.
    """
    if bool(row.get("wind_low_flag", False)) and bool(row.get("diurnal_range_low_flag", False)):
        return "Calm Night"
    if bool(row.get("front_like_flag", False)):
        return "Frontal Passage"
    if bool(row.get("cold_surge_flag", False)):
        return "Cold Surge"
    return "Other"


def robust_bool_series(index) -> pd.Series:
    return pd.Series(False, index=index, dtype=bool)


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    return out.sort_index()


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

        # Optional wind if available
        if "WND" in df.columns:
            wnd_split = df["WND"].astype(str).str.split(",", expand=True)
            if wnd_split.shape[1] >= 4:
                # wind speed in tenths m/s in ISD WND field
                df["WND_speed_raw"] = pd.to_numeric(wnd_split[3], errors="coerce")
                df["wind_speed_ms"] = df["WND_speed_raw"].replace(9999, np.nan) / 10.0
            else:
                df["wind_speed_ms"] = np.nan
        else:
            df["wind_speed_ms"] = np.nan

        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
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


def compute_pairwise_climatology_correlation(primary_df: pd.DataFrame, neighbor_df: pd.DataFrame) -> float:
    common = primary_df.index.intersection(neighbor_df.index)
    if len(common) < 200:
        return np.nan

    p = pd.to_numeric(primary_df.loc[common, "T_air"], errors="coerce")
    n = pd.to_numeric(neighbor_df.loc[common, "T_air"], errors="coerce")
    mask = p.notna() & n.notna()

    if mask.sum() < 200:
        return np.nan

    try:
        r, _ = pearsonr(p[mask], n[mask])
        return float(r)
    except Exception:
        return np.nan


def merge_neighbor_series(primary_df: pd.DataFrame, neighbor_dict: Dict[str, dict]) -> Tuple[pd.DataFrame, List[Tuple[str, dict]]]:
    out = primary_df.copy()
    neighbor_cols: List[Tuple[str, dict]] = []

    for i, (fname, info) in enumerate(neighbor_dict.items(), start=1):
        ndf = info["data"][["T_air"]].rename(columns={"T_air": f"neighbor_{i}_T"})
        out = out.join(ndf, how="left")
        neighbor_cols.append((f"neighbor_{i}_T", info))

    return out, neighbor_cols


# =============================================================================
# EVENT FEATURES AND RULE FEATURES
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

    # Dynamic percentile bounds
    out["within_seasonal_bounds"] = (
        (out["T_air"] >= out["clim_p05"]) &
        (out["T_air"] <= out["clim_p95"])
    )

    out["far_outside_seasonal_bounds"] = (
        (out["T_air"] < out["clim_p01"] - 2.0) |
        (out["T_air"] > out["clim_p99"] + 2.0)
    )

    # Season-aware spike thresholds
    seasonal_spike = {
        "Winter": 10.0,
        "Spring": 8.0,
        "Summer": 6.5,
        "Fall": 8.0
    }
    out["spike_thresh_1h"] = out["season"].map(seasonal_spike).astype(float)
    out["spike_dynamic_flag"] = out["dT_1h"].abs() > out["spike_thresh_1h"]

    # Flatline / persistence
    out["flatline_flag"] = (
        (out["rolling_std_6h"] < 0.15) &
        (out["rolling_std_24h"] < 0.25)
    )

    # Event helpers
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

    # -------------------------------------------------------------------------
    # No usable neighbors
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Build weighted neighbor fields
    # -------------------------------------------------------------------------
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

    # Weighted mean neighbor temperature
    weights = np.array(robust_weights, dtype=float)
    weighted_num = stacked.mul(weights, axis=1).sum(axis=1, skipna=True)
    weighted_den = stacked.notna().mul(weights, axis=1).sum(axis=1, skipna=True)
    out["neighbor_weighted_mean"] = weighted_num / weighted_den.replace(0, np.nan)

    # Robust ensemble summaries
    out["neighbor_median"] = stacked.median(axis=1)
    out["neighbor_neighbor_spread"] = stacked.std(axis=1)

    # -------------------------------------------------------------------------
    # Spatial anomaly diagnostics
    # -------------------------------------------------------------------------
    out["primary_anom"] = out["T_air"] - out["rolling_mean_24h"]
    out["neighbor_median_anom"] = (
        out["neighbor_median"] - out["neighbor_median"].rolling(24, min_periods=12).mean()
    )
    out["spatial_diff_to_median"] = out["T_air"] - out["neighbor_median"]
    out["spatial_anom_diff"] = out["primary_anom"] - out["neighbor_median_anom"]

    mean_corr = np.nanmean(corr_vals) if len(corr_vals) > 0 else np.nan
    if np.isnan(mean_corr):
        mean_corr = 0.5

    # Higher correlation -> tighter threshold
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

    # -------------------------------------------------------------------------
    # Implausible jump not supported by neighbors
    # -------------------------------------------------------------------------
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
        return pd.DataFrame(columns=["DATE", "event_type"])

    out = metadata_events.copy()
    out["DATE"] = pd.to_datetime(out["DATE"], errors="coerce")
    out = out.dropna(subset=["DATE"]).sort_values("DATE")

    if "event_type" not in out.columns:
        out["event_type"] = "metadata_event"

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
# UNSUPERVISED AUXILIARY ANOMALY MODEL
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
# SILVER LABEL GENERATION
# =============================================================================
def generate_silver_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Silver labels:
      1 = high-confidence bad
      0 = high-confidence good
      NaN = uncertain / unlabeled

    This version is intentionally more permissive for the GOOD class so that
    supervised ML can train on both classes while still keeping labels conservative.
    """
    out = df.copy()

    # -------------------------------------------------------------------------
    # Safety defaults in case some columns are missing
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # BAD LABEL LOGIC (kept strong / conservative)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # MULTI-METHOD ANOMALY VOTE
    # -------------------------------------------------------------------------
    out["anomaly_vote_count"] = (
        out["rule_flag"].fillna(False).astype(int) +
        out["aux_iforest_flag"].fillna(False).astype(int) +
        out["spatial_instant_flag"].fillna(False).astype(int)
    )

    # -------------------------------------------------------------------------
    # GOOD LABEL LOGIC (relaxed so you can actually get class 0)
    #
    # Changes from before:
    # 1. neighbors >= 2 instead of 3
    # 2. allow missing spatial data
    # 3. allow slightly looser spatial agreement
    # 4. allow slightly looser neighbor spread
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # INITIAL LABEL ASSIGNMENT
    # -------------------------------------------------------------------------
    out["silver_label"] = np.nan
    out.loc[high_conf_bad, "silver_label"] = 1.0
    out.loc[high_conf_good & ~high_conf_bad, "silver_label"] = 0.0

    # -------------------------------------------------------------------------
    # FALLBACK CLEAN SAMPLE
    #
    # If no good labels survive, create a conservative bootstrap good sample from:
    # - within seasonal bounds
    # - no rule flag
    # - no metadata event
    # This keeps the pipeline usable for ML bootstrapping.
    # -------------------------------------------------------------------------
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

            # Choose a moderate fallback size:
            # at least 200 if available, otherwise up to number of bad labels, capped at 1000
            fallback_n = min(
                len(fallback_candidates),
                max(200, min(1000, n_bad if n_bad > 0 else 500))
            )

            fallback_idx = fallback_candidates.sample(
                n=fallback_n,
                random_state=42
            ).index

            out.loc[fallback_idx, "silver_label"] = 0.0
            out.loc[fallback_idx, "silver_reason"] = "Fallback high-confidence good"

    # -------------------------------------------------------------------------
    # REASON STRINGS
    # -------------------------------------------------------------------------
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

        # keep fallback reason if already assigned
        existing_reason = row.get("silver_reason", None)
        if isinstance(existing_reason, str) and existing_reason.strip():
            return existing_reason

        return "High-confidence good"

    if "silver_reason" not in out.columns:
        out["silver_reason"] = ""

    needs_reason = out["silver_reason"].isna() | (out["silver_reason"].astype(str).str.strip() == "")
    out.loc[needs_reason, "silver_reason"] = out.loc[needs_reason].apply(_silver_reason, axis=1)

    # -------------------------------------------------------------------------
    # REVIEW PRIORITY
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # DEBUG COUNTS
    # -------------------------------------------------------------------------
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
# ML FEATURES AND TRAINING
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
# SIDEBAR
# =============================================================================
st.sidebar.header("🔧 Configuration")

stations = load_station_metadata()

state_code = st.sidebar.text_input("State Code", value="NM").upper().strip()
state_stations = stations[stations["STATE"] == state_code].copy()

if state_stations.empty:
    st.error(f"No stations found for state {state_code}")
    st.stop()

station_options = state_stations[
    ["USAF", "WBAN", "STATION NAME", "LAT", "LON", "ELEV_M", "FILENAME"]
].reset_index(drop=True)

primary_idx = st.sidebar.selectbox(
    "Primary station",
    options=range(len(station_options)),
    format_func=lambda x: f"{station_options.iloc[x]['STATION NAME']} ({station_options.iloc[x]['USAF']}-{station_options.iloc[x]['WBAN']})",
    index=0
)
primary_station = station_options.iloc[primary_idx]

year_range = st.sidebar.slider("Year Range", 2000, 2025, (2018, 2023))
years = tuple(range(year_range[0], year_range[1] + 1))

st.sidebar.subheader("Neighbor controls")
max_neighbors = st.sidebar.slider("Use top neighbors", 2, 4, 3)
max_distance_km = st.sidebar.selectbox("Max neighbor distance (km)", [10, 20, 30, 50, 100, 200], index=2)
max_elev_diff = st.sidebar.selectbox("Max elevation difference (m)", [100, 200, 300, 500, 800], index=2)
min_corr = st.sidebar.slider("Minimum correlation", 0.30, 0.95, 0.60, 0.05)

st.sidebar.subheader("Auxiliary anomaly model")
aux_contamination = st.sidebar.slider("Isolation Forest contamination", 0.005, 0.10, 0.02, 0.005)

st.sidebar.subheader("Optional expert inputs")
metadata_events_file = st.sidebar.file_uploader(
    "Upload metadata events CSV",
    type=["csv"],
    help="Optional. Must contain DATE and optional event_type columns."
)

expert_labels_file = st.sidebar.file_uploader(
    "Upload expert true labels CSV",
    type=["csv"],
    help="Optional. If available, must contain DATE and expert_label columns."
)

expert_label_col = st.sidebar.text_input("Expert label column", value="expert_label")

train_ml_on = st.sidebar.selectbox(
    "Train ML on",
    options=[
        "Silver labels only",
        "Expert labels only",
        "Expert labels if available, else silver labels"
    ],
    index=2
)

# =============================================================================
# LOAD DATA
# =============================================================================
st.header("📡 Loading Station Data")

with st.spinner("Loading primary station data..."):
    df_primary = load_station_years(primary_station["FILENAME"], years)

if df_primary.empty:
    st.error("No primary station data found for selected years.")
    st.stop()

st.success(f"Loaded {len(df_primary):,} records for primary station: {primary_station['STATION NAME']}")

# =============================================================================
# CLIMATOLOGY AND PRIMARY FEATURES
# =============================================================================
with st.spinner("Building station climatology and dynamic features..."):
    primary_clim = compute_station_climatology(df_primary)
    df_primary = attach_climatology(df_primary, primary_clim)
    df_primary = build_dynamic_rule_features(df_primary)

# =============================================================================
# OPTIONAL METADATA EVENTS
# =============================================================================
metadata_events = None
if metadata_events_file is not None:
    metadata_events = pd.read_csv(metadata_events_file)

df_primary = attach_metadata_event_flag(df_primary, metadata_events, window_days=3)

# =============================================================================
# FIND / LOAD NEIGHBORS
# =============================================================================
st.header("👥 Multi-Neighbor Spatial QC")

candidate_neighbors = find_candidate_neighbors(
    primary_station,
    station_options,
    max_candidates=10,
    max_distance_km=max_distance_km,
    max_elev_diff=max_elev_diff
)

neighbor_dict: Dict[str, dict] = {}

if candidate_neighbors.empty:
    st.warning("No candidate neighbors found under the chosen distance/elevation settings.")
else:
    rows = []
    with st.spinner("Loading candidate neighbors and estimating correlation..."):
        for _, nbr in candidate_neighbors.iterrows():
            ndf = load_station_years(nbr["FILENAME"], years)
            if ndf.empty:
                continue

            r = compute_pairwise_climatology_correlation(df_primary, ndf)

            rows.append({
                "name": nbr["STATION NAME"],
                "filename": nbr["FILENAME"],
                "distance_km": nbr["distance_km"],
                "elev_diff_m": nbr["elev_diff_m"],
                "corr": r,
                "n_points": len(ndf)
            })

            neighbor_dict[nbr["FILENAME"]] = {
                "name": nbr["STATION NAME"],
                "distance_km": nbr["distance_km"],
                "elev_diff_m": nbr["elev_diff_m"],
                "correlation": r,
                "data": ndf
            }

    neighbor_summary = pd.DataFrame(rows)

    if neighbor_summary.empty:
        st.warning("Neighbor files were not available.")
    else:
        neighbor_summary = neighbor_summary.sort_values(["corr", "distance_km"], ascending=[False, True], na_position="last")
        neighbor_summary["selected"] = (
            neighbor_summary["corr"].fillna(-999) >= min_corr
        )

        st.dataframe(neighbor_summary, use_container_width=True, hide_index=True)

        selected_files = neighbor_summary[neighbor_summary["selected"]].head(max_neighbors)["filename"].tolist()
        if len(selected_files) < 3:
            st.warning("Fewer than 3 suitable neighbors met the threshold. Spatial QC will be weaker.")

        neighbor_dict = {k: v for k, v in neighbor_dict.items() if k in selected_files}


# =============================================================================
# SPATIAL QC APPLICATION
# =============================================================================
if len(neighbor_dict) < 2:
    st.warning(
        "Limited spatial neighbors available — QC relies more heavily on climatology "
        "and rule-based diagnostics."
    )
else:
    st.info(f"Spatial QC using {len(neighbor_dict)} neighboring stations.")

if len(neighbor_dict) > 0:
    with st.spinner("Building weighted multi-neighbor spatial QC features..."):
        df_primary, neighbor_cols = merge_neighbor_series(df_primary, neighbor_dict)
        df_primary = build_spatial_qc(
            df_primary,
            neighbor_cols,
            min_neighbors_required=min(2, len(neighbor_dict))
        )
else:
    df_primary = build_spatial_qc(
        df_primary,
        [],
        min_neighbors_required=2
    )


# =============================================================================
# APPLY RULE-BASED QC
# =============================================================================
with st.spinner("Applying advanced rule-based QC..."):
    df_primary = apply_advanced_rule_qc(df_primary)

# =============================================================================
# AUXILIARY UNSUPERVISED FLAGS
# =============================================================================
with st.spinner("Running auxiliary anomaly detector..."):
    df_primary["aux_iforest_flag"] = run_auxiliary_isolation_forest(df_primary, contamination=aux_contamination)

# =============================================================================
# GENERATE SILVER LABELS
# =============================================================================
with st.spinner("Generating silver labels..."):
    df_primary = generate_silver_labels(df_primary)
debug_counts = df_primary.attrs.get("silver_debug_counts", {})

st.subheader("Silver Label Rule Diagnostics")
st.write("within_seasonal_bounds:", debug_counts.get("within_seasonal_bounds", "N/A"))
st.write("not rule_flag:", debug_counts.get("not_rule_flag", "N/A"))
st.write("available_neighbors >= 2:", debug_counts.get("available_neighbors_ge_2", "N/A"))
st.write("spatial acceptable:", debug_counts.get("spatial_acceptable", "N/A"))
st.write("neighbor spread acceptable:", debug_counts.get("neighbor_spread_acceptable", "N/A"))
st.write("metadata_event_flag == 0:", debug_counts.get("metadata_event_flag_eq_0", "N/A"))
st.write("anomaly_vote_count <= 1:", debug_counts.get("anomaly_vote_count_le_1", "N/A"))
st.write("high_conf_bad:", debug_counts.get("high_conf_bad", "N/A"))
st.write("high_conf_good:", debug_counts.get("high_conf_good", "N/A"))
st.write("silver_bad_final:", debug_counts.get("silver_bad_final", "N/A"))
st.write("silver_good_final:", debug_counts.get("silver_good_final", "N/A"))
st.write("silver_uncertain_final:", debug_counts.get("silver_uncertain_final", "N/A"))  

st.subheader("Silver Label Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Silver Bad (1)", int((df_primary["silver_label"] == 1).sum()))
c2.metric("Silver Good (0)", int((df_primary["silver_label"] == 0).sum()))
c3.metric("Uncertain", int(df_primary["silver_label"].isna().sum()))

st.info(
    "Silver labels are high-confidence proxy labels used because expert true labels are not yet available. "
    "They should guide screening and model bootstrapping, not be treated as final truth."
)

# =============================================================================
# OPTIONAL EXPERT LABELS
# =============================================================================
if expert_labels_file is not None:
    expert_df = pd.read_csv(expert_labels_file)
    expert_df["DATE"] = pd.to_datetime(expert_df["DATE"], errors="coerce")
    expert_df = expert_df.dropna(subset=["DATE"]).set_index("DATE").sort_index()

    if expert_label_col not in expert_df.columns:
        st.error(f"Column '{expert_label_col}' not found in expert labels file.")
        st.stop()

    df_primary = df_primary.join(expert_df[[expert_label_col]], how="left")
else:
    df_primary[expert_label_col] = np.nan

# =============================================================================
# TRAINING TARGET SELECTION
# =============================================================================
if train_ml_on == "Silver labels only":
    training_label_col = "silver_label"
elif train_ml_on == "Expert labels only":
    training_label_col = expert_label_col
else:
    # Prefer expert when present, otherwise silver
    df_primary["training_label"] = df_primary[expert_label_col]
    fill_mask = df_primary["training_label"].isna()
    df_primary.loc[fill_mask, "training_label"] = df_primary.loc[fill_mask, "silver_label"]
    training_label_col = "training_label"

# =============================================================================
# SUPERVISED ML
# =============================================================================
st.header("🤖 Supervised ML QC")

ml_enabled = st.checkbox("Train supervised ML model", value=True)

if ml_enabled:
    ml_features = prepare_ml_features(df_primary)

    train_df = ml_features.join(df_primary[[training_label_col]], how="left")
    train_df = train_df.dropna(subset=[training_label_col]).copy()

    if train_df.empty:
        st.warning("No labels available for supervised ML training.")
        df_primary["ml_flag"] = np.nan
        ml_trained = False

    else:
        train_df[training_label_col] = train_df[training_label_col].astype(int)
        train_df = train_df.sort_index()
        train_df["year"] = train_df.index.year

        if train_df[training_label_col].nunique() < 2:
            st.warning("Training labels contain only one class. ML training skipped.")
            df_primary["ml_flag"] = np.nan
            ml_trained = False

        else:
            st.subheader("Train/Test Split by Whole Years")

            split_method = st.radio(
                "Choose year-based split method",
                options=[
                    "Train on earliest years, test on latest years",
                    "80/20 split by selected years"
                ],
                index=0
            )

            unique_years = sorted(train_df["year"].dropna().unique().tolist())

            if len(unique_years) < 2:
                st.warning("Need at least 2 years with labels for year-based train/test split.")
                df_primary["ml_flag"] = np.nan
                ml_trained = False

            else:
                if split_method == "Train on earliest years, test on latest years":
                    n_test_years = st.slider(
                        "Number of latest years to use for testing",
                        min_value=1,
                        max_value=max(1, len(unique_years) - 1),
                        value=max(1, int(round(len(unique_years) * 0.2))),
                        step=1
                    )

                    test_years = unique_years[-n_test_years:]
                    train_years = unique_years[:-n_test_years]

                else:
                    # 80/20 split by selected years
                    n_test_years = max(1, int(np.ceil(len(unique_years) * 0.2)))
                    test_years = unique_years[-n_test_years:]
                    train_years = unique_years[:-n_test_years]

                st.info(
                    "ML evaluation uses a whole-year split. "
                    "This is more realistic than random hourly splitting because it reduces temporal leakage."
                )

                st.write(f"**Training years:** {train_years}")
                st.write(f"**Testing years:** {test_years}")

                if len(train_years) == 0 or len(test_years) == 0:
                    st.warning("Invalid year split. Adjust selected years.")
                    df_primary["ml_flag"] = np.nan
                    ml_trained = False
                else:
                    train_mask = train_df["year"].isin(train_years)
                    test_mask = train_df["year"].isin(test_years)

                    X = train_df.drop(columns=[training_label_col, "year"]).copy()
                    y = train_df[training_label_col].copy()

                    X_train = X.loc[train_mask].copy()
                    X_test = X.loc[test_mask].copy()
                    y_train = y.loc[train_mask].copy()
                    y_test = y.loc[test_mask].copy()

                    # Safety check: both train and test need both classes
                    if len(X_train) == 0 or len(X_test) == 0:
                        st.warning("Empty train or test split after year filtering.")
                        df_primary["ml_flag"] = np.nan
                        ml_trained = False

                    elif y_train.nunique() < 2 or y_test.nunique() < 2:
                        st.warning(
                            "Year-based split produced only one class in the training or testing years. "
                            "Try increasing the year range or generating more balanced labels."
                        )
                        df_primary["ml_flag"] = np.nan
                        ml_trained = False

                    else:
                        # Show class balance
                        c1, c2 = st.columns(2)
                        with c1:
                            st.write("Training label distribution:")
                            st.write(y_train.value_counts(dropna=False).sort_index())
                        with c2:
                            st.write("Testing label distribution:")
                            st.write(y_test.value_counts(dropna=False).sort_index())

                        pipe = Pipeline([
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                            ("clf", RandomForestClassifier(
                                n_estimators=400,
                                max_depth=14,
                                min_samples_leaf=3,
                                class_weight="balanced",
                                random_state=42,
                                n_jobs=-1
                            ))
                        ])

                        with st.spinner("Training Random Forest model..."):
                            pipe.fit(X_train, y_train)
                            y_pred = pipe.predict(X_test)
                            y_pred_proba = pipe.predict_proba(X_test)[:, 1]

                        # Predict on the full feature set
                        full_feature_df = ml_features.copy()
                        full_pred = pipe.predict(full_feature_df)
                        df_primary["ml_flag"] = full_pred.astype(int)
                        ml_trained = True

                        # -----------------------------------------------------------------
                        # EVALUATION
                        # -----------------------------------------------------------------
                        st.subheader("ML Evaluation on Year-Based Holdout")

                        cm = confusion_matrix(y_test, y_pred)
                        kappa = cohen_kappa_score(y_test, y_pred)
                        mcc = matthews_corrcoef(y_test, y_pred)

                        precision, recall, f1, _ = precision_recall_fscore_support(
                            y_test, y_pred, average="binary", zero_division=0
                        )

                        holdout_start = X_test.index.min()
                        holdout_end = X_test.index.max()

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Precision", f"{precision:.3f}")
                        c2.metric("Recall", f"{recall:.3f}")
                        c3.metric("F1", f"{f1:.3f}")
                        c4.metric("MCC", f"{mcc:.3f}")

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Cohen's Kappa", f"{kappa:.3f}")
                        c2.metric("Holdout Samples", f"{len(y_test):,}")
                        c3.metric("Holdout Bad Rate", f"{(y_test.mean() * 100):.2f}%")

                        st.write(f"**Holdout period:** {holdout_start} to {holdout_end}")
                        st.text(classification_report(y_test, y_pred, zero_division=0))

                        fig_cm = go.Figure(data=go.Heatmap(
                            z=cm,
                            x=["Pred Good", "Pred Bad"],
                            y=["True Good", "True Bad"],
                            text=cm,
                            texttemplate="%{text}",
                            colorscale="Blues",
                            showscale=False
                        ))
                        fig_cm.update_layout(
                            title="Supervised ML Confusion Matrix (Year-Based Holdout)",
                            height=350
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)

                        # -----------------------------------------------------------------
                        # HOLDOUT PROBABILITY PLOT
                        # -----------------------------------------------------------------
                        eval_plot_df = pd.DataFrame({
                            "time": X_test.index,
                            "true_label": y_test.values,
                            "pred_label": y_pred,
                            "pred_prob_bad": y_pred_proba
                        }).set_index("time")

                        fig_prob = go.Figure()
                        fig_prob.add_trace(go.Scatter(
                            x=eval_plot_df.index,
                            y=eval_plot_df["pred_prob_bad"],
                            mode="lines",
                            name="Predicted probability of bad"
                        ))
                        fig_prob.add_trace(go.Scatter(
                            x=eval_plot_df.index[eval_plot_df["true_label"] == 1],
                            y=eval_plot_df.loc[eval_plot_df["true_label"] == 1, "pred_prob_bad"],
                            mode="markers",
                            name="True bad in holdout"
                        ))
                        fig_prob.update_layout(
                            title="Holdout Years: Predicted Probability of Bad Observations",
                            yaxis_title="Probability",
                            height=350
                        )
                        st.plotly_chart(fig_prob, use_container_width=True)

                        # -----------------------------------------------------------------
                        # YEARLY COMPARISON PLOTS
                        # -----------------------------------------------------------------
                        st.subheader("Yearly Comparison of Original vs ML Flags")

                        plot_mode = st.radio(
                            "Plot comparison for",
                            options=[
                                "Entire selected year range",
                                "Test years only"
                            ],
                            index=0
                        )

                        plot_df = df_primary.copy()
                        plot_df["original_label"] = df_primary[training_label_col]

                        if plot_mode == "Test years only":
                            plot_df = plot_df[plot_df.index.year.isin(test_years)].copy()

                        yearly_summary = pd.DataFrame(index=sorted(plot_df.index.year.unique()))
                        yearly_summary["n_obs"] = plot_df.groupby(plot_df.index.year).size()
                        yearly_summary["original_flag_rate_pct"] = (
                            plot_df.groupby(plot_df.index.year)["original_label"].mean() * 100
                        )
                        yearly_summary["ml_flag_rate_pct"] = (
                            plot_df.groupby(plot_df.index.year)["ml_flag"].mean() * 100
                        )

                        yearly_summary = yearly_summary.reset_index().rename(columns={"index": "year"})

                        fig_year = go.Figure()
                        fig_year.add_trace(go.Bar(
                            x=yearly_summary["year"],
                            y=yearly_summary["original_flag_rate_pct"],
                            name="Original label bad rate (%)"
                        ))
                        fig_year.add_trace(go.Bar(
                            x=yearly_summary["year"],
                            y=yearly_summary["ml_flag_rate_pct"],
                            name="ML bad rate (%)"
                        ))
                        fig_year.update_layout(
                            title=f"Yearly Bad-Rate Comparison ({plot_mode})",
                            xaxis_title="Year",
                            yaxis_title="Bad rate (%)",
                            barmode="group",
                            height=400
                        )
                        st.plotly_chart(fig_year, use_container_width=True)

                        # -----------------------------------------------------------------
                        # FULL TIME SERIES / TEST-YEAR COMPARISON
                        # -----------------------------------------------------------------
                        st.subheader("Time-Series Comparison of Original vs ML Flags")

                        ts_plot_df = df_primary.copy()
                        ts_plot_df["original_label"] = df_primary[training_label_col]

                        if plot_mode == "Test years only":
                            ts_plot_df = ts_plot_df[ts_plot_df.index.year.isin(test_years)].copy()

                        # keep a manageable number of points for plotting
                        ts_plot_df = ts_plot_df.copy()
                        ts_plot_df["year"] = ts_plot_df.index.year

                        fig_ts = go.Figure()
                        fig_ts.add_trace(go.Scatter(
                            x=ts_plot_df.index,
                            y=ts_plot_df["T_air"],
                            mode="lines",
                            name="Temperature"
                        ))

                        orig_bad_mask = pd.to_numeric(ts_plot_df["original_label"], errors="coerce").fillna(0).astype(int) == 1
                        ml_bad_mask = pd.to_numeric(ts_plot_df["ml_flag"], errors="coerce").fillna(0).astype(int) == 1

                        fig_ts.add_trace(go.Scatter(
                            x=ts_plot_df.index[orig_bad_mask],
                            y=ts_plot_df.loc[orig_bad_mask, "T_air"],
                            mode="markers",
                            name="Original bad labels",
                            marker=dict(size=6, symbol="x")
                        ))

                        fig_ts.add_trace(go.Scatter(
                            x=ts_plot_df.index[ml_bad_mask],
                            y=ts_plot_df.loc[ml_bad_mask, "T_air"],
                            mode="markers",
                            name="ML bad labels",
                            marker=dict(size=5, symbol="circle-open")
                        ))

                        fig_ts.update_layout(
                            title=f"Temperature and Flag Comparison ({plot_mode})",
                            xaxis_title="Time",
                            yaxis_title="Temperature (°C)",
                            height=500
                        )
                        st.plotly_chart(fig_ts, use_container_width=True)

                        # -----------------------------------------------------------------
                        # FEATURE IMPORTANCE
                        # -----------------------------------------------------------------
                        clf = pipe.named_steps["clf"]
                        feature_names = X_train.columns.tolist()
                        importances = pd.DataFrame({
                            "feature": feature_names,
                            "importance": clf.feature_importances_
                        }).sort_values("importance", ascending=False).head(20)

                        st.subheader("Top ML Features")
                        st.dataframe(importances, use_container_width=True, hide_index=True)

                        fig_imp = go.Figure()
                        fig_imp.add_trace(go.Bar(
                            x=importances["importance"][::-1],
                            y=importances["feature"][::-1],
                            orientation="h",
                            name="Importance"
                        ))
                        fig_imp.update_layout(
                            title="Top 20 Feature Importances",
                            height=500,
                            yaxis_title=""
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)

else:
    df_primary["ml_flag"] = np.nan
    ml_trained = False
    


# =============================================================================
# COMPARISON TABLE
# =============================================================================
st.header("📊 QC Output Table")

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

comparison_df["expert_label"] = df_primary[expert_label_col]
comparison_df["training_label"] = df_primary[training_label_col] if training_label_col in df_primary.columns else df_primary[training_label_col]
comparison_df["ml_flag"] = df_primary["ml_flag"]

if ml_trained:
    comparison_df["comparison_type"] = "Neither"
    comparison_df.loc[(comparison_df["rule_flag"] == 1) & (comparison_df["ml_flag"] == 1), "comparison_type"] = "Both"
    comparison_df.loc[(comparison_df["rule_flag"] == 1) & (comparison_df["ml_flag"] == 0), "comparison_type"] = "Rule Only"
    comparison_df.loc[(comparison_df["rule_flag"] == 0) & (comparison_df["ml_flag"] == 1), "comparison_type"] = "ML Only"
else:
    comparison_df["comparison_type"] = np.where(comparison_df["rule_flag"] == 1, "Rule Flagged", "Rule Clean")

# =============================================================================
# SUMMARY METRICS
# =============================================================================
st.subheader("Summary Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rule Flags", int(comparison_df["rule_flag"].sum()))
c2.metric("Silver Bad", int((comparison_df["silver_label"] == 1).sum()))
c3.metric("Silver Good", int((comparison_df["silver_label"] == 0).sum()))
c4.metric("ML Flags", int(pd.to_numeric(comparison_df["ml_flag"], errors="coerce").fillna(0).sum()) if ml_trained else 0)

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Silver Label Evaluation",
    "Seasonal Evaluation",
    "Event-Based Evaluation",
    "Time Series",
    "Download"
])

with tab1:
    st.subheader("Overview")
    st.dataframe(comparison_df.head(100), use_container_width=True)

    flag_counts = comparison_df["comparison_type"].value_counts()
    fig = go.Figure(data=[go.Pie(
        labels=flag_counts.index,
        values=flag_counts.values,
        hole=0.4,
        textinfo="label+percent"
    )])
    fig.update_layout(title="Comparison Type Distribution", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### High-priority uncertain cases for expert review")
    review_df = comparison_df[comparison_df["review_priority"] == "High"].head(50)
    st.dataframe(review_df, use_container_width=True)

with tab2:
    st.subheader("Silver Label Evaluation")
    st.markdown("""
This section evaluates QC methods against **silver labels**, which are proxy labels used because expert true labels are still needed.
Interpret these as **screening performance**, not definitive truth performance.
""")

    silver_eval = comparison_df.dropna(subset=["silver_label"]).copy()
    if silver_eval.empty or silver_eval["silver_label"].nunique() < 2:
        st.info("Not enough silver labels for evaluation.")
    else:
        y_true = silver_eval["silver_label"].astype(int)
        y_rule = silver_eval["rule_flag"].astype(int)

        pr, rc, f1, _ = precision_recall_fscore_support(
            y_true, y_rule, average="binary", zero_division=0
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Rule Precision vs Silver", f"{pr:.3f}")
        c2.metric("Rule Recall vs Silver", f"{rc:.3f}")
        c3.metric("Rule F1 vs Silver", f"{f1:.3f}")

        if ml_trained:
            y_ml = silver_eval["ml_flag"].astype(int)
            pr, rc, f1, _ = precision_recall_fscore_support(
                y_true, y_ml, average="binary", zero_division=0
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("ML Precision vs Silver", f"{pr:.3f}")
            c2.metric("ML Recall vs Silver", f"{rc:.3f}")
            c3.metric("ML F1 vs Silver", f"{f1:.3f}")

with tab3:
    st.subheader("Seasonal Evaluation")

    eval_df = comparison_df.dropna(subset=["silver_label"]).copy()
    if eval_df.empty:
        st.info("No silver labels available for seasonal evaluation.")
    else:
        eval_df["silver_label"] = eval_df["silver_label"].astype(int)
        season_rule = seasonal_metrics(eval_df, "silver_label", "rule_flag")
        st.markdown("### Rule-based seasonal metrics vs silver labels")
        st.dataframe(season_rule, use_container_width=True, hide_index=True)

        if ml_trained:
            season_ml = seasonal_metrics(eval_df, "silver_label", "ml_flag")
            st.markdown("### ML seasonal metrics vs silver labels")
            st.dataframe(season_ml, use_container_width=True, hide_index=True)

            if not season_ml.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=season_ml["season"], y=season_ml["f1"], name="ML F1"))
                if not season_rule.empty:
                    fig.add_trace(go.Bar(x=season_rule["season"], y=season_rule["f1"], name="Rule F1"))
                fig.update_layout(title="Seasonal F1 Scores vs Silver Labels", barmode="group", height=400)
                st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Event-Based Evaluation")

    eval_df = comparison_df.dropna(subset=["silver_label"]).copy()
    if eval_df.empty:
        st.info("No silver labels available for event-based evaluation.")
    else:
        eval_df["silver_label"] = eval_df["silver_label"].astype(int)
        event_rule = event_metrics(eval_df, "silver_label", "rule_flag")
        st.markdown("### Rule-based event metrics vs silver labels")
        st.dataframe(event_rule, use_container_width=True, hide_index=True)

        if ml_trained:
            event_ml = event_metrics(eval_df, "silver_label", "ml_flag")
            st.markdown("### ML event metrics vs silver labels")
            st.dataframe(event_ml, use_container_width=True, hide_index=True)

            if not event_ml.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=event_ml["event_type"], y=event_ml["f1"], name="ML F1"))
                if not event_rule.empty:
                    fig.add_trace(go.Bar(x=event_rule["event_type"], y=event_rule["f1"], name="Rule F1"))
                fig.update_layout(title="Event-Based F1 Scores vs Silver Labels", barmode="group", height=400)
                st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Time Series View")

    recent = comparison_df.tail(1500).copy()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["Temperature"],
            mode="lines",
            name="Temperature"
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=recent.index[recent["rule_flag"] == 1],
            y=recent.loc[recent["rule_flag"] == 1, "Temperature"],
            mode="markers",
            name="Rule Flags",
            marker=dict(size=7, symbol="x")
        ),
        secondary_y=False
    )

    silver_bad_mask = recent["silver_label"] == 1
    fig.add_trace(
        go.Scatter(
            x=recent.index[silver_bad_mask.fillna(False)],
            y=recent.loc[silver_bad_mask.fillna(False), "Temperature"],
            mode="markers",
            name="Silver Bad",
            marker=dict(size=7, symbol="diamond")
        ),
        secondary_y=False
    )

    if ml_trained:
        ml_mask = pd.to_numeric(recent["ml_flag"], errors="coerce").fillna(0).astype(int) == 1
        fig.add_trace(
            go.Scatter(
                x=recent.index[ml_mask],
                y=recent.loc[ml_mask, "Temperature"],
                mode="markers",
                name="ML Flags",
                marker=dict(size=6, symbol="circle-open")
            ),
            secondary_y=False
        )

    if recent["spatial_anom_diff"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=recent.index,
                y=recent["spatial_anom_diff"],
                mode="lines",
                name="Spatial anomaly diff"
            ),
            secondary_y=True
        )

    fig.update_layout(title="Recent Time Series with QC Flags", height=550)
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
    fig.update_yaxes(title_text="Spatial anomaly difference", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.subheader("Download Results")

    st.markdown("""
Recommended workflow:
1. Use **silver labels** as a bootstrapping layer.
2. Export the **high-priority uncertain cases** for expert review.
3. Replace silver labels with **expert true labels** over time.
4. Re-train and re-evaluate by **season** and **event type**.
""")

    csv = comparison_df.to_csv()
    st.download_button(
        "Download QC Results CSV",
        data=csv,
        file_name="advanced_temperature_qc_silver_labels.csv",
        mime="text/csv"
    )
