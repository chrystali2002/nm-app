#!/usr/bin/env python3
"""
Streamlit app for Advanced Temperature QC with Silver Labels.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import math
import zipfile
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
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


st.set_page_config(
    page_title="Advanced Temperature QC",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Advanced Temperature QC with Silver Labels")
st.write(
    """
A Streamlit interface for NOAA hourly temperature QA/QC using:
- climatology-aware rules
- spatial neighbor checks
- silver labels
- optional supervised machine learning
"""
)

# -----------------------------------------------------------------------------
# Cached metadata loader for UI
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_station_metadata():
    return load_station_metadata()

@st.cache_data(show_spinner=False)
def get_station_preview_cached(
    state: str,
    start_year: int,
    end_year: int,
    primary_filename: str | None,
    primary_name: str | None,
    max_distance_km: float,
    max_elev_diff: float,
):
    stations = load_station_metadata()
    return build_station_preview(
        stations=stations,
        state=state,
        start_year=start_year,
        end_year=end_year,
        primary_filename=primary_filename,
        primary_name=primary_name,
        max_distance_km=max_distance_km,
        max_elev_diff=max_elev_diff,
        max_candidates=10,
    )

stations_all = get_station_metadata()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    state = st.text_input("State", value="NM").strip().upper()

    stations_state = stations_all[stations_all["STATE"] == state].copy() if state else stations_all.copy()

    station_mode = st.radio(
        "Station selection mode",
        ["Select from list", "Filename", "Name contains"],
        index=0
    )

    primary_filename = None
    primary_name = None

    if station_mode == "Select from list":
        if not stations_state.empty:
            stations_state = stations_state.sort_values("STATION NAME").copy()
            stations_state["display"] = stations_state["STATION NAME"] + " | " + stations_state["FILENAME"]
            selected_display = st.selectbox("Choose station", stations_state["display"].tolist())
            selected_row = stations_state[stations_state["display"] == selected_display].iloc[0]
            primary_filename = selected_row["FILENAME"]
        else:
            st.warning("No stations found for this state.")
    elif station_mode == "Filename":
        primary_filename = st.text_input("Primary filename", value="72253923044.csv").strip() or None
    else:
        primary_name = st.text_input("Station name contains", value="HOBBS").strip() or None

    st.subheader("Year range")
    c1, c2 = st.columns(2)
    with c1:
        start_year = st.number_input("Start year", min_value=1900, max_value=2100, value=2007, step=1)
    with c2:
        end_year = st.number_input("End year", min_value=1900, max_value=2100, value=2012, step=1)

    st.subheader("ML options")
    ml_model = st.selectbox(
        "ML model",
        ["random_forest", "extra_trees", "gradient_boosting", "logistic_regression"],
        index=0
    )
    train_ml_on = st.selectbox(
        "Train ML on",
        ["silver", "expert", "expert_else_silver"],
        index=2
    )
    split_method = st.selectbox(
        "Split method",
        ["latest_years", "eighty_twenty_years"],
        index=0
    )
    n_test_years = st.number_input("Number of test years (0 = auto)", min_value=0, max_value=20, value=0, step=1)
    ml_prob_threshold = st.slider("ML bad-probability threshold", 0.0, 1.0, 0.80, 0.01)
    aux_contamination = st.slider("Isolation Forest contamination", 0.001, 0.20, 0.02, 0.001)

    st.subheader("Neighbor options")
    max_neighbors = st.number_input("Max neighbors", min_value=1, max_value=10, value=3, step=1)
    max_distance_km = st.number_input("Max distance (km)", min_value=1.0, max_value=500.0, value=30.0, step=1.0)
    max_elev_diff = st.number_input("Max elevation diff (m)", min_value=0.0, max_value=5000.0, value=300.0, step=10.0)
    min_corr = st.slider("Minimum correlation", 0.0, 1.0, 0.40, 0.01)

    st.subheader("Optional uploaded inputs")
    metadata_upload = st.file_uploader("Metadata events CSV", type=["csv"])
    expert_upload = st.file_uploader("Expert labels CSV", type=["csv"])
    expert_label_col = st.text_input("Expert label column", value="expert_label")

    st.subheader("Figure generation")
    generate_figures = st.checkbox("Generate figures", value=True)

    if generate_figures:
        time_series = st.checkbox("Time-series comparison", value=True)
        zooms = st.checkbox("Automatic zoom figures", value=True)
        flatline = st.checkbox("Flatline figure", value=True)
        spike = st.checkbox("Spike figure", value=True)
        climatology = st.checkbox("Climatology/z-score figure", value=True)
        spatial = st.checkbox("Spatial consistency figure", value=True)
        daily_counts = st.checkbox("Daily rule-flag counts", value=True)
        confusion_matrix = st.checkbox("Confusion matrix", value=True)
        holdout_probability = st.checkbox("Holdout probability plot", value=True)
        yearly_comparison = st.checkbox("Yearly bad-rate comparison", value=True)
        feature_importance = st.checkbox("Feature importance", value=True)
    else:
        time_series = zooms = flatline = spike = climatology = False
        spatial = daily_counts = confusion_matrix = False
        holdout_probability = yearly_comparison = feature_importance = False

    st.subheader("Actions")
    preview_button = st.button("Preview Station Availability", use_container_width=True)
    run_button = st.button("Run QC Pipeline", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# PREVIEW PANEL
# -----------------------------------------------------------------------------
preview_requested = preview_button or (
    primary_filename is not None or primary_name is not None
)

if end_year < start_year:
    st.error("End year must be greater than or equal to start year.")
    st.stop()

if preview_requested:
    st.subheader("🔎 Station Availability Preview")

    try:
        preview = get_station_preview_cached(
            state=state,
            start_year=int(start_year),
            end_year=int(end_year),
            primary_filename=primary_filename,
            primary_name=primary_name,
            max_distance_km=float(max_distance_km),
            max_elev_diff=float(max_elev_diff),
        )

        summary = preview["summary"]
        meta = preview["station_metadata"]
        year_coverage = preview["year_coverage"]
        neighbor_preview = preview["neighbor_preview"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Years requested", summary["years_requested_count"])
        c2.metric("Years found", summary["years_found_count"])
        c3.metric("Total rows found", f"{summary['total_rows_found']:,}")
        c4.metric("Candidate neighbors", summary["candidate_neighbors_found"])

        st.markdown("**Station metadata**")
        st.json(meta)

        st.markdown("**Approximate yearly availability**")
        display_cov = year_coverage.copy()
        for col in ["first_timestamp", "last_timestamp"]:
            if col in display_cov.columns:
                display_cov[col] = pd.to_datetime(display_cov[col], errors="coerce")
                display_cov[col] = display_cov[col].astype(str).replace("NaT", "")
        st.dataframe(display_cov, use_container_width=True)

        st.markdown("**Nearby candidate neighbors (preview)**")
        if neighbor_preview is not None and not neighbor_preview.empty:
            st.dataframe(neighbor_preview, use_container_width=True)
        else:
            st.info("No nearby candidate neighbors were found with the current distance/elevation filters.")

    except Exception as e:
        st.warning(f"Preview unavailable: {e}")

# -----------------------------------------------------------------------------
# MAIN RUN
# -----------------------------------------------------------------------------
if not run_button:
    st.info("Use the preview above to inspect station availability, then click **Run QC Pipeline**.")
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

    fig_opts = FigureOptions(
        generate_figures=generate_figures,
        time_series=time_series,
        zooms=zooms,
        flatline=flatline,
        spike=spike,
        climatology=climatology,
        spatial=spatial,
        daily_counts=daily_counts,
        confusion_matrix=confusion_matrix,
        holdout_probability=holdout_probability,
        yearly_comparison=yearly_comparison,
        feature_importance=feature_importance,
    )

    args = QCArgs(
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
        figure_options=fig_opts,
    )

    progress_placeholder = st.empty()
    log_box = st.container()

    def logger(msg: str):
        log_box.write(msg)

    try:
        progress_placeholder.info("Running pipeline... please wait.")
        results = run_pipeline(args, logger=logger)
        progress_placeholder.empty()

        outdir = Path(results["output_dir"])
        comparison_df = results["comparison_df"]
        neighbor_summary = results["neighbor_summary"]
        holdout_metrics_df = results["holdout_metrics_df"]
        yearly_summary = results["yearly_summary"]
        importances = results["importances"]
        primary_station = results["primary_station"]
        ml_trained = results["ml_trained"]

        st.success(f"Finished successfully for {primary_station['STATION NAME']}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Observations", f"{len(comparison_df):,}")
        c2.metric("Rule flags", f"{int(comparison_df['rule_flag'].fillna(0).sum()):,}")
        c3.metric("Silver bad", f"{int((comparison_df['silver_label'] == 1).sum()):,}")
        c4.metric("ML trained", "Yes" if ml_trained else "No")

        st.subheader("Selected station")
        st.write({
            "station_name": primary_station["STATION NAME"],
            "filename": primary_station["FILENAME"],
            "state": primary_station.get("STATE", ""),
            "lat": primary_station["LAT"],
            "lon": primary_station["LON"],
            "elev_m": primary_station.get("ELEV_M", None),
        })

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "QC Output Table",
            "Neighbors",
            "ML Metrics",
            "Figures",
            "Downloads"
        ])

        with tab1:
            st.dataframe(comparison_df.head(500), use_container_width=True)

        with tab2:
            if neighbor_summary is not None and not neighbor_summary.empty:
                st.dataframe(neighbor_summary, use_container_width=True)
            else:
                st.warning("No usable neighbors found.")

        with tab3:
            if holdout_metrics_df is not None and not holdout_metrics_df.empty:
                st.markdown("**Holdout metrics**")
                st.dataframe(holdout_metrics_df, use_container_width=True)
            else:
                st.info("No supervised ML evaluation metrics were generated.")

            if yearly_summary is not None and not yearly_summary.empty:
                st.markdown("**Yearly bad-rate comparison table**")
                st.dataframe(yearly_summary, use_container_width=True)

            if importances is not None and not importances.empty:
                st.markdown("**Top feature importances**")
                st.dataframe(importances, use_container_width=True)

        with tab4:
            png_files = sorted(outdir.glob("*.png"))
            if png_files:
                for png in png_files:
                    st.markdown(f"**{png.name}**")
                    st.image(str(png), use_container_width=True)
            else:
                st.info("No figures were generated.")

        with tab5:
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
        progress_placeholder.empty()
        st.error(f"Pipeline failed: {e}")
