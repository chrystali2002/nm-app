#!/usr/bin/env python3
"""
Streamlit app for Advanced Temperature QC with Silver Labels.

Run:
    streamlit run app.py
"""
#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from appcore.qc_core import (
    QCArgs,
    FigureOptions,
    load_station_metadata,
    run_pipeline,
    zip_directory,
    build_station_preview,
)


st.set_page_config(
    page_title="New Mexico Temperature QA/QC",
    page_icon="🌡️",
    layout="wide"
)

# -------------------------------
# TITLE
# -------------------------------
st.title("🌡️ New Mexico Hourly Surface Air Temperature QA/QC Analysis")

st.markdown("""
This application performs enhanced multi-tier quality assurance and quality control (QA/QC) 
analysis on hourly surface air temperature observations from ASOS (Automated Surface Observing System) 
stations in New Mexico.

It includes:
- **Multi-tier QA/QC checks** (range, spike, flatline, and spatial consistency)
- **Station-level summary statistics**
- **Flag pattern analysis** (clustering, recurrence, and seasonality)
- **Raw vs validated data comparison**
- **Decision-matrix-based classification** for flagged observations
- **Enhanced visualizations** with statistical overlays
- **Flag type distribution analysis** with pie charts
""")

# -------------------------------
# DATA SOURCE SECTION
# -------------------------------
with st.expander("📡 Data Source: ASOS Network", expanded=True):

    col1, col2 = st.columns([2,1])

    with col1:
        st.markdown("""
        ### Automated Surface Observing System (ASOS)

        The **Automated Surface Observing System (ASOS)** is a nationwide network of automated
        weather observing stations operated primarily by the **National Weather Service (NWS)**,
        **Federal Aviation Administration (FAA)**, and **Department of Defense (DoD)**.

        **Key Characteristics**

        • Nationwide station coverage including multiple stations in New Mexico  
        • Hourly observations (some stations report more frequently)  
        • Measures temperature, dew point, wind, precipitation, pressure, and visibility  
        • Raw observations may include automated QC indicators  

        **Data Access**

        • **Source:** NOAA Integrated Surface Database (ISD)  
        • **Format:** Global Hourly (CSV)  
        • **Period analyzed:** 2000-–2024  
        • **Variable used in this analysis:** Surface air temperature (`TMP`)
        """)

    with col2:
        st.markdown("""
        ### Station Map Legend

        🟢 **<5% flags** — Good quality  
        🟡 **5–10% flags** — Moderate issues  
        🟠 **10–20% flags** — Problematic  
        🔴 **>20% flags** — Poor quality  

        **Processing Workflow**

        1. Raw data ingestion  
        2. Multi-tier QC checks  
        3. Flag classification  
        4. Decision matrix application  
        5. Final validation  
        """)


# -----------------------------------------------------------------------------
# Helper functions for pie charts
# -----------------------------------------------------------------------------
def create_flag_distribution_pie_charts(comparison_df: pd.DataFrame) -> dict:
    """
    Create pie charts showing flag type distributions.
    
    Returns:
        Dictionary of matplotlib figures
    """
    figures = {}
    
    # Define flag columns and their display names
    flag_columns = {
        'spatial_flag': 'Spatial',
        'flatline_flag': 'Flatline',
        'spike_flag': 'Spike',
        'range_flag': 'Range (Rule)',
        'climatology_flag': 'Climatology (Rule)'
    }
    
    # 1. Overall flag type distribution (counts of each flag type)
    flag_counts = {}
    for col, label in flag_columns.items():
        if col in comparison_df.columns:
            flag_counts[label] = comparison_df[col].sum()
    
    if flag_counts and sum(flag_counts.values()) > 0:
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(flag_counts)))
        
        wedges, texts, autotexts = ax1.pie(
            flag_counts.values(),
            labels=flag_counts.keys(),
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        
        # Enhance text appearance
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_color('white')
            autotext.set_weight('bold')
        
        ax1.set_title('Flag Type Distribution\n(Total flags by type)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.axis('equal')
        
        # Add legend with counts
        legend_labels = [f'{k}: {v:,} ({v/sum(flag_counts.values())*100:.1f}%)' 
                        for k, v in flag_counts.items()]
        ax1.legend(wedges, legend_labels, title="Flag Counts", 
                  loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        figures['flag_type_distribution'] = fig1
    
    # 2. ML vs Rule-based flags distribution (if ML column exists)
    if 'ml_bad' in comparison_df.columns:
        ml_flags = comparison_df['ml_bad'].sum()
        rule_flags = comparison_df['rule_flag'].sum() if 'rule_flag' in comparison_df.columns else 0
        
        if ml_flags > 0 or rule_flags > 0:
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            data = {'Machine Learning': ml_flags, 'Rule-based': rule_flags}
            colors = ['#ff9999', '#66b3ff']
            
            # Filter out zero values
            data = {k: v for k, v in data.items() if v > 0}
            
            if data:
                wedges, texts, autotexts = ax2.pie(
                    data.values(),
                    labels=data.keys(),
                    autopct='%1.1f%%',
                    colors=colors[:len(data)],
                    startangle=90,
                    explode=(0.05,) if len(data) == 1 else (0.05, 0)
                )
                
                for text in texts:
                    text.set_fontsize(11)
                for autotext in autotexts:
                    autotext.set_fontsize(10)
                    autotext.set_color('white')
                    autotext.set_weight('bold')
                
                ax2.set_title('ML vs Rule-based Flag Distribution', 
                             fontsize=14, fontweight='bold', pad=20)
                ax2.axis('equal')
                
                # Add count labels
                count_labels = [f'{k}: {v:,} ({v/sum(data.values())*100:.1f}%)' 
                               for k, v in data.items()]
                ax2.legend(wedges, count_labels, title="Flag Counts", 
                          loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                
                plt.tight_layout()
                figures['ml_vs_rule_distribution'] = fig2
    
    # 3. Flag type distribution for observations flagged as "bad" (silver_label == 1)
    if 'silver_label' in comparison_df.columns:
        bad_obs = comparison_df[comparison_df['silver_label'] == 1]
        
        if not bad_obs.empty:
            bad_flag_counts = {}
            for col, label in flag_columns.items():
                if col in bad_obs.columns:
                    bad_flag_counts[label] = bad_obs[col].sum()
            
            if bad_flag_counts and sum(bad_flag_counts.values()) > 0:
                fig3, ax3 = plt.subplots(figsize=(10, 8))
                colors = plt.cm.Paired(np.linspace(0, 1, len(bad_flag_counts)))
                
                wedges, texts, autotexts = ax3.pie(
                    bad_flag_counts.values(),
                    labels=bad_flag_counts.keys(),
                    autopct='%1.1f%%',
                    colors=colors,
                    startangle=90
                )
                
                for text in texts:
                    text.set_fontsize(10)
                for autotext in autotexts:
                    autotext.set_fontsize(9)
                    autotext.set_color('white')
                    autotext.set_weight('bold')
                
                ax3.set_title('Flag Type Distribution for "Bad" Observations\n(Silver Label = 1)', 
                             fontsize=14, fontweight='bold', pad=20)
                ax3.axis('equal')
                
                legend_labels = [f'{k}: {v:,} ({v/sum(bad_flag_counts.values())*100:.1f}%)' 
                                for k, v in bad_flag_counts.items()]
                ax3.legend(wedges, legend_labels, title="Flag Counts", 
                          loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                
                plt.tight_layout()
                figures['bad_obs_flag_distribution'] = fig3
    
    return figures

def save_pie_charts(figures: dict, output_dir: Path):
    """Save pie chart figures to the output directory."""
    for name, fig in figures.items():
        fig.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

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
        flag_distribution = st.checkbox("Flag distribution pie charts", value=True)
    else:
        time_series = zooms = flatline = spike = climatology = False
        spatial = daily_counts = confusion_matrix = False
        holdout_probability = yearly_comparison = feature_importance = False
        flag_distribution = False

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

        # Generate pie charts for flag distributions
        if flag_distribution:
            pie_charts = create_flag_distribution_pie_charts(comparison_df)
            # Save pie charts to output directory
            save_pie_charts(pie_charts, outdir)
        else:
            pie_charts = {}

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

        # Create tabs - include Flag Distribution if we have charts
        if pie_charts:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "QC Output Table",
                "Neighbors",
                "ML Metrics",
                "Flag Distribution",
                "Figures",
                "Downloads"
            ])
        else:
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

        # Flag Distribution tab (if charts exist)
        if pie_charts:
            with tab4:
                if pie_charts:
                    st.subheader("Flag Type Distribution Analysis")
                    
                    # Display each pie chart with explanation
                    if 'flag_type_distribution' in pie_charts:
                        st.markdown("### Overall Flag Type Distribution")
                        st.markdown("This chart shows the distribution of different QC flag types across all observations.")
                        st.pyplot(pie_charts['flag_type_distribution'])
                        
                        # Calculate and display summary statistics
                        total_flags = sum([comparison_df[col].sum() for col in ['spatial_flag', 'flatline_flag', 
                                                                               'spike_flag', 'range_flag', 
                                                                               'climatology_flag'] if col in comparison_df.columns])
                        st.markdown(f"**Total flags detected:** {int(total_flags):,}")
                        st.markdown("---")
                    
                    if 'ml_vs_rule_distribution' in pie_charts:
                        st.markdown("### ML vs Rule-based Flag Distribution")
                        st.markdown("Comparison between machine learning detected flags and traditional rule-based flags.")
                        st.pyplot(pie_charts['ml_vs_rule_distribution'])
                        
                        ml_flags = comparison_df['ml_bad'].sum() if 'ml_bad' in comparison_df.columns else 0
                        rule_flags = comparison_df['rule_flag'].sum() if 'rule_flag' in comparison_df.columns else 0
                        st.markdown(f"**ML flags:** {int(ml_flags):,}")
                        st.markdown(f"**Rule-based flags:** {int(rule_flags):,}")
                        st.markdown("---")
                    
                    if 'bad_obs_flag_distribution' in pie_charts:
                        st.markdown("### Flag Distribution for 'Bad' Observations")
                        st.markdown("Distribution of flag types specifically for observations classified as 'bad' (silver_label = 1).")
                        st.pyplot(pie_charts['bad_obs_flag_distribution'])
                        
                        bad_obs_count = (comparison_df['silver_label'] == 1).sum()
                        st.markdown(f"**Total 'bad' observations:** {bad_obs_count:,}")
                    
                    # Add summary statistics
                    st.markdown("### Summary Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    total_obs = len(comparison_df)
                    total_flags = sum([comparison_df[col].sum() for col in ['spatial_flag', 'flatline_flag', 
                                                                           'spike_flag', 'range_flag', 
                                                                           'climatology_flag'] if col in comparison_df.columns])
                    
                    col1.metric("Total Observations", f"{total_obs:,}")
                    col2.metric("Total Flags", f"{int(total_flags):,}")
                    col3.metric("Flags per Observation", f"{total_flags/total_obs:.2f}")
                    
                else:
                    st.info("No flag data available to generate distribution charts.")

        # Figures tab
        if pie_charts:
            figures_tab = tab5
        else:
            figures_tab = tab4
            
        with figures_tab:
            png_files = sorted(outdir.glob("*.png"))
            if png_files:
                for png in png_files:
                    # Skip pie charts here since they're in their own tab
                    if png.name not in ['flag_type_distribution.png', 
                                       'ml_vs_rule_distribution.png', 
                                       'bad_obs_flag_distribution.png']:
                        st.markdown(f"**{png.name}**")
                        st.image(str(png), use_container_width=True)
            else:
                st.info("No figures were generated.")

        # Downloads tab
        if pie_charts:
            downloads_tab = tab6
        else:
            downloads_tab = tab5
            
        with downloads_tab:
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
        import traceback
        st.code(traceback.format_exc())
