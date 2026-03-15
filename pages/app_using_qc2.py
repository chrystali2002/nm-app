#!/usr/bin/env python3
"""
Streamlit Interface for Advanced Temperature QC System
Uses qc2_core.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, precision_recall_curve

from qc2_core import QCArgs, FigureOptions, run_pipeline



st.set_page_config(
    page_title="Advanced Temperature QC",
    layout="wide"
)

st.title("Advanced Temperature QC System")

st.write(
"""
This application performs **automated quality control of temperature observations** using:

• rule-based QC  
• spatial consistency checks  
• silver labels  
• supervised machine learning  

The model is optimized to detect **bad observations (anomalies)**.
"""
)

# ---------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------

st.sidebar.header("Station Selection")

state = st.sidebar.text_input("State", "NM")
primary_filename = st.sidebar.text_input("Station Filename (optional)", "")
primary_name = st.sidebar.text_input("Station Name (optional)", "")

start_year = st.sidebar.number_input("Start Year", 2000, 2025, 2018)
end_year = st.sidebar.number_input("End Year", 2000, 2025, 2023)

st.sidebar.header("Neighbor Settings")

max_neighbors = st.sidebar.slider("Max neighbors", 1, 6, 3)
max_distance_km = st.sidebar.slider("Max neighbor distance (km)", 5.0, 200.0, 30.0)
min_corr = st.sidebar.slider("Minimum correlation", 0.0, 1.0, 0.40)

st.sidebar.header("Machine Learning")

ml_model = st.sidebar.selectbox(
    "ML Model",
    ["random_forest", "extra_trees", "gradient_boosting", "logistic_regression"]
)

train_ml_on = st.sidebar.selectbox(
    "Training label source",
    ["expert_else_silver", "expert", "silver"]
)

split_method = st.sidebar.selectbox(
    "Train/Test split method",
    ["latest_years"]
)

ml_prob_threshold = st.sidebar.slider(
    "Probability threshold (manual)",
    0.0,
    1.0,
    0.80
)

st.sidebar.header("Imbalance Handling")

use_class_weighting = st.sidebar.checkbox(
    "Use class weighting",
    value=True
)

use_smote = st.sidebar.checkbox(
    "Use SMOTE oversampling",
    value=True
)

smote_k = st.sidebar.slider(
    "SMOTE neighbors",
    2,
    10,
    5
)

st.sidebar.header("Threshold Optimization")

auto_tune = st.sidebar.checkbox(
    "Auto optimize probability threshold",
    True
)

threshold_metric = st.sidebar.selectbox(
    "Optimization target",
    ["f1_bad", "recall_bad", "balanced_accuracy"]
)

min_precision = st.sidebar.slider(
    "Minimum precision for bad class",
    0.0,
    1.0,
    0.50
)

st.sidebar.header("Run QC")

run_button = st.sidebar.button("Run QC Pipeline")

# ---------------------------------------------------------
# Run Pipeline
# ---------------------------------------------------------

if run_button:

    with st.spinner("Running QC pipeline..."):

        args = QCArgs(
            output_dir="qc_output",
            state=state,
            primary_filename=primary_filename if primary_filename else None,
            primary_name=primary_name if primary_name else None,
            start_year=start_year,
            end_year=end_year,
            max_neighbors=max_neighbors,
            max_distance_km=max_distance_km,
            min_corr=min_corr,
            ml_model=ml_model,
            train_ml_on=train_ml_on,
            split_method=split_method,
            ml_prob_threshold=ml_prob_threshold,
            use_class_weighting=use_class_weighting,
            use_smote=use_smote,
            smote_k_neighbors=smote_k,
            auto_tune_threshold=auto_tune,
            threshold_metric=threshold_metric,
            min_precision_bad=min_precision,
        )

        result = run_pipeline(args)

    st.success("QC pipeline completed")

    df = result["comparison_df"]
    metrics = result["holdout_metrics_df"]
    importances = result["importances"]

    # -----------------------------------------------------
    # Holdout metrics
    # -----------------------------------------------------

    st.header("Holdout Performance Metrics")

    if not metrics.empty:
        st.dataframe(metrics)

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    st.header("Confusion Matrix")

    if not metrics.empty:

        tn = metrics.iloc[0]["tn"]
        fp = metrics.iloc[0]["fp"]
        fn = metrics.iloc[0]["fn"]
        tp = metrics.iloc[0]["tp"]

        cm = np.array([[tn, fp], [fn, tp]])

        fig, ax = plt.subplots()

        im = ax.imshow(cm)

        ax.set_xticks([0,1])
        ax.set_yticks([0,1])

        ax.set_xticklabels(["Pred Good", "Pred Bad"])
        ax.set_yticklabels(["True Good", "True Bad"])

        for i in range(2):
            for j in range(2):
                ax.text(j,i,str(cm[i,j]),ha="center",va="center")

        st.pyplot(fig)

    # -----------------------------------------------------
    # ROC Curve
    # -----------------------------------------------------

    if result["ml_trained"]:

        st.header("ROC Curve")

        holdout = result["holdout_predictions_df"]

        if not holdout.empty:

            fpr, tpr, _ = roc_curve(
                holdout["y_true"],
                holdout["y_prob"]
            )

            fig, ax = plt.subplots()

            ax.plot(fpr, tpr)
            ax.plot([0,1],[0,1],"--")

            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")

            st.pyplot(fig)

    # -----------------------------------------------------
    # Precision Recall Curve
    # -----------------------------------------------------

        st.header("Precision–Recall Curve")

        precision, recall, _ = precision_recall_curve(
            holdout["y_true"],
            holdout["y_prob"]
        )

        fig, ax = plt.subplots()

        ax.plot(recall, precision)

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")

        st.pyplot(fig)

    # -----------------------------------------------------
    # Threshold optimization plot
    # -----------------------------------------------------

        st.header("Threshold Optimization")

        thresh = result["threshold_metrics_df"]

        if not thresh.empty:

            fig, ax = plt.subplots()

            ax.plot(
                thresh["threshold"],
                thresh["precision_bad"],
                label="Precision"
            )

            ax.plot(
                thresh["threshold"],
                thresh["recall_bad"],
                label="Recall"
            )

            ax.plot(
                thresh["threshold"],
                thresh["f1_bad"],
                label="F1"
            )

            ax.legend()

            ax.set_xlabel("Threshold")
            ax.set_ylabel("Metric")

            st.pyplot(fig)

            st.dataframe(thresh)

    # -----------------------------------------------------
    # Probability distribution
    # -----------------------------------------------------

        st.header("Probability Distribution")

        fig, ax = plt.subplots()

        good = holdout[holdout["y_true"]==0]["y_prob"]
        bad = holdout[holdout["y_true"]==1]["y_prob"]

        ax.hist(good,bins=40,alpha=0.5,label="Good")
        ax.hist(bad,bins=40,alpha=0.5,label="Bad")

        ax.legend()

        st.pyplot(fig)

    # -----------------------------------------------------
    # Feature importance
    # -----------------------------------------------------

    st.header("Feature Importance")

    if not importances.empty:

        fig, ax = plt.subplots()

        plot_df = importances.sort_values("importance")

        ax.barh(
            plot_df["feature"],
            plot_df["importance"]
        )

        st.pyplot(fig)

        st.dataframe(importances)

    # -----------------------------------------------------
    # QC output table
    # -----------------------------------------------------

    st.header("QC Output")

    st.dataframe(df)

    csv = df.to_csv(index=True).encode("utf-8")

    st.download_button(
        "Download QC Output",
        csv,
        "qc_results.csv",
        "text/csv"
    )
