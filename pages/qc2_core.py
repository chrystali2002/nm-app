#!/usr/bin/env python3
"""
qc2_core.py

Enhanced QC pipeline with:

• class weighting
• SMOTE oversampling
• automatic probability threshold tuning
• anomaly-focused metrics
• ROC + PR evaluation support

Designed to be used by Streamlit app.py
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
    balanced_accuracy_score,
    average_precision_score,
    roc_auc_score,
)

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
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
@dataclass
class FigureOptions:
    generate_figures: bool = True
    time_series: bool = True
    zooms: bool = True
    flatline: bool = True
    spike: bool = True
    climatology: bool = True
    spatial: bool = True
    daily_counts: bool = True
    confusion_matrix: bool = True
    holdout_probability: bool = True
    yearly_comparison: bool = True
    feature_importance: bool = True


@dataclass
class QCArgs:
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

    # NEW OPTIONS
    use_class_weighting: bool = True
    use_smote: bool = True
    smote_k_neighbors: int = 5

    auto_tune_threshold: bool = True
    threshold_metric: str = "f1_bad"
    min_precision_bad: float = 0.50

    threshold_scan_min: float = 0.05
    threshold_scan_max: float = 0.95
    threshold_scan_steps: int = 37

    figure_options: FigureOptions = field(default_factory=FigureOptions)
    
def fit_resample_with_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_smote: bool = True,
    smote_k_neighbors: int = 5,
):

    if not use_smote:
        return X_train, y_train

    bad_count = int((y_train == 1).sum())

    if bad_count < 3:
        return X_train, y_train

    k_neighbors = min(smote_k_neighbors, bad_count - 1)

    smote = SMOTE(
        random_state=42,
        k_neighbors=k_neighbors
    )

    X_res, y_res = smote.fit_resample(X_train, y_train)

    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)

    return X_res, y_res


def threshold_search(y_true, y_prob, args: QCArgs):

    thresholds = np.linspace(
        args.threshold_scan_min,
        args.threshold_scan_max,
        args.threshold_scan_steps
    )

    rows = []

    for t in thresholds:

        y_pred = (y_prob >= t).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0
        )

        rows.append({
            "threshold": t,
            "precision_bad": precision,
            "recall_bad": recall,
            "f1_bad": f1,
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        })

    return pd.DataFrame(rows)


def choose_best_threshold(df, args: QCArgs):

    if args.threshold_metric == "recall_bad":

        eligible = df[df["precision_bad"] >= args.min_precision_bad]

        if not eligible.empty:
            return float(
                eligible.sort_values(
                    "recall_bad",
                    ascending=False
                ).iloc[0]["threshold"]
            )

        return float(
            df.sort_values(
                "recall_bad",
                ascending=False
            ).iloc[0]["threshold"]
        )

    if args.threshold_metric == "balanced_accuracy":

        return float(
            df.sort_values(
                "balanced_accuracy",
                ascending=False
            ).iloc[0]["threshold"]
        )

    return float(
        df.sort_values(
            "f1_bad",
            ascending=False
        ).iloc[0]["threshold"]
    )


def make_classifier(model_name: str, use_class_weighting=True):

    cw = "balanced" if use_class_weighting else None

    if model_name == "random_forest":

        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=14,
            min_samples_leaf=3,
            class_weight=cw,
            random_state=42,
            n_jobs=-1
        )

    elif model_name == "extra_trees":

        clf = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=14,
            min_samples_leaf=3,
            class_weight=cw,
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
            class_weight=cw,
            random_state=42
        )

    else:
        raise ValueError("Unsupported model")

pipe = make_classifier(
    args.ml_model,
    args.use_class_weighting
)

X_train_res, y_train_res = fit_resample_with_smote(
    X_train,
    y_train,
    args.use_smote,
    args.smote_k_neighbors
)

pipe.fit(X_train_res, y_train_res)

y_prob = pipe.predict_proba(X_test)[:, 1]

threshold_df = threshold_search(
    y_test,
    y_prob,
    args
)

if args.auto_tune_threshold:

    best_threshold = choose_best_threshold(
        threshold_df,
        args
    )

else:

    best_threshold = args.ml_prob_threshold

y_pred = (y_prob >= best_threshold).astype(int)


warnings.filterwarnings("ignore")
