import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix, matthews_corrcoef
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup
from geopy.distance import geodesic
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="ML vs Rule-Based Comparison",
    page_icon="🔄",
    layout="wide"
)

st.title("🔄 ML vs Rule-Based: Comprehensive Comparison")
st.markdown("""
This page allows you to select specific stations and compare multiple ML-based vs rule-based anomaly detection methods.
The rule-based system includes **spatial checks using nearest neighbor stations** for robust quality control.
""")

# =============================================================================
# Helper Functions
# =============================================================================
@st.cache_data(show_spinner=False)
def load_station_metadata(state_code):
    """Load and filter station metadata."""
    isd_metadata_url = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
    stations = pd.read_csv(isd_metadata_url)

    state_stations = stations[
        (stations["STATE"] == state_code) &
        (~stations["LAT"].isna()) &
        (~stations["LON"].isna())
    ].copy()

    # Make sure key fields exist and are usable
    for col in ["USAF", "WBAN", "STATION NAME", "LAT", "LON"]:
        if col not in state_stations.columns:
            raise ValueError(f"Required column '{col}' not found in station metadata.")

    return state_stations


@st.cache_data(show_spinner=False)
def get_access_files(year):
    """Get list of Access files for a given year."""
    access_base_url = "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/"
    url = access_base_url.format(year=year)

    try:
        res = requests.get(url, timeout=30)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        files = [a.text for a in soup.find_all("a") if a.text.endswith(".csv")]
        return files, url
    except Exception:
        return [], None


@st.cache_data(show_spinner=False)
def load_access_csv(url: str) -> pd.DataFrame:
    """Load and process ACCESS CSV file safely."""
    try:
        df = pd.read_csv(url, low_memory=False)

        if "DATE" not in df.columns or "TMP" not in df.columns:
            return pd.DataFrame()

        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.dropna(subset=["DATE"]).copy()
        df = df.set_index("DATE").sort_index()

        # TMP format is usually something like "0123,1"
        tmp_split = df["TMP"].astype(str).str.split(",", expand=True)
        df["TMP_val"] = pd.to_numeric(tmp_split[0], errors="coerce")
        df["TMP_val"] = df["TMP_val"].replace(9999, np.nan)
        df["T_air"] = df["TMP_val"] / 10.0  # °C

        # Keep original columns for reference, but T_air is the main analysis variable
        return df

    except Exception:
        return pd.DataFrame()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create clean numeric features for ML models only."""
    out = pd.DataFrame(index=df.index)

    out["T_air"] = pd.to_numeric(df["T_air"], errors="coerce")

    # Time features
    out["hour"] = df.index.hour
    out["day"] = df.index.day
    out["dayofweek"] = df.index.dayofweek
    out["month"] = df.index.month
    out["dayofyear"] = df.index.dayofyear

    # Cyclical encoding
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    # Lag features
    for lag in [1, 3, 6, 12, 24]:
        out[f"temp_lag_{lag}h"] = out["T_air"].shift(lag)

    # Rolling statistics
    for window in [6, 12, 24]:
        out[f"rolling_mean_{window}h"] = out["T_air"].rolling(window, min_periods=3).mean()
        out[f"rolling_std_{window}h"] = out["T_air"].rolling(window, min_periods=3).std()
        out[f"rolling_min_{window}h"] = out["T_air"].rolling(window, min_periods=3).min()
        out[f"rolling_max_{window}h"] = out["T_air"].rolling(window, min_periods=3).max()

    # Rate of change
    out["temp_diff_1h"] = out["T_air"].diff(1)
    out["temp_diff_3h"] = out["T_air"].diff(3)
    out["temp_diff_6h"] = out["T_air"].diff(6)

    out = out.apply(pd.to_numeric, errors="coerce")
    return out


def prepare_numeric_features(df: pd.DataFrame, features: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ensure features are numeric, finite, and aligned."""
    if features is None:
        feat = pd.DataFrame(index=df.index)
        feat["T_air"] = pd.to_numeric(df["T_air"], errors="coerce")
    else:
        feat = features.copy()

    feat = feat.select_dtypes(include=[np.number])
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()

    return feat


def get_ml_flags_isolation_forest(df, contamination=0.05, features=None):
    """Generate ML flags using Isolation Forest."""
    features = prepare_numeric_features(df, features)

    if len(features) < 50 or features.shape[1] == 0:
        return None, None

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
        bootstrap=True
    )
    preds = iso_forest.fit_predict(features_scaled)
    scores = iso_forest.score_samples(features_scaled)

    ml_flags = pd.Series(False, index=df.index)
    ml_flags.loc[features.index] = (preds == -1)

    return ml_flags, scores


def get_ml_flags_lof(df, contamination=0.05, n_neighbors=20, features=None):
    """Generate ML flags using Local Outlier Factor."""
    features = prepare_numeric_features(df, features)

    if len(features) < 50 or features.shape[1] == 0:
        return None, None

    n_neighbors = min(n_neighbors, max(2, len(features) - 1))

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    lof = LocalOutlierFactor(
        contamination=contamination,
        n_neighbors=n_neighbors,
        novelty=False
    )
    preds = lof.fit_predict(features_scaled)

    ml_flags = pd.Series(False, index=df.index)
    ml_flags.loc[features.index] = (preds == -1)

    return ml_flags, None


def get_ml_flags_one_class_svm(df, contamination=0.05, features=None):
    """Generate ML flags using One-Class SVM."""
    features = prepare_numeric_features(df, features)

    if len(features) < 50 or features.shape[1] == 0:
        return None, None

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    svm = OneClassSVM(
        nu=contamination,
        kernel="rbf",
        gamma="auto"
    )
    preds = svm.fit_predict(features_scaled)

    ml_flags = pd.Series(False, index=df.index)
    ml_flags.loc[features.index] = (preds == -1)

    return ml_flags, None


def get_ml_flags_ensemble(df, contamination=0.05, features=None):
    """Ensemble method combining multiple algorithms."""
    features = prepare_numeric_features(df, features)

    if len(features) < 50 or features.shape[1] == 0:
        return None, None

    n_neighbors = min(20, max(2, len(features) - 1))

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    lof = LocalOutlierFactor(contamination=contamination, n_neighbors=n_neighbors, novelty=False)
    svm = OneClassSVM(nu=contamination, kernel="rbf", gamma="auto")

    iso_preds = iso_forest.fit_predict(features_scaled)
    lof_preds = lof.fit_predict(features_scaled)
    svm_preds = svm.fit_predict(features_scaled)

    ensemble_votes = (
        (iso_preds == -1).astype(int) +
        (lof_preds == -1).astype(int) +
        (svm_preds == -1).astype(int)
    )
    ensemble_preds = ensemble_votes >= 2

    ml_flags = pd.Series(False, index=df.index)
    ml_flags.loc[features.index] = ensemble_preds

    return ml_flags, ensemble_votes


def get_ml_flags_hybrid(df, rule_flags, contamination=0.05, features=None):
    """Hybrid approach: uses rule flags as training labels for supervised learning."""
    features = prepare_numeric_features(df, features)

    if len(features) < 100 or features.shape[1] == 0:
        return None, None

    aligned_rules = rule_flags.reindex(features.index).fillna(False).astype(int)

    if aligned_rules.sum() < 10 or aligned_rules.nunique() < 2:
        return None, None

    X = features.values
    y = aligned_rules.values

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    except ValueError:
        return None, None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X_train_scaled, y_train)

    X_all_scaled = scaler.transform(X)
    pred_proba = rf.predict_proba(X_all_scaled)[:, 1]

    threshold = np.percentile(pred_proba, (1 - contamination) * 100)
    hybrid_preds = pred_proba >= threshold

    ml_flags = pd.Series(False, index=df.index)
    ml_flags.loc[features.index] = hybrid_preds

    return ml_flags, pred_proba


# =============================================================================
# Sidebar - Station Selection
# =============================================================================
st.sidebar.header("🔧 Station Selection")

state_code = st.sidebar.text_input("State Code", value="NM").upper().strip()

with st.spinner(f"Loading {state_code} stations..."):
    state_stations = load_station_metadata(state_code)

if len(state_stations) == 0:
    st.sidebar.error(f"No stations found for state {state_code}")
    st.stop()

station_options = []
for _, row in state_stations.iterrows():
    station_options.append({
        "USAF": row["USAF"],
        "WBAN": row["WBAN"],
        "Name": row["STATION NAME"],
        "LAT": row["LAT"],
        "LON": row["LON"]
    })

station_df = pd.DataFrame(station_options)

primary_idx = st.sidebar.selectbox(
    "Choose primary station",
    options=range(len(station_df)),
    format_func=lambda x: f"{station_df.iloc[x]['Name']} ({station_df.iloc[x]['USAF']}-{station_df.iloc[x]['WBAN']})",
    index=0
)

primary_station = station_df.iloc[primary_idx]

st.sidebar.subheader("👥 Neighbor Station")

neighbor_option = st.sidebar.radio(
    "Neighbor selection",
    options=["Auto-find nearest", "Manual select"]
)

neighbor_station = None
min_dist = None

if neighbor_option == "Auto-find nearest":
    st.sidebar.info("Finding nearest station...")

    other_stations = station_df[station_df.index != primary_idx].copy()

    min_dist = float("inf")
    nearest_idx = None

    for idx, row in other_stations.iterrows():
        dist = geodesic(
            (primary_station["LAT"], primary_station["LON"]),
            (row["LAT"], row["LON"])
        ).km

        if dist < min_dist:
            min_dist = dist
            nearest_idx = idx

    if nearest_idx is not None:
        neighbor_station = station_df.loc[nearest_idx]
        st.sidebar.success(f"Found: {neighbor_station['Name']} ({min_dist:.1f} km away)")
    else:
        st.sidebar.error("No neighbor stations found")
        neighbor_station = None
        min_dist = None

else:
    neighbor_idx = st.sidebar.selectbox(
        "Choose neighbor station",
        options=[i for i in range(len(station_df)) if i != primary_idx],
        format_func=lambda x: f"{station_df.iloc[x]['Name']} ({station_df.iloc[x]['USAF']}-{station_df.iloc[x]['WBAN']})"
    )
    neighbor_station = station_df.iloc[neighbor_idx]

    min_dist = geodesic(
        (primary_station["LAT"], primary_station["LON"]),
        (neighbor_station["LAT"], neighbor_station["LON"])
    ).km
    st.sidebar.info(f"Distance: {min_dist:.1f} km")

st.sidebar.subheader("📅 Year Range")
year_range = st.sidebar.slider("Select Years", 2000, 2025, (2018, 2023))
years = range(year_range[0], year_range[1] + 1)

# =============================================================================
# ML Method Selection
# =============================================================================
st.sidebar.header("🤖 ML Method Selection")

ml_method = st.sidebar.selectbox(
    "Choose ML Method",
    options=[
        "Isolation Forest (Default)",
        "Local Outlier Factor (LOF)",
        "One-Class SVM",
        "Ensemble (Voting)",
        "Hybrid (Rule-based + ML)"
    ],
    help="Different methods have different strengths"
)

st.sidebar.subheader("⚙️ Parameter Tuning")

col1, col2 = st.sidebar.columns(2)
with col1:
    contamination = st.number_input(
        "Contamination",
        min_value=0.01,
        max_value=0.30,
        value=0.05,
        step=0.01,
        format="%.2f",
        help="Expected proportion of anomalies"
    )

with col2:
    if ml_method == "Local Outlier Factor (LOF)":
        n_neighbors = st.number_input(
            "Neighbors",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Number of neighbors for LOF"
        )
    else:
        n_neighbors = 20

use_engineered_features = st.sidebar.checkbox(
    "Use Engineered Features",
    value=True,
    help="Add time features, lags, and rolling statistics"
)

with st.sidebar.expander("📘 Parameter Tuning Guide"):
    st.markdown("""
    ### When to Adjust Parameters

    **Decrease Contamination (0.01-0.03):**
    - Data has clear patterns with few anomalies
    - ML flags > 15% of data
    - Many false positives in review
    - Temperature range is large (>60°C)

    **Increase Contamination (0.08-0.15):**
    - Data is noisy with many anomalies
    - ML flags < 2% but you expect more
    - Missing known events
    - Temperature range is small (<30°C)

    **Default (0.05):**
    - Starting point for new data
    - Balanced sensitivity
    - General purpose use
    """)

# =============================================================================
# Load Data for Selected Stations
# =============================================================================
st.header("📡 Loading Station Data")

progress_bar = st.progress(0)
status_text = st.empty()

status_text.text(f"Loading primary station: {primary_station['Name']}...")

usaf = str(primary_station["USAF"]).zfill(6)
wban = str(primary_station["WBAN"]).zfill(5)
filename = f"{usaf}{wban}.csv"

all_primary_data = []

for j, year in enumerate(years):
    progress = j / max(1, (len(years) * 2))
    progress_bar.progress(progress)

    url = f"https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{filename}"
    df_year = load_access_csv(url)

    if not df_year.empty:
        all_primary_data.append(df_year)

if all_primary_data:
    df_primary = pd.concat(all_primary_data).sort_index()
    df_primary = df_primary[~df_primary.index.duplicated(keep="first")].copy()
    st.success(f"✅ Loaded {len(df_primary):,} records for primary station: {primary_station['Name']}")
else:
    st.error("❌ No data found for primary station in selected years")
    st.stop()

df_neighbor = None
neighbor_name = None
neighbor_dist = None

if neighbor_station is not None:
    status_text.text(f"Loading neighbor station: {neighbor_station['Name']}...")

    usaf_neighbor = str(neighbor_station["USAF"]).zfill(6)
    wban_neighbor = str(neighbor_station["WBAN"]).zfill(5)
    neighbor_filename = f"{usaf_neighbor}{wban_neighbor}.csv"

    all_neighbor_data = []

    for j, year in enumerate(years):
        progress = 0.5 + (j / max(1, len(years)) * 0.5)
        progress_bar.progress(progress)

        url = f"https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{neighbor_filename}"
        df_year = load_access_csv(url)

        if not df_year.empty:
            all_neighbor_data.append(df_year)

    if all_neighbor_data:
        df_neighbor = pd.concat(all_neighbor_data).sort_index()
        df_neighbor = df_neighbor[~df_neighbor.index.duplicated(keep="first")].copy()
        neighbor_name = neighbor_station["Name"]
        neighbor_dist = min_dist
        st.success(f"✅ Loaded {len(df_neighbor):,} records for neighbor: {neighbor_name} ({neighbor_dist:.1f} km)")
    else:
        st.warning("⚠️ No data found for neighbor station - spatial checks will be disabled")

progress_bar.progress(1.0)
status_text.text("Data loading complete!")

st.markdown("---")

# =============================================================================
# Data Diagnostics
# =============================================================================
st.sidebar.subheader("📊 Data Diagnostics")

total_points = len(df_primary)
data_years = df_primary.index.year.nunique()
data_range = df_primary["T_air"].max() - df_primary["T_air"].min()
data_std = df_primary["T_air"].std()

st.sidebar.info(f"""
**Primary Station:**
- Points: {total_points:,}
- Years: {data_years}
- Range: {data_range:.1f}°C
- Std: {data_std:.2f}°C
""")

if df_neighbor is not None:
    neighbor_points = len(df_neighbor)
    st.sidebar.info(f"""
**Neighbor Station:**
- Points: {neighbor_points:,}
- Distance: {neighbor_dist:.1f} km
""")

st.sidebar.subheader("💡 Recommendations")

if data_range > 60:
    st.sidebar.info("🌡️ Large range - consider lower contamination (0.02-0.03)")
elif data_range < 30:
    st.sidebar.info("🌡️ Small range - consider higher contamination (0.05-0.08)")

if data_std > 10:
    st.sidebar.info("📊 High variability - start with lower contamination (0.03)")
elif data_std < 5:
    st.sidebar.info("📊 Low variability - start with higher contamination (0.07)")

# =============================================================================
# Apply Rule-Based QA/QC
# =============================================================================
st.header("📊 Applying QA/QC Rules")

with st.spinner("Applying rule-based QA/QC with spatial checks..."):
    df_primary["T_air"] = pd.to_numeric(df_primary["T_air"], errors="coerce")

    df_primary["flag_range"] = (df_primary["T_air"] < -40) | (df_primary["T_air"] > 55)
    df_primary["dT"] = df_primary["T_air"].diff()
    df_primary["flag_spike"] = df_primary["dT"].abs() > 8
    df_primary["flag_flat"] = df_primary["T_air"].rolling(12, min_periods=10).std() < 0.1
    df_primary["heatwave"] = df_primary["T_air"] > 40

    if df_neighbor is not None:
        df_neighbor = df_neighbor.copy()
        df_neighbor["T_air"] = pd.to_numeric(df_neighbor["T_air"], errors="coerce")

        common_index = df_primary.index.intersection(df_neighbor.index)

        if len(common_index) > 0:
            df_primary_aligned = df_primary.loc[common_index, ["T_air"]].copy()
            df_neighbor_aligned = df_neighbor.loc[common_index, ["T_air"]].copy()

            df_primary_aligned["anom"] = (
                df_primary_aligned["T_air"] -
                df_primary_aligned["T_air"].rolling(24, min_periods=18).mean()
            )
            df_neighbor_aligned["anom"] = (
                df_neighbor_aligned["T_air"] -
                df_neighbor_aligned["T_air"].rolling(24, min_periods=18).mean()
            )

            spatial_flags_aligned = (
                (df_primary_aligned["anom"] - df_neighbor_aligned["anom"]).abs() > 6
            ).fillna(False)

            df_primary["flag_spatial"] = False
            df_primary.loc[common_index, "flag_spatial"] = spatial_flags_aligned.values

            st.info(f"✅ Spatial check enabled using {neighbor_name} ({neighbor_dist:.1f} km away)")
        else:
            df_primary["flag_spatial"] = False
            st.warning("⚠️ No overlapping timestamps with neighbor station - spatial check disabled")
    else:
        df_primary["flag_spatial"] = False
        st.warning("⚠️ Spatial check disabled - no neighbor data available")

    df_primary["final_flag"] = (
        df_primary["flag_range"] |
        df_primary["flag_spike"] |
        df_primary["flag_flat"] |
        (df_primary["flag_spatial"] & ~df_primary["heatwave"])
    )

    rule_based_flags = df_primary["final_flag"].fillna(False).astype(bool).copy()

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Range Flags", f"{int(df_primary['flag_range'].sum()):,}")
with col2:
    st.metric("Spike Flags", f"{int(df_primary['flag_spike'].sum()):,}")
with col3:
    st.metric("Flatline Flags", f"{int(df_primary['flag_flat'].sum()):,}")
with col4:
    st.metric("Spatial Flags", f"{int(df_primary['flag_spatial'].sum()):,}")
with col5:
    st.metric("Heatwaves", f"{int(df_primary['heatwave'].sum()):,}")

st.markdown("---")

# =============================================================================
# Prepare ML Features
# =============================================================================
if use_engineered_features:
    df_features = engineer_features(df_primary)
    features = prepare_numeric_features(df_primary, df_features)
else:
    features = prepare_numeric_features(df_primary, None)

with st.expander("🔎 Feature Diagnostics"):
    st.write("Feature shape:", features.shape)
    st.write("Feature columns:", list(features.columns))
    st.write("Feature dtypes:")
    st.dataframe(features.dtypes.astype(str).reset_index().rename(columns={"index": "Feature", 0: "Dtype"}), use_container_width=True)

if features.empty or features.shape[1] == 0:
    st.error("No usable numeric features were created for ML analysis.")
    st.stop()

# =============================================================================
# Generate ML Flags
# =============================================================================
with st.spinner(f"Generating {ml_method} flags..."):
    if ml_method == "Isolation Forest (Default)":
        ml_flags, ml_scores = get_ml_flags_isolation_forest(df_primary, contamination, features)
    elif ml_method == "Local Outlier Factor (LOF)":
        ml_flags, ml_scores = get_ml_flags_lof(df_primary, contamination, n_neighbors, features)
    elif ml_method == "One-Class SVM":
        ml_flags, ml_scores = get_ml_flags_one_class_svm(df_primary, contamination, features)
    elif ml_method == "Ensemble (Voting)":
        ml_flags, ml_scores = get_ml_flags_ensemble(df_primary, contamination, features)
    elif ml_method == "Hybrid (Rule-based + ML)":
        ml_flags, ml_scores = get_ml_flags_hybrid(df_primary, rule_based_flags, contamination, features)
    else:
        ml_flags, ml_scores = None, None

if ml_flags is None:
    st.error("Insufficient or unsuitable data for ML analysis.")
    st.stop()

ml_flags = ml_flags.fillna(False).astype(bool)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Current Contamination", f"{contamination:.2f}")

with col2:
    flag_rate = ml_flags.sum() / len(ml_flags) * 100
    st.metric("Actual ML Flag Rate", f"{flag_rate:.2f}%",
              delta=f"{flag_rate - contamination * 100:.1f}% vs target")

with col3:
    if abs(flag_rate - contamination * 100) > 5:
        st.warning("⚠️ Large difference from target")
        if flag_rate > contamination * 100:
            st.info("💡 Consider decreasing contamination")
        else:
            st.info("💡 Consider increasing contamination")

# =============================================================================
# Sensitivity Analysis
# =============================================================================
if st.sidebar.checkbox("Show Parameter Sensitivity Analysis"):
    st.subheader("📈 Parameter Sensitivity Analysis")

    test_contaminations = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15]
    test_rates = []

    with st.spinner("Testing different contamination values..."):
        for test_contam in test_contaminations:
            if ml_method == "Isolation Forest (Default)":
                test_flags, _ = get_ml_flags_isolation_forest(df_primary, test_contam, features)
            elif ml_method == "Local Outlier Factor (LOF)":
                test_flags, _ = get_ml_flags_lof(df_primary, test_contam, n_neighbors, features)
            elif ml_method == "One-Class SVM":
                test_flags, _ = get_ml_flags_one_class_svm(df_primary, test_contam, features)
            elif ml_method == "Ensemble (Voting)":
                test_flags, _ = get_ml_flags_ensemble(df_primary, test_contam, features)
            elif ml_method == "Hybrid (Rule-based + ML)":
                test_flags, _ = get_ml_flags_hybrid(df_primary, rule_based_flags, test_contam, features)
            else:
                test_flags = None

            if test_flags is not None:
                test_rates.append(test_flags.sum() / len(test_flags) * 100)
            else:
                test_rates.append(0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_contaminations,
        y=test_rates,
        mode="lines+markers",
        name="Actual Flag Rate",
        line=dict(color="blue", width=2),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 0.16],
        y=[0, 16],
        mode="lines",
        name="Ideal 1:1 Line",
        line=dict(color="red", dash="dash")
    ))

    fig.update_layout(
        title=f"Parameter Sensitivity - {ml_method}",
        xaxis_title="Contamination Parameter",
        yaxis_title="Actual Flag Rate (%)",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# Create Comparison DataFrame
# =============================================================================
comparison_df = pd.DataFrame(index=df_primary.index)
comparison_df["Temperature"] = pd.to_numeric(df_primary["T_air"], errors="coerce")
comparison_df["Rule Flag"] = rule_based_flags.astype(int)
comparison_df["ML Flag"] = ml_flags.astype(int)
comparison_df["Flag Type"] = "Neither"

comparison_df.loc[
    (comparison_df["Rule Flag"] == 1) & (comparison_df["ML Flag"] == 1), "Flag Type"
] = "Both"
comparison_df.loc[
    (comparison_df["Rule Flag"] == 1) & (comparison_df["ML Flag"] == 0), "Flag Type"
] = "Rule Only"
comparison_df.loc[
    (comparison_df["Rule Flag"] == 0) & (comparison_df["ML Flag"] == 1), "Flag Type"
] = "ML Only"

comparison_df["Hour"] = comparison_df.index.hour
comparison_df["Month"] = comparison_df.index.month
comparison_df["DayOfWeek"] = comparison_df.index.dayofweek
comparison_df["Season"] = comparison_df["Month"].map({
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall"
})

for window in [6, 12, 24]:
    comparison_df[f"Rolling Mean {window}h"] = df_primary["T_air"].rolling(window, min_periods=3).mean()
    comparison_df[f"Rolling Std {window}h"] = df_primary["T_air"].rolling(window, min_periods=3).std()

# Metrics used in multiple tabs
y_true = comparison_df["Rule Flag"].values
y_pred = comparison_df["ML Flag"].values

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
kappa = cohen_kappa_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

# =============================================================================
# Tabs
# =============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview Statistics",
    "🎯 Agreement Analysis",
    "📈 Temporal Patterns",
    "🌡️ Temperature Analysis",
    "📉 Statistical Comparison",
    "🔍 Case Studies"
])

with tab1:
    st.header("📊 Overview Statistics")

    total_points = len(comparison_df)
    rule_count = int((comparison_df["Rule Flag"] == 1).sum())
    ml_count = int((comparison_df["ML Flag"] == 1).sum())
    both_count = int(((comparison_df["Rule Flag"] == 1) & (comparison_df["ML Flag"] == 1)).sum())
    rule_only = int(((comparison_df["Rule Flag"] == 1) & (comparison_df["ML Flag"] == 0)).sum())
    ml_only = int(((comparison_df["Rule Flag"] == 0) & (comparison_df["ML Flag"] == 1)).sum())

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Points", f"{total_points:,}")
    with col2:
        st.metric("Rule Flags", f"{rule_count:,}", delta=f"{(rule_count / total_points * 100):.1f}%")
    with col3:
        st.metric("ML Flags", f"{ml_count:,}", delta=f"{(ml_count / total_points * 100):.1f}%", delta_color="inverse")
    with col4:
        st.metric("Both", f"{both_count:,}")
    with col5:
        agreement = (both_count + (total_points - rule_count - ml_count + both_count)) / total_points * 100
        st.metric("Agreement Rate", f"{agreement:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        flag_counts = comparison_df["Flag Type"].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=flag_counts.index,
            values=flag_counts.values,
            hole=0.4,
            marker_colors=["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"],
            textinfo="label+percent",
            textposition="outside"
        )])
        fig.update_layout(title="Distribution of Flag Types", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        summary_stats = []
        for flag_type in ["ML Only", "Rule Only", "Both", "Neither"]:
            subset = comparison_df[comparison_df["Flag Type"] == flag_type]
            if len(subset) > 0:
                summary_stats.append({
                    "Flag Type": flag_type,
                    "Count": len(subset),
                    "Percentage": f"{len(subset) / total_points * 100:.2f}%",
                    "Mean Temp": f"{subset['Temperature'].mean():.2f}°C",
                    "Std Dev": f"{subset['Temperature'].std():.2f}°C"
                })
        st.dataframe(pd.DataFrame(summary_stats), use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("🔍 Key Insights")
    insights = []

    if ml_count > rule_count and rule_count > 0:
        ratio = ml_count / rule_count
        insights.append(f"✅ **ML is {ratio:.1f}x more sensitive** than rule-based")
    elif ml_count > 0:
        ratio = rule_count / ml_count
        insights.append(f"⚠️ **Rule-based is {ratio:.1f}x more sensitive** than ML")

    overlap_ratio = both_count / min(rule_count, ml_count) * 100 if min(rule_count, ml_count) > 0 else 0
    if overlap_ratio > 80:
        insights.append(f"🎯 **High agreement** ({overlap_ratio:.1f}% overlap)")
    elif overlap_ratio > 50:
        insights.append(f"🟡 **Moderate agreement** ({overlap_ratio:.1f}% overlap)")
    else:
        insights.append(f"🔴 **Low agreement** ({overlap_ratio:.1f}% overlap)")

    if ml_only > rule_only:
        insights.append(f"🤖 **ML specializes** in finding {ml_only} unique patterns")
    else:
        insights.append(f"📏 **Rules specialize** in finding {rule_only} unique patterns")

    for insight in insights:
        st.info(insight)

with tab2:
    st.header("🎯 Agreement Analysis")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
        st.metric("Precision", f"{precision:.3f}")
    with col2:
        st.metric("Recall", f"{recall:.3f}")
        st.metric("Specificity", f"{specificity:.3f}")
    with col3:
        st.metric("F1 Score", f"{f1:.3f}")
        st.metric("Cohen's Kappa", f"{kappa:.3f}")
    with col4:
        st.metric("MCC", f"{mcc:.3f}")
        st.metric("Youden's J", f"{recall + specificity - 1:.3f}")

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["ML: Normal", "ML: Flagged"],
            y=["Rule: Normal", "Rule: Flagged"],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            showscale=False
        ))
        fig.update_layout(title="Confusion Matrix", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📖 Interpretation")
        interpretations = [
            f"**True Negatives:** {tn:,} points - Both agree normal",
            f"**True Positives:** {tp:,} points - Both agree anomalous",
            f"**False Positives:** {fp:,} points - ML flagged only",
            f"**False Negatives:** {fn:,} points - Rules flagged only"
        ]
        for interp in interpretations:
            st.markdown(interp)

        if kappa > 0.8:
            st.success(f"**Almost perfect agreement** (Kappa = {kappa:.3f})")
        elif kappa > 0.6:
            st.success(f"**Substantial agreement** (Kappa = {kappa:.3f})")
        elif kappa > 0.4:
            st.warning(f"**Moderate agreement** (Kappa = {kappa:.3f})")
        else:
            st.error(f"**Slight agreement** (Kappa = {kappa:.3f})")

with tab3:
    st.header("📈 Temporal Pattern Analysis")

    hourly_stats = comparison_df.groupby("Hour").agg({
        "Rule Flag": "mean",
        "ML Flag": "mean",
        "Temperature": "mean"
    }).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=hourly_stats["Hour"],
            y=hourly_stats["Rule Flag"] * 100,
            name="Rule Flag Rate",
            marker_color="red",
            opacity=0.7
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Bar(
            x=hourly_stats["Hour"],
            y=hourly_stats["ML Flag"] * 100,
            name="ML Flag Rate",
            marker_color="orange",
            opacity=0.7
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=hourly_stats["Hour"],
            y=hourly_stats["Temperature"],
            name="Avg Temperature",
            line=dict(color="blue", width=2),
            mode="lines+markers"
        ),
        secondary_y=True
    )

    fig.update_layout(title="Flag Rates by Hour of Day", height=400, barmode="group")
    fig.update_yaxes(title_text="Flag Rate (%)", secondary_y=False)
    fig.update_yaxes(title_text="Average Temperature (°C)", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    hourly_diff = hourly_stats.copy()
    hourly_diff["Diff"] = abs(hourly_stats["ML Flag"] - hourly_stats["Rule Flag"]) * 100
    max_diff_hour = hourly_diff.loc[hourly_diff["Diff"].idxmax()]

    st.info(f"""
**Peak Disagreement Hour**: {int(max_diff_hour['Hour']):02d}:00
- Rule Rate: {max_diff_hour['Rule Flag'] * 100:.1f}%
- ML Rate: {max_diff_hour['ML Flag'] * 100:.1f}%
- Difference: {max_diff_hour['Diff']:.1f}%
""")

with tab4:
    st.header("🌡️ Temperature Analysis")

    fig = go.Figure()

    for flag_type, color in zip(
        ["ML Only", "Rule Only", "Both", "Neither"],
        ["orange", "red", "purple", "gray"]
    ):
        subset = comparison_df[comparison_df["Flag Type"] == flag_type]["Temperature"].dropna()
        if len(subset) > 0:
            fig.add_trace(go.Violin(
                y=subset,
                name=flag_type,
                box_visible=True,
                meanline_visible=True,
                fillcolor=color,
                opacity=0.7,
                line_color="black"
            ))

    fig.update_layout(title="Temperature Distribution by Flag Type", height=500)
    st.plotly_chart(fig, use_container_width=True)

    bins = np.arange(-30, 41, 5)
    labels = [f"{bins[i]}-{bins[i + 1]}°C" for i in range(len(bins) - 1)]

    temp_bin_df = comparison_df.copy()
    temp_bin_df["TempBin"] = pd.cut(temp_bin_df["Temperature"], bins=bins, labels=labels)

    bin_stats = temp_bin_df.groupby("TempBin", observed=False).agg({
        "Rule Flag": "sum",
        "ML Flag": "sum",
        "Temperature": "count"
    }).rename(columns={"Temperature": "Total"})

    bin_stats["Rule Rate"] = np.where(bin_stats["Total"] > 0, (bin_stats["Rule Flag"] / bin_stats["Total"] * 100).round(1), 0)
    bin_stats["ML Rate"] = np.where(bin_stats["Total"] > 0, (bin_stats["ML Flag"] / bin_stats["Total"] * 100).round(1), 0)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=bin_stats.index.astype(str), y=bin_stats["Rule Rate"],
        name="Rule Rate", marker_color="red", opacity=0.7
    ))
    fig2.add_trace(go.Bar(
        x=bin_stats.index.astype(str), y=bin_stats["ML Rate"],
        name="ML Rate", marker_color="orange", opacity=0.7
    ))

    fig2.update_layout(title="Flag Rates by Temperature Range", height=400, barmode="group")
    st.plotly_chart(fig2, use_container_width=True)

with tab5:
    st.header("📉 Statistical Comparison")

    ml_only_temp = comparison_df[comparison_df["Flag Type"] == "ML Only"]["Temperature"].dropna()
    rule_only_temp = comparison_df[comparison_df["Flag Type"] == "Rule Only"]["Temperature"].dropna()

    if len(ml_only_temp) > 1 and len(rule_only_temp) > 1:
        t_stat, p_value = stats.ttest_ind(ml_only_temp, rule_only_temp, equal_var=False, nan_policy="omit")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("T-Statistic", f"{t_stat:.3f}")
            st.metric("P-Value", f"{p_value:.4f}")

            if p_value < 0.05:
                st.success("✅ **Significant difference** (p < 0.05)")
                st.markdown("ML and Rule-based detect **different temperature distributions**")
            else:
                st.warning("⚠️ **No significant difference** (p >= 0.05)")
                st.markdown("ML and Rule-based detect **similar temperature distributions**")

        with col2:
            pooled_std = np.sqrt((ml_only_temp.std() ** 2 + rule_only_temp.std() ** 2) / 2)
            cohens_d = (ml_only_temp.mean() - rule_only_temp.mean()) / pooled_std if pooled_std > 0 else 0

            st.metric("Cohen's d", f"{cohens_d:.3f}")

            if abs(cohens_d) > 0.8:
                st.info("Large effect size - **practically significant**")
            elif abs(cohens_d) > 0.5:
                st.info("Medium effect size")
            elif abs(cohens_d) > 0.2:
                st.info("Small effect size")
            else:
                st.info("Negligible effect size")
    else:
        st.warning("Not enough ML Only and Rule Only cases for statistical comparison.")

with tab6:
    st.header("🔍 Case Studies")

    st.subheader("🤖 ML Only Detections")
    ml_only_examples = comparison_df[comparison_df["Flag Type"] == "ML Only"].head(10)
    if len(ml_only_examples) > 0:
        st.dataframe(
            ml_only_examples[["Temperature", "Hour", "Month", "Rolling Mean 6h", "Rolling Std 6h"]],
            use_container_width=True
        )
        st.markdown("""
**What ML Only detections represent:**
- Contextual anomalies (unusual for time of day)
- Subtle patterns too small for spike detection
- Transition periods with rapid changes
- Boundary cases near range limits
""")
    else:
        st.info("No ML-only detections found.")

    st.markdown("---")

    st.subheader("📏 Rule Only Detections")
    rule_only_examples = comparison_df[comparison_df["Flag Type"] == "Rule Only"].head(10)
    if len(rule_only_examples) > 0:
        st.dataframe(
            rule_only_examples[["Temperature", "Hour", "Month", "Rolling Mean 6h", "Rolling Std 6h"]],
            use_container_width=True
        )
        st.markdown("""
**What Rule Only detections represent:**
- Flatlining (zero rolling std)
- Extreme temperature spikes
- Range violations
- Spatial inconsistencies
""")
    else:
        st.info("No Rule-only detections found.")

    st.markdown("---")

    st.subheader("📊 Method Comparison Summary")

    summary_data = {
        "Aspect": ["Agreement Recall", "Agreement Specificity", "Pattern Detection", "Spatial Awareness", "Best For"],
        ml_method: [
            f"{recall:.2f}",
            f"{specificity:.2f}",
            "Contextual anomalies",
            "Indirect only",
            "Subtle patterns"
        ],
        "Rule-Based": [
            "Reference method",
            "Reference method",
            "Clear violations",
            "Yes" if df_neighbor is not None else "No",
            "Obvious errors"
        ]
    }

    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    csv = comparison_df.to_csv()
    st.download_button(
        label="📥 Download Complete Comparison Data",
        data=csv,
        file_name=f"ml_rule_comparison_{ml_method.replace(' ', '_').replace('/', '_')}.csv",
        mime="text/csv"
    )

# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.markdown("""
**How to use this analysis:**
1. Select a primary station and its neighbor (auto-found or manual)
2. Data is loaded for both stations
3. Rule-based QA/QC includes spatial checks using neighbor data
4. Choose ML method and tune parameters
5. Review the 6 analysis tabs
6. Use insights to improve your QA/QC workflow

The spatial check compares anomalies between stations to detect inconsistencies.
""")
