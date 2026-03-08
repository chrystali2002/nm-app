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
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="ML vs Rule-Based Comparison",
    page_icon="🔄",
    layout="wide"
)

st.title("🔄 ML vs Rule-Based: Comprehensive Comparison")
st.markdown("""
This page allows you to select specific stations and compare multiple ML-based vs rule-based anomaly detection methods.
The rule-based system includes **spatial checks using configurable neighbor station distances** for robust quality control.
""")

# ============================================================================
# Helper Functions
# ============================================================================
@st.cache_data
def load_station_metadata(state_code):
    """Load and filter station metadata."""
    ISD_METADATA_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
    
    with st.spinner("Loading station metadata..."):
        stations = pd.read_csv(ISD_METADATA_URL)
        state_stations = stations[(stations['STATE'] == state_code) &
                                  (~stations['LAT'].isna()) & 
                                  (~stations['LON'].isna())]
    return state_stations

@st.cache_data
def get_access_files(year):
    """Get list of Access files for a given year."""
    ACCESS_BASE_URL = "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/"
    url = ACCESS_BASE_URL.format(year=year)
    
    try:
        res = requests.get(url)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        files = [a.text for a in soup.find_all('a') if a.text.endswith('.csv')]
        return files, url
    except:
        return [], None

def load_access_csv(url: str) -> pd.DataFrame:
    """Load and process Access CSV file."""
    try:
        df = pd.read_csv(url, low_memory=False)
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.set_index('DATE')
        df['TMP_val'] = df['TMP'].str.split(',').str[0].astype(float)
        df['TMP_val'] = df['TMP_val'].replace(9999, np.nan)
        df['T_air'] = df['TMP_val'] / 10.0  # °C
        return df
    except Exception as e:
        return pd.DataFrame()

def find_stations_by_distance(primary_lat, primary_lon, stations_df, exclude_filename, max_distance):
    """Find all stations within a specified distance of the primary station."""
    stations_within_distance = []
    
    for _, row in stations_df.iterrows():
        if row['FILENAME'] == exclude_filename:
            continue
        
        dist = geodesic((primary_lat, primary_lon), (row['LAT'], row['LON'])).km
        
        if dist <= max_distance:
            stations_within_distance.append({
                'name': row['STATION NAME'],
                'filename': row['FILENAME'],
                'lat': row['LAT'],
                'lon': row['LON'],
                'distance': dist
            })
    
    # Sort by distance
    stations_within_distance.sort(key=lambda x: x['distance'])
    return stations_within_distance

def engineer_features(df):
    """Create enhanced features for ML models"""
    df_features = df.copy()
    
    # Time features
    df_features['hour'] = df_features.index.hour
    df_features['day'] = df_features.index.day
    df_features['dayofweek'] = df_features.index.dayofweek
    df_features['month'] = df_features.index.month
    df_features['dayofyear'] = df_features.index.dayofyear
    
    # Cyclical encoding
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    
    # Lag features
    for lag in [1, 3, 6, 12, 24]:
        df_features[f'temp_lag_{lag}h'] = df_features['T_air'].shift(lag)
    
    # Rolling statistics
    for window in [6, 12, 24]:
        df_features[f'rolling_mean_{window}h'] = df_features['T_air'].rolling(window, min_periods=3).mean()
        df_features[f'rolling_std_{window}h'] = df_features['T_air'].rolling(window, min_periods=3).std()
        df_features[f'rolling_min_{window}h'] = df_features['T_air'].rolling(window, min_periods=3).min()
        df_features[f'rolling_max_{window}h'] = df_features['T_air'].rolling(window, min_periods=3).max()
    
    # Rate of change
    df_features['temp_diff_1h'] = df_features['T_air'].diff(1)
    df_features['temp_diff_3h'] = df_features['T_air'].diff(3)
    df_features['temp_diff_6h'] = df_features['T_air'].diff(6)
    
    return df_features

# ML Methods
def get_ml_flags_isolation_forest(df, contamination=0.05, features=None):
    """Generate ML flags using Isolation Forest"""
    if features is None:
        features = df[['T_air']].copy().dropna()
    else:
        features = features.dropna()
    
    if len(features) < 50:
        return None
    
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
    """Generate ML flags using Local Outlier Factor"""
    if features is None:
        features = df[['T_air']].copy().dropna()
    else:
        features = features.dropna()
    
    if len(features) < 50:
        return None
    
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
    """Generate ML flags using One-Class SVM"""
    if features is None:
        features = df[['T_air']].copy().dropna()
    else:
        features = features.dropna()
    
    if len(features) < 50:
        return None
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    svm = OneClassSVM(
        nu=contamination,
        kernel='rbf',
        gamma='auto'
    )
    preds = svm.fit_predict(features_scaled)
    
    ml_flags = pd.Series(False, index=df.index)
    ml_flags.loc[features.index] = (preds == -1)
    
    return ml_flags, None

def get_ml_flags_ensemble(df, contamination=0.05, features=None):
    """Ensemble method combining multiple algorithms"""
    if features is None:
        features = df[['T_air']].copy().dropna()
    else:
        features = features.dropna()
    
    if len(features) < 50:
        return None
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    lof = LocalOutlierFactor(contamination=contamination, n_neighbors=20, novelty=False)
    svm = OneClassSVM(nu=contamination, kernel='rbf', gamma='auto')
    
    iso_preds = iso_forest.fit_predict(features_scaled)
    lof_preds = lof.fit_predict(features_scaled)
    svm_preds = svm.fit_predict(features_scaled)
    
    ensemble_votes = (iso_preds == -1).astype(int) + (lof_preds == -1).astype(int) + (svm_preds == -1).astype(int)
    ensemble_preds = ensemble_votes >= 2
    
    ml_flags = pd.Series(False, index=df.index)
    ml_flags.loc[features.index] = ensemble_preds
    
    return ml_flags, ensemble_votes

def get_ml_flags_hybrid(df, rule_flags, contamination=0.05, features=None):
    """Hybrid approach: Uses rule flags as training labels for supervised learning"""
    if features is None:
        features = df[['T_air']].copy().dropna()
    else:
        features = features.dropna()
    
    aligned_rules = rule_flags.loc[features.index]
    
    if len(features) < 100 or aligned_rules.sum() < 10:
        return None, None
    
    X = features.values
    y = aligned_rules.values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train_scaled, y_train)
    
    X_all_scaled = scaler.transform(features.values)
    pred_proba = rf.predict_proba(X_all_scaled)[:, 1]
    
    threshold = np.percentile(pred_proba, (1 - contamination) * 100)
    hybrid_preds = pred_proba >= threshold
    
    ml_flags = pd.Series(False, index=df.index)
    ml_flags.loc[features.index] = hybrid_preds
    
    return ml_flags, pred_proba

# ============================================================================
# Sidebar - Station Selection
# ============================================================================
st.sidebar.header("🔧 Station Selection")

# State selection
state_code = st.sidebar.text_input("State Code", value="NM").upper()

# Load available stations for the state
with st.spinner(f"Loading {state_code} stations..."):
    state_stations = load_station_metadata(state_code)

if len(state_stations) == 0:
    st.sidebar.error(f"No stations found for state {state_code}")
    st.stop()

# Create station selection with checkboxes
st.sidebar.subheader("Select Primary Station")

station_options = []
for idx, row in state_stations.iterrows():
    station_options.append({
        'USAF': row['USAF'],
        'WBAN': row['WBAN'],
        'Name': row['STATION NAME'],
        'LAT': row['LAT'],
        'LON': row['LON']
    })

station_df = pd.DataFrame(station_options)

# Add filename column for easy reference
station_df['FILENAME'] = station_df.apply(
    lambda row: f"{str(row['USAF']).zfill(6)}{str(row['WBAN']).zfill(5)}.csv", 
    axis=1
)

primary_idx = st.sidebar.selectbox(
    "Choose primary station",
    options=range(len(station_df)),
    format_func=lambda x: f"{station_df.iloc[x]['Name']} ({station_df.iloc[x]['USAF']}-{station_df.iloc[x]['WBAN']})",
    index=0
)

primary_station = station_df.iloc[primary_idx]

# ============================================================================
# Neighbor Station Selection by Distance
# ============================================================================
st.sidebar.subheader("👥 Neighbor Station Configuration")

distance_option = st.sidebar.radio(
    "Neighbor selection method",
    options=["Nearest station (any distance)", "Within specified distance"]
)

if distance_option == "Nearest station (any distance)":
    # Find absolute nearest station
    st.sidebar.info("Finding nearest station regardless of distance...")
    
    other_stations = station_df[station_df.index != primary_idx].copy()
    
    min_dist = float('inf')
    nearest_idx = None
    nearest_info = None
    
    for idx, row in other_stations.iterrows():
        dist = geodesic(
            (primary_station['LAT'], primary_station['LON']),
            (row['LAT'], row['LON'])
        ).km
        
        if dist < min_dist:
            min_dist = dist
            nearest_idx = idx
            nearest_info = {
                'name': row['Name'],
                'filename': row['FILENAME'],
                'lat': row['LAT'],
                'lon': row['LON'],
                'distance': dist
            }
    
    if nearest_info:
        st.sidebar.success(f"Found: {nearest_info['name']} ({nearest_info['distance']:.1f} km)")
        selected_neighbors = [nearest_info]
        selection_method = f"nearest ({nearest_info['distance']:.1f} km)"
    else:
        st.sidebar.error("No neighbor stations found")
        selected_neighbors = []
        selection_method = "none"

else:
    # Select by distance threshold
    distance_options = {
        "1 km": 1,
        "2 km": 2,
        "5 km": 5,
        "10 km": 10,
        "20 km": 20,
        "30 km": 30,
        "50 km": 50,
        "100 km": 100,
        "200 km": 200
    }
    
    selected_distance = st.sidebar.selectbox(
        "Maximum distance",
        options=list(distance_options.keys()),
        index=2  # Default to 5 km
    )
    
    max_distance = distance_options[selected_distance]
    
    st.sidebar.info(f"Finding stations within {max_distance} km...")
    
    # Find all stations within the specified distance
    stations_in_range = find_stations_by_distance(
        primary_station['LAT'],
        primary_station['LON'],
        station_df,
        primary_station['FILENAME'],
        max_distance
    )
    
    if stations_in_range:
        st.sidebar.success(f"Found {len(stations_in_range)} stations within {max_distance} km")
        
        # Display stations found
        for i, station in enumerate(stations_in_range):
            st.sidebar.text(f"{i+1}. {station['name']} ({station['distance']:.1f} km)")
        
        # Let user select which one to use (default to closest)
        if len(stations_in_range) > 1:
            selected_idx = st.sidebar.selectbox(
                "Select neighbor station",
                options=range(len(stations_in_range)),
                format_func=lambda x: f"{stations_in_range[x]['name']} ({stations_in_range[x]['distance']:.1f} km)",
                index=0
            )
            selected_neighbors = [stations_in_range[selected_idx]]
            selection_method = f"selected within {max_distance} km"
        else:
            selected_neighbors = stations_in_range
            selection_method = f"closest within {max_distance} km"
    else:
        st.sidebar.warning(f"No stations found within {max_distance} km")
        st.sidebar.info("Falling back to nearest station...")
        
        # Fallback to nearest station
        other_stations = station_df[station_df.index != primary_idx].copy()
        
        min_dist = float('inf')
        fallback_info = None
        
        for idx, row in other_stations.iterrows():
            dist = geodesic(
                (primary_station['LAT'], primary_station['LON']),
                (row['LAT'], row['LON'])
            ).km
            
            if dist < min_dist:
                min_dist = dist
                fallback_info = {
                    'name': row['Name'],
                    'filename': row['FILENAME'],
                    'lat': row['LAT'],
                    'lon': row['LON'],
                    'distance': dist
                }
        
        if fallback_info:
            st.sidebar.info(f"Using nearest: {fallback_info['name']} ({fallback_info['distance']:.1f} km)")
            selected_neighbors = [fallback_info]
            selection_method = f"nearest fallback ({fallback_info['distance']:.1f} km)"
        else:
            selected_neighbors = []
            selection_method = "none"

# Year range selection
st.sidebar.subheader("📅 Year Range")
year_range = st.sidebar.slider("Select Years", 2000, 2025, (2018, 2023))
years = range(year_range[0], year_range[1] + 1)

# ============================================================================
# ML Method Selection
# ============================================================================
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

# ML Parameters
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

# Feature engineering option
use_engineered_features = st.sidebar.checkbox(
    "Use Engineered Features",
    value=True,
    help="Add time features, lags, and rolling statistics"
)

# ============================================================================
# Parameter Tuning Guide
# ============================================================================
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
    
    **Distance Selection:**
    - **1-5 km**: Urban areas, dense station networks
    - **10-30 km**: Rural areas, moderate station density
    - **50-100 km**: Remote areas, sparse networks
    - **>100 km**: Only for very sparse networks
    """)

# ============================================================================
# Load Data for Selected Stations
# ============================================================================
st.header("📡 Loading Station Data")

progress_bar = st.progress(0)
status_text = st.empty()

# Load primary station data
status_text.text(f"Loading primary station: {primary_station['Name']}...")

usaf = str(primary_station['USAF']).zfill(6)
wban = str(primary_station['WBAN']).zfill(5)
filename = f"{usaf}{wban}.csv"

all_primary_data = []

for j, year in enumerate(years):
    progress = j / (len(years) * 2)  # First half of progress for primary
    progress_bar.progress(progress)
    
    url = f"https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{filename}"
    df_year = load_access_csv(url)
    
    if not df_year.empty:
        all_primary_data.append(df_year)

if all_primary_data:
    df_primary = pd.concat(all_primary_data).sort_index()
    st.success(f"✅ Loaded {len(df_primary):,} records for primary station: {primary_station['Name']}")
else:
    st.error(f"❌ No data found for primary station in selected years")
    st.stop()

# Load neighbor station data
df_neighbor = None
neighbor_name = None
neighbor_dist = None
neighbor_info_text = ""

if selected_neighbors:
    neighbor_info = selected_neighbors[0]
    neighbor_name = neighbor_info['name']
    neighbor_dist = neighbor_info['distance']
    neighbor_filename = neighbor_info['filename']
    
    status_text.text(f"Loading neighbor station: {neighbor_name}...")
    
    all_neighbor_data = []
    
    for j, year in enumerate(years):
        progress = 0.5 + (j / len(years) * 0.5)  # Second half of progress for neighbor
        progress_bar.progress(progress)
        
        url = f"https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{neighbor_filename}"
        df_year = load_access_csv(url)
        
        if not df_year.empty:
            all_neighbor_data.append(df_year)
    
    if all_neighbor_data:
        df_neighbor = pd.concat(all_neighbor_data).sort_index()
        neighbor_info_text = f"({neighbor_dist:.1f} km, {selection_method})"
        st.success(f"✅ Loaded {len(df_neighbor):,} records for neighbor: {neighbor_name} {neighbor_info_text}")
    else:
        st.warning(f"⚠️ No data found for selected neighbor station - spatial checks will be disabled")
else:
    st.warning("⚠️ No neighbor station available - spatial checks will be disabled")

progress_bar.progress(1.0)
status_text.text("Data loading complete!")

st.markdown("---")

# ============================================================================
# Data Diagnostics
# ============================================================================
st.sidebar.subheader("📊 Data Diagnostics")

total_points = len(df_primary)
data_years = df_primary.index.year.nunique()
data_range = df_primary['T_air'].max() - df_primary['T_air'].min()
data_std = df_primary['T_air'].std()

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
    - Method: {selection_method}
    """)

# Recommendations based on data
st.sidebar.subheader("💡 Recommendations")

if data_range > 60:
    st.sidebar.info("🌡️ Large range - consider lower contamination (0.02-0.03)")
elif data_range < 30:
    st.sidebar.info("🌡️ Small range - consider higher contamination (0.05-0.08)")

if data_std > 10:
    st.sidebar.info("📊 High variability - start with lower contamination (0.03)")
elif data_std < 5:
    st.sidebar.info("📊 Low variability - start with higher contamination (0.07)")

if df_neighbor is not None:
    if neighbor_dist <= 10:
        st.sidebar.success(f"✅ Excellent spatial comparison (within {neighbor_dist:.1f} km)")
    elif neighbor_dist <= 30:
        st.sidebar.info(f"🟡 Good spatial comparison (within {neighbor_dist:.1f} km)")
    elif neighbor_dist <= 100:
        st.sidebar.warning(f"⚠️ Distant neighbor ({neighbor_dist:.1f} km) - spatial check may be less reliable")
    else:
        st.sidebar.error(f"🔴 Very distant neighbor ({neighbor_dist:.1f} km) - spatial check may not be meaningful")

# ============================================================================
# Apply Rule-Based QA/QC (with spatial check)
# ============================================================================
st.header("📊 Applying QA/QC Rules")

with st.spinner("Applying rule-based QA/QC with spatial checks..."):
    df_primary['flag_range'] = (df_primary['T_air'] < -40) | (df_primary['T_air'] > 55)
    df_primary['dT'] = df_primary['T_air'].diff()
    df_primary['flag_spike'] = df_primary['dT'].abs() > 8
    df_primary['flag_flat'] = df_primary['T_air'].rolling(12, min_periods=10).std() < 0.1
    df_primary['heatwave'] = df_primary['T_air'] > 40
    
    # Spatial check (if neighbor data available)
    if df_neighbor is not None:
        # Align both dataframes to common time index
        common_index = df_primary.index.intersection(df_neighbor.index)
        df_primary_aligned = df_primary.loc[common_index]
        df_neighbor_aligned = df_neighbor.loc[common_index]
        
        if len(common_index) > 0:
            # Calculate anomalies (deviation from 24h rolling mean)
            df_primary_aligned['anom'] = df_primary_aligned['T_air'] - df_primary_aligned['T_air'].rolling(24, min_periods=18).mean()
            df_neighbor_aligned['anom'] = df_neighbor_aligned['T_air'] - df_neighbor_aligned['T_air'].rolling(24, min_periods=18).mean()
            
            # Spatial flag: difference in anomalies > 6°C AND not a heatwave
            spatial_flags_aligned = (df_primary_aligned['anom'] - df_neighbor_aligned['anom']).abs() > 6
            
            # Reindex to full dataframe
            df_primary['flag_spatial'] = False
            df_primary.loc[common_index, 'flag_spatial'] = spatial_flags_aligned
            
            st.info(f"✅ Spatial check enabled using {neighbor_name} {neighbor_info_text}")
            st.info(f"   • Common timestamps: {len(common_index):,} points")
            st.info(f"   • Spatial flags: {spatial_flags_aligned.sum():,} ({spatial_flags_aligned.sum()/len(common_index)*100:.2f}%)")
        else:
            df_primary['flag_spatial'] = False
            st.warning("⚠️ No overlapping timestamps with neighbor - spatial check disabled")
    else:
        df_primary['flag_spatial'] = False
        st.warning("⚠️ Spatial check disabled - no neighbor data available")
    
    # Final flag (spatial flags are ignored during heatwaves)
    df_primary['final_flag'] = (
        df_primary['flag_range'] |
        df_primary['flag_spike'] |
        df_primary['flag_flat'] |
        (df_primary['flag_spatial'] & ~df_primary['heatwave'])
    )
    
    rule_based_flags = df_primary['final_flag'].copy()

# Display rule flag breakdown
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Range Flags", f"{df_primary['flag_range'].sum():,}")
with col2:
    st.metric("Spike Flags", f"{df_primary['flag_spike'].sum():,}")
with col3:
    st.metric("Flatline Flags", f"{df_primary['flag_flat'].sum():,}")
with col4:
    st.metric("Spatial Flags", f"{df_primary['flag_spatial'].sum():,}")
with col5:
    st.metric("Heatwaves", f"{df_primary['heatwave'].sum():,}")

st.markdown("---")

# ============================================================================
# Prepare ML Features
# ============================================================================
if use_engineered_features:
    df_features = engineer_features(df_primary)
    feature_cols = [col for col in df_features.columns if col not in ['T_air']]
    features = df_features[feature_cols].dropna()
else:
    features = df_primary[['T_air']].copy().dropna()

# ============================================================================
# Generate ML Flags
# ============================================================================
with st.spinner(f"Generating {ml_method} flags..."):
    if ml_method == "Isolation Forest (Default)":
        ml_flags, ml_scores = get_ml_flags_isolation_forest(df_primary, contamination, features)
    elif ml_method == "Local Outlier Factor (LOF)":
        ml_flags, _ = get_ml_flags_lof(df_primary, contamination, n_neighbors, features)
    elif ml_method == "One-Class SVM":
        ml_flags, _ = get_ml_flags_one_class_svm(df_primary, contamination, features)
    elif ml_method == "Ensemble (Voting)":
        ml_flags, _ = get_ml_flags_ensemble(df_primary, contamination, features)
    elif ml_method == "Hybrid (Rule-based + ML)":
        ml_flags, ml_scores = get_ml_flags_hybrid(df_primary, rule_based_flags, contamination, features)

if ml_flags is None:
    st.error("Insufficient data for ML analysis")
    st.stop()

# Parameter impact analysis
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Current Contamination", f"{contamination:.2f}")

with col2:
    flag_rate = ml_flags.sum() / len(ml_flags) * 100
    st.metric("Actual ML Flag Rate", f"{flag_rate:.2f}%",
             delta=f"{flag_rate - contamination*100:.1f}% vs target")

with col3:
    if abs(flag_rate - contamination*100) > 5:
        st.warning("⚠️ Large difference from target")
        if flag_rate > contamination*100:
            st.info("💡 Consider decreasing contamination")
        else:
            st.info("💡 Consider increasing contamination")

# ============================================================================
# Sensitivity Analysis
# ============================================================================
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
            
            if test_flags is not None:
                test_rates.append(test_flags.sum() / len(test_flags) * 100)
            else:
                test_rates.append(0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_contaminations,
        y=test_rates,
        mode='lines+markers',
        name='Actual Flag Rate',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 0.16],
        y=[0, 16],
        mode='lines',
        name='Ideal 1:1 Line',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f"Parameter Sensitivity - {ml_method}",
        xaxis_title="Contamination Parameter",
        yaxis_title="Actual Flag Rate (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# Create Comparison DataFrame
# ============================================================================
comparison_df = pd.DataFrame(index=df_primary.index)
comparison_df['Temperature'] = df_primary['T_air']
comparison_df['Rule Flag'] = rule_based_flags.astype(int)
comparison_df['ML Flag'] = ml_flags.astype(int)
comparison_df['Flag Type'] = 'Neither'
comparison_df.loc[(comparison_df['Rule Flag'] == 1) & (comparison_df['ML Flag'] == 1), 'Flag Type'] = 'Both'
comparison_df.loc[(comparison_df['Rule Flag'] == 1) & (comparison_df['ML Flag'] == 0), 'Flag Type'] = 'Rule Only'
comparison_df.loc[(comparison_df['Rule Flag'] == 0) & (comparison_df['ML Flag'] == 1), 'Flag Type'] = 'ML Only'

# Add temporal features
comparison_df['Hour'] = comparison_df.index.hour
comparison_df['Month'] = comparison_df.index.month
comparison_df['DayOfWeek'] = comparison_df.index.dayofweek
comparison_df['Season'] = comparison_df['Month'].map({12:'Winter',1:'Winter',2:'Winter',
                                                      3:'Spring',4:'Spring',5:'Spring',
                                                      6:'Summer',7:'Summer',8:'Summer',
                                                      9:'Fall',10:'Fall',11:'Fall'})

# Add rolling statistics
for window in [6, 12, 24]:
    comparison_df[f'Rolling Mean {window}h'] = df_primary['T_air'].rolling(window, min_periods=3).mean()
    comparison_df[f'Rolling Std {window}h'] = df_primary['T_air'].rolling(window, min_periods=3).std()

# ============================================================================
# Create tabs for analysis
# ============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview Statistics",
    "🎯 Agreement Analysis",
    "📈 Temporal Patterns",
    "🌡️ Temperature Analysis",
    "
