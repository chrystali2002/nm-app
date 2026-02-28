import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import os
import re
from scipy import stats
import matplotlib.gridspec as gridspec
import warnings
from io import BytesIO
from datetime import datetime
import random
warnings.filterwarnings('ignore')

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="NM Temperature QA/QC Analysis",
    page_icon="🌡️",
    layout="wide"
)
# Initialize status_text at the top level
status_text = st.empty()
st.title("🌡️ New Mexico Hourly Air Temperature QA/QC Analysis")
st.markdown("""
This application performs enhanced multi-tier QA/QC analysis on hourly air temperature data 
from New Mexico ASOS (Automated Surface Observing System) weather stations. 
It includes:
- **Multi-tier QA/QC** (range, spike, flatline, spatial checks)
- **Comprehensive statistics** for each station
- **Flag pattern analysis** (clustering, seasonality)
- **Raw vs validated data comparison**
- **Decision matrix** for flagged data classification
- **Enhanced visualizations** with statistics overlay
""")

# -------------------------------
# Data Source Description
# -------------------------------
with st.expander("📡 Data Source: ASOS Network", expanded=True):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Automated Surface Observing System (ASOS)
        
        The **ASOS** is a network of automated weather stations operated by the National Weather Service (NWS), 
        Federal Aviation Administration (FAA), and Department of Defense (DOD). It serves as the primary 
        climatological observing network in the United States.
        
        **Key Characteristics:**
        - **Coverage:** Over 900 stations nationwide, including 25+ in New Mexico
        - **Frequency:** Hourly observations, with some stations reporting more frequently
        - **Parameters:** Temperature, dew point, wind speed/direction, precipitation, pressure, visibility
        - **Quality:** Raw data includes automated QC flags; this analysis applies additional multi-tier QC
        
        **Data Access:**
        - Source: NOAA Integrated Surface Database (ISD)
        - Format: Global Hourly Access (CSV)
        - Period: 2011-2024 (selected range)
        - Variables: Air temperature (TMP) at 2m height
        """)
    
    with col2:
        st.markdown("""
        ### Station Map Legend
        - 🟢 **<5% flags**: Good quality
        - 🟡 **5-10% flags**: Moderate issues
        - 🟠 **10-20% flags**: Problematic
        - 🔴 **>20% flags**: Poor quality
        
        **Processing Steps:**
        1. Raw data ingestion
        2. Multi-tier QC checks
        3. Flag classification
        4. Decision matrix application
        5. Final validation
        """)

# -------------------------------
# Decision Matrix
# -------------------------------
with st.expander("📋 Decision Matrix for Flagged Data", expanded=True):
    st.markdown("""
    ### Flag Classification and Action Matrix
    
    This matrix determines how flagged data should be treated based on the dominant flag type 
    and secondary flag patterns. The decisions are based on extensive analysis of ASOS 
    sensor behavior and known failure modes.
    """)
    
    # Create the decision matrix table
    decision_data = {
        "Dominant Flag": ["Flatline", "Flatline", "Spatial", "Spatial", "Spike", "Spike", "No dominant"],
        "Secondary Flag": ["Spatial", "Spatial", "Flatline", "Flatline", "Spatial", "Flatline", "—"],
        "Typical Share": ["≥70% flatline", "40–70% flatline", "≥60% spatial", "40–60% spatial", "≥50% spike", "≥50% spike", "All <40%"],
        "Interpretation": [
            "Sensor stuck / frozen", 
            "Intermittent sensor stagnation", 
            "Systematic bias / siting issue", 
            "Mixed degradation", 
            "Transmission or power instability", 
            "Electrical noise", 
            "Minor QC noise"
        ],
        "Operational Action": ["Remove", "Remove", "Flag & bias-check", "Remove", "Remove", "Remove", "Retain"],
        "Research Use": ["Exclude", "Use only with caution", "Conditional", "Limited diagnostics only", "Exclude", "Exclude", "Retain"]
    }
    
    decision_df = pd.DataFrame(decision_data)
    
    # Style the dataframe for better visualization
    st.dataframe(
        decision_df,
        column_config={
            "Dominant Flag": st.column_config.TextColumn("Dominant Flag", width="medium"),
            "Secondary Flag": st.column_config.TextColumn("Secondary Flag", width="medium"),
            "Typical Share": st.column_config.TextColumn("Typical Share", width="medium"),
            "Interpretation": st.column_config.TextColumn("Interpretation", width="large"),
            "Operational Action": st.column_config.TextColumn("Operational Action", width="medium"),
            "Research Use": st.column_config.TextColumn("Research Use", width="medium")
        },
        hide_index=True,
        use_container_width=True
    )
    
    st.markdown("""
    **How to use this matrix:**
    1. For each station, identify the dominant flag type (highest percentage)
    2. Identify the secondary flag type (second highest percentage)
    3. Match the pattern in the matrix to determine the likely cause
    4. Apply the recommended operational action
    5. Consider the research use recommendation for further analysis
    
    **Special Cases:**
    - **Heatwaves (T > 40°C)**: Spatial flags are ignored during extreme heat events
    - **Multiple flags**: When multiple flags exceed thresholds, use the most restrictive action
    - **Seasonal patterns**: Consider seasonal variations in flag patterns (see monthly analysis)
    """)

# -------------------------------
# Sidebar Configuration
# -------------------------------
st.sidebar.header("⚙️ Configuration")

STATE_CODE = st.sidebar.text_input("State Code", value="NM")
year_range = st.sidebar.slider("Year Range", 2000, 2025, (2011, 2024))
YEARS = range(year_range[0], year_range[1] + 1)

# Station Sampling Options
st.sidebar.subheader("🎯 Station Sampling")
sampling_options = {
    "Test Run (2% of stations)": 0.02,
    "Quick Preview (5% of stations)": 0.05,
    "Sample Analysis (10% of stations)": 0.10,
    "Quarter Analysis (25% of stations)": 0.25,
    "Half Analysis (50% of stations)": 0.50,
    "Full Analysis (100% of stations)": 1.00
}

selected_sampling = st.sidebar.selectbox(
    "Select analysis scope",
    options=list(sampling_options.keys()),
    index=0  # Default to Test Run
)

sampling_fraction = sampling_options[selected_sampling]

if sampling_fraction < 1.0:
    st.sidebar.info(f"⚠️ Running with {int(sampling_fraction*100)}% of stations for testing")
    st.sidebar.warning("Select 'Full Analysis' for complete results")

# Output options
st.sidebar.subheader("📊 Output Options")
SAVE_SUMMARY_CSV = st.sidebar.checkbox("Save station summary CSV", value=True)
SAVE_NEIGHBOR_CSV = st.sidebar.checkbox("Save neighbor CSV", value=True)
SAVE_STATS_CSV = st.sidebar.checkbox("Save statistics CSV", value=True)

# Display options
st.sidebar.subheader("🎨 Display Options")
FIG_DPI = st.sidebar.slider("Figure DPI", 100, 300, 150)

# Decision matrix options
st.sidebar.subheader("⚖️ Decision Matrix Options")
apply_decision_matrix = st.sidebar.checkbox("Apply decision matrix recommendations", value=True)
highlight_problematic = st.sidebar.checkbox("Highlight problematic stations", value=True)

# Advanced options
with st.sidebar.expander("🔧 Advanced Options"):
    random_seed = st.number_input("Random seed for sampling", value=42, min_value=0, max_value=999)
    show_progress_details = st.checkbox("Show detailed progress", value=True)

# -------------------------------
# Helper Functions
# -------------------------------
def sanitize_filename(name):
    """Sanitize filename for saving."""
    return re.sub(r'[^A-Za-z0-9_\-]', '_', name)

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

def calculate_flag_statistics(df, flags):
    """Calculate comprehensive flag statistics."""
    stats_dict = {}
    total_points = len(df)
    stats_dict['total_points'] = total_points
    
    if total_points > 0:
        stats_dict['flag_range_count'] = flags['flag_range'].sum()
        stats_dict['flag_spike_count'] = flags['flag_spike'].sum()
        stats_dict['flag_flat_count'] = flags['flag_flat'].sum()
        stats_dict['flag_spatial_count'] = flags['flag_spatial'].sum()
        stats_dict['heatwave_count'] = flags['heatwave'].sum()
        stats_dict['final_flag_count'] = flags['final_flag'].sum()
        
        stats_dict['flag_range_pct'] = stats_dict['flag_range_count'] / total_points * 100
        stats_dict['flag_spike_pct'] = stats_dict['flag_spike_count'] / total_points * 100
        stats_dict['flag_flat_pct'] = stats_dict['flag_flat_count'] / total_points * 100
        stats_dict['flag_spatial_pct'] = stats_dict['flag_spatial_count'] / total_points * 100
        stats_dict['heatwave_pct'] = stats_dict['heatwave_count'] / total_points * 100
        stats_dict['final_flag_pct'] = stats_dict['final_flag_count'] / total_points * 100
        
        # Determine dominant flag type
        flag_types = {
            'Flatline': stats_dict['flag_flat_pct'],
            'Spatial': stats_dict['flag_spatial_pct'],
            'Spike': stats_dict['flag_spike_pct'],
            'Range': stats_dict['flag_range_pct']
        }
        
        # Sort by percentage to find dominant and secondary
        sorted_flags = sorted(flag_types.items(), key=lambda x: x[1], reverse=True)
        stats_dict['dominant_flag'] = sorted_flags[0][0] if sorted_flags[0][1] > 0 else 'None'
        stats_dict['dominant_flag_pct'] = sorted_flags[0][1]
        stats_dict['secondary_flag'] = sorted_flags[1][0] if len(sorted_flags) > 1 and sorted_flags[1][1] > 0 else 'None'
        stats_dict['secondary_flag_pct'] = sorted_flags[1][1] if len(sorted_flags) > 1 else 0
        
        # Apply decision matrix logic
        stats_dict['decision'] = apply_decision_matrix_logic(
            stats_dict['dominant_flag'],
            stats_dict['dominant_flag_pct'],
            stats_dict['secondary_flag'],
            stats_dict['secondary_flag_pct']
        )
    else:
        for key in ['flag_range', 'flag_spike', 'flag_flat', 'flag_spatial', 
                    'heatwave', 'final_flag']:
            stats_dict[f'{key}_count'] = 0
            stats_dict[f'{key}_pct'] = 0
        stats_dict['dominant_flag'] = 'None'
        stats_dict['secondary_flag'] = 'None'
        stats_dict['decision'] = 'No data'
    
    return stats_dict

def apply_decision_matrix_logic(dominant, dominant_pct, secondary, secondary_pct):
    """Apply the decision matrix to determine action."""
    
    # Case 1: Flatline dominant
    if dominant == 'Flatline' and dominant_pct >= 70:
        return 'REMOVE - Sensor stuck/frozen'
    elif dominant == 'Flatline' and dominant_pct >= 40:
        if secondary == 'Spatial':
            return 'REMOVE - Intermittent sensor stagnation'
        else:
            return 'REMOVE - Flatline issues'
    
    # Case 2: Spatial dominant
    elif dominant == 'Spatial' and dominant_pct >= 60:
        if secondary == 'Flatline':
            return 'FLAG & BIAS-CHECK - Systematic bias'
        else:
            return 'FLAG & VERIFY - Spatial issues'
    elif dominant == 'Spatial' and dominant_pct >= 40:
        if secondary == 'Flatline':
            return 'REMOVE - Mixed degradation'
        else:
            return 'USE WITH CAUTION - Minor spatial issues'
    
    # Case 3: Spike dominant
    elif dominant == 'Spike' and dominant_pct >= 50:
        if secondary == 'Spatial':
            return 'REMOVE - Transmission instability'
        elif secondary == 'Flatline':
            return 'REMOVE - Electrical noise'
        else:
            return 'REMOVE - Spike issues'
    
    # Case 4: No dominant flag
    elif dominant_pct < 40:
        return 'RETAIN - Minor QC noise'
    
    # Default
    return 'REVIEW - Further analysis needed'

def analyze_flag_clustering(flags, window_hours=24):
    """Analyze if flags are clustered in time."""
    flag_series = flags['final_flag'].astype(int)
    
    if len(flag_series) > 0:
        flag_intervals = flag_series.diff().abs().sum()
        flag_clusters = (flag_series.diff() == 1).sum()
        
        if len(flag_series) >= window_hours:
            rolling_flags = flag_series.rolling(window_hours).sum()
            max_consecutive_flags = rolling_flags.max()
        else:
            max_consecutive_flags = flag_series.sum()
        
        if len(flag_series) > 1:
            autocorr = flag_series.autocorr(lag=1)
        else:
            autocorr = np.nan
    else:
        flag_clusters = 0
        max_consecutive_flags = 0
        autocorr = np.nan
        flag_intervals = 0
    
    return {
        'flag_clusters': flag_clusters,
        'max_consecutive_flags': max_consecutive_flags,
        'autocorrelation_lag1': autocorr,
        'flag_transitions': flag_intervals
    }

def compare_raw_vs_validated(df_raw, df_validated):
    """Compare statistics between raw and validated data."""
    stats_comparison = {}
    
    if len(df_raw) > 0:
        raw_data = df_raw['T_air'].dropna()
        stats_comparison['raw_mean'] = raw_data.mean() if len(raw_data) > 0 else np.nan
        stats_comparison['raw_std'] = raw_data.std() if len(raw_data) > 0 else np.nan
        stats_comparison['raw_min'] = raw_data.min() if len(raw_data) > 0 else np.nan
        stats_comparison['raw_max'] = raw_data.max() if len(raw_data) > 0 else np.nan
        stats_comparison['raw_range'] = (raw_data.max() - raw_data.min()) if len(raw_data) > 0 else np.nan
        stats_comparison['raw_q25'] = raw_data.quantile(0.25) if len(raw_data) > 0 else np.nan
        stats_comparison['raw_median'] = raw_data.median() if len(raw_data) > 0 else np.nan
        stats_comparison['raw_q75'] = raw_data.quantile(0.75) if len(raw_data) > 0 else np.nan
    else:
        for stat in ['raw_mean', 'raw_std', 'raw_min', 'raw_max', 'raw_range', 
                    'raw_q25', 'raw_median', 'raw_q75']:
            stats_comparison[stat] = np.nan
    
    if len(df_validated.dropna()) > 0:
        validated_data = df_validated.dropna()
        stats_comparison['validated_mean'] = validated_data.mean() if len(validated_data) > 0 else np.nan
        stats_comparison['validated_std'] = validated_data.std() if len(validated_data) > 0 else np.nan
        stats_comparison['validated_min'] = validated_data.min() if len(validated_data) > 0 else np.nan
        stats_comparison['validated_max'] = validated_data.max() if len(validated_data) > 0 else np.nan
        stats_comparison['validated_range'] = (validated_data.max() - validated_data.min()) if len(validated_data) > 0 else np.nan
        stats_comparison['validated_q25'] = validated_data.quantile(0.25) if len(validated_data) > 0 else np.nan
        stats_comparison['validated_median'] = validated_data.median() if len(validated_data) > 0 else np.nan
        stats_comparison['validated_q75'] = validated_data.quantile(0.75) if len(validated_data) > 0 else np.nan
    else:
        for stat in ['validated_mean', 'validated_std', 'validated_min', 'validated_max', 
                    'validated_range', 'validated_q25', 'validated_median', 'validated_q75']:
            stats_comparison[stat] = np.nan
    
    if not np.isnan(stats_comparison.get('raw_mean', np.nan)) and not np.isnan(stats_comparison.get('validated_mean', np.nan)):
        stats_comparison['mean_diff'] = stats_comparison['validated_mean'] - stats_comparison['raw_mean']
        stats_comparison['std_diff'] = stats_comparison['validated_std'] - stats_comparison['raw_std']
        stats_comparison['range_diff'] = stats_comparison['validated_range'] - stats_comparison['raw_range']
    else:
        stats_comparison['mean_diff'] = np.nan
        stats_comparison['std_diff'] = np.nan
        stats_comparison['range_diff'] = np.nan
    
    if len(df_raw) > 0:
        valid_count = len(df_validated.dropna())
        raw_count = len(df_raw)
        stats_comparison['data_loss_pct'] = ((raw_count - valid_count) / raw_count * 100) if raw_count > 0 else np.nan
    else:
        stats_comparison['data_loss_pct'] = np.nan
    
    return stats_comparison

def analyze_seasonal_patterns(df, flags):
    """Analyze flag patterns by month."""
    monthly_stats = []
    
    if len(df) == 0:
        return pd.DataFrame(monthly_stats)
    
    df_monthly = df.copy()
    df_monthly['month'] = df_monthly.index.month
    df_monthly['final_flag'] = flags['final_flag']
    
    months_present = sorted(df_monthly['month'].unique())
    
    for month in months_present:
        month_data = df_monthly[df_monthly['month'] == month]
        if len(month_data) > 0:
            month_flags = month_data['final_flag'].sum()
            month_total = len(month_data)
            monthly_stats.append({
                'month': month,
                'flag_count': month_flags,
                'flag_pct': month_flags / month_total * 100 if month_total > 0 else 0,
                'mean_temp': month_data['T_air'].mean(),
                'flag_rate_per_100': (month_flags / month_total * 100) if month_total > 0 else 0
            })
    
    return pd.DataFrame(monthly_stats)

def sample_stations(stations_df, fraction, random_seed=42):
    """Randomly sample a fraction of stations."""
    if fraction >= 1.0:
        return stations_df
    
    random.seed(random_seed)
    n_samples = max(1, int(len(stations_df) * fraction))
    sampled_indices = random.sample(range(len(stations_df)), n_samples)
    return stations_df.iloc[sampled_indices].reset_index(drop=True)

# -------------------------------
# Main Processing
# -------------------------------
if st.button("🚀 Run QA/QC Analysis"):
    # Initialize session state for storing results
    if 'results' not in st.session_state:
        st.session_state.results = {}
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Load station metadata
    status_text.text("Loading station metadata...")
    nm_stations = load_station_metadata(STATE_CODE)
    st.write(f"Found {len(nm_stations)} {STATE_CODE} stations in metadata.")
    
    if len(nm_stations) == 0:
        st.error("No stations found for the selected state!")
        st.stop()
    
    # Step 2: Find first year with data
    # Step 2: Find stations with data in the selected years
if 'results' not in st.session_state:
        st.session_state.results = {}
status_text.text("Identifying stations with data in selected years...")
stations_with_data = set()
base_url = None
years_with_data_dict = {}

for i, year in enumerate(YEARS):
    progress = i / len(YEARS) * 0.2
    (progress)
    
    files, url = get_access_files(year)
    if not files:
        continue
    
    if base_url is None:
        base_url = url
    
    for _, row in nm_stations.iterrows():
        usaf = str(row['USAF']).zfill(6)
        wban = str(row['WBAN']).zfill(5)
        fname = f"{usaf}{wban}.csv"
        
        if fname in files:
            station_key = f"{row['STATION NAME']}|{usaf}|{wban}"
            stations_with_data.add(station_key)
            
            if station_key not in years_with_data_dict:
                years_with_data_dict[station_key] = []
            years_with_data_dict[station_key].append(year)

if not stations_with_data:
    st.error("No stations found with data in the selected year range!")
    st.stop()

# Create stations list
nm_stations_files_list = []
for _, row in nm_stations.iterrows():
    usaf = str(row['USAF']).zfill(6)
    wban = str(row['WBAN']).zfill(5)
    station_key = f"{row['STATION NAME']}|{usaf}|{wban}"
    
    if station_key in stations_with_data:
        years_available = years_with_data_dict[station_key]
        year_range_str = f"{min(years_available)}-{max(years_available)}"
        
        nm_stations_files_list.append({
            "STATION NAME": row['STATION NAME'],
            "USAF": usaf,
            "WBAN": wban,
            "LAT": row['LAT'],
            "LON": row['LON'],
            "FILENAME": f"{usaf}{wban}.csv",
            "YEARS_AVAILABLE": years_available,
            "YEAR_RANGE": year_range_str
        })

nm_stations_files = pd.DataFrame(nm_stations_files_list)

st.success(f"Found {len(nm_stations_files)} stations with data in {year_range[0]}-{year_range[1]}")

# Show summary of years coverage
year_coverage = {}
for _, row in nm_stations_files.iterrows():
    for year in row['YEARS_AVAILABLE']:
        year_coverage[year] = year_coverage.get(year, 0) + 1

if year_coverage:
    coverage_df = pd.DataFrame([
        {"Year": year, "Stations": count} 
        for year, count in sorted(year_coverage.items())
    ])
    st.info(f"Yearly coverage: {', '.join([f'{y}: {c}' for y, c in sorted(year_coverage.items())])}")

# Apply station sampling
original_station_count = len(nm_stations_files)
if sampling_fraction < 1.0:
    nm_stations_files = sample_stations(nm_stations_files, sampling_fraction, random_seed)
    st.info(f"📊 Sampled {len(nm_stations_files)} out of {original_station_count} stations ({int(sampling_fraction*100)}%) for this analysis")

# Step 3: Determine nearest neighbors 
status_text.text("Loading station metadata...")
    nm_stations = load_station_metadata(STATE_CODE)
    st.write(f"Found {len(nm_stations)} {STATE_CODE} stations in metadata.")
    
    if len(nm_stations) == 0:
        st.error("No stations found for the selected state!")
        st.stop()
        
status_text.text("Calculating nearest neighbors...")
neighbor_list = []

for idx, primary in nm_stations_files.iterrows():
    primary_coords = (primary['LAT'], primary['LON'])
    temp_df = nm_stations_files.copy()
    temp_df['DIST'] = temp_df.apply(
        lambda row: geodesic(primary_coords, (row['LAT'], row['LON'])).km, axis=1
    )
    neighbors = temp_df[temp_df['FILENAME'] != primary['FILENAME']]
    
    if not neighbors.empty:
        nearest = neighbors.sort_values('DIST').iloc[0]
        neighbor_list.append({
            "PRIMARY NAME": primary['STATION NAME'],
            "PRIMARY FILE": primary['FILENAME'],
            "PRIMARY LAT": primary['LAT'],
            "PRIMARY LON": primary['LON'],
            "NEIGHBOR NAME": nearest['STATION NAME'],
            "NEIGHBOR FILE": nearest['FILENAME'],
            "NEIGHBOR LAT": nearest['LAT'],
            "NEIGHBOR LON": nearest['LON'],
            "DIST_KM": nearest['DIST']
        })

neighbor_df = pd.DataFrame(neighbor_list)

# Step 4: Process each station for ALL selected years
status_text.text("Processing stations with QA/QC...")
all_station_stats = []
all_flag_analyses = []
all_comparisons = []

total_stations = len(neighbor_df)

if show_progress_details:
    progress_details = st.empty()

for idx, row in neighbor_df.iterrows():
    progress = 0.2 + (idx / total_stations * 0.7)
    progress_bar.progress(progress)
    
    primary_file = row['PRIMARY FILE']
    neighbor_file = row['NEIGHBOR FILE']
    station_name = row['PRIMARY NAME']
    
    status_text.text(f"Processing {station_name} ({idx+1}/{total_stations})...")
    
    if show_progress_details:
        progress_details.info(f"📈 Processing station {idx+1}/{total_stations}: {station_name}")
    
    # Load data for ALL selected years
    all_primary_data = []
    all_neighbor_data = []
    
    for year in YEARS:  # Use the exact years the user selected
        year_url = f"https://www.ncei.noaa.gov/data/global-hourly/access/{year}/"
        
        df_primary_year = load_access_csv(year_url + primary_file)
        if not df_primary_year.empty:
            all_primary_data.append(df_primary_year)
        
        if pd.notna(neighbor_file):
            df_neighbor_year = load_access_csv(year_url + neighbor_file)
            if not df_neighbor_year.empty:
                all_neighbor_data.append(df_neighbor_year)
    
    if not all_primary_data:
        continue
    
    # Combine all years
    df_primary = pd.concat(all_primary_data)
    df_primary = df_primary.sort_index()
    
    df_neighbor = None
    if all_neighbor_data:
        df_neighbor = pd.concat(all_neighbor_data)
        df_neighbor = df_neighbor.sort_index()
    
    # Align data
    if df_neighbor is not None and not df_neighbor.empty:
        df_primary, df_neighbor = df_primary.align(df_neighbor, join='inner')
    
    
        
        # Multi-tier QA/QC
        df_primary['flag_range'] = (df_primary['T_air'] < -40) | (df_primary['T_air'] > 55)
        df_primary['dT'] = df_primary['T_air'].diff()
        df_primary['flag_spike'] = df_primary['dT'].abs() > 8
        df_primary['flag_flat'] = df_primary['T_air'].rolling(12, min_periods=10).std() < 0.1
        
        if df_neighbor is not None and not df_neighbor.empty:
            df_primary['anom'] = df_primary['T_air'] - df_primary['T_air'].rolling(24, min_periods=18).mean()
            df_neighbor['anom'] = df_neighbor['T_air'] - df_neighbor['T_air'].rolling(24, min_periods=18).mean()
            df_primary['flag_spatial'] = (df_primary['anom'] - df_neighbor['anom']).abs() > 6
        else:
            df_primary['flag_spatial'] = False
        
        df_primary['heatwave'] = df_primary['T_air'] > 40
        df_primary['final_flag'] = (
            df_primary['flag_range'] |
            df_primary['flag_spike'] |
            df_primary['flag_flat'] |
            (df_primary['flag_spatial'] & ~df_primary['heatwave'])
        )
        
        # Create validated dataset
        df_validated = df_primary['T_air'].where(~df_primary['final_flag'])
        
        # Calculate statistics
        flags_df = df_primary[['flag_range', 'flag_spike', 'flag_flat', 'flag_spatial', 'heatwave', 'final_flag']]
        
        flag_stats = calculate_flag_statistics(df_primary, flags_df)
        flag_stats['station_name'] = station_name
        flag_stats['neighbor_name'] = row['NEIGHBOR NAME']
        flag_stats['neighbor_dist'] = row['DIST_KM']
        all_station_stats.append(flag_stats)
        
        clustering_stats = analyze_flag_clustering(flags_df)
        clustering_stats['station_name'] = station_name
        all_flag_analyses.append(clustering_stats)
        
        comparison_stats = compare_raw_vs_validated(df_primary, df_validated)
        comparison_stats['station_name'] = station_name
        all_comparisons.append(comparison_stats)
        
        # Store results for display
        st.session_state.results[station_name] = {
            'df_primary': df_primary,
            'df_validated': df_validated,
            'flags_df': flags_df,
            'neighbor_name': row['NEIGHBOR NAME'],
            'dist_km': row['DIST_KM'],
            'flag_stats': flag_stats,
            'comparison_stats': comparison_stats,
            'clustering_stats': clustering_stats
        }
    
    progress_bar.progress(1.0)
    if show_progress_details:
        progress_details.empty()
    status_text.text("Analysis complete!")
    
    # Store aggregated results
    # Store aggregated results
    st.session_state.all_station_stats = pd.DataFrame(all_station_stats)
    st.session_state.all_flag_analyses = pd.DataFrame(all_flag_analyses)
    st.session_state.all_comparisons = pd.DataFrame(all_comparisons)
    st.session_state.neighbor_df = neighbor_df
# For backward compatibility, set first_year to the start of the range
    st.session_state.first_year = year_range[0]
    st.session_state.year_end = year_range[1]  # Also store end year
    st.session_state.sampling_fraction = sampling_fraction
    st.session_state.original_station_count = original_station_count

    st.success(f"✅ Processed {len(all_station_stats)} stations successfully for years {year_range[0]}-{year_range[1]}!")

   
    
    if sampling_fraction < 1.0:
        st.info(f"💡 This was a {int(sampling_fraction*100)}% sample analysis. Select 'Full Analysis' from the sidebar to process all {original_station_count} stations.")

# -------------------------------
# Results Display with Tabs
# -------------------------------
if 'all_station_stats' in st.session_state:
    st.header("📊 Analysis Results")
    
    # Display sampling info if applicable
    if st.session_state.sampling_fraction < 1.0:
        st.info(f"📊 Showing results for {len(st.session_state.all_station_stats)} stations ({int(st.session_state.sampling_fraction*100)}% sample of {st.session_state.original_station_count} total stations)")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Station Overview", 
        "Individual Station Analysis",
        "Summary Statistics",
        "Flag Analysis",
        "Decision Matrix Results",
        "Station Map"
    ])
    
    with tab1:
        st.subheader("Station Overview")
        
        # Check if both DataFrames exist
        if st.session_state.all_station_stats is not None and st.session_state.all_comparisons is not None:
            # Ensure we have data
            if not st.session_state.all_station_stats.empty and not st.session_state.all_comparisons.empty:
                # Check if station_name column exists in both DataFrames
                if 'station_name' in st.session_state.all_station_stats.columns and 'station_name' in st.session_state.all_comparisons.columns:
                    # Merge all statistics
                    overview_df = pd.merge(
                        st.session_state.all_station_stats,
                        st.session_state.all_comparisons,
                        on='station_name',
                        how='outer'
                    )
                else:
                    # If station_name is missing, use index or create a default
                    st.warning("Station name column missing in some data. Using index for merging.")
                    overview_df = pd.concat([
                        st.session_state.all_station_stats.reset_index(drop=True),
                        st.session_state.all_comparisons.reset_index(drop=True)
                    ], axis=1)
            else:
                st.error("One or more dataframes are empty")
                st.stop()
        else:
            st.error("Missing data for overview")
            st.stop()
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Stations", len(overview_df))
        with col2:
            avg_flag = overview_df['final_flag_pct'].mean() if 'final_flag_pct' in overview_df.columns else 0
            st.metric("Avg Flag %", f"{avg_flag:.1f}%")
        with col3:
            avg_loss = overview_df['data_loss_pct'].mean() if 'data_loss_pct' in overview_df.columns else 0
            st.metric("Avg Data Loss", f"{avg_loss:.1f}%")
        with col4:
            total_points = overview_df['total_points'].sum() if 'total_points' in overview_df.columns else 0
            st.metric("Total Points", f"{total_points:,}")
        
        # Display station table with decision
        display_cols = []
        available_cols = overview_df.columns.tolist()
        
        # Build display columns based on what's available
        col_mapping = {
            'station_name': 'station_name',
            'neighbor_dist': 'neighbor_dist',
            'total_points': 'total_points',
            'final_flag_pct': 'final_flag_pct',
            'dominant_flag': 'dominant_flag',
            'secondary_flag': 'secondary_flag',
            'decision': 'decision',
            'data_loss_pct': 'data_loss_pct',
            'heatwave_count': 'heatwave_count'
        }
        
        for display_name, col_name in col_mapping.items():
            if col_name in available_cols:
                display_cols.append(col_name)
        
        if display_cols:
            df_display = overview_df[display_cols].copy()
            
            # Color code based on decision if decision column exists
            if 'decision' in df_display.columns:
                def color_decision(val):
                    if pd.isna(val):
                        return ''
                    if 'REMOVE' in str(val):
                        return 'background-color: #ffcccc'
                    elif 'RETAIN' in str(val):
                        return 'background-color: #ccffcc'
                    elif 'CAUTION' in str(val):
                        return 'background-color: #ffffcc'
                    else:
                        return ''
                
                # Sort by flag percentage if available
                if 'final_flag_pct' in df_display.columns:
                    df_display = df_display.sort_values('final_flag_pct', ascending=False)
                
                styled_df = df_display.style.applymap(color_decision, subset=['decision'])
                st.dataframe(styled_df, use_container_width=True)
            else:
                # Sort by flag percentage if available
                if 'final_flag_pct' in df_display.columns:
                    df_display = df_display.sort_values('final_flag_pct', ascending=False)
                st.dataframe(df_display, use_container_width=True)
        else:
            st.warning("No display columns available")
            st.dataframe(overview_df)
        
        # Download button
        csv = overview_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Overview CSV",
            data=csv,
            file_name=f"station_overview_{st.session_state.first_year}.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("Individual Station Analysis")
        
        # Check if results exist
        if st.session_state.results and len(st.session_state.results) > 0:
            # Station selector
            selected_station = st.selectbox(
                "Select Station for Detailed Analysis",
                options=sorted(list(st.session_state.results.keys()))
            )
            
            if selected_station:
                station_data = st.session_state.results[selected_station]
                station_stats = station_data.get('flag_stats', {})
                
                # Display station info with decision
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_points = station_stats.get('total_points', 0)
                    st.metric("Total Points", f"{total_points:,}")
                with col2:
                    flagged_points = station_stats.get('final_flag_count', 0)
                    st.metric("Flagged Points", f"{flagged_points:,}")
                with col3:
                    flag_pct = station_stats.get('final_flag_pct', 0)
                    st.metric("Flag Percentage", f"{flag_pct:.1f}%")
                with col4:
                    decision = station_stats.get('decision', 'Unknown')
                    decision_color = "🔴" if "REMOVE" in decision else "🟢" if "RETAIN" in decision else "🟡"
                    st.metric("Decision", f"{decision_color} {decision}")
                
                dominant_flag = station_stats.get('dominant_flag', 'Unknown')
                dominant_pct = station_stats.get('dominant_flag_pct', 0)
                secondary_flag = station_stats.get('secondary_flag', 'Unknown')
                secondary_pct = station_stats.get('secondary_flag_pct', 0)
                
                st.info(f"**Decision Matrix Result:** {decision} based on dominant flag '{dominant_flag}' ({dominant_pct:.1f}%) and secondary flag '{secondary_flag}' ({secondary_pct:.1f}%)")
                
                # Check if we have data to plot
                if 'df_primary' in station_data and station_data['df_primary'] is not None and not station_data['df_primary'].empty:
                    # Create enhanced plot for selected station
                    safe_name = sanitize_filename(selected_station)
                    
                    fig = plt.figure(figsize=(16, 12))
                    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
                    
                    # Main QA/QC plot
                    ax1 = plt.subplot(gs[0, :])
                    
                    df_primary = station_data['df_primary']
                    df_validated = station_data.get('df_validated', pd.Series(index=df_primary.index))
                    flags_df = station_data.get('flags_df', pd.DataFrame())
                    
                    ax1.plot(df_primary.index, df_primary['T_air'], 
                            color='lightgray', alpha=0.7, label='Raw', linewidth=0.5)
                    
                    if df_validated is not None and not df_validated.empty:
                        ax1.plot(df_primary.index, df_validated, 
                                color='tab:blue', label='Validated', linewidth=1)
                    
                    if not flags_df.empty and 'final_flag' in flags_df.columns:
                        flagged_idx = df_primary.index[flags_df['final_flag']]
                        if len(flagged_idx) > 0:
                            flagged_vals = df_primary.loc[flags_df['final_flag'], 'T_air']
                            ax1.scatter(flagged_idx, flagged_vals, 
                                       color='red', s=8, alpha=0.6, label='Flagged', zorder=5)
                    
                    if 'heatwave' in df_primary.columns:
                        extreme_idx = df_primary.index[df_primary['heatwave']]
                        if len(extreme_idx) > 0:
                            extreme_vals = df_primary.loc[df_primary['heatwave'], 'T_air']
                            ax1.scatter(extreme_idx, extreme_vals, 
                                       color='orange', s=15, alpha=0.4, label='Extreme heat (>40°C)', zorder=4)
                    
                    ax1.axhspan(40, 55, color='orange', alpha=0.1, label='Extreme preserved zone')
                    
                    stats_text = (
                        f"Total points: {station_stats.get('total_points', 0):,}\n"
                        f"Flagged: {station_stats.get('final_flag_count', 0):,} ({station_stats.get('final_flag_pct', 0):.1f}%)\n"
                        f"Range flags: {station_stats.get('flag_range_count', 0):,}\n"
                        f"Spike flags: {station_stats.get('flag_spike_count', 0):,}\n"
                        f"Flatline flags: {station_stats.get('flag_flat_count', 0):,}\n"
                        f"Spatial flags: {station_stats.get('flag_spatial_count', 0):,}\n"
                        f"Heatwaves: {station_stats.get('heatwave_count', 0):,}\n"
                        f"Neighbor: {station_data.get('neighbor_name', 'None')} ({station_data.get('dist_km', 0):.1f} km)\n"
                        f"Decision: {station_stats.get('decision', 'Unknown')}"
                    )
                    
                    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                            fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    ax1.set_ylabel('Air Temperature (°C)')
                    
                    ax1.set_title(f"{st.session_state.first_year}-{st.session_state.year_end} - {selected_station} - Hourly Temperature with QA/QC")
                    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
                    ax1.grid(True, alpha=0.3)
                    
                    # Monthly flag percentage
                    ax2 = plt.subplot(gs[1, 0])
                    if not flags_df.empty:
                        seasonal_stats = analyze_seasonal_patterns(df_primary, flags_df)
                        
                        if not seasonal_stats.empty:
                            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                            months = seasonal_stats['month'].values
                            flag_pcts = seasonal_stats['flag_pct'].values
                            
                            x_positions = np.arange(len(months))
                            ax2.bar(x_positions, flag_pcts, color='skyblue', edgecolor='navy', alpha=0.7)
                            ax2.set_xlabel('Month')
                            ax2.set_ylabel('Flag Percentage (%)')
                            ax2.set_title('Monthly Flag Distribution')
                            ax2.set_xticks(x_positions)
                            ax2.set_xticklabels([month_labels[int(m)-1] for m in months], rotation=45)
                            ax2.grid(True, alpha=0.3)
                            
                            if len(flag_pcts) > 0:
                                max_idx = np.argmax(flag_pcts)
                                max_month = months[max_idx]
                                max_pct = flag_pcts[max_idx]
                                ax2.text(0.02, 0.98, f"Max: {month_labels[int(max_month)-1]} ({max_pct:.1f}%)",
                                        transform=ax2.transAxes, fontsize=9,
                                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Flag type distribution
                    ax3 = plt.subplot(gs[1, 1])
                    flag_types = ['Range', 'Spike', 'Flatline', 'Spatial']
                    flag_counts = [
                        station_stats.get('flag_range_count', 0),
                        station_stats.get('flag_spike_count', 0),
                        station_stats.get('flag_flat_count', 0),
                        station_stats.get('flag_spatial_count', 0)
                    ]
                    
                    if sum(flag_counts) > 0:
                        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
                        non_zero_idx = [i for i, count in enumerate(flag_counts) if count > 0]
                        if len(non_zero_idx) > 0:
                            non_zero_counts = [flag_counts[i] for i in non_zero_idx]
                            non_zero_labels = [flag_types[i] for i in non_zero_idx]
                            non_zero_colors = [colors[i] for i in non_zero_idx]
                            
                            wedges, texts, autotexts = ax3.pie(non_zero_counts, labels=non_zero_labels, 
                                                               colors=non_zero_colors,
                                                               autopct='%1.1f%%', startangle=90)
                            ax3.set_title(f'Flag Distribution\nDominant: {station_stats.get("dominant_flag", "None")}')
                        else:
                            ax3.text(0.5, 0.5, 'No flags\nfound', ha='center', va='center')
                            ax3.set_title('Flag Type Distribution')
                    else:
                        ax3.text(0.5, 0.5, 'No flags\nfound', ha='center', va='center')
                        ax3.set_title('Flag Type Distribution')
                    
                    ax3.axis('equal')
                    
                    # Raw vs Validated comparison
                    ax4 = plt.subplot(gs[2, 0])
                    comparison_stats = station_data.get('comparison_stats', {})
                    
                    if df_validated is not None and not df_validated.empty and len(df_validated.dropna()) > 0:
                        data_to_plot = [df_primary['T_air'].dropna(), df_validated.dropna()]
                        bp = ax4.boxplot(data_to_plot, patch_artist=True, labels=['Raw', 'Validated'])
                        
                        colors = ['lightgray', 'lightblue']
                        for patch, color in zip(bp['boxes'], colors):
                            patch.set_facecolor(color)
                        
                        ax4.set_ylabel('Temperature (°C)')
                        ax4.set_title('Raw vs Validated Distribution')
                        ax4.grid(True, alpha=0.3)
                        
                        mean_diff = comparison_stats.get('mean_diff', np.nan)
                        if not np.isnan(mean_diff):
                            stats_text = (
                                f"Mean diff: {mean_diff:.2f}°C\n"
                                f"Std diff: {comparison_stats.get('std_diff', np.nan):.2f}°C\n"
                                f"Range diff: {comparison_stats.get('range_diff', np.nan):.2f}°C\n"
                                f"Data loss: {comparison_stats.get('data_loss_pct', 0):.1f}%"
                            )
                            ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes,
                                    fontsize=9, verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Flag clustering
                    ax5 = plt.subplot(gs[2, 1])
                    clustering_stats = station_data.get('clustering_stats', {})
                    
                    if len(df_primary) > 0 and not flags_df.empty and 'final_flag' in flags_df.columns:
                        flag_series = flags_df['final_flag'].astype(int)
                        
                        if len(flag_series) >= 24:
                            rolling_flags = flag_series.rolling(window=24).sum()
                            ax5.plot(df_primary.index, rolling_flags, color='red', linewidth=1)
                            ax5.fill_between(df_primary.index, 0, rolling_flags, color='red', alpha=0.3)
                            ax5.set_ylabel('Flags in 24h window')
                        else:
                            ax5.plot(df_primary.index, flag_series, color='red', linewidth=1)
                            ax5.fill_between(df_primary.index, 0, flag_series, color='red', alpha=0.3)
                            ax5.set_ylabel('Flags (binary)')
                        
                        ax5.set_xlabel('Date')
                        ax5.set_title(f'Flag Clustering Analysis')
                        ax5.grid(True, alpha=0.3)
                        
                        cluster_text = (
                            f"Flag clusters: {clustering_stats.get('flag_clusters', 0)}\n"
                            f"Max consecutive: {clustering_stats.get('max_consecutive_flags', 0)}\n"
                            f"Autocorr (lag1): {clustering_stats.get('autocorrelation_lag1', np.nan):.3f}"
                        )
                        ax5.text(0.02, 0.98, cluster_text, transform=ax5.transAxes,
                                fontsize=9, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Download individual station plot
                    buf = BytesIO()
                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    
                    st.download_button(
                        label=f"📥 Download {selected_station} Analysis Plot",
                        data=buf,
                        file_name=f"{safe_name}_analysis_{st.session_state.first_year}.png",
                        mime="image/png"
                    )
                else:
                    st.warning(f"No data available to plot for {selected_station}")
        else:
            st.warning("No station results available. Please run the analysis first.")
    
    with tab3:
        st.subheader("Summary Statistics")
        
        if st.session_state.all_station_stats is not None and not st.session_state.all_station_stats.empty:
            # Flag percentage distribution
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Histogram of flag percentages
            ax1 = axes[0]
            if 'final_flag_pct' in st.session_state.all_station_stats.columns:
                flag_pcts = st.session_state.all_station_stats['final_flag_pct'].dropna()
                
                if len(flag_pcts) > 0:
                    ax1.hist(flag_pcts, bins=20, color='skyblue', edgecolor='navy', alpha=0.7)
                    ax1.axvline(flag_pcts.mean(), color='red', linestyle='--', linewidth=2, 
                               label=f"Mean: {flag_pcts.mean():.1f}%")
                    ax1.axvline(flag_pcts.median(), color='green', linestyle='--', linewidth=2, 
                               label=f"Median: {flag_pcts.median():.1f}%")
                    ax1.set_xlabel('Flag Percentage (%)')
                    ax1.set_ylabel('Number of Stations')
                    ax1.set_title('Distribution of Flag Percentages')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Box plot of flag percentages
                    ax2 = axes[1]
                    bp = ax2.boxplot(flag_pcts, patch_artist=True)
                    bp['boxes'][0].set_facecolor('lightblue')
                    ax2.set_ylabel('Flag Percentage (%)')
                    ax2.set_title('Box Plot of Flag Percentages')
                    ax2.grid(True, alpha=0.3)
                    ax2.set_xticklabels(['All Stations'])
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Summary table
                    summary_stats = pd.DataFrame({
                        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3'],
                        'Flag %': [
                            f"{flag_pcts.mean():.2f}",
                            f"{flag_pcts.median():.2f}",
                            f"{flag_pcts.std():.2f}",
                            f"{flag_pcts.min():.2f}",
                            f"{flag_pcts.max():.2f}",
                            f"{flag_pcts.quantile(0.25):.2f}",
                            f"{flag_pcts.quantile(0.75):.2f}"
                        ]
                    })
                    
                    # Add data loss stats if available
                    if st.session_state.all_comparisons is not None and 'data_loss_pct' in st.session_state.all_comparisons.columns:
                        loss_pcts = st.session_state.all_comparisons['data_loss_pct'].dropna()
                        if len(loss_pcts) > 0:
                            summary_stats['Data Loss %'] = [
                                f"{loss_pcts.mean():.2f}",
                                f"{loss_pcts.median():.2f}",
                                f"{loss_pcts.std():.2f}",
                                f"{loss_pcts.min():.2f}",
                                f"{loss_pcts.max():.2f}",
                                f"{loss_pcts.quantile(0.25):.2f}",
                                f"{loss_pcts.quantile(0.75):.2f}"
                            ]
                        else:
                            summary_stats['Data Loss %'] = ["N/A"] * 7
                    
                    st.table(summary_stats)
                else:
                    st.warning("No flag percentage data available")
            else:
                st.warning("Flag percentage column not found in data")
        else:
            st.warning("No station statistics available")
    
    with tab4:
        st.subheader("Flag Pattern Analysis")
        
        if st.session_state.all_flag_analyses is not None and not st.session_state.all_flag_analyses.empty:
            # Merge flag analysis with station stats
            if 'station_name' in st.session_state.all_flag_analyses.columns and 'station_name' in st.session_state.all_station_stats.columns:
                # Get only the columns we need from station stats
                station_cols = ['station_name']
                for col in ['final_flag_pct', 'dominant_flag', 'decision']:
                    if col in st.session_state.all_station_stats.columns:
                        station_cols.append(col)
                
                flag_analysis = pd.merge(
                    st.session_state.all_flag_analyses,
                    st.session_state.all_station_stats[station_cols],
                    on='station_name',
                    how='left'
                )
                
                # Correlation between flag percentage and clustering
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                ax1 = axes[0]
                if 'final_flag_pct' in flag_analysis.columns and 'max_consecutive_flags' in flag_analysis.columns:
                    valid_data = flag_analysis[['final_flag_pct', 'max_consecutive_flags']].dropna()
                    
                    if len(valid_data) > 0:
                        scatter = ax1.scatter(valid_data['final_flag_pct'], valid_data['max_consecutive_flags'], 
                                   alpha=0.6, c=valid_data['final_flag_pct'], cmap='viridis', 
                                   edgecolor='black', s=50)
                        ax1.set_xlabel('Flag Percentage (%)')
                        ax1.set_ylabel('Max Consecutive Flags (24h window)')
                        ax1.set_title('Flag Percentage vs Clustering')
                        ax1.grid(True, alpha=0.3)
                        plt.colorbar(scatter, ax=ax1, label='Flag %')
                        
                        # Add trend line with error handling
                        if len(valid_data) > 1:
                            try:
                                # Check if there's enough variation in the data
                                if valid_data['final_flag_pct'].std() > 0 and valid_data['max_consecutive_flags'].std() > 0:
                                    z = np.polyfit(valid_data['final_flag_pct'], valid_data['max_consecutive_flags'], 1)
                                    p = np.poly1d(z)
                                    x_trend = np.linspace(valid_data['final_flag_pct'].min(), valid_data['final_flag_pct'].max(), 100)
                                    ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, label='Trend')
                                    ax1.legend()
                                else:
                                    # If no variation, just show a horizontal line at the mean
                                    mean_val = valid_data['max_consecutive_flags'].mean()
                                    ax1.axhline(y=mean_val, color='r', linestyle='--', alpha=0.8, label='Mean value')
                                    ax1.legend()
                            except (np.linalg.LinAlgError, ValueError) as e:
                                # If polyfit fails, just show a horizontal line at the mean
                                mean_val = valid_data['max_consecutive_flags'].mean()
                                ax1.axhline(y=mean_val, color='r', linestyle='--', alpha=0.8, label='Mean value (trend unavailable)')
                                ax1.legend()
                                # Optionally log the error
                                print(f"Trend line calculation failed: {e}")
                
                # Histogram of flag clusters
                ax2 = axes[1]
                if 'flag_clusters' in flag_analysis.columns:
                    clusters = flag_analysis['flag_clusters'].dropna()
                    if len(clusters) > 0:
                        ax2.hist(clusters, bins=min(20, len(clusters)), color='lightgreen', 
                                edgecolor='darkgreen', alpha=0.7)
                        ax2.set_xlabel('Number of Flag Clusters')
                        ax2.set_ylabel('Number of Stations')
                        ax2.set_title('Distribution of Flag Clusters')
                        ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Display flag analysis table
                display_cols = []
                for col in ['station_name', 'flag_clusters', 'max_consecutive_flags', 
                           'autocorrelation_lag1', 'final_flag_pct', 'dominant_flag', 'decision']:
                    if col in flag_analysis.columns:
                        display_cols.append(col)
                
                if display_cols:
                    # Sort by max_consecutive_flags if available
                    if 'max_consecutive_flags' in flag_analysis.columns:
                        st.dataframe(flag_analysis[display_cols].sort_values('max_consecutive_flags', ascending=False))
                    else:
                        st.dataframe(flag_analysis[display_cols])
                else:
                    st.dataframe(flag_analysis)
            else:
                st.warning("Cannot merge flag analysis: missing station_name column")
                st.dataframe(st.session_state.all_flag_analyses)
        else:
            st.warning("No flag analysis data available")
    
    with tab5:
        st.subheader("Decision Matrix Results")
        
        if st.session_state.all_station_stats is not None and not st.session_state.all_station_stats.empty:
            if 'decision' in st.session_state.all_station_stats.columns:
                # Summarize decisions
                decision_summary = st.session_state.all_station_stats['decision'].value_counts().reset_index()
                decision_summary.columns = ['Decision', 'Count']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.bar_chart(decision_summary.set_index('Decision'))
                
                with col2:
                    st.dataframe(decision_summary)
                
                # Show stations by decision category
                st.subheader("Stations by Decision Category")
                
                unique_decisions = ['All'] + sorted([d for d in st.session_state.all_station_stats['decision'].unique() if pd.notna(d)])
                decision_filter = st.selectbox(
                    "Filter by decision",
                    options=unique_decisions
                )
                
                if decision_filter == 'All':
                    filtered_stats = st.session_state.all_station_stats
                else:
                    filtered_stats = st.session_state.all_station_stats[
                        st.session_state.all_station_stats['decision'] == decision_filter
                    ]
                
                display_cols = []
                for col in ['station_name', 'neighbor_name', 'neighbor_dist', 'final_flag_pct', 
                           'dominant_flag', 'dominant_flag_pct', 'secondary_flag', 'decision']:
                    if col in filtered_stats.columns:
                        display_cols.append(col)
                
                if display_cols:
                    st.dataframe(filtered_stats[display_cols])
                else:
                    st.dataframe(filtered_stats)
            else:
                st.warning("Decision column not found in data")
                st.dataframe(st.session_state.all_station_stats)
        else:
            st.warning("No decision matrix data available")
    
    with tab6:
        st.subheader("Station Location Map")
        
        if st.session_state.neighbor_df is not None and not st.session_state.neighbor_df.empty:
            # Create simple map
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Set map bounds for New Mexico
            ax.set_xlim(-109, -103)
            ax.set_ylim(31.5, 37)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(f"NM Stations with QA/QC Results ({st.session_state.first_year})")
            ax.grid(True, alpha=0.3)
            
            # Add background
            ax.fill_between([-109, -103], 31.5, 37, color='lightyellow', alpha=0.3)
            
            # Plot stations
            stats_df = st.session_state.all_station_stats
            
            for _, row in st.session_state.neighbor_df.iterrows():
                station_name = row['PRIMARY NAME']
                
                if stats_df is not None and 'station_name' in stats_df.columns:
                    station_stats = stats_df[stats_df['station_name'] == station_name]
                    
                    if not station_stats.empty:
                        flag_pct = station_stats['final_flag_pct'].iloc[0] if 'final_flag_pct' in station_stats.columns else 0
                        decision = station_stats['decision'].iloc[0] if 'decision' in station_stats.columns else 'Unknown'
                        
                        # Color by decision
                        if 'REMOVE' in str(decision):
                            color = 'red'
                            marker_size = 100
                        elif 'RETAIN' in str(decision):
                            color = 'green'
                            marker_size = 70
                        elif 'CAUTION' in str(decision):
                            color = 'orange'
                            marker_size = 85
                        else:
                            color = 'blue'
                            marker_size = 70
                        
                        # Size by data points (additional scaling)
                        total_points = station_stats['total_points'].iloc[0] if 'total_points' in station_stats.columns else 0
                        size = marker_size * (1 + total_points / 15000)
                        
                        ax.scatter(row['PRIMARY LON'], row['PRIMARY LAT'], 
                                  color=color, s=size, edgecolor='black', linewidth=0.5, 
                                  zorder=5, alpha=0.7)
                        
                        # Add station name for stations with high flag rates
                        if flag_pct > 15 or 'REMOVE' in str(decision):
                            ax.annotate(station_name, 
                                       xy=(row['PRIMARY LON'], row['PRIMARY LAT']),
                                       xytext=(5, 5), textcoords='offset points',
                                       fontsize=8, bbox=dict(boxstyle='round,pad=0.2', 
                                                             facecolor='white', alpha=0.7))
                    else:
                        # Plot station even if no stats (just use default color)
                        ax.scatter(row['PRIMARY LON'], row['PRIMARY LAT'], 
                                  color='gray', s=50, edgecolor='black', linewidth=0.5, 
                                  zorder=5, alpha=0.7)
                else:
                    # Plot station even if no stats (just use default color)
                    ax.scatter(row['PRIMARY LON'], row['PRIMARY LAT'], 
                              color='gray', s=50, edgecolor='black', linewidth=0.5, 
                              zorder=5, alpha=0.7)
            
            # Add lines to neighbors (optional)
            show_neighbors = st.checkbox("Show neighbor connections", value=False, key="map_neighbors")
            if show_neighbors:
                for _, row in st.session_state.neighbor_df.iterrows():
                    if pd.notna(row['NEIGHBOR NAME']):
                        ax.plot([row['PRIMARY LON'], row['NEIGHBOR LON']],
                               [row['PRIMARY LAT'], row['NEIGHBOR LAT']],
                               color='gray', linewidth=0.5, alpha=0.3, zorder=1)
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                          markersize=10, label='RETAIN'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                          markersize=10, label='USE WITH CAUTION'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                          markersize=10, label='REMOVE'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                          markersize=10, label='Unknown')
            ]
            ax.legend(handles=legend_elements, loc='lower left', framealpha=0.8)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Map download button
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            
            st.download_button(
                label="📥 Download Map",
                data=buf,
                file_name=f"station_map_{st.session_state.first_year}.png",
                mime="image/png"
            )
        else:
            st.warning("No neighbor data available for map")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("""
**Data Source:** NOAA Integrated Surface Database (ISD) - ASOS Network  
**Analysis:** Multi-tier QA/QC for hourly air temperature with decision matrix  
**Contact:** For questions or issues, please contact the developer
""")
