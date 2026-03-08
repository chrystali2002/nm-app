import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix, matthews_corrcoef
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
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
This page allows you to select specific stations and compare ML-based vs rule-based anomaly detection.
You can load 1-2 stations at a time for detailed analysis.
""")

# ============================================================================
# Helper Functions (adapted from main app)
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

def get_ml_flags(df, contamination=0.05):
    """Generate ML flags using Isolation Forest"""
    features = df[['T_air']].copy().dropna()
    if len(features) < 50:
        return None
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    preds = iso_forest.fit_predict(features_scaled)
    
    ml_flags = pd.Series(False, index=df.index)
    ml_flags.loc[features.index] = (preds == -1)
    return ml_flags

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
st.sidebar.subheader("Select Stations (max 2)")

# Create a dataframe with station info for display
station_options = []
for idx, row in state_stations.iterrows():
    station_options.append({
        'USAF': row['USAF'],
        'WBAN': row['WBAN'],
        'Name': row['STATION NAME'],
        'Location': f"{row['LAT']:.2f}, {row['LON']:.2f}"
    })

station_df = pd.DataFrame(station_options)

# Multi-select for stations (limit to 2)
selected_indices = st.sidebar.multiselect(
    "Choose 1-2 stations",
    options=range(len(station_df)),
    format_func=lambda x: f"{station_df.iloc[x]['Name']} ({station_df.iloc[x]['USAF']}-{station_df.iloc[x]['WBAN']})",
    max_selections=2
)

if len(selected_indices) == 0:
    st.sidebar.warning("Please select at least one station")
    st.stop()

# Year range selection
st.sidebar.subheader("📅 Year Range")
year_range = st.sidebar.slider("Select Years", 2000, 2025, (2018, 2023))
years = range(year_range[0], year_range[1] + 1)

# ML Parameters
st.sidebar.subheader("🤖 ML Parameters")
contamination = st.sidebar.slider(
    "Expected Anomaly Rate",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01,
    help="Proportion of data expected to be anomalies"
)

# ============================================================================
# Load Data for Selected Stations
# ============================================================================
st.header("📡 Loading Station Data")

# Progress tracking
progress_bar = st.progress(0)
status_text = st.empty()

station_data_dict = {}

for i, idx in enumerate(selected_indices):
    station_info = station_df.iloc[idx]
    status_text.text(f"Loading data for {station_info['Name']}...")
    
    usaf = str(station_info['USAF']).zfill(6)
    wban = str(station_info['WBAN']).zfill(5)
    filename = f"{usaf}{wban}.csv"
    
    all_data = []
    
    for j, year in enumerate(years):
        progress = (i * len(years) + j) / (len(selected_indices) * len(years))
        progress_bar.progress(progress)
        
        url = f"https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{filename}"
        df_year = load_access_csv(url)
        
        if not df_year.empty:
            all_data.append(df_year)
    
    if all_data:
        df_station = pd.concat(all_data).sort_index()
        station_data_dict[station_info['Name']] = {
            'df': df_station,
            'usaf': usaf,
            'wban': wban,
            'lat': station_info['Location'].split(',')[0],
            'lon': station_info['Location'].split(',')[1]
        }
        st.success(f"✅ Loaded {len(df_station):,} records for {station_info['Name']}")
    else:
        st.error(f"❌ No data found for {station_info['Name']} in selected years")

progress_bar.progress(1.0)
status_text.text("Data loading complete!")

st.markdown("---")

# ============================================================================
# Process and Compare Stations
# ============================================================================
if len(station_data_dict) == 1:
    # Single station analysis
    station_name = list(station_data_dict.keys())[0]
    df_primary = station_data_dict[station_name]['df']
    
    st.header(f"📊 Analyzing: {station_name}")
    
    # Apply rule-based QA/QC
    with st.spinner("Applying rule-based QA/QC..."):
        df_primary['flag_range'] = (df_primary['T_air'] < -40) | (df_primary['T_air'] > 55)
        df_primary['dT'] = df_primary['T_air'].diff()
        df_primary['flag_spike'] = df_primary['dT'].abs() > 8
        df_primary['flag_flat'] = df_primary['T_air'].rolling(12, min_periods=10).std() < 0.1
        df_primary['heatwave'] = df_primary['T_air'] > 40
        
        # For spatial check (simplified - using rolling stats as proxy)
        df_primary['flag_spatial'] = False  # Simplified for single station
        
        df_primary['final_flag'] = (
            df_primary['flag_range'] |
            df_primary['flag_spike'] |
            df_primary['flag_flat'] 
        )
        
        rule_based_flags = df_primary['final_flag'].copy()
    
    # Generate ML flags
    with st.spinner("Generating ML flags..."):
        ml_flags = get_ml_flags(df_primary, contamination)
    
    if ml_flags is None:
        st.error("Insufficient data for ML analysis")
        st.stop()
    
    # Create comparison dataframe
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

elif len(station_data_dict) == 2:
    # Two station comparison
    station_names = list(station_data_dict.keys())
    st.header(f"📊 Comparing: {station_names[0]} vs {station_names[1]}")
    
    # Get both dataframes and align them
    df1 = station_data_dict[station_names[0]]['df']
    df2 = station_data_dict[station_names[1]]['df']
    
    # Align on common time index
    common_index = df1.index.intersection(df2.index)
    df1_aligned = df1.loc[common_index]
    df2_aligned = df2.loc[common_index]
    
    # Apply rule-based QA/QC to both
    for df in [df1_aligned, df2_aligned]:
        df['flag_range'] = (df['T_air'] < -40) | (df['T_air'] > 55)
        df['dT'] = df['T_air'].diff()
        df['flag_spike'] = df['dT'].abs() > 8
        df['flag_flat'] = df['T_air'].rolling(12, min_periods=10).std() < 0.1
        df['heatwave'] = df['T_air'] > 40
        
        # Spatial check between stations
        df['anom'] = df['T_air'] - df['T_air'].rolling(24, min_periods=18).mean()
        df2_aligned['anom'] = df2_aligned['T_air'] - df2_aligned['T_air'].rolling(24, min_periods=18).mean()
        
        df['flag_spatial'] = (df['anom'] - df2_aligned['anom']).abs() > 6
        
        df['final_flag'] = (
            df['flag_range'] |
            df['flag_spike'] |
            df['flag_flat'] |
            (df['flag_spatial'] & ~df['heatwave'])
        )
    
    # Generate ML flags
    ml_flags1 = get_ml_flags(df1_aligned, contamination)
    ml_flags2 = get_ml_flags(df2_aligned, contamination)
    
    if ml_flags1 is None or ml_flags2 is None:
        st.error("Insufficient data for ML analysis")
        st.stop()
    
    # Create comparison for both stations
    st.subheader("Station 1 Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Station 1", station_names[0])
        st.metric("Rule Flags", f"{(df1_aligned['final_flag']).sum():,}")
        st.metric("ML Flags", f"{ml_flags1.sum():,}")
    
    with col2:
        st.metric("Station 2", station_names[1])
        st.metric("Rule Flags", f"{(df2_aligned['final_flag']).sum():,}")
        st.metric("ML Flags", f"{ml_flags2.sum():,}")
    
    # Create comparison dataframe for station 1 (for detailed analysis)
    comparison_df = pd.DataFrame(index=df1_aligned.index)
    comparison_df['Temperature'] = df1_aligned['T_air']
    comparison_df['Rule Flag'] = df1_aligned['final_flag'].astype(int)
    comparison_df['ML Flag'] = ml_flags1.astype(int)
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
        comparison_df[f'Rolling Mean {window}h'] = df1_aligned['T_air'].rolling(window, min_periods=3).mean()
        comparison_df[f'Rolling Std {window}h'] = df1_aligned['T_air'].rolling(window, min_periods=3).std()
    
    st.info("📌 Showing detailed analysis for Station 1. Switch stations using the sidebar to analyze Station 2.")

# ============================================================================
# Create tabs for different analysis views (same as before)
# ============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview Statistics",
    "🎯 Agreement Analysis",
    "📈 Temporal Patterns",
    "🌡️ Temperature Analysis",
    "📉 Statistical Comparison",
    "🔍 Case Studies"
])

# ============================================================================
# TAB 1: Overview Statistics
# ============================================================================
with tab1:
    st.header("📊 Overview Statistics")
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_points = len(comparison_df)
    rule_count = (comparison_df['Rule Flag'] == 1).sum()
    ml_count = (comparison_df['ML Flag'] == 1).sum()
    both_count = ((comparison_df['Rule Flag'] == 1) & (comparison_df['ML Flag'] == 1)).sum()
    rule_only = ((comparison_df['Rule Flag'] == 1) & (comparison_df['ML Flag'] == 0)).sum()
    ml_only = ((comparison_df['Rule Flag'] == 0) & (comparison_df['ML Flag'] == 1)).sum()
    
    with col1:
        st.metric("Total Points", f"{total_points:,}")
    with col2:
        st.metric("Rule Flags", f"{rule_count:,}", 
                 delta=f"{(rule_count/total_points*100):.1f}%")
    with col3:
        st.metric("ML Flags", f"{ml_count:,}", 
                 delta=f"{(ml_count/total_points*100):.1f}%",
                 delta_color="inverse")
    with col4:
        st.metric("Both", f"{both_count:,}")
    with col5:
        agreement = (both_count + (total_points - rule_count - ml_count + both_count)) / total_points * 100
        st.metric("Agreement Rate", f"{agreement:.1f}%")
    
    st.markdown("---")
    
    # Flag type distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart of flag types
        flag_counts = comparison_df['Flag Type'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=flag_counts.index,
            values=flag_counts.values,
            hole=0.4,
            marker_colors=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Distribution of Flag Types",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Summary statistics table
        summary_stats = []
        for flag_type in ['ML Only', 'Rule Only', 'Both', 'Neither']:
            subset = comparison_df[comparison_df['Flag Type'] == flag_type]
            if len(subset) > 0:
                summary_stats.append({
                    'Flag Type': flag_type,
                    'Count': len(subset),
                    'Percentage': f"{len(subset)/total_points*100:.2f}%",
                    'Mean Temp': f"{subset['Temperature'].mean():.2f}°C",
                    'Temp Range': f"{subset['Temperature'].min():.1f} to {subset['Temperature'].max():.1f}°C",
                    'Std Dev': f"{subset['Temperature'].std():.2f}°C"
                })
        
        st.dataframe(pd.DataFrame(summary_stats), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Key insights
    st.subheader("🔍 Key Insights")
    
    insights = []
    
    if ml_count > rule_count:
        ratio = ml_count / rule_count
        insights.append(f"✅ **ML is {ratio:.1f}x more sensitive** than rule-based, detecting {ml_count - rule_count} additional points")
    else:
        ratio = rule_count / ml_count
        insights.append(f"⚠️ **Rule-based is {ratio:.1f}x more sensitive** than ML")
    
    overlap_ratio = both_count / min(rule_count, ml_count) * 100 if min(rule_count, ml_count) > 0 else 0
    if overlap_ratio > 80:
        insights.append(f"🎯 **High agreement** ({overlap_ratio:.1f}% overlap) - methods detect similar patterns")
    elif overlap_ratio > 50:
        insights.append(f"🟡 **Moderate agreement** ({overlap_ratio:.1f}% overlap) - methods have some common ground")
    else:
        insights.append(f"🔴 **Low agreement** ({overlap_ratio:.1f}% overlap) - methods detect fundamentally different patterns")
    
    if ml_only > rule_only:
        insights.append(f"🤖 **ML specializes** in finding {ml_only} unique patterns that rules miss")
    else:
        insights.append(f"📏 **Rules specialize** in finding {rule_only} unique patterns that ML misses")
    
    for insight in insights:
        st.info(insight)

# ============================================================================
# TAB 2: Agreement Analysis
# ============================================================================
with tab2:
    st.header("🎯 Agreement Analysis")
    
    # Calculate agreement metrics
    y_true = comparison_df['Rule Flag'].values
    y_pred = comparison_df['ML Flag'].values
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Display metrics
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
    
    st.markdown("---")
    
    # Confusion Matrix Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['ML: Normal', 'ML: Flagged'],
            y=['Rule: Normal', 'Rule: Flagged'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16},
            showscale=False
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            height=400,
            xaxis_title="ML Prediction",
            yaxis_title="Rule-Based Truth"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📖 Interpretation")
        
        interpretations = [
            f"**True Negatives:** {tn:,} points - Both methods agree data is normal",
            f"**True Positives:** {tp:,} points - Both methods agree data is anomalous",
            f"**False Positives:** {fp:,} points - ML flagged but rules didn't",
            f"**False Negatives:** {fn:,} points - Rules flagged but ML didn't"
        ]
        
        for interp in interpretations:
            st.markdown(interp)
        
        st.markdown("---")
        
        if kappa > 0.8:
            kappa_interp = "Almost perfect agreement"
        elif kappa > 0.6:
            kappa_interp = "Substantial agreement"
        elif kappa > 0.4:
            kappa_interp = "Moderate agreement"
        elif kappa > 0.2:
            kappa_interp = "Fair agreement"
        else:
            kappa_interp = "Slight agreement"
        
        st.info(f"**Cohen's Kappa**: {kappa:.3f} - {kappa_interp}")

# ============================================================================
# TAB 3: Temporal Patterns (simplified - same as before but using comparison_df)
# ============================================================================
with tab3:
    st.header("📈 Temporal Pattern Analysis")
    
    # Hourly patterns
    st.subheader("Hour of Day Analysis")
    
    hourly_stats = comparison_df.groupby('Hour').agg({
        'Rule Flag': 'mean',
        'ML Flag': 'mean',
        'Temperature': 'mean'
    }).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=hourly_stats['Hour'], y=hourly_stats['Rule Flag'] * 100,
               name='Rule Flag Rate', marker_color='red', opacity=0.7),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(x=hourly_stats['Hour'], y=hourly_stats['ML Flag'] * 100,
               name='ML Flag Rate', marker_color='orange', opacity=0.7),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=hourly_stats['Hour'], y=hourly_stats['Temperature'],
                  name='Avg Temperature', line=dict(color='blue', width=2),
                  mode='lines+markers'),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Flag Rates by Hour of Day",
        xaxis_title="Hour",
        barmode='group',
        height=400
    )
    
    fig.update_yaxes(title_text="Flag Rate (%)", secondary_y=False)
    fig.update_yaxes(title_text="Average Temperature (°C)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 4: Temperature Analysis
# ============================================================================
with tab4:
    st.header("🌡️ Temperature Analysis")
    
    # Temperature distribution by flag type
    fig = go.Figure()
    
    for flag_type, color in zip(['ML Only', 'Rule Only', 'Both', 'Neither'],
                                 ['orange', 'red', 'purple', 'gray']):
        subset = comparison_df[comparison_df['Flag Type'] == flag_type]['Temperature'].dropna()
        if len(subset) > 0:
            fig.add_trace(go.Violin(
                y=subset,
                name=flag_type,
                box_visible=True,
                meanline_visible=True,
                fillcolor=color,
                opacity=0.7,
                line_color='black'
            ))
    
    fig.update_layout(
        title="Temperature Distribution by Flag Type",
        yaxis_title="Temperature (°C)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 5: Statistical Comparison
# ============================================================================
with tab5:
    st.header("📉 Statistical Comparison")
    
    # Separate data by flag type
    ml_only_temp = comparison_df[comparison_df['Flag Type'] == 'ML Only']['Temperature'].dropna()
    rule_only_temp = comparison_df[comparison_df['Flag Type'] == 'Rule Only']['Temperature'].dropna()
    
    if len(ml_only_temp) > 0 and len(rule_only_temp) > 0:
        t_stat, p_value = stats.ttest_ind(ml_only_temp, rule_only_temp)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("T-Statistic", f"{t_stat:.3f}")
            st.metric("P-Value", f"{p_value:.4f}")
        
        with col2:
            if p_value < 0.05:
                st.success("✅ **Significant difference** (p < 0.05)")
            else:
                st.warning("⚠️ **No significant difference** (p >= 0.05)")

# ============================================================================
# TAB 6: Case Studies
# ============================================================================
with tab6:
    st.header("🔍 Case Studies")
    
    # ML Only Examples
    st.subheader("🤖 ML Only Detections")
    ml_only_examples = comparison_df[comparison_df['Flag Type'] == 'ML Only'].head(10)
    if len(ml_only_examples) > 0:
        st.dataframe(ml_only_examples[['Temperature', 'Hour', 'Month', 'Rolling Mean 6h', 'Rolling Std 6h']])
    
    st.markdown("---")
    
    # Rule Only Examples
    st.subheader("📏 Rule Only Detections")
    rule_only_examples = comparison_df[comparison_df['Flag Type'] == 'Rule Only'].head(10)
    if len(rule_only_examples) > 0:
        st.dataframe(rule_only_examples[['Temperature', 'Hour', 'Month', 'Rolling Mean 6h', 'Rolling Std 6h']])
    
    # Download full comparison data
    st.markdown("---")
    csv = comparison_df.to_csv()
    st.download_button(
        label="📥 Download Complete Comparison Data",
        data=csv,
        file_name=f"ml_rule_comparison.csv",
        mime="text/csv"
    )
