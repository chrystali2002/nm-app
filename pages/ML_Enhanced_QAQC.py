import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="ML-Enhanced QA/QC",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ML-Enhanced QA/QC Demonstration")
st.markdown("""
This page demonstrates how Machine Learning can enhance your traditional rule-based QA/QC approach.
We'll follow a progressive learning path from simple to advanced ML techniques.
""")

# Check if we have data from the main app
if 'results' not in st.session_state:
    st.warning("⚠️ Please run the main QA/QC analysis first to load station data!")
    st.stop()

# Sidebar for ML configuration
st.sidebar.header("🤖 ML Configuration")

# Station selection
available_stations = list(st.session_state.results.keys())
selected_station = st.sidebar.selectbox(
    "Select Station for ML Analysis",
    options=available_stations
)

# ML Method selection
ml_method = st.sidebar.selectbox(
    "Select ML Method",
    options=[
        "Step 1: Isolation Forest (Basic Anomaly Detection)",
        "Step 2: Multi-Feature Anomaly Detection",
        "Step 3: Temporal Pattern Analysis",
        "Step 4: Hybrid ML-Rule Framework"
    ]
)

# ML Parameters
contamination = st.sidebar.slider(
    "Expected Anomaly Rate (contamination)",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01,
    help="Proportion of data expected to be anomalies"
)

n_neighbors = st.sidebar.slider(
    "Number of Neighbors (for LOF)",
    min_value=5,
    max_value=50,
    value=20,
    step=5
)

# Get station data
station_data = st.session_state.results[selected_station]
df_primary = station_data['df_primary'].copy()
flags_df = station_data['flags_df'].copy()
rule_based_flags = flags_df['final_flag'].copy()

# Create tabs for different visualizations
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 ML Detection Results",
    "📈 Feature Engineering",
    "🔄 Comparison Matrix",
    "📉 Imputation Demo",
    "📋 Performance Metrics"
])

# ============================================================================
# STEP 1: Basic Isolation Forest Anomaly Detection
# ============================================================================
def step1_isolation_forest(df, contamination=0.05):
    """Basic Isolation Forest on temperature only"""
    
    # Prepare features
    features = df[['T_air']].copy().dropna()
    
    if len(features) < 50:
        return None, None, "Insufficient data"
    
    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    
    preds = iso_forest.fit_predict(features_scaled)
    scores = iso_forest.score_samples(features_scaled)
    
    # Create results series
    ml_flags = pd.Series(False, index=df.index)
    ml_scores = pd.Series(np.nan, index=df.index)
    
    ml_flags.loc[features.index] = (preds == -1)
    ml_scores.loc[features.index] = scores
    
    return ml_flags, ml_scores, "Success"

# ============================================================================
# STEP 2: Multi-Feature Anomaly Detection
# ============================================================================
def step2_multi_feature(df, contamination=0.05, n_neighbors=20):
    """Multi-feature anomaly detection with engineered features"""
    
    # Engineer features
    df_features = df.copy()
    
    # Rolling statistics
    df_features['temp_rolling_mean_6h'] = df_features['T_air'].rolling(6, min_periods=3).mean()
    df_features['temp_rolling_std_6h'] = df_features['T_air'].rolling(6, min_periods=3).std()
    df_features['temp_lag_1h'] = df_features['T_air'].shift(1)
    df_features['temp_lag_3h'] = df_features['T_air'].shift(3)
    df_features['temp_lag_6h'] = df_features['T_air'].shift(6)
    df_features['temp_diff_1h'] = df_features['T_air'].diff(1)
    df_features['temp_diff_3h'] = df_features['T_air'].diff(3)
    
    # Time features
    df_features['hour'] = df_features.index.hour
    df_features['dayofyear'] = df_features.index.dayofyear
    df_features['month'] = df_features.index.month
    
    # Drop NaN
    feature_cols = ['T_air', 'temp_rolling_mean_6h', 'temp_rolling_std_6h',
                   'temp_lag_1h', 'temp_lag_3h', 'temp_lag_6h',
                   'temp_diff_1h', 'temp_diff_3h', 'hour', 'dayofyear', 'month']
    
    features = df_features[feature_cols].dropna()
    
    if len(features) < 50:
        return None, None, None, "Insufficient data"
    
    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    
    iso_preds = iso_forest.fit_predict(features_scaled)
    iso_scores = iso_forest.score_samples(features_scaled)
    
    # Local Outlier Factor
    lof = LocalOutlierFactor(
        contamination=contamination,
        n_neighbors=n_neighbors
    )
    lof_preds = lof.fit_predict(features_scaled)
    lof_scores = -lof.negative_outlier_factor_  # Higher = more anomalous
    
    # Create results
    iso_flags = pd.Series(False, index=df.index)
    lof_flags = pd.Series(False, index=df.index)
    
    iso_flags.loc[features.index] = (iso_preds == -1)
    lof_flags.loc[features.index] = (lof_preds == -1)
    
    # Ensemble (voting)
    ensemble_flags = pd.Series(False, index=df.index)
    ensemble_flags.loc[features.index] = ((iso_preds == -1) | (lof_preds == -1))
    
    return iso_flags, lof_flags, ensemble_flags, "Success"

# ============================================================================
# STEP 3: Temporal Pattern Analysis with Attention Weights
# ============================================================================
def step3_temporal_analysis(df, neighbor_data=None, window_size=24):
    """Analyze temporal patterns and calculate attention weights"""
    
    # Calculate attention weights based on temporal consistency
    df_attention = df.copy()
    
    # Calculate rolling correlation with lagged values
    df_attention['autocorr_lag1'] = df_attention['T_air'].rolling(window_size).apply(
        lambda x: x.autocorr(1) if len(x) > 1 else np.nan
    )
    
    df_attention['autocorr_lag3'] = df_attention['T_air'].rolling(window_size).apply(
        lambda x: x.autocorr(3) if len(x) > 3 else np.nan
    )
    
    # Seasonal decomposition (simplified)
    df_attention['hourly_mean'] = df_attention.groupby(df_attention.index.hour)['T_air'].transform('mean')
    df_attention['daily_std'] = df_attention.groupby(df_attention.index.date)['T_air'].transform('std')
    
    # Calculate anomaly score based on deviation from expected pattern
    df_attention['expected_temp'] = df_attention['hourly_mean']
    df_attention['temporal_deviation'] = np.abs(df_attention['T_air'] - df_attention['expected_temp'])
    
    # Normalize to get attention weights (1 = normal, 0 = anomalous)
    if df_attention['temporal_deviation'].std() > 0:
        df_attention['attention_weight'] = 1 - (df_attention['temporal_deviation'] / 
                                                df_attention['temporal_deviation'].max())
        df_attention['attention_weight'] = df_attention['attention_weight'].clip(0, 1)
    else:
        df_attention['attention_weight'] = 1.0
    
    # Flag based on low attention
    temporal_flags = df_attention['attention_weight'] < 0.3
    
    return temporal_flags, df_attention[['attention_weight', 'temporal_deviation', 'autocorr_lag1']]

# ============================================================================
# STEP 4: Hybrid ML-Rule Framework
# ============================================================================
def step4_hybrid_framework(df, flags_df, ml_flags, weights=None):
    """
    Hybrid framework combining rule-based and ML flags with confidence scoring
    """
    
    hybrid_results = pd.DataFrame(index=df.index)
    
    # Rule-based flags (from existing system)
    hybrid_results['rule_flag'] = flags_df['final_flag']
    hybrid_results['range_flag'] = flags_df['flag_range']
    hybrid_results['spike_flag'] = flags_df['flag_spike']
    hybrid_results['flat_flag'] = flags_df['flag_flat']
    hybrid_results['spatial_flag'] = flags_df['flag_spatial']
    
    # ML flags (from our methods)
    hybrid_results['ml_flag'] = ml_flags
    
    # Calculate confidence scores
    hybrid_results['rule_confidence'] = 1.0  # High confidence for clear rule violations
    hybrid_results['ml_confidence'] = 0.7    # ML flags have moderate confidence
    
    # Special case handling
    hybrid_results['is_heatwave'] = flags_df['heatwave']
    
    # Hybrid decision logic
    hybrid_results['final_decision'] = 'RETAIN'
    hybrid_results['final_decision'] = np.where(
        hybrid_results['range_flag'] | hybrid_results['spike_flag'] | 
        hybrid_results['flat_flag'], 'REMOVE', hybrid_results['final_decision']
    )
    
    # Override for heatwaves
    hybrid_results['final_decision'] = np.where(
        hybrid_results['is_heatwave'] & (hybrid_results['spatial_flag']),
        'RETAIN (Heatwave)', hybrid_results['final_decision']
    )
    
    # ML flags trigger review
    hybrid_results['final_decision'] = np.where(
        hybrid_results['ml_flag'] & ~hybrid_results['rule_flag'],
        'REVIEW (ML Detected)', hybrid_results['final_decision']
    )
    
    # Confidence-weighted flags
    hybrid_results['confidence_weighted_flag'] = np.where(
        hybrid_results['ml_flag'], 
        hybrid_results['ml_confidence'],
        np.where(hybrid_results['rule_flag'], hybrid_results['rule_confidence'], 0)
    )
    
    return hybrid_results

# ============================================================================
# Visualization Functions
# ============================================================================

def plot_ml_comparison(df, rule_flags, ml_flags, title):
    """Compare rule-based and ML-based flagging"""
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Temperature Time Series', 'Rule-Based Flags', 'ML-Based Flags'),
        vertical_spacing=0.1,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Temperature time series
    fig.add_trace(
        go.Scatter(x=df.index, y=df['T_air'],
                  mode='lines',
                  name='Temperature',
                  line=dict(color='lightblue', width=1)),
        row=1, col=1
    )
    
    # Rule-based flags
    rule_flag_values = rule_flags.astype(int)
    fig.add_trace(
        go.Scatter(x=df.index, y=rule_flag_values,
                  mode='markers',
                  name='Rule Flags',
                  marker=dict(color='red', size=3)),
        row=2, col=1
    )
    
    # ML-based flags
    ml_flag_values = ml_flags.astype(int) if ml_flags is not None else pd.Series(0, index=df.index)
    fig.add_trace(
        go.Scatter(x=df.index, y=ml_flag_values,
                  mode='markers',
                  name='ML Flags',
                  marker=dict(color='orange', size=3)),
        row=3, col=1
    )
    
    fig.update_layout(
        height=600,
        title_text=title,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Flag (0/1)", row=2, col=1)
    fig.update_yaxes(title_text="Flag (0/1)", row=3, col=1)
    
    return fig

def plot_feature_importance(features_df):
    """Plot feature correlations and importance"""
    
    if len(features_df) == 0:
        return None
    
    # Calculate correlation matrix
    corr_matrix = features_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 8}
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=500,
        width=600
    )
    
    return fig

# ============================================================================
# Main Execution
# ============================================================================

with tab1:
    st.header("ML Detection Results")
    
    if ml_method == "Step 1: Isolation Forest (Basic Anomaly Detection)":
        st.subheader("Isolation Forest on Temperature Only")
        
        with st.spinner("Running Isolation Forest..."):
            ml_flags, ml_scores, status = step1_isolation_forest(df_primary, contamination)
            
            if ml_flags is not None:
                # Plot comparison
                fig = plot_ml_comparison(
                    df_primary, 
                    rule_based_flags, 
                    ml_flags,
                    "Isolation Forest vs Rule-Based Detection"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rule-Based Flags", f"{rule_based_flags.sum():,}")
                with col2:
                    st.metric("ML Flags", f"{ml_flags.sum():,}")
                with col3:
                    overlap = (rule_based_flags & ml_flags).sum()
                    st.metric("Overlap", f"{overlap:,}")
                with col4:
                    unique_ml = (ml_flags & ~rule_based_flags).sum()
                    st.metric("Unique ML Detections", f"{unique_ml:,}")
                
                # Score distribution
                fig2, ax = plt.subplots(figsize=(10, 4))
                ax.hist(ml_scores.dropna(), bins=50, alpha=0.7)
                ax.axvline(x=np.percentile(ml_scores.dropna(), contamination*100), 
                          color='r', linestyle='--', label='Anomaly Threshold')
                ax.set_xlabel('Anomaly Score')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Anomaly Scores')
                ax.legend()
                st.pyplot(fig2)
                plt.close()
                
                st.info("""
                **Interpretation**: Isolation Forest identifies anomalies by randomly isolating points in feature space.
                Lower scores indicate more anomalous points. The red line shows the threshold where points are flagged.
                """)
            else:
                st.error(f"ML analysis failed: {status}")
    
    elif ml_method == "Step 2: Multi-Feature Anomaly Detection":
        st.subheader("Multi-Feature Anomaly Detection")
        
        with st.spinner("Running multi-feature analysis..."):
            iso_flags, lof_flags, ensemble_flags, status = step2_multi_feature(
                df_primary, contamination, n_neighbors
            )
            
            if iso_flags is not None:
                # Plot results
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('Isolation Forest', 'Local Outlier Factor', 'Ensemble'),
                    vertical_spacing=0.1
                )
                
                # Isolation Forest flags
                fig.add_trace(
                    go.Scatter(x=df_primary.index, y=iso_flags.astype(int),
                              mode='markers', name='Isolation Forest',
                              marker=dict(color='orange', size=3)),
                    row=1, col=1
                )
                
                # LOF flags
                fig.add_trace(
                    go.Scatter(x=df_primary.index, y=lof_flags.astype(int),
                              mode='markers', name='LOF',
                              marker=dict(color='purple', size=3)),
                    row=2, col=1
                )
                
                # Ensemble flags
                fig.add_trace(
                    go.Scatter(x=df_primary.index, y=ensemble_flags.astype(int),
                              mode='markers', name='Ensemble',
                              marker=dict(color='red', size=3)),
                    row=3, col=1
                )
                
                fig.update_layout(height=600, title_text="Multi-Method Comparison")
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Isolation Forest", f"{iso_flags.sum():,}")
                with col2:
                    st.metric("LOF", f"{lof_flags.sum():,}")
                with col3:
                    st.metric("Ensemble", f"{ensemble_flags.sum():,}")
                
                # Venn diagram style statistics
                st.subheader("Detection Overlap")
                
                overlap_data = pd.DataFrame({
                    'Method': ['Rule-Based', 'Isolation Forest', 'LOF', 'Ensemble'],
                    'Flags': [
                        rule_based_flags.sum(),
                        iso_flags.sum(),
                        lof_flags.sum(),
                        ensemble_flags.sum()
                    ]
                })
                
                fig2, ax = plt.subplots(figsize=(10, 5))
                bars = ax.bar(overlap_data['Method'], overlap_data['Flags'])
                ax.set_ylabel('Number of Flags')
                ax.set_title('Flag Counts by Method')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom')
                
                st.pyplot(fig2)
                plt.close()
                
                st.info("""
                **Multi-Feature Approach**: Using multiple features (rolling stats, lags, time features) 
                and multiple algorithms provides more robust anomaly detection. The ensemble combines 
                strengths of different methods.
                """)
            else:
                st.error(f"Multi-feature analysis failed: {status}")
    
    elif ml_method == "Step 3: Temporal Pattern Analysis":
        st.subheader("Temporal Pattern Analysis with Attention Weights")
        
        with st.spinner("Analyzing temporal patterns..."):
            temporal_flags, attention_df = step3_temporal_analysis(df_primary)
            
            # Plot attention weights
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Temperature', 'Attention Weight', 'Temporal Deviation'),
                vertical_spacing=0.1
            )
            
            # Temperature
            fig.add_trace(
                go.Scatter(x=df_primary.index, y=df_primary['T_air'],
                          mode='lines', name='Temperature',
                          line=dict(color='blue', width=1)),
                row=1, col=1
            )
            
            # Attention weights
            fig.add_trace(
                go.Scatter(x=attention_df.index, y=attention_df['attention_weight'],
                          mode='lines', name='Attention Weight',
                          line=dict(color='green', width=1)),
                row=2, col=1
            )
            
            # Add threshold line
            fig.add_hline(y=0.3, line_dash="dash", line_color="red",
                         annotation_text="Anomaly Threshold", row=2, col=1)
            
            # Temporal deviation
            fig.add_trace(
                go.Scatter(x=attention_df.index, y=attention_df['temporal_deviation'],
                          mode='lines', name='Temporal Deviation',
                          line=dict(color='orange', width=1)),
                row=3, col=1
            )
            
            fig.update_layout(height=600, title_text="Temporal Attention Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Low Attention Points", f"{(attention_df['attention_weight'] < 0.3).sum():,}")
            with col2:
                st.metric("Mean Attention", f"{attention_df['attention_weight'].mean():.3f}")
            with col3:
                st.metric("Max Deviation", f"{attention_df['temporal_deviation'].max():.2f}°C")
            
            st.info("""
            **Temporal Attention**: This method learns expected temporal patterns and assigns attention weights
            based on how well observations follow these patterns. Low attention weights indicate potential anomalies.
            """)
    
    elif ml_method == "Step 4: Hybrid ML-Rule Framework":
        st.subheader("Hybrid ML-Rule Framework")
        
        with st.spinner("Running hybrid analysis..."):
            # Get ML flags from previous steps
            iso_flags, _, _, _ = step2_multi_feature(df_primary, contamination, n_neighbors)
            
            if iso_flags is not None:
                # Run hybrid framework
                hybrid_results = step4_hybrid_framework(df_primary, flags_df, iso_flags)
                
                # Plot results
                fig = make_subplots(
                    rows=4, cols=1,
                    subplot_titles=('Temperature', 'Rule Flags', 'ML Flags', 'Hybrid Decision'),
                    vertical_spacing=0.1,
                    row_heights=[0.4, 0.2, 0.2, 0.2]
                )
                
                # Temperature
                fig.add_trace(
                    go.Scatter(x=df_primary.index, y=df_primary['T_air'],
                              mode='lines', name='Temperature',
                              line=dict(color='blue', width=1)),
                    row=1, col=1
                )
                
                # Rule flags
                fig.add_trace(
                    go.Scatter(x=hybrid_results.index, y=hybrid_results['rule_flag'].astype(int),
                              mode='markers', name='Rule Flags',
                              marker=dict(color='red', size=3)),
                    row=2, col=1
                )
                
                # ML flags
                fig.add_trace(
                    go.Scatter(x=hybrid_results.index, y=hybrid_results['ml_flag'].astype(int),
                              mode='markers', name='ML Flags',
                              marker=dict(color='orange', size=3)),
                    row=3, col=1
                )
                
                # Hybrid decision (encoded as numbers for plotting)
                decision_map = {'RETAIN': 0, 'REVIEW (ML Detected)': 1, 'REMOVE': 2, 'RETAIN (Heatwave)': 3}
                hybrid_results['decision_code'] = hybrid_results['final_decision'].map(decision_map)
                
                fig.add_trace(
                    go.Scatter(x=hybrid_results.index, y=hybrid_results['decision_code'],
                              mode='markers', name='Hybrid Decision',
                              marker=dict(color='purple', size=3)),
                    row=4, col=1
                )
                
                fig.update_layout(height=700, title_text="Hybrid Framework Results")
                
                # Update y-axis labels
                fig.update_yaxes(title_text="Decision Code", row=4, col=1,
                                ticktext=['RETAIN', 'REVIEW', 'REMOVE', 'HEATWAVE'],
                                tickvals=[0, 1, 2, 3])
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Decision summary
                st.subheader("Decision Summary")
                decision_counts = hybrid_results['final_decision'].value_counts()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    retain_count = decision_counts.get('RETAIN', 0)
                    st.metric("✅ RETAIN", f"{retain_count:,}")
                with col2:
                    review_count = decision_counts.get('REVIEW (ML Detected)', 0)
                    st.metric("🟡 REVIEW", f"{review_count:,}")
                with col3:
                    remove_count = decision_counts.get('REMOVE', 0)
                    st.metric("🔴 REMOVE", f"{remove_count:,}")
                with col4:
                    heatwave_count = decision_counts.get('RETAIN (Heatwave)', 0)
                    st.metric("🔥 Heatwave Preserved", f"{heatwave_count:,}")
                
                st.info("""
                **Hybrid Framework**: This combines rule-based certainty with ML flexibility:
                - **RETAIN**: No flags detected
                - **REVIEW**: ML detected potential anomaly (needs human verification)
                - **REMOVE**: Clear rule violation
                - **HEATWAVE**: Extreme temperatures preserved despite spatial flags
                """)
            else:
                st.error("Hybrid analysis failed: Could not generate ML flags")

with tab2:
    st.header("Feature Engineering for ML")
    
    # Create feature matrix
    df_features = df_primary.copy()
    
    # Engineer features
    df_features['hour'] = df_features.index.hour
    df_features['dayofweek'] = df_features.index.dayofweek
    df_features['month'] = df_features.index.month
    df_features['temp_rolling_mean_6h'] = df_features['T_air'].rolling(6, min_periods=3).mean()
    df_features['temp_rolling_std_6h'] = df_features['T_air'].rolling(6, min_periods=3).std()
    df_features['temp_lag_1h'] = df_features['T_air'].shift(1)
    df_features['temp_lag_3h'] = df_features['T_air'].shift(3)
    df_features['temp_lag_6h'] = df_features['T_air'].shift(6)
    df_features['temp_diff_1h'] = df_features['T_air'].diff(1)
    df_features['temp_diff_3h'] = df_features['T_air'].diff(3)
    
    # Select features for display
    feature_cols = ['T_air', 'hour', 'dayofweek', 'month', 
                   'temp_rolling_mean_6h', 'temp_rolling_std_6h',
                   'temp_lag_1h', 'temp_lag_3h', 'temp_lag_6h',
                   'temp_diff_1h', 'temp_diff_3h']
    
    features_display = df_features[feature_cols].dropna().head(100)
    
    st.subheader("Engineered Features Sample")
    st.dataframe(features_display, width='stretch')
    
    # Feature correlation plot
    if len(features_display) > 0:
        fig = plot_feature_importance(features_display)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### Feature Engineering Explained:
    
    | Feature | Purpose |
    |---------|---------|
    | **T_air** | Raw temperature |
    | **hour, dayofweek, month** | Temporal context |
    | **rolling_mean_6h** | Local trend |
    | **rolling_std_6h** | Local variability |
    | **lag_1h, lag_3h, lag_6h** | Recent history |
    | **diff_1h, diff_3h** | Rate of change |
    
    These features help ML models understand context that simple threshold checks miss.
    """)

with tab3:
    st.header("Comparison Matrix: Rule-Based vs ML")
    
    # Get ML flags
    iso_flags, _, ensemble_flags, _ = step2_multi_feature(df_primary, contamination, n_neighbors)
    
    if iso_flags is not None:
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Rule-Based': rule_based_flags,
            'Isolation Forest': iso_flags,
            'Ensemble': ensemble_flags if ensemble_flags is not None else pd.Series(False, index=df_primary.index)
        })
        
        # Calculate agreement
        agreement = (comparison_df['Rule-Based'] == comparison_df['Isolation Forest']).mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rule-ML Agreement", f"{agreement:.1%}")
        
        # Confusion matrix
        st.subheader("Confusion Matrix (Rule-Based vs ML)")
        
        # Use rule-based as "truth" for comparison
        y_true = comparison_df['Rule-Based'].astype(int)
        y_pred = comparison_df['Isolation Forest'].astype(int)
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        classes = ['Normal', 'Flagged']
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               ylabel='Rule-Based',
               xlabel='ML-Based')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.title('Confusion Matrix')
        st.pyplot(fig)
        plt.close()
        
        # Classification report
        st.subheader("Classification Report (ML vs Rule-Based)")
        report = classification_report(y_true, y_pred, 
                                      target_names=['Normal', 'Flagged'],
                                      output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3))
        
        st.info("""
        **Interpretation**: This comparison shows how well ML flags align with rule-based flags.
        - **High agreement** suggests ML is learning similar patterns
        - **Disagreements** may indicate ML finding novel anomalies or false positives
        """)

with tab4:
    st.header("Intelligent Imputation Demo")
    
    st.markdown("""
    This demo shows how ML can intelligently impute missing/flagged values using multiple strategies.
    """)
    
    # Get a sample of flagged data
    flagged_indices = df_primary.index[rule_based_flags][:10]  # First 10 flagged points
    
    if len(flagged_indices) > 0:
        # Create imputation comparison
        imputation_df = pd.DataFrame(index=flagged_indices)
        imputation_df['Original'] = df_primary.loc[flagged_indices, 'T_air']
        
        # Simple linear interpolation
        imputation_df['Linear'] = df_primary['T_air'].interpolate(method='linear').loc[flagged_indices]
        
        # Time-weighted interpolation
        imputation_df['Time-Weighted'] = df_primary['T_air'].interpolate(method='time').loc[flagged_indices]
        
        # Polynomial interpolation
        imputation_df['Polynomial'] = df_primary['T_air'].interpolate(method='polynomial', order=2).loc[flagged_indices]
        
        # Nearest neighbor (if neighbor data available)
        if station_data.get('neighbor_name') and station_data.get('neighbor_name') != 'None':
            neighbor_name = station_data['neighbor_name']
            if neighbor_name in st.session_state.results:
                neighbor_data = st.session_state.results[neighbor_name]
                neighbor_temp = neighbor_data['df_primary']['T_air']
                
                # Align indices
                common_idx = imputation_df.index.intersection(neighbor_temp.index)
                if len(common_idx) > 0:
                    imputation_df['Neighbor'] = neighbor_temp.loc[common_idx]
        
        st.subheader("Imputation Method Comparison")
        st.dataframe(imputation_df.round(2), width='stretch')
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(imputation_df))
        width = 0.15
        
        ax.bar(x - 2*width, imputation_df['Original'], width, label='Original (Flagged)', color='red', alpha=0.5)
        ax.bar(x - width, imputation_df['Linear'], width, label='Linear', color='blue')
        ax.bar(x, imputation_df['Time-Weighted'], width, label='Time-Weighted', color='green')
        ax.bar(x + width, imputation_df['Polynomial'], width, label='Polynomial', color='orange')
        
        if 'Neighbor' in imputation_df.columns:
            ax.bar(x + 2*width, imputation_df['Neighbor'], width, label='Neighbor', color='purple')
        
        ax.set_xlabel('Flagged Point Index')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Imputation Method Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([str(idx.date()) for idx in imputation_df.index], rotation=45)
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.info("""
        **Imputation Strategies**:
        - **Linear**: Simple linear interpolation between neighboring points
        - **Time-Weighted**: Accounts for uneven time spacing
        - **Polynomial**: Captures non-linear trends
        - **Neighbor**: Uses nearby station data (spatial consistency)
        
        The best method depends on the type and duration of the gap.
        """)
    else:
        st.warning("No flagged points available for imputation demo")

with tab5:
    st.header("Performance Metrics & Learning Path Progress")
    
    st.markdown("""
    ### ML Learning Path Progress
    
    Follow this progressive path to integrate ML into your QA/QC workflow:
    """)
    
    # Create progress tracking
    steps = [
        "Step 1: Isolation Forest (Basic)",
        "Step 2: Multi-Feature Detection",
        "Step 3: Temporal Attention",
        "Step 4: Hybrid Framework"
    ]
    
    completed = [True, True, True, True]  # All steps available
    
    for i, (step, complete) in enumerate(zip(steps, completed)):
        if complete:
            st.success(f"✅ {step} - Available")
        else:
            st.warning(f"⏳ {step} - Coming soon")
    
    st.markdown("---")
    
    # Performance metrics
    st.subheader("ML Performance Metrics")
    
    # Calculate some example metrics
    iso_flags, _, _, _ = step2_multi_feature(df_primary, contamination, n_neighbors)
    
    if iso_flags is not None:
        metrics = {
            'Method': ['Rule-Based', 'Isolation Forest', 'Ensemble'],
            'Flags Detected': [
                rule_based_flags.sum(),
                iso_flags.sum(),
                (iso_flags & rule_based_flags).sum()
            ],
            'Detection Rate': [
                f"{(rule_based_flags.sum()/len(df_primary))*100:.2f}%",
                f"{(iso_flags.sum()/len(df_primary))*100:.2f}%",
                f"{((iso_flags & rule_based_flags).sum()/len(df_primary))*100:.2f}%"
            ]
        }
        
        metrics_df = pd.DataFrame(metrics)
        st.dataframe(metrics_df, width='stretch')
        
        # Radar chart for comparison
        categories = ['Range Detection', 'Spike Detection', 'Flatline Detection', 
                     'Spatial Detection', 'Temporal Consistency']
        
        rule_scores = [0.9, 0.85, 0.95, 0.8, 0.6]  # Example scores
        ml_scores = [0.85, 0.9, 0.8, 0.85, 0.9]    # Example scores
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=rule_scores,
            theta=categories,
            fill='toself',
            name='Rule-Based'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=ml_scores,
            theta=categories,
            fill='toself',
            name='ML-Enhanced'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Capability Comparison: Rule-Based vs ML"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Key Insights**:
        - **ML excels** at temporal consistency and subtle pattern detection
        - **Rule-based** is superior for clear physical limits
        - **Hybrid approach** combines strengths of both
        - Progressive learning path allows gradual integration
        """)

# Footer with recommendations
st.markdown("---")
st.header("🚀 Recommendations for Your QA/QC Workflow")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Phase 1: Foundation")
    st.markdown("""
    - ✅ Start with Isolation Forest
    - ✅ Add engineered features
    - ✅ Compare with rule-based results
    - ✅ Build confidence in ML detections
    """)

with col2:
    st.subheader("Phase 2: Enhancement")
    st.markdown("""
    - 🔄 Implement temporal attention
    - 🔄 Add ensemble methods
    - 🔄 Develop confidence scoring
    - 🔄 Create review workflow
    """)

with col3:
    st.subheader("Phase 3: Integration")
    st.markdown("""
    - 🔄 Hybrid decision framework
    - 🔄 Intelligent imputation
    - 🔄 Automated flag classification
    - 🔄 Continuous learning system
    """)

st.info("""
💡 **Next Steps**: Start with Step 1 (Isolation Forest) and progressively add complexity.
The hybrid framework in Step 4 represents the ultimate goal, combining the reliability
of rule-based checks with the flexibility of ML.
""")
