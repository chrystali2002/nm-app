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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 ML Detection Results",
    "📈 Feature Engineering",
    "🔄 Comparison Matrix",
    "📉 Imputation Demo",
    "📋 Performance Metrics",
    "🔍 Detailed Flag Analysis"
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
# Main Execution
# ============================================================================

with tab1:
    st.header("ML Detection Results")
    
    if ml_method == "Step 1: Isolation Forest (Basic Anomaly Detection)":
        st.subheader("Isolation Forest on Temperature Only")
        
        with st.spinner("Running Isolation Forest..."):
            ml_flags, ml_scores, status = step1_isolation_forest(df_primary, contamination)
            
            if ml_flags is not None:
                # Calculate unique ML detections
                unique_ml = (ml_flags & ~rule_based_flags)
                
                # Display key metrics at the top
                st.subheader("📊 Flag Comparison Summary")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                total_points = len(df_primary)
                rule_count = rule_based_flags.sum()
                ml_count = ml_flags.sum()
                overlap_count = (rule_based_flags & ml_flags).sum()
                unique_count = unique_ml.sum()
                
                with col1:
                    st.metric("Rule-Based Flags", f"{rule_count:,}", 
                             delta=f"{(rule_count/total_points*100):.1f}%")
                with col2:
                    st.metric("ML Flags", f"{ml_count:,}", 
                             delta=f"{(ml_count/total_points*100):.1f}%",
                             delta_color="inverse")
                with col3:
                    st.metric("Overlap", f"{overlap_count:,}")
                with col4:
                    st.metric("Unique ML Detections", f"{unique_count:,}", 
                             delta=f"{(unique_count/ml_count*100):.1f}% of ML",
                             delta_color="inverse")
                with col5:
                    multiplier = ml_count/rule_count if rule_count > 0 else 0
                    st.metric("ML Sensitivity", f"{multiplier:.1f}x", 
                             delta="More sensitive" if multiplier > 1 else "Less sensitive")
                
                st.markdown("---")
                
                # Enhanced visualization with 4 subplots
                fig = make_subplots(
                    rows=4, cols=1,
                    subplot_titles=('Temperature Time Series', 'Rule-Based Flags', 
                                  'ML Flags', 'Unique ML Detections'),
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.4, 0.2, 0.2, 0.2]
                )

                # Temperature
                fig.add_trace(
                    go.Scatter(x=df_primary.index, y=df_primary['T_air'],
                              mode='lines', name='Temperature',
                              line=dict(color='blue', width=1),
                              hovertemplate='Date: %{x}<br>Temperature: %{y:.1f}°C<extra></extra>'),
                    row=1, col=1
                )

                # Rule-based flags
                rule_flags_int = rule_based_flags.astype(int)
                fig.add_trace(
                    go.Scatter(x=df_primary.index, y=rule_flags_int,
                              mode='markers', name='Rule Flags',
                              marker=dict(color='red', size=4, symbol='square'),
                              hovertemplate='Date: %{x}<br>Rule Flag: %{y}<extra></extra>'),
                    row=2, col=1
                )

                # ML flags
                ml_flags_int = ml_flags.astype(int)
                fig.add_trace(
                    go.Scatter(x=df_primary.index, y=ml_flags_int,
                              mode='markers', name='ML Flags',
                              marker=dict(color='orange', size=3, opacity=0.6),
                              hovertemplate='Date: %{x}<br>ML Flag: %{y}<extra></extra>'),
                    row=3, col=1
                )

                # Unique ML detections
                unique_ml_int = unique_ml.astype(int)
                fig.add_trace(
                    go.Scatter(x=df_primary.index, y=unique_ml_int,
                              mode='markers', name='Unique ML',
                              marker=dict(color='purple', size=5, symbol='diamond'),
                              hovertemplate='Date: %{x}<br>Unique ML: %{y}<extra></extra>'),
                    row=4, col=1
                )

                fig.update_layout(
                    height=800, 
                    title_text=f"ML vs Rule-Based Flag Comparison for {selected_station}",
                    showlegend=True,
                    hovermode='x unified'
                )

                fig.update_xaxes(title_text="Date", row=4, col=1)
                fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
                fig.update_yaxes(title_text="Flag (0/1)", row=2, col=1, range=[-0.1, 1.1])
                fig.update_yaxes(title_text="Flag (0/1)", row=3, col=1, range=[-0.1, 1.1])
                fig.update_yaxes(title_text="Flag (0/1)", row=4, col=1, range=[-0.1, 1.1])

                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Sample analysis of unique ML detections
                st.subheader("🔍 Sample Analysis of Unique ML Detections")
                
                if unique_count > 0:
                    # Get sample of unique ML detections
                    sample_size = min(20, unique_count)
                    sample_indices = df_primary.index[unique_ml == 1][:sample_size]
                    
                    sample_df = df_primary.loc[sample_indices, ['T_air']].copy()
                    sample_df['hour'] = sample_df.index.hour
                    sample_df['day'] = sample_df.index.day
                    sample_df['month'] = sample_df.index.month
                    sample_df['dayofweek'] = sample_df.index.dayofweek
                    
                    # Add context: rolling mean for comparison
                    sample_df['rolling_mean_6h'] = df_primary['T_air'].rolling(6, min_periods=3).mean().loc[sample_indices].values
                    sample_df['deviation'] = sample_df['T_air'] - sample_df['rolling_mean_6h']
                    
                    st.dataframe(sample_df.round(2), width='stretch')
                    
                    # Visualize these samples in context
                    fig2 = make_subplots(rows=1, cols=1)
                    
                    # Plot full temperature series
                    fig2.add_trace(
                        go.Scatter(x=df_primary.index, y=df_primary['T_air'],
                                  mode='lines', name='Full Series',
                                  line=dict(color='lightgray', width=0.5),
                                  opacity=0.5)
                    )
                    
                    # Highlight sample points
                    fig2.add_trace(
                        go.Scatter(x=sample_indices, y=sample_df['T_air'],
                                  mode='markers', name='Sample ML Detections',
                                  marker=dict(color='purple', size=8, symbol='diamond'),
                                  hovertemplate='Date: %{x}<br>Temp: %{y:.1f}°C<br>Deviation: %{customdata:.1f}°C<extra></extra>',
                                  customdata=sample_df['deviation'])
                    )
                    
                    fig2.update_layout(
                        title=f"Sample of {sample_size} Unique ML Detections in Context",
                        xaxis_title="Date",
                        yaxis_title="Temperature (°C)",
                        height=400
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Interpretation of samples
                    avg_deviation = sample_df['deviation'].abs().mean()
                    st.info(f"**Insight**: These ML-detected points deviate from their 6-hour rolling average by an average of {avg_deviation:.2f}°C. This suggests ML is finding points that break local patterns.")
                else:
                    st.warning("No unique ML detections found for this station.")
                
                st.markdown("---")
                
                # Temporal pattern analysis
                st.subheader("⏰ Temporal Pattern Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Flag distribution by hour
                    rule_hours = df_primary.index[rule_based_flags].hour
                    ml_hours = df_primary.index[ml_flags].hour
                    
                    fig3, ax = plt.subplots(figsize=(10, 5))
                    ax.hist(rule_hours, bins=24, alpha=0.7, label='Rule Flags', color='red', edgecolor='black')
                    ax.hist(ml_hours, bins=24, alpha=0.5, label='ML Flags', color='orange', edgecolor='black')
                    ax.set_xlabel('Hour of Day')
                    ax.set_ylabel('Flag Count')
                    ax.set_title('Flag Distribution by Hour')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig3)
                    plt.close()
                
                with col2:
                    # Flag distribution by month
                    rule_months = df_primary.index[rule_based_flags].month
                    ml_months = df_primary.index[ml_flags].month
                    
                    fig4, ax = plt.subplots(figsize=(10, 5))
                    ax.hist(rule_months, bins=12, alpha=0.7, label='Rule Flags', color='red', edgecolor='black')
                    ax.hist(ml_months, bins=12, alpha=0.5, label='ML Flags', color='orange', edgecolor='black')
                    ax.set_xlabel('Month')
                    ax.set_ylabel('Flag Count')
                    ax.set_title('Flag Distribution by Month')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig4)
                    plt.close()
                
                st.markdown("---")
                
                # Statistical metrics
                st.subheader("📈 Statistical Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                # Calculate metrics
                rule_rate = rule_count / total_points * 100
                ml_rate = ml_count / total_points * 100
                
                # Assuming rule-based flags as "truth" for comparison
                tp = overlap_count
                fp = unique_count
                fn = rule_count - overlap_count
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                with col1:
                    st.metric("Rule Flag Rate", f"{rule_rate:.2f}%")
                with col2:
                    st.metric("ML Flag Rate", f"{ml_rate:.2f}%")
                with col3:
                    st.metric("Precision (vs Rules)", f"{precision:.2%}")
                with col4:
                    st.metric("Recall (vs Rules)", f"{recall:.2%}")
                
                col5, col6, col7, col8 = st.columns(4)
                with col5:
                    st.metric("F1 Score", f"{f1:.2%}")
                with col6:
                    st.metric("True Positives", f"{tp:,}")
                with col7:
                    st.metric("False Positives", f"{fp:,}")
                with col8:
                    st.metric("False Negatives", f"{fn:,}")
                
                # Interpretation guide
                st.markdown("""
                ### 📖 How to Interpret These Results
                
                | Metric | Value | Interpretation |
                |--------|-------|----------------|
                | **Rule-Based Flags** | {} | Your traditional QA/QC flagged this many points |
                | **ML Flags** | {} | ML detected {} potential anomalies ({}x more!) |
                | **Overlap** | {} | Points flagged by BOTH methods |
                | **Unique ML Detections** | {} | ML found {} anomalies that rules missed |
                
                **What this means:**
                - ML is **{}** than your rule-based system
                - The {} unique detections represent a **new category** of potential quality issues
                - These are points that passed all your checks but ML suspects something is wrong
                
                **Next steps:**
                1. Review the sample of unique ML detections above
                2. Check temporal patterns (hourly/monthly distributions)
                3. Decide if ML sensitivity should be adjusted
                """.format(
                    f"{rule_count:,} ({rule_rate:.1f}%)",
                    f"{ml_count:,} ({ml_rate:.1f}%)",
                    f"{ml_count:,}",
                    f"{ml_rate/rule_rate:.1f}" if rule_rate > 0 else "∞",
                    f"{overlap_count:,}",
                    f"{unique_count:,}",
                    f"{unique_count:,}",
                    f"{ml_rate/rule_rate:.1f}x more sensitive" if ml_rate > rule_rate else f"{rule_rate/ml_rate:.1f}x less sensitive",
                    f"{unique_count:,}"
                ))
                
            else:
                st.error(f"ML analysis failed: {status}")
    
    elif ml_method == "Step 2: Multi-Feature Anomaly Detection":
        st.subheader("Multi-Feature Anomaly Detection")
        
        with st.spinner("Running multi-feature analysis..."):
            iso_flags, lof_flags, ensemble_flags, status = step2_multi_feature(
                df_primary, contamination, n_neighbors
            )
            
            if iso_flags is not None:
                # Calculate unique detections for each method
                unique_iso = (iso_flags & ~rule_based_flags)
                unique_lof = (lof_flags & ~rule_based_flags)
                unique_ensemble = (ensemble_flags & ~rule_based_flags)
                
                # Display comparison
                st.subheader("📊 Multi-Method Comparison")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Isolation Forest", f"{iso_flags.sum():,}")
                with col2:
                    st.metric("LOF", f"{lof_flags.sum():,}")
                with col3:
                    st.metric("Ensemble", f"{ensemble_flags.sum():,}")
                with col4:
                    st.metric("Rule-Based", f"{rule_based_flags.sum():,}")
                
                # Visualization
                fig = make_subplots(
                    rows=4, cols=1,
                    subplot_titles=('Temperature', 'Isolation Forest', 'LOF', 'Ensemble'),
                    shared_xaxes=True,
                    vertical_spacing=0.05
                )
                
                # Temperature
                fig.add_trace(
                    go.Scatter(x=df_primary.index, y=df_primary['T_air'],
                              mode='lines', name='Temperature',
                              line=dict(color='blue', width=1)),
                    row=1, col=1
                )
                
                # Isolation Forest
                fig.add_trace(
                    go.Scatter(x=df_primary.index, y=iso_flags.astype(int),
                              mode='markers', name='Isolation Forest',
                              marker=dict(color='orange', size=3)),
                    row=2, col=1
                )
                
                # LOF
                fig.add_trace(
                    go.Scatter(x=df_primary.index, y=lof_flags.astype(int),
                              mode='markers', name='LOF',
                              marker=dict(color='purple', size=3)),
                    row=3, col=1
                )
                
                # Ensemble
                fig.add_trace(
                    go.Scatter(x=df_primary.index, y=ensemble_flags.astype(int),
                              mode='markers', name='Ensemble',
                              marker=dict(color='red', size=3)),
                    row=4, col=1
                )
                
                fig.update_layout(height=700, title_text="Multi-Method Comparison")
                st.plotly_chart(fig, use_container_width=True)
                
                # Sample analysis
                st.subheader("🔍 Sample Unique Ensemble Detections")
                if unique_ensemble.sum() > 0:
                    sample_size = min(20, unique_ensemble.sum())
                    sample_indices = df_primary.index[unique_ensemble == 1][:sample_size]
                    sample_df = df_primary.loc[sample_indices, ['T_air']].copy()
                    sample_df['hour'] = sample_df.index.hour
                    sample_df['month'] = sample_df.index.month
                    st.dataframe(sample_df)
                
                st.info("""
                **Multi-Feature Approach**: Using multiple features (rolling stats, lags, time features) 
                and multiple algorithms provides more robust anomaly detection. The ensemble combines 
                strengths of different methods.
                """)
    
    elif ml_method == "Step 3: Temporal Pattern Analysis":
        st.subheader("Temporal Pattern Analysis with Attention Weights")
        
        with st.spinner("Analyzing temporal patterns..."):
            temporal_flags, attention_df = step3_temporal_analysis(df_primary)
            
            # Calculate unique temporal detections
            unique_temporal = (temporal_flags & ~rule_based_flags)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Low Attention Points", f"{(attention_df['attention_weight'] < 0.3).sum():,}")
            with col2:
                st.metric("Mean Attention", f"{attention_df['attention_weight'].mean():.3f}")
            with col3:
                st.metric("Unique Temporal Flags", f"{unique_temporal.sum():,}")
            
            # Visualization
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Temperature', 'Attention Weight', 'Temporal Deviation'),
                shared_xaxes=True,
                vertical_spacing=0.05
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
            
            # Sample analysis
            if unique_temporal.sum() > 0:
                st.subheader("🔍 Sample Low Attention Points")
                sample_size = min(20, unique_temporal.sum())
                sample_indices = df_primary.index[unique_temporal == 1][:sample_size]
                sample_df = df_primary.loc[sample_indices, ['T_air']].copy()
                sample_df['attention'] = attention_df.loc[sample_indices, 'attention_weight']
                sample_df['deviation'] = attention_df.loc[sample_indices, 'temporal_deviation']
                st.dataframe(sample_df.round(3))
            
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
                
                # Decision summary
                st.subheader("📊 Decision Summary")
                decision_counts = hybrid_results['final_decision'].value_counts()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    retain_count = decision_counts.get('RETAIN', 0)
                    st.metric("✅ RETAIN", f"{retain_count:,}", 
                             delta=f"{(retain_count/len(hybrid_results)*100):.1f}%")
                with col2:
                    review_count = decision_counts.get('REVIEW (ML Detected)', 0)
                    st.metric("🟡 REVIEW", f"{review_count:,}")
                with col3:
                    remove_count = decision_counts.get('REMOVE', 0)
                    st.metric("🔴 REMOVE", f"{remove_count:,}")
                with col4:
                    heatwave_count = decision_counts.get('RETAIN (Heatwave)', 0)
                    st.metric("🔥 Heatwave Preserved", f"{heatwave_count:,}")
                
                # Visualization
                fig = make_subplots(
                    rows=4, cols=1,
                    subplot_titles=('Temperature', 'Rule Flags', 'ML Flags', 'Hybrid Decision'),
                    shared_xaxes=True,
                    vertical_spacing=0.05
                )
                
                # Temperature
                fig.add_trace(
                    go.Scatter(x=hybrid_results.index, y=df_primary['T_air'],
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
                
                # Hybrid decision (encoded)
                decision_map = {'RETAIN': 0, 'REVIEW (ML Detected)': 1, 'REMOVE': 2, 'RETAIN (Heatwave)': 3}
                colors = ['green', 'yellow', 'red', 'orange']
                
                for decision, code in decision_map.items():
                    mask = hybrid_results['final_decision'] == decision
                    if mask.any():
                        fig.add_trace(
                            go.Scatter(x=hybrid_results.index[mask], 
                                      y=[code] * mask.sum(),
                                      mode='markers', name=decision,
                                      marker=dict(color=colors[code], size=4, symbol='square')),
                            row=4, col=1
                        )
                
                fig.update_layout(height=700, title_text="Hybrid Framework Results")
                fig.update_yaxes(title_text="Decision", row=4, col=1,
                                ticktext=['RETAIN', 'REVIEW', 'REMOVE', 'HEATWAVE'],
                                tickvals=[0, 1, 2, 3])
                
                st.plotly_chart(fig, use_container_width=True)
                
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
        st.subheader("Feature Correlation Matrix")
        corr_matrix = features_display.corr()
        
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
            height=600,
            width=700
        )
        
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
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Normal', 'Flagged'],
            y=['Normal', 'Flagged'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16}
        ))
        
        fig.update_layout(
            title='Confusion Matrix: Rule-Based vs ML',
            xaxis_title='ML Prediction',
            yaxis_title='Rule-Based Truth',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
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
        fig = go.Figure()
        
        for method in imputation_df.columns:
            fig.add_trace(go.Scatter(
                x=imputation_df.index,
                y=imputation_df[method],
                mode='lines+markers',
                name=method,
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="Imputation Method Comparison",
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
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
    
    # Get ML flags for metrics
    iso_flags, _, _, _ = step2_multi_feature(df_primary, contamination, n_neighbors)
    
    if iso_flags is not None:
        st.subheader("ML Performance Metrics")
        
        # Calculate comprehensive metrics
        total_points = len(df_primary)
        rule_count = rule_based_flags.sum()
        ml_count = iso_flags.sum()
        overlap = (rule_based_flags & iso_flags).sum()
        unique_ml = (iso_flags & ~rule_based_flags).sum()
        unique_rule = (rule_based_flags & ~iso_flags).sum()
        
        # Create metrics dataframe
        metrics_df = pd.DataFrame({
            'Metric': [
                'Total Points',
                'Rule-Based Flags',
                'ML Flags',
                'Overlap',
                'Unique ML Detections',
                'Unique Rule Detections',
                'ML Detection Rate',
                'Rule Detection Rate'
            ],
            'Value': [
                f"{total_points:,}",
                f"{rule_count:,} ({(rule_count/total_points*100):.2f}%)",
                f"{ml_count:,} ({(ml_count/total_points*100):.2f}%)",
                f"{overlap:,} ({(overlap/total_points*100):.2f}%)",
                f"{unique_ml:,} ({(unique_ml/total_points*100):.2f}%)",
                f"{unique_rule:,} ({(unique_rule/total_points*100):.2f}%)",
                f"{(ml_count/total_points*100):.2f}%",
                f"{(rule_count/total_points*100):.2f}%"
            ]
        })
        
        st.dataframe(metrics_df, width='stretch', hide_index=True)
        
        # Radar chart for comparison
        categories = ['Range Detection', 'Spike Detection', 'Flatline Detection', 
                     'Spatial Detection', 'Temporal Consistency']
        
        # Calculate scores based on actual data
        rule_scores = [
            flags_df['flag_range'].mean() * 10 if flags_df['flag_range'].any() else 0.5,
            flags_df['flag_spike'].mean() * 10 if flags_df['flag_spike'].any() else 0.5,
            flags_df['flag_flat'].mean() * 10 if flags_df['flag_flat'].any() else 0.5,
            flags_df['flag_spatial'].mean() * 10 if flags_df['flag_spatial'].any() else 0.5,
            0.6  # Baseline for temporal
        ]
        
        ml_scores = [
            flags_df['flag_range'].mean() * 8 if flags_df['flag_range'].any() else 0.5,  # Slightly lower for range
            flags_df['flag_spike'].mean() * 9 if flags_df['flag_spike'].any() else 0.5,
            flags_df['flag_flat'].mean() * 7 if flags_df['flag_flat'].any() else 0.5,
            flags_df['flag_spatial'].mean() * 9 if flags_df['flag_spatial'].any() else 0.5,
            0.9  # ML excels at temporal
        ]
        
        # Normalize to 0-1 scale
        rule_scores = np.clip(rule_scores, 0, 1)
        ml_scores = np.clip(ml_scores, 0, 1)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=rule_scores,
            theta=categories,
            fill='toself',
            name='Rule-Based',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=ml_scores,
            theta=categories,
            fill='toself',
            name='ML-Enhanced',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Capability Comparison: Rule-Based vs ML",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Key Insights**:
        - **ML excels** at temporal consistency and subtle pattern detection
        - **Rule-based** is superior for clear physical limits
        - **Hybrid approach** combines strengths of both
        - Progressive learning path allows gradual integration
        """)

with tab6:
    st.header("🔍 Detailed Flag Analysis")
    
    # Get ML flags
    iso_flags, _, _, _ = step2_multi_feature(df_primary, contamination, n_neighbors)
    
    if iso_flags is not None:
        # Create detailed flag dataframe
        flag_details = pd.DataFrame(index=df_primary.index)
        flag_details['Temperature'] = df_primary['T_air']
        flag_details['Rule Flag'] = rule_based_flags.astype(int)
        flag_details['ML Flag'] = iso_flags.astype(int)
        flag_details['Flag Type'] = 'None'
        
        # Categorize flags
        flag_details.loc[rule_based_flags & iso_flags, 'Flag Type'] = 'Both'
        flag_details.loc[rule_based_flags & ~iso_flags, 'Flag Type'] = 'Rule Only'
        flag_details.loc[~rule_based_flags & iso_flags, 'Flag Type'] = 'ML Only'
        
        # Add context features
        flag_details['Hour'] = flag_details.index.hour
        flag_details['Month'] = flag_details.index.month
        flag_details['DayOfWeek'] = flag_details.index.dayofweek
        
        # Add rolling statistics
        flag_details['Rolling Mean 6h'] = df_primary['T_air'].rolling(6, min_periods=3).mean()
        flag_details['Rolling Std 6h'] = df_primary['T_air'].rolling(6, min_periods=3).std()
        
        # Filter controls
        st.subheader("Filter Flags")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            flag_type_filter = st.multiselect(
                "Flag Type",
                options=['Both', 'Rule Only', 'ML Only', 'None'],
                default=['Both', 'Rule Only', 'ML Only']
            )
        
        with col2:
            hour_range = st.slider("Hour Range", 0, 23, (0, 23))
        
        with col3:
            month_range = st.slider("Month Range", 1, 12, (1, 12))
        
        # Apply filters
        filtered_df = flag_details[
            (flag_details['Flag Type'].isin(flag_type_filter)) &
            (flag_details['Hour'].between(hour_range[0], hour_range[1])) &
            (flag_details['Month'].between(month_range[0], month_range[1]))
        ]
        
        st.subheader(f"Filtered Results ({len(filtered_df):,} records)")
        st.dataframe(filtered_df, width='stretch')
        
        # Download button
        csv = filtered_df.to_csv()
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name=f"flag_analysis_{selected_station}.csv",
            mime="text/csv"
        )
        
        # Summary statistics by flag type
        st.subheader("Summary by Flag Type")
        
        summary = filtered_df.groupby('Flag Type').agg({
            'Temperature': ['count', 'mean', 'std', 'min', 'max'],
            'Hour': 'mean',
            'Month': 'mean'
        }).round(2)
        
        st.dataframe(summary, width='stretch')

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
