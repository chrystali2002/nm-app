import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="ML Train-Test Validation",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 ML Train-Test Validation for Temperature QA/QC")
st.markdown("""
This page implements proper machine learning validation by training on historical data 
and predicting future observations. You can evaluate how well ML models perform on 
unseen data and compare with your rule-based QA/QC.
""")

# Check if we have data from the main app
if 'results' not in st.session_state:
    st.warning("⚠️ Please run the main QA/QC analysis first to load station data!")
    st.stop()

# Sidebar configuration
st.sidebar.header("🎯 Model Configuration")

# State selection
state_options = ['NM']  # You can expand this based on available data
selected_state = st.sidebar.selectbox(
    "Select State",
    options=state_options,
    index=0
)

# Station selection
available_stations = list(st.session_state.results.keys())
selected_station = st.sidebar.selectbox(
    "Select Station for Validation",
    options=available_stations,
    index=0
)

# Train-test split configuration
st.sidebar.subheader("📊 Data Split")
split_method = st.sidebar.radio(
    "Split Method",
    options=["Temporal Split (Recommended)", "Random Split"],
    help="Temporal split preserves time order for realistic forecasting"
)

if split_method == "Temporal Split (Recommended)":
    train_years = st.sidebar.slider(
        "Number of Years for Training",
        min_value=2,
        max_value=8,
        value=7,
        help="First N years used for training, remaining for testing"
    )
else:
    test_size = st.sidebar.slider(
        "Test Size (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5
    ) / 100

# Model selection
st.sidebar.subheader("🤖 Model Selection")
model_type = st.sidebar.selectbox(
    "Prediction Model",
    options=[
        "Isolation Forest (Anomaly Detection)",
        "Random Forest (Temperature Prediction)",
        "Ensemble (Both)"
    ]
)

# Advanced parameters
with st.sidebar.expander("🔧 Advanced Parameters"):
    contamination = st.slider(
        "Contamination (anomaly rate)",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01
    )
    
    n_estimators = st.slider(
        "Number of Trees",
        min_value=50,
        max_value=300,
        value=100,
        step=50
    )
    
    random_state = st.number_input("Random Seed", value=42, min_value=0, max_value=999)

# Get station data
station_data = st.session_state.results[selected_station]
df_primary = station_data['df_primary'].copy()
flags_df = station_data['flags_df'].copy()
rule_based_flags = flags_df['final_flag'].copy()

# ============================================================================
# Feature Engineering Functions
# ============================================================================
def engineer_features(df):
    """Create features for ML models"""
    df_features = df.copy()
    
    # Time features
    df_features['hour'] = df_features.index.hour
    df_features['day'] = df_features.index.day
    df_features['dayofweek'] = df_features.index.dayofweek
    df_features['month'] = df_features.index.month
    df_features['dayofyear'] = df_features.index.dayofyear
    df_features['weekend'] = (df_features['dayofweek'] >= 5).astype(int)
    
    # Cyclical encoding for time features
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
    
    # Seasonal averages
    df_features['hourly_avg'] = df_features.groupby('hour')['T_air'].transform('mean')
    df_features['monthly_avg'] = df_features.groupby('month')['T_air'].transform('mean')
    
    # Deviation from expected
    df_features['deviation_from_hourly'] = df_features['T_air'] - df_features['hourly_avg']
    df_features['deviation_from_monthly'] = df_features['T_air'] - df_features['monthly_avg']
    
    return df_features

# ============================================================================
# Prepare data with features
# ============================================================================
with st.spinner("Engineering features..."):
    df_features = engineer_features(df_primary)
    
    # Define feature columns (exclude target and NaN-prone columns)
    exclude_cols = ['T_air', 'hourly_avg', 'monthly_avg']  # Keep these for reference but not as features
    feature_cols = [col for col in df_features.columns if col not in exclude_cols and not df_features[col].isna().all()]
    
    # Drop rows with NaN (from lag/rolling features)
    df_clean = df_features.dropna()
    
    if len(df_clean) < 100:
        st.error("Insufficient data after feature engineering. Need at least 100 data points.")
        st.stop()
    
    X = df_clean[feature_cols]
    y = df_clean['T_air']

# ============================================================================
# Split data based on method
# ============================================================================
st.subheader("📊 Data Split Configuration")

if split_method == "Temporal Split (Recommended)":
    # Temporal split based on years
    df_clean['year'] = df_clean.index.year
    available_years = sorted(df_clean['year'].unique())
    
    if len(available_years) < 3:
        st.warning(f"Only {len(available_years)} years available. Using random split instead.")
        split_method = "Random Split"
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        train_idx = X_train.index
        test_idx = X_test.index
    else:
        train_years_list = available_years[:train_years]
        test_years_list = available_years[train_years:]
        
        train_mask = df_clean['year'].isin(train_years_list)
        test_mask = df_clean['year'].isin(test_years_list)
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        train_idx = X_train.index
        test_idx = X_test.index
        
        # Display split info
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Training Years**: {', '.join(map(str, train_years_list))}")
            st.metric("Training Samples", f"{len(X_train):,}")
        with col2:
            st.info(f"**Testing Years**: {', '.join(map(str, test_years_list)) if test_years_list else 'None'}")
            st.metric("Testing Samples", f"{len(X_test):,}")
else:
    # Random split
    test_size_pct = test_size * 100
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    train_idx = X_train.index
    test_idx = X_test.index
    
    st.info(f"**Random Split**: {len(X_train)} training ({100-test_size_pct:.0f}%), {len(X_test)} testing ({test_size_pct:.0f}%)")

# ============================================================================
# Train models and make predictions
# ============================================================================
st.subheader("🤖 Model Training & Prediction")

if len(X_test) == 0:
    st.error("No data available for testing. Adjust your split parameters.")
    st.stop()

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

if model_type in ["Isolation Forest (Anomaly Detection)", "Ensemble (Both)"]:
    with st.spinner("Training Isolation Forest..."):
        # Train Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators
        )
        
        # Fit on training data
        iso_forest.fit(X_train_scaled)
        
        # Predict anomalies on test data
        test_preds = iso_forest.predict(X_test_scaled)
        test_scores = iso_forest.score_samples(X_test_scaled)
        
        # Convert to flags (1 for anomaly, 0 for normal)
        test_anomalies = (test_preds == -1)
        
        results['iso_forest'] = {
            'predictions': test_preds,
            'scores': test_scores,
            'anomalies': test_anomalies,
            'anomaly_count': test_anomalies.sum(),
            'anomaly_rate': test_anomalies.sum() / len(test_anomalies) * 100
        }

if model_type in ["Random Forest (Temperature Prediction)", "Ensemble (Both)"]:
    with st.spinner("Training Random Forest Regressor..."):
        # Train Random Forest for temperature prediction
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test_scaled)
        
        # Calculate errors
        errors = np.abs(y_pred - y_test)
        
        # Define anomaly threshold (e.g., errors > 2 standard deviations)
        error_threshold = np.mean(errors) + 2 * np.std(errors)
        prediction_anomalies = errors > error_threshold
        
        results['random_forest'] = {
            'predictions': y_pred,
            'actual': y_test,
            'errors': errors,
            'anomalies': prediction_anomalies,
            'anomaly_count': prediction_anomalies.sum(),
            'anomaly_rate': prediction_anomalies.sum() / len(prediction_anomalies) * 100,
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'error_threshold': error_threshold
        }

# Add this to your analysis to visualize the differences
st.subheader("ML vs Rule Flag Characteristics")

col1, col2 = st.columns(2)

with col1:
    # Temperature distribution comparison
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df_ml_only['Temperature'],
        name='ML Only',
        opacity=0.7,
        marker_color='orange',
        nbinsx=30
    ))
    fig.add_trace(go.Histogram(
        x=df_rule_only['Temperature'],
        name='Rule Only',
        opacity=0.7,
        marker_color='red',
        nbinsx=30
    ))
    fig.update_layout(
        title="Temperature Distribution: ML vs Rule",
        xaxis_title="Temperature (°C)",
        yaxis_title="Frequency",
        barmode='overlay'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Hour of day pattern
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=df_ml_only['Hour'],
        name='ML Only',
        opacity=0.7,
        marker_color='orange',
        nbinsx=24
    ))
    fig2.add_trace(go.Histogram(
        x=df_rule_only['Hour'],
        name='Rule Only',
        opacity=0.7,
        marker_color='red',
        nbinsx=24
    ))
    fig2.update_layout(
        title="Flag Distribution by Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Frequency",
        barmode='overlay'
    )
    st.plotly_chart(fig2, use_container_width=True)

# Rolling statistics comparison
st.subheader("Rolling Statistics Comparison")

fig3 = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Rolling Mean', 'Rolling Std')
)

# Rolling Mean
fig3.add_trace(
    go.Box(y=df_ml_only['Rolling Mean 6h'].dropna(), name='ML Only', marker_color='orange'),
    row=1, col=1
)
fig3.add_trace(
    go.Box(y=df_rule_only['Rolling Mean 6h'].dropna(), name='Rule Only', marker_color='red'),
    row=1, col=1
)

# Rolling Std
fig3.add_trace(
    go.Box(y=df_ml_only['Rolling Std 6h'].dropna(), name='ML Only', marker_color='orange'),
    row=1, col=2
)
fig3.add_trace(
    go.Box(y=df_rule_only['Rolling Std 6h'].dropna(), name='Rule Only', marker_color='red'),
    row=1, col=2
)

fig3.update_layout(height=400, showlegend=False)
st.plotly_chart(fig3, use_container_width=True)

# ============================================================================
# Display Results
# ============================================================================
st.header("📈 Validation Results")

# Create tabs for different views
result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs([
    "📊 Model Performance",
    "🔍 Anomaly Detection",
    "📉 Prediction Accuracy",
    "⚖️ Comparison with Rule-Based"
])

with result_tab1:
    st.subheader("Model Performance Metrics")
    
    if 'random_forest' in results:
        rf_res = results['random_forest']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAE", f"{rf_res['mae']:.2f}°C")
        with col2:
            st.metric("RMSE", f"{rf_res['rmse']:.2f}°C")
        with col3:
            st.metric("R² Score", f"{rf_res['r2']:.3f}")
        with col4:
            st.metric("Error Threshold", f"{rf_res['error_threshold']:.2f}°C")
    
    if 'iso_forest' in results:
        iso_res = results['iso_forest']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Isolation Forest Anomalies", f"{iso_res['anomaly_count']:,}")
        with col2:
            st.metric("Anomaly Rate", f"{iso_res['anomaly_rate']:.2f}%")

with result_tab2:
    st.subheader("Anomaly Detection on Test Data")
    
    if 'iso_forest' in results:
        iso_res = results['iso_forest']
        
        # Create visualization
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Temperature (Test Set)', 'Isolation Forest Anomaly Score', 'Detected Anomalies'),
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        # Temperature
        fig.add_trace(
            go.Scatter(x=test_idx, y=y_test,
                      mode='lines', name='Temperature',
                      line=dict(color='blue', width=1)),
            row=1, col=1
        )
        
        # Anomaly scores
        fig.add_trace(
            go.Scatter(x=test_idx, y=iso_res['scores'],
                      mode='lines', name='Anomaly Score',
                      line=dict(color='orange', width=1)),
            row=2, col=1
        )
        
        # Add threshold line
        threshold = np.percentile(iso_res['scores'], contamination * 100)
        fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                     annotation_text="Threshold", row=2, col=1)
        
        # Anomalies
        anomaly_y = [1 if a else 0 for a in iso_res['anomalies']]
        fig.add_trace(
            go.Scatter(x=test_idx, y=anomaly_y,
                      mode='markers', name='Anomalies',
                      marker=dict(color='red', size=4, symbol='square')),
            row=3, col=1
        )
        
        fig.update_layout(height=600, title_text="Isolation Forest Anomaly Detection on Test Data")
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=2, col=1)
        fig.update_yaxes(title_text="Anomaly (0/1)", row=3, col=1, range=[-0.1, 1.1])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample anomalies
        if iso_res['anomaly_count'] > 0:
            st.subheader("🔍 Sample Anomalies Detected")
            anomaly_indices = test_idx[iso_res['anomalies']]
            sample_size = min(20, len(anomaly_indices))
            sample_idx = anomaly_indices[:sample_size]
            
            sample_df = pd.DataFrame({
                'Date': sample_idx,
                'Temperature': y_test.loc[sample_idx],
                'Anomaly Score': iso_res['scores'][iso_res['anomalies']][:sample_size]
            })
            st.dataframe(sample_df)
    
    elif 'random_forest' in results:
        rf_res = results['random_forest']
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Temperature (Test Set)', 'Prediction Error', 'Detected Anomalies'),
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        # Actual vs Predicted
        fig.add_trace(
            go.Scatter(x=test_idx, y=y_test,
                      mode='lines', name='Actual',
                      line=dict(color='blue', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=test_idx, y=rf_res['predictions'],
                      mode='lines', name='Predicted',
                      line=dict(color='red', width=1, dash='dash')),
            row=1, col=1
        )
        
        # Errors
        fig.add_trace(
            go.Scatter(x=test_idx, y=rf_res['errors'],
                      mode='lines', name='Absolute Error',
                      line=dict(color='orange', width=1)),
            row=2, col=1
        )
        
        # Add threshold line
        fig.add_hline(y=rf_res['error_threshold'], line_dash="dash", line_color="red",
                     annotation_text="Threshold", row=2, col=1)
        
        # Anomalies
        anomaly_y = [1 if a else 0 for a in rf_res['anomalies']]
        fig.add_trace(
            go.Scatter(x=test_idx, y=anomaly_y,
                      mode='markers', name='Anomalies',
                      marker=dict(color='red', size=4, symbol='square')),
            row=3, col=1
        )
        
        fig.update_layout(height=600, title_text="Random Forest Prediction Errors on Test Data")
        fig.update_xaxes(title_text="Date", row=3, col=1)
        st.plotly_chart(fig, use_container_width=True)

with result_tab3:
    st.subheader("Prediction Accuracy Analysis")
    
    if 'random_forest' in results:
        rf_res = results['random_forest']
        
        # Scatter plot: Actual vs Predicted
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=y_test, y=rf_res['predictions'],
            mode='markers',
            marker=dict(
                color=rf_res['errors'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Error (°C)")
            ),
            text=[f"Date: {idx}<br>Actual: {act:.1f}°C<br>Pred: {pred:.1f}°C<br>Error: {err:.2f}°C" 
                  for idx, act, pred, err in zip(test_idx, y_test, rf_res['predictions'], rf_res['errors'])],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add perfect prediction line
        min_val = min(y_test.min(), rf_res['predictions'].min())
        max_val = max(y_test.max(), rf_res['predictions'].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="Actual vs Predicted Temperature",
            xaxis_title="Actual Temperature (°C)",
            yaxis_title="Predicted Temperature (°C)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Error distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=rf_res['errors'],
                nbinsx=50,
                marker_color='lightblue',
                opacity=0.7
            ))
            fig2.add_vline(x=rf_res['error_threshold'], line_dash="dash", line_color="red")
            fig2.update_layout(
                title="Error Distribution",
                xaxis_title="Absolute Error (°C)",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Error by hour
            test_hours = pd.Series(test_idx).dt.hour.values
            error_by_hour = pd.DataFrame({
                'hour': test_hours,
                'error': rf_res['errors']
            }).groupby('hour').mean()
            
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=error_by_hour.index,
                y=error_by_hour['error'],
                marker_color='coral'
            ))
            fig3.update_layout(
                title="Mean Error by Hour of Day",
                xaxis_title="Hour",
                yaxis_title="Mean Absolute Error (°C)",
                height=400
            )
            st.plotly_chart(fig3, use_container_width=True)

with result_tab4:
    st.subheader("Comparison with Rule-Based QA/QC")
    
    # Get rule-based flags for test set
    test_rule_flags = rule_based_flags.loc[test_idx]
    
    if 'iso_forest' in results:
        iso_res = results['iso_forest']
        
        # Create comparison dataframe
        comparison = pd.DataFrame(index=test_idx)
        comparison['Rule Flag'] = test_rule_flags.astype(int)
        comparison['ML Flag'] = iso_res['anomalies'].astype(int)
        comparison['Flag Type'] = 'None'
        comparison.loc[(comparison['Rule Flag'] == 1) & (comparison['ML Flag'] == 1), 'Flag Type'] = 'Both'
        comparison.loc[(comparison['Rule Flag'] == 1) & (comparison['ML Flag'] == 0), 'Flag Type'] = 'Rule Only'
        comparison.loc[(comparison['Rule Flag'] == 0) & (comparison['ML Flag'] == 1), 'Flag Type'] = 'ML Only'
        
        # Summary statistics
        st.subheader("Flag Comparison on Test Data")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rule Flags", f"{(comparison['Rule Flag'] == 1).sum():,}")
        with col2:
            st.metric("ML Flags", f"{(comparison['ML Flag'] == 1).sum():,}")
        with col3:
            both = ((comparison['Rule Flag'] == 1) & (comparison['ML Flag'] == 1)).sum()
            st.metric("Both", f"{both:,}")
        with col4:
            ml_only = ((comparison['Rule Flag'] == 0) & (comparison['ML Flag'] == 1)).sum()
            st.metric("ML Only", f"{ml_only:,}")
        
        # Visualization
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Temperature (Test Set)', 'Rule-Based Flags', 'ML Flags'),
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        # Temperature
        fig.add_trace(
            go.Scatter(x=test_idx, y=y_test,
                      mode='lines', name='Temperature',
                      line=dict(color='blue', width=1)),
            row=1, col=1
        )
        
        # Rule flags
        fig.add_trace(
            go.Scatter(x=test_idx, y=comparison['Rule Flag'],
                      mode='markers', name='Rule Flags',
                      marker=dict(color='red', size=4, symbol='square')),
            row=2, col=1
        )
        
        # ML flags
        fig.add_trace(
            go.Scatter(x=test_idx, y=comparison['ML Flag'],
                      mode='markers', name='ML Flags',
                      marker=dict(color='orange', size=3, opacity=0.6)),
            row=3, col=1
        )
        
        fig.update_layout(height=500, title_text="Rule-Based vs ML Flag Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        # Flag type distribution
        flag_counts = comparison['Flag Type'].value_counts()
        
        fig2 = go.Figure(data=[go.Pie(
            labels=flag_counts.index,
            values=flag_counts.values,
            hole=0.3,
            marker_colors=['purple', 'red', 'orange', 'gray']
        )])
        
        fig2.update_layout(title="Flag Type Distribution on Test Data", height=400)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Download comparison data
        csv = comparison.to_csv()
        st.download_button(
            label="📥 Download Comparison Data",
            data=csv,
            file_name=f"ml_rule_comparison_{selected_station}.csv",
            mime="text/csv"
        )

# ============================================================================
# Model Export and Recommendations
# ============================================================================
st.header("💾 Model Export & Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Performance Summary")
    
    summary_text = f"""
    **Station:** {selected_station}
    **Training Period:** {len(X_train)} samples
    **Testing Period:** {len(X_test)} samples
    
    """
    
    if 'random_forest' in results:
        rf_res = results['random_forest']
        summary_text += f"""
        **Random Forest:**
        - MAE: {rf_res['mae']:.2f}°C
        - RMSE: {rf_res['rmse']:.2f}°C
        - R²: {rf_res['r2']:.3f}
        - Anomaly Rate: {rf_res['anomaly_rate']:.2f}%
        """
    
    if 'iso_forest' in results:
        iso_res = results['iso_forest']
        summary_text += f"""
        **Isolation Forest:**
        - Anomalies Detected: {iso_res['anomaly_count']}
        - Anomaly Rate: {iso_res['anomaly_rate']:.2f}%
        """
    
    st.info(summary_text)

with col2:
    st.subheader("Recommendations")
    
    recommendations = []
    
    if 'random_forest' in results:
        rf_res = results['random_forest']
        if rf_res['r2'] > 0.8:
            recommendations.append("✅ Random Forest shows excellent predictive power (R² > 0.8)")
        elif rf_res['r2'] > 0.6:
            recommendations.append("🟡 Random Forest shows good predictive power (R² > 0.6)")
        else:
            recommendations.append("🔴 Random Forest shows poor predictive power - consider more features")
        
        if rf_res['mae'] < 2:
            recommendations.append(f"✅ Prediction error is low (MAE = {rf_res['mae']:.2f}°C)")
        else:
            recommendations.append(f"⚠️ High prediction error (MAE = {rf_res['mae']:.2f}°C)")
    
    if 'iso_forest' in results:
        iso_res = results['iso_forest']
        rule_rate = rule_based_flags.loc[test_idx].mean() * 100
        
        if iso_res['anomaly_rate'] > rule_rate * 2:
            recommendations.append("🔍 ML detects 2x more anomalies than rule-based - review for false positives")
        elif iso_res['anomaly_rate'] < rule_rate / 2:
            recommendations.append("📊 ML is more conservative than rule-based - may miss some anomalies")
        else:
            recommendations.append("✅ ML anomaly rate aligns with rule-based expectations")
    
    for rec in recommendations:
        st.markdown(rec)
    
    if not recommendations:
        st.info("Run the models to see recommendations")

# Footer
st.markdown("---")
st.markdown("""
**How to interpret these results:**

1. **Temporal Split**: Models are trained on older data and tested on newer data - this simulates real-world forecasting
2. **Anomaly Detection**: ML finds patterns that deviate from learned behavior in the test set
3. **Prediction Accuracy**: Lower MAE/RMSE indicates better temperature prediction
4. **Comparison with Rules**: See how ML flags compare to your existing QA/QC system on unseen data

The goal is to validate that ML can generalize to new, unseen data before deploying it in production.
""")
