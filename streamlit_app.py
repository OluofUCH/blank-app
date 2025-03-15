import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from supabase import create_client
import datetime
import calendar
from statsmodels.nonparametric.smoothers_lowess import lowess

# Set page configuration
st.set_page_config(
    page_title="Running Analytics Dashboard",
    page_icon="🏃",
    layout="wide"
)

# Supabase connection
@st.cache_resource
def init_connection():
    url = st.secrets["supabase_url"]
    key = st.secrets["supabase_key"]
    return create_client(url, key)

# Fetch data from Supabase
@st.cache_data(ttl=600)
def fetch_data():
    supabase = init_connection()
    response = supabase.table("activities").select("*").execute()
    
    # Convert to DataFrame
    data = pd.DataFrame(response.data)
    
    # Convert date strings to datetime objects
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
    
    return data

# Main function
def main():
    st.title("🏃 Running Performance Analytics")
    
    try:
        # Load data
        df = fetch_data()
        
        # Check if data is available
        if df.empty:
            st.error("No data found in the activities table.")
            return
        
        # Add sidebar for filtering
        st.sidebar.header("Filter Data")
        
        # Date range filter
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            df_filtered = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
        else:
            df_filtered = df
        
        # Display data summary
        st.sidebar.markdown(f"**Total Activities:** {len(df_filtered)}")
        st.sidebar.markdown(f"**Date Range:** {df_filtered['date'].min().date()} to {df_filtered['date'].max().date()}")
        
        if 'distance' in df_filtered.columns:
            total_distance = df_filtered['distance'].sum()
            st.sidebar.markdown(f"**Total Distance:** {total_distance:.2f} km")
        
        # Create tabs for different visualization categories
        tab1, tab2 = st.tabs(["Tracking Visualizations", "Research Visualizations"])
        
        with tab1:
            st.header("Training Tracking Visualizations")
            
            # Visualization 5: Running Consistency Calendar Heatmap
            st.subheader("5. Running Consistency Calendar Heatmap")
            render_consistency_heatmap(df_filtered)
            
        with tab2:
            st.header("Research-Focused Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Visualization 6: Performance Over Time Regression Line
                st.subheader("6. Performance Over Time Regression")
                render_performance_regression(df_filtered)
                
                # Visualization 8: Box Plot of Running Performance by Training Type
                st.subheader("8. Box Plot by Training Type")
                render_boxplot_by_training_type(df_filtered)
                
            with col2:
                # Visualization 7: Correlation Matrix
                st.subheader("7. Correlation Matrix")
                render_correlation_matrix(df_filtered)
                
                # Visualization 9: Outlier Detection
                st.subheader("9. Outlier Detection")
                render_outlier_detection(df_filtered)
            
            # Visualization 10: Clustering Analysis
            st.subheader("10. Clustering Analysis")
            render_clustering_analysis(df_filtered)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please check your Supabase connection and data structure.")

# Visualization 5: Running Consistency Calendar Heatmap
def render_consistency_heatmap(df):
    if 'date' not in df.columns:
        st.error("Date column not found in the data.")
        return
    
    # Create a calendar heatmap
    if not df.empty:
        # Get the year range
        min_year = df['date'].dt.year.min()
        max_year = df['date'].dt.year.max()
        
        # Year selection
        selected_year = st.selectbox("Select Year", range(min_year, max_year + 1), index=len(range(min_year, max_year + 1)) - 1)
        
        # Filter data for the selected year
        year_data = df[df['date'].dt.year == selected_year]
        
        if year_data.empty:
            st.info(f"No running data available for {selected_year}.")
            return
        
        # Create daily activity count
        daily_counts = year_data.groupby(year_data['date'].dt.date).size().reset_index()
        daily_counts.columns = ['date', 'count']
        
        # Create a complete date range for the year
        all_dates = pd.date_range(start=f"{selected_year}-01-01", end=f"{selected_year}-12-31", freq='D')
        all_dates_df = pd.DataFrame({'date': all_dates})
        all_dates_df['date'] = all_dates_df['date'].dt.date
        
        # Merge with actual data to include zeros for days with no activity
        merged_data = all_dates_df.merge(daily_counts, on='date', how='left').fillna(0)
        
        # Format for calendar heatmap
        merged_data['month'] = pd.to_datetime(merged_data['date']).dt.month
        merged_data['day'] = pd.to_datetime(merged_data['date']).dt.day
        merged_data['weekday'] = pd.to_datetime(merged_data['date']).dt.weekday
        
        # Custom darker blue colorscale
        colorscale = [
            [0, 'rgb(240, 240, 255)'],  # Very light blue for 0
            [0.2, 'rgb(200, 200, 240)'],
            [0.4, 'rgb(150, 150, 220)'],
            [0.6, 'rgb(100, 100, 200)'],
            [0.8, 'rgb(50, 50, 180)'],
            [1, 'rgb(0, 0, 150)']        # Dark blue for max
        ]
        
        # Create monthly calendar plots
        months_per_row = 3
        num_rows = (12 + months_per_row - 1) // months_per_row
        
        for row in range(num_rows):
            cols = st.columns(months_per_row)
            for i, month_idx in enumerate(range(row * months_per_row + 1, min(12 + 1, (row + 1) * months_per_row + 1))):
                month_name = calendar.month_name[month_idx]
                month_data = merged_data[merged_data['month'] == month_idx].copy()
                
                # Handle case where month has no data
                if month_data.empty:
                    cols[i].info(f"No data for {month_name}")
                    continue
                
                # Create heatmap
                fig = go.Figure()
                
                # Get weekday names
                weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                
                # Create heatmap
                fig = px.imshow(
                    month_data.pivot(index='weekday', columns='day', values='count'),
                    labels=dict(x="Day of Month", y="Day of Week", color="Activities"),
                    x=sorted(month_data['day'].unique()),
                    y=weekday_names,
                    color_continuous_scale=colorscale,
                    height=200
                )
                
                fig.update_layout(
                    title=month_name,
                    margin=dict(l=10, r=10, t=30, b=10),
                    coloraxis_showscale=True if month_idx % 3 == 0 else False,
                )
                
                cols[i].plotly_chart(fig, use_container_width=True)

# Visualization 6: Performance Over Time Regression Line
def render_performance_regression(df):
    if 'date' not in df.columns:
        st.error("Date column not found in the data.")
        return
    
    # Select metrics
    metric_options = [col for col in df.columns if col not in ['date', 'id', 'name', 'type', 'notes']]
    if not metric_options:
        st.error("No numerical metrics found in the data.")
        return
    
    selected_metric = st.selectbox("Select Performance Metric", metric_options, key="regression_metric")
    
    if selected_metric not in df.columns:
        st.error(f"Selected metric '{selected_metric}' not found in data.")
        return
    
    # Create a copy of the data with only relevant columns
    regression_data = df[['date', selected_metric]].dropna().copy()
    
    if len(regression_data) < 3:
        st.warning("Not enough data points for regression analysis.")
        return
    
    # Calculate days since first activity
    regression_data['days_since_start'] = (regression_data['date'] - regression_data['date'].min()).dt.days
    
    # Create the scatter plot
    fig = px.scatter(
        regression_data, 
        x='days_since_start', 
        y=selected_metric,
        title=f"{selected_metric} Over Time",
        labels={
            'days_since_start': 'Days Since First Activity',
            selected_metric: selected_metric.replace('_', ' ').title()
        }
    )
    
    # Add regression line
    X = regression_data['days_since_start'].values.reshape(-1, 1)
    y = regression_data[selected_metric].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    x_range = np.linspace(0, X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)
    
    # Create trendline with loess smoothing for more realistic curve
    lowess_data = lowess(y, X.flatten(), frac=0.3)
    
    # Add the trendlines
    fig.add_trace(
        go.Scatter(
            x=x_range.flatten(), 
            y=y_pred, 
            mode='lines', 
            line=dict(color='blue', width=2),
            name='Linear Trend'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=lowess_data[:, 0], 
            y=lowess_data[:, 1], 
            mode='lines', 
            line=dict(color='red', width=2),
            name='Smoothed Trend'
        )
    )
    
    # Add projection for next 30 days
    last_day = regression_data['days_since_start'].max()
    future_days = np.linspace(last_day + 1, last_day + 30, 30).reshape(-1, 1)
    future_pred = model.predict(future_days)
    
    fig.add_trace(
        go.Scatter(
            x=future_days.flatten(), 
            y=future_pred, 
            mode='lines', 
            line=dict(color='green', dash='dash', width=2),
            name='30-Day Projection'
        )
    )
    
    # Calculate improvement metrics
    if len(regression_data) >= 2:
        first_value = regression_data.iloc[0][selected_metric]
        last_value = regression_data.iloc[-1][selected_metric]
        
        # Calculate monthly improvement rate
        days_total = regression_data['days_since_start'].max()
        if days_total >= 30:
            monthly_improvement = model.coef_[0] * 30
            monthly_pct = (monthly_improvement / first_value) * 100 if first_value != 0 else 0
            
            # Add annotation about improvement rate
            annotation_text = f"Monthly improvement: {monthly_improvement:.2f} ({monthly_pct:.1f}%)"
            fig.add_annotation(
                x=X.max() * 0.5,
                y=regression_data[selected_metric].max(),
                text=annotation_text,
                showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.8)"
            )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Days Since First Activity",
        yaxis_title=selected_metric.replace('_', ' ').title(),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display regression stats
    slope = model.coef_[0]
    intercept = model.intercept_
    
    stats_col1, stats_col2 = st.columns(2)
    stats_col1.metric("Daily Improvement Rate", f"{slope:.4f} units/day")
    
    if slope > 0:
        stats_col2.metric("Trend", "Improving 📈", delta="positive")
    elif slope < 0:
        stats_col2.metric("Trend", "Declining 📉", delta="negative")
    else:
        stats_col2.metric("Trend", "Stable ↔️", delta="none")

# Visualization 7: Correlation Matrix
def render_correlation_matrix(df):
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['id']]
    
    if len(numeric_cols) < 2:
        st.error("Not enough numerical columns for correlation analysis.")
        return
    
    # Allow user to select columns
    selected_cols = st.multiselect(
        "Select metrics for correlation analysis:",
        options=numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))],
        key="corr_metrics"
    )
    
    if len(selected_cols) < 2:
        st.warning("Please select at least 2 metrics for correlation analysis.")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[selected_cols].corr()
    
    # Create the heatmap using plotly
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    corr_z = corr_matrix.mask(mask).round(2)
    
    # Custom colorscale from white to dark blue
    colorscale = [
        [0, 'rgb(198, 219, 239)'],  # Light
    [0, 'rgb(198, 219, 239)'],  # Light blue for negative correlations
        [0.5, 'rgb(255, 255, 255)'],  # White for zero correlation
        [1, 'rgb(8, 48, 107)']        # Dark blue for positive correlations
    ]
    
    fig = go.Figure()
    
    # Add heatmap trace
    heatmap = go.Heatmap(
        z=corr_z,
        x=corr_z.columns,
        y=corr_z.columns,
        colorscale=colorscale,
        zmin=-1, zmax=1,
        text=corr_z.values,
        hoverinfo='text',
        texttemplate='%{text:.2f}'
    )
    fig.add_trace(heatmap)
    
    # Update layout
    fig.update_layout(
        title='Correlation Matrix',
        height=500,
        width=500,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_title='',
        yaxis_title='',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display key insights
    st.subheader("Key Correlation Insights")
    
    # Find strongest positive and negative correlations
    corr_pairs = []
    for i in range(len(selected_cols)):
        for j in range(i+1, len(selected_cols)):
            corr_value = corr_matrix.iloc[i, j]
            corr_pairs.append((selected_cols[i], selected_cols[j], corr_value))
    
    # Sort by absolute correlation value
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # Display top correlations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Strongest Positive Correlations:**")
        positive_corrs = [pair for pair in corr_pairs if pair[2] > 0][:3]
        
        if positive_corrs:
            for pair in positive_corrs:
                var1, var2, corr = pair
                st.write(f"• {var1} & {var2}: {corr:.2f}")
        else:
            st.write("No positive correlations found.")
    
    with col2:
        st.markdown("**Strongest Negative Correlations:**")
        negative_corrs = [pair for pair in corr_pairs if pair[2] < 0][:3]
        
        if negative_corrs:
            for pair in negative_corrs:
                var1, var2, corr = pair
                st.write(f"• {var1} & {var2}: {corr:.2f}")
        else:
            st.write("No negative correlations found.")
            
    # Interpretation of key correlations
    if corr_pairs:
        st.markdown("**Interpretation:**")
        top_corr = corr_pairs[0]
        var1, var2, corr = top_corr
        
        if corr > 0.7:
            st.write(f"There is a strong positive relationship between {var1} and {var2}, suggesting that as {var1} increases, {var2} tends to increase as well.")
        elif corr > 0.3:
            st.write(f"There is a moderate positive relationship between {var1} and {var2}.")
        elif corr > -0.3:
            st.write(f"There is a weak or no significant relationship between {var1} and {var2}.")
        elif corr > -0.7:
            st.write(f"There is a moderate negative relationship between {var1} and {var2}.")
        else:
            st.write(f"There is a strong negative relationship between {var1} and {var2}, suggesting that as {var1} increases, {var2} tends to decrease.")

# Visualization 8: Box Plot of Running Performance by Training Type
def render_boxplot_by_training_type(df):
    if 'type' not in df.columns:
        st.error("Training type column not found in the data.")
        return
    
    # Get unique training types
    training_types = df['type'].unique()
    
    if len(training_types) < 2:
        st.warning("Not enough training types for comparison.")
        return
    
    # Select performance metric
    metric_options = [col for col in df.columns if col not in ['date', 'id', 'name', 'type', 'notes']]
    
    if not metric_options:
        st.error("No numerical metrics found in the data.")
        return
    
    selected_metric = st.selectbox("Select Performance Metric", metric_options, key="boxplot_metric")
    
    # Create box plot
    fig = px.box(
        df,
        x='type',
        y=selected_metric,
        color='type',
        title=f"{selected_metric.replace('_', ' ').title()} by Training Type",
        labels={
            'type': 'Training Type',
            selected_metric: selected_metric.replace('_', ' ').title()
        },
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Training Type",
        yaxis_title=selected_metric.replace('_', ' ').title(),
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate and display statistics
    stats = df.groupby('type')[selected_metric].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
    stats = stats.round(2)
    
    st.subheader("Statistical Summary by Training Type")
    st.dataframe(stats, use_container_width=True)
    
    # ANOVA test for statistical significance
    if len(training_types) >= 2 and len(df) >= 10:
        st.markdown("**Statistical Significance:**")
        
        from scipy import stats as scipy_stats
        import numpy as np
        
        # Prepare data for ANOVA
        groups = []
        for t_type in training_types:
            type_data = df[df['type'] == t_type][selected_metric].dropna()
            if len(type_data) > 0:
                groups.append(type_data)
        
        if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
            # Perform one-way ANOVA
            f_stat, p_value = scipy_stats.f_oneway(*groups)
            
            if p_value < 0.05:
                st.write(f"There is a statistically significant difference in {selected_metric} between training types (p-value: {p_value:.4f}).")
            else:
                st.write(f"No statistically significant difference detected in {selected_metric} between training types (p-value: {p_value:.4f}).")
        else:
            st.write("Not enough data in each group for statistical comparison.")

# Visualization 9: Outlier Detection
def render_outlier_detection(df):
    # Select metric for outlier detection
    metric_options = [col for col in df.columns if col not in ['date', 'id', 'name', 'type', 'notes']]
    
    if not metric_options:
        st.error("No numerical metrics found in the data.")
        return
    
    selected_metric = st.selectbox("Select Metric for Outlier Detection", metric_options, key="outlier_metric")
    
    # Create a copy of the dataframe with relevant columns
    outlier_df = df[['date', selected_metric]].dropna().copy()
    
    if len(outlier_df) < 5:
        st.warning("Not enough data points for outlier analysis.")
        return
    
    # Calculate Z-scores
    outlier_df['z_score'] = (outlier_df[selected_metric] - outlier_df[selected_metric].mean()) / outlier_df[selected_metric].std()
    
    # Define outliers as points with absolute Z-score > 2
    outlier_threshold = 2.0
    outlier_df['is_outlier'] = outlier_df['z_score'].abs() > outlier_threshold
    
    # Sort by date
    outlier_df = outlier_df.sort_values('date')
    
    # Create scatter plot
    fig = px.scatter(
        outlier_df,
        x=outlier_df.index,
        y=selected_metric,
        color='is_outlier',
        color_discrete_map={True: 'red', False: 'blue'},
        labels={
            selected_metric: selected_metric.replace('_', ' ').title(),
            'index': 'Activity Number'
        },
        title=f"Outlier Detection for {selected_metric.replace('_', ' ').title()}",
        hover_data=['date', 'z_score']
    )
    
    # Add horizontal lines for mean and standard deviation bounds
    mean_val = outlier_df[selected_metric].mean()
    std_val = outlier_df[selected_metric].std()
    
    fig.add_hline(y=mean_val, line_dash="dash", line_color="green", annotation_text="Mean")
    fig.add_hline(y=mean_val + outlier_threshold * std_val, line_dash="dot", line_color="orange", annotation_text=f"+{outlier_threshold} SD")
    fig.add_hline(y=mean_val - outlier_threshold * std_val, line_dash="dot", line_color="orange", annotation_text=f"-{outlier_threshold} SD")
    
    # Update layout
    fig.update_layout(
        xaxis_title="Activity Number",
        yaxis_title=selected_metric.replace('_', ' ').title(),
        showlegend=True,
        legend_title="Is Outlier",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display outlier summary
    outlier_count = outlier_df['is_outlier'].sum()
    outlier_percentage = (outlier_count / len(outlier_df)) * 100
    
    col1, col2 = st.columns(2)
    col1.metric("Number of Outliers", f"{outlier_count}")
    col2.metric("Percentage of Outliers", f"{outlier_percentage:.1f}%")
    
    # Show outlier details
    if outlier_count > 0:
        st.subheader("Outlier Details")
        outliers = outlier_df[outlier_df['is_outlier']].sort_values('z_score', ascending=False)
        
        for i, (_, row) in enumerate(outliers.iterrows()):
            if i < 5:  # Show only top 5 outliers
                direction = "above" if row['z_score'] > 0 else "below"
                st.write(f"• {row['date'].date()}: {row[selected_metric]:.2f} ({direction} average by {abs(row['z_score']):.2f} standard deviations)")

# Visualization 10: Clustering Analysis
def render_clustering_analysis(df):
    st.write("This visualization groups your running activities into clusters based on similar characteristics.")
    
    # Select features for clustering
    feature_options = [col for col in df.columns if col not in ['date', 'id', 'name', 'type', 'notes']]
    
    if len(feature_options) < 2:
        st.error("Not enough numerical features for clustering analysis.")
        return
    
    # Allow user to select columns for clustering
    selected_features = st.multiselect(
        "Select features for clustering:",
        options=feature_options,
        default=feature_options[:min(3, len(feature_options))],
        key="cluster_features"
    )
    
    if len(selected_features) < 2:
        st.warning("Please select at least 2 features for clustering.")
        return
    
    # Number of clusters
    num_clusters = st.slider("Number of Clusters", min_value=2, max_value=6, value=3)
    
    # Prepare data for clustering
    cluster_data = df[selected_features].dropna().copy()
    
    if len(cluster_data) < num_clusters:
        st.warning(f"Not enough complete data points for {num_clusters} clusters.")
        return
    
    # Scale the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Add cluster labels to the data
    cluster_data['cluster'] = cluster_labels
    
    # Select two features for visualization
    if len(selected_features) >= 2:
        x_feature = st.selectbox("X-axis feature:", selected_features, index=0)
        y_feature = st.selectbox("Y-axis feature:", selected_features, index=min(1, len(selected_features)-1))
        
        # Create scatter plot
        fig = px.scatter(
            cluster_data,
            x=x_feature,
            y=y_feature,
            color='cluster',
            labels={
                x_feature: x_feature.replace('_', ' ').title(),
                y_feature: y_feature.replace('_', ' ').title(),
                'cluster': 'Cluster'
            },
            title=f"Clustering of Running Activities ({num_clusters} clusters)",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        # Add cluster centroids
        centroids = kmeans.cluster_centers_
        
        # Convert centroids back to original scale
        centroids_original = scaler.inverse_transform(centroids)
        
        # Extract the coordinates for the selected features
        feature_indices = [selected_features.index(x_feature), selected_features.index(y_feature)]
        
        # Add centroids to the plot
        for i in range(num_clusters):
            fig.add_trace(
                go.Scatter(
                    x=[centroids_original[i, feature_indices[0]]],
                    y=[centroids_original[i, feature_indices[1]]],
                    mode='markers',
                    marker=dict(
                        symbol='star',
                        size=15,
                        color=px.colors.qualitative.Bold[i % len(px.colors.qualitative.Bold)],
                        line=dict(color='black', width=1)
                    ),
                    name=f'Centroid {i}'
                )
            )
        
        # Update layout
        fig.update_layout(
            xaxis_title=x_feature.replace('_', ' ').title(),
            yaxis_title=y_feature.replace('_', ' ').title(),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display cluster statistics
        st.subheader("Cluster Statistics")
        
        # Calculate cluster summaries
        cluster_stats = cluster_data.groupby('cluster')[selected_features].agg(['mean', 'count']).reset_index()
        
        # Flatten the MultiIndex
        cluster_stats.columns = ['_'.join(col).strip('_') for col in cluster_stats.columns.values]
        
        # Format table
        formatted_stats = pd.DataFrame()
        formatted_stats['Cluster'] = cluster_
