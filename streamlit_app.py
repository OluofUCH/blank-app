import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import datetime
import calendar

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
    try:
        supabase = init_connection()
        response = supabase.table("activities").select("*").execute()
        
        # Convert to DataFrame
        data = pd.DataFrame(response.data)
        
        # Debug: Print raw data types
        print("Original data types:")
        print(data.dtypes)
        
        # Rename start_date to date if it exists
        if 'start_date' in data.columns and 'date' not in data.columns:
            data = data.rename(columns={'start_date': 'date'})
        
        # Convert date strings to datetime objects
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
        
        # Identify problematic columns and catch conversion errors for each column separately
        numeric_columns = ['distance', 'duration', 'pace', 'heart_rate', 'elevation_gain']
        
        # Process each column individually
        for col in numeric_columns:
            if col in data.columns:
                try:
                    # First check for problematic values
                    print(f"Processing column: {col}")
                    print(f"Sample values in {col}: {data[col].head().tolist()}")
                    
                    # Convert with extra safety
                    data[col] = data[col].astype(str).str.replace(',', '.').str.extract('([-+]?\d*\.?\d+)', expand=False)
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    
                except Exception as col_error:
                    print(f"Error processing column {col}: {col_error}")
                    # If conversion fails, set column to NaN
                    data[col] = np.nan
        
        # For debugging, print any rows with extremely large strings
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check for suspiciously long strings
                max_len = data[col].astype(str).map(len).max()
                if max_len > 1000:  # Arbitrary threshold for "too long"
                    print(f"Column {col} has suspicious long values (max length: {max_len})")
                    # Find the problematic rows
                    long_value_indices = data[col].astype(str).map(len) > 1000
                    print(f"Rows with long values: {long_value_indices.sum()}")
                    # Set these to NaN
                    data.loc[long_value_indices, col] = np.nan
        
        return data
    
    except Exception as e:
        print(f"Error in fetch_data: {e}")
        st.error(f"Error fetching data: {e}")
        # Return empty DataFrame for debugging
        return pd.DataFrame()

# For debugging: Use sample data if Supabase connection fails
def get_sample_data():
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='3D')
    
    data = {
        'date': dates,
        'distance': np.random.uniform(3, 15, size=len(dates)),
        'duration': np.random.uniform(20, 120, size=len(dates)),
        'pace': np.random.uniform(4, 7, size=len(dates)),
        'heart_rate': np.random.uniform(120, 180, size=len(dates)),
        'elevation_gain': np.random.uniform(0, 300, size=len(dates)),
        'type': np.random.choice(['Easy Run', 'Tempo Run', 'Long Run', 'Interval', 'Recovery'], size=len(dates))
    }
    
    return pd.DataFrame(data)

# 5. Running Consistency Calendar Heatmap
def render_consistency_heatmap(df):
    # First, check if 'date' column exists and is datetime
    if 'date' not in df.columns:
        st.error("No 'date' column found. Available columns: " + ", ".join(df.columns.tolist()))
        return
    
    # Check if date column has valid datetime values
    if not pd.api.types.is_datetime64_dtype(df['date']):
        st.error("'date' column is not in datetime format. Attempting to convert...")
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # Check if conversion worked
            if df['date'].isna().all():
                st.error("Could not convert 'date' column to datetime format.")
                return
        except Exception as e:
            st.error(f"Error converting dates: {e}")
            return
    
    # Rest of your function remains the same
    # ...
    
    # Create a calendar heatmap
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
            fig = px.imshow(
                month_data.pivot(index='weekday', columns='day', values='count'),
                labels=dict(x="Day of Month", y="Day of Week", color="Activities"),
                x=sorted(month_data['day'].unique()),
                y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                color_continuous_scale='Blues',
                height=200
            )
            
            fig.update_layout(
                title=month_name,
                margin=dict(l=10, r=10, t=30, b=10),
                coloraxis_showscale=True if month_idx % 3 == 0 else False,
            )
            
            cols[i].plotly_chart(fig, use_container_width=True)

# 6. Performance Over Time Regression Line
def render_performance_regression(df):
    if 'date' not in df.columns or df.empty:
        st.error("No date data available for regression analysis.")
        return
    
    # Select metrics
    metric_options = [col for col in df.columns if col not in ['date', 'id', 'name', 'type', 'notes']]
    if not metric_options:
        st.error("No numerical metrics found in the data.")
        return
    
    # Completely rewrite this line with consistent indentation
    selected_metric = st.selectbox("Select Performance Metric", metric_options, 
                                  index=metric_options.index('distance') if 'distance' in metric_options else 0, 
                                  key="regression_metric")
    
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
    
    # Add the trendline
    fig.add_trace(
        go.Scatter(
            x=x_range.flatten(), 
            y=y_pred, 
            mode='lines', 
            line=dict(color='blue', width=2),
            name='Linear Trend'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"{selected_metric} Over Time",
        xaxis_title="Days Since First Activity",
        yaxis_title=selected_metric.replace('_', ' ').title(),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# 7. Correlation Matrix
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
        default=numeric_cols[:min(5, len(numeric_cols))]
    )
    
    if len(selected_cols) < 2:
        st.warning("Please select at least 2 metrics for correlation analysis.")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[selected_cols].corr()
    
    # Create the heatmap using plotly
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        text_auto=True
    )
    
    # Update layout
    fig.update_layout(
        title='Correlation Matrix',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# 8. Box Plot of Running Performance by Training Type
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
    
    selected_metric = st.selectbox("Select Performance Metric", metric_options, 
                                  index=metric_options.index('distance') if 'distance' in metric_options else 0, 
                                  key="boxplot_metric")
    
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
        }
    )
    
    # Update layout
    fig.update_layout(
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# 9. Outlier Detection
# 9. Outlier Detection
def render_outlier_detection(df):
    # Select metric for outlier detection
    metric_options = [col for col in df.columns if col not in ['date', 'id', 'name', 'type', 'notes']]
    
    if not metric_options:
        st.error("No numerical metrics found in the data.")
        return
    
    selected_metric = st.selectbox("Select Metric for Outlier Detection", metric_options, 
                                  index=metric_options.index('distance') if 'distance' in metric_options else 0,
                                  key="outlier_metric")
    
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
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# 10. Clustering Analysis
def render_clustering_analysis(df):
    # Select features for clustering
    feature_options = [col for col in df.columns if col not in ['date', 'id', 'name', 'type', 'notes']]
    
    if len(feature_options) < 2:
        st.error("Not enough numerical features for clustering analysis.")
        return
    
    # Allow user to select columns for clustering
    selected_features = st.multiselect(
        "Select features for clustering:",
        options=feature_options,
        default=feature_options[:min(3, len(feature_options))]
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
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Add cluster labels to the data
    cluster_data['cluster'] = cluster_labels
    
    # Select two features for visualization
    if len(selected_features) >= 2:
        x_feature = st.selectbox
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
            title=f"Clustering of Running Activities ({num_clusters} clusters)"
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
                        color=f'rgba({i*50}, {i*30}, {200-(i*30)}, 1)',
                        line=dict(color='black', width=1)
                    ),
                    name=f'Centroid {i}'
                )
            )
        
        # Update layout
        fig.update_layout(
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Main function
def main():
    st.title("🏃 Running Performance Analytics")
    
    try:
        # First try to fetch data from Supabase
        df = fetch_data()
        
        # Add extensive diagnostics
        with st.expander("Data Diagnostics", expanded=True):
            st.write("Data Shape:", df.shape)
            st.write("Columns:", df.columns.tolist())
            
            # Check for date column specifically
            if 'date' in df.columns:
                st.write("Date column exists")
                st.write("Date type:", df['date'].dtype)
                st.write("First few dates:", df['date'].head())
            elif 'start_date' in df.columns:
                st.write("start_date column exists (should be renamed to date)")
                st.write("start_date type:", df['start_date'].dtype)
            else:
                st.write("⚠️ No date column found!")
            
            # Preview the data
         
        
        # Rest of your code
        # ...
                
                # Check for problematic values
                for col in df.columns:
                    if df[col].dtype != 'datetime64[ns]' and df[col].dtype != 'object':
                        try:
                            min_val = df[col].min()
                            max_val = df[col].max()
                            st.write(f"{col}: min={min_val}, max={max_val}")
                        except Exception as e:
                            st.write(f"Could not compute stats for {col}: {e}")
        
        # Rest of your code remains the same...
        
        # Add sidebar for filtering
        st.sidebar.header("Filter Data")
        
        # Date range filter
        if 'date' in df.columns:
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
        else:
            df_filtered = df
            st.sidebar.warning("Date column not found. Cannot filter by date.")
        
        # Display data summary
        st.sidebar.markdown("### Data Summary")
        st.sidebar.markdown(f"**Total Activities:** {len(df_filtered)}")
        
        if 'date' in df_filtered.columns:
            st.sidebar.markdown(f"**Date Range:** {df_filtered['date'].min().date()} to {df_filtered['date'].max().date()}")
        
        if 'distance' in df_filtered.columns:
            total_distance = df_filtered['distance'].sum()
            st.sidebar.markdown(f"**Total Distance:** {total_distance:.2f} km")
        
        # Create tabs for different visualization categories
        tab1, tab2 = st.tabs(["Training Tracking", "Performance Analysis"])
        
        with tab1:
            st.header("1. Running Consistency Calendar Heatmap")
            render_consistency_heatmap(df_filtered)
            
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.header("2. Performance Trend Prediction")
                render_performance_regression(df_filtered)
                
                st.header("3. Performance by Training Type")
                render_boxplot_by_training_type(df_filtered)
                
            with col2:
                st.header("4. Correlation Matrix")
                render_correlation_matrix(df_filtered)
                
                st.header("5. Outlier Detection")
                render_outlier_detection(df_filtered)
            
            st.header("6. Clustering Analysis")
            render_clustering_analysis(df_filtered)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("If you're seeing this error, please check:")
        st.markdown("""
        1. Your Supabase connection settings in the `.streamlit/secrets.toml` file
        2. The structure of your activities table
        3. That all required packages are installed
        """)
        
        # Show sample data for demonstration even on error
        st.subheader("Demo Data Preview")
            # st.dataframe(df.head())
        st.dataframe(get_sample_data().head())

if __name__ == "__main__":
    main()
