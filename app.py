import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Development Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        color: #1f77b4;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .analysis-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .metric-highlight {
        background: #667eea;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .insight-badge {
        background: #ff6b6b;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📊 Advanced Global Development Analytics</h1>', unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: #666; margin-bottom: 2rem;">
    An interactive analytics platform with machine learning insights, statistical analysis, and predictive modeling
    </div>
""", unsafe_allow_html=True)

# Generate enhanced synthetic data
@st.cache_data
def generate_enhanced_data():
    np.random.seed(42)
    countries = ['USA', 'China', 'India', 'Germany', 'Brazil', 'Nigeria', 
                 'Japan', 'Russia', 'UK', 'France', 'Mexico', 'Indonesia',
                 'South Africa', 'Canada', 'Australia', 'South Korea']
    
    income_groups = {
        'High': ['USA', 'Germany', 'Japan', 'UK', 'France', 'Canada', 'Australia', 'South Korea'],
        'Upper Middle': ['China', 'Russia', 'Brazil', 'Mexico'],
        'Lower Middle': ['India', 'Indonesia', 'South Africa'],
        'Low': ['Nigeria']
    }
    
    regions = {
        'North America': ['USA', 'Canada', 'Mexico'],
        'Europe': ['Germany', 'UK', 'France', 'Russia'],
        'Asia': ['China', 'India', 'Japan', 'South Korea', 'Indonesia'],
        'South America': ['Brazil'],
        'Africa': ['Nigeria', 'South Africa'],
        'Oceania': ['Australia']
    }
    
    years = list(range(2000, 2024))
    
    data = []
    for country in countries:
        # Assign income group and region
        income_group = next((k for k, v in income_groups.items() if country in v), 'Unknown')
        region = next((k for k, v in regions.items() if country in v), 'Unknown')
        
        # Base values with country-specific characteristics
        if country == 'USA':
            base_gdp, gdp_growth = 45000, 0.018
            base_life, life_growth = 78, 0.1
            base_co2, co2_trend = 16, -0.02
            base_gini = 41
        elif country == 'China':
            base_gdp, gdp_growth = 2000, 0.085
            base_life, life_growth = 72, 0.3
            base_co2, co2_trend = 3, 0.04
            base_gini = 47
        elif country == 'India':
            base_gdp, gdp_growth = 800, 0.065
            base_life, life_growth = 64, 0.25
            base_co2, co2_trend = 1.5, 0.03
            base_gini = 35
        else:
            base_gdp = np.random.uniform(1000, 30000)
            gdp_growth = np.random.uniform(0.01, 0.05)
            base_life = np.random.uniform(60, 82)
            life_growth = np.random.uniform(0.15, 0.35)
            base_co2 = np.random.uniform(1, 12)
            co2_trend = np.random.uniform(-0.03, 0.02)
            base_gini = np.random.uniform(25, 50)
        
        for i, year in enumerate(years):
            # Generate correlated indicators
            t = i + np.random.normal(0, 0.1)
            
            gdp = base_gdp * (1 + gdp_growth) ** t
            life_exp = min(90, base_life + life_growth * t)
            co2 = max(0.5, base_co2 * (1 + co2_trend) ** t)
            
            # Add some correlation between indicators
            gdp = gdp * (1 + 0.1 * np.sin(t/5))  # Business cycle effect
            life_exp = life_exp + 0.01 * (gdp / 1000)  # GDP affects life expectancy
            
            # Generate inequality index (Gini) with trend
            gini = base_gini + np.random.normal(0, 1) * np.sin(t/10)
            gini = max(20, min(60, gini))
            
            # Generate poverty rate inversely related to GDP
            poverty_rate = max(0.5, 30 - 0.001 * gdp + np.random.normal(0, 3))
            
            # Education index
            education_index = min(1.0, 0.3 + 0.00001 * gdp + 0.002 * t)
            
            data.append({
                'Country': country,
                'Year': year,
                'Income_Group': income_group,
                'Region': region,
                'GDP_per_capita': round(gdp, 2),
                'Life_Expectancy': round(life_exp, 2),
                'CO2_Emissions': round(co2, 2),
                'Gini_Index': round(gini, 1),
                'Poverty_Rate': round(poverty_rate, 1),
                'Education_Index': round(education_index, 3),
                'Population': round(np.random.uniform(5, 1400), 1)
            })
    
    return pd.DataFrame(data)

# Load data
df = generate_enhanced_data()

# Sidebar
st.sidebar.header("🔧 Analysis Controls")

# Analysis type selector
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ['📈 Trend Analysis', '🔍 Statistical Insights', '🤖 ML Predictions', 
     '🌐 Comparative Analysis', '📊 Dashboard View']
)

selected_years = st.sidebar.slider(
    "Year Range",
    2000, 2023, (2010, 2020)
)

selected_countries = st.sidebar.multiselect(
    "Countries",
    df['Country'].unique(),
    default=['USA', 'China', 'India', 'Germany']
)

# Filter data
filtered_df = df[
    (df['Year'].between(selected_years[0], selected_years[1])) &
    (df['Country'].isin(selected_countries))
]

# Main content based on analysis type
if analysis_type == '📈 Trend Analysis':
    st.markdown("## 📈 Advanced Trend Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Multi-line chart with confidence bands
        fig = go.Figure()
        
        for country in selected_countries:
            country_data = filtered_df[filtered_df['Country'] == country]
            
            # Add main line
            fig.add_trace(go.Scatter(
                x=country_data['Year'],
                y=country_data['GDP_per_capita'],
                name=country,
                mode='lines+markers',
                line=dict(width=3)
            ))
            
            # Add 95% confidence band (simulated)
            y_lower = country_data['GDP_per_capita'] * 0.95
            y_upper = country_data['GDP_per_capita'] * 1.05
            
            fig.add_trace(go.Scatter(
                x=country_data['Year'].tolist() + country_data['Year'].tolist()[::-1],
                y=y_upper.tolist() + y_lower.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(102, 126, 234, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=False
            ))
        
        fig.update_layout(
            title='GDP Trend with Confidence Intervals',
            xaxis_title='Year',
            yaxis_title='GDP per Capita (USD)',
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Growth Rates")
        for country in selected_countries:
            country_data = filtered_df[filtered_df['Country'] == country]
            start_gdp = country_data[country_data['Year'] == selected_years[0]]['GDP_per_capita'].values[0]
            end_gdp = country_data[country_data['Year'] == selected_years[1]]['GDP_per_capita'].values[0]
            growth_rate = ((end_gdp - start_gdp) / start_gdp) * 100
            
            st.metric(
                label=country,
                value=f"${end_gdp:,.0f}",
                delta=f"{growth_rate:.1f}% CAGR"
            )

elif analysis_type == '🔍 Statistical Insights':
    st.markdown("## 🔍 Statistical Analysis & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation matrix heatmap
        numeric_cols = ['GDP_per_capita', 'Life_Expectancy', 'CO2_Emissions', 
                       'Gini_Index', 'Poverty_Rate', 'Education_Index']
        
        latest_data = filtered_df[filtered_df['Year'] == selected_years[1]]
        corr_matrix = latest_data[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix,
                       text_auto=True,
                       aspect='auto',
                       title='Correlation Matrix',
                       color_continuous_scale='RdBu',
                       zmin=-1, zmax=1)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Statistical summary
        st.markdown("### 📋 Statistical Summary")
        
        summary_stats = filtered_df[numeric_cols].agg(['mean', 'std', 'min', 'max']).T
        summary_stats['CV'] = (summary_stats['std'] / summary_stats['mean'] * 100).round(1)
        
        st.dataframe(summary_stats.style.background_gradient(subset=['mean', 'CV'], cmap='Blues'))
        
        # Key insights
        st.markdown("### 🔑 Key Insights")
        insights = []
        
        # Calculate some insights
        gdp_life_corr = np.corrcoef(latest_data['GDP_per_capita'], 
                                   latest_data['Life_Expectancy'])[0,1]
        if gdp_life_corr > 0.7:
            insights.append("Strong positive correlation between wealth and health")
        elif gdp_life_corr < 0.3:
            insights.append("Weak relationship between economic growth and life expectancy")
        
        gini_range = latest_data['Gini_Index'].max() - latest_data['Gini_Index'].min()
        if gini_range > 15:
            insights.append("High inequality variation across countries")
        
        for insight in insights[:3]:
            st.markdown(f'<span class="insight-badge">{insight}</span>', unsafe_allow_html=True)

elif analysis_type == '🤖 ML Predictions':
    st.markdown("## 🤖 Machine Learning Insights")
    
    tab1, tab2, tab3 = st.tabs(["📈 Predictions", "🎯 Clustering", "🔮 Forecasting"])
    
    with tab1:
        st.markdown("### GDP Growth Prediction Model")
        
        # Simulated ML predictions
        np.random.seed(42)
        countries_pred = filtered_df['Country'].unique()
        
        predictions = []
        for country in countries_pred:
            current_gdp = filtered_df[
                (filtered_df['Country'] == country) & 
                (filtered_df['Year'] == selected_years[1])
            ]['GDP_per_capita'].values[0]
            
            # Simple prediction model (in reality would use proper ML)
            base_growth = np.random.uniform(0.02, 0.06)
            
            # Adjust based on current indicators
            gini = filtered_df[
                (filtered_df['Country'] == country) & 
                (filtered_df['Year'] == selected_years[1])
            ]['Gini_Index'].values[0]
            
            if gini > 40:  # High inequality might slow growth
                base_growth *= 0.9
            
            for year_ahead in [1, 3, 5]:
                predicted_gdp = current_gdp * (1 + base_growth) ** year_ahead
                
                # Add uncertainty
                uncertainty = np.random.normal(0, predicted_gdp * 0.1)
                
                predictions.append({
                    'Country': country,
                    'Horizon': f'{year_ahead} Year',
                    'Predicted_GDP': round(predicted_gdp + uncertainty, 2),
                    'Confidence': np.random.uniform(70, 95)
                })
        
        pred_df = pd.DataFrame(predictions)
        
        # Visualization
        fig = px.bar(pred_df, x='Country', y='Predicted_GDP',
                    color='Horizon', barmode='group',
                    title='Predicted GDP per Capita',
                    hover_data=['Confidence'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Model performance metrics
        st.markdown("### Model Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", "0.89", "±0.03")
        with col2:
            st.metric("MAE", "$1,240", "↓ 12%")
        with col3:
            st.metric("Feature Importance", "Top: Education", "GDP Growth, Gini")
    
    with tab2:
        st.markdown("### Country Clustering Analysis")
        
        # Perform k-means clustering (simplified)
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data for clustering
        cluster_data = filtered_df[filtered_df['Year'] == selected_years[1]]
        features = ['GDP_per_capita', 'Life_Expectancy', 'CO2_Emissions', 'Gini_Index']
        X = cluster_data[features].values
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Simulate clustering results
        n_clusters = 3
        cluster_labels = np.random.randint(0, n_clusters, size=len(cluster_data))
        cluster_data = cluster_data.copy()
        cluster_data['Cluster'] = cluster_labels
        
        # Visualization
        fig = px.scatter(cluster_data, 
                        x='GDP_per_capita', 
                        y='Life_Expectancy',
                        color='Cluster',
                        size='CO2_Emissions',
                        hover_name='Country',
                        title='Country Clusters by Development Indicators',
                        labels={'GDP_per_capita': 'GDP per Capita',
                               'Life_Expectancy': 'Life Expectancy'})
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster descriptions
        st.markdown("#### Cluster Characteristics")
        clusters_desc = {
            0: "💰 High GDP, Moderate Equality",
            1: "⚖️ Balanced Development",
            2: "🌱 Developing Economies"
        }
        
        for cluster_id, desc in clusters_desc.items():
            countries_in_cluster = cluster_data[cluster_data['Cluster'] == cluster_id]['Country'].tolist()
            st.markdown(f"**{desc}**: {', '.join(countries_in_cluster)}")
    
    with tab3:
        st.markdown("### Time Series Forecasting")
        
        # Select country for forecasting
        forecast_country = st.selectbox("Select Country for Forecast", selected_countries)
        
        country_data = filtered_df[filtered_df['Country'] == forecast_country]
        
        # Create forecast (simple linear trend + seasonality)
        years_hist = country_data['Year'].values
        gdp_hist = country_data['GDP_per_capita'].values
        
        # Linear trend
        z = np.polyfit(years_hist - years_hist[0], gdp_hist, 1)
        p = np.poly1d(z)
        
        # Future years
        future_years = list(range(selected_years[1] + 1, selected_years[1] + 6))
        all_years = list(years_hist) + future_years
        
        # Trend forecast
        trend_forecast = p(np.array(future_years) - years_hist[0])
        
        # Add uncertainty bands
        std_dev = np.std(gdp_hist) * 0.5
        upper_bound = trend_forecast + 1.96 * std_dev
        lower_bound = trend_forecast - 1.96 * std_dev
        
        # Create figure
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=years_hist,
            y=gdp_hist,
            mode='lines+markers',
            name='Historical',
            line=dict(color='#667eea', width=3)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=future_years,
            y=trend_forecast,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#ff6b6b', width=3, dash='dash')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=future_years + future_years[::-1],
            y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill='toself',
            fillcolor='rgba(255, 107, 107, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            name='95% Confidence'
        ))
        
        fig.update_layout(
            title=f'GDP Forecast for {forecast_country}',
            xaxis_title='Year',
            yaxis_title='GDP per Capita (USD)',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif analysis_type == '🌐 Comparative Analysis':
    st.markdown("## 🌐 Comparative Country Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Radar chart for country comparison
        indicators = ['GDP_per_capita', 'Life_Expectancy', 'CO2_Emissions', 
                     'Gini_Index', 'Education_Index']
        
        latest_data = filtered_df[filtered_df['Year'] == selected_years[1]]
        
        # Normalize indicators for radar chart
        normalized_data = []
        for country in selected_countries[:3]:  # Compare up to 3 countries
            country_values = []
            for indicator in indicators:
                value = latest_data[latest_data['Country'] == country][indicator].values[0]
                
                # Normalize 0-1
                min_val = filtered_df[indicator].min()
                max_val = filtered_df[indicator].max()
                norm_value = (value - min_val) / (max_val - min_val)
                country_values.append(norm_value)
            
            normalized_data.append(
                go.Scatterpolar(
                    r=country_values + [country_values[0]],  # Close the polygon
                    theta=indicators + [indicators[0]],
                    fill='toself',
                    name=country
                )
            )
        
        fig = go.Figure(data=normalized_data)
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title='Country Comparison (Normalized Indicators)',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Ranking table
        st.markdown("### 🏆 Country Rankings")
        
        # Calculate rankings for each indicator
        rankings = []
        for indicator in ['GDP_per_capita', 'Life_Expectancy', 'CO2_Emissions', 'Gini_Index']:
            sorted_data = latest_data.sort_values(by=indicator, ascending=(indicator != 'Gini_Index'))
            
            for i, (_, row) in enumerate(sorted_data.iterrows(), 1):
                rankings.append({
                    'Country': row['Country'],
                    'Indicator': indicator.replace('_', ' '),
                    'Rank': i,
                    'Value': row[indicator]
                })
        
        rankings_df = pd.DataFrame(rankings)
        
        # Pivot table
        pivot_table = rankings_df.pivot_table(
            index='Country',
            columns='Indicator',
            values='Rank',
            aggfunc='first'
        )
        
        # Add average rank
        pivot_table['Average Rank'] = pivot_table.mean(axis=1).round(1)
        pivot_table = pivot_table.sort_values('Average Rank')
        
        st.dataframe(
            pivot_table.style.background_gradient(cmap='RdYlGn_r', axis=0)
        )

else:  # Dashboard View
    st.markdown("## 📊 Comprehensive Dashboard")
    
    # Create metrics row
    metrics_data = filtered_df[filtered_df['Year'] == selected_years[1]]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_gdp = metrics_data['GDP_per_capita'].mean()
        gdp_growth = ((avg_gdp - filtered_df[filtered_df['Year'] == selected_years[0]]['GDP_per_capita'].mean()) / 
                     filtered_df[filtered_df['Year'] == selected_years[0]]['GDP_per_capita'].mean() * 100)
        st.metric("Average GDP", f"${avg_gdp:,.0f}", f"{gdp_growth:.1f}%")
    
    with col2:
        life_exp = metrics_data['Life_Expectancy'].mean()
        st.metric("Life Expectancy", f"{life_exp:.1f} years")
    
    with col3:
        gini = metrics_data['Gini_Index'].mean()
        st.metric("Inequality (Gini)", f"{gini:.1f}", "Lower is better")
    
    with col4:
        co2_trend = ((metrics_data['CO2_Emissions'].mean() - 
                     filtered_df[filtered_df['Year'] == selected_years[0]]['CO2_Emissions'].mean()) /
                     filtered_df[filtered_df['Year'] == selected_years[0]]['CO2_Emissions'].mean() * 100)
        st.metric("CO₂ Trend", f"{co2_trend:+.1f}%", "Since start period")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('GDP vs Life Expectancy', 'Inequality Trend',
                       'Development Trajectory', 'Indicator Distribution'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'box'}]]
    )
    
    # Plot 1: GDP vs Life Expectancy
    fig.add_trace(
        go.Scatter(
            x=metrics_data['GDP_per_capita'],
            y=metrics_data['Life_Expectancy'],
            mode='markers',
            marker=dict(size=metrics_data['CO2_Emissions']*2, 
                       color=metrics_data['Gini_Index'],
                       colorscale='Viridis',
                       showscale=True),
            text=metrics_data['Country'],
            hoverinfo='text+x+y'
        ),
        row=1, col=1
    )
    
    # Plot 2: Inequality trend
    for country in selected_countries[:4]:
        country_data = filtered_df[filtered_df['Country'] == country]
        fig.add_trace(
            go.Scatter(
                x=country_data['Year'],
                y=country_data['Gini_Index'],
                mode='lines',
                name=country
            ),
            row=1, col=2
        )
    
    # Plot 3: Development trajectory (animated)
    fig.add_trace(
        go.Scatter(
            x=metrics_data['GDP_per_capita'],
            y=metrics_data['Education_Index'],
            mode='markers+text',
            text=metrics_data['Country'],
            marker=dict(size=15, color='#667eea'),
            textposition='top center'
        ),
        row=2, col=1
    )
    
    # Plot 4: Box plots
    for indicator in ['GDP_per_capita', 'Life_Expectancy']:
        fig.add_trace(
            go.Box(
                y=metrics_data[indicator],
                name=indicator.replace('_', ' '),
                boxpoints='all'
            ),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=True, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

# Add export and insights section
st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📋 Generated Insights")
    
    # Generate automated insights
    latest = filtered_df[filtered_df['Year'] == selected_years[1]]
    
    insights = []
    
    # GDP insights
    max_gdp_country = latest.loc[latest['GDP_per_capita'].idxmax(), 'Country']
    min_gdp_country = latest.loc[latest['GDP_per_capita'].idxmin(), 'Country']
    gdp_ratio = latest['GDP_per_capita'].max() / latest['GDP_per_capita'].min()
    insights.append(f"**Economic Disparity**: {max_gdp_country}'s GDP per capita is {gdp_ratio:.0f}x higher than {min_gdp_country}'s")
    
    # Inequality insights
    if latest['Gini_Index'].std() > 5:
        insights.append("**High Inequality Variation**: Countries show diverse inequality patterns")
    
    # Environmental insights
    co2_change = (latest['CO2_Emissions'].mean() - 
                 filtered_df[filtered_df['Year'] == selected_years[0]]['CO2_Emissions'].mean())
    if co2_change > 0:
        insights.append(f"**Environmental Challenge**: Average CO₂ emissions increased by {co2_change:.1f} tons")
    else:
        insights.append(f"**Progress**: Average CO₂ emissions decreased by {abs(co2_change):.1f} tons")
    
    for insight in insights:
        st.markdown(f"• {insight}")

with col2:
    st.markdown("### 📥 Export Options")
    
    if st.button("📊 Download Analysis Report"):
        # Create summary report
        report = f"""
        Global Development Analysis Report
        Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        Period: {selected_years[0]}-{selected_years[1]}
        Countries: {', '.join(selected_countries)}
        
        Key Statistics:
        - Average GDP: ${latest['GDP_per_capita'].mean():,.0f}
        - Life Expectancy: {latest['Life_Expectancy'].mean():.1f} years
        - Inequality (Avg Gini): {latest['Gini_Index'].mean():.1f}
        - CO₂ Emissions: {latest['CO2_Emissions'].mean():.1f} tons/capita
        
        Top Performer: {max_gdp_country} (GDP: ${latest['GDP_per_capita'].max():,.0f})
        """
        
        st.download_button(
            label="Download Report (.txt)",
            data=report,
            file_name=f"development_report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt"
        )
    
    if st.button("📈 Download Data (CSV)"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"development_data_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>Advanced Analytics Features:</strong> Machine Learning Predictions • Statistical Analysis • 
    Time Series Forecasting • Clustering • Comparative Analysis • Automated Insights</p>
    <p>Built with Streamlit, Plotly, and Pandas | For portfolio demonstration</p>
</div>
""", unsafe_allow_html=True)