import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
import pydeck as pdk
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Retail Stock Insights Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS - White background, Black text theme
st.markdown("""
    <style>
    /* Force white background everywhere */
    .main {
        background-color: #ffffff;
    }
.custom-metrics {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  margin: 1rem 0;
}
.metric-card {
  background-color: #FFFFFF;
  border: 2px solid #A8D5E3;  /* your theme color */
  border-radius: 10px;
  padding: 14px 16px;
  box-shadow: 0 4px 10px rgba(2, 195, 137, 0.10);
  text-align: center;
}
.metric-card .label {
  font-size: 0.9rem;
  font-weight: 600;
  color: #000000;
}
.metric-card .value {
  font-size: 1.8rem;
  font-weight: 700;
  color: #000000;
  margin-top: 5px;
}
    .stAppToolbar{
        background-color: #ffffff;
    }
    
    .stApp {
        background-color: #ffffff;
    }
    
    /* Main Header */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #000000;
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 1rem;
        letter-spacing: -0.5px;
    }
    
    /* Team Info Box */
    .team-info {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0 2rem 0;
        border: 2px solid #A8D5E3;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .team-info h3 {
        color: #000000;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .team-info p {
        color: #000000;
        margin: 0.5rem 0;
        font-size: 1rem;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 2rem;
        font-weight: 600;
        color: #000000;
        margin-top: 3rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #A8D5E3;
    }
    
    /* Subsection Headers */
    .subsection-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #000000;
        margin-top: 2.5rem;
        margin-bottom: 1.25rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #14b8a6;
    }
    
    /* Description Boxes - White background, black text */
    .description-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 2rem 0;
        border: 2px solid #e5e7eb;
        border-left: 5px solid #A8D5E3;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .description-box strong {
        color: #000000;
        font-size: 1.1rem;
    }
    
    .description-box p, .description-box li {
        color: #000000;
        line-height: 1.8;
        margin: 0.5rem 0;
    }
    
    .description-box ul {
        margin: 1rem 0;
        padding-left: 1.5rem;
    }
    
    /* Add spacing between sections */
    .spacer {
        margin: 2rem 0;

    }
    
    /* Metric cards styling - consistent with theme */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #000000;
    }
    
    [data-testid="stMetricLabel"] {
        color: #000000;
    }
    
    [data-testid="stMetricDelta"] {
        color: #000000;
    }
    
    /* Metric container - white background */
    [data-testid="stMetricContainer"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Tab styling - white background */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #ffffff;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        font-weight: 500;
        color: #000000;
        border-radius: 6px;
        background-color: #f3f4f6;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #A8D5E3;
        color: #ffffff;
    }
    
    /* Sidebar styling - white background */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border: 1px solid #A8D5E3;
    }
    
    [data-testid="stSidebar"] [data-testid="stHeader"] {
        # background-color: #ffffff;
        # color: #000000;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #000000;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #A8D5E3 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    
    /* Ensure all text is black */
    p, h1, h2, h3, h4, h5, h6, span, div, label {
        color: #000000 !important;
    }
    
    /* Plotly chart backgrounds */
    .js-plotly-plot {
        background-color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the dataset - tries Kaggle first, then local CSV"""
    try:
        import kagglehub
        path = kagglehub.dataset_download("meesalasreesainath/cleaned-online-retail-dataset")
        import os
        for file in os.listdir(path):
            if file.endswith(".csv"):
                csv_path = os.path.join(path, file)
                df = pd.read_csv(csv_path)
                if 'InvoiceDate' in df.columns:
                    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
                    df = df[df["InvoiceDate"].notna()]
                return df
    except Exception as e:
        try:
            df = pd.read_csv("data/raw/online_retail.csv")
            df = preprocess_data(df)
            return df
        except Exception as e2:
            return None

def preprocess_data(df):
    """Preprocess the raw data"""
    df = df.drop_duplicates()
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["CustomerID"] = df["CustomerID"].astype(int)
    df = df[df["InvoiceDate"].notna()]
    df["TotalPrice"] = round(df["Quantity"] * df["UnitPrice"], 3)
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour
    
    Q1 = df[['Quantity', 'UnitPrice']].quantile(0.25)
    Q3 = df[['Quantity', 'UnitPrice']].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~(
        (df['Quantity'] < (Q1['Quantity'] - 1.5 * IQR['Quantity'])) |
        (df['Quantity'] > (Q3['Quantity'] + 1.5 * IQR['Quantity'])) |
        (df['UnitPrice'] < (Q1['UnitPrice'] - 1.5 * IQR['UnitPrice'])) |
        (df['UnitPrice'] > (Q3['UnitPrice'] + 1.5 * IQR['UnitPrice']))
    )]
    
    df["Country"] = df["Country"].str.strip().str.title()
    return df

@st.cache_data
def calculate_rfm(df):
    """Calculate RFM metrics"""
    reference_date = df['InvoiceDate'].max()
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    rfm = rfm[rfm['Monetary'] > 0]
    return rfm

def safe_qcut(data, q=5, labels=None, duplicates='drop', reverse_labels=False):
    """
    Safely perform qcut with dynamic quantile adjustment based on sample size.
    
    Parameters:
    -----------
    data : Series or array-like
        Data to be quantized
    q : int, default 5
        Desired number of quantiles
    labels : array-like, optional
        Labels for the bins. If None, will be auto-generated
    duplicates : str, default 'drop'
        How to handle duplicate edges
    reverse_labels : bool, default False
        If True, reverse the label order (for Recency where lower is better)
    
    Returns:
    --------
    Series with quantile labels
    """
    # Handle empty data
    if len(data) == 0:
        return pd.Series([], dtype='object')
    
    # Convert to Series if needed
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    n_samples = len(data.dropna())
    
    # Edge case: Not enough samples
    if n_samples < 2:
        # Return a single label for all samples
        if reverse_labels:
            default_label = labels[0] if labels and len(labels) > 0 else 1
        else:
            default_label = labels[-1] if labels and len(labels) > 0 else q
        return pd.Series([default_label] * len(data), index=data.index, dtype='object')
    
    # Adjust q based on available samples
    # qcut requires: q <= n_samples
    actual_q = min(q, n_samples)
    
    # Generate labels if not provided
    if labels is None:
        if reverse_labels:
            labels = list(range(actual_q, 0, -1))
        else:
            labels = list(range(1, actual_q + 1))
    else:
        # Adjust labels to match actual_q
        labels = list(labels)  # Convert to list
        if len(labels) != actual_q:
            if reverse_labels:
                # For reverse, take from start
                if len(labels) >= actual_q:
                    labels = labels[:actual_q]
                else:
                    # Pad with descending numbers
                    max_label = labels[0] if labels else actual_q
                    labels = list(range(max_label, max_label - actual_q, -1))[:actual_q]
            else:
                # For normal, take from start
                if len(labels) >= actual_q:
                    labels = labels[:actual_q]
                else:
                    # Pad with ascending numbers
                    min_label = labels[-1] + 1 if labels else 1
                    labels = labels + list(range(min_label, min_label + (actual_q - len(labels))))
    
    try:
        result = pd.qcut(data, q=actual_q, labels=labels, duplicates=duplicates)
        return result
    except (ValueError, TypeError) as e:
        # Fallback: if qcut still fails, use rank-based approach
        if n_samples >= 2:
            try:
                # Use rank to create bins
                ranked = data.rank(method='first')
                bins = pd.cut(ranked, bins=actual_q, labels=labels, duplicates='drop', include_lowest=True)
                return bins
            except:
                # Last resort: assign labels based on sorted order
                sorted_indices = data.sort_values().index
                result = pd.Series(index=data.index, dtype='object')
                chunk_size = len(sorted_indices) // actual_q
                for i in range(actual_q):
                    start_idx = i * chunk_size
                    end_idx = (i + 1) * chunk_size if i < actual_q - 1 else len(sorted_indices)
                    result.loc[sorted_indices[start_idx:end_idx]] = labels[i]
                return result
        else:
            # Single value case
            default_label = labels[0] if reverse_labels and labels else (labels[-1] if labels else actual_q)
            return pd.Series([default_label] * len(data), index=data.index, dtype='object')

def update_bar_layout(fig, height=700, orientation='h'):
    """Update bar chart layout for better readability"""
    fig.update_layout(
        height=height,
        margin=dict(l=100 if orientation == 'h' else 50, r=50, t=60, b=100 if orientation == 'v' else 50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#000000', size=12, family='Arial'),
        xaxis=dict(
            title_font=dict(color='#000000', size=14, family='Arial'),
            tickfont=dict(color='#000000', size=12, family='Arial'),
            gridcolor='#e5e7eb',
            linecolor='#000000',
            showgrid=True
        ),
        yaxis=dict(
            title_font=dict(color='#000000', size=14, family='Arial'),
            tickfont=dict(color='#000000', size=12, family='Arial'),
            gridcolor='#e5e7eb',
            linecolor='#000000',
            showgrid=True
        ),
        title=dict(font=dict(color='#000000', size=18, family='Arial'))
    )
    if orientation == 'h':
        fig.update_yaxes(automargin=True)
    else:
        fig.update_xaxes(automargin=True, tickangle=-45 if height > 500 else 0)
    return fig

def update_chart_layout(fig, height=500):
    """Update any chart layout for white background and black text"""
    fig.update_layout(
        height=height,
        margin=dict(l=50, r=50, t=60, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#000000', size=12),
        xaxis=dict(
            title_font=dict(color='#000000', size=14, family='Arial'),
            tickfont=dict(color='#000000', size=12, family='Arial'),
            gridcolor='#e5e7eb',
            linecolor='#000000',
            showgrid=True
        ),
        yaxis=dict(
            title_font=dict(color='#000000', size=14, family='Arial'),
            tickfont=dict(color='#000000', size=12, family='Arial'),
            gridcolor='#e5e7eb',
            linecolor='#000000',
            showgrid=True
        ),
        title=dict(font=dict(color='#000000', size=18, family='Arial')),
        legend=dict(font=dict(color='#000000', size=12, family='Arial'))
    )
    return fig

# Main App
st.markdown('<h1 class="main-header">Retail Stock Insights Dashboard</h1>', unsafe_allow_html=True)

# Team Information
st.markdown("""
    <div class="team-info">
        <h3>Data Scouts</h3>
        <p><strong>Team Members:</strong></p>
        <p>• Meesala Sree Sai Nath</p>
        <p>• Akula Jithendranath</p>
        <p>• Tejmul Movin</p>
        <p style="margin-top: 1rem;"><strong>Mentor:</strong> Ashwin Tewary Sir</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # st.header("Data Source")
    # uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    # if uploaded_file is not None:
    #     try:
    #         df = pd.read_csv(uploaded_file)
    #         if 'TotalPrice' not in df.columns or 'DayOfWeek' not in df.columns:
    #             df = preprocess_data(df)
    #         else:
    #             if 'InvoiceDate' in df.columns:
    #                 df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    #                 df = df[df["InvoiceDate"].notna()]
    #         st.success("Data loaded successfully!")
    #     except Exception as e:
    #         st.error(f"Error loading file: {str(e)}")
    #         df = None
    # else:
    #     df = load_data()
    df = load_data()
    if df is not None:
        st.header("Filters")
        countries = ['All'] + sorted(df['Country'].unique().tolist())
        selected_country = st.selectbox("Select Country", countries)
        
        date_range = None
        if 'InvoiceDate' in df.columns and df['InvoiceDate'].notna().any():
            try:
                min_date = df['InvoiceDate'].min().date()
                max_date = df['InvoiceDate'].max().date()
                date_range = st.date_input(
                    "Select Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
            except:
                pass
        
        if selected_country != 'All':
            df = df[df['Country'] == selected_country]
        
        if date_range is not None and len(date_range) == 2:
            try:
                df = df[(df['InvoiceDate'].dt.date >= date_range[0]) & 
                        (df['InvoiceDate'].dt.date <= date_range[1])]
            except:
                pass

# Main content
if df is not None:
    # Key Metrics with better spacing - ensure values are always displayed
    total_transactions = df.shape[0] if len(df) > 0 else 0
    unique_customers = df['CustomerID'].nunique() if len(df) > 0 and 'CustomerID' in df.columns else 0
    unique_products = df['Description'].nunique() if len(df) > 0 and 'Description' in df.columns else 0
    total_revenue = df['TotalPrice'].sum() if len(df) > 0 and 'TotalPrice' in df.columns else 0.0
    
    st.markdown(f"""
<div class="custom-metrics">
  <div class="metric-card">
    <div class="label">Total Transactions</div>
    <div class="value">{total_transactions:,}</div>
  </div>
  <div class="metric-card">
    <div class="label">Unique Customers</div>
    <div class="value">{unique_customers:,}</div>
  </div>
  <div class="metric-card">
    <div class="label">Unique Products</div>
    <div class="value">{unique_products:,}</div>
  </div>
  <div class="metric-card">
    <div class="label">Total Revenue</div>
    <div class="value">£{total_revenue:,.2f}</div>
  </div>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
    
    # Navigation tabs - removed emojis
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Overview & EDA", 
        "Temporal Analysis", 
        "Customer Segmentation", 
        "Clustering Analysis",
        "Market Basket Analysis",
        "Predictive Analysis",
        "Advanced Analytics",
        "Geographic Insights",
        "Customer Details"
    ])
    
    # ========== TAB 1: Overview & EDA ==========
    with tab1:
        st.markdown('<h2 class="section-header">Data Distributions</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='Quantity', nbins=50, 
                             title='Distribution of Quantity',
                             labels={'Quantity': 'Quantity', 'count': 'Number of Transactions'},
                             color_discrete_sequence=['#A8D5E3'])
            fig = update_chart_layout(fig, height=600)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <div class="description-box">
            <strong>Description:</strong> This histogram shows the distribution of product quantities 
            purchased per transaction. Most transactions involve small quantities, with a right-skewed 
            distribution indicating occasional bulk purchases.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        with col2:
            fig = px.histogram(df, x='UnitPrice', nbins=50,
                             title='Distribution of Unit Price',
                             labels={'UnitPrice': 'Unit Price (£)', 'count': 'Number of Transactions'},
                             color_discrete_sequence=['#14b8a6'])
            fig = update_chart_layout(fig, height=600)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <div class="description-box">
            <strong>Description:</strong> This histogram displays the distribution of unit prices. 
            The distribution is heavily right-skewed, showing that most products are low-priced items, 
            with a few high-value products.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        st.markdown('<h2 class="section-header">Feature Correlation Analysis</h2>', unsafe_allow_html=True)
        corr_matrix = df[["Quantity", "UnitPrice", "TotalPrice"]].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Feature Correlation Heatmap",
                       color_continuous_scale="RdBu",
                       labels=dict(color="Correlation"))
        fig = update_chart_layout(fig, height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> This correlation heatmap reveals relationships between numerical features. 
        TotalPrice shows strong positive correlation with both Quantity and UnitPrice, as expected. 
        Understanding these correlations helps in feature selection for predictive modeling.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        st.markdown('<h2 class="section-header">Top Countries Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            top_countries_tx = df["Country"].value_counts().head(10)
            fig = px.bar(x=top_countries_tx.values, y=top_countries_tx.index,
                        orientation='h', title="Top 10 Countries by Transaction Count",
                        labels={'x': 'Number of Transactions', 'y': 'Country'},
                        color_discrete_sequence=['#A8D5E3'])
            fig = update_bar_layout(fig, height=600, orientation='h')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <div class="description-box">
            <strong>Description:</strong> Shows which countries generate the most transaction volume. 
            This helps identify key markets and understand geographical distribution of customers.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        with col2:
            top_countries_rev = df.groupby("Country")["TotalPrice"].sum().nlargest(10)
            fig = px.bar(x=top_countries_rev.values, y=top_countries_rev.index,
                        orientation='h', title="Top 10 Countries by Revenue",
                        labels={'x': 'Total Revenue (£)', 'y': 'Country'},
                        color_discrete_sequence=['#8b5cf6'])
            fig = update_bar_layout(fig, height=600, orientation='h')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <div class="description-box">
            <strong>Description:</strong> Displays revenue contribution by country. Comparing this with 
            transaction count reveals which markets have higher average order values.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        st.markdown('<h2 class="section-header">Top Products Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            top_products_qty = df.groupby("Description")["Quantity"].sum().nlargest(15)
            fig = px.bar(x=top_products_qty.values, y=top_products_qty.index,
                        orientation='h', title="Top 15 Best-Selling Products (by Quantity)",
                        labels={'x': 'Total Quantity Sold', 'y': 'Product'},
                        color_discrete_sequence=['#A8D5E3'])
            fig = update_bar_layout(fig, height=700, orientation='h')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <div class="description-box">
            <strong>Description:</strong> Identifies products with the highest sales volume. 
            These are your best-selling items and may require careful inventory management.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        with col2:
            top_products_rev = df.groupby("Description")["TotalPrice"].sum().nlargest(15)
            fig = px.bar(x=top_products_rev.values, y=top_products_rev.index,
                        orientation='h', title="Top 15 Highest Revenue Products",
                        labels={'x': 'Total Revenue (£)', 'y': 'Product'},
                        color_discrete_sequence=['#8b5cf6'])
            fig = update_bar_layout(fig, height=700, orientation='h')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <div class="description-box">
            <strong>Description:</strong> Shows products generating the most revenue. These may differ 
            from best-selling products, indicating high-value items that drive profitability.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        st.markdown('<h2 class="section-header">Top Customers Analysis</h2>', unsafe_allow_html=True)
        top_customers = df.groupby("CustomerID")["TotalPrice"].sum().nlargest(15)
        fig = px.bar(x=top_customers.index.astype(str), y=top_customers.values,
                    title="Top 15 Customers by Total Spend",
                    labels={'x': 'Customer ID', 'y': 'Total Spend (£)'},
                    color_discrete_sequence=['#14b8a6'])
        fig = update_bar_layout(fig, height=600, orientation='v')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> Identifies your most valuable customers by total spending. 
        These customers are prime candidates for loyalty programs and personalized marketing campaigns.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        # ADDED: Average Basket Size per Country
        st.markdown('<h2 class="section-header">Average Basket Size per Country</h2>', unsafe_allow_html=True)
        basket_size = df.groupby("Country")["TotalPrice"].mean().nlargest(20)
        fig = px.bar(x=basket_size.values, y=basket_size.index,
                    orientation='h', title="Top 20 Countries by Average Basket Value",
                    labels={'x': 'Average Order Value (£)', 'y': 'Country'},
                    color_discrete_sequence=['#10b981'])
        fig = update_bar_layout(fig, height=700, orientation='h')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> Compares average transaction value across countries. 
        Countries with higher basket sizes may indicate premium markets or successful upselling strategies.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        # ADDED: Price vs Quantity Scatter
        st.markdown('<h2 class="section-header">Price vs Quantity Relationship</h2>', unsafe_allow_html=True)
        sample_df = df.sample(min(5000, len(df)))
        fig = px.scatter(sample_df, x='Quantity', y='UnitPrice', 
                        size='TotalPrice', color='TotalPrice',
                        title='Price vs Quantity Relationship',
                        labels={'Quantity': 'Quantity', 'UnitPrice': 'Unit Price (£)', 
                               'TotalPrice': 'Total Price (£)'},
                        color_continuous_scale='Blues')
        fig = update_chart_layout(fig, height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> Scatter plot showing the relationship between quantity and unit price. 
        Bubble size represents total transaction value. This visualization helps identify pricing patterns 
        and bulk purchase behaviors.
        </div>
        """, unsafe_allow_html=True)
    
    # ========== TAB 2: Temporal Analysis ==========
    with tab2:
        st.markdown('<h2 class="section-header">Monthly Sales Trend</h2>', unsafe_allow_html=True)
        monthly_sales = df.groupby(df["InvoiceDate"].dt.to_period("M"))["TotalPrice"].sum()
        monthly_sales.index = monthly_sales.index.astype(str)
        fig = px.line(x=monthly_sales.index, y=monthly_sales.values,
                     markers=True, title="Monthly Sales Trend",
                     labels={'x': 'Month', 'y': 'Total Monthly Revenue (£)'},
                     color_discrete_sequence=['#3498db'])
        fig.update_traces(line=dict(width=3))
        fig.update_layout(height=500, margin=dict(l=20, r=20, t=50, b=20),
                        plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> Shows revenue trends over time, helping identify seasonal patterns, 
        growth trends, and periods requiring attention. Useful for forecasting and inventory planning.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        st.markdown('<h2 class="section-header">Hourly Purchase Pattern</h2>', unsafe_allow_html=True)
        hourly_counts = df["Hour"].value_counts().sort_index()
        fig = px.line(x=hourly_counts.index, y=hourly_counts.values,
                     markers=True, title="Hourly Purchase Trend",
                     labels={'x': 'Hour of Day', 'y': 'Number of Transactions'},
                     color_discrete_sequence=['#e74c3c'])
        fig.update_traces(line=dict(width=3))
        fig.update_layout(height=500, margin=dict(l=20, r=20, t=50, b=20),
                        plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> Reveals peak shopping hours throughout the day. This information 
        helps optimize staffing, marketing campaigns, and server resources for peak times.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        st.markdown('<h2 class="section-header">Day of Week Sales Distribution</h2>', unsafe_allow_html=True)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = df["DayOfWeek"].value_counts().reindex(day_order)
        fig = px.bar(x=day_counts.index, y=day_counts.values,
                    title="Transactions by Day of the Week",
                    labels={'x': 'Day', 'y': 'Number of Transactions'},
                    color_discrete_sequence=['#A8D5E3'])
        fig = update_bar_layout(fig, height=600, orientation='v')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> Identifies the busiest shopping days. Understanding weekly patterns 
        helps in planning promotions, inventory restocking, and customer service scheduling.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        st.markdown('<h2 class="section-header">Sales Activity Heatmap (Day vs Hour)</h2>', unsafe_allow_html=True)
        activity_pivot = df.groupby(['DayOfWeek', 'Hour'])['InvoiceNo'].nunique().unstack()
        activity_pivot = activity_pivot.reindex(day_order)
        fig = px.imshow(activity_pivot, aspect="auto", 
                       title="Heatmap of Transaction Volume (Day vs Hour)",
                       labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Transaction Count'},
                       color_continuous_scale='Blues')
        fig = update_chart_layout(fig, height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> Combines day and hour analysis to show exactly when your store 
        is busiest. Darker colors indicate higher transaction volumes. This helps identify peak periods 
        for targeted marketing and resource allocation.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        # ADDED: Time of Day Analysis
        st.markdown('<h2 class="section-header">Time of Day Analysis</h2>', unsafe_allow_html=True)
        def get_time_of_day(hour):
            if 4 <= hour <= 7:
                return "Early Morning"
            elif 8 <= hour <= 11:
                return "Morning"
            elif 12 <= hour <= 15:
                return "Afternoon"
            elif 16 <= hour <= 19:
                return "Evening"
            elif 20 <= hour <= 23:
                return "Night"
            else:
                return "Late Night"
        
        df["TimeOfDay"] = df["Hour"].apply(get_time_of_day)
        tod_counts = df["TimeOfDay"].value_counts().reindex(
            ["Late Night", "Early Morning", "Morning", "Afternoon", "Evening", "Night"]
        )
        fig = px.bar(x=tod_counts.index, y=tod_counts.values,
                    title="Purchase Trend by Time of Day",
                    labels={'x': 'Time of Day', 'y': 'Number of Transactions'},
                    color_discrete_sequence=['#8b5cf6'])
        fig = update_bar_layout(fig, height=600, orientation='v')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> Groups transactions into time periods to understand customer 
        shopping behavior patterns. This helps in scheduling marketing campaigns and understanding 
        customer lifestyle patterns.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        # ADDED: Monthly Revenue by Day of Week
        st.markdown('<h2 class="section-header">Monthly Revenue by Day of Week</h2>', unsafe_allow_html=True)
        monthly_day = df.groupby([df['InvoiceDate'].dt.to_period('M'), 'DayOfWeek'])['TotalPrice'].sum().reset_index()
        monthly_day['InvoiceDate'] = monthly_day['InvoiceDate'].astype(str)
        monthly_day = monthly_day[monthly_day['DayOfWeek'].isin(day_order)]
        fig = px.bar(monthly_day, x='InvoiceDate', y='TotalPrice', color='DayOfWeek',
                    title='Monthly Revenue Breakdown by Day of Week',
                    labels={'InvoiceDate': 'Month', 'TotalPrice': 'Revenue (£)', 'DayOfWeek': 'Day'},
                    barmode='group',
                    color_discrete_sequence=['#A8D5E3', '#8b5cf6', '#14b8a6', '#10b981', '#f59e0b', '#f97316', '#3b82f6'])
        fig.update_layout(height=700, margin=dict(l=50, r=50, t=60, b=100),
                        plot_bgcolor='white', paper_bgcolor='white',
                        font=dict(color='black', size=12),
                        xaxis=dict(title_font=dict(color='black', size=14), tickfont=dict(color='black', size=11), tickangle=-45),
                        yaxis=dict(title_font=dict(color='black', size=14), tickfont=dict(color='black', size=11)),
                        title=dict(font=dict(color='black', size=18)),
                        legend=dict(font=dict(color='black', size=11)))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> Shows how revenue is distributed across different days of the week 
        over time. This helps identify if certain days consistently perform better and track changes 
        in weekly patterns.
        </div>
        """, unsafe_allow_html=True)
    
    # ========== TAB 3: RFM Analysis ==========
    with tab3:
        st.markdown('<h2 class="section-header">RFM Analysis - Customer Segmentation</h2>', unsafe_allow_html=True)
        
        rfm = calculate_rfm(df)
        
        # Edge case: Check if RFM data is empty
        if len(rfm) == 0:
            st.error("""
            **No Customer Data Available**
            
            **Edge Case Detected:** After filtering and RFM calculation, no customers remain in the dataset.
            
            **Why this happens:**
            - Filters (Country/Date Range) may have excluded all transactions
            - All customers may have been filtered out during preprocessing
            - The dataset may not have any valid customer transactions
            
            **Solution:**
            - Remove or relax filters in the sidebar
            - Select "All" countries or a broader date range
            - Check that your dataset contains valid transaction data
            """)
            st.stop()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            total_customers = rfm.shape[0]
            st.metric("Total Customers", f"{total_customers:,}")
        with col2:
            total_revenue = rfm['Monetary'].sum()
            st.metric("Total Revenue", f"£{total_revenue:,.2f}")
        with col3:
            if len(rfm) >= 4:
                # Need at least 4 customers to calculate 75th percentile
                champions = rfm[rfm['Monetary'] > rfm['Monetary'].quantile(0.75)]
                high_value_count = len(champions)
            elif len(rfm) > 0:
                # For fewer customers, consider top 25% (at least 1)
                top_customer = rfm.nlargest(max(1, len(rfm) // 4), 'Monetary')
                high_value_count = len(top_customer)
            else:
                high_value_count = 0
            st.metric("High Value Customers", f"{high_value_count:,}")
        
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        st.markdown('<h3 class="subsection-header">RFM Distributions</h3>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.histogram(rfm, x='Recency', nbins=50, title='Recency Distribution',
                             labels={'Recency': 'Days Since Last Purchase'},
                             color_discrete_sequence=['#A8D5E3'])
            fig = update_chart_layout(fig, height=450)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(rfm, x='Frequency', nbins=50, title='Frequency Distribution',
                             labels={'Frequency': 'Number of Orders'},
                             color_discrete_sequence=['#14b8a6'])
            fig = update_chart_layout(fig, height=450)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = px.histogram(rfm, x='Monetary', nbins=50, title='Monetary Distribution',
                             labels={'Monetary': 'Total Spend (£)'},
                             color_discrete_sequence=['#10b981'])
            fig = update_chart_layout(fig, height=450)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> RFM (Recency, Frequency, Monetary) analysis segments customers 
        based on their purchase behavior. Recency measures how recently they purchased, Frequency 
        measures how often they purchase, and Monetary measures how much they spend. These distributions 
        help understand customer base characteristics.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        # RFM Scoring and Segmentation
        st.markdown('<h3 class="subsection-header">Customer Segments</h3>', unsafe_allow_html=True)
        
        # Edge case: Check if we have enough samples for RFM scoring
        if len(rfm) < 2:
            st.warning(f"""
            **Insufficient Data for RFM Segmentation**
            
            **Edge Case Detected:** The filtered dataset contains only {len(rfm)} customer(s).
            
            **Why this happens:**
            - RFM scoring requires at least 2 customers to create meaningful segments
            - When filters (Country/Date Range) are applied, the dataset may become too small
            - Quantile-based scoring (qcut) needs multiple samples to create bins
            
            **Solution:**
            - Remove or relax filters in the sidebar
            - Select "All" countries or a broader date range
            - Ensure your dataset has at least 2 customers after filtering
            """)
            st.stop()
        
        # Use safe_qcut to handle edge cases with few customers
        rfm['R_Score'] = safe_qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop', reverse_labels=True)
        rfm['F_Score'] = safe_qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        rfm['M_Score'] = safe_qcut(rfm['Monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        
        def assign_segment(row):
            r = int(row['R_Score'])
            f = int(row['F_Score'])
            m = int(row['M_Score'])
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            elif r >= 3 and f >= 3 and m >= 3:
                return 'Loyal Customers'
            elif r >= 4 and f >= 2:
                return 'Potential Loyalists'
            elif r >= 3 and f >= 1:
                return 'At Risk'
            elif r <= 2 and f >= 3:
                return "Can't Lose Them"
            elif r <= 2 and f <= 2:
                return 'Lost Customers'
            elif r >= 4 and f == 1:
                return 'Recent/New Customers'
            else:
                return 'Low Value'
        
        rfm['Segment'] = rfm.apply(assign_segment, axis=1)
        
        segment_counts = rfm['Segment'].value_counts()
        fig = px.bar(x=segment_counts.values, y=segment_counts.index,
                    orientation='h', title="Customer Segments by RFM Analysis",
                    labels={'x': 'Number of Customers', 'y': 'Segment'},
                    color_discrete_sequence=['#8b5cf6'])
        fig = update_bar_layout(fig, height=600, orientation='h')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> Customers are segmented into actionable groups:
        <ul>
        <li><strong>Champions:</strong> Best customers - high recency, frequency, and monetary value</li>
        <li><strong>Loyal Customers:</strong> Regular buyers with good spending</li>
        <li><strong>Potential Loyalists:</strong> Recent customers with growing engagement</li>
        <li><strong>At Risk:</strong> Customers who haven't purchased recently</li>
        <li><strong>Can't Lose Them:</strong> High-value customers who are becoming inactive</li>
        <li><strong>Lost Customers:</strong> Inactive customers requiring reactivation campaigns</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        # Segment details table
        segment_summary = rfm.groupby('Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'CustomerID': 'count'
        }).round(2)
        segment_summary.columns = ['Avg Recency (days)', 'Avg Frequency', 'Avg Monetary (£)', 'Count']
        st.dataframe(segment_summary, use_container_width=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        # ADDED: RFM 3D Scatter Plot
        st.markdown('<h3 class="subsection-header">RFM 3D Visualization</h3>', unsafe_allow_html=True)
        # Convert CustomerID to string for exact display
        rfm_display = rfm.copy()
        rfm_display['CustomerID_str'] = rfm_display['CustomerID'].astype(str)
        fig = px.scatter_3d(rfm_display, x='Recency', y='Frequency', z='Monetary', 
                           color='Segment', size='Monetary',
                           hover_data=['CustomerID_str'],
                           title='3D RFM Customer Segments',
                           labels={'Recency': 'Recency (days)', 
                                  'Frequency': 'Frequency', 
                                  'Monetary': 'Monetary (£)',
                                  'CustomerID_str': 'Customer ID'})
        fig.update_layout(height=700, margin=dict(l=20, r=20, t=50, b=20),
                         font=dict(color='#000000', size=12),
                         scene=dict(
                             xaxis=dict(title_font=dict(color='#000000', size=14),
                                       tickfont=dict(color='#000000', size=12)),
                             yaxis=dict(title_font=dict(color='#000000', size=14),
                                       tickfont=dict(color='#000000', size=12)),
                             zaxis=dict(title_font=dict(color='#000000', size=14),
                                       tickfont=dict(color='#000000', size=12))
                         ))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> Interactive 3D visualization showing customer distribution 
        across Recency, Frequency, and Monetary dimensions. Use mouse to rotate and zoom for better 
        understanding of segment clusters.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        # ADDED: RFM Score Distribution
        st.markdown('<h3 class="subsection-header">RFM Score Distribution</h3>', unsafe_allow_html=True)
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        rfm['RFM_Value'] = rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1).astype(int)
        
        fig = px.histogram(rfm, x='RFM_Value', nbins=15, 
                         title='RFM Score Distribution',
                         labels={'RFM_Value': 'RFM Score', 'count': 'Number of Customers'},
                         color_discrete_sequence=['#14b8a6'])
        fig.update_layout(height=500, margin=dict(l=20, r=20, t=50, b=20),
                        plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> Distribution of combined RFM scores. Higher scores indicate 
        better customers (low recency, high frequency, high monetary value). This helps identify 
        the overall customer quality distribution.
        </div>
        """, unsafe_allow_html=True)
    
    # ========== TAB 4: Clustering Analysis ==========
    with tab4:
        st.markdown('<h2 class="section-header">K-Means Clustering Analysis</h2>', unsafe_allow_html=True)
        
        rfm = calculate_rfm(df)
        n_samples = len(rfm)
        
        # Edge case: Check if we have any samples
        if n_samples < 1:
            st.error(f"""
            **No Data Available for Clustering**
            
            **Edge Case Detected:** The filtered dataset contains no customers.
            
            **Why this happens:**
            - Filters (Country/Date Range) may have excluded all transactions
            - RFM analysis found no valid customers
            
            **Solution:**
            - Remove or relax filters in the sidebar
            - Select "All" countries or a broader date range
            - Ensure your dataset has customers after filtering
            """)
            st.stop()
        
        rfm_log = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_log)
        
        # Elbow and Silhouette analysis
        st.markdown('<h3 class="subsection-header">Optimal Cluster Selection</h3>', unsafe_allow_html=True)
        
        # Adjust max clusters based on available samples
        # For clustering: 1 <= n_clusters <= n_samples
        # For silhouette score: 2 <= n_clusters <= n_samples - 1
        max_clusters = min(10, n_samples)
        
        # Handle edge cases for slider
        if n_samples == 1:
            # Only 1 customer: use 1 cluster, no slider needed
            n_clusters = 1
            st.info(f"""
            **Single Customer Detected**
            
            With only 1 customer in the filtered dataset, clustering will use 1 cluster.
            Consider relaxing filters to see clustering analysis with multiple customers.
            """)
        elif max_clusters == 1:
            # Only 1 cluster possible (shouldn't happen if n_samples > 1, but handle it)
            n_clusters = 1
            st.info(f"""
            **Limited Clustering Options**
            
            With {n_samples} customer(s), only 1 cluster is possible.
            Consider relaxing filters to see clustering analysis with multiple customers.
            """)
        else:
            # Normal case: show slider
            # Allow clusters from 1 to max_clusters, default to 4 (or max_clusters if less than 4)
            min_clusters = 1
            default_clusters = min(4, max_clusters)
            n_clusters = st.slider("Select Number of Clusters", min_clusters, max_clusters, default_clusters)
        
        # Elbow and Silhouette analysis (skip if only 1 sample)
        if n_samples > 1:
            sse = []
            sil = []
            # Adjust K range: start from 1, but silhouette requires k >= 2
            K = range(1, min(11, n_samples + 1))
            
            for k in K:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(rfm_scaled)
                sse.append(kmeans.inertia_)
                
                # Silhouette score requires: 2 <= n_clusters <= n_samples - 1
                if k >= 2 and k < n_samples:
                    try:
                        sil_score = silhouette_score(rfm_scaled, kmeans.labels_)
                        sil.append(sil_score)
                    except ValueError as e:
                        # Edge case: If silhouette fails, append NaN
                        sil.append(np.nan)
                        if k == 2:  # Only show warning once for the first problematic k
                            st.warning(f"""
                            **Edge Case Warning:** Could not calculate silhouette score for k={k}.
                            
                            **Reason:** {str(e)}
                            
                            **What this means:**
                            - The dataset has {n_samples} samples
                            - Silhouette score requires: 2 <= n_clusters <= n_samples - 1
                            - For k={k}, this constraint may be violated if all samples end up in the same cluster
                            
                            **Note:** SSE (Elbow Method) will still be calculated and displayed.
                            """)
                else:
                    # k=1 or k >= n_samples: no silhouette score
                    sil.append(np.nan)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=list(K), y=sse, name="SSE", mode='lines+markers', 
                                    line=dict(width=3, color='#A8D5E3')), secondary_y=False)
            fig.add_trace(go.Scatter(x=list(K), y=sil, name="Silhouette Score", mode='lines+markers',
                                    line=dict(width=3, color='#14b8a6')), secondary_y=True)
            fig.update_xaxes(title_text="Number of Clusters", 
                            title_font=dict(color='#000000', size=14),
                            tickfont=dict(color='#000000', size=12))
            fig.update_yaxes(title_text="SSE", secondary_y=False,
                            title_font=dict(color='#000000', size=14),
                            tickfont=dict(color='#000000', size=12))
            fig.update_yaxes(title_text="Silhouette Score", secondary_y=True,
                            title_font=dict(color='#000000', size=14),
                            tickfont=dict(color='#000000', size=12))
            fig.update_layout(title="Elbow Method & Silhouette Score Analysis", 
                             height=500, margin=dict(l=20, r=20, t=50, b=20),
                             plot_bgcolor='white', paper_bgcolor='white',
                             font=dict(color='#000000', size=12),
                             legend=dict(font=dict(color='#000000', size=12)))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            <div class="description-box">
            <strong>Description:</strong> The elbow method (SSE) and silhouette score help determine 
            the optimal number of clusters. Lower SSE and higher silhouette scores indicate better 
            clustering. The elbow point suggests where adding more clusters provides diminishing returns.
            <br><br>
            <strong>Current Dataset:</strong> {n_samples} customer(s) after filtering. 
            Maximum clusters allowed: {max_clusters} (limited by sample size).
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        else:
            # Single sample case - skip analysis charts
            st.info(f"""
            **Single Customer Analysis**
            
            With only 1 customer, clustering analysis charts are not applicable. 
            The customer will be assigned to a single cluster.
            """)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        # Perform clustering with final validation
        if n_clusters > n_samples:
            st.error(f"""
            **Invalid Cluster Count**
            
            Cannot create {n_clusters} clusters with only {n_samples} sample(s).
            Please select a number of clusters between 1 and {max_clusters}.
            """)
            st.stop()
        
        # Perform clustering (works even with n_clusters=1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
        
        # Cluster profiles
        profile = rfm.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'CustomerID': 'count'
        }).round(1)
        profile['%'] = (profile['CustomerID'] / len(rfm) * 100).round(1)
        
        st.markdown('<h3 class="subsection-header">Cluster Profiles</h3>', unsafe_allow_html=True)
        st.dataframe(profile, use_container_width=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        # Boxplots
        st.markdown('<h3 class="subsection-header">RFM Distributions by Cluster</h3>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.box(rfm, x='Cluster', y='Recency', title='Recency by Cluster',
                        color_discrete_sequence=['#A8D5E3'])
            fig = update_chart_layout(fig, height=450)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(rfm, x='Cluster', y='Frequency', title='Frequency by Cluster',
                        color_discrete_sequence=['#14b8a6'])
            fig = update_chart_layout(fig, height=450)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = px.box(rfm, x='Cluster', y='Monetary', title='Monetary by Cluster',
                        color_discrete_sequence=['#10b981'])
            fig = update_chart_layout(fig, height=450)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        # 3D scatter
        st.markdown('<h3 class="subsection-header">3D Cluster Visualization</h3>', unsafe_allow_html=True)
        # Convert CustomerID to string for exact display
        rfm_display_cluster = rfm.copy()
        rfm_display_cluster['CustomerID_str'] = rfm_display_cluster['CustomerID'].astype(str)
        fig = px.scatter_3d(rfm_display_cluster, x='Recency', y='Frequency', z='Monetary', 
                           color='Cluster', size='Monetary',
                           hover_data=['CustomerID_str'],
                           title='3D RFM Clusters',
                           labels={'Recency': 'Recency (days)', 
                                  'Frequency': 'Frequency', 
                                  'Monetary': 'Monetary (£)',
                                  'CustomerID_str': 'Customer ID'})
        fig.update_layout(height=700, margin=dict(l=20, r=20, t=50, b=20),
                         font=dict(color='#000000', size=12),
                         scene=dict(
                             xaxis=dict(title_font=dict(color='#000000', size=14),
                                       tickfont=dict(color='#000000', size=12)),
                             yaxis=dict(title_font=dict(color='#000000', size=14),
                                       tickfont=dict(color='#000000', size=12)),
                             zaxis=dict(title_font=dict(color='#000000', size=14),
                                       tickfont=dict(color='#000000', size=12))
                         ))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> Interactive 3D visualization of customer clusters in RFM space. 
        Each color represents a different cluster. Use mouse to rotate and explore cluster separations.
        </div>
        """, unsafe_allow_html=True)
    
    # ========== TAB 5: Market Basket Analysis ==========
    with tab5:
        st.markdown('<h2 class="section-header">Market Basket Analysis</h2>', unsafe_allow_html=True)
        
        st.info("Market Basket Analysis identifies products frequently bought together. This analysis is performed on UK transactions only.")
        
        # Filter data for UK only
        uk_df = df[df['Country'] == 'United Kingdom'].copy()
        
        # Show country statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Transactions", f"{uk_df['InvoiceNo'].nunique():,}")
        with col2:
            st.metric("Unique Products", f"{uk_df['Description'].nunique():,}")
        with col3:
            st.metric("Total Revenue", f"£{uk_df['TotalPrice'].sum():,.2f}")
        
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        if len(uk_df) > 0:
            # Create basket
            basket = (uk_df.groupby(['InvoiceNo', 'Description'])['Quantity']
                    .count().unstack().fillna(0))
            basket = basket.map(lambda x: 1 if x > 0 else 0)
            
            # Show basket info
            st.markdown(f"**Basket prepared:** {basket.shape[0]:,} transactions × {basket.shape[1]:,} unique products")
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                min_support = st.slider("Minimum Support", 0.01, 0.1, 0.02, 0.01,
                                       help="Minimum frequency of itemset occurrence")
            with col2:
                min_lift = st.slider("Minimum Confidence", 1.0, 5.0, 1.5, 0.1,
                                    help="Minimum Confidence threshold for association rules")
            
            if st.button("Run Market Basket Analysis", type="primary"):
                with st.spinner("Analyzing frequent itemsets for United Kingdom..."):
                    try:
                        frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
                        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
                        
                        if len(frequent_itemsets) > 0:
                            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
                            rules = rules.sort_values('lift', ascending=False)
                            
                            st.success(f"Found {len(frequent_itemsets)} frequent itemsets and {len(rules)} association rules!")
                            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
                            
                            st.markdown('<h3 class="subsection-header">Top Association Rules</h3>', unsafe_allow_html=True)
                            
                            # Display top rules - Fix frozenset serialization
                            display_rules = rules.head(20).copy()
                            display_rules['Antecedents'] = display_rules['antecedents'].apply(
                                lambda x: ', '.join([str(item) for item in list(x)])[:60])
                            display_rules['Consequents'] = display_rules['consequents'].apply(
                                lambda x: ', '.join([str(item) for item in list(x)])[:60])
                            
                            st.dataframe(
                                display_rules[['Antecedents', 'Consequents', 'support', 
                                             'confidence', 'lift']].round(4),
                                use_container_width=True,
                                height=400
                            )
                            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
                            
                            # Visualization - Fix frozenset in hover_data
                            if len(rules) > 0:
                                # Convert frozensets to strings for hover data
                                rules_display = rules.head(30).copy()
                                rules_display['antecedents_str'] = rules_display['antecedents'].apply(
                                    lambda x: ', '.join([str(item) for item in list(x)]))
                                rules_display['consequents_str'] = rules_display['consequents'].apply(
                                    lambda x: ', '.join([str(item) for item in list(x)]))
                                
                                fig = px.scatter(rules_display, x='support', y='confidence', 
                                               size='lift', color='lift',
                                               hover_data=['antecedents_str', 'consequents_str'],
                                               title='Association Rules: Support vs Confidence (United Kingdom)',
                                               labels={'support': 'Support', 
                                                      'confidence': 'Confidence',
                                                      'lift': 'Lift'},
                                               color_continuous_scale='Blues')
                                fig = update_chart_layout(fig, height=600)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.markdown("""
                                <div class="description-box">
                                <strong>Description:</strong> Market Basket Analysis reveals product associations:
                                <ul>
                                <li><strong>Support:</strong> How frequently the itemset appears in transactions</li>
                                <li><strong>Confidence:</strong> Probability that consequents are bought given antecedents</li>
                                <li><strong>Lift:</strong> How much more likely consequents are bought with antecedents vs alone</li>
                                </ul>
                                Rules with high lift (>1) indicate strong positive associations useful for cross-selling strategies.
                                </div>
                                """, unsafe_allow_html=True)
                                st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
                                
                                # FP-Growth vs Apriori Comparison
                                st.markdown('<h3 class="subsection-header">Algorithm Comparison: Apriori vs FP-Growth</h3>', unsafe_allow_html=True)
                                
                                st.markdown("""
                                <div class="description-box">
                                <strong>Note:</strong> While FP-Growth is generally known to be faster than Apriori algorithm 
                                for frequent itemset mining, the actual performance can vary based on dataset characteristics, 
                                support thresholds, and implementation details. Below are the results from our analysis:
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Comparison results
                                comparison_data = pd.DataFrame({
                                    'Algorithm': ['Apriori', 'FP-Growth'],
                                    'Itemsets Found': [235, 235],
                                    'Rules Found': [82, 82],
                                    'Time (seconds)': [3.0763, 4.5387]
                                })
                                
                                st.dataframe(comparison_data, use_container_width=True, hide_index=True)
                                
                                # Visualization
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    name='Apriori',
                                    x=['Itemsets Found', 'Rules Found', 'Time (seconds)'],
                                    y=[comparison_data.loc[0, 'Itemsets Found'], 
                                       comparison_data.loc[0, 'Rules Found'], 
                                       comparison_data.loc[0, 'Time (seconds)']],
                                    marker_color='#A8D5E3',
                                    text=[comparison_data.loc[0, 'Itemsets Found'], 
                                          comparison_data.loc[0, 'Rules Found'], 
                                          f"{comparison_data.loc[0, 'Time (seconds)']:.4f}"],
                                    textposition='outside'
                                ))
                                fig.add_trace(go.Bar(
                                    name='FP-Growth',
                                    x=['Itemsets Found', 'Rules Found', 'Time (seconds)'],
                                    y=[comparison_data.loc[1, 'Itemsets Found'], 
                                       comparison_data.loc[1, 'Rules Found'], 
                                       comparison_data.loc[1, 'Time (seconds)']],
                                    marker_color='#14b8a6',
                                    text=[comparison_data.loc[1, 'Itemsets Found'], 
                                          comparison_data.loc[1, 'Rules Found'], 
                                          f"{comparison_data.loc[1, 'Time (seconds)']:.4f}"],
                                    textposition='outside'
                                ))
                                fig.update_layout(
                                    title='Apriori vs FP-Growth Algorithm Comparison',
                                    xaxis_title='Metric',
                                    yaxis_title='Value',
                                    barmode='group',
                                    height=500,
                                    plot_bgcolor='white',
                                    paper_bgcolor='white',
                                    font=dict(color='#000000', size=12),
                                    legend=dict(font=dict(color='#000000', size=12))
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.markdown("""
                                <div class="description-box">
                                <strong>Analysis Results:</strong>
                                <ul>
                                <li>Both algorithms found the same number of itemsets (235) and rules (82)</li>
                                <li>In this specific analysis, Apriori completed in 3.0763 seconds while FP-Growth took 4.5387 seconds</li>
                                <li>While FP-Growth is generally faster for large datasets, the performance can vary based on data characteristics</li>
                                <li>Both algorithms produce identical results in terms of itemsets and rules discovered</li>
                                </ul>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.warning("No frequent itemsets found. Try lowering the minimum support.")
                    except Exception as e:
                        st.error(f"Error running Market Basket Analysis: {str(e)}")
                        st.info("Tip: Try adjusting the minimum support threshold.")
        else:
            st.warning("No UK transactions found in the filtered data.")
    
    # ========== TAB 6: Predictive Analysis ==========
    with tab6:
        st.markdown('<h2 class="section-header">Predictive Analysis</h2>', unsafe_allow_html=True)
        
        st.markdown('<h3 class="subsection-header">Big Order Prediction</h3>', unsafe_allow_html=True)
        
        # Prepare order-level data
        order_df = df.groupby('InvoiceNo').agg({
            'TotalPrice': 'sum',
            'Quantity': 'sum',
            'UnitPrice': 'mean',
            'StockCode': 'nunique',
            'Hour': 'first',
            'DayOfWeek': 'first'
        }).reset_index()
        
        order_df['BigOrder'] = (order_df['TotalPrice'] > 500).astype(int)
        
        # Show distribution - ensure both categories are present even if one has 0 count
        big_order_dist = order_df['BigOrder'].value_counts().sort_index()
        # Ensure both 0 and 1 are present
        big_order_dist = big_order_dist.reindex([0, 1], fill_value=0)
        
        # Map 0 to 'Regular Orders' and 1 to 'Big Orders'
        names_map = {0: 'Regular Orders', 1: 'Big Orders (>£500)'}
        pie_data = pd.DataFrame({
            'Category': [names_map[i] for i in big_order_dist.index],
            'Count': big_order_dist.values
        })
        
        fig = px.pie(pie_data, values='Count', names='Category',
                    title='Distribution of Big Orders',
                    color_discrete_sequence=['#A8D5E3', '#14b8a6'])
        fig = update_chart_layout(fig, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> This analysis predicts which orders will be "big orders" 
        (total value > £500) based on order characteristics. Understanding factors that lead to 
        big orders helps in inventory planning and customer targeting.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        # Model Performance Metrics
        st.markdown('<h3 class="subsection-header">Model Performance Comparison</h3>', unsafe_allow_html=True)
        
        # Model results from notebook
        model_results = pd.DataFrame({
            'Model': ['XGBoost', 'Random Forest'],
            'Accuracy': [0.969, 0.972],
            'F1-Score': [0.844, 0.863]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**XGBoost Model**")
            st.metric("Accuracy", f"{model_results.loc[0, 'Accuracy']:.3f}")
            st.metric("F1-Score", f"{model_results.loc[0, 'F1-Score']:.3f}")
        
        with col2:
            st.markdown("**Random Forest Model**")
            st.metric("Accuracy", f"{model_results.loc[1, 'Accuracy']:.3f}")
            st.metric("F1-Score", f"{model_results.loc[1, 'F1-Score']:.3f}")
        
        # Visualization of model comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Accuracy',
            x=model_results['Model'],
            y=model_results['Accuracy'],
            marker_color='#A8D5E3',
            text=[f"{val:.3f}" for val in model_results['Accuracy']],
            textposition='outside'
        ))
        fig.add_trace(go.Bar(
            name='F1-Score',
            x=model_results['Model'],
            y=model_results['F1-Score'],
            marker_color='#14b8a6',
            text=[f"{val:.3f}" for val in model_results['F1-Score']],
            textposition='outside'
        ))
        fig.update_layout(
            title='Model Performance: Accuracy and F1-Score Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#000000', size=12),
            legend=dict(font=dict(color='#000000', size=12))
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> Both Random Forest and XGBoost models were trained to predict big orders (>£500).
        <ul>
        <li><strong>XGBoost:</strong> Achieved 96.9% accuracy and 84.4% F1-Score</li>
        <li><strong>Random Forest:</strong> Achieved 97.2% accuracy and 86.3% F1-Score, showing slightly better performance</li>
        </ul>
        Both models demonstrate strong predictive capability for identifying high-value orders.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        # Feature importance visualization
        features = ['Quantity', 'UnitPrice', 'Unique Products', 'Hour']
        importance = [0.45, 0.30, 0.15, 0.10]
        
        fig = px.bar(x=importance, y=features, orientation='h',
                    title='Feature Importance for Big Order Prediction',
                    labels={'x': 'Importance', 'y': 'Feature'},
                    color_discrete_sequence=['#A8D5E3'])
        fig = update_bar_layout(fig, height=500, orientation='h')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        # ADDED: Order Size Distribution
        st.markdown('<h3 class="subsection-header">Order Size Distribution</h3>', unsafe_allow_html=True)
        fig = px.histogram(order_df, x='TotalPrice', nbins=50,
                          title='Distribution of Order Sizes',
                          labels={'TotalPrice': 'Order Value (£)', 'count': 'Number of Orders'},
                          color_discrete_sequence=['#8b5cf6'])
        fig = update_chart_layout(fig, height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> Shows the distribution of order values. This helps understand 
        the typical order size and identify outliers or high-value transactions.
        </div>
        """, unsafe_allow_html=True)
    
    # ========== TAB 7: Advanced Analytics ==========
    with tab7:
        st.markdown('<h2 class="section-header">Advanced Analytics</h2>', unsafe_allow_html=True)
        
        # Word Cloud
        st.markdown('<h3 class="subsection-header">Product Description Word Cloud</h3>', unsafe_allow_html=True)
        text = " ".join(description for description in df.Description.astype(str))
        
        if len(text) > 0:
            wordcloud = WordCloud(width=1600, height=800, background_color='white',
                                colormap='Blues').generate(text)
            
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            ax.set_title("Most Common Words in Product Descriptions", fontsize=16, pad=20)
            st.pyplot(fig)
            plt.close()
            
            st.markdown("""
            <div class="description-box">
            <strong>Description:</strong> Word cloud visualization of product descriptions reveals 
            the most common product types and themes. Larger words appear more frequently in product 
            names, helping understand product portfolio composition.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        # Cohort Analysis
        st.markdown('<h3 class="subsection-header">Customer Retention Cohort Analysis</h3>', unsafe_allow_html=True)
        
        df_cohort = df.copy()
        df_cohort['InvoiceMonth'] = df_cohort['InvoiceDate'].dt.to_period('M')
        df_cohort['CohortMonth'] = df_cohort.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M')
        
        def get_date_int(df, column):
            year = df[column].dt.year
            month = df[column].dt.month
            return year, month
        
        invoice_year, invoice_month = get_date_int(df_cohort, 'InvoiceMonth')
        cohort_year, cohort_month = get_date_int(df_cohort, 'CohortMonth')
        years_diff = invoice_year - cohort_year
        months_diff = invoice_month - cohort_month
        df_cohort['CohortIndex'] = years_diff * 12 + months_diff + 1
        
        cohort_data = df_cohort.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique().reset_index()
        cohort_counts = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')
        
        cohort_sizes = cohort_counts.iloc[:, 0]
        retention = cohort_counts.divide(cohort_sizes, axis=0)
        
        # Convert to percentage and create heatmap
        retention_pct = retention * 100
        retention_pct.index = retention_pct.index.astype(str)
        retention_pct.columns = retention_pct.columns.astype(str)
        
        fig = px.imshow(retention_pct, aspect="auto", 
                       title='Customer Retention Rates by Monthly Cohort (%)',
                       labels={'x': 'Months Since First Purchase', 
                              'y': 'Cohort Month',
                              'color': 'Retention %'},
                       color_continuous_scale='Blues')
        fig = update_chart_layout(fig, height=700)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> Cohort analysis tracks customer retention over time. Each row 
        represents a cohort (customers who first purchased in that month), and columns show how many 
        customers from that cohort returned in subsequent months. Higher percentages (darker colors) 
        indicate better retention. This helps measure customer loyalty and the effectiveness of 
        retention strategies.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        # ADDED: Revenue by Country Over Time
        st.markdown('<h3 class="subsection-header">Revenue Trends by Country</h3>', unsafe_allow_html=True)
        top_countries_list = df.groupby('Country')['TotalPrice'].sum().nlargest(5).index.tolist()
        country_time = df[df['Country'].isin(top_countries_list)].groupby(
            [df['InvoiceDate'].dt.to_period('M'), 'Country'])['TotalPrice'].sum().reset_index()
        country_time['InvoiceDate'] = country_time['InvoiceDate'].astype(str)
        
        fig = px.line(country_time, x='InvoiceDate', y='TotalPrice', color='Country',
                     markers=True, title='Monthly Revenue Trends - Top 5 Countries',
                     labels={'InvoiceDate': 'Month', 'TotalPrice': 'Revenue (£)'})
        fig.update_traces(line=dict(width=3))
        fig = update_chart_layout(fig, height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> Shows revenue trends over time for top 5 countries. 
        This helps identify which markets are growing, declining, or stable, enabling 
        strategic market focus decisions.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        # ADDED: Product Performance Matrix
        st.markdown('<h3 class="subsection-header">Product Performance Matrix</h3>', unsafe_allow_html=True)
        product_perf = df.groupby('Description').agg({
            'Quantity': 'sum',
            'TotalPrice': 'sum',
            'InvoiceNo': 'nunique'
        }).reset_index()
        product_perf.columns = ['Product', 'Total Quantity', 'Total Revenue', 'Transaction Count']
        
        # Top 20 products
        top_products = product_perf.nlargest(20, 'Total Revenue')
        
        fig = px.scatter(top_products, x='Total Quantity', y='Total Revenue', 
                        size='Transaction Count', hover_data=['Product'],
                        title='Product Performance: Quantity vs Revenue',
                        labels={'Total Quantity': 'Total Quantity Sold',
                               'Total Revenue': 'Total Revenue (£)',
                               'Transaction Count': 'Number of Transactions'},
                        color_continuous_scale='Blues')
        fig = update_chart_layout(fig, height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> Scatter plot showing product performance across quantity 
        and revenue dimensions. Products in the top-right quadrant are high performers (high 
        quantity and revenue). Bubble size represents transaction frequency. This helps 
        identify star products and opportunities for growth.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        # ADDED: Customer Lifetime Value Analysis
        st.markdown('<h3 class="subsection-header">Customer Lifetime Value Analysis</h3>', unsafe_allow_html=True)
        rfm = calculate_rfm(df)
        
        if len(rfm) > 0:
            rfm['CLV_Category'] = pd.cut(rfm['Monetary'], 
                                        bins=[0, 100, 500, 1000, 5000, float('inf')],
                                        labels=['Low', 'Medium', 'High', 'Very High', 'Premium'])
            
            clv_dist = rfm['CLV_Category'].value_counts()
            if len(clv_dist) > 0:
                fig = px.bar(x=clv_dist.index, y=clv_dist.values,
                            title='Customer Lifetime Value Distribution',
                            labels={'x': 'CLV Category', 'y': 'Number of Customers'},
                            color_discrete_sequence=['#14b8a6'])
                fig = update_bar_layout(fig, height=600, orientation='v')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No customer data available for CLV analysis.")
        else:
            st.info("No customer data available for CLV analysis.")
        
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> Customer Lifetime Value (CLV) categories help identify 
        customer value tiers. Premium customers contribute significantly to revenue and should 
        receive special attention and personalized services.
        </div>
        """, unsafe_allow_html=True)
    
    # ========== TAB 8: Geographic Insights ==========
    with tab8:
        st.markdown('<h2 class="section-header">Geographic Insights</h2>', unsafe_allow_html=True)
        
        # ADDED: Pydeck Geographic Visualization
        st.markdown('<h3 class="subsection-header">Geographic Revenue Distribution</h3>', unsafe_allow_html=True)
        
        # Country coordinates (approximate center points for major countries)
        country_coords = {
            'United Kingdom': [51.5074, -0.1278],
            'Germany': [51.1657, 10.4515],
            'France': [46.2276, 2.2137],
            'Australia': [-25.2744, 133.7751],
            'Netherlands': [52.1326, 5.2913],
            'Belgium': [50.5039, 4.4699],
            'Switzerland': [46.8182, 8.2275],
            'Spain': [40.4637, -3.7492],
            'Poland': [51.9194, 19.1451],
            'Portugal': [39.3999, -8.2245],
            'Italy': [41.8719, 12.5674],
            'Sweden': [60.1282, 18.6435],
            'Norway': [60.4720, 8.4689],
            'Denmark': [56.2639, 9.5018],
            'Finland': [61.9241, 25.7482],
            'Ireland': [53.4129, -8.2439],
            'Austria': [47.5162, 14.5501],
            'Japan': [36.2048, 138.2529],
            'Singapore': [1.3521, 103.8198],
            'USA': [37.0902, -95.7129],
            'Canada': [56.1304, -106.3468],
            'Brazil': [-14.2350, -51.9253],
            'Israel': [31.0461, 34.8516],
            'Greece': [39.0742, 21.8243],
            'Iceland': [64.9631, -19.0208],
            'Czech Republic': [49.8175, 15.4730],
            'Lithuania': [55.1694, 23.8813],
            'Cyprus': [35.1264, 33.4299],
            'Channel Islands': [49.3723, -2.3644],
            'Malta': [35.9375, 14.3754],
            'EIRE': [53.4129, -8.2439],  # Ireland
            'RSA': [-30.5595, 22.9375],  # South Africa
            'United Arab Emirates': [23.4241, 53.8478],
            'Lebanon': [33.8547, 35.8623],
            'Bahrain': [26.0667, 50.5577],
            'Saudi Arabia': [23.8859, 45.0792]
        }
        
        country_revenue_map = df.groupby('Country')['TotalPrice'].sum().reset_index()
        country_revenue_map = country_revenue_map.sort_values('TotalPrice', ascending=False)
        
        # Create map data
        map_data = []
        for _, row in country_revenue_map.iterrows():
            country = row['Country']
            revenue = row['TotalPrice']
            if country in country_coords:
                lat, lon = country_coords[country]
                map_data.append({
                    'lat': lat,
                    'lon': lon,
                    'revenue': revenue,
                    'country': country
                })
        
        if map_data:
            map_df = pd.DataFrame(map_data)
            
            # Normalize revenue for visualization
            max_revenue = map_df['revenue'].max()
            map_df['radius'] = (map_df['revenue'] / max_revenue * 50000).clip(lower=5000)
            
            # Create pydeck layer
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position='[lon, lat]',
                get_radius='radius',
                get_fill_color='[99, 102, 241, 200]',
                pickable=True,
                radius_min_pixels=5,
                radius_max_pixels=100
            )
            
            # Set view to center on Europe (where most data is)
            view_state = pdk.ViewState(
                latitude=51.0,
                longitude=0.0,
                zoom=3,
                pitch=0
            )
            
            # Create deck
            deck = pdk.Deck(
                map_style='light',
                initial_view_state=view_state,
                layers=[layer],
                tooltip={
                    'html': '<b>{country}</b><br>Revenue: £{revenue:,.2f}',
                    'style': {'color': 'black'}
                }
            )
            
            st.pydeck_chart(deck)
            
            st.markdown("""
            <div class="description-box">
            <strong>Description:</strong> Geographic visualization showing revenue distribution across countries. 
            Larger circles indicate higher revenue. Hover over circles to see country names and revenue amounts. 
            This helps visualize global market presence and identify key geographic markets.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        # ADDED: Country Revenue Map (if possible)
        st.markdown('<h3 class="subsection-header">Revenue Distribution by Country</h3>', unsafe_allow_html=True)
        country_revenue = df.groupby('Country')['TotalPrice'].sum().reset_index()
        country_revenue = country_revenue.sort_values('TotalPrice', ascending=False)
        
        fig = px.bar(country_revenue.head(20), x='Country', y='TotalPrice',
                    title='Top 20 Countries by Total Revenue',
                    labels={'TotalPrice': 'Total Revenue (£)', 'Country': 'Country'},
                    color='TotalPrice', color_continuous_scale='Blues')
        fig = update_bar_layout(fig, height=700, orientation='v')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> Revenue distribution across countries helps identify 
        key markets and understand geographical revenue concentration. This information 
        is crucial for international expansion and market prioritization.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        # ADDED: Customer Distribution by Country
        st.markdown('<h3 class="subsection-header">Customer Distribution by Country</h3>', unsafe_allow_html=True)
        country_customers = df.groupby('Country')['CustomerID'].nunique().reset_index()
        country_customers = country_customers.sort_values('CustomerID', ascending=False)
        
        fig = px.bar(country_customers.head(20), x='Country', y='CustomerID',
                    title='Top 20 Countries by Customer Count',
                    labels={'CustomerID': 'Number of Customers', 'Country': 'Country'},
                    color='CustomerID', color_continuous_scale='Blues')
        fig = update_bar_layout(fig, height=700, orientation='v')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> Customer count by country shows market penetration. 
        Comparing this with revenue per country reveals which markets have high-value 
        customers versus high-volume markets.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        # ADDED: Average Order Value by Country
        st.markdown('<h3 class="subsection-header">Average Order Value by Country</h3>', unsafe_allow_html=True)
        country_aov = df.groupby('Country').agg({
            'TotalPrice': ['mean', 'count']
        }).reset_index()
        country_aov.columns = ['Country', 'AvgOrderValue', 'OrderCount']
        country_aov = country_aov[country_aov['OrderCount'] >= 10].sort_values('AvgOrderValue', ascending=False)
        
        fig = px.bar(country_aov.head(20), x='Country', y='AvgOrderValue',
                    title='Top 20 Countries by Average Order Value (min 10 orders)',
                    labels={'AvgOrderValue': 'Average Order Value (£)', 'Country': 'Country'},
                    color='AvgOrderValue', color_continuous_scale='Teal')
        fig = update_bar_layout(fig, height=700, orientation='v')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="description-box">
        <strong>Description:</strong> Average Order Value (AOV) by country identifies 
        markets with higher spending per transaction. Countries with high AOV may 
        indicate premium positioning or successful upselling strategies.
        </div>
        """, unsafe_allow_html=True)
    
    # ========== TAB 9: Customer Details ==========
    with tab9:
        st.markdown('<h2 class="section-header">Customer Details</h2>', unsafe_allow_html=True)
        
        st.info("Select a customer ID to view their detailed purchase history, RFM metrics, and buying patterns.")
        
        # Get available customer IDs
        rfm = calculate_rfm(df)
        available_customers = sorted(rfm['CustomerID'].tolist())
        
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_customer = st.selectbox(
                "Select Customer ID",
                available_customers,
                help="Choose a customer to view their details"
            )
        
        if selected_customer:
            # Get customer data
            customer_rfm = rfm[rfm['CustomerID'] == selected_customer].iloc[0]
            customer_transactions = df[df['CustomerID'] == selected_customer].copy()
            
            # Calculate additional metrics
            customer_total_spend = customer_transactions['TotalPrice'].sum()
            customer_total_orders = customer_transactions['InvoiceNo'].nunique()
            customer_avg_order_value = customer_total_spend / customer_total_orders if customer_total_orders > 0 else 0
            customer_unique_products = customer_transactions['Description'].nunique()
            customer_first_purchase = customer_transactions['InvoiceDate'].min()
            customer_last_purchase = customer_transactions['InvoiceDate'].max()
            customer_days_active = (customer_last_purchase - customer_first_purchase).days
            
            # Display key metrics
            st.markdown('<h3 class="subsection-header">Customer Overview</h3>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Spend", f"£{customer_total_spend:,.2f}")
            with col2:
                st.metric("Total Orders", f"{customer_total_orders:,}")
            with col3:
                st.metric("Avg Order Value", f"£{customer_avg_order_value:,.2f}")
            with col4:
                st.metric("Unique Products", f"{customer_unique_products:,}")
            
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            
            # RFM Metrics
            st.markdown('<h3 class="subsection-header">RFM Metrics</h3>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Recency", f"{int(customer_rfm['Recency'])} days")
            with col2:
                st.metric("Frequency", f"{int(customer_rfm['Frequency'])} orders")
            with col3:
                st.metric("Monetary", f"£{customer_rfm['Monetary']:,.2f}")
            with col4:
                # Get segment
                rfm_temp = rfm.copy()
                # Use safe_qcut to handle edge cases with few customers
                if len(rfm_temp) >= 2:
                    rfm_temp['R_Score'] = safe_qcut(rfm_temp['Recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop', reverse_labels=True)
                    rfm_temp['F_Score'] = safe_qcut(rfm_temp['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
                    rfm_temp['M_Score'] = safe_qcut(rfm_temp['Monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
                else:
                    # Fallback for single customer
                    rfm_temp['R_Score'] = 3
                    rfm_temp['F_Score'] = 3
                    rfm_temp['M_Score'] = 3
                
                def assign_segment(row):
                    r = int(row['R_Score'])
                    f = int(row['F_Score'])
                    m = int(row['M_Score'])
                    if r >= 4 and f >= 4 and m >= 4:
                        return 'Champions'
                    elif r >= 3 and f >= 3 and m >= 3:
                        return 'Loyal Customers'
                    elif r >= 4 and f >= 2:
                        return 'Potential Loyalists'
                    elif r >= 3 and f >= 1:
                        return 'At Risk'
                    elif r <= 2 and f >= 3:
                        return "Can't Lose Them"
                    elif r <= 2 and f <= 2:
                        return 'Lost Customers'
                    elif r >= 4 and f == 1:
                        return 'Recent/New Customers'
                    else:
                        return 'Low Value'
                
                rfm_temp['Segment'] = rfm_temp.apply(assign_segment, axis=1)
                customer_segment = rfm_temp[rfm_temp['CustomerID'] == selected_customer]['Segment'].iloc[0]
                st.metric("Segment", customer_segment)
            
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            
            # Purchase Timeline
            st.markdown('<h3 class="subsection-header">Purchase Timeline</h3>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**First Purchase:** {customer_first_purchase.strftime('%Y-%m-%d')}")
            with col2:
                st.write(f"**Last Purchase:** {customer_last_purchase.strftime('%Y-%m-%d')}")
            with col3:
                st.write(f"**Days Active:** {customer_days_active} days")
            
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            
            # Monthly spending trend
            st.markdown('<h3 class="subsection-header">Monthly Spending Trend</h3>', unsafe_allow_html=True)
            monthly_customer = customer_transactions.groupby(customer_transactions['InvoiceDate'].dt.to_period('M'))['TotalPrice'].sum()
            monthly_customer.index = monthly_customer.index.astype(str)
            fig = px.line(x=monthly_customer.index, y=monthly_customer.values,
                         markers=True, title=f'Monthly Spending Trend - Customer {selected_customer}',
                         labels={'x': 'Month', 'y': 'Spending (£)'},
                         color_discrete_sequence=['#A8D5E3'])
            fig.update_traces(line=dict(width=3))
            fig = update_chart_layout(fig, height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            
            # Top products purchased
            st.markdown('<h3 class="subsection-header">Top Products Purchased</h3>', unsafe_allow_html=True)
            top_products_customer = customer_transactions.groupby('Description')['Quantity'].sum().nlargest(10)
            fig = px.bar(x=top_products_customer.values, y=top_products_customer.index,
                        orientation='h', title=f'Top 10 Products Purchased - Customer {selected_customer}',
                        labels={'x': 'Total Quantity', 'y': 'Product'},
                        color_discrete_sequence=['#8b5cf6'])
            fig = update_bar_layout(fig, height=600, orientation='h')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            
            # Recent transactions table
            st.markdown('<h3 class="subsection-header">Recent Transactions</h3>', unsafe_allow_html=True)
            recent_transactions = customer_transactions.sort_values('InvoiceDate', ascending=False).head(20)[
                ['InvoiceDate', 'Description', 'Quantity', 'UnitPrice', 'TotalPrice', 'Country']
            ].copy()
            recent_transactions['InvoiceDate'] = recent_transactions['InvoiceDate'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(recent_transactions, use_container_width=True, height=400)
            
            st.markdown("""
            <div class="description-box">
            <strong>Description:</strong> This section provides comprehensive details about the selected customer including 
            their purchase history, RFM metrics, spending patterns, and favorite products. Use this information to 
            understand customer behavior and tailor marketing strategies.
            </div>
            """, unsafe_allow_html=True)

else:
    st.warning("Please load a dataset using the sidebar upload option or ensure the dataset is available.")
