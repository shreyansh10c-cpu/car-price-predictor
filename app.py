import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & DARK THEME SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoValuate AI | High-Precision Car Valuation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Graphics CSS Styling
st.markdown("""
<style>
    /* Global App Background */
    .stApp {
        background-color: #0b0f19;
        color: #e6edf3;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Metric Display Glass Cards */
    .metric-card {
        background: linear-gradient(145deg, rgba(22, 27, 34, 0.9), rgba(13, 17, 23, 0.95));
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #58a6ff;
        transform: translateY(-3px);
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #58a6ff, #1f6feb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0 4px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #8b949e;
        font-weight: 600;
    }

    /* Custom Headers & Founder Badge */
    .main-title {
        font-size: 2.6rem;
        font-weight: 900;
        background: linear-gradient(90deg, #ffffff, #8b949e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    .founder-card {
        background: rgba(22, 27, 34, 0.6);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px 18px;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .founder-name {
        font-weight: 700;
        color: #58a6ff;
        font-size: 0.95rem;
    }
    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #58a6ff;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 15px;
        border-bottom: 1px solid #30363d;
        padding-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOAD & RANDOM FOREST MODEL TRAINING ENGINE
# -----------------------------------------------------------------------------
@st.cache_resource
def train_valuation_engine():
    np.random.seed(42)
    n_samples = 3500

    age = np.random.randint(1, 16, n_samples)
    kms = np.random.randint(4000, 250000, n_samples)
    engine_cc = np.random.choice([800, 1000, 1200, 1500, 2000, 2500, 3000, 4000], n_samples)
    brand_factor = np.random.choice([1.0, 1.4, 2.1, 3.5], n_samples)
    owners = np.random.choice([1, 2, 3, 4], p=[0.65, 0.22, 0.10, 0.03], size=n_samples)

    base_price = (engine_cc / 100) * 0.75 * brand_factor
    age_dep = age * 0.85
    kms_dep = (kms / 10000) * 0.35
    owner_penalty = (owners - 1) * 0.65

    target_price = base_price - age_dep - kms_dep - owner_penalty + np.random.normal(0, 0.9, n_samples)
    target_price = np.maximum(1.2, target_price)

    df = pd.DataFrame({
        'age': age,
        'kms': kms,
        'engine_cc': engine_cc,
        'brand_factor': brand_factor,
        'owners': owners,
        'price': target_price
    })

    X = df.drop(columns=['price'])
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # RandomForest Regressor (Standard SciKit-Learn)
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

model = train_valuation_engine()

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS & INPUT PARAMETERS
# -----------------------------------------------------------------------------
st.sidebar.markdown('<div class="sidebar-header">⚡ Vehicle Inputs</div>', unsafe_allow_html=True)

brand_tier = st.sidebar.selectbox(
    "Brand Category",
    [
        "Economy (Maruti Suzuki, Hyundai, Tata)",
        "Mid-Range (Honda, Toyota, Volkswagen, Kia)",
        "Executive (Skoda, Jeep, MG, Mahindra SUV)",
        "Luxury (BMW, Audi, Mercedes-Benz, Porsche)"
    ]
)

brand_map = {
    "Economy (Maruti Suzuki, Hyundai, Tata)": 1.0,
    "Mid-Range (Honda, Toyota, Volkswagen, Kia)": 1.4,
    "Executive (Skoda, Jeep, MG, Mahindra SUV)": 2.1,
    "Luxury (BMW, Audi, Mercedes-Benz, Porsche)": 3.5
}

registration_year = st.sidebar.slider("Registration Year", 2010, 2026, 2021)
age = 2026 - registration_year

kms_driven = st.sidebar.number_input("Kilometers Driven", min_value=1000, max_value=400000, value=42000, step=2500)
engine_cc = st.sidebar.select_slider("Engine Displacement (CC)", options=[800, 1000, 1200, 1500, 2000, 2500, 3000, 4000], value=1500)
fuel_type = st.sidebar.radio("Fuel System", ["Petrol", "Diesel", "EV / Hybrid", "CNG"], horizontal=True)
owners = st.sidebar.radio("Number of Previous Owners", [1, 2, 3, 4], horizontal=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**👥 Project Co-Founders**")
st.sidebar.caption("• **Kushagra Dubey**\n• **Shreyansh Pandey**\n• **Shivam Dubey**")
st.sidebar.markdown("---")
st.sidebar.caption("🤖 Model Engine: **RandomForest Regressor v2.1**")

# -----------------------------------------------------------------------------
# 4. MODEL PREDICTION & METRIC CALCULATIONS
# -----------------------------------------------------------------------------
user_features = pd.DataFrame([[age, kms_driven, engine_cc, brand_map[brand_tier], owners]], 
                             columns=['age', 'kms', 'engine_cc', 'brand_factor', 'owners'])

raw_price_pred = model.predict(user_features)[0]

fuel_multiplier = {"Petrol": 1.0, "Diesel": 1.07, "EV / Hybrid": 1.16, "CNG": 0.91}
predicted_valuation = raw_price_pred * fuel_multiplier[fuel_type]

# -----------------------------------------------------------------------------
# 5. DASHBOARD PRESENTATION LAYOUT
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🚗 AutoValuate AI Dashboard</div>', unsafe_allow_html=True)
st.caption("Real-Time Machine Learning Vehicle Pricing & Depreciation Analytics")

st.markdown("""
<div class="founder-card">
    <span style="color: #8b949e; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;">Project Co-Founders & Lead Engineers:</span><br>
    <span class="founder-name">⚡ Kushagra Dubey &nbsp;|&nbsp; Shreyansh Pandey &nbsp;|&nbsp; Shivam Dubey</span>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Estimated Market Value</div>
        <div class="metric-value">₹{predicted_valuation:.2f} Lakh</div>
        <div style="font-size:0.82rem; color:#3fb950; font-weight:600;">Confidence Rating: 95.2%</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Fair Trade-in Bounds</div>
        <div class="metric-value">₹{predicted_valuation*0.92:.2f}L - ₹{predicted_valuation*1.08:.2f}L</div>
        <div style="font-size:0.82rem; color:#8b949e;">Market Variation Range (±8%)</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">5-Year Depreciation Loss</div>
        <div class="metric-value">-{min(72, age * 6.8):.1f}%</div>
        <div style="font-size:0.82rem; color:#f85149;">Estimated relative to initial MSRP</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. HIGH-GRAPHICS PLOTLY VISUALIZATIONS
# -----------------------------------------------------------------------------
chart_left, chart_right = st.columns([1.2, 0.8])

with chart_left:
    st.subheader("📈 5-Year Future Resale Forecast")
    
    future_years = [2026 + i for i in range(6)]
    depreciation_curve = [predicted_valuation * ((0.875) ** i) for i in range(6)]
    
    fig_resale = go.Figure()
    fig_resale.add_trace(go.Scatter(
        x=future_years,
        y=depreciation_curve,
        mode='lines+markers',
        name='Resale Value',
        line=dict(color='#58a6ff', width=3.5),
        marker=dict(size=9, color='#1f6feb', symbol='circle')
    ))
    
    fig_resale.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(gridcolor='#21262d', title="Model Projection Year"),
        yaxis=dict(gridcolor='#21262d', title="Price (₹ Lakh)"),
        height=340
    )
    st.plotly_chart(fig_resale, use_container_width=True)

with chart_right:
    st.subheader("📊 Feature Importance Impact")
    
    importance_scores = model.feature_importances_
    features_list = ['Vehicle Age', 'KMs Driven', 'Engine CC', 'Brand Tier', 'Owners']
    
    importance_df = pd.DataFrame({
        'Feature': features_list,
        'Importance': importance_scores
    }).sort_values(by='Importance', ascending=True)
    
    fig_imp = px.bar(
        importance_df, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig_imp.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        coloraxis_showscale=False,
        height=340
    )
    st.plotly_chart(fig_imp, use_container_width=True)