import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

CURRENT_YEAR = 2026

# -----------------------------------------------------------------------------
# 1. REAL INDIAN CAR MODEL DATABASE  (ex-showroom price ₹ Lakh, engine cc, fuels)
# -----------------------------------------------------------------------------
MODEL_DATABASE = {
    "Maruti Suzuki": {
        "Alto K10":        (4.09,  998,  ["Petrol", "CNG"]),
        "S-Presso":        (4.26,  998,  ["Petrol", "CNG"]),
        "Wagon R":         (5.54,  1197, ["Petrol", "CNG"]),
        "Celerio":         (5.64,  998,  ["Petrol", "CNG"]),
        "Swift":           (6.49,  1197, ["Petrol", "CNG"]),
        "Baleno":          (6.66,  1197, ["Petrol", "CNG"]),
        "Fronx":           (7.46,  1197, ["Petrol", "CNG"]),
        "Dzire":           (6.79,  1197, ["Petrol", "CNG"]),
        "Brezza":          (8.34,  1462, ["Petrol", "CNG"]),
        "Ertiga":          (8.69,  1462, ["Petrol", "CNG"]),
        "Ciaz":            (9.30,  1462, ["Petrol"]),
        "Grand Vitara":    (11.02, 1462, ["Petrol", "Hybrid"]),
    },
    "Hyundai": {
        "Grand i10 Nios":  (5.92,  1197, ["Petrol", "CNG"]),
        "Exter":           (6.00,  1197, ["Petrol", "CNG"]),
        "i20":             (7.04,  1197, ["Petrol"]),
        "Aura":            (6.48,  1197, ["Petrol", "CNG"]),
        "Venue":           (7.89,  1197, ["Petrol", "Diesel"]),
        "Verna":           (9.42,  1482, ["Petrol"]),
        "Creta":           (10.87, 1497, ["Petrol", "Diesel"]),
        "Alcazar":         (17.00, 1499, ["Petrol", "Diesel"]),
        "Tucson":          (30.05, 1999, ["Petrol", "Diesel"]),
    },
    "Tata": {
        "Tiago":           (5.64,  1199, ["Petrol", "CNG", "EV"]),
        "Tigor":           (7.19,  1199, ["Petrol", "CNG", "EV"]),
        "Altroz":          (6.99,  1199, ["Petrol", "Diesel", "CNG"]),
        "Punch":           (6.12,  1199, ["Petrol", "CNG", "EV"]),
        "Nexon":           (7.99,  1199, ["Petrol", "Diesel", "EV"]),
        "Harrier":         (14.99, 1956, ["Petrol", "Diesel"]),
        "Safari":          (15.99, 1956, ["Petrol", "Diesel"]),
    },
    "Mahindra": {
        "Bolero":          (9.89,  1498, ["Diesel"]),
        "XUV 3XO":         (7.99,  1197, ["Petrol", "Diesel"]),
        "Thar":            (11.34, 1497, ["Petrol", "Diesel"]),
        "Scorpio-N":       (13.59, 1997, ["Petrol", "Diesel"]),
        "XUV700":          (13.99, 1999, ["Petrol", "Diesel"]),
        "XUV400 EV":       (15.49, 0,    ["EV"]),
    },
    "Toyota": {
        "Glanza":          (6.98,  1197, ["Petrol", "CNG"]),
        "Taisor":          (7.73,  1197, ["Petrol", "CNG"]),
        "Urban Cruiser Hyryder": (11.14, 1462, ["Petrol", "Hybrid"]),
        "Innova Crysta":   (19.99, 2393, ["Petrol", "Diesel"]),
        "Innova HyCross":  (25.21, 1987, ["Petrol", "Hybrid"]),
        "Fortuner":        (33.43, 2694, ["Petrol", "Diesel"]),
        "Hilux":           (30.40, 2393, ["Diesel"]),
        "Camry":           (48.00, 2487, ["Hybrid"]),
    },
    "Honda": {
        "Amaze":           (7.57,  1199, ["Petrol"]),
        "City":            (11.82, 1498, ["Petrol"]),
        "Elevate":         (11.73, 1498, ["Petrol"]),
    },
    "Kia": {
        "Syros":           (9.00,  1197, ["Petrol", "Diesel"]),
        "Sonet":           (7.99,  1197, ["Petrol", "Diesel"]),
        "Seltos":          (10.89, 1497, ["Petrol", "Diesel"]),
        "Carens":          (10.45, 1497, ["Petrol", "Diesel"]),
        "Carnival":        (32.00, 2151, ["Diesel"]),
        "EV6":             (65.00, 0,    ["EV"]),
    },
    "Volkswagen": {
        "Virtus":          (11.56, 999,  ["Petrol"]),
        "Taigun":          (11.70, 999,  ["Petrol"]),
        "Tiguan":          (35.17, 1984, ["Petrol"]),
    },
    "Skoda": {
        "Slavia":          (11.53, 999,  ["Petrol"]),
        "Kushaq":          (11.89, 999,  ["Petrol"]),
        "Kodiaq":          (39.99, 1984, ["Petrol"]),
    },
    "MG": {
        "Astor":           (9.98,  1349, ["Petrol"]),
        "Hector":          (13.99, 1451, ["Petrol", "Diesel"]),
        "ZS EV":           (18.98, 0,    ["EV"]),
        "Windsor EV":      (13.99, 0,    ["EV"]),
        "Gloster":         (37.50, 1996, ["Diesel"]),
    },
    "Jeep": {
        "Compass":         (20.69, 1368, ["Petrol", "Diesel"]),
        "Meridian":        (33.00, 1956, ["Diesel"]),
    },
    "BMW": {
        "3 Series":        (61.90, 1998, ["Petrol", "Diesel"]),
        "5 Series":        (74.90, 1998, ["Petrol", "Diesel"]),
        "X1":              (49.00, 1499, ["Petrol", "Diesel"]),
        "X3":              (68.50, 1998, ["Petrol", "Diesel"]),
    },
    "Audi": {
        "A4":              (46.99, 1984, ["Petrol"]),
        "Q3":              (43.99, 1984, ["Petrol"]),
    },
    "Mercedes-Benz": {
        "C-Class":         (59.00, 1496, ["Petrol"]),
        "E-Class":         (78.50, 1993, ["Petrol"]),
        "GLC":             (74.50, 1993, ["Petrol", "Diesel"]),
    },
}

# Resale-strength by brand (lower = holds value better, Indian used-car market)
BRAND_RETENTION = {
    "Maruti Suzuki": 0.90, "Toyota": 0.80, "Honda": 0.88,
    "Mahindra": 0.95, "Hyundai": 0.95, "Kia": 0.95,
    "Tata": 1.00, "Volkswagen": 1.02, "Skoda": 1.02,
    "MG": 1.05, "Jeep": 1.05, "BMW": 1.10, "Audi": 1.10,
    "Mercedes-Benz": 1.08,
}

# -----------------------------------------------------------------------------
# 2. REALISTIC MARKET VALUATION ENGINE  (age curve + km + owners + fuel)
# -----------------------------------------------------------------------------
def depreciation_factor(age, brand):
    """Compound on-road depreciation for 'age' years, adjusted for brand resale strength."""
    rf = BRAND_RETENTION.get(brand, 1.0)
    dep = 1.0
    for y in range(1, age + 1):
        if y == 1:    rate = 0.13
        elif y == 2:  rate = 0.11
        elif y <= 5:  rate = 0.085
        elif y <= 10: rate = 0.06
        else:         rate = 0.045
        dep *= (1 - rate * rf)
    return dep


def compute_market_value(ex_showroom, age, kms, owners, fuel, engine_cc, brand):
    """Estimated current used-car market value in ₹ Lakh (Indian market logic)."""
    dep = depreciation_factor(age, brand)
    expected_km = 14000 if fuel == "Diesel" else 11000
    km_penalty = max(0.0, kms - age * expected_km) * 0.00003   # ~₹0.3L per 10k km over-use
    owner_pen = {1: 0.0, 2: 0.04, 3: 0.09, 4: 0.14}[owners]
    fuel_adj = {"Petrol": 1.0,
                "Diesel": 1.02 if engine_cc >= 1500 else 1.0,
                "CNG": 0.96, "EV": 0.94, "Hybrid": 1.06}[fuel]
    value = ex_showroom * dep * (1 - owner_pen) * fuel_adj - km_penalty
    return max(0.35, value)


# -----------------------------------------------------------------------------
# 3. TRAINING DATA GENERATION (built on the real model database)
# -----------------------------------------------------------------------------
def generate_training_data(n_samples=8000, seed=42):
    rng = np.random.default_rng(seed)
    brands = list(MODEL_DATABASE.keys())
    rows = []
    for _ in range(n_samples):
        brand = brands[rng.integers(len(brands))]
        model_name = list(MODEL_DATABASE[brand].keys())[rng.integers(len(MODEL_DATABASE[brand]))]
        ex_showroom, engine_cc, fuels = MODEL_DATABASE[brand][model_name]
        fuel = fuels[rng.integers(len(fuels))]
        year = int(rng.integers(2010, CURRENT_YEAR + 1))
        age = CURRENT_YEAR - year
        if year == CURRENT_YEAR:
            kms = int(rng.integers(500, 9000))
            owners = 1
        else:
            kms = int(rng.integers(6, 22) * 1000 * age)
            if age <= 2:       owners = 1
            elif age <= 5:     owners = int(rng.choice([1, 2], p=[0.75, 0.25]))
            elif age <= 8:     owners = int(rng.choice([1, 2, 3], p=[0.50, 0.40, 0.10]))
            else:              owners = int(rng.choice([1, 2, 3, 4], p=[0.35, 0.40, 0.20, 0.05]))
        value = compute_market_value(ex_showroom, max(age, 1), kms, owners, fuel, engine_cc, brand)
        value *= (1 + rng.normal(0, 0.02))
        rows.append([age, kms, engine_cc, ex_showroom, owners, fuel, max(0.35, value)])
    df = pd.DataFrame(rows, columns=["age", "kms", "engine_cc", "ex_showroom", "owners", "fuel", "price"])
    fuel_dummies = pd.get_dummies(df["fuel"], prefix="fuel").astype(float)
    X = pd.concat([df[["age", "kms", "engine_cc", "ex_showroom", "owners"]], fuel_dummies], axis=1)
    return df, X, df["price"].astype(float)


# -----------------------------------------------------------------------------
# 4. MODEL TRAINING
# -----------------------------------------------------------------------------
@st.cache_resource
def train_valuation_engine():
    df, X, y = generate_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    model = RandomForestRegressor(n_estimators=250, max_depth=16, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    test_r2 = r2_score(y_test, model.predict(X_test))
    return model, list(X.columns), test_r2


def predict_price(model, feature_cols, age, kms, engine_cc, ex_showroom, owners, fuel):
    row = pd.DataFrame([[age, kms, engine_cc, ex_showroom, owners]],
                       columns=["age", "kms", "engine_cc", "ex_showroom", "owners"])
    for c in feature_cols:
        if c.startswith("fuel_"):
            row[c] = 0.0
    row[f"fuel_{fuel}"] = 1.0
    return float(model.predict(row[feature_cols])[0])


# -----------------------------------------------------------------------------
# 4b. LIVE CAR IMAGE (Wikipedia API - free, no API key required)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def get_car_image(brand, model):
    """Fetch the car's photo from Wikipedia. Returns image URL or None."""
    queries = [f"{brand} {model} car", f"{model} car"]
    headers = {"User-Agent": "AutoValuateAI/3.1 (educational project)"}
    for q in queries:
        try:
            r = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": q,
                        "srlimit": 1, "format": "json"},
                timeout=8, headers=headers)
            hits = r.json().get("query", {}).get("search", [])
            if not hits:
                continue
            r2 = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "titles": hits[0]["title"],
                        "prop": "pageimages", "pithumbsize": 800, "format": "json"},
                timeout=8, headers=headers)
            for p in r2.json().get("query", {}).get("pages", {}).values():
                if "thumbnail" in p:
                    return p["thumbnail"]["source"]
        except Exception:
            continue
    return None


# -----------------------------------------------------------------------------
# 4c. AI ADVISOR ENGINE (rule-based insights, works offline)
# -----------------------------------------------------------------------------
def generate_advice(brand, model, fuel, age, kms, owners, value, ex_showroom):
    """Auto-generated buying / selling / negotiation advice from valuation data."""
    cc = MODEL_DATABASE[brand][model][1]
    yearly_km = kms / max(age, 1)
    expected = compute_market_value(ex_showroom, max(age, 1), kms, owners, fuel, cc, brand)
    tips = []

    if value <= expected * 0.95:
        tips.append("🟢 **Deal verdict:** Below typical market value — this looks like a *good deal*.")
    elif value >= expected * 1.05:
        tips.append("🔴 **Deal verdict:** Above typical market range — *negotiate harder* or compare more listings.")
    else:
        tips.append("🟡 **Deal verdict:** Within the normal market range — fair if the condition checks out.")

    if yearly_km > 15000:
        tips.append(f"🔧 **High usage:** {yearly_km:,.0f} km/year (avg is ~10–12k). Use this as leverage to negotiate down.")
    elif yearly_km < 7000 and age > 0:
        tips.append(f"✨ **Low usage:** only {yearly_km:,.0f} km/year — a lightly driven car, slight premium is justified.")

    if owners >= 3:
        tips.append(f"👥 **Multiple owners:** {owners} owners raises transfer & maintenance risk — insist on full service records.")
    elif owners == 2:
        tips.append("👥 **Second owner:** normal, but verify the RC transfer history before paying.")

    if age >= 10:
        tips.append("🛠️ **Age:** 10+ year old car — budget for upcoming major service (suspension, tyres, belts).")

    if fuel == "EV":
        tips.append("🔋 **EV check:** ask for a battery State-of-Health (SOH) report — battery is 30–40% of the car's value.")
    if fuel == "Diesel" and age >= 8:
        tips.append("⚠️ **Diesel rule:** 10+ year old diesels are banned in Delhi-NCR — resale will be region-restricted.")

    rf = BRAND_RETENTION.get(brand, 1.0)
    if rf <= 0.85:
        tips.append("📈 **Resale strength:** this brand holds value exceptionally well — low risk of a bad resale later.")
    elif rf >= 1.08:
        tips.append("📉 **Depreciation note:** luxury cars depreciate steeply — a *certified pre-owned* unit from an authorized dealer is safer.")

    if value / ex_showroom < 0.35:
        tips.append("💰 **Depreciated deep:** the car has already taken its biggest value hits — from here, depreciation is slow. Good long-term hold.")

    tips.append(f"🤝 **Negotiation band:** if buying, open at **₹{value * 0.93:,.2f}L**; if selling, list at **₹{value * 1.06:,.2f}L** and settle near ₹{value:,.2f}L.")
    return tips


# -----------------------------------------------------------------------------
# 4d. LLM CHAT ASSISTANT (optional - Google Gemini free API key)
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are AutoValuate AI, an expert Indian used-car valuation assistant. "
    "Give short, practical advice on pricing, negotiation, inspection and resale. "
    "Use INR Lakh for prices. Be honest about uncertainty."
)


def ask_gemini(api_key, car_context, history, question):
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-2.0-flash:generateContent?key={api_key}")
    contents = [{"role": "user", "parts": [{"text": car_context}]}]
    for m in history[-6:]:
        contents.append({"role": "user" if m["role"] == "user" else "model",
                         "parts": [{"text": m["content"]}]})
    contents.append({"role": "user", "parts": [{"text": question}]})
    payload = {"system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
               "contents": contents}
    r = requests.post(url, json=payload, timeout=30)
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


# -----------------------------------------------------------------------------
# 5. STREAMLIT UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="AutoValuate AI | Real-Model Car Valuation",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
        .stApp { background-color: #0b0f19; color: #e6edf3;
                 font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
        .metric-card {
            background: linear-gradient(145deg, rgba(22,27,34,0.9), rgba(13,17,23,0.95));
            border: 1px solid #30363d; border-radius: 14px; padding: 22px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5); text-align: center;
            transition: transform 0.2s ease, border-color 0.2s ease; }
        .metric-card:hover { border-color: #58a6ff; transform: translateY(-3px); }
        .metric-value { font-size: 2.4rem; font-weight: 800;
            background: linear-gradient(90deg, #58a6ff, #1f6feb);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin: 10px 0 4px 0; }
        .metric-label { font-size: 0.85rem; text-transform: uppercase;
            letter-spacing: 1.5px; color: #8b949e; font-weight: 600; }
        .main-title { font-size: 2.6rem; font-weight: 900;
            background: linear-gradient(90deg, #ffffff, #8b949e);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            letter-spacing: -1px; }
        .founder-card { background: rgba(22,27,34,0.6); border: 1px solid #30363d;
            border-radius: 10px; padding: 12px 18px; margin-top: 10px; margin-bottom: 20px; }
        .founder-name { font-weight: 700; color: #58a6ff; font-size: 0.95rem; }
        .sidebar-header { font-size: 1.1rem; font-weight: 700; color: #58a6ff;
            letter-spacing: 1px; text-transform: uppercase; margin-bottom: 15px;
            border-bottom: 1px solid #30363d; padding-bottom: 8px; }
    </style>
    """, unsafe_allow_html=True)

    with st.spinner("⚡ Loading real-model valuation engine..."):
        model, feature_cols, test_r2 = train_valuation_engine()

    # ---------------- Sidebar ----------------
    st.sidebar.markdown('<div class="sidebar-header">⚡ Vehicle Inputs</div>', unsafe_allow_html=True)

    brand = st.sidebar.selectbox("Brand", list(MODEL_DATABASE.keys()))
    model_name = st.sidebar.selectbox("Model", list(MODEL_DATABASE[brand].keys()))
    ex_showroom, engine_cc, available_fuels = MODEL_DATABASE[brand][model_name]

    fuel = st.sidebar.radio("Fuel", available_fuels, horizontal=True)
    registration_year = st.sidebar.slider("Registration Year", 2010, CURRENT_YEAR, 2021)
    age = max(0, CURRENT_YEAR - registration_year)
    kms_driven = st.sidebar.number_input("Kilometers Driven", min_value=500,
                                         max_value=400000, value=42000, step=2500)
    owners = st.sidebar.radio("Previous Owners", [1, 2, 3, 4], horizontal=True,
                              disabled=(age == 0), index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**👥 Project Co-Founders**")
    st.sidebar.caption("• **Kushagra Dubey**\n• **Shreyansh Pandey**\n• **Shivam Dubey**")
    st.sidebar.markdown("---")
    st.sidebar.caption(f"🤖 RandomForest v3.2 · Real-model DB · Live images · Test R² = {test_r2:.1%}")
    gemini_key = st.sidebar.text_input("🔑 Gemini API key (optional, enables chat)",
                                       type="password", placeholder="Get a free key at aistudio.google.com")

    # ---------------- Prediction ----------------
    predicted_valuation = predict_price(model, feature_cols, age, kms_driven,
                                        engine_cc, ex_showroom, owners, fuel)

    # ---------------- Dashboard ----------------
    st.markdown('<div class="main-title">🚗 AutoValuate AI Dashboard</div>', unsafe_allow_html=True)
    st.caption("Real-Model Machine Learning Vehicle Pricing & Depreciation Analytics")

    st.markdown("""
    <div class="founder-card">
        <span style="color: #8b949e; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;">Project Co-Founders & Lead Engineers:</span><br>
        <span class="founder-name">⚡ Kushagra Dubey &nbsp;|&nbsp; Shreyansh Pandey &nbsp;|&nbsp; Shivam Dubey</span>
    </div>
    """, unsafe_allow_html=True)

    img_url = get_car_image(brand, model_name)
    img_col, info_col = st.columns([1, 2.4])
    with img_col:
        if img_url:
            st.image(img_url, use_container_width=True,
                     caption=f"{brand} {model_name} (image: Wikipedia)")
        else:
            st.markdown('<div style="background:#161b22;border:1px solid #30363d;border-radius:12px;'
                        'height:230px;display:flex;align-items:center;justify-content:center;'
                        'font-size:3.5rem;">🚗</div>', unsafe_allow_html=True)
    with info_col:
        st.markdown(f"### {brand} {model_name}")
        st.markdown(f"**Fuel:** {fuel} &nbsp;·&nbsp; **Engine:** "
                    f"{str(engine_cc) + ' cc' if engine_cc else 'Electric motor'}")
        st.markdown(f"**Registration:** {registration_year} ({age} yr old) &nbsp;·&nbsp; "
                    f"**KMs:** {kms_driven:,} km &nbsp;·&nbsp; **Owners:** {owners}")
        st.markdown(f"**Current ex-showroom:** ₹{ex_showroom:,.2f} Lakh &nbsp;·&nbsp; "
                    f"**Est. on-road:** ₹{ex_showroom * 1.12:,.2f} Lakh")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Estimated Market Value</div>
            <div class="metric-value">₹{predicted_valuation:,.2f} Lakh</div>
            <div style="font-size:0.82rem; color:#3fb950; font-weight:600;">Model R² (test): {test_r2:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Fair Trade-in Bounds</div>
            <div class="metric-value">₹{predicted_valuation*0.93:,.2f}L - ₹{predicted_valuation*1.07:,.2f}L</div>
            <div style="font-size:0.82rem; color:#8b949e;">Market Variation Range (±7%)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        dep_pct = max(0.0, (1 - predicted_valuation / ex_showroom) * 100)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Depreciation</div>
            <div class="metric-value">-{dep_pct:.1f}%</div>
            <div style="font-size:0.82rem; color:#f85149;">vs ex-showroom ₹{ex_showroom:.2f}L</div>
        </div>
        """, unsafe_allow_html=True)

    c4, c5 = st.columns(2)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Ex-Showroom Price</div>
            <div class="metric-value">₹{ex_showroom:,.2f} Lakh</div>
            <div style="font-size:0.82rem; color:#8b949e;">Latest official price (model database)</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Estimated On-Road Price</div>
            <div class="metric-value">₹{ex_showroom * 1.12:,.2f} Lakh</div>
            <div style="font-size:0.82rem; color:#8b949e;">Incl. RTO registration, insurance & taxes (+12%)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- Charts ----------------
    chart_left, chart_right = st.columns([1.2, 0.8])

    with chart_left:
        st.subheader("📈 Future Resale Forecast")
        future_years = [CURRENT_YEAR + i for i in range(6)]
        future_rates = [0.10, 0.10, 0.09, 0.08, 0.07]
        curve, v = [], predicted_valuation
        for r in future_rates:
            v *= (1 - r)
            curve.append(v)
        curve = [predicted_valuation] + curve

        fig_resale = go.Figure()
        fig_resale.add_trace(go.Scatter(
            x=future_years, y=curve, mode="lines+markers", name="Resale Value",
            line=dict(color="#58a6ff", width=3.5),
            marker=dict(size=9, color="#1f6feb", symbol="circle")))
        fig_resale.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(gridcolor="#21262d", title="Model Projection Year"),
            yaxis=dict(gridcolor="#21262d", title="Price (₹ Lakh)"), height=340)
        st.plotly_chart(fig_resale, use_container_width=True)

    with chart_right:
        st.subheader("📊 Feature Importance Impact")
        features_list = ["Vehicle Age", "KMs Driven", "Engine CC",
                         "Ex-Showroom Price", "Owners", "Fuel Type (one-hot)"][:len(model.feature_importances_)]
        importance_df = pd.DataFrame({
            "Feature": ["Vehicle Age", "KMs Driven", "Engine CC", "Ex-Showroom Price", "Owners"]
                        + [c for c in feature_cols if c.startswith("fuel_")],
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=True)

        fig_imp = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Blues")
        fig_imp.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=20, b=20),
            coloraxis_showscale=False, height=340)
        st.plotly_chart(fig_imp, use_container_width=True)

    # ---------------- AI Assistant ----------------
    st.divider()
    st.subheader("🤖 AutoValuate AI Assistant")

    with st.expander("📋 Instant smart advice for this car", expanded=True):
        for tip in generate_advice(brand, model_name, fuel, age, kms_driven,
                                   owners, predicted_valuation, ex_showroom):
            st.markdown(f"- {tip}")

    if gemini_key:
        st.markdown("###### 💬 Chat with the AI assistant about this car")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        car_context = (f"Car being evaluated: {brand} {model_name}, {fuel}, "
                       f"{age} years old, {kms_driven:,} km, {owners} owner(s). "
                       f"Ex-showroom ₹{ex_showroom:.2f}L. Estimated current market value "
                       f"₹{predicted_valuation:.2f}L. Test R²={test_r2:.1%}.")
        for m in st.session_state.chat_history:
            st.chat_message(m["role"]).write(m["content"])
        question = st.chat_input("e.g. Is this a fair price? What should I check before buying?")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.spinner("Thinking..."):
                try:
                    answer = ask_gemini(gemini_key, car_context,
                                        st.session_state.chat_history[:-1], question)
                except Exception as e:
                    answer = (f"⚠️ Could not reach the Gemini API ({e}). "
                              "Check that the API key is valid and has quota.")
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()
    else:
        st.caption("💡 Add a free Gemini API key in the sidebar to unlock the conversational "
                   "chat assistant (aistudio.google.com → Get API key).")


if __name__ == "__main__":
    main()
