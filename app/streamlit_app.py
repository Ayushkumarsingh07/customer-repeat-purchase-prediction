"""
Customer Repeat-Purchase Predictor
Streamlit app for the Olist e-commerce repeat-purchase prediction project.

Run locally with:  streamlit run streamlit_app.py
Requires these files in the same folder:
  - churn_model.pkl
  - category_encoding.json
  - state_encoding.json
  - global_mean.json
  - feature_columns.json
"""

import json
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(
    page_title="Repeat Purchase Predictor",
    page_icon="🛒",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load model + lookups (cached so this only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(base / "churn_model.pkl")
    with open(base / "category_encoding.json") as f:
        category_encoding = json.load(f)
    with open(base / "state_encoding.json") as f:
        state_encoding = json.load(f)
    with open(base / "global_mean.json") as f:
        global_mean = json.load(f)["global_mean"]
    with open(base / "feature_columns.json") as f:
        feature_columns = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, category_encoding, state_encoding, global_mean, feature_columns, explainer


model, category_encoding, state_encoding, global_mean, feature_columns, explainer = load_artifacts()

CATEGORY_LABELS = {
    "bed_bath_table": "Bed, Bath & Table",
    "health_beauty": "Health & Beauty",
    "sports_leisure": "Sports & Leisure",
    "computers_accessories": "Computers & Accessories",
    "furniture_decor": "Furniture & Decor",
    "housewares": "Housewares",
    "watches_gifts": "Watches & Gifts",
    "telephony": "Telephony",
    "auto": "Auto",
    "toys": "Toys",
    "cool_stuff": "Cool Stuff",
    "garden_tools": "Garden Tools",
    "perfumery": "Perfumery",
    "baby": "Baby",
    "electronics": "Electronics",
    "stationery": "Stationery",
    "fashion_bags_accessories": "Fashion Bags & Accessories",
    "pet_shop": "Pet Shop",
    "office_furniture": "Office Furniture",
    "consoles_games": "Consoles & Games",
    "luggage_accessories": "Luggage & Accessories",
    "other": "Other",
    "unknown": "Unknown",
}

STATE_LABELS = {
    "SP": "São Paulo", "RJ": "Rio de Janeiro", "MG": "Minas Gerais",
    "RS": "Rio Grande do Sul", "PR": "Paraná", "SC": "Santa Catarina",
    "BA": "Bahia", "ES": "Espírito Santo", "GO": "Goiás", "DF": "Distrito Federal",
    "PE": "Pernambuco", "CE": "Ceará", "PA": "Pará", "MT": "Mato Grosso",
    "MA": "Maranhão", "MS": "Mato Grosso do Sul", "PB": "Paraíba",
    "PI": "Piauí", "RN": "Rio Grande do Norte", "AL": "Alagoas",
    "SE": "Sergipe", "TO": "Tocantins", "RO": "Rondônia", "AM": "Amazonas",
    "AC": "Acre", "AP": "Amapá", "RR": "Roraima",
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🛒 Customer Repeat-Purchase Predictor")
st.markdown(
    """
Predicts the probability that a **first-time Olist customer** will make a
**second purchase**, based only on information available right after their
first order — delivery experience, price point, product category, and
payment behavior.

Built on 56,000+ real Brazilian e-commerce orders. Model: XGBoost,
ROC-AUC ≈ 0.61 — modest by design, since first-order data alone has a real,
honestly-reported ceiling for predicting future behavior (see *Model Notes*
in the sidebar).
"""
)

# ---------------------------------------------------------------------------
# Sidebar: model notes / honesty section
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📊 Model Notes")
    st.markdown(
        """
**Why such a modest ROC-AUC?**

Repeat-purchase behavior is driven heavily by things this dataset doesn't
capture — life circumstances, competitor offers, whether the customer
needed anything else. First-order delivery/price/category data carries
*real* signal (confirmed in EDA), but it's a soft signal, not a strong one.

**What was tried to push past the ceiling:**
- Logistic Regression → XGBoost (no meaningful gain)
- One-hot → target encoding for category (small gain)
- Added customer state (small gain)

**Honest takeaway:** model complexity wasn't the bottleneck — feature
richness was. A production version would benefit from browsing behavior,
marketing touchpoints, or multi-order history once available.
"""
    )
    st.divider()
    st.caption("Built with XGBoost + SHAP · Olist Brazilian E-Commerce dataset")

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.subheader("Customer's First Order Details")

col1, col2, col3 = st.columns(3)

with col1:
    category_key = st.selectbox(
        "Product Category",
        options=sorted(CATEGORY_LABELS, key=lambda k: CATEGORY_LABELS[k]),
        format_func=lambda k: CATEGORY_LABELS[k],
        index=sorted(CATEGORY_LABELS, key=lambda k: CATEGORY_LABELS[k]).index("bed_bath_table"),
    )
    state_key = st.selectbox(
        "Customer State",
        options=sorted(STATE_LABELS, key=lambda k: STATE_LABELS[k]),
        format_func=lambda k: f"{STATE_LABELS[k]} ({k})",
        index=sorted(STATE_LABELS, key=lambda k: STATE_LABELS[k]).index("SP"),
    )
    payment_value = st.number_input("Total Order Value (R$)", min_value=1.0, max_value=10000.0, value=120.0, step=10.0)

with col2:
    n_items = st.number_input("Number of Items", min_value=1, max_value=20, value=1, step=1)
    payment_installments = st.number_input("Payment Installments", min_value=1, max_value=24, value=1, step=1)
    payment_type_count = st.selectbox("Number of Payment Methods Used", options=[1, 2, 3], index=0)

with col3:
    delivery_delay_days = st.slider(
        "Delivery Delay (days)", min_value=-60, max_value=60, value=-12,
        help="Negative = delivered early. Olist customers average about 12 days early.",
    )
    delivery_speed_days = st.slider("Delivery Speed (days, purchase to delivery)", min_value=0, max_value=60, value=10)
    review_score = st.select_slider("Review Score Given", options=[1, 2, 3, 4, 5], value=5)

col4, col5 = st.columns(2)
with col4:
    purchase_month = st.selectbox("Purchase Month", options=list(range(1, 13)), index=0,
                                   format_func=lambda m: pd.Timestamp(2018, m, 1).strftime("%B"))
with col5:
    purchase_dayofweek = st.selectbox("Purchase Day of Week", options=list(range(7)), index=0,
                                       format_func=lambda d: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][d])

has_review = 1  # assume a review exists for simplicity in this demo
approval_speed_hours = 1.0  # typical fast approval; not exposed as input (low importance feature)
total_price = payment_value * 0.85  # rough split, since user enters total payment_value
total_freight = payment_value * 0.15
avg_item_price = total_price / n_items

predict_clicked = st.button("🔮 Predict Repeat Purchase Probability", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if predict_clicked:
    category_enc = category_encoding.get(category_key, global_mean)
    state_enc = state_encoding.get(state_key, global_mean)

    row = {
        "payment_value": payment_value,
        "payment_installments_max": payment_installments,
        "payment_type_count": payment_type_count,
        "n_items": n_items,
        "total_price": total_price,
        "total_freight": total_freight,
        "avg_item_price": avg_item_price,
        "delivery_delay_days": delivery_delay_days,
        "delivery_speed_days": delivery_speed_days,
        "approval_speed_hours": approval_speed_hours,
        "purchase_dayofweek": purchase_dayofweek,
        "purchase_month": purchase_month,
        "has_review": has_review,
        "review_score_rounded": review_score,
        "state_target_enc": state_enc,
        "category_target_enc": category_enc,
    }

    X_input = pd.DataFrame([row])[feature_columns]

    proba = model.predict_proba(X_input)[0, 1]

    st.divider()
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.metric("Repeat Purchase Probability", f"{proba*100:.1f}%")
        baseline = global_mean * 100
        delta = proba * 100 - baseline
        st.caption(f"Baseline rate across all customers: {baseline:.1f}%")
        if delta > 0:
            st.success(f"⬆ {delta:.1f} points above baseline")
        else:
            st.warning(f"⬇ {abs(delta):.1f} points below baseline")

    with res_col2:
        st.markdown("**Top factors driving this prediction (SHAP):**")
        shap_values = explainer.shap_values(X_input)
        shap_row = shap_values[0]
        shap_df = pd.DataFrame({
            "feature": feature_columns,
            "shap_value": shap_row,
        }).sort_values("shap_value", key=abs, ascending=False).head(6)

        for _, r in shap_df.iterrows():
            direction = "increases" if r["shap_value"] > 0 else "decreases"
            color = "green" if r["shap_value"] > 0 else "red"
            st.markdown(f"- **{r['feature']}** {direction} repeat-purchase likelihood "
                        f"(:{color}[{r['shap_value']:+.4f}])")

    st.divider()
    st.caption(
        "Note: this is a demo prediction based on a model with a deliberately "
        "documented performance ceiling (ROC-AUC ≈ 0.61). Treat outputs as "
        "directional signal, not precise forecasts — see sidebar for why."
    )
