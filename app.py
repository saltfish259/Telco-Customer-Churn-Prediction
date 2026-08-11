import streamlit as st
import pandas as pd
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📡",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📡 Telco Customer Churn Prediction")
st.write("Predict customer churn probability using a trained LightGBM model.")
st.divider()

# ── Load pipeline (inside render flow, not at module level) ───────────────────
PIPELINE_PATH = "models/lightgbm_pipeline.pkl"

@st.cache_resource
def load_pipeline():
    import joblib
    return joblib.load(PIPELINE_PATH)

if not os.path.exists(PIPELINE_PATH):
    st.error(
        "❌ Pipeline file not found. "
        "Please place `lightgbm_pipeline.pkl` inside the `models/` folder and redeploy."
    )
    st.stop()

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"❌ Failed to load the pipeline. Details: {e}")
    st.stop()

# ── Customer input form ───────────────────────────────────────────────────────
st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    internet_service = st.selectbox(
        "Internet Service",
        ["No", "DSL", "Fiber optic"]
    )
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )
    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=500.0,
        value=65.0,
        step=0.5
    )

with col2:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=120,
        value=12,
        step=1
    )
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

# ── Auto-calculated TotalCharges ──────────────────────────────────────────────
total_charges = tenure * monthly_charges
st.info(f"💡 Estimated Total Charges: **${total_charges:,.2f}**  *(tenure × monthly charges)*")

st.divider()

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("🔍 Predict Churn", use_container_width=True, type="primary"):
    try:
        # Build the input DataFrame with exact feature names the pipeline expects
        input_df = pd.DataFrame([{
            "InternetService":  internet_service,
            "PaymentMethod":    payment_method,
            "tenure":           tenure,
            "Contract":         contract,
            "MonthlyCharges":   monthly_charges,
            "TotalCharges":     total_charges,
            "MultipleLines":    multiple_lines,
            "StreamingTV":      streaming_tv,
            "StreamingMovies":  streaming_movies,
            "PhoneService":     phone_service
        }])

        prediction  = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]
        probability_pct = probability * 100

        predicted_class = "Churn" if prediction == 1 else "Not Churn"

        if probability_pct < 30:
            risk_level = "Low"
        elif probability_pct < 60:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # ── Result display ────────────────────────────────────────────────────
        st.subheader("Prediction Result")

        if risk_level == "High":
            st.error("🔴 High Churn Risk")
        elif risk_level == "Medium":
            st.warning("🟡 Medium Churn Risk")
        else:
            st.success("🟢 Low Churn Risk")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Predicted Class",    predicted_class)
        col_b.metric("Churn Probability",  f"{probability_pct:.1f}%")
        col_c.metric("Risk Level",         risk_level)

    except Exception as e:
        st.error("❌ Prediction failed. Please check your inputs and try again.")
        st.caption(f"Details: {e}")

# ── About the model ───────────────────────────────────────────────────────────
st.divider()
with st.expander("ℹ️ About the Model"):
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("**Model:** LightGBM")
        st.markdown("**Purpose:** Customer churn prediction")
        st.markdown("**Selected Features:** 10")
    with col_m2:
        st.markdown("**Model Performance**")
        st.markdown("- ROC AUC: 0.8360")
        st.markdown("- F1 Score: 0.6274")
        st.markdown("- Recall: 0.7016")
        st.markdown("- Precision: 0.5674")

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.caption(
    "⚠️ This application is for demonstration and educational purposes. "
    "Predictions should not be used as the only basis for business decisions."
)
