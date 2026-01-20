import streamlit as st
import pandas as pd
import joblib
import numpy as np  # ✅ FIX 1: required for NaN handling

# 🔑 REQUIRED FOR PICKLE (DO NOT REMOVE)
from preprocessing_utils import to_string

# ----------------------------------
# 1. Page Config
# ----------------------------------
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="centered"
)

# ----------------------------------
# 2. Custom CSS (UI UNCHANGED)
# ----------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
}

.glass {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 20px;
    margin-bottom: 20px;
    color: #e0e0e0;
}

.title {
    font-size: 28px;
    font-weight: 700;
    color: #ffffff;
}

.subtitle {
    font-size: 14px;
    color: #a0a0a0;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# 3. Load Model (PIPELINE)
# ----------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model/final_credit_model.pkl")

model = load_model()
FINAL_THRESHOLD = 0.35

# ----------------------------------
# 4. Helper (NO LAMBDA)
# ----------------------------------
def job_label(x):
    return {
        0: "Unskilled / Non-resident",
        1: "Unskilled / Resident",
        2: "Skilled",
        3: "Highly Skilled"
    }[x]

# ✅ FIX 2: Normalize NA values to match training data
def clean_na(val):
    return np.nan if val == "NA" else val

# ----------------------------------
# 5. Sidebar
# ----------------------------------
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/633/633611.png",
        width=100
    )
    st.title("Credit Risk Prediction")

    st.markdown("### ℹ️ About")
    st.info(
        """
        **Model:** Random Forest  
        **Dataset:** German Credit  
        **Output:** Probability of Default
        """
    )

    st.markdown("### ⚙️ Decision Rule")
    st.write(f"Threshold = **{FINAL_THRESHOLD}**")
    st.caption("Probability ≥ threshold → High Risk")

# ----------------------------------
# 6. Header
# ----------------------------------
st.markdown(
    """
    <div class="glass" style="text-align:center;">
        <div class="title">💳 Credit Risk Prediction</div>
        <div class="subtitle">Evaluate borrower creditworthiness</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------------
# 7. Input Form (RAW FEATURES ONLY)
# ----------------------------------
st.markdown("### 📝 Applicant Details")

with st.container():
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Sex", ["male", "female"])
        job = st.selectbox(
            "Job Type",
            [0, 1, 2, 3],
            format_func=job_label
        )
        housing = st.selectbox("Housing", ["own", "rent", "free"])
        saving_accounts = st.selectbox(
            "Saving accounts",
            ["little", "moderate", "quite rich", "rich", "NA"]
        )

    with col2:
        credit_amount = st.number_input(
            "Credit amount", min_value=0, value=1000, step=500
        )
        duration = st.number_input(
            "Duration (months)", min_value=1, max_value=72, value=12
        )
        checking_account = st.selectbox(
            "Checking account",
            ["little", "moderate", "rich", "NA"]
        )
        purpose = st.selectbox(
            "Purpose",
            [
                "radio/TV", "education", "furniture/equipment",
                "car", "business", "domestic appliances",
                "repairs", "vacation/others"
            ]
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------
# 8. Prediction
# ----------------------------------
if st.button("🔍 Predict Credit Risk", use_container_width=True):

    # ---- Business Rule ----
    if credit_amount == 0:
        st.success("LOW RISK ✅ (No credit exposure)")
        st.stop()

    # ---- BUILD INPUT (MATCH TRAINING EXACTLY) ----
    X_input = pd.DataFrame([{
        "Age": age,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Sex": sex,
        "Job": job,
        "Housing": housing,
        "Saving accounts": clean_na(saving_accounts),   # ✅ FIX
        "Checking account": clean_na(checking_account), # ✅ FIX
        "Purpose": purpose
    }])

    # ---- Predict ----
    risk_prob = model.predict_proba(X_input)[0][1]

    # ---- Decision ----
    if risk_prob >= FINAL_THRESHOLD:
        label = "HIGH RISK ⚠️"
        color = "#FF4B4B"
        msg = "Applicant probability exceeds acceptable risk threshold."
    else:
        label = "LOW RISK ✅"
        color = "#4CAF50"
        msg = "Applicant falls within safe risk limits."

    # ----------------------------------
    # 9. Result Display
    # ----------------------------------
    st.markdown("---")
    st.metric("Default Probability", f"{risk_prob:.2%}")
    st.progress(risk_prob)

    st.markdown(
        f"""
        <div class="glass" style="border-left:5px solid {color};">
            <h2 style="color:{color};">{label}</h2>
            <p>{msg}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
