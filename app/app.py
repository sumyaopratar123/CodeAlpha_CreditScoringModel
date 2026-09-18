import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "credit_scoring_model.joblib"

st.set_page_config(
    page_title="CreditScore AI",
    page_icon="💳",
    layout="wide"
)

# =========================================================
# STYLE
# =========================================================

st.html("""
<style>

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(59,130,246,.12), transparent 28%),
        radial-gradient(circle at 90% 20%, rgba(99,102,241,.12), transparent 30%),
        linear-gradient(135deg, #f8fbff, #eef4ff, #f8f7ff);
}

.main .block-container {
    max-width: 1200px;
    padding-top: 30px;
    padding-bottom: 50px;
}

/* TOP BAR */

.topbar {
    background: white;
    border: 1px solid #e5eaf2;
    border-radius: 18px;
    padding: 16px 22px;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(15,23,42,.06);
}

.brand {
    color: #172554;
    font-size: 21px;
    font-weight: 800;
}

.blue {
    color: #2563eb;
}

.ready {
    color: #16a34a;
    font-size: 13px;
    font-weight: 700;
    margin-left: 15px;
}

/* HERO */

.hero {
    background: linear-gradient(135deg, #0f172a, #172554, #4338ca);
    border-radius: 26px;
    padding: 45px;
    margin-bottom: 35px;
    color: white;
    box-shadow: 0 20px 50px rgba(37,99,235,.18);
    animation: hero 0.7s ease;
}

.badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 30px;
    background: rgba(255,255,255,.10);
    border: 1px solid rgba(255,255,255,.20);
    color: #bfdbfe;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}

.hero-title {
    font-size: 44px;
    font-weight: 850;
    margin-top: 15px;
}

.hero-text {
    color: #cbd5e1;
    font-size: 16px;
    margin-top: 8px;
}

@keyframes hero {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* NORMAL TEXT */

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    color: #172033 !important;
}

.stApp p {
    color: #334155 !important;
}

[data-testid="stCaptionContainer"] {
    color: #64748b !important;
}

/* INPUTS */

div[data-baseweb="select"] > div {
    background: white !important;
    border: 1px solid #dbe4f0 !important;
    border-radius: 11px !important;
}

.stNumberInput input {
    background: white !important;
    color: #172033 !important;
    border: 1px solid #dbe4f0 !important;
    border-radius: 11px !important;
}

.stSelectbox label,
.stNumberInput label,
.stSlider label {
    color: #334155 !important;
    font-weight: 650 !important;
}

/* BUTTON */

.stButton > button {
    width: 100%;
    min-height: 56px;
    border: none;
    border-radius: 14px;
    background: linear-gradient(90deg, #2563eb, #4f46e5);
    color: white;
    font-size: 16px;
    font-weight: 800;
    box-shadow: 0 12px 30px rgba(37,99,235,.25);
    transition: all .25s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 18px 38px rgba(37,99,235,.32);
}

/* RESULT */

.result-box {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 22px;
    padding: 25px;
    margin-top: 25px;
    text-align: center;
    box-shadow: 0 15px 40px rgba(15,23,42,.08);
    animation: result .5s ease;
}

.result-title {
    color: #172033;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

@keyframes result {
    from {
        opacity: 0;
        transform: scale(.95);
    }

    to {
        opacity: 1;
        transform: scale(1);
    }
}

/* FOOTER */

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    margin-top: 40px;
    padding-top: 25px;
    border-top: 1px solid #dbe4f0;
}

</style>
""")

# =========================================================
# TOP BAR
# =========================================================

st.html("""
<div class="topbar">
    <span class="brand">
        💳 Credit<span class="blue">Score</span> AI
    </span>

    <span class="ready">
        ● Model Ready
    </span>
</div>
""")

# =========================================================
# HERO
# =========================================================

st.html("""
<div class="hero">

    <div class="badge">
        AI • MACHINE LEARNING • CREDIT ANALYTICS
    </div>

    <div class="hero-title">
        CreditScore AI
    </div>

    <div class="hero-text">
        Intelligent creditworthiness assessment powered by machine learning.
    </div>

</div>
""")

# =========================================================
# MODEL
# =========================================================

if not MODEL_PATH.exists():
    st.error("⚠️ Model not found. Run src/train_models.py first.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# =========================================================
# CUSTOMER PROFILE
# =========================================================

st.markdown("## 👤 Customer Profile")

st.caption(
    "Enter the customer's basic credit and financial information."
)

left, right = st.columns(2, gap="large")

with left:

    checking_status = st.selectbox(
        "Checking account status",
        ["A11", "A12", "A13", "A14"]
    )

    duration = st.number_input(
        "Credit duration (months)",
        4,
        72,
        24
    )

    credit_history = st.selectbox(
        "Credit history",
        ["A30", "A31", "A32", "A33", "A34"]
    )

    purpose = st.selectbox(
        "Purpose",
        [
            "A40",
            "A41",
            "A410",
            "A42",
            "A43",
            "A44",
            "A45",
            "A46",
            "A47",
            "A48",
            "A49"
        ]
    )

    credit_amount = st.number_input(
        "Credit amount",
        250,
        20000,
        3000,
        step=100
    )

    savings = st.selectbox(
        "Savings status",
        ["A61", "A62", "A63", "A64", "A65"]
    )

    employment = st.selectbox(
        "Employment",
        ["A71", "A72", "A73", "A74", "A75"]
    )

    installment = st.slider(
        "Installment rate",
        1,
        4,
        2
    )

    personal = st.selectbox(
        "Personal status / sex",
        ["A91", "A92", "A93", "A94"]
    )

    other_debtors = st.selectbox(
        "Other debtors",
        ["A101", "A102", "A103"]
    )


with right:

    residence = st.slider(
        "Residence since",
        1,
        4,
        2
    )

    property_ = st.selectbox(
        "Property",
        ["A121", "A122", "A123", "A124"]
    )

    age = st.number_input(
        "Age",
        18,
        80,
        35
    )

    other_plans = st.selectbox(
        "Other installment plans",
        ["A141", "A142", "A143"]
    )

    housing = st.selectbox(
        "Housing",
        ["A151", "A152", "A153"]
    )

    existing = st.slider(
        "Existing credits",
        1,
        4,
        1
    )

    job = st.selectbox(
        "Job",
        ["A171", "A172", "A173", "A174"]
    )

    dependents = st.slider(
        "Dependents",
        1,
        2,
        1
    )

    telephone = st.selectbox(
        "Telephone",
        ["A191", "A192"]
    )

    foreign = st.selectbox(
        "Foreign worker",
        ["A201", "A202"]
    )

# =========================================================
# DATA
# =========================================================

row = pd.DataFrame([{
    "checking_status": checking_status,
    "duration_months": duration,
    "credit_history": credit_history,
    "purpose": purpose,
    "credit_amount": credit_amount,
    "savings_status": savings,
    "employment": employment,
    "installment_rate": installment,
    "personal_status_sex": personal,
    "other_debtors": other_debtors,
    "residence_since": residence,
    "property": property_,
    "age": age,
    "other_installment_plans": other_plans,
    "housing": housing,
    "existing_credits": existing,
    "job": job,
    "dependents": dependents,
    "telephone": telephone,
    "foreign_worker": foreign
}])

# =========================================================
# PREDICTION
# =========================================================

st.markdown("---")

st.markdown("## 🤖 AI Credit Assessment")

st.caption(
    "Analyze the submitted profile using the trained machine-learning model."
)

c1, c2, c3 = st.columns([1, 2, 1])

with c2:

    predict = st.button(
        "🔍 Analyze Credit Profile"
    )

if predict:

    with st.spinner("🤖 Analyzing credit profile..."):

        prediction = int(model.predict(row)[0])

        probability = float(
            model.predict_proba(row)[0, 1]
        )

    percentage = probability * 100

    st.html("""
    <div class="result-box">
        <div class="result-title">
            AI Assessment Result
        </div>
    </div>
    """)

    if prediction == 1:

        st.success("✅ Good Creditworthiness")

    else:

        st.warning("⚠️ Higher Credit Risk")

    st.metric(
        "Estimated Probability",
        f"{percentage:.1f}%"
    )

    st.progress(
        probability
    )

    st.caption(
        "Prediction generated from the submitted credit profile."
    )

# =========================================================
# FOOTER
# =========================================================

st.html("""
<div class="footer">
    💡 CreditScore AI • Machine Learning Internship Project
    <br>
    Educational demonstration only.
    Not intended for real-world lending or financial decisions.
</div>
""")