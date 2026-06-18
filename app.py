import streamlit as st
import pandas as pd
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.hero {
    padding: 2rem;
    border-radius: 15px;
    background: linear-gradient(135deg,#1e3c72,#2a5298);
    color:white;
    text-align:center;
    margin-bottom:20px;
}

.card {
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 10px rgba(0,0,0,0.1);
    text-align:center;
}

.metric-card{
    background:white;
    padding:15px;
    border-radius:12px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.1);
}

h1,h2,h3{
    color:#1e3c72;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load("artifacts/model.pkl")
    scaler = joblib.load("artifacts/scaler.pkl")
    selected_cols = joblib.load("artifacts/selected_features.pkl")

    return model, scaler, selected_cols

model, scaler, selected_cols = load_artifacts()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
    width=120
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "",
    [
        "🏠 Home",
        "📊 Dataset",
        "🤖 Prediction",
        "📈 Model Info",
        "ℹ️ About"
    ]
)

# =====================================================
# HOME PAGE
# =====================================================

if page == "🏠 Home":

    st.markdown("""
    <div class='hero'>
        <h1>Customer Churn Prediction System</h1>
        <h4>Predict customer attrition before it happens.</h4>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Project Overview")

    st.write("""
    Customer churn prediction helps businesses identify customers
    who are likely to stop using their services.

    By proactively identifying at-risk customers, organizations can
    improve retention and reduce revenue loss.
    """)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model", "Logistic Regression")

    with col2:
        st.metric("Features", "9")

    with col3:
        st.metric("Prediction", "Real-Time")

    with col4:
        st.metric("Deployment", "Streamlit")

    st.markdown("---")

    st.subheader("Key Features")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info("📉 Churn Prediction")

    with c2:
        st.info("📊 Probability Analysis")

    with c3:
        st.info("⚡ Instant Results")

# =====================================================
# DATASET PAGE
# =====================================================

elif page == "📊 Dataset":

    st.title("📊 Dataset Information")

    st.write("""
    This dataset contains customer demographic,
    subscription and service information used to predict churn.
    """)

    feature_df = pd.DataFrame({
        "Feature":[
            "Age",
            "Tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Gender",
            "Internet Service",
            "Tech Support",
            "Contract Type"
        ],
        "Description":[
            "Customer age",
            "Months with company",
            "Monthly bill amount",
            "Total spending",
            "Gender",
            "Internet package",
            "Support subscription",
            "Contract duration"
        ]
    })

    st.dataframe(feature_df, use_container_width=True)

    st.markdown("---")

    st.subheader("Target Variable")

    st.success("0 → Customer Stays")
    st.error("1 → Customer Churns")

# =====================================================
# MODEL PAGE
# =====================================================

elif page == "📈 Model Info":

    st.title("📈 Model Information")

    st.subheader("Algorithm")

    st.write("""
    Logistic Regression is used as the classification model.
    """)

    st.subheader("Preprocessing")

    st.write("""
    ✔ Feature Engineering

    ✔ Encoding

    ✔ StandardScaler

    ✔ SMOTE Balancing
    """)

    st.subheader("Workflow")

    st.code("""
Dataset
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
Scaling
   ↓
Training
   ↓
Prediction
""")

# =====================================================
# ABOUT PAGE
# =====================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About Project")

    st.write("""
    Customer Churn Prediction System developed using
    Machine Learning and Streamlit.
    """)

    st.subheader("Technologies Used")

    st.write("""
    • Python

    • Pandas

    • NumPy

    • Scikit-Learn

    • Streamlit

    • Joblib
    """)

    st.subheader("Developer")

    st.info("""
    Name: Chinmay Wade

    MCA Student

    Bharati Vidyapeeth Institute of Management &
    Information Technology

    Internship Project
    """)

# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "🤖 Prediction":

    st.title("🤖 Customer Churn Prediction")

    st.write("Enter customer details below.")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )

        tenure = st.number_input(
            "Tenure",
            min_value=0,
            max_value=120,
            value=24
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            value=65.0
        )

        total_charges = st.number_input(
            "Total Charges",
            value=1500.0
        )

    with col2:

        gender = st.selectbox(
            "Gender",
            ["Male","Female"]
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL","Fiber Optic","NA"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes","No"]
        )

        contract = st.selectbox(
            "Contract Type",
            ["Month-to-Month","One-Year","Two-Year"]
        )

    def build_input():

        gender_female = 1 if gender == "Female" else 0
        internet_fiber = 1 if internet == "Fiber Optic" else 0
        internet_na = 1 if internet == "NA" else 0
        tech_support_yes = 1 if tech_support == "Yes" else 0

        contract_map = {
            "Month-to-Month":0,
            "One-Year":1,
            "Two-Year":2
        }

        data = {
            "Age": age,
            "Tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "ContractType": contract_map[contract],
            "Gender_Male": 1 - gender_female,
            "InternetService_Fiber optic": internet_fiber,
            "InternetService_NA": internet_na,
            "TechSupport_Yes": tech_support_yes
        }

        df = pd.DataFrame([data])

        for col in selected_cols:
            if col not in df.columns:
                df[col] = 0

        return df[selected_cols]

    if st.button("🔍 Predict", use_container_width=True):

        input_df = build_input()

        scaled = scaler.transform(input_df)

        pred = model.predict(scaled)[0]

        prob = model.predict_proba(scaled)[0]

        churn_prob = prob[1] * 100
        retain_prob = prob[0] * 100

        st.markdown("---")

        if pred == 1:

            st.error(
                f"⚠️ Customer likely to churn ({churn_prob:.2f}%)"
            )

        else:

            st.success(
                f"✅ Customer likely to stay ({retain_prob:.2f}%)"
            )

        st.subheader("Probability")

        st.progress(int(churn_prob))

        c1, c2 = st.columns(2)

        c1.metric(
            "Churn Probability",
            f"{churn_prob:.2f}%"
        )

        c2.metric(
            "Retention Probability",
            f"{retain_prob:.2f}%"
        )

        with st.expander("View Input Data"):
            st.dataframe(input_df)