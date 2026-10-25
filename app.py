import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"
META_PATH = BASE_DIR / "models" / "metadata.json"

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error("Model files missing. Run Step 5 in your notebook first.")
    st.stop()

model, metadata = load_artifacts()
features = metadata["features"]

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}
.hero {
    padding: 1.6rem;
    border-radius: 18px;
    background: linear-gradient(120deg, #14213d, #264b72);
    color: white;
    margin-bottom: 1.3rem;
}
.hero h1 { color: white; margin-bottom: 0.3rem; }
.hero p { color: #dce8f5; margin-bottom: 0; }
.metric-card {
    padding: 1rem;
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 14px;
}
div[data-testid="stForm"] {
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.title("📊 Churn Intelligence")
    st.caption("Customer retention analytics")
    st.divider()

    st.markdown("### Prediction settings")
    threshold = st.slider(
        "Churn alert threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Controls when a customer is flagged as likely to churn.",
    )

    st.info(
        "The threshold changes the displayed classification, "
        "not the model's estimated probability."
    )

    st.divider()
    st.caption(f"Model: {metadata['selected_model']}")
    st.caption("Dataset: IBM Telco Customer Churn")

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1>Customer Churn Prediction</h1>
    <p>Understand customer risk. Identify retention opportunities.</p>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_guide = st.tabs(
    ["🎯 Customer Prediction", "📘 How to use"]
)

with tab_guide:
    st.markdown("""
    ### How this application works

    1. Enter a customer's service, contract, and billing details.
    2. Select **Predict Churn** to run the saved ML pipeline.
    3. Review the estimated probability and risk category.
    4. Use the result to prioritize further customer review.

    **Important:** A prediction is an estimate, not a guarantee.
    The selected threshold should be validated against business costs.
    """)

with tab_predict:
    st.subheader("Customer profile")
    st.caption(
        "Adjust the fields below. The prediction updates when you submit."
    )

    with st.form("customer_form"):
        st.markdown("#### 1. Customer details")
        a, b, c = st.columns(3)

        with a:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox(
                "Senior citizen", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )
            partner = st.selectbox("Has partner", ["Yes", "No"])
            dependents = st.selectbox("Has dependents", ["Yes", "No"])

        with b:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple lines",
                ["No", "Yes", "No phone service"]
            )

        with c:
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox(
                "Paperless billing", ["Yes", "No"]
            )
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            )

        st.divider()
        st.markdown("#### 2. Internet and subscribed services")

        internet = st.selectbox(
            "Internet service",
            ["DSL", "Fiber optic", "No"]
        )

        if internet == "No":
            online_security = "No internet service"
            online_backup = "No internet service"
            device_protection = "No internet service"
            tech_support = "No internet service"
            streaming_tv = "No internet service"
            streaming_movies = "No internet service"

            st.info(
                "Internet-dependent services are set to "
                "'No internet service'."
            )
        else:
            x1, x2, x3 = st.columns(3)

            with x1:
                online_security = st.selectbox(
                    "Online security", ["No", "Yes"]
                )
                online_backup = st.selectbox(
                    "Online backup", ["No", "Yes"]
                )

            with x2:
                device_protection = st.selectbox(
                    "Device protection", ["No", "Yes"]
                )
                tech_support = st.selectbox(
                    "Tech support", ["No", "Yes"]
                )

            with x3:
                streaming_tv = st.selectbox(
                    "Streaming TV", ["No", "Yes"]
                )
                streaming_movies = st.selectbox(
                    "Streaming movies", ["No", "Yes"]
                )

        st.divider()
        st.markdown("#### 3. Billing information")

        charge1, charge2 = st.columns(2)

        with charge1:
            monthly = st.number_input(
                "Monthly charges ($)",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=1.0,
            )

        with charge2:
            total = st.number_input(
                "Total charges ($)",
                min_value=0.0,
                value=float(tenure * monthly),
                step=10.0,
                help="Enter the customer's actual total charges.",
            )

        submitted = st.form_submit_button(
            "🚀 Predict customer churn",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        customer = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": (
                "No phone service" if phone == "No"
                else multiple_lines
            ),
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }

        input_df = pd.DataFrame([customer]).reindex(
            columns=features
        )

        probability = float(
            model.predict_proba(input_df)[0, 1]
        )

        flagged = probability >= threshold

        if probability < 0.30:
            risk = "Low"
        elif probability < 0.60:
            risk = "Medium"
        else:
            risk = "High"

        st.session_state["churn_result"] = {
            "probability": probability,
            "flagged": flagged,
            "risk": risk,
            "customer": customer,
        }

    result = st.session_state.get("churn_result")

    if result:
        probability = result["probability"]
        flagged = probability >= threshold
        risk = result["risk"]

        st.divider()
        st.subheader("Prediction dashboard")

        m1, m2, m3 = st.columns(3)

        m1.metric("Churn probability", f"{probability:.1%}")
        m2.metric("Risk category", risk)
        m3.metric(
            "Decision",
            "Flag for review" if flagged else "Below threshold"
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%"},
                title={"text": "Estimated churn probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#4776a8"},
                    "steps": [
                        {"range": [0, 30], "color": "#d9eee3"},
                        {"range": [30, 60], "color": "#fff0c2"},
                        {"range": [60, 100], "color": "#f6d6d6"},
                    ],
                    "threshold": {
                        "line": {"color": "#26384a", "width": 4},
                        "thickness": 0.8,
                        "value": threshold * 100,
                    },
                },
            )
        )

        gauge.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=65, b=15),
        )

        left, right = st.columns([1, 1])

        with left:
            st.plotly_chart(gauge, use_container_width=True)

        with right:
            st.markdown("#### Customer snapshot")
            st.write(f"**Contract:** {result['customer']['Contract']}")
            st.write(f"**Tenure:** {result['customer']['tenure']} months")
            st.write(
                f"**Monthly charges:** "
                f"${result['customer']['MonthlyCharges']:.2f}"
            )
            st.write(
                f"**Internet:** {result['customer']['InternetService']}"
            )

            st.markdown("#### Suggested next action")

            if flagged:
                st.warning(
                    "Prioritize a customer review. Check billing concerns, "
                    "service quality, support history, and relevant offers."
                )
            else:
                st.success(
                    "The probability is below the selected alert threshold. "
                    "Continue normal service monitoring."
                )

        st.caption(
            "Risk bands are illustrative. The alert threshold is adjustable "
            "and is not automatically optimized for business costs."
        )