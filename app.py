
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("churn_prediction_model.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction System")
st.write(
    "Enter customer details to estimate whether the customer "
    "is likely to leave the service."
)

st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines", ["No", "Yes", "No phone service"]
    )

with col2:
    internet_service = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )
    online_security = st.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )
    tech_support = st.selectbox(
        "Tech Support", ["Yes", "No", "No internet service"]
    )
    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

st.subheader("Billing Information")

monthly_charges = st.number_input(
    "Monthly Charges ($)", min_value=0.0, value=70.0
)
total_charges = st.number_input(
    "Total Charges ($)", min_value=0.0, value=840.0
)

# Build a row using the original dataset's column names
customer = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "TechSupport": tech_support,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}])

# Add any dataset features not entered in the interface
# with valid default values.
defaults = {
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No"
}

feature_columns = model.named_steps["preprocessor"].feature_names_in_

for column, value in defaults.items():
    if column in feature_columns and column not in customer.columns:
        customer[column] = value

customer = customer.reindex(columns=feature_columns)

st.write("Review your customer information below:")
st.dataframe(customer)

if st.button("Predict Customer Churn", type="primary"):
    prediction = model.predict(customer)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ This customer is predicted to churn.")
    else:
        st.success("✅ This customer is predicted to stay.")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(customer)[0]
        churn_probability = probabilities[list(model.classes_).index(1)]

        st.metric(
            "Estimated Churn Probability",
            f"{churn_probability:.1%}"
        )

        st.progress(float(churn_probability))

st.caption(
    "Educational project: predictions are estimates, not guarantees "
    "of customer behavior."
)
