import streamlit as st
import tensorflow as tf
import joblib
import pandas as pd



model = tf.keras.models.load_model("model/churn_model.keras")

Lable_encoders = joblib.load("model/label_encoders.pkl")
OneHot_encoders = joblib.load("model/onehot_encoders.pkl")
scaler = joblib.load("model/scaler.pkl")



st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)



st.markdown("""
<style>

.main{
    background:#f7f9fc;
}

h1{
    color:#0E4C92;
    text-align:center;
}

.stButton>button{
    width:100%;
    height:55px;
    font-size:20px;
    font-weight:bold;
    border-radius:12px;
}

div[data-testid="stMetric"]{
    border-radius:12px;
    padding:10px;
    background:#ffffff;
    box-shadow:0px 0px 8px rgba(0,0,0,0.15);
}

</style>
""",unsafe_allow_html=True)



with st.sidebar:

    st.title("📊 Customer Churn")

    st.markdown("---")

    st.markdown("""
### Deep Learning Project

**Model**
- Artificial Neural Network

**Framework**
- TensorFlow

**Frontend**
- Streamlit

**Developer**
- Aayush Patidar
""")

    st.markdown("---")

    st.info(
        "Enter customer information and click Predict."
    )



st.title("📊 Customer Churn Prediction")

st.write(
    "Predict whether a telecom customer is likely to churn using an Artificial Neural Network."
)

st.divider()



col1,col2,col3=st.columns(3)



with col1:

    st.subheader("👤 Customer")

    gender=st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    senior=st.selectbox(
        "Senior Citizen",
        [0,1]
    )

    partner=st.selectbox(
        "Partner",
        ["Yes","No"]
    )

    dependents=st.selectbox(
        "Dependents",
        ["Yes","No"]
    )

    tenure=st.number_input(
        "Tenure (Months)",
        0,
        100,
        12
    )

    phone=st.selectbox(
        "Phone Service",
        ["Yes","No"]
    )

    multiple=st.selectbox(
        "Multiple Lines",
        ["No","Yes","No phone service"]
    )



with col2:

    st.subheader("🌐 Services")

    internet=st.selectbox(
        "Internet Service",
        ["DSL","Fiber optic","No"]
    )

    online_security=st.selectbox(
        "Online Security",
        ["Yes","No","No internet service"]
    )

    online_backup=st.selectbox(
        "Online Backup",
        ["Yes","No","No internet service"]
    )

    device=st.selectbox(
        "Device Protection",
        ["Yes","No","No internet service"]
    )

    tech=st.selectbox(
        "Tech Support",
        ["Yes","No","No internet service"]
    )

    tv=st.selectbox(
        "Streaming TV",
        ["Yes","No","No internet service"]
    )

    movies=st.selectbox(
        "Streaming Movies",
        ["Yes","No","No internet service"]
    )


with col3:

    st.subheader("💳 Billing")

    contract=st.selectbox(
        "Contract",
        ["Month-to-month","One year","Two year"]
    )

    paperless=st.selectbox(
        "Paperless Billing",
        ["Yes","No"]
    )

    payment=st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly=st.number_input(
        "Monthly Charges",
        value=70.0
    )

    total=st.number_input(
        "Total Charges",
        value=1000.0
    )

st.divider()

predict=st.button(
    "🚀 Predict Customer Churn",
    use_container_width=True
)
if predict:

    with st.spinner("Predicting Customer Churn..."):

        

        gender = Lable_encoders["gender"].transform([gender])[0]
        partner = Lable_encoders["Partner"].transform([partner])[0]
        dependents = Lable_encoders["Dependents"].transform([dependents])[0]
        phone = Lable_encoders["PhoneService"].transform([phone])[0]
        paperless = Lable_encoders["PaperlessBilling"].transform([paperless])[0]

       

        internet = OneHot_encoders["InternetService"].transform([[internet]]).toarray()[0]

        multiple = OneHot_encoders["MultipleLines"].transform([[multiple]]).toarray()[0]

        online_security = OneHot_encoders["OnlineSecurity"].transform([[online_security]]).toarray()[0]

        online_backup = OneHot_encoders["OnlineBackup"].transform([[online_backup]]).toarray()[0]

        device = OneHot_encoders["DeviceProtection"].transform([[device]]).toarray()[0]

        tech = OneHot_encoders["TechSupport"].transform([[tech]]).toarray()[0]

        tv = OneHot_encoders["StreamingTV"].transform([[tv]]).toarray()[0]

        movies = OneHot_encoders["StreamingMovies"].transform([[movies]]).toarray()[0]

        contract = OneHot_encoders["Contract"].transform([[contract]]).toarray()[0]

        payment = OneHot_encoders["PaymentMethod"].transform([[payment]]).toarray()[0]

        

        data = pd.DataFrame({

            "gender":[gender],
            "SeniorCitizen":[senior],
            "Partner":[partner],
            "Dependents":[dependents],
            "tenure":[tenure],
            "PhoneService":[phone],

            "No_MultipleLines":[multiple[0]],
            "No phone service_MultipleLines":[multiple[1]],
            "Yes_MultipleLines":[multiple[2]],

            "DSL_InternetService":[internet[0]],
            "Fiber optic_InternetService":[internet[1]],
            "No_InternetService":[internet[2]],

            "No_OnlineSecurity":[online_security[0]],
            "No internet service_OnlineSecurity":[online_security[1]],
            "Yes_OnlineSecurity":[online_security[2]],

            "No_OnlineBackup":[online_backup[0]],
            "No internet service_OnlineBackup":[online_backup[1]],
            "Yes_OnlineBackup":[online_backup[2]],

            "No_DeviceProtection":[device[0]],
            "No internet service_DeviceProtection":[device[1]],
            "Yes_DeviceProtection":[device[2]],

            "No_TechSupport":[tech[0]],
            "No internet service_TechSupport":[tech[1]],
            "Yes_TechSupport":[tech[2]],

            "No_StreamingTV":[tv[0]],
            "No internet service_StreamingTV":[tv[1]],
            "Yes_StreamingTV":[tv[2]],

            "No_StreamingMovies":[movies[0]],
            "No internet service_StreamingMovies":[movies[1]],
            "Yes_StreamingMovies":[movies[2]],

            "Month-to-month_Contract":[contract[0]],
            "One year_Contract":[contract[1]],
            "Two year_Contract":[contract[2]],

            "PaperlessBilling":[paperless],

            "Bank transfer (automatic)_PaymentMethod":[payment[0]],
            "Credit card (automatic)_PaymentMethod":[payment[1]],
            "Electronic check_PaymentMethod":[payment[2]],
            "Mailed check_PaymentMethod":[payment[3]],

            "MonthlyCharges":[monthly],
            "TotalCharges":[total]

        })

       

        data[["tenure","MonthlyCharges","TotalCharges"]] = scaler.transform(
            data[["tenure","MonthlyCharges","TotalCharges"]]
        )

        

        prediction = model.predict(data, verbose=0)

        probability = float(prediction[0][0])

    st.divider()

    st.subheader("📈 Prediction Result")

    st.progress(probability)

    colA,colB=st.columns(2)

    with colA:

        st.metric(
            "Churn Probability",
            f"{probability*100:.2f}%"
        )

    with colB:

        st.metric(
            "Confidence",
            f"{max(probability,1-probability)*100:.2f}%"
        )

    st.divider()

    if probability>=0.5:

        st.error("🔴 High Risk Customer (Likely to Churn)")

        st.warning("""
### Recommended Actions

- Offer Discount
- Provide Better Support
- Upgrade Plan
- Retention Campaign
- Personalized Offers
""")

    else:

        st.success("🟢 Customer is Likely to Stay")

        st.balloons()

        st.info("""
### Customer Status

- Loyal Customer
- Low Churn Risk
- Continue Existing Plan
""")

    st.divider()

    with st.expander("📋 Processed Input Data"):

        st.dataframe(data)