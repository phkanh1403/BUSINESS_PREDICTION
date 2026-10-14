import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title("BUSINESS PREDICTION")

@st.cache_resource
def load_model():
    model = joblib.load("xgb_revenue_model.pkl")
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_model()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age of the business owner", min_value=18, max_value=100, value=30)
    experience = st.number_input("Years of experience in the business", min_value=0, max_value=80, value=5)
    team_size = st.number_input("Size of the team", min_value=1, max_value=10000, value=10)
    customer_rating = st.slider("Customer rating (1->5)", min_value=1, max_value=5, value=1)
with col2:
    monthly_ad_spend = st.number_input("Monthly ad spend in USD", min_value=0, value=1000)
    website_visits = st.number_input("Average monthly website visits",min_value=0, value=5000)
    region = st.selectbox("Region", ["North","South","East","West"])
    subscription_plan = st.selectbox("Subscription plan",["Basic","Standard","Premium"])
    business_type = st.selectbox("Business type", ["Retail","SaaS","Service","E-commerce"])
if st.button("Predict"):
    input_data = pd.DataFrame([{
        'age': age,
        'experience_years': experience,
        'monthly_ad_spend': monthly_ad_spend,
        'website_visits': website_visits,
        'team_size': team_size,
        'customer_rating': customer_rating,
        'region': region,
        'business_type': business_type,
        'subscription_plan': subscription_plan
    }])
    input_data_preprocessed = preprocessor.transform(input_data)
    prediction = model.predict(input_data_preprocessed)[0]
    st.success(f"Prediction venue: ${prediction:,.2f}")