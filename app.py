import streamlit as st
import pandas as pd
import joblib


# Load model and encoders
model = joblib.load('model.pkl')
encoders = joblib.load('encoders.pkl')

st.title("🏠 House Price Prediction")

# User input
district = st.selectbox("Select District", encoders['district'].classes_)
tahsil = st.selectbox("Select Tahsil", encoders['tahsil'].classes_)
village = st.selectbox("Select Village", encoders['village'].classes_)
area = st.number_input("Enter House Area (sq ft)", min_value=100)
bedroom = st.selectbox("Number of Bedrooms", encoders['bedrooms'].classes_)

if st.button("Know Your House Price"):
    # Encode inputs
    input_data = pd.DataFrame([[
        encoders['district'].transform([district])[0],
        encoders['tahsil'].transform([tahsil])[0],
        encoders['village'].transform([village])[0],
        area,
        encoders['bedrooms'].transform([bedroom])[0]
    ]], columns=['district', 'tahsil', 'village', 'house_area', 'bedrooms'])

    # Predict
    price = model.predict(input_data)[0]
    st.success(f"🏷️ Estimated Price: ₹{price:,.2f}")
