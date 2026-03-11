
import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")

st.title("🚗 Car Price Prediction")
st.write("Enter car details and click **Predict Price**.")

# Load trained model
model = pickle.load(open("model.pkl","rb"))

col1, col2 = st.columns(2)

with col1:
    km_driven = st.number_input("KM Driven", min_value=0, value=50000)
    engine = st.number_input("Engine (CC)", min_value=500, value=1200)

with col2:
    max_power = st.number_input("Max Power (bhp)", min_value=10.0, value=80.0)

st.markdown("---")

if st.button("Predict Price"):
    input_data = np.array([[km_driven, engine, max_power]])
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Car Price: ₹ {int(prediction):,}")
