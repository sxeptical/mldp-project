import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load the Saved Model ---
try:
    model = joblib.load('depression_model.pkl')
    # scaler = joblib.load('scaler.pkl') # Uncomment if you used a scaler
except FileNotFoundError:
    st.error("Model files not found. Please run Step 1 to save 'depression_model.pkl'.")
    st.stop()

# --- 2. App Title & Description ---
st.title("🧠 Student Mental Health Screening Tool")
st.write("Enter student details below to assess the risk of depression.")

# --- 3. Sidebar Inputs (Raw Features) ---
st.sidebar.header("Student Profile")

# Sliders for key stress factors
academic_pressure = st.sidebar.slider("Academic Pressure (1-5)", 1, 5, 3)
study_hours = st.sidebar.slider("Work/Study Hours (per day)", 0, 16, 6)
financial_stress = st.sidebar.slider("Financial Stress (1-5)", 1, 5, 3)
study_satisfaction = st.sidebar.slider("Study Satisfaction (1-5)", 1, 5, 3)

# Demographic inputs
cgpa = st.sidebar.number_input("CGPA", 0.0, 5.0, 3.5, step=0.01)
age = st.sidebar.number_input("Age", 18, 35, 21)

# --- 4. Feature Engineering (Crucial Step!) ---
# We must calculate the SAME engineered features you used in training
burnout_index = (academic_pressure * study_hours) / (study_satisfaction + 1)
total_stress = financial_stress * academic_pressure

# Create a DataFrame for the model
input_data = pd.DataFrame({
    'Academic Pressure': [academic_pressure],
    'Work/Study Hours': [study_hours],
    'Financial Stress': [financial_stress],
    'Study Satisfaction': [study_satisfaction],
    'CGPA': [cgpa],
    'Age': [age],
    'Burnout_Index': [burnout_index], # Engineered Feature 1
    'Total_Stress': [total_stress]    # Engineered Feature 2
})

# Scaling (Optional: Only if you scaled in your notebook)
# input_data_scaled = scaler.transform(input_data) 

# --- 5. Real-Time Prediction Section ---
st.subheader("Assessment Results")

# Display the engineered metrics to show "Intelligence"
col1, col2 = st.columns(2)
col1.metric("Calculated Burnout Index", f"{burnout_index:.2f}")
col2.metric("Total Stress Score", f"{total_stress:.2f}")

# Predict Button
if st.button("Analyze Risk"):
    # Make Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.write("---")
    
    # Dynamic Output
    if prediction == 1:
        st.error(f"⚠️ **High Risk Detected** (Probability: {probability:.2%})")
        st.write("Recommendation: This student shows signs of high stress and potential depression. Immediate counseling intervention is recommended.")
    else:
        st.success(f"✅ **Low Risk** (Probability: {probability:.2%})")
        st.write("Recommendation: Student appears stable. Continue monitoring academic pressure levels.")

else:
    st.info("👈 Adjust details in the sidebar and click 'Analyze Risk' to see predictions.")