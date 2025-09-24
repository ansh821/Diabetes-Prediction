#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import joblib


# In[6]:


# 1. Load saved model & scaler
# -------------------------------
log_reg = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")


# In[7]:


# 2. Streamlit App UI
# -------------------------------
st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict diabetes risk:")


# In[8]:


# Input fields for user
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)


# In[9]:


# 3. Predict button
# -------------------------------
if st.button("Predict"):
    # Collect input features
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, dpf, age]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = log_reg.predict(features_scaled)[0]
    probability = log_reg.predict_proba(features_scaled)[0][1]
    
    # Display result
    if prediction == 1:
        st.error(f"⚠️ High Risk: The patient is **Diabetic** (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk: The patient is **Non-Diabetic** (Probability: {probability:.2f})")


# In[ ]:




