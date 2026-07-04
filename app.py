#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import pickle


# In[2]:


# 1. Load saved model & scaler
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


# In[3]:


# 2. Streamlit App UI
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")
st.title("🩺 Diabetes Prediction System")
st.write("Enter patient details to predict diabetes risk:")


# In[4]:


# Input fields for user
preg = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose")
bp = st.number_input("Blood Pressure")
skin = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

if st.button("Predict"):

    patient = pd.DataFrame({
        "Pregnancies":[preg],
        "Glucose":[glucose],
        "BloodPressure":[bp],
        "SkinThickness":[skin],
        "Insulin":[insulin],
        "BMI":[bmi],
        "DiabetesPedigreeFunction":[dpf],
        "Age":[age]
    })
    
    patient_scaled = scaler.transform(patient)

    prediction = model.predict(patient_scaled)

    probability = model.predict_proba(patient_scaled)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ The patient is likely to have Diabetes.")
    else:
        st.success("✅ The patient is NOT likely to have Diabetes.")

    st.write("### Probability")

    st.write(f"Diabetes : {probability[0][1]*100:.2f}%")
    st.write(f"No Diabetes : {probability[0][0]*100:.2f}%")
    
    


# In[5]:


"""# 3. Predict button
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
"""


# In[ ]:




