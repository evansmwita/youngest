import streamlit as st
import joblib
import numpy as np
from pathlib import Path

# Set page layout
st.set_page_config(layout="wide")

# Cache the model loading to avoid reloading each time
@st.cache_resource
def load_model():
    model_path = Path("random_forest_model.pkl")
    if not model_path.exists():
        st.error("Model file not found! Please ensure 'random_forest_model.pkl' is in the repo.")
        return None
    return joblib.load(model_path)

model = load_model()

# Sidebar navigation
page = st.sidebar.selectbox("Select a page", ["Home", "Prediction"])

# Home page
if page == "Home":
    st.title("Diabetes Prediction Model Overview")
    st.write("""
    Welcome to the **Diabetes Prediction Application**.
    
    This tool uses a machine learning model (**Random Forest Classifier**) 
    trained on health metrics such as:
    - Pregnancies
    - Glucose
    - Blood Pressure
    - Skin Thickness
    - Insulin
    - BMI
    - Diabetes Pedigree Function
    - Age
    
    Navigate to the **Prediction** page to enter patient details and get predictions.
    """)

# Prediction page
elif page == "Prediction":
    st.title("Diabetes Prediction App - Prediction")
    st.write("Enter the patient's details to get a diabetes prediction:")

    # Input fields
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=130, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("Age", min_value=0, max_value=120, value=30)

    # Predict button
    if st.button("Predict"):
        if model is not None:
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                    insulin, bmi, diabetes_pedigree_function, age]])
            
            prediction = model.predict(input_data)

            if prediction[0] == 1:
                st.error("🚨 The model predicts that the patient is **likely to have diabetes**.")
            else:
                st.success("✅ The model predicts that the patient is **unlikely to have diabetes**.")
