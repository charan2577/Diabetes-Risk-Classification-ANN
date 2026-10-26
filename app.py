"""
=============================================================================
DIABETES RISK CLASSIFICATION USING MULTI-LAYER PERCEPTRON (MLP)
Interactive Web Application: app.py
=============================================================================
Academic Project: CDC BRFSS 2015 Diabetes Health Indicators Prediction
Framework: Streamlit
Model: Artificial Neural Network (MLPClassifier) trained via Full-Batch SGD

This Streamlit application provides an interactive clinical prediction interface:
  1. Accepts user inputs across all 21 CDC health indicators in the sidebar
  2. Loads the serialized MLP model and feature scaler from models/
  3. Preprocesses input indicators to match training scale
  4. Generates multi-class risk predictions (Healthy, Prediabetes, Diabetes)
  5. Renders color-coded diagnostic alert cards and probability bar charts
  6. Displays detailed guidance on running and interpreting the application

Usage:
    streamlit run app.py
=============================================================================
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path Configuration: Ensure project root is present in Python module search path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Classification — MLP Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Model and Scaler Loading Utilities
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mlp_model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")

# NOTE ON FEATURE SCALER:
# Neural networks (MLP) are sensitive to feature scales. The scaler (e.g., StandardScaler)
# fitted during model training in train.py should be saved to models/scaler.pkl.
# In app.py, new patient inputs must be transformed using this saved scaler before
# passing them to model.predict() or model.predict_proba().

@st.cache_resource(show_spinner=False)
def load_saved_model(path):
    """Loads and caches the serialized MLP classifier model."""
    if not os.path.exists(path):
        return None
    try:
        # Attempt to import load_model from src.model if available
        try:
            from src.model import load_model
            return load_model(path)
        except Exception:
            return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model from {path}: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_saved_scaler(path):
    """Loads and caches the serialized feature scaler."""
    if not os.path.exists(path):
        return None
    try:
        # Attempt to import load_scaler from src.preprocessing if available
        try:
            from src.preprocessing import load_scaler
            return load_scaler(path)
        except Exception:
            return joblib.load(path)
    except Exception as e:
        st.warning(f"Could not load scaler from {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Header & Introductory Overview
# ---------------------------------------------------------------------------
st.title("🩺 Diabetes Risk Classification — MLP Predictor")
st.markdown("""
This academic application implements an **Artificial Neural Network (Multi-Layer Perceptron)** 
trained on the **CDC Behavioral Risk Factor Surveillance System (BRFSS 2015)** dataset 
comprising **253,680 records** and **21 health indicators**.

The neural network predicts diabetes risk across three clinical categories:
- **Class 0:** Healthy (No Diabetes or Gestational Only)
- **Class 1:** Prediabetes (Elevated blood glucose / borderline)
- **Class 2:** Diabetes (Diagnosed type 1 or type 2 diabetes)
""")

st.divider()

# ---------------------------------------------------------------------------
# Sidebar: User Input Controls for all 21 Features
# ---------------------------------------------------------------------------
st.sidebar.header("📋 Patient Health Indicators (21 Features)")
st.sidebar.markdown("Adjust the clinical, lifestyle, and demographic inputs below:")

# Feature 1: HighBP
high_bp = st.sidebar.selectbox(
    "1. High Blood Pressure (HighBP)",
    options=[0, 1],
    index=0,
    format_func=lambda x: "0 - Normal / No High BP" if x == 0 else "1 - Diagnosed High BP",
    help="Adults who have been told they have high blood pressure."
)

# Feature 2: HighChol
high_chol = st.sidebar.selectbox(
    "2. High Cholesterol (HighChol)",
    options=[0, 1],
    index=0,
    format_func=lambda x: "0 - Normal / No High Cholesterol" if x == 0 else "1 - Diagnosed High Cholesterol",
    help="Adults who have been told they have high blood cholesterol."
)

# Feature 3: CholCheck
chol_check = st.sidebar.selectbox(
    "3. Cholesterol Checked (CholCheck)",
    options=[0, 1],
    index=1,
    format_func=lambda x: "0 - Not Checked in Past 5 Years" if x == 0 else "1 - Checked in Past 5 Years",
    help="Cholesterol check completed within past five years."
)

# Feature 4: BMI
bmi = st.sidebar.slider(
    "4. Body Mass Index (BMI)",
    min_value=10.0,
    max_value=80.0,
    value=25.0,
    step=0.5,
    help="Body Mass Index calculated as weight in kg / (height in meters)^2."
)

# Feature 5: Smoker
smoker = st.sidebar.selectbox(
    "5. Smoker Status (Smoker)",
    options=[0, 1],
    index=0,
    format_func=lambda x: "0 - No (< 100 cigarettes in lifetime)" if x == 0 else "1 - Yes (>= 100 cigarettes in lifetime)",
    help="Have you smoked at least 100 cigarettes in your entire life?"
)

# Feature 6: Stroke
stroke = st.sidebar.selectbox(
    "6. History of Stroke (Stroke)",
    options=[0, 1],
    index=0,
    format_func=lambda x: "0 - No History of Stroke" if x == 0 else "1 - History of Stroke",
    help="Ever told by a medical professional that you had a stroke."
)

# Feature 7: HeartDiseaseorAttack
heart_disease = st.sidebar.selectbox(
    "7. Heart Disease or Attack (HeartDiseaseorAttack)",
    options=[0, 1],
    index=0,
    format_func=lambda x: "0 - No Heart Disease / MI" if x == 0 else "1 - Coronary Heart Disease or MI",
    help="Reported coronary heart disease (CHD) or myocardial infarction (MI)."
)

# Feature 8: PhysActivity
phys_activity = st.sidebar.selectbox(
    "8. Physical Activity (PhysActivity)",
    options=[0, 1],
    index=1,
    format_func=lambda x: "0 - No Leisure Activity" if x == 0 else "1 - Active in Past 30 Days",
    help="Engaged in physical activity or exercise during past 30 days other than regular job."
)

# Feature 9: Fruits
fruits = st.sidebar.selectbox(
    "9. Daily Fruit Consumption (Fruits)",
    options=[0, 1],
    index=1,
    format_func=lambda x: "0 - Less than 1 serving/day" if x == 0 else "1 - Consumes fruit >= 1 time/day",
    help="Consume fruit 1 or more times per day."
)

# Feature 10: Veggies
veggies = st.sidebar.selectbox(
    "10. Daily Vegetable Consumption (Veggies)",
    options=[0, 1],
    index=1,
    format_func=lambda x: "0 - Less than 1 serving/day" if x == 0 else "1 - Consumes vegetables >= 1 time/day",
    help="Consume vegetables 1 or more times per day."
)

# Feature 11: HvyAlcoholConsump
heavy_alcohol = st.sidebar.selectbox(
    "11. Heavy Alcohol Consumption (HvyAlcoholConsump)",
    options=[0, 1],
    index=0,
    format_func=lambda x: "0 - No (Moderate or Non-Drinker)" if x == 0 else "1 - Yes (Heavy Drinker: Men >14 drinks/wk, Women >7 drinks/wk)",
    help="Adult men having >14 drinks per week; adult women having >7 drinks per week."
)

# Feature 12: AnyHealthcare
any_healthcare = st.sidebar.selectbox(
    "12. Healthcare Coverage (AnyHealthcare)",
    options=[0, 1],
    index=1,
    format_func=lambda x: "0 - No Healthcare Coverage" if x == 0 else "1 - Has Any Healthcare Coverage",
    help="Any health care insurance or coverage, including HMOs or government plans."
)

# Feature 13: NoDocbcCost
no_doc_cost = st.sidebar.selectbox(
    "13. Doctor Visit Cost Barrier (NoDocbcCost)",
    options=[0, 1],
    index=0,
    format_func=lambda x: "0 - No Cost Barrier" if x == 0 else "1 - Needed Doctor but Could Not Afford",
    help="Was there a time in past 12 months needed to see a doctor but could not due to cost?"
)

# Feature 14: GenHlth
gen_hlth = st.sidebar.slider(
    "14. General Health Rating (GenHlth: 1-5)",
    min_value=1,
    max_value=5,
    value=2,
    step=1,
    help="Self-reported health: 1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor."
)

# Feature 15: MentHlth
ment_hlth = st.sidebar.slider(
    "15. Poor Mental Health Days (MentHlth: 0-30)",
    min_value=0,
    max_value=30,
    value=0,
    step=1,
    help="Number of days during the past 30 days mental health was not good (stress, depression)."
)

# Feature 16: PhysHlth
phys_hlth = st.sidebar.slider(
    "16. Physical Illness/Injury Days (PhysHlth: 0-30)",
    min_value=0,
    max_value=30,
    value=0,
    step=1,
    help="Number of days during past 30 days physical health was not good (illness, injury)."
)

# Feature 17: DiffWalk
diff_walk = st.sidebar.selectbox(
    "17. Difficulty Walking / Stairs (DiffWalk)",
    options=[0, 1],
    index=0,
    format_func=lambda x: "0 - No Difficulty Walking" if x == 0 else "1 - Serious Difficulty Walking/Climbing Stairs",
    help="Do you have serious difficulty walking or climbing stairs?"
)

# Feature 18: Sex
sex = st.sidebar.selectbox(
    "18. Biological Sex (Sex: 0=Female, 1=Male)",
    options=[0, 1],
    index=0,
    format_func=lambda x: "0 - Female" if x == 0 else "1 - Male",
    help="Biological sex of the participant."
)

# Feature 19: Age
age = st.sidebar.slider(
    "19. Age Category (Age: 1-13)",
    min_value=1,
    max_value=13,
    value=7,
    step=1,
    help="13-level age categories: 1=18-24, 2=25-29, 3=30-34, ..., 7=50-54, ..., 13=80+."
)

# Feature 20: Education
education = st.sidebar.slider(
    "20. Education Level (Education: 1-6)",
    min_value=1,
    max_value=6,
    value=4,
    step=1,
    help="1=Never attended school, 2=Elementary, 3=Some high school, 4=High school grad, 5=Some college, 6=College 4+ yrs."
)

# Feature 21: Income
income = st.sidebar.slider(
    "21. Household Income Scale (Income: 1-8)",
    min_value=1,
    max_value=8,
    value=5,
    step=1,
    help="1=<$10k, 2=<$15k, 3=<$20k, 4=<$25k, 5=<$35k, 6=<$50k, 7=<$75k, 8=>$75k."
)


# ---------------------------------------------------------------------------
# Construct Patient Input DataFrame (Exact 21 Feature Schema)
# ---------------------------------------------------------------------------
patient_data = {
    'HighBP': [high_bp],
    'HighChol': [high_chol],
    'CholCheck': [chol_check],
    'BMI': [float(bmi)],
    'Smoker': [smoker],
    'Stroke': [stroke],
    'HeartDiseaseorAttack': [heart_disease],
    'PhysActivity': [phys_activity],
    'Fruits': [fruits],
    'Veggies': [veggies],
    'HvyAlcoholConsump': [heavy_alcohol],
    'AnyHealthcare': [any_healthcare],
    'NoDocbcCost': [no_doc_cost],
    'GenHlth': [gen_hlth],
    'MentHlth': [ment_hlth],
    'PhysHlth': [phys_hlth],
    'DiffWalk': [diff_walk],
    'Sex': [sex],
    'Age': [age],
    'Education': [education],
    'Income': [income]
}

input_df = pd.DataFrame(patient_data)

# ---------------------------------------------------------------------------
# Main Display Layout: Input Review & Prediction Trigger
# ---------------------------------------------------------------------------
col_main, col_summary = st.columns([2, 1])

with col_main:
    st.subheader("Selected Clinical Indicators Overview")
    st.dataframe(input_df, use_container_width=True)

with col_summary:
    st.subheader("Patient Quick Facts")
    sex_label = "Female" if sex == 0 else "Male"
    bp_label = "High BP" if high_bp == 1 else "Normal BP"
    chol_label = "High Chol" if high_chol == 1 else "Normal Chol"
    st.info(f"""
    - **Sex / BMI:** {sex_label}, BMI = {bmi:.1f}
    - **Vitals:** {bp_label}, {chol_label}
    - **General Health:** Rating {gen_hlth}/5
    - **Age Category:** Level {age}/13
    """)

# ---------------------------------------------------------------------------
# Prediction Execution & Results Visualization
# ---------------------------------------------------------------------------
st.subheader("Generate Clinical Risk Classification")
predict_button = st.button("🔮 Predict Diabetes Risk", type="primary", use_container_width=True)

if predict_button:
    # 1. Error Handling: Verify existence of trained model
    if not os.path.exists(MODEL_PATH):
        st.error(f"""
        ❌ **Model File Not Found!**
        
        The trained MLP model could not be found at `{MODEL_PATH}`.
        
        **To resolve this issue:**
        1. Open your terminal in the project directory:
           `c:\\Users\\thumm\\OneDrive\\Desktop\\Diabetes-Risk-Classification-ANN`
        2. Run the training script:
           ```bash
           python train.py
           ```
        3. Once training completes, return here and re-click **Predict Diabetes Risk**.
        """)
    else:
        with st.spinner("Loading MLP model and analyzing clinical risk factors..."):
            model = load_saved_model(MODEL_PATH)
            scaler = load_saved_scaler(SCALER_PATH)
            
            # Apply feature scaling if scaler exists
            if scaler is not None:
                try:
                    scaled_features = scaler.transform(input_df)
                except Exception as e:
                    st.warning(f"Feature scaling warning: {e}. Passing raw features to model.")
                    scaled_features = input_df.values
            else:
                # Scaler file not found comment & fallback
                st.info("ℹ️ Scaler file (`models/scaler.pkl`) not detected. Using raw input values. (For highest accuracy, ensure scaler is saved during `python train.py` execution.)")
                scaled_features = input_df.values
                
            # Compute classification prediction and class probabilities
            try:
                prediction = model.predict(scaled_features)[0]
                probabilities = model.predict_proba(scaled_features)[0]
            except Exception as e:
                st.error(f"Inference error during model prediction: {e}")
                prediction = None
                probabilities = None

        if prediction is not None and probabilities is not None:
            pred_class = int(prediction)
            
            # Map probabilities for the 3 target classes
            prob_healthy = probabilities[0]
            prob_prediabetes = probabilities[1] if len(probabilities) > 1 else 0.0
            prob_diabetes = probabilities[2] if len(probabilities) > 2 else (1.0 - prob_healthy)
            
            st.divider()
            st.subheader("Classification Outcome")

            # Colored Markdown Alert Cards for each condition
            if pred_class == 0:
                st.markdown("""
                <div style="background-color: #d4edda; color: #155724; padding: 20px; border-radius: 10px; border-left: 8px solid #28a745; margin-bottom: 20px;">
                    <h2 style="margin: 0 0 10px 0; color: #155724;">🟢 Predicted Class: 0 — Healthy (No Diabetes)</h2>
                    <p style="font-size: 16px; margin: 0;">
                        The Multi-Layer Perceptron neural network predicts that the patient does <strong>not</strong> exhibit significant indicators of prediabetes or diabetes.
                        Continued regular exercise, a balanced diet, and periodic health screenings are recommended to sustain metabolic health.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            elif pred_class == 1:
                st.markdown("""
                <div style="background-color: #fff3cd; color: #856404; padding: 20px; border-radius: 10px; border-left: 8px solid #fd7e14; margin-bottom: 20px;">
                    <h2 style="margin: 0 0 10px 0; color: #856404;">🟠 Predicted Class: 1 — Prediabetes</h2>
                    <p style="font-size: 16px; margin: 0;">
                        The Multi-Layer Perceptron neural network detects clinical and lifestyle patterns consistent with <strong>prediabetes</strong> (elevated glycemic risk).
                        Early intervention through nutritional counseling, targeted weight management, and routine laboratory screening (HbA1c / fasting glucose) is strongly recommended.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            else:  # pred_class == 2
                st.markdown("""
                <div style="background-color: #f8d7da; color: #721c24; padding: 20px; border-radius: 10px; border-left: 8px solid #dc3545; margin-bottom: 20px;">
                    <h2 style="margin: 0 0 10px 0; color: #721c24;">🔴 Predicted Class: 2 — Diabetes</h2>
                    <p style="font-size: 16px; margin: 0;">
                        The Multi-Layer Perceptron neural network identifies high-probability indicators associated with <strong>diabetes mellitus</strong>.
                        A clinical medical evaluation, comprehensive diagnostic laboratory workup (fasting plasma glucose, HbA1c, lipid panel), and physician follow-up are strongly advised.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Probabilities Visualization
            st.subheader("📊 Class Confidence & Probability Breakdown")
            
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                st.metric(
                    label="Class 0: Healthy",
                    value=f"{prob_healthy * 100:.2f}%",
                    delta=f"{'+' if pred_class == 0 else ''}{(prob_healthy - 0.33) * 100:.1f}% vs random"
                )
            with p_col2:
                st.metric(
                    label="Class 1: Prediabetes",
                    value=f"{prob_prediabetes * 100:.2f}%",
                    delta=f"{'+' if pred_class == 1 else ''}{(prob_prediabetes - 0.33) * 100:.1f}% vs random"
                )
            with p_col3:
                st.metric(
                    label="Class 2: Diabetes",
                    value=f"{prob_diabetes * 100:.2f}%",
                    delta=f"{'+' if pred_class == 2 else ''}{(prob_diabetes - 0.33) * 100:.1f}% vs random"
                )

            # Probability Bar Chart
            prob_chart_data = pd.DataFrame({
                "Risk Category": ["Healthy (0)", "Prediabetes (1)", "Diabetes (2)"],
                "Confidence Probability": [prob_healthy, prob_prediabetes, prob_diabetes]
            }).set_index("Risk Category")

            st.bar_chart(prob_chart_data, y="Confidence Probability", color="#2b5c8f")

# ---------------------------------------------------------------------------
# Instructions & Technical Project Notes
# ---------------------------------------------------------------------------
st.divider()
st.subheader("💡 Application Notes & Instructions")

with st.expander("📖 How to Run This Application", expanded=False):
    st.markdown("""
    To launch this Streamlit web interface locally from your terminal or command prompt:
    ```bash
    streamlit run app.py
    ```
    Ensure dependencies (`streamlit`, `scikit-learn`, `pandas`, `numpy`, `joblib`) are installed.
    """)

with st.expander("🔬 Model & Optimization Details (Full-Batch Gradient Descent)", expanded=False):
    st.markdown("""
    - **Architecture:** Multi-Layer Perceptron (Artificial Neural Network)
    - **Optimization Technique:** Full-Batch Gradient Descent (`solver='sgd'`, `batch_size=N_train`)
    - **Data Source:** CDC Behavioral Risk Factor Surveillance System (BRFSS 2015), 253,680 records, 21 health indicator features.
    - **Target Column:** `Diabetes_012` (0=Healthy, 1=Prediabetes, 2=Diabetes).
    - **Academic Disclaimer:** This software is developed strictly for educational and academic research purposes. It does not replace professional clinical diagnostic procedures.
    """)

st.caption("Diabetes Risk Classification Using MLP | Academic Project BRFSS 2015 | Full-Batch Gradient Descent")
