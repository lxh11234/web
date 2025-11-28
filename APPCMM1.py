# Load the new model
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Load the new model
model = joblib.load('stacking_classifier_model.pkl')

# Load the test data from X_test.csv to create LIME explainer
X_test = pd.read_csv('X_test.csv')

Duration_of_diabetes = {
    1: '≤1year',
    2: '1-5years',
    3: '＞5years'
}


# Define feature names from the new dataset
feature_names = ['PBG', 'HbAlc','Age','VFA','METS_IR','AST_ALT','PLT','Duration_of_diabetes','SCII']

# Streamlit user interface
st.title("CMM Predictor")

# Create two columns
col1, col2 = st.columns(2)

# Input fields in the left column
with col1:
    st.subheader("Input Features")
    Age = st.number_input("Age:", min_value=0, max_value=120, value=41)  # 统一为浮点数
    Duration_of_diabetes = st.selectbox("Duration.of.diabetes:", options=list(Duration_of_diabetes.keys()), format_func=lambda x: Duration_of_diabetes[x])
    SCII = st.selectbox("SCII (NO, YES):", options=[0, 1], format_func=lambda x: 'NO' if x == 0 else 'YES')
    PLT = st.number_input("PLT:", min_value=0.0, max_value=10000.0, value=157.0)  # 统一为浮点数
    AST_ALT = st.number_input("AST/ALT:", min_value=0.00, max_value=1000.00, value=1.00)  # 已经是浮点数
    PBG = st.number_input("PBG:", min_value=0.00, max_value=100.00, value=8.00)  # 已经是浮点数
    HbAlc = st.number_input("HbAlc:", min_value=0.0, max_value=50.0, value=6.0)
    VFA = st.number_input("VFA:", min_value=0.0, max_value=10000.0, value=90.0)  # 统一为浮点数
    METS_IR = st.number_input("METS_IR:", min_value=0.00, max_value=500.00, value=22.00)
with col2:
    st.subheader("Prediction Results")
    feature_values = [PBG, HbAlc, Age, VFA, METS_IR, AST_ALT, PLT, Duration_of_diabetes, SCII]
    features = np.array([feature_values]).reshape(1, -1)

    if st.button("Predict"):
        # Predict class and probabilities
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]

        # Display prediction results
        st.write(f"**Predicted Class:** {predicted_class} (0: No Disease, 1: Disease)")
        st.write(f"**Prediction Probabilities:** {predicted_proba}")

        # Generate advice based on prediction results
        probability = predicted_proba[predicted_class] * 100

        if predicted_class == 1:
            advice = (
                f"According to our model, you have a high risk of CMM disease. "
                f"The model predicts that your probability of having CMM disease is {probability:.1f}%. "
                "It's advised to consult with your healthcare provider for further evaluation and possible intervention."
            )
        else:
            advice = (
                f"According to our model, you have a low risk of CMM disease. "
                f"The model predicts that your probability of not having CMM disease is {probability:.1f}%. "
                "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."
            )

        st.write(advice)

        # SHAP Explanation
        st.subheader("SHAP Force Plot Explanation")
        explainer_shap = shap.KernelExplainer(model.predict_proba, X_test)
        shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))

        # Display the SHAP force plot for the predicted class
        if predicted_class == 1:
            shap.force_plot(explainer_shap.expected_value[1], shap_values[:, :, 1],
                            pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
        else:
            shap.force_plot(explainer_shap.expected_value[0], shap_values[:, :, 0],
                            pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)

        plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
        st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')

        # LIME Explanation
        st.subheader("LIME Explanation")
        lime_explainer = LimeTabularExplainer(
            training_data=X_test.values,
            feature_names=X_test.columns.tolist(),
            class_names=['Not sick', 'sick'],  # Adjust class names to match your classification task
            mode='classification'
        )

        # Explain the instance
        lime_exp = lime_explainer.explain_instance(
            data_row=features.flatten(),
            predict_fn=model.predict_proba
        )

        # Display the LIME explanation without the feature value table
        lime_html = lime_exp.as_html(show_table=False)  # Disable feature value table
        st.components.v1.html(lime_html, height=800, scrolling=True)
