import gradio as gr
import joblib
import numpy as np

model = joblib.load("model/heart_model.pkl")
scaler = joblib.load("model/scaler.pkl")

def predict_heart_attack(age, sex, cp, trestbps, chol,
                          fbs, restecg, thalach, exang,
                          oldpeak, slope, ca, thal):

    features = np.array([[age, sex, cp, trestbps, chol,
                           fbs, restecg, thalach, exang,
                           oldpeak, slope, ca, thal]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    if prediction == 1:
        return f"⚠️ High Risk — {probability*100:.1f}% probability of heart disease"
    else:
        return f"✅ Low Risk — {probability*100:.1f}% probability of heart disease"
    

interface = gr.Interface(
    fn=predict_heart_attack,
    inputs=[
        gr.Slider(29, 77, label="Age"),
        gr.Radio([0, 1], label="Sex (0=Female, 1=Male)"),
        gr.Slider(0, 3, step=1, label="Chest Pain Type (0-3)"),
        gr.Slider(90, 200, label="Resting Blood Pressure"),
        gr.Slider(100, 600, label="Cholesterol"),
        gr.Radio([0, 1], label="Fasting Blood Sugar > 120mg (0=No, 1=Yes)"),
        gr.Slider(0, 2, step=1, label="Resting ECG (0-2)"),
        gr.Slider(70, 210, label="Max Heart Rate"),
        gr.Radio([0, 1], label="Exercise Induced Angina (0=No, 1=Yes)"),
        gr.Slider(0.0, 6.2, label="ST Depression (Oldpeak)"),
        gr.Slider(0, 2, step=1, label="Slope of ST Segment (0-2)"),
        gr.Slider(0, 4, step=1, label="Number of Major Vessels (0-4)"),
        gr.Slider(0, 3, step=1, label="Thalassemia (0-3)"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Heart Attack Risk Predictor",
    description="Enter patient details to predict heart attack risk."
)

interface.launch()