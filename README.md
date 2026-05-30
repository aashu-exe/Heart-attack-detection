# Heart Attack Risk Predictor

A machine learning model that predicts whether a patient is at 
risk of a heart attack based on clinical measurements like age, 
cholesterol, blood pressure, and heart rate. Built with a 
Gradio web UI so anyone can enter patient details and get an 
instant prediction.

## Tech Stack

- **Python** — core language
- **Pandas** — loading and exploring the dataset
- **Scikit-learn** — preprocessing, train/test split, 
   feature scaling, model training and evaluation
- **Gradio** — interactive web UI for making predictions
- **Joblib** — saving and loading the trained model to disk

## How to Run

1. Clone the repo
   git clone https://github.com/aashu-exe/Heart-attack-detection
   cd heart-attack-prediction

2. Install dependencies
   pip install pandas scikit-learn gradio joblib

3. Download the dataset
   Get heart.csv from:
   https://www.kaggle.com/datasets/rashikrahmanpritom/
   heart-attack-analysis-prediction-dataset
   Place it in the data/ folder.

4. Train the model
   python train.py
   This saves the model and scaler to the model/ folder.

5. Launch the app
   python app.py
   Open http://127.0.0.1:7860 in your browser.

## Model Performance

Trained on 303 patients, 80/20 train/test split.

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.85  |
| Precision | 0.87  |
| Recall    | 0.84  |
| F1 Score  | 0.86  |

Confusion Matrix:
[[25  4]
 [ 5 27]]

Recall of 0.84 means the model correctly identified 84% of 
actual heart disease cases. The 5 missed cases (false 
negatives) are the critical failure mode — a patient with 
heart disease being told they are healthy. In a production 
system I would lower the prediction threshold from 0.5 to 0.3 
to prioritize catching more sick patients even at the cost of 
more false alarms.

## What I'd Improve

- **Lower prediction threshold** from 0.5 to 0.3 to improve 
  recall in a healthcare context
- **Try ensemble models** like Random Forest or XGBoost to 
  compare against Logistic Regression
- **Add feature importance chart** to show which symptoms 
  contribute most to the prediction
- **Deploy on Hugging Face Spaces** so it's accessible online 
  without local setup
