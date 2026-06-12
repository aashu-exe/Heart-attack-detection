import pandas as pd

df=pd.read_csv('C:\\College\\ML Pro\\HeartAttackPrediction\\data\\heart.csv')
print(df.head())
print(df.shape)
print(df.describe())
print(df.isnull().sum())

X = df.drop('target', axis=1)
Y = df['target']

print("Features shape: ", X.shape)
print("Target distribution:\n", Y.value_counts())

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("Training features shape: ", X_train.shape)
print("Testing features shape: ", X_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("First patients faetures before were raw numbers")
print("First patients features now: ", X_train[0])

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# model = LogisticRegression(max_iter=1000)
model = KNeighborsClassifier(n_neighbors=5)

model = model.fit(X_train, Y_train)

print("Model training completed.")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
Y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))

import joblib
import os

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/heart_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(X.columns.tolist(), "model/columns.pkl")

print("Model saved to model/heart_model.pkl")
print("Scaler saved to model/scaler.pkl")
print("Columns saved to model/columns.pkl")