import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight


# -----------------------------
# Load Dataset
# -----------------------------

data = pd.read_csv("cardio_train.csv", sep=";")

print("Dataset Preview:")
print(data.head())


# -----------------------------
# Convert Age (days → years)
# -----------------------------

data["age"] = data["age"] / 365


# -----------------------------
# BMI Calculation
# -----------------------------

data["BMI"] = data["weight"] / ((data["height"] / 100) ** 2)


# -----------------------------
# Rename columns to match API
# -----------------------------

data.rename(columns={

    "age": "Age",
    "gender": "Sex",
    "cholesterol": "Cholesterol",
    "ap_hi": "Systolic_BP",
    "ap_lo": "Diastolic_BP",
    "smoke": "Smoking",
    "alco": "Alcohol",
    "active": "Physical_Activity",
    "gluc": "Glucose",
    "cardio": "Heart_Attack_Risk"

}, inplace=True)


# -----------------------------
# Feature Engineering
# -----------------------------

data["Age_Cholesterol_Risk"] = data["Age"] * data["Cholesterol"]
data["BMI_Stress_Risk"] = data["BMI"] * data["Glucose"]


# -----------------------------
# Target
# -----------------------------

y = data["Heart_Attack_Risk"]


# -----------------------------
# Feature Selection
# -----------------------------

X = data[[
    "Age",
    "Sex",
    "Cholesterol",
    "Smoking",
    "Alcohol",
    "Physical_Activity",
    "Glucose",
    "BMI",
    "Systolic_BP",
    "Diastolic_BP",
    "Age_Cholesterol_Risk",
    "BMI_Stress_Risk"
]]

print("\nFeatures used for training:")
print(X.columns.tolist())


# -----------------------------
# Train/Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42

)


# -----------------------------
# Feature Scaling
# -----------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -----------------------------
# Handle Class Imbalance
# -----------------------------

classes = np.unique(y_train)

weights = compute_class_weight(

    class_weight="balanced",
    classes=classes,
    y=y_train

)

class_weights = dict(zip(classes, weights))

sample_weights = np.array([class_weights[i] * 0.7 for i in y_train])


# -----------------------------
# Deep Learning Model (ANN)
# -----------------------------

model = MLPClassifier(

    hidden_layer_sizes=(96, 48),

    activation="relu",

    solver="adam",

    learning_rate="adaptive",

    batch_size=32,

    alpha=0.01,

    max_iter=4000,

    early_stopping=True,

    validation_fraction=0.1,

    random_state=42

)


# -----------------------------
# Train Model
# -----------------------------

model.fit(X_train, y_train, sample_weight=sample_weights)


# -----------------------------
# Evaluate Model
# -----------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# -----------------------------
# Save Model
# -----------------------------

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, "heart_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "heart_scaler.pkl"))

print("\nHeart disease model saved successfully!")