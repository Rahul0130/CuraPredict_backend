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
# Load NHANES Tables
# -----------------------------

demo = pd.read_csv("demographic.csv")
exam = pd.read_csv("examination.csv")
quest = pd.read_csv("questionnaire.csv")
labs = pd.read_csv("labs.csv")


# -----------------------------
# Merge Tables
# -----------------------------

data = demo.merge(exam, on="SEQN", how="inner")
data = data.merge(quest, on="SEQN", how="inner")
data = data.merge(labs, on="SEQN", how="inner")

print("Merged dataset size:", data.shape)


# -----------------------------
# Target Variable
# -----------------------------

data["target"] = data["HSD010"].map({
    1: 0,
    2: 0,
    3: 0,
    4: 1,
    5: 1
})

data = data.dropna(subset=["target"])


# -----------------------------
# Feature Engineering
# -----------------------------

data["Age"] = data["RIDAGEYR"]

data["sex"] = data["RIAGENDR"].map({
    1: 1,
    2: 0
})

data["bmi"] = data["BMXBMI"]


# -----------------------------
# Smoking
# -----------------------------

data["smoking"] = data["SMQ020"].apply(
    lambda x: 1 if x == 1 else 0
)


# -----------------------------
# Alcohol Consumption
# -----------------------------

data["alcohol"] = data["ALQ101"]

# remove invalid NHANES codes
data["alcohol"] = data["alcohol"].replace([7, 9, 77, 99], np.nan)

# limit extreme values
data["alcohol"] = data["alcohol"].clip(0, 20)


# -----------------------------
# Physical Activity
# -----------------------------

# PAQ605: 1 = Yes exercise, 2 = No exercise
data["physical_activity"] = data["PAQ605"].map({
    1: 5,
    2: 0
})


# -----------------------------
# Sleep Duration
# -----------------------------

data["sleep_duration"] = data["SLD010H"]

data["sleep_duration"] = data["sleep_duration"].replace([77, 99], np.nan)


# -----------------------------
# Diabetes
# -----------------------------

data["diabetes"] = data["DIQ010"].apply(
    lambda x: 1 if x == 1 else 0
)


# -----------------------------
# Stress Proxy
# -----------------------------

data["stress_level"] = data["BPXDI1"]


# -----------------------------
# Optional Medical Features
# -----------------------------

if "LBXTC" in data.columns:
    data["cholesterol"] = data["LBXTC"]

if "LBXTR" in data.columns:
    data["triglycerides"] = data["LBXTR"]

if "BPXSY1" in data.columns:
    data["systolic_bp"] = data["BPXSY1"]

if "BPXDI1" in data.columns:
    data["diastolic_bp"] = data["BPXDI1"]

if "BPXPLS" in data.columns:
    data["heart_rate"] = data["BPXPLS"]


# -----------------------------
# Select Features
# -----------------------------

features = [
    "Age",
    "sex",
    "bmi",
    "smoking",
    "alcohol",
    "physical_activity",
    "sleep_duration",
    "diabetes",
    "stress_level"
]

optional_features = [
    "cholesterol",
    "triglycerides",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate"
]

for col in optional_features:
    if col in data.columns:
        features.append(col)

data = data[features + ["target"]]


# -----------------------------
# Clean Data
# -----------------------------

data = data.replace([np.inf, -np.inf], np.nan)

data = data.dropna()

print("Final dataset size:", data.shape)


# -----------------------------
# Split Features / Target
# -----------------------------

X = data.drop("target", axis=1)
y = data["target"]


# -----------------------------
# Train Test Split
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

sample_weights = np.array([class_weights[i] for i in y_train])


# -----------------------------
# ANN Model
# -----------------------------

model = MLPClassifier(

    hidden_layer_sizes=(128, 64, 32),

    activation="relu",

    solver="adam",

    learning_rate="adaptive",

    batch_size=32,

    alpha=0.01,

    max_iter=3000,

    early_stopping=True,

    random_state=42
)


# -----------------------------
# Train Model
# -----------------------------

model.fit(X_train, y_train, sample_weight=sample_weights)


# -----------------------------
# Evaluate Model
# -----------------------------

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, pred))


# -----------------------------
# Save Model
# -----------------------------

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, "health_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "health_scaler.pkl"))

print("\nHealth model saved successfully!")