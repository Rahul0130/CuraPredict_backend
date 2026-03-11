import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# 1 Load Dataset
# -----------------------------

data = pd.read_csv("heart_attack_dataset.csv")

# Remove missing values
data = data.dropna()

print("Dataset Preview:")
print(data.head())


# -----------------------------
# 2 Remove Irrelevant Columns
# -----------------------------

data = data.drop(
    columns=[
        "Patient ID",
        "Country",
        "Continent",
        "Hemisphere"
    ],
    errors="ignore"
)


# -----------------------------
# 3 Split Blood Pressure
# -----------------------------

bp_split = data["Blood Pressure"].str.split("/", expand=True)

data["Systolic_BP"] = bp_split[0].astype(int)
data["Diastolic_BP"] = bp_split[1].astype(int)

data = data.drop(columns=["Blood Pressure"])


# -----------------------------
# 4 Encode Categorical Columns
# -----------------------------

encoders = {}

categorical_cols = [
    "Sex",
    "Smoking",
    "Diabetes",
    "Family History",
    "Obesity",
    "Alcohol Consumption",
    "Diet",
    "Previous Heart Problems",
    "Medication Use",
    "Stress Level"
]

for col in categorical_cols:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])
    encoders[col] = encoder


# -----------------------------
# 5 Define Features and Target
# -----------------------------

target_column = "Heart Attack Risk"

X = data.drop(target_column, axis=1)
y = data[target_column].astype(int)

feature_columns = X.columns.tolist()

print("\nFeatures used for training:")
print(feature_columns)


# -----------------------------
# 6 Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# 7 Feature Scaling
# -----------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -----------------------------
# 8 Train ANN Model
# -----------------------------

model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=800,
    random_state=42
)

model.fit(X_train, y_train)


# -----------------------------
# 9 Evaluate Model
# -----------------------------

predictions = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))


# -----------------------------
# 10 Save Model + Preprocessing
# -----------------------------

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Save model
joblib.dump(model, os.path.join(MODEL_DIR, "heart_model.pkl"))

# Save scaler
joblib.dump(scaler, os.path.join(MODEL_DIR, "heart_scaler.pkl"))

# Save encoders
for col, encoder in encoders.items():
    joblib.dump(
        encoder,
        os.path.join(MODEL_DIR, f"heart_{col}_encoder.pkl")
    )

# Save feature order
joblib.dump(
    feature_columns,
    os.path.join(MODEL_DIR, "heart_features.pkl")
)

print("\nHeart disease model saved successfully!")