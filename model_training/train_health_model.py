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

data = pd.read_csv("healthcare_dataset.csv")

print("Dataset Preview:")
print(data.head())


# -----------------------------
# 2 Encode Categorical Columns
# -----------------------------

label_encoder_gender = LabelEncoder()
data["Gender"] = label_encoder_gender.fit_transform(data["Gender"])

label_encoder_smoking = LabelEncoder()
data["Smoking Status"] = label_encoder_smoking.fit_transform(data["Smoking Status"])

label_encoder_chronic = LabelEncoder()
data["Chronic Disease History"] = label_encoder_chronic.fit_transform(data["Chronic Disease History"])

label_encoder_target = LabelEncoder()
data["Health Risk Level"] = label_encoder_target.fit_transform(data["Health Risk Level"])


# -----------------------------
# 3 Define Features and Target
# -----------------------------

X = data.drop("Health Risk Level", axis=1)
y = data["Health Risk Level"]

feature_columns = X.columns.tolist()


# -----------------------------
# 4 Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# 5 Feature Scaling
# -----------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -----------------------------
# 6 Train ANN Model
# -----------------------------

model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=500,
    random_state=42
)

model.fit(X_train, y_train)


# -----------------------------
# 7 Model Evaluation
# -----------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# -----------------------------
# 8 Save Model + Preprocessing Objects
# -----------------------------

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# create models folder if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, "health_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "health_scaler.pkl"))

joblib.dump(label_encoder_gender, os.path.join(MODEL_DIR, "gender_encoder.pkl"))
joblib.dump(label_encoder_smoking, os.path.join(MODEL_DIR, "smoking_encoder.pkl"))
joblib.dump(label_encoder_chronic, os.path.join(MODEL_DIR, "chronic_encoder.pkl"))
joblib.dump(label_encoder_target, os.path.join(MODEL_DIR, "target_encoder.pkl"))

joblib.dump(feature_columns, os.path.join(MODEL_DIR, "health_features.pkl"))
print("\nModel and preprocessing files saved successfully!")