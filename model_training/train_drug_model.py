import pandas as pd
import joblib
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# 1 Load Train + Test Dataset
# -----------------------------

train_data = pd.read_csv("drugsComTrain_raw.csv")
test_data = pd.read_csv("drugsComTest_raw.csv")

# Remove missing values
train_data = train_data.dropna()
test_data = test_data.dropna()

print("Train Dataset Preview:")
print(train_data.head())


# -----------------------------
# 2 Convert Rating → Effectiveness
# -----------------------------

train_data["drug_effectiveness"] = train_data["rating"].apply(lambda x: 1 if x >= 7 else 0)
test_data["drug_effectiveness"] = test_data["rating"].apply(lambda x: 1 if x >= 7 else 0)


# -----------------------------
# 3 Keep Important Columns
# -----------------------------

columns = ["drugName", "condition", "usefulCount", "drug_effectiveness"]

train_data = train_data[columns]
test_data = test_data[columns]


# -----------------------------
# 4 Reduce Dataset Noise
# -----------------------------

# keep only top 30 drugs
top_drugs = train_data["drugName"].value_counts().head(30).index

train_data = train_data[train_data["drugName"].isin(top_drugs)]
test_data = test_data[test_data["drugName"].isin(top_drugs)]

# keep top 20 conditions
top_conditions = train_data["condition"].value_counts().head(20).index

train_data = train_data[train_data["condition"].isin(top_conditions)]
test_data = test_data[test_data["condition"].isin(top_conditions)]


# -----------------------------
# 5 Encode Categorical Columns
# -----------------------------

# combine datasets to fit encoder safely
combined = pd.concat([train_data, test_data])

drug_encoder = LabelEncoder()
condition_encoder = LabelEncoder()

drug_encoder.fit(combined["drugName"])
condition_encoder.fit(combined["condition"])

train_data["drugName"] = drug_encoder.transform(train_data["drugName"])
train_data["condition"] = condition_encoder.transform(train_data["condition"])

test_data["drugName"] = drug_encoder.transform(test_data["drugName"])
test_data["condition"] = condition_encoder.transform(test_data["condition"])


# -----------------------------
# 6 Define Features and Target
# -----------------------------

X_train = train_data.drop("drug_effectiveness", axis=1)
y_train = train_data["drug_effectiveness"].astype(int)

X_test = test_data.drop("drug_effectiveness", axis=1)
y_test = test_data["drug_effectiveness"].astype(int)

feature_columns = X_train.columns.tolist()

print("\nFeatures used for training:")
print(feature_columns)


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
    hidden_layer_sizes=(128, 64),
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

# save model
joblib.dump(model, os.path.join(MODEL_DIR, "drug_model.pkl"))

# save scaler
joblib.dump(scaler, os.path.join(MODEL_DIR, "drug_scaler.pkl"))

# save encoders
joblib.dump(drug_encoder, os.path.join(MODEL_DIR, "drug_encoder.pkl"))
joblib.dump(condition_encoder, os.path.join(MODEL_DIR, "condition_encoder.pkl"))

# save feature order
joblib.dump(feature_columns, os.path.join(MODEL_DIR, "drug_features.pkl"))

print("\nDrug effectiveness model saved successfully!")