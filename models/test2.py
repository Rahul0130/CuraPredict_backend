import joblib
import os

MODEL_DIR = "."

print("Current folder:", os.getcwd())

# load encoders
drug_encoder = joblib.load(os.path.join(MODEL_DIR, "drug_encoder.pkl"))
condition_encoder = joblib.load(os.path.join(MODEL_DIR, "condition_encoder.pkl"))

print("\nTotal Supported Drugs:", len(drug_encoder.classes_))
print("=================================")

for drug in drug_encoder.classes_:
    print(drug)

print("\n\nTotal Supported Conditions:", len(condition_encoder.classes_))
print("=================================")

for condition in condition_encoder.classes_:
    print(condition)