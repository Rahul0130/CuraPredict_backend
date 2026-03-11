import joblib
import os

# path to models folder
MODEL_DIR = r"R:\personalized_medicine_backend\models"

# load feature list
health_features = joblib.load(os.path.join(MODEL_DIR, "health_features.pkl"))

print("\nHealth Model Features Used:")
print("================================")

for i, feature in enumerate(health_features, 1):
    print(f"{i}. {feature}")

print("\nTotal Features:", len(health_features))