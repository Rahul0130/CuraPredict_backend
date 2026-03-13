import joblib

# load heart model
model = joblib.load("heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")

print("Heart Model Type:", type(model))
print()

# number of features
if hasattr(model, "n_features_in_"):
    print("Heart model expects features:", model.n_features_in_)

if hasattr(scaler, "n_features_in_"):
    print("Heart scaler expects features:", scaler.n_features_in_)

print()

# feature names if stored
if hasattr(scaler, "feature_names_in_"):
    print("Heart model features:")
    for f in scaler.feature_names_in_:
        print("-", f)
else:
    print("Feature names not stored in scaler.")