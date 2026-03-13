import joblib

# load model and scaler
model = joblib.load("health_model.pkl")
scaler = joblib.load("health_scaler.pkl")

print("Model type:", type(model))
print()

# how many features model expects
if hasattr(model, "n_features_in_"):
    print("Model expects features:", model.n_features_in_)

if hasattr(scaler, "n_features_in_"):
    print("Scaler expects features:", scaler.n_features_in_)

print()

# feature names if stored
if hasattr(scaler, "feature_names_in_"):
    print("Features used during training:")
    for f in scaler.feature_names_in_:
        print("-", f)
else:
    print("Feature names not stored in scaler.")