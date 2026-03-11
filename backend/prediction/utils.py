import numpy as np


def preprocess_features(data):

    age = data["age"]

    gender = 1 if data["gender"] == "Male" else 0

    height = data["height"] / 100
    weight = data["weight"]

    bmi = weight / (height ** 2)

    blood_pressure = data["blood_pressure"]
    cholesterol = data["cholesterol"]
    blood_sugar = data["blood_sugar"]

    smoking = 1 if data["smoking"] else 0
    alcohol = 1 if data["alcohol"] else 0
    physical_activity = 1 if data["physical_activity"] else 0

    diabetes = 1 if data["diabetes"] else 0
    family_history = 1 if data["family_history"] else 0

    features = np.array([
        age,
        gender,
        bmi,
        blood_pressure,
        cholesterol,
        blood_sugar,
        smoking,
        alcohol,
        physical_activity,
        diabetes,
        family_history
    ]).reshape(1, -1)

    return features