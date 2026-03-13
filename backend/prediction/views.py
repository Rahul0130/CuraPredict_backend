import joblib
import os
import warnings

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

warnings.filterwarnings("ignore")

# -----------------------------
# Paths
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

print("MODEL_DIR =", MODEL_DIR)

# -----------------------------
# Load Models
# -----------------------------

health_model = joblib.load(os.path.join(MODEL_DIR, "health_model.pkl"))
health_scaler = joblib.load(os.path.join(MODEL_DIR, "health_scaler.pkl"))

heart_model = joblib.load(os.path.join(MODEL_DIR, "heart_model.pkl"))
heart_scaler = joblib.load(os.path.join(MODEL_DIR, "heart_scaler.pkl"))

drug_model = joblib.load(os.path.join(MODEL_DIR, "drug_model.pkl"))
drug_scaler = joblib.load(os.path.join(MODEL_DIR, "drug_scaler.pkl"))

# -----------------------------
# Encoders
# -----------------------------

drug_encoder = joblib.load(os.path.join(MODEL_DIR, "drug_encoder.pkl"))
condition_encoder = joblib.load(os.path.join(MODEL_DIR, "condition_encoder.pkl"))

# -----------------------------
# Condition → Drug Mapping
# -----------------------------

condition_drug_map = {
"Depression":["Sertraline","Escitalopram","Citalopram","Venlafaxine","Duloxetine","Lexapro","Zoloft"],
"Major Depressive Disorder":["Sertraline","Escitalopram","Citalopram","Venlafaxine","Duloxetine"],
"Anxiety":["Sertraline","Escitalopram","Citalopram","Venlafaxine","Clonazepam"],
"Anxiety and Stress":["Sertraline","Escitalopram","Clonazepam","Venlafaxine"],
"Generalized Anxiety Disorder":["Sertraline","Escitalopram","Venlafaxine","Clonazepam"],
"Panic Disorder":["Clonazepam","Sertraline","Venlafaxine"],
"Social Anxiety Disorder":["Sertraline","Escitalopram","Venlafaxine"],
"Bipolar Disorder":["Quetiapine"],
"Insomnia":["Quetiapine","Clonazepam"],
"Pain":["Tramadol","Gabapentin"],
"Chronic Pain":["Tramadol","Gabapentin","Duloxetine"],
"Fibromyalgia":["Duloxetine","Gabapentin","Tramadol"],
"Obesity":["Phentermine","Bupropion / naltrexone","Contrave"],
"Weight Loss":["Phentermine","Contrave"]
}

# -----------------------------
# API
# -----------------------------

@api_view(["POST"])
def predict_all(request):

    try:

        data = request.data

        age = float(data["age"])
        height = float(data["height"])
        weight = float(data["weight"])

        cholesterol = float(data["cholesterol"])
        triglycerides = float(data["triglycerides"])
        heart_rate = float(data["heart_rate"])

        glucose = float(data.get("glucose",90))

        blood_pressure = data["blood_pressure"]

        alcohol = float(data.get("alcohol_consumption",0))
        physical_activity = float(data.get("physical_activity",0))
        sleep_duration = float(data.get("sleep_duration",7))
        stress_level = float(data.get("stress_level",5))

        condition = str(data["condition"]).strip()
        drug_name = str(data["drug_name"]).strip()

        sex = 1 if data["sex"] == "Male" else 0
        smoking = 1 if data["smoking"] == "Yes" else 0
        diabetes = 1 if data["diabetes"] == "Yes" else 0
        family_history = 1 if data["family_history"] == "Yes" else 0

        # -----------------------------
        # BMI
        # -----------------------------

        height_m = height / 100

        if height_m <= 0:
            return Response({"error":"Height must be greater than 0"},status=400)

        bmi = weight / (height_m ** 2)

        # -----------------------------
        # Blood Pressure
        # -----------------------------

        try:
            systolic, diastolic = blood_pressure.split("/")
            systolic = float(systolic)
            diastolic = float(diastolic)
        except:
            return Response({"error":"Invalid blood pressure format. Use 120/80"},status=400)

        # -----------------------------
        # Encode Drug / Condition
        # -----------------------------

        try:
            drug_encoded = drug_encoder.transform([drug_name])[0]
            condition_encoded = condition_encoder.transform([condition])[0]
        except:
            return Response({"error":"Unknown drug or condition"},status=400)

        # -----------------------------
        # HEALTH MODEL
        # -----------------------------

        health_features = [[
        age,sex,bmi,smoking,alcohol,physical_activity,
        sleep_duration,diabetes,stress_level,
        cholesterol,triglycerides,systolic,diastolic,heart_rate
        ]]

        health_features = health_scaler.transform(health_features)

        health_prediction = health_model.predict_proba(health_features)[0][1]

        health_prediction = health_prediction ** 1.5
        health_prediction = min(max(health_prediction,0),1)

        # -----------------------------
        # HEART MODEL
        # -----------------------------

        if cholesterol < 200:
            cholesterol_cat = 1
        elif cholesterol < 240:
            cholesterol_cat = 2
        else:
            cholesterol_cat = 3

        glucose_risk = 1 if glucose > 125 else 0

        age_cholesterol_risk = age * cholesterol_cat
        bmi_stress_risk = bmi * stress_level

        heart_features = [[
        age,sex,cholesterol_cat,smoking,alcohol,
        physical_activity,glucose_risk,bmi,
        systolic,diastolic,
        age_cholesterol_risk,bmi_stress_risk
        ]]

        heart_features = heart_scaler.transform(heart_features)

        heart_prediction = heart_model.predict_proba(heart_features)[0][1]

        # -----------------------------
        # DRUG MODEL
        # -----------------------------

        drug_features = [[drug_encoded,condition_encoded,10]]

        drug_features = drug_scaler.transform(drug_features)

        drug_prediction = drug_model.predict_proba(drug_features)[0][1]

        # -----------------------------
        # DRUG RECOMMENDATION
        # -----------------------------

        drug_recommendations = []

        for drug in condition_drug_map.get(condition,[]):

            try:

                drug_id = drug_encoder.transform([drug])[0]

                features = [[drug_id,condition_encoded,10]]

                features = drug_scaler.transform(features)

                score = drug_model.predict_proba(features)[0][1]

                drug_recommendations.append({
                "drug":drug,
                "effectiveness":round(score*100,2)
                })

            except:
                continue

        drug_recommendations = sorted(
        drug_recommendations,
        key=lambda x:x["effectiveness"],
        reverse=True
        )[:3]

        # -----------------------------
        # RISK CALCULATIONS
        # -----------------------------

        general_risk = float(health_prediction)
        heart_risk = float(heart_prediction)
        drug_effect = float(drug_prediction)

        combined_risk = (0.7 * heart_risk) + (0.3 * general_risk)

        complication_risk = combined_risk
        hospitalization_risk = combined_risk * (1 - (drug_effect * 0.25))
        overall_health_score = 1 - combined_risk

        if combined_risk < 0.2:
            severity = "Low"
        elif combined_risk < 0.4:
            severity = "Moderate"
        elif combined_risk < 0.65:
            severity = "High"
        else:
            severity = "Critical"

        # -----------------------------
        # LIFESTYLE RECOMMENDATIONS
        # -----------------------------

        recommendations = []

        if bmi > 30:
            recommendations.append("BMI indicates obesity. Weight reduction recommended.")

        if cholesterol > 200:
            recommendations.append("High cholesterol detected. Reduce saturated fat intake.")

        if triglycerides > 200:
            recommendations.append("High triglycerides detected. Reduce sugar and processed food intake.")

        if systolic > 140:
            recommendations.append("High blood pressure detected. Reduce salt intake.")

        if glucose > 125:
            recommendations.append("Elevated blood glucose detected. Screen for diabetes.")

        if smoking:
            recommendations.append("Smoking cessation strongly recommended.")

        if sleep_duration < 6:
            recommendations.append("Increase sleep duration to at least 7 hours.")

        if physical_activity < 2:
            recommendations.append("Increase weekly physical activity.")

        if stress_level > 7:
            recommendations.append("High stress detected. Practice stress management.")

        if alcohol > 7:
            recommendations.append("Reduce alcohol consumption.")

        # Severity advice

        if severity == "Critical":
            recommendations.append("Immediate medical consultation recommended.")

        elif severity == "High":
            recommendations.append("Consult a doctor soon for further evaluation.")

        elif severity == "Moderate":
            recommendations.append("Adopt preventive lifestyle changes to reduce risk.")

        # Default recommendation

        if not recommendations:
            recommendations.append(
            "Your health indicators look stable. Maintain balanced nutrition, regular exercise, and adequate sleep."
            )

        # -----------------------------
        # RESPONSE
        # -----------------------------

        result = {
        "general_health_risk":round(general_risk*100,2),
        "heart_attack_risk":round(heart_risk*100,2),
        "drug_effectiveness":round(drug_effect*100,2),
        "complication_risk":round(complication_risk*100,2),
        "hospitalization_risk":round(hospitalization_risk*100,2),
        "overall_health_score":round(overall_health_score*100,2),
        "disease_severity":severity,
        "recommended_drugs":drug_recommendations,
        "lifestyle_recommendations":recommendations
        }

        return Response(result)

    except Exception as e:

        return Response({"error":str(e)},status=status.HTTP_400_BAD_REQUEST)