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

    "Depression": ["Sertraline","Escitalopram","Citalopram","Venlafaxine","Duloxetine","Lexapro","Zoloft"],
    "Major Depressive Disorde": ["Sertraline","Escitalopram","Citalopram","Venlafaxine","Duloxetine"],

    "Anxiety": ["Sertraline","Escitalopram","Citalopram","Venlafaxine","Clonazepam"],
    "Anxiety and Stress": ["Sertraline","Escitalopram","Clonazepam","Venlafaxine"],
    "Generalized Anxiety Disorde": ["Sertraline","Escitalopram","Venlafaxine","Clonazepam"],
    "Panic Disorde": ["Clonazepam","Sertraline","Venlafaxine"],
    "Social Anxiety Disorde": ["Sertraline","Escitalopram","Venlafaxine"],

    "Bipolar Disorde": ["Quetiapine"],
    "Insomnia": ["Quetiapine","Clonazepam"],

    "Pain": ["Tramadol","Gabapentin"],
    "Chronic Pain": ["Tramadol","Gabapentin","Duloxetine"],
    "ibromyalgia": ["Duloxetine","Gabapentin","Tramadol"],

    "Obesity": ["Phentermine","Bupropion / naltrexone","Contrave"],
    "Weight Loss": ["Phentermine","Contrave"],

    "Birth Control": [
        "Levonorgestrel",
        "Drospirenone / ethinyl estradiol",
        "Ethinyl estradiol / levonorgestrel",
        "Ethinyl estradiol / norethindrone",
        "Ethinyl estradiol / norgestimate",
        "Etonogestrel",
        "Implanon",
        "Nexplanon",
        "Skyla",
        "Mirena"
    ],

    "Emergency Contraception": ["Levonorgestrel"],

    "Abnormal Uterine Bleeding": ["Medroxyprogesterone","Depo-Provera"],

    "Bacterial Vaginitis": ["Metronidazole"],
    "Vaginal Yeast Infection": ["Miconazole"],

    "Acne": [
        "Ethinyl estradiol / norgestimate",
        "Drospirenone / ethinyl estradiol"
    ]
}

# -----------------------------
# API
# -----------------------------

@api_view(["POST"])
def predict_all(request):

    try:

        data = request.data

        age = int(float(data["age"]))
        height = float(data["height"])
        weight = float(data["weight"])

        cholesterol = float(data["cholesterol"])
        heart_rate = float(data["heart_rate"])
        triglycerides = float(data["triglycerides"])

        blood_pressure = data["blood_pressure"]

        alcohol = float(data.get("alcohol_consumption", 0))
        physical_activity = float(data.get("physical_activity", 0))
        sleep_duration = float(data.get("sleep_duration", 7))
        stress_level = float(data.get("stress_level", 5))

        condition = data["condition"].strip()
        drug_name = data["drug_name"]

        sex = 1 if data["sex"] == "Male" else 0
        smoking = 1 if data["smoking"] == "Yes" else 0
        diabetes = 1 if data["diabetes"] == "Yes" else 0
        family_history = 1 if data["family_history"] == "Yes" else 0

        height_m = height / 100
        if height_m <= 0:
            return Response({"error": "Height must be greater than 0"}, status=400)

        bmi = weight / (height_m ** 2)

        try:
            systolic, diastolic = blood_pressure.split("/")
            systolic = int(float(systolic))
            diastolic = int(float(diastolic))
        except:
            return Response({"error": "blood_pressure must be in format 120/80"}, status=400)

        try:
            drug_encoded = drug_encoder.transform([drug_name])[0]
        except:
            return Response({"error": f"Drug '{drug_name}' not supported"}, status=400)

        try:
            condition_encoded = condition_encoder.transform([condition])[0]
        except:
            return Response({"error": f"Condition '{condition}' not supported"}, status=400)

        # -----------------------------
        # Health Model
        # -----------------------------

        health_features = [[
            age, sex, bmi, smoking,
            alcohol, physical_activity,
            sleep_duration, diabetes, stress_level
        ]]

        health_features = health_scaler.transform(health_features)
        health_prediction = health_model.predict_proba(health_features)[0][1]

        # -----------------------------
        # Heart Model
        # -----------------------------

        heart_features = [[
            age, sex, cholesterol, heart_rate, diabetes,
            family_history, smoking,
            0,0,3,1,0,0,1,6,50000,
            bmi, triglycerides,3,7,
            systolic, diastolic
        ]]

        heart_features = heart_scaler.transform(heart_features)
        heart_prediction = heart_model.predict_proba(heart_features)[0][1]

        # -----------------------------
        # Drug Effectiveness
        # -----------------------------

        drug_features = [[drug_encoded, condition_encoded, 10]]
        drug_features = drug_scaler.transform(drug_features)
        drug_prediction = drug_model.predict_proba(drug_features)[0][1]

        # -----------------------------
        # Drug Recommendation
        # -----------------------------

        drug_recommendations = []

        valid_drugs = condition_drug_map.get(condition)

        if not valid_drugs:
            return Response(
                {"error": f"No drug mapping available for condition '{condition}'"},
                status=status.HTTP_400_BAD_REQUEST
            )

        for drug in valid_drugs:

            try:

                drug_id = drug_encoder.transform([drug])[0]

                features = [[drug_id, condition_encoded, 10]]
                features = drug_scaler.transform(features)

                score = drug_model.predict_proba(features)[0][1]

                drug_recommendations.append({
                    "drug": drug,
                    "effectiveness": score
                })

            except:
                continue

        drug_recommendations = sorted(
            drug_recommendations,
            key=lambda x: x["effectiveness"],
            reverse=True
        )

        top_drugs = drug_recommendations[:3]

        for d in top_drugs:
            d["effectiveness"] = round(d["effectiveness"] * 100, 2)

        # -----------------------------
        # Risk calculations
        # -----------------------------

        general_risk = float(health_prediction)
        heart_risk = float(heart_prediction)
        drug_effect = float(drug_prediction)

        complication_risk = (general_risk + heart_risk) / 2

        hospitalization_risk = (0.6 * heart_risk) + (0.4 * general_risk)
        hospitalization_risk *= (1 - (drug_effect * 0.3))

        overall_health_score = 1 - ((general_risk + heart_risk) / 2)

        combined_risk = (general_risk + heart_risk) / 2

        if combined_risk < 0.15:
            severity = "Low"
        elif combined_risk < 0.4:
            severity = "Moderate"
        elif combined_risk < 0.6:
            severity = "High"
        else:
            severity = "Critical"

        # -----------------------------
        # Lifestyle recommendations
        # -----------------------------

        recommendations = []

        if bmi > 30:
            recommendations.append("BMI indicates obesity. Weight reduction recommended.")

        if cholesterol > 200:
            recommendations.append("High cholesterol detected. Reduce saturated fat intake.")

        if systolic > 140 or diastolic > 90:
            recommendations.append("High blood pressure detected. Reduce salt intake.")

        if smoking == 1:
            recommendations.append("Smoking cessation strongly recommended.")

        if sleep_duration < 6:
            recommendations.append("Increase sleep duration to at least 7 hours.")

        if physical_activity < 2:
            recommendations.append("Increase weekly physical activity.")

        if stress_level > 7:
            recommendations.append("High stress detected. Practice stress management.")

        if severity == "Critical":
            recommendations.append("CRITICAL RISK: Seek immediate medical attention.")
        elif severity == "High":
            recommendations.append("High disease severity detected. Consult a doctor immediately.")
        elif severity == "Moderate":
            recommendations.append("Moderate risk detected. Schedule a medical checkup.")
        else:
            recommendations.append("Health risk currently low. Maintain healthy lifestyle.")

        # -----------------------------
        # Response
        # -----------------------------

        result = {
            "general_health_risk": round(general_risk * 100, 2),
            "heart_attack_risk": round(heart_risk * 100, 2),
            "drug_effectiveness": round(drug_effect * 100, 2),
            "complication_risk": round(complication_risk * 100, 2),
            "hospitalization_risk": round(hospitalization_risk * 100, 2),
            "overall_health_score": round(overall_health_score * 100, 2),
            "disease_severity": severity,
            "recommended_drugs": top_drugs,
            "lifestyle_recommendations": recommendations
        }

        return Response(result)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)