from rest_framework import serializers

class PredictionSerializer(serializers.Serializer):

    age = serializers.IntegerField()
    sex = serializers.CharField()

    height = serializers.FloatField()
    weight = serializers.FloatField()

    blood_pressure = serializers.CharField()
    cholesterol = serializers.FloatField()
    heart_rate = serializers.FloatField()
    triglycerides = serializers.FloatField()

    diabetes = serializers.BooleanField()
    family_history = serializers.BooleanField()
    previous_heart_problems = serializers.BooleanField()
    medication_use = serializers.BooleanField()

    smoking = serializers.BooleanField()
    alcohol_consumption = serializers.CharField()

    exercise_hours_per_week = serializers.FloatField()
    physical_activity_days_per_week = serializers.IntegerField()

    sedentary_hours_per_day = serializers.FloatField()
    sleep_hours_per_day = serializers.FloatField()

    diet = serializers.CharField()
    stress_level = serializers.CharField()

    obesity = serializers.BooleanField()
    income = serializers.FloatField()

    condition = serializers.CharField()
    drug_name = serializers.CharField()