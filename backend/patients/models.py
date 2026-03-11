from django.db import models

class Patient(models.Model):

    # Basic Info
    age = models.IntegerField()
    sex = models.CharField(max_length=10)

    height = models.FloatField()
    weight = models.FloatField()

    # Cardiovascular
    blood_pressure = models.CharField(max_length=10)
    cholesterol = models.FloatField()
    heart_rate = models.FloatField()
    triglycerides = models.FloatField()

    # Medical history
    diabetes = models.BooleanField()
    family_history = models.BooleanField()
    previous_heart_problems = models.BooleanField()
    medication_use = models.BooleanField()

    # Lifestyle
    smoking = models.BooleanField()
    alcohol_consumption = models.CharField(max_length=50)

    exercise_hours_per_week = models.FloatField()
    physical_activity_days_per_week = models.IntegerField()

    sedentary_hours_per_day = models.FloatField()
    sleep_hours_per_day = models.FloatField()

    diet = models.CharField(max_length=50)
    stress_level = models.CharField(max_length=50)

    # Other risk factors
    obesity = models.BooleanField()
    income = models.FloatField()

    # Drug prediction
    condition = models.CharField(max_length=100)
    drug_name = models.CharField(max_length=100)