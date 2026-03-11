from django.urls import path
from .views import predict_all

urlpatterns = [
    path("predict/", predict_all, name="predict"),
]