from django.urls import path
from .views import predict_pdf

urlpatterns = [
    path("predict/", predict_pdf, name="predict-pdf"),
]