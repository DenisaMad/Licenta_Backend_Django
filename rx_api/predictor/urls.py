from django.urls import path
from .views import predict_pdf, predict_pdf_file

urlpatterns = [
    path("predict/", predict_pdf, name="predict-pdf"),
    path("predict_file/", predict_pdf_file, name="predict-pdf_file"),
]