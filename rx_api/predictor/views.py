import os
import re
import tempfile

import joblib
from pdfminer.high_level import extract_text

from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status

from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rx_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
print(model)
print(model.named_steps['tfidf'])
print(hasattr(model.named_steps['tfidf'], "idf_"))

def split_lines(text: str) -> list[str]:
    return [line.strip() for line in text.split("\n") if len(line.strip()) > 3]


def pdf_to_text(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        for chunk in uploaded_file.chunks():
            tmp.write(chunk)
        temp_path = tmp.name

    try:
        text = extract_text(temp_path)
        return text
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def extract_drug_and_dose(line: str):
    patterns = [
        r"([A-Za-zĂÂÎȘȚăâîșț0-9\-\+]+(?:\s+[A-Za-zĂÂÎȘȚăâîșț0-9\-\+]+){0,3}).*?(\d/\d/\d)",
        r"([A-Za-zĂÂÎȘȚăâîșț0-9\-\+]+(?:\s+[A-Za-zĂÂÎȘȚăâîșț0-9\-\+]+){0,3}).*?(\d-\d-\d)",
    ]

    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            drug = match.group(1).strip(" -:;|,")
            dose = match.group(2)
            dose = dose.replace("-", "/")
            return {
                "drug": drug,
                "dose": dose,
                "raw_line": line
            }

    return None


upload_param = openapi.Parameter(
    name="file",
    in_=openapi.IN_FORM,
    description="PDF file to analyze",
    type=openapi.TYPE_FILE,
    required=True,
)


@swagger_auto_schema(
    method="post",
    manual_parameters=[upload_param],
    operation_summary="Upload PDF and extract medication lines",
    operation_description="Accepts a PDF, extracts text, classifies relevant lines using the trained model, and returns detected medications with dose patterns.",
    responses={
        200: openapi.Response(
            description="Prediction successful",
            examples={
                "application/json": {
                    "filename": "example.pdf",
                    "total_lines": 24,
                    "positive_lines": 3,
                    "results": [
                        {
                            "drug": "Paracetamol 500mg",
                            "dose": "1/0/1",
                            "raw_line": "Paracetamol 500mg - 1/0/1"
                        }
                    ]
                }
            },
        ),
        400: "Bad request",
        500: "Server error",
    },
)
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def predict_pdf_file(request):
    uploaded_file = request.FILES.get("file")

    if not uploaded_file:
        return Response(
            {"error": "No file uploaded. Use form-data with key 'file'."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if not uploaded_file.name.lower().endswith(".pdf"):
        return Response(
            {"error": "Only PDF files are allowed."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        text = pdf_to_text(uploaded_file)

        if not text or not text.strip():
            return Response(
                {"error": "Could not extract text from PDF."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        lines = split_lines(text)

        if not lines:
            return Response(
                {"error": "No usable lines found in PDF."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        predictions = model.predict(lines)

        results = []
        for line, pred in zip(lines, predictions):
            if int(pred) == 1:
                extracted = extract_drug_and_dose(line)
                if extracted:
                    results.append(extracted)

        return Response(
            {
                "filename": uploaded_file.name,
                "total_lines": len(lines),
                "positive_lines": len(results),
                "results": results,
            },
            status=status.HTTP_200_OK,
        )

    except Exception as exc:
        return Response(
            {"error": f"Internal error: {str(exc)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    
@api_view(["POST"])
def predict_pdf(request):
    try:
        # 🔥 aici citim direct raw body
        file_bytes = request.body

        if not file_bytes:
            return Response(
                {"error": "Empty request body"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 🔥 salvăm temporar PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        try:
            text = extract_text(temp_path)

            if not text or not text.strip():
                return Response(
                    {"error": "Could not extract text from PDF."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            lines = split_lines(text)

            if not lines:
                return Response(
                    {"error": "No usable lines found in PDF."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            predictions = model.predict(lines)

            results = []
            for line, pred in zip(lines, predictions):
                if int(pred) == 1:
                    extracted = extract_drug_and_dose(line)
                    if extracted:
                        results.append(extracted)

            return Response(
                {
                    "total_lines": len(lines),
                    "positive_lines": len(results),
                    "results": results,
                },
                status=status.HTTP_200_OK,
            )

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as exc:
        return Response(
            {"error": f"Internal error: {str(exc)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )