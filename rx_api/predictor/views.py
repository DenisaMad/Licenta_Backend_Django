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

NER_MODEL_PATH = os.path.join(BASE_DIR, "ner_mlp_model.pkl")
if os.path.exists(NER_MODEL_PATH):
    ner_data = joblib.load(NER_MODEL_PATH)
    ner_pipeline = ner_data['pipeline']
    ner_label_encoder = ner_data['label_encoder']
    print("Loaded NER MLP Model")
else:
    print(f"NER Model not found at {NER_MODEL_PATH}")
    ner_pipeline = None
    ner_label_encoder = None

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


def word2features(sent, i):
    word = sent[i]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.isnumeric()': any(char.isdigit() for char in word)
    }
    if i > 0:
        word1 = sent[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True
        
    if i < len(sent)-1:
        word1 = sent[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True
                
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def extract_drug_and_dose(line: str, is_med_predicted: bool = False):
    if not ner_pipeline or not ner_label_encoder:
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

    tokens = re.findall(r'\w+|[^\w\s]', line)
    if not tokens:
        return None
        
    features = sent2features(tokens)
    preds_encoded = ner_pipeline.predict(features)
    preds = ner_label_encoder.inverse_transform(preds_encoded)
    
    drug_tokens = []
    dose_tokens = []
    
    for token, tag in zip(tokens, preds):
        if tag == 'B-DRUG' or tag == 'I-DRUG':
            drug_tokens.append(token)
        elif tag == 'B-DOSE' or tag == 'I-DOSE':
            dose_tokens.append(token)
            
    drug = " ".join(drug_tokens).strip(" -:;|,/")
    drug = drug.replace(" / ", " ").replace(" - ", " ").replace("/", "").replace("\\", "").strip()
    
    if not drug:
        return None
        
    # Filter out common false positives (words that are not drugs)
    drug_lower = drug.lower()
    blacklist = ["recomandare", "control", "diagnosc", "diagnostic", "semnatura", "telefon", "varsta", "nume", "data", "zile"]
    if any(b in drug_lower for b in blacklist):
        return None
        
    if not dose_tokens and not is_med_predicted:
        return None
        
    raw_dose = " ".join(dose_tokens).strip()
    raw_dose = " ".join(dose_tokens).strip()
    
    # Normalize dose for .NET backend which strictly expects x/y/z
    def normalize_dose(d: str) -> str:
        if not d:
            return "1/0/0"
        digits = re.findall(r'\d', d)
        if len(digits) >= 3:
            return f"{digits[0]}/{digits[1]}/{digits[2]}"
        d_lower = d.lower()
        if "diminea" in d_lower:
            return f"{digits[0] if digits else '1'}/0/0"
        elif "sear" in d_lower:
            return f"0/0/{digits[0] if digits else '1'}"
        elif "pranz" in d_lower or "prânz" in d_lower or "amiaz" in d_lower:
            return f"0/{digits[0] if digits else '1'}/0"
        elif "zi" in d_lower or "nevoie" in d_lower:
            return f"{digits[0] if digits else '1'}/0/0"
        elif len(digits) == 2:
            return f"{digits[0]}/{digits[1]}/0"
        elif len(digits) == 1:
            return f"{digits[0]}/0/0"
        return "1/0/0"

    dose = normalize_dose(raw_dose)
    
    if drug or dose != "1/0/0":
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
            is_med = int(pred) == 1
            has_dose_pattern = re.search(r'\d[\/\-]\d|\d\s*(zi|seara|diminea|pranz)', line.lower())
            
            if is_med or has_dose_pattern:
                extracted = extract_drug_and_dose(line, is_med_predicted=is_med)
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
                is_med = int(pred) == 1
                has_dose_pattern = re.search(r'\d[\/\-]\d|\d\s*(zi|seara|diminea|pranz)', line.lower())
                
                if is_med or has_dose_pattern:
                    extracted = extract_drug_and_dose(line, is_med_predicted=is_med)
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