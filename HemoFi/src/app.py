from flask import Flask, render_template, request, send_file
from pathlib import Path
import pandas as pd
import os
import json
import plotly.express as px
from datetime import datetime
from io import BytesIO
import re

try:
    from reportlab.lib.pagesizes import A4  # type: ignore[import]
    from reportlab.lib import colors  # type: ignore[import]
    from reportlab.pdfgen import canvas  # type: ignore[import]
except ImportError:
    A4 = None
    colors = None
    canvas = None

try:
    from PIL import Image  # type: ignore[import]
except ImportError:
    Image = None

try:
    from PIL import ImageOps, ImageFilter  # type: ignore[import]
except ImportError:
    ImageOps = None
    ImageFilter = None

try:
    import pytesseract  # type: ignore[import]
except ImportError:
    pytesseract = None

try:
    from pypdf import PdfReader  # type: ignore[import]
except ImportError:
    PdfReader = None

# AI libraries
try:
    from groq import Groq  # type: ignore[import]
except ImportError:
    Groq = None

try:
    from dotenv import load_dotenv  # type: ignore[import]
except ImportError:
    load_dotenv = None

from utils import load_dataset, estimate_anemia_risk

app = Flask(__name__, template_folder="templates", static_folder="static")
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "anemia.csv"

if load_dotenv is not None:
    env_path = BASE_DIR.parent / "api.env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

app.config["DATA_PATH"] = str(DATA_PATH)
DATAFRAME_CACHE = None
MAX_ASSISTANT_HISTORY_MESSAGES = 12
MAX_ASSISTANT_QUERY_CHARS = 1200
ASSISTANT_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
]
ASSISTANT_SYSTEM_PROMPT = (
    "You are an AI medical education assistant for anemia screening. "
    "You must only answer medical questions, especially questions about anemia, CBC parameters, blood health, "
    "symptoms, prevention, and clinician follow-up guidance. "
    "If the user asks about non-medical topics (coding, business, entertainment, travel, general chat, etc.), "
    "politely refuse and ask them to submit a medical question instead. "
    "Provide clear, evidence-aligned educational guidance in plain language. "
    "Do not provide diagnosis, prescriptions, or emergency triage decisions. "
    "When needed, advise consulting a licensed clinician. "
    "Keep responses concise and practical."
)

MEDICAL_QUERY_KEYWORDS = {
    "anemia", "anaemia", "cbc", "blood", "hemoglobin", "haemoglobin", "mch", "mchc", "mcv",
    "hematocrit", "haematocrit", "rbc", "wbc", "platelet", "ferritin", "iron", "transferrin",
    "vitamin b12", "folate", "fatigue", "pallor", "dizziness", "shortness of breath", "symptom",
    "diagnosis", "diagnostic", "screening", "test", "lab", "clinical", "medicine", "medical",
    "doctor", "clinician", "treatment", "therapy", "supplement", "nutrition", "diet"
}

SELF_CHECK_QUESTIONS = [
    {
        "key": "fatigue_frequency",
        "question": "How often do you feel unusual fatigue or low energy?",
        "options": [
            {"value": 0, "label": "Rarely or never"},
            {"value": 1, "label": "Sometimes"},
            {"value": 2, "label": "Most days"},
            {"value": 3, "label": "Almost always"},
        ],
    },
    {
        "key": "breathlessness",
        "question": "Do you get short of breath during mild activity?",
        "options": [
            {"value": 0, "label": "No"},
            {"value": 1, "label": "Occasionally"},
            {"value": 2, "label": "Frequently"},
            {"value": 3, "label": "Very frequently"},
        ],
    },
    {
        "key": "pale_signs",
        "question": "Have you noticed pale skin, lips, or inner eyelids?",
        "options": [
            {"value": 0, "label": "No"},
            {"value": 1, "label": "Not sure"},
            {"value": 2, "label": "Yes, mildly"},
            {"value": 3, "label": "Yes, clearly"},
        ],
    },
    {
        "key": "dizziness",
        "question": "How often do you feel dizzy, lightheaded, or weak?",
        "options": [
            {"value": 0, "label": "Rarely or never"},
            {"value": 1, "label": "Sometimes"},
            {"value": 2, "label": "Frequently"},
            {"value": 3, "label": "Very frequently"},
        ],
    },
    {
        "key": "diet_iron",
        "question": "How would you describe your iron-rich food intake?",
        "options": [
            {"value": 0, "label": "Good and balanced"},
            {"value": 1, "label": "Average"},
            {"value": 2, "label": "Often low"},
            {"value": 3, "label": "Very low / restrictive"},
        ],
    },
    {
        "key": "blood_loss_history",
        "question": "Do you have recent heavy menstrual bleeding or blood-loss history?",
        "options": [
            {"value": 0, "label": "No"},
            {"value": 1, "label": "Unsure"},
            {"value": 2, "label": "Mild history"},
            {"value": 3, "label": "Clear/recent history"},
        ],
    },
]

NUTRITION_DAILY_TARGETS = [
    {"label": "Iron",
        "value": "Adult men: ~8 mg/day, Women (19-50): ~18 mg/day"},
    {"label": "Vitamin C", "value": "75-90 mg/day to improve iron absorption"},
    {"label": "Folate", "value": "~400 mcg/day"},
    {"label": "Vitamin B12", "value": "~2.4 mcg/day"},
]

NUTRITION_FOODS = [
    {"food": "Spinach (cooked)", "portion": "1 cup", "iron": "~6.4 mg",
     "tips": "Pair with lemon/orange for better absorption"},
    {"food": "Lentils (cooked)", "portion": "1 cup",
     "iron": "~6.6 mg", "tips": "Great vegetarian iron source"},
    {"food": "Chickpeas (cooked)", "portion": "1 cup",
     "iron": "~4.7 mg", "tips": "Use in salads/soups"},
    {"food": "Lean red meat",
        "portion": "90 g (3 oz)", "iron": "~2.1 mg", "tips": "Heme iron absorbs efficiently"},
    {"food": "Chicken liver", "portion": "75 g",
        "iron": "~6-9 mg", "tips": "High iron and B12"},
    {"food": "Pumpkin seeds", "portion": "30 g",
        "iron": "~2.5 mg", "tips": "Easy snack option"},
    {"food": "Tofu", "portion": "100 g", "iron": "~3.4 mg",
        "tips": "Good for plant-based diets"},
    {"food": "Eggs", "portion": "2 eggs", "iron": "~1.8 mg",
        "tips": "Add vitamin C source in same meal"},
    {"food": "Fortified cereal", "portion": "1 bowl",
        "iron": "~4-8 mg", "tips": "Check label for iron content"},
]

NUTRITION_MEAL_EXAMPLE = [
    "Breakfast: Fortified cereal + citrus fruit",
    "Lunch: Lentil curry + spinach salad + lemon dressing",
    "Snack: Pumpkin seeds + guava/orange",
    "Dinner: Lean meat/tofu + chickpeas + vegetables",
]


def _selfcheck_assessment(selected_answers: dict[str, int]) -> dict[str, str | int | bool | list[str]]:
    score = 0
    selected_labels: list[str] = []

    for question in SELF_CHECK_QUESTIONS:
        q_key = str(question["key"])
        q_value = int(selected_answers.get(q_key, 0))
        score += q_value
        selected_option = next(
            (option for option in question["options"] if int(
                option["value"]) == q_value),
            None,
        )
        if selected_option is not None:
            selected_labels.append(
                f"{question['question']} — {selected_option['label']}")

    if score >= 13:
        level = "High"
        status = "Potential anemia-related health issue detected"
        guidance = "Please arrange a CBC test and clinical review as soon as possible."
        alert_class = "danger"
    elif score >= 8:
        level = "Moderate"
        status = "Possible anemia-related concern"
        guidance = "Consider CBC testing soon and monitor symptoms closely."
        alert_class = "warning"
    elif score >= 3:
        level = "Mild"
        status = "Low-to-moderate symptom burden"
        guidance = "Improve nutrition and monitor symptoms; seek testing if symptoms persist."
        alert_class = "info"
    else:
        level = "Low"
        status = "No strong anemia-related symptom pattern detected"
        guidance = "Continue healthy habits and routine health follow-up."
        alert_class = "success"

    return {
        "score": score,
        "level": level,
        "status": status,
        "guidance": guidance,
        "alert_class": alert_class,
        "selected_labels": selected_labels,
        "has_issue": score >= 7,
    }


def _is_medical_query(query: str) -> bool:
    normalized_query = " ".join(query.lower().split())
    return any(keyword in normalized_query for keyword in MEDICAL_QUERY_KEYWORDS)


def _friendly_dtype_label(dtype) -> str:
    if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype):
        return "Numeric"
    if pd.api.types.is_bool_dtype(dtype):
        return "Category"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "Date/Time"
    return "Text"


def _friendly_column_type(column_name: str, series: pd.Series) -> str:
    if column_name in {"Gender", "Result"}:
        return "Category"
    return _friendly_dtype_label(series.dtype)


def get_dataset():
    global DATAFRAME_CACHE
    if DATAFRAME_CACHE is None:
        DATAFRAME_CACHE = load_dataset(app.config["DATA_PATH"])
    return DATAFRAME_CACHE


def _parse_chat_history(raw_history: str) -> list[dict[str, str]]:
    if not raw_history:
        return []
    try:
        parsed = json.loads(raw_history)
    except Exception:
        return []

    if not isinstance(parsed, list):
        return []

    sanitized_messages = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"}:
            continue
        if not content:
            continue
        sanitized_messages.append({"role": role, "content": content})

    return sanitized_messages[-MAX_ASSISTANT_HISTORY_MESSAGES:]


def _assistant_reply(query: str, history: list[dict[str, str]]) -> tuple[str | None, str | None]:
    api_key = os.getenv("GROQ_API_KEY")
    if Groq is None or not api_key:
        return None, "AI service not available. Set GROQ_API_KEY and install groq package."

    messages = [{"role": "system", "content": ASSISTANT_SYSTEM_PROMPT}]
    messages.extend(history[-MAX_ASSISTANT_HISTORY_MESSAGES:])
    messages.append({"role": "user", "content": query})

    client = Groq(api_key=api_key)
    last_error = "Unknown error"

    for model_name in ASSISTANT_MODELS:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=700,
            )
            content = ""
            if completion and getattr(completion, "choices", None):
                message = completion.choices[0].message
                content = (message.content or "") if message else ""
            if isinstance(content, list):
                content = " ".join(str(part) for part in content)
            content = str(content).strip()
            if content:
                return content, None
            last_error = "No response generated."
        except Exception as exc:
            last_error = str(exc)

    return None, f"AI request failed: {last_error}"


def _build_health_report_pdf(
    *,
    timestamp: str,
    gender_label: str,
    hemoglobin: float,
    mch: float,
    mchc: float,
    mcv: float,
    risk_percent: float,
    risk_level: str,
    prediction: int,
    neighbors_used: int,
    recommendation: str,
) -> bytes:
    global canvas, A4, colors
    if canvas is None or A4 is None or colors is None:
        try:
            # type: ignore[import]
            from reportlab.lib.pagesizes import A4 as _A4
            from reportlab.lib import colors as _colors  # type: ignore[import]
            # type: ignore[import]
            from reportlab.pdfgen import canvas as _canvas
            A4 = _A4
            colors = _colors
            canvas = _canvas
        except ImportError as exc:
            raise RuntimeError("reportlab is not installed") from exc

    if canvas is None or A4 is None or colors is None:
        raise RuntimeError("reportlab is not installed")

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    page_width, page_height = A4

    margin = 42
    content_left = margin
    content_right = page_width - margin
    cursor_y = page_height - 55

    pdf.setFillColor(colors.HexColor("#E63946"))
    pdf.roundRect(content_left, cursor_y - 32, content_right -
                  content_left, 42, 10, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(content_left + 14, cursor_y - 6, "HEMOSCAN Health Report")
    pdf.setFont("Helvetica", 10)
    pdf.drawRightString(content_right - 12, cursor_y -
                        5, f"Generated: {timestamp}")

    cursor_y -= 62

    def section_title(title: str):
        nonlocal cursor_y
        pdf.setFillColor(colors.HexColor("#1F2937"))
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(content_left, cursor_y, title)
        pdf.setStrokeColor(colors.HexColor("#E5E7EB"))
        pdf.line(content_left, cursor_y - 4, content_right, cursor_y - 4)
        cursor_y -= 20

    def key_value(label: str, value: str):
        nonlocal cursor_y
        pdf.setFillColor(colors.HexColor("#374151"))
        pdf.setFont("Helvetica-Bold", 10.5)
        pdf.drawString(content_left, cursor_y, f"{label}:")
        pdf.setFont("Helvetica", 10.5)
        pdf.setFillColor(colors.black)
        pdf.drawString(content_left + 120, cursor_y, value)
        cursor_y -= 16

    section_title("Patient Inputs")
    key_value("Gender", gender_label)
    key_value("Hemoglobin (g/dL)", f"{hemoglobin:.1f}")
    key_value("MCH (pg)", f"{mch:.1f}")
    key_value("MCHC (g/dL)", f"{mchc:.1f}")
    key_value("MCV (fL)", f"{mcv:.1f}")

    cursor_y -= 8
    section_title("Assessment Output")
    key_value("Risk Score", f"{risk_percent:.1f}%")
    key_value("Risk Level", risk_level)
    key_value("Prediction", "Anemia Risk Detected" if prediction ==
              1 else "No Anemia Risk Detected")
    key_value("Similar Cases Used", str(neighbors_used))

    badge_color = {
        "LOW": "#28A745",
        "MILD": "#F59E0B",
        "MODERATE": "#F97316",
        "SEVERE": "#DC2626",
    }.get(risk_level, "#6B7280")

    cursor_y -= 2
    pdf.setFillColor(colors.HexColor(badge_color))
    pdf.roundRect(content_left, cursor_y - 8, 118, 22, 7, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(content_left + 12, cursor_y +
                   1, f"Risk Level: {risk_level}")

    cursor_y -= 34
    section_title("Guidance")
    guidance_lines = [
        recommendation,
        "",
        "Disclaimer: This report is for educational screening support only and is not a diagnosis.",
    ]
    text_object = pdf.beginText(content_left, cursor_y)
    text_object.setFont("Helvetica", 10.5)
    text_object.setFillColor(colors.black)
    for line in guidance_lines:
        text_object.textLine(line)
    pdf.drawText(text_object)

    pdf.setStrokeColor(colors.HexColor("#E5E7EB"))
    pdf.line(content_left, 46, content_right, 46)
    pdf.setFont("Helvetica-Oblique", 9)
    pdf.setFillColor(colors.HexColor("#6B7280"))
    pdf.drawString(content_left, 32,
                   "HEMOSCAN | Anemia screening support report")

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


def _ensure_tesseract_ready() -> tuple[bool, str | None]:
    if pytesseract is None:
        return False, "pytesseract package is not installed."

    try:
        pytesseract.get_tesseract_version()
        return True, None
    except Exception:
        pass

    common_paths = [
        r"C:/Program Files/Tesseract-OCR/tesseract.exe",
        r"C:/Program Files (x86)/Tesseract-OCR/tesseract.exe",
    ]
    for candidate in common_paths:
        if os.path.exists(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            try:
                pytesseract.get_tesseract_version()
                return True, None
            except Exception:
                continue

    return False, "Tesseract OCR engine is not available in PATH."


def _extract_report_text(uploaded_file) -> tuple[str, str | None]:
    filename = (uploaded_file.filename or "").lower()
    if not filename:
        return "", "Missing file name."

    if filename.endswith(".pdf"):
        if PdfReader is None:
            return "", "pypdf package is not installed."
        uploaded_file.stream.seek(0)
        reader = PdfReader(uploaded_file.stream)
        extracted_pages = []
        for page in reader.pages:
            extracted_pages.append((page.extract_text() or "").strip())
        pdf_text = "\n".join(part for part in extracted_pages if part)
        if pdf_text.strip():
            return pdf_text, None
        return "", "PDF text could not be extracted. If this is a scanned PDF, upload as image or use OCR-capable PDF conversion."

    if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")):
        if Image is None:
            return "", "Pillow package is not installed."
        tesseract_ok, tesseract_error = _ensure_tesseract_ready()
        if not tesseract_ok:
            return "", tesseract_error
        uploaded_file.stream.seek(0)
        image = Image.open(uploaded_file.stream)
        ocr_candidates: list[str] = []

        try:
            ocr_candidates.append(pytesseract.image_to_string(
                image, config="--oem 3 --psm 6"))
        except Exception:
            pass

        if ImageOps is not None and ImageFilter is not None:
            try:
                processed = ImageOps.exif_transpose(image).convert("L")
                processed = ImageOps.autocontrast(processed)
                processed = processed.filter(ImageFilter.SHARPEN)
                binary = processed.point(
                    lambda pixel: 255 if pixel > 150 else 0, mode="1")
                ocr_candidates.append(pytesseract.image_to_string(
                    binary, config="--oem 3 --psm 6"))
                ocr_candidates.append(pytesseract.image_to_string(
                    binary, config="--oem 3 --psm 11"))
            except Exception:
                pass

        extracted = "\n".join(part.strip()
                              for part in ocr_candidates if part and part.strip())
        if extracted.strip():
            return extracted, None
        return "", "OCR ran but no readable text was detected. Try a clearer image."

    return "", "Unsupported file type. Use PDF or image formats."


def _normalize_lab_value(metric: str, value: float, detected_unit: str) -> tuple[float, str, str]:
    clean_unit = (detected_unit or "").strip().lower().replace(" ", "")

    if metric in {"Hemoglobin", "MCHC"}:
        if clean_unit in {"g/l", "gl", "gm/l"}:
            return value / 10.0, "g/dL", "g/L → g/dL"
        if clean_unit in {"mmol/l", "mmoll"} and metric == "Hemoglobin":
            return value * 1.611, "g/dL", "mmol/L → g/dL"
        return value, "g/dL", "standardized"

    if metric == "MCH":
        if clean_unit in {"fmol"}:
            return value * 16.11, "pg", "fmol → pg"
        return value, "pg", "standardized"

    if metric == "MCV":
        return value, "fL", "standardized"

    return value, detected_unit or "", "as-detected"


def _safe_float(number_text: str) -> float | None:
    try:
        return float(number_text.replace(",", "."))
    except Exception:
        return None


def _metric_label_pattern(metric: str) -> str:
    patterns = {
        "Hemoglobin": r"(?:ha?em[o0]g(?:l|i|1)o(?:b|8)i?n|hemogiobin|haemogiobin|h\s*e\s*m\s*o\s*g\s*(?:l|i|1)\s*o\s*(?:b|8)\s*i?\s*n|hgb|h\s*g|\bhg\b|h\s*b|\bhb\b)",
        "MCHC": r"(?:m\s*c\s*h\s*c|mchc|mean\s*corpuscular\s*hemoglobin\s*concentration)",
        "MCH": r"(?:m\s*c\s*h(?!\s*c)|mch(?!c)|mean\s*corpuscular\s*hemoglobin(?!\s*concentration))",
        "MCV": r"(?:m\s*c\s*v|mcv|mean\s*corpuscular\s*volume)",
    }
    return patterns.get(metric, "")


def _has_explicit_hemoglobin_label(text: str) -> bool:
    scrubbed = re.sub(
        r"mean\s*corpuscular\s*ha?em[o0]g(?:l|i|1)o(?:b|8)i?n(?:\s*concentration)?",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    return bool(
        re.search(
            r"\bha?em[o0]g\w*\b|\bhemogiobin\b|\bhaemogiobin\b|\bhgb\b|\bh\s*g\b|\bhg\b|\bh\s*b\b|\bhb\b",
            scrubbed,
            flags=re.IGNORECASE,
        )
    )


def _extract_metric_value(
    lines: list[str],
    full_text: str,
    metric: str,
    used_line_indexes: set[int],
) -> tuple[float | None, str, int | None]:
    label_pattern = _metric_label_pattern(metric)
    metric_pattern = rf"(?i){label_pattern}\s*[:\-]?\s*([-+]?\d+(?:[\.,]\d+)?)\s*([a-zA-Zµμ/%]+(?:/[a-zA-Z]+)?)?"

    for idx, line in enumerate(lines):
        if idx in used_line_indexes:
            continue

        line_lower = line.lower()

        if metric == "Hemoglobin":
            if re.search(r"mean\s*corpuscular\s*ha?em[o0]g", line_lower):
                continue
            has_hemo_hint = _has_explicit_hemoglobin_label(line_lower)
            if "corpuscular" in line_lower and not has_hemo_hint:
                continue
            if not has_hemo_hint:
                continue

        if not label_pattern:
            continue

        match = re.search(metric_pattern, line)
        if match:
            parsed_value = _safe_float(match.group(1))
            if parsed_value is not None:
                return parsed_value, (match.group(2) or ""), idx

    for idx, line in enumerate(lines):
        if idx in used_line_indexes:
            continue

        line_lower = line.lower()
        if metric == "MCHC" and not re.search(r"\bmchc\b|\bmch\s*c\b", line_lower):
            continue
        if metric == "MCH" and not re.search(r"\bmch\b", line_lower):
            continue
        if metric == "MCH" and re.search(r"\bmchc\b|\bmch\s*c\b", line_lower):
            continue
        if metric == "MCV" and not re.search(r"\bmcv\b", line_lower):
            continue
        if metric == "Hemoglobin":
            if not _has_explicit_hemoglobin_label(line_lower):
                continue
            if re.search(r"mean\s*corpuscular\s*ha?em[o0]g", line_lower):
                continue
            has_hemo_hint = _has_explicit_hemoglobin_label(line_lower)
            if "corpuscular" in line_lower and not has_hemo_hint:
                continue

        fallback = re.search(
            r"([-+]?\d+(?:[\.,]\d+)?)\s*([a-zA-Zµμ/%]+(?:/[a-zA-Z]+)?)?", line)
        if fallback:
            parsed_value = _safe_float(fallback.group(1))
            if parsed_value is not None:
                return parsed_value, (fallback.group(2) or ""), idx

    if label_pattern:
        for global_match in re.finditer(metric_pattern, full_text):
            if metric == "Hemoglobin":
                left_context = full_text[max(
                    0, global_match.start() - 45):global_match.start()].lower()
                if "corpuscular" in left_context:
                    continue
                context_window = full_text[max(0, global_match.start(
                ) - 70):min(len(full_text), global_match.end() + 40)]
                if not _has_explicit_hemoglobin_label(context_window):
                    continue
            parsed_value = _safe_float(global_match.group(1))
            if parsed_value is not None:
                return parsed_value, (global_match.group(2) or ""), None

    return None, "", None


def _parse_blood_report(raw_text: str) -> tuple[dict[str, float | str], list[dict[str, str]]]:
    normalized_text = raw_text.replace("\r", "\n")
    normalized_text = re.sub(r"\bH\s*G\b", "Hemoglobin",
                             normalized_text, flags=re.IGNORECASE)
    normalized_text = re.sub(r"\bH\s*e\s*m\s*o\s*g\s*l\s*o\s*b\s*i\s*n\b",
                             "Hemoglobin", normalized_text, flags=re.IGNORECASE)
    normalized_text = re.sub(r"\bM\s*C\s*H\s*C\b",
                             "MCHC", normalized_text, flags=re.IGNORECASE)
    normalized_text = re.sub(r"\bM\s*C\s*H\b", "MCH",
                             normalized_text, flags=re.IGNORECASE)
    normalized_text = re.sub(r"\bM\s*C\s*V\b", "MCV",
                             normalized_text, flags=re.IGNORECASE)

    cleaned_text = " ".join(normalized_text.split())
    lines = [line.strip()
             for line in normalized_text.splitlines() if line.strip()]
    parsed: dict[str, float | str] = {}
    formatting_rows: list[dict[str, str]] = []
    used_line_indexes: set[int] = set()

    gender_match = re.search(r"\b(male|female)\b",
                             cleaned_text, flags=re.IGNORECASE)
    if gender_match:
        parsed["GenderLabel"] = gender_match.group(1).capitalize()

    metric_order = ["Hemoglobin", "MCHC", "MCH", "MCV"]

    for metric in metric_order:
        raw_value, detected_unit, line_index = _extract_metric_value(
            lines, cleaned_text, metric, used_line_indexes)
        if raw_value is None:
            continue
        if line_index is not None:
            used_line_indexes.add(line_index)
        normalized_value, normalized_unit, conversion_note = _normalize_lab_value(
            metric, raw_value, detected_unit)
        normalized_value = round(normalized_value, 2)
        parsed[metric] = normalized_value
        formatting_rows.append({
            "metric": metric,
            "raw": f"{raw_value:.2f} {detected_unit or '-'}",
            "normalized": f"{normalized_value:.2f} {normalized_unit}",
            "note": conversion_note,
        })

    return parsed, formatting_rows


@app.route("/")
def index():
    df = get_dataset()
    total = len(df) if df is not None else 0
    anemia_cases = 0
    if df is not None and "Result" in df.columns:
        anemia_cases = int(
            (pd.to_numeric(df.get("Result"), errors="coerce") == 1).sum())
    return render_template("index.html", total=total, anemia_cases=anemia_cases)


@app.route("/patient", methods=["GET"])
def patient_assessment():
    df = get_dataset()
    total = len(df) if df is not None else 0
    return render_template("patient_assessment.html", total=total)


@app.route("/scan-report", methods=["POST"])
def scan_report():
    df = get_dataset()
    total = len(df) if df is not None else 0
    uploaded_file = request.files.get("report_file")

    if uploaded_file is None or not (uploaded_file.filename or "").strip():
        return render_template(
            "patient_assessment.html",
            total=total,
            error="Please upload a blood report image or PDF file."
        )

    try:
        extracted_text, extraction_error = _extract_report_text(uploaded_file)
        if not extracted_text:
            return render_template(
                "patient_assessment.html",
                total=total,
                error=extraction_error or "Could not read report text. Try a clearer image/PDF."
            )

        parsed, formatting_rows = _parse_blood_report(extracted_text)
        gender_label = str(parsed.get("GenderLabel", "Female"))
        hemoglobin = float(parsed.get("Hemoglobin", 12.5))
        mch = float(parsed.get("MCH", 27.0))
        mchc = float(parsed.get("MCHC", 33.0))
        mcv = float(parsed.get("MCV", 85.0))

        missing_metrics = [
            metric for metric in ["Hemoglobin", "MCH", "MCHC", "MCV"] if metric not in parsed
        ]

        if missing_metrics:
            return render_template(
                "patient_assessment.html",
                total=total,
                gender_label=gender_label,
                hemoglobin=hemoglobin,
                mch=mch,
                mchc=mchc,
                mcv=mcv,
                error=f"Missing required values from report: {', '.join(missing_metrics)}",
                missing_metrics=missing_metrics,
                scan_formatting_rows=formatting_rows,
                scan_extracted_preview=" ".join(extracted_text.split())[:500],
            )

        gender_map = {"Female": 0.0, "Male": 1.0}
        user_values = {
            "Gender": gender_map.get(gender_label, 0.0),
            "Hemoglobin": hemoglobin,
            "MCH": mch,
            "MCHC": mchc,
            "MCV": mcv,
        }

        risk_percent, prediction, neighbors_used = estimate_anemia_risk(
            df, user_values)

        return render_template(
            "patient_assessment.html",
            total=total,
            result=True,
            gender_label=gender_label,
            hemoglobin=hemoglobin,
            mch=mch,
            mchc=mchc,
            mcv=mcv,
            risk_percent=risk_percent,
            prediction=prediction,
            neighbors_used=neighbors_used,
            scan_success="Report scanned successfully. Values extracted and assessment completed.",
            scan_formatting_rows=formatting_rows,
            scan_extracted_preview=" ".join(extracted_text.split())[:500],
        )
    except Exception:
        return render_template(
            "patient_assessment.html",
            total=total,
            error="Failed to process the uploaded report. Please try another file."
        )


@app.route("/assess", methods=["POST"])
def assess():
    df = get_dataset()
    try:
        gender_label = request.form.get("gender", "Female")
        gender_map = {"Female": 0.0, "Male": 1.0}

        raw_values = {
            "Hemoglobin": (request.form.get("hemoglobin") or "").strip(),
            "MCH": (request.form.get("mch") or "").strip(),
            "MCHC": (request.form.get("mchc") or "").strip(),
            "MCV": (request.form.get("mcv") or "").strip(),
        }
        missing_metrics = [metric for metric,
                           value in raw_values.items() if not value]
        if missing_metrics:
            return render_template(
                "patient_assessment.html",
                error=f"Missing required input values: {', '.join(missing_metrics)}",
                missing_metrics=missing_metrics,
                gender_label=gender_label,
                hemoglobin=raw_values["Hemoglobin"] or 12.5,
                mch=raw_values["MCH"] or 27.0,
                mchc=raw_values["MCHC"] or 33.0,
                mcv=raw_values["MCV"] or 85.0,
            )

        hemoglobin = float(raw_values["Hemoglobin"])
        mch = float(raw_values["MCH"])
        mchc = float(raw_values["MCHC"])
        mcv = float(raw_values["MCV"])

        user_values = {
            "Gender": gender_map.get(gender_label, 0.0),
            "Hemoglobin": hemoglobin,
            "MCH": mch,
            "MCHC": mchc,
            "MCV": mcv,
        }

        risk_percent, prediction, neighbors_used = estimate_anemia_risk(
            df, user_values)

        return render_template(
            "patient_assessment.html",
            result=True,
            gender_label=gender_label,
            hemoglobin=hemoglobin,
            mch=mch,
            mchc=mchc,
            mcv=mcv,
            risk_percent=risk_percent,
            prediction=prediction,
            neighbors_used=neighbors_used,
        )
    except Exception:
        return render_template(
            "patient_assessment.html",
            error="Invalid input. Please check your values and try again."
        )


@app.route("/download-report", methods=["POST"])
def download_report():
    def to_float(value, default=0.0):
        try:
            return float(value)
        except Exception:
            return default

    try:
        gender_label = request.form.get("gender_label", "Unknown")
        hemoglobin = to_float(request.form.get("hemoglobin"))
        mch = to_float(request.form.get("mch"))
        mchc = to_float(request.form.get("mchc"))
        mcv = to_float(request.form.get("mcv"))
        risk_percent = to_float(request.form.get("risk_percent"))
        prediction = int(to_float(request.form.get("prediction"), 0.0))
        neighbors_used = int(to_float(request.form.get("neighbors_used"), 0.0))

        if risk_percent < 25:
            risk_level = "LOW"
        elif risk_percent < 50:
            risk_level = "MILD"
        elif risk_percent < 75:
            risk_level = "MODERATE"
        else:
            risk_level = "SEVERE"

        recommendation = (
            "Anemia risk detected. Consult a licensed clinician for medical evaluation."
            if prediction == 1
            else "No anemia risk detected. Maintain a healthy lifestyle and periodic checkups."
        )

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_name = f"hemoscan_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        report_buffer = BytesIO(
            _build_health_report_pdf(
                timestamp=timestamp,
                gender_label=gender_label,
                hemoglobin=hemoglobin,
                mch=mch,
                mchc=mchc,
                mcv=mcv,
                risk_percent=risk_percent,
                risk_level=risk_level,
                prediction=prediction,
                neighbors_used=neighbors_used,
                recommendation=recommendation,
            )
        )

        return send_file(
            report_buffer,
            as_attachment=True,
            download_name=file_name,
            mimetype="application/pdf"
        )
    except Exception:
        return render_template(
            "patient_assessment.html",
            error="Unable to generate report right now. Please run an assessment again and retry download."
        )


@app.route("/dataset")
def dataset():
    df = get_dataset()
    total = len(df) if df is not None else 0
    anemia_cases = 0
    graphs_html = {}
    columns_info = []
    sample_data_html = ""
    chart_font_color = "#1f2937"

    if df is not None:
        # Calculate statistics
        if "Result" in df.columns:
            anemia_cases = int(
                (pd.to_numeric(df.get("Result"), errors="coerce") == 1).sum())

        # Get column information
        columns_info = [
            {
                "name": col,
                "type": _friendly_column_type(col, df[col]),
                "missing": int(df[col].isna().sum()),
                "unique": int(df[col].nunique())
            }
            for col in df.columns
        ]

        # Create graphs with proper sizing and ranges
        try:
            # Anemia distribution pie chart
            if "Result" in df.columns:
                result_series = pd.to_numeric(
                    df.get("Result"), errors="coerce").dropna()
                if not result_series.empty:
                    result_counts = result_series.value_counts().reindex(
                        [0.0, 1.0], fill_value=0)
                    fig_pie = px.pie(
                        values=result_counts.values,
                        names=["Healthy", "Anemia"],
                        title="Anemia Distribution",
                        color_discrete_sequence=["#28a745", "#e63946"],
                        labels={0: "Healthy", 1: "Anemia"}
                    )
                    fig_pie.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        textfont=dict(size=14, color='white'),
                        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>'
                    )
                    fig_pie.update_layout(
                        height=500,
                        width=None,
                        margin=dict(l=50, r=50, t=80, b=50),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Arial, sans-serif',
                                  size=12, color=chart_font_color),
                        title_font=dict(size=16, color=chart_font_color),
                        showlegend=True,
                        hovermode='closest'
                    )
                    graphs_html["pie"] = fig_pie.to_html(
                        full_html=False, include_plotlyjs='cdn', div_id='pie-chart')
        except Exception as e:
            print(f"Error generating pie chart: {e}")

        try:
            # Hemoglobin distribution
            if "Hemoglobin" in df.columns:
                hemo_df = df[["Hemoglobin"]].copy()
                hemo_df["Hemoglobin"] = pd.to_numeric(
                    hemo_df["Hemoglobin"], errors="coerce")
                hemo_df = hemo_df.dropna()
                if not hemo_df.empty:
                    fig_hemo = px.histogram(
                        hemo_df,
                        x="Hemoglobin",
                        nbins=30,
                        title="Hemoglobin Distribution",
                        labels={
                            "Hemoglobin": "Hemoglobin (g/dL)", "count": "Frequency"},
                        color_discrete_sequence=["#667eea"]
                    )
                    fig_hemo.update_traces(
                        marker_line_width=0,
                        hovertemplate='<b>Hemoglobin:</b> %{x:.2f} g/dL<br><b>Frequency:</b> %{y}<extra></extra>'
                    )
                    fig_hemo.update_layout(
                        height=500,
                        width=None,
                        margin=dict(l=60, r=50, t=80, b=60),
                        bargap=0.15,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Arial, sans-serif',
                                  size=11, color=chart_font_color),
                        title_font=dict(size=16, color=chart_font_color),
                        xaxis=dict(
                            title_font=dict(size=12, color=chart_font_color),
                            tickfont=dict(size=11, color=chart_font_color),
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(128,128,128,0.2)'
                        ),
                        yaxis=dict(
                            title_font=dict(size=12, color=chart_font_color),
                            tickfont=dict(size=11, color=chart_font_color),
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(128,128,128,0.2)',
                            rangemode='tozero'
                        ),
                        hovermode='closest'
                    )
                    graphs_html["hemoglobin"] = fig_hemo.to_html(
                        full_html=False, include_plotlyjs=False, div_id='hemo-chart')
        except Exception as e:
            print(f"Error generating hemoglobin chart: {e}")

        try:
            # MCV distribution
            if "MCV" in df.columns:
                mcv_df = df[["MCV"]].copy()
                mcv_df["MCV"] = pd.to_numeric(mcv_df["MCV"], errors="coerce")
                mcv_df = mcv_df.dropna()
                if not mcv_df.empty:
                    fig_mcv = px.histogram(
                        mcv_df,
                        x="MCV",
                        nbins=25,
                        title="Mean Corpuscular Volume (MCV) Distribution",
                        labels={"MCV": "MCV (fL)", "count": "Frequency"},
                        color_discrete_sequence=["#17a2b8"]
                    )
                    fig_mcv.update_traces(
                        marker_line_width=0,
                        hovertemplate='<b>MCV:</b> %{x:.2f} fL<br><b>Frequency:</b> %{y}<extra></extra>'
                    )
                    fig_mcv.update_layout(
                        height=500,
                        width=None,
                        margin=dict(l=60, r=50, t=80, b=60),
                        bargap=0.15,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Arial, sans-serif',
                                  size=11, color=chart_font_color),
                        title_font=dict(size=16, color=chart_font_color),
                        xaxis=dict(
                            title_font=dict(size=12, color=chart_font_color),
                            tickfont=dict(size=11, color=chart_font_color),
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(128,128,128,0.2)'
                        ),
                        yaxis=dict(
                            title_font=dict(size=12, color=chart_font_color),
                            tickfont=dict(size=11, color=chart_font_color),
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(128,128,128,0.2)',
                            rangemode='tozero'
                        ),
                        hovermode='closest'
                    )
                    graphs_html["mcv"] = fig_mcv.to_html(
                        full_html=False, include_plotlyjs=False, div_id='mcv-chart')
        except Exception as e:
            print(f"Error generating MCV chart: {e}")

        try:
            # Gender distribution
            if "Gender" in df.columns:
                gender_series = pd.to_numeric(
                    df["Gender"], errors="coerce").dropna()
                if not gender_series.empty:
                    gender_counts = gender_series.value_counts().sort_index()
                    fig_gender = px.bar(
                        x=["Female", "Male"],
                        y=[gender_counts.get(
                            0.0, 0), gender_counts.get(1.0, 0)],
                        title="Gender Distribution",
                        labels={"x": "Gender", "y": "Count"},
                        color_discrete_sequence=["#FF69B4", "#4169E1"]
                    )
                    fig_gender.update_traces(
                        marker_line_width=0,
                        hovertemplate='<b>%{x}</b><br><b>Count:</b> %{y}<extra></extra>'
                    )
                    fig_gender.update_layout(
                        height=500,
                        width=None,
                        margin=dict(l=60, r=50, t=80, b=60),
                        bargap=0.3,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Arial, sans-serif',
                                  size=11, color=chart_font_color),
                        title_font=dict(size=16, color=chart_font_color),
                        xaxis=dict(
                            title_font=dict(size=12, color=chart_font_color),
                            tickfont=dict(size=11, color=chart_font_color)
                        ),
                        yaxis=dict(
                            title_font=dict(size=12, color=chart_font_color),
                            tickfont=dict(size=11, color=chart_font_color),
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(128,128,128,0.2)',
                            rangemode='tozero'
                        ),
                        hovermode='closest'
                    )
                    graphs_html["gender"] = fig_gender.to_html(
                        full_html=False, include_plotlyjs=False, div_id='gender-chart')
        except Exception as e:
            print(f"Error generating gender chart: {e}")

        # Generate sample data table HTML
        try:
            expected_columns = ["Gender", "Hemoglobin",
                                "MCH", "MCHC", "MCV", "Result"]

            display_df = df.copy()
            if all(column in display_df.columns for column in expected_columns):
                display_df = display_df[expected_columns]

            display_df = display_df.head(10).copy()

            # Round only numeric columns safely
            numeric_cols = display_df.select_dtypes(
                include=["float64", "int64"]).columns
            display_df[numeric_cols] = display_df[numeric_cols].round(2)

            # OPTIONAL: map labels without breaking alignment
            if "Gender" in display_df.columns:
                display_df["Gender"] = display_df["Gender"].replace(
                    {0: "Female", 1: "Male"})

            if "Result" in display_df.columns:
                display_df["Result"] = display_df["Result"].replace(
                    {0: "Healthy", 1: "Anemia"})

            sample_data_html = display_df.to_html(
                classes="table table-sm table-striped",
                index=False,
                border=0,
                justify="center"
            )
        except Exception:
            # Fallback: just render without rounding
            try:
                sample_data_html = df.head(10).to_html(
                    classes="table table-sm table-striped", index=False, border=0)
            except Exception:
                sample_data_html = ""

    return render_template(
        "dataset_explorer.html",
        total=total,
        anemia_cases=anemia_cases,
        graphs_html=graphs_html,
        columns_info=columns_info,
        sample_data_html=sample_data_html,
        df_download_available=df is not None
    )


@app.route("/assistant", methods=["GET", "POST"])
def assistant():
    response = None
    query = ""
    error = None
    chat_history = []

    if request.method == "POST":
        action = request.form.get("action", "").strip().lower()
        raw_history = request.form.get("chat_history", "")
        chat_history = _parse_chat_history(raw_history)

        if action == "clear":
            return render_template(
                "ai_assistant.html",
                response=None,
                query="",
                error=None,
                chat_history=[],
            )

        query = request.form.get("query", "").strip()

        if not query:
            error = "Please enter a question."
        elif len(query) > MAX_ASSISTANT_QUERY_CHARS:
            error = f"Please keep your question under {MAX_ASSISTANT_QUERY_CHARS} characters."
        elif not _is_medical_query(query):
            error = "This assistant is restricted to medical topics. Please ask a medical question related to anemia, CBC, symptoms, labs, or treatment guidance."
        else:
            response, error = _assistant_reply(query, chat_history)
            if response and not error:
                chat_history.append({"role": "user", "content": query})
                chat_history.append({"role": "assistant", "content": response})
                chat_history = chat_history[-MAX_ASSISTANT_HISTORY_MESSAGES:]
                query = ""

    return render_template(
        "ai_assistant.html",
        response=response,
        query=query,
        error=error,
        chat_history=chat_history,
    )


@app.route("/selfcheck", methods=["GET", "POST"])
def selfcheck():
    result = None
    selected_answers: dict[str, str] = {}
    error = None

    if request.method == "POST":
        missing_questions: list[str] = []
        numeric_answers: dict[str, int] = {}

        for question in SELF_CHECK_QUESTIONS:
            q_key = str(question["key"])
            raw_value = (request.form.get(q_key) or "").strip()
            selected_answers[q_key] = raw_value
            if raw_value == "":
                missing_questions.append(str(question["question"]))
                continue
            try:
                numeric_answers[q_key] = int(raw_value)
            except Exception:
                missing_questions.append(str(question["question"]))

        if missing_questions:
            error = "Please answer all quiz questions before submitting."
        else:
            result = _selfcheck_assessment(numeric_answers)

    return render_template(
        "selfcheck.html",
        questions=SELF_CHECK_QUESTIONS,
        selected_answers=selected_answers,
        result=result,
        error=error,
    )


@app.route("/nutrition")
def nutrition():
    return render_template(
        "nutrition.html",
        daily_targets=NUTRITION_DAILY_TARGETS,
        foods=NUTRITION_FOODS,
        meal_examples=NUTRITION_MEAL_EXAMPLE,
    )


@app.route("/faq")
def faq():
    return render_template("faq.html")


@app.route("/download-dataset")
def download_dataset():
    try:
        return send_file(app.config["DATA_PATH"], as_attachment=True, download_name="anemia_dataset.csv")
    except Exception as e:
        return render_template("error.html", error=str(e)), 500


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)
