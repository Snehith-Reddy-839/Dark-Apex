from flask import Flask, render_template, request, redirect, url_for, send_file
from pathlib import Path
import pandas as pd
import os
import json
import plotly.express as px

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
    "Provide clear, evidence-aligned educational guidance in plain language. "
    "Do not provide diagnosis, prescriptions, or emergency triage decisions. "
    "When needed, advise consulting a licensed clinician. "
    "Keep responses concise and practical."
)


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


@app.route("/assess", methods=["POST"])
def assess():
    df = get_dataset()
    try:
        gender_label = request.form.get("gender", "Female")
        gender_map = {"Female": 0.0, "Male": 1.0}
        hemoglobin = float(request.form.get("hemoglobin", 12.5))
        mch = float(request.form.get("mch", 27.0))
        mchc = float(request.form.get("mchc", 33.0))
        mcv = float(request.form.get("mcv", 85.0))

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
