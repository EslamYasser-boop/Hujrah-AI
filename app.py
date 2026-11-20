import os
import sys
import json
import uuid
import base64
import importlib.util
import html

import streamlit as st
import streamlit.components.v1 as components

from pathlib import Path
from PIL import Image, ImageOps
from dotenv import load_dotenv
from google import genai
from google.genai import types

try:
    import pillow_heif  # pip install pillow-heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Florence-2-large"
RESULTS_DIR = BASE_DIR / "Results"
FONT_DIR = BASE_DIR / "fonts"

RESULTS_DIR.mkdir(exist_ok=True)

load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    try:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        GEMINI_API_KEY = None

# Model names can be changed from .env without touching the code:
#   GEMINI_MODEL=gemini-3.5-flash
#   GEMINI_FALLBACK_MODELS=gemini-3.5-flash-lite
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite").split(",")
    if m.strip()
]
RETRY_DELAYS = [2, 5, 10]

# USE_FLORENCE in .env:  auto (default, only with GPU) | 1 (always) | 0 (never)
USE_FLORENCE = os.getenv("USE_FLORENCE", "auto").strip().lower()

MIN_IMAGES = 1
MAX_IMAGES = 6

st.set_page_config(
    page_title="حُجرة | تحليل الشقق",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4().hex[:8]

SESSION_DIR = RESULTS_DIR / st.session_state["session_id"]
SESSION_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# HTML FIX
# Streamlit's markdown treats lines indented by 4+ spaces as code.
# This flattens every HTML snippet (except <style>).
# =====================================================

_original_markdown = st.markdown


def _flat_markdown(body, *args, **kwargs):

    if (
        kwargs.get("unsafe_allow_html")
        and isinstance(body, str)
        and not body.lstrip().startswith("<style")
    ):
        body = " ".join(
            line.strip()
            for line in body.splitlines()
            if line.strip()
        )

    return _original_markdown(body, *args, **kwargs)


st.markdown = _flat_markdown


# =====================================================
# 2. DESIGN SYSTEM  (Midnight Emerald & Gold)
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Readex+Pro:wght@300;400;500;600&display=swap');

:root {
    --void: #060D0A;
    --night: #0C1813;
    --forest: #12241C;
    --line: rgba(201, 162, 77, 0.22);
    --gold: #C9A24D;
    --gold-light: #EBD290;
    --champagne: #EFE7D2;
    --muted: #9BA89E;
    --glow: #1F8A68;
    --serif: 'Cormorant Garamond', Georgia, serif;
    --sans: 'Readex Pro', 'Segoe UI', Tahoma, sans-serif;
}

.stApp {
    background: var(--void);
    color: var(--champagne);
    font-family: var(--sans);
}

.stApp::before {
    content: "";
    position: fixed;
    inset: -20%;
    z-index: 0;
    pointer-events: none;
    background:
        radial-gradient(600px 400px at 15% 20%, rgba(31, 138, 104, 0.22), transparent 70%),
        radial-gradient(500px 380px at 85% 35%, rgba(201, 162, 77, 0.13), transparent 70%),
        radial-gradient(700px 450px at 50% 95%, rgba(31, 138, 104, 0.16), transparent 70%);
    animation: drift 26s ease-in-out infinite alternate;
}

@keyframes drift {
    from { transform: translate3d(-3%, -2%, 0) scale(1); }
    to   { transform: translate3d(3%, 3%, 0) scale(1.08); }
}

.block-container {
    position: relative;
    z-index: 1;
    max-width: 1060px;
    padding-top: 1.6rem;
    padding-bottom: 4rem;
}

h1, h2, h3, h4, p, label, input, textarea, button {
    font-family: var(--sans);
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

/* ---------- Hero ---------- */
.hero {
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 24px;
    padding: 46px 52px 0 52px;
    margin-bottom: 34px;
    border-radius: 26px;
    background: linear-gradient(135deg, #0F2A20 0%, #08130F 70%);
    border: 1px solid var(--line);
    box-shadow: 0 30px 80px rgba(0, 0, 0, 0.55);
    animation: rise 0.9s cubic-bezier(.2, .8, .2, 1) both;
}

@keyframes rise {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: none; }
}

.hero::after {
    content: "";
    position: absolute;
    top: 0;
    bottom: 0;
    width: 40%;
    left: -50%;
    background: linear-gradient(100deg, transparent, rgba(235, 210, 144, 0.10), transparent);
    transform: skewX(-18deg);
    animation: sweep 9s ease-in-out infinite;
    pointer-events: none;
}

@keyframes sweep {
    0%, 55% { left: -50%; }
    100%    { left: 130%; }
}

.hero-text {
    position: relative;
    z-index: 1;
    padding-bottom: 46px;
}

.hero h1 {
    margin: 0;
    padding: 0;
    font-family: var(--serif);
    font-size: 66px;
    font-weight: 700;
    line-height: 1;
    letter-spacing: 1px;
    background: linear-gradient(90deg, #B8913F, #F3DEA0, #B8913F, #F3DEA0);
    background-size: 300% 100%;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 7s linear infinite;
}

@keyframes shimmer {
    to { background-position: -300% 0; }
}

.hero p {
    margin: 16px 0 0 0;
    max-width: 440px;
    font-size: 16px;
    font-weight: 300;
    line-height: 1.7;
    color: #B9C6BC;
}

.arch {
    position: relative;
    z-index: 1;
    flex-shrink: 0;
    width: 170px;
    height: 200px;
    display: flex;
    align-items: flex-end;
    justify-content: center;
    padding-bottom: 26px;
    border: 1.5px solid var(--gold);
    border-bottom: none;
    border-radius: 85px 85px 0 0;
    background: linear-gradient(to top, rgba(235, 210, 144, 0.28), rgba(31, 138, 104, 0.10) 70%);
    animation: glow 4s ease-in-out infinite, float 6s ease-in-out infinite;
}

@keyframes glow {
    0%, 100% { box-shadow: 0 0 22px rgba(201, 162, 77, 0.18); }
    50%      { box-shadow: 0 0 52px rgba(201, 162, 77, 0.42); }
}

@keyframes float {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-12px); }
}

.arch span {
    font-size: 46px;
    font-weight: 500;
    line-height: 1;
    color: var(--gold-light);
}

/* ---------- Loader (single calm message) ---------- */
.loader {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 18px;
    padding: 42px 20px;
    margin: 14px 0;
    border: 1px solid var(--line);
    border-radius: 22px;
    background: linear-gradient(160deg, rgba(18, 36, 28, 0.9), rgba(12, 24, 19, 0.9));
}

.loader-arch {
    width: 64px;
    height: 84px;
    border: 1.5px solid var(--gold);
    border-bottom: none;
    border-radius: 32px 32px 0 0;
    background: linear-gradient(to top, rgba(235, 210, 144, 0.45), rgba(31, 138, 104, 0.08) 75%);
    background-size: 100% 200%;
    animation: glow 1.8s ease-in-out infinite, fill 2.4s ease-in-out infinite alternate;
}

@keyframes fill {
    from { background-position: 0 100%; }
    to   { background-position: 0 0; }
}

.loader-text {
    font-size: 20px;
    font-weight: 500;
    color: var(--gold-light);
}

.loader-text i {
    font-style: normal;
    animation: blink 1.2s infinite;
}

.loader-text i:nth-child(2) { animation-delay: 0.2s; }
.loader-text i:nth-child(3) { animation-delay: 0.4s; }

@keyframes blink {
    0%, 100% { opacity: 0.15; }
    50%      { opacity: 1; }
}

/* ---------- Section titles ---------- */
.section-title {
    margin: 30px 0 14px 0;
    font-family: var(--serif);
    font-size: 30px;
    font-weight: 700;
    letter-spacing: 0.3px;
    color: var(--champagne);
}

/* ---------- Property card ---------- */
@property --ang {
    syntax: '<angle>';
    initial-value: 0deg;
    inherits: false;
}

.property-card {
    margin-top: 18px;
    padding: 28px 30px 16px 30px;
    border: 1px solid transparent;
    border-radius: 20px;
    background:
        linear-gradient(160deg, #12241C, #0C1813) padding-box,
        conic-gradient(from var(--ang), rgba(201, 162, 77, 0.15) 55%, #F3DEA0 78%, rgba(201, 162, 77, 0.15) 100%) border-box;
    animation: spin 6s linear infinite;
}

@keyframes spin {
    to { --ang: 360deg; }
}

.property-title {
    font-family: var(--serif);
    font-size: 38px;
    font-weight: 700;
    line-height: 1.15;
    color: var(--champagne);
    margin-bottom: 6px;
}

.property-location {
    color: var(--muted);
    font-size: 15px;
    margin-bottom: 16px;
}

.price {
    font-family: var(--serif);
    font-size: 34px;
    font-weight: 700;
    line-height: 1.1;
    color: var(--gold-light);
}

.price-caption,
.rating-caption {
    color: var(--muted);
    font-size: 13px;
    margin-top: 4px;
}

.tag {
    display: inline-block;
    margin: 3px 6px 3px 0;
    padding: 6px 14px;
    border-radius: 999px;
    border: 1px solid var(--line);
    background: rgba(201, 162, 77, 0.08);
    color: var(--gold-light);
    font-size: 13px;
}

.stars {
    display: inline-block;
    font-size: 28px;
    letter-spacing: 3px;
    line-height: 1.2;
    background: linear-gradient(90deg, var(--gold-light) var(--pct, 100%), #2B3A32 var(--pct, 100%));
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
    animation: twinkle 3s ease-in-out infinite;
}

@keyframes twinkle {
    50% { filter: brightness(1.4) drop-shadow(0 0 8px rgba(235, 210, 144, 0.65)); }
}

.pdi-box,
.info-box {
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid var(--line);
    border-radius: 16px;
    transition: border-color 0.25s ease, background 0.25s ease;
}

.pdi-box:hover,
.info-box:hover {
    border-color: rgba(235, 210, 144, 0.55);
    background: rgba(201, 162, 77, 0.06);
}

.pdi-box { padding: 14px 18px; }

.info-box {
    padding: 18px 8px;
    text-align: center;
}

.pdi-number,
.info-number {
    font-family: var(--serif);
    font-size: 34px;
    font-weight: 700;
    line-height: 1.1;
    color: var(--gold-light);
}

.pdi-label,
.info-label {
    font-size: 13px;
    color: var(--muted);
    margin-top: 2px;
}

.details-heading {
    margin-bottom: 10px;
    font-family: var(--serif);
    font-size: 24px;
    font-weight: 700;
    color: var(--champagne);
}

.description-box {
    max-width: 78ch;
    padding: 20px 24px;
    border: 1px solid var(--line);
    border-left: 3px solid var(--gold);
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.03);
    color: #DCD5C2;
    line-height: 1.95;
}

/* ---------- Buttons ---------- */
.stButton > button,
.stDownloadButton > button {
    position: relative;
    overflow: hidden;
    min-height: 52px;
    border: 1px solid rgba(235, 210, 144, 0.6);
    border-radius: 14px;
    background: linear-gradient(135deg, #B8913F, #EBD290 50%, #B8913F);
    background-size: 200% 100%;
    color: #1A1405;
    font-size: 16px;
    font-weight: 600;
    box-shadow: 0 8px 26px rgba(201, 162, 77, 0.18);
    transition: transform 0.2s ease, box-shadow 0.2s ease, background-position 0.5s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-3px);
    background-position: 100% 0;
    border-color: #F3DEA0;
    color: #1A1405;
    box-shadow: 0 14px 38px rgba(201, 162, 77, 0.38);
}

.stButton > button:active,
.stDownloadButton > button:active {
    transform: translateY(0) scale(0.985);
}

.stButton > button p,
.stDownloadButton > button p {
    color: inherit;
}

/* ---------- Inputs ---------- */
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 10px;
}

div[data-baseweb="input"] > div {
    border: 1px solid var(--line);
}

div[data-baseweb="input"] > div:focus-within {
    border-color: var(--gold-light);
    box-shadow: 0 0 0 1px var(--gold-light), 0 0 18px rgba(201, 162, 77, 0.22);
}

div[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid var(--line);
    border-radius: 16px;
}

div[data-testid="stAlert"] {
    border-radius: 14px;
    border: 1px solid var(--line);
}

div[data-testid="stFileUploader"] section {
    padding: 26px;
    border: 1.5px dashed rgba(201, 162, 77, 0.45);
    border-radius: 18px;
    background: rgba(201, 162, 77, 0.04);
    transition: border-color 0.25s ease, background 0.25s ease;
}

div[data-testid="stFileUploader"] section:hover {
    border-color: var(--gold-light);
    background: rgba(201, 162, 77, 0.09);
}

div[data-testid="stFileUploader"] section button {
    border: 1px solid var(--gold);
    background: transparent;
    color: var(--gold-light);
}

div[data-testid="stFileUploader"] section button:hover {
    background: var(--gold);
    color: #1A1405;
}

div[data-testid="stImage"] img {
    width: 100%;
    max-width: 260px;
    height: auto;
    object-fit: cover;
    border: 1px solid var(--line);
    border-radius: 14px;
    transition: transform 0.3s ease, border-color 0.3s ease;
}

div[data-testid="stImage"] img:hover {
    transform: scale(1.03);
    border-color: var(--gold-light);
}

div[data-testid="stImage"] { overflow: visible; }

hr { border-color: var(--line); }

@media (max-width: 700px) {
    .hero { padding: 30px 24px 0 24px; }
    .hero-text { padding-bottom: 30px; }
    .hero h1 { font-size: 42px; }
    .arch { width: 100px; height: 120px; border-radius: 50px 50px 0 0; padding-bottom: 14px; }
    .arch span { font-size: 28px; }
    .property-card { padding: 20px 18px 10px 18px; }
    .property-title { font-size: 28px; }
    div[data-testid="stImage"] img { max-width: 100%; }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation: none !important;
        transition: none !important;
    }
}


/* Arabic RTL layout */
html, body, [class*="css"], .stApp {
    direction: rtl !important;
    text-align: right !important;
    font-family: 'Readex Pro', Tahoma, Arial, sans-serif !important;
}
.block-container, .stMarkdown, .stText, .stCaption, label, p, h1, h2, h3, h4 {
    direction: rtl;
    text-align: right;
}
div[data-testid="stExpander"], div[data-testid="stFileUploader"],
div[data-testid="stAlert"], div[data-testid="stMetric"],
div[data-testid="stTextInput"], div[data-testid="stNumberInput"] {
    direction: rtl;
    text-align: right;
}
input, textarea { direction: rtl !important; text-align: right !important; }
.stButton > button, .stDownloadButton > button {
    direction: rtl;
    font-family: 'Readex Pro', Tahoma, Arial, sans-serif !important;
}
div[data-testid="stImage"] figcaption { direction: rtl; text-align: center; }
.property-card, .description-box, .pdi-box, .info-box {
    direction: rtl;
    text-align: right;
}
.tag { margin: 3px 0 3px 6px; }
.description-box {
    border-left: 1px solid var(--line);
    border-right: 3px solid var(--gold);
}

</style>
""", unsafe_allow_html=True)


LOADER_HTML = """
<div class="loader">
    <div class="loader-arch"></div>
    <div class="loader-text">جاري التحليل<i>.</i><i>.</i><i>.</i></div>
</div>
"""


# =====================================================
# 3. HELPER FUNCTIONS
# =====================================================

def save_json(filename, content):

    path = SESSION_DIR / filename

    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=4)

    return path


def safe_text(value):
    return html.escape(str(value))


def prepare_image(uploaded_file):

    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)

    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background
    else:
        image = image.convert("RGB")

    image.thumbnail((1024, 1024))

    return image


def make_thumb(image, size=(480, 360)):
    return ImageOps.fit(image, size, method=Image.LANCZOS)


def get_verified_description(report):

    description = report.get("final_description", "")

    if not description:
        description = "No verified description was generated."

    return str(description)


def auto_download(pdf_bytes, filename):
    """Starts the PDF download in the browser as soon as it is ready."""

    b64 = base64.b64encode(pdf_bytes).decode()

    components.html(
        f"""
        <script>
        (function () {{
            const bin = atob("{b64}");
            const bytes = new Uint8Array(bin.length);
            for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
            const blob = new Blob([bytes], {{ type: "application/pdf" }});
            const a = document.createElement("a");
            a.href = URL.createObjectURL(blob);
            a.download = "{filename}";
            document.body.appendChild(a);
            a.click();
        }})();
        </script>
        """,
        height=0
    )


# =====================================================
# 4. GEMINI (retry + fallback model)
# =====================================================

def is_network_error(error):

    text = (type(error).__name__ + " " + str(error)).upper()

    return (
        "GETADDRINFO" in text
        or "CONNECTERROR" in text
        or "CONNECTTIMEOUT" in text
        or "NAME OR SERVICE NOT KNOWN" in text
        or "NETWORK IS UNREACHABLE" in text
    )


def is_temporary_error(error):

    code = getattr(error, "code", None) or getattr(error, "status_code", None)

    text = str(error).upper()

    return (
        is_network_error(error)
        or code in (429, 500, 502, 503, 504)
        or "UNAVAILABLE" in text
        or "RESOURCE_EXHAUSTED" in text
        or "OVERLOADED" in text
        or "HIGH DEMAND" in text
    )


def create_gemini_client():

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing. Add it to your .env file.")

    return genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(api_version="v1beta")
    )


def _make_config(fast):

    kwargs = {
        "response_mime_type": "application/json",
        "max_output_tokens": 8192,
    }

    if fast:
        # Gemini 3.x uses thinking_level (not thinking_budget)
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")

    return types.GenerateContentConfig(**kwargs)


def _generate_once(client, model_name, contents):
    """Tries the fast (low thinking) setup; if the model or the installed
    SDK rejects it, retries immediately without the thinking setting."""

    try:
        return client.models.generate_content(
            model=model_name,
            contents=contents,
            config=_make_config(True)
        )

    except Exception as error:

        if "THINKING" in str(error).upper():
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=_make_config(False)
            )

        raise


def call_gemini_json(client, contents):

    import time

    last_error = None

    for model_name in [GEMINI_MODEL] + FALLBACK_MODELS:

        for delay in [0] + RETRY_DELAYS:

            if delay:
                time.sleep(delay)

            try:
                return _generate_once(client, model_name, contents)

            except Exception as error:

                last_error = error

                if is_network_error(error) and delay == RETRY_DELAYS[-1]:
                    raise

                if not is_temporary_error(error):

                    # A real error on the main model must be shown;
                    # a failing backup model is simply skipped.
                    if model_name == GEMINI_MODEL:
                        raise

                    break

    raise RuntimeError(
        f"All Gemini models failed. Last error: {last_error}"
    ) from last_error


def extract_json(response):

    response_text = response.text

    if not response_text:
        raise ValueError("Gemini returned an empty response.")

    text = response_text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "")

    text = text.strip()

    try:
        return json.loads(text, strict=False)

    except json.JSONDecodeError:

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end <= start:
            raise

        return json.loads(text[start:end + 1], strict=False)


def generate_json(client, contents, attempts=2):

    last_error = None

    for _ in range(attempts):

        response = call_gemini_json(client, contents)

        try:
            return extract_json(response)

        except (json.JSONDecodeError, ValueError) as error:
            last_error = error

    raise ValueError(f"Gemini returned invalid JSON: {last_error}")


# =====================================================
# 5. FLORENCE (optional, only used when it is fast)
# =====================================================

def florence_enabled():

    if not (MODEL_DIR / "config.json").exists():
        return False

    if USE_FLORENCE in ("1", "true", "yes", "on"):
        return True

    if USE_FLORENCE in ("0", "false", "no", "off"):
        return False

    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def load_florence():

    import torch
    from transformers import CLIPImageProcessor, BartTokenizerFast

    device = "cuda" if torch.cuda.is_available() else "cpu"

    spec = importlib.util.spec_from_file_location(
        "florence_pkg",
        str(MODEL_DIR / "__init__.py"),
        submodule_search_locations=[str(MODEL_DIR)]
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Florence local package.")

    pkg = importlib.util.module_from_spec(spec)
    sys.modules["florence_pkg"] = pkg
    spec.loader.exec_module(pkg)

    from florence_pkg.configuration_florence2 import Florence2Config
    from florence_pkg.modeling_florence2 import Florence2ForConditionalGeneration
    from florence_pkg.processing_florence2 import Florence2Processor

    config = Florence2Config.from_pretrained(str(MODEL_DIR), local_files_only=True)

    image_processor = CLIPImageProcessor.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )

    tokenizer = BartTokenizerFast.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )

    processor = Florence2Processor(
        image_processor=image_processor,
        tokenizer=tokenizer
    )

    dtype = torch.float16 if device == "cuda" else torch.float32

    model = Florence2ForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        config=config,
        local_files_only=True,
        torch_dtype=dtype
    ).to(device)

    model.eval()

    return processor, model, device, dtype


def analyze_with_florence(image, processor, model, device, dtype):

    import torch

    prompt = "<MORE_DETAILED_CAPTION>"

    inputs = processor(text=prompt, images=image, return_tensors="pt")

    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs["pixel_values"].to(device, dtype)

    with torch.inference_mode():

        generated_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=256,
            num_beams=1
        )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False
    )[0]

    parsed = processor.post_process_generation(
        generated_text,
        task=prompt,
        image_size=(image.width, image.height)
    )

    return parsed.get(prompt, generated_text)


# =====================================================
# 6. ONE-SHOT LISTING ANALYSIS
#    (structure + verification + listing details in ONE call)
# =====================================================

def run_listing_analysis(client, images, location, florence_results):

    names = ", ".join(images.keys())

    hints = ""

    if florence_results:
        hints = (
            "\nSecondary hints from an image captioning model "
            "(may contain mistakes; trust the photos over these):\n"
            + json.dumps(florence_results, ensure_ascii=False)
        )

    prompt = f"""
أنت مساعد متخصص في تحليل العقارات لمنصة سكن طلابي في مصر.
افحص صور الشقة ({names}) وأنشئ إعلانًا دقيقًا ومراجعًا وجاهزًا لمراجعة المالك.

عنوان الشقة الذي أدخله المالك: {location}
ملاحظات مساعدة من نموذج تحليل الصور (قد تكون بالإنجليزية وقد تحتوي على أخطاء؛ اعتمد على الصور الأصلية أولًا، واكتب كل المخرجات بالعربية):
{hints}

شرط اللغة الأساسي:
- جميع القيم النصية التي ستظهر للمستخدم يجب أن تكون باللغة العربية فقط.
- استخدم العربية الفصحى الواضحة والمهنية.
- اسم الإعلان والوصف والمميزات وأسباب التقييم والإيجار والسعة والثقة وعدم اليقين كلها بالعربية.
- أسماء مفاتيح JSON يجب أن تبقى باللغة الإنجليزية كما هي تمامًا حتى لا يتعطل التطبيق.
- لا تكتب أي جملة إنجليزية داخل القيم النصية.
- اكتب أسماء الغرف والأثاث والألوان بالعربية.

قواعد صارمة:
- اذكر فقط ما يظهر بوضوح. لا تخترع أثاثًا أو ألوانًا أو ديكورات أو مرافق.
- إذا لم تكن متأكدًا من نوع خامة أو جهاز، فاستخدم وصفًا عامًا مثل كرسي أو جهاز مثبت على الحائط.
- لا تحدد نوع الخامة أو الجهاز دون دليل بصري واضح.
- لا تصف قطعة قماش بأنها سجادة إلا إذا كانت على الأرض.
- لا تدّعِ مساحة أو نظافة أو إضاءة طبيعية أو موقعًا أو جودة دون دليل بصري مباشر.
- عدم ظهور شيء في الصور لا يعني عدم وجوده في الشقة.
- لا تفترض أن الصور المختلفة تُظهر غرفًا مختلفة.
- أعد JSON صالحًا فقط، ولا تضع علامات اقتباس مزدوجة داخل القيم النصية.

الحقول المطلوبة:
- listing_name: اسم عربي قصير وطبيعي للإعلان، دون اختلاق موقع أو مرافق.
- verified_description: وصف عقاري عربي مهني من 60 إلى 110 كلمات، كوصف مترابط واحد للشقة وليس قائمة صورة بصورة. اذكر العناصر الظاهرة بوضوح فقط والمهمة للطالب مثل الأثاث والسرير والمكتب وألوان الجدران والأرضيات والستائر والتجهيزات الظاهرة. تجاهل الأشياء العابرة مثل الزجاجات والملابس والحقائب والأسلاك والأحذية. لا تذكر الصور أو اللقطات ولا تضف جملة تسويقية ختامية.
- highlights: من 3 إلى 5 مميزات قصيرة بالعربية، ظاهرة فقط.
- estimated_rating: تقييم من 0 إلى 5 بزيادات 0.5 بناءً على الأثاث والتشطيبات والتنوع الظاهر. لا تذكر النظافة. هذا تقدير آلي.
- estimated_rent_min_egp و estimated_rent_max_egp: نطاق إيجار شهري واقعي بالجنيه المصري لسكن طلابي في المنطقة التي أدخلها المالك.
- estimated_capacity: عدد الأفراد الذين يمكن أن تستوعبهم أماكن النوم الظاهرة بشكل معقول، عدد صحيح لا يقل عن 1.
- *_reason: جملة عربية قصيرة لكل سبب، اعتمادًا على العناصر الظاهرة فقط، وعلى عنوان المالك عند تقدير الإيجار.
- confidence: استخدم "مرتفع" فقط إذا كانت كل الصور واضحة وكل الادعاءات مؤكدة. استخدم "متوسط" أو "منخفض" عند وجود أي غموض.
- images: عنصر لكل صورة، مع room_type أو "غير واضح"، ومستوى الثقة بالعربية، والأثاث الظاهر فقط، والألوان الرئيسية بالعربية.
- uncertainty: حتى 3 ملاحظات قصيرة بالعربية حول ما لا يمكن تأكيده.

أعد هذا الهيكل بالضبط، مع إبقاء مفاتيح JSON كما هي:
{{
  "listing_name": "",
  "verified_description": "",
  "highlights": [],
  "estimated_rating": 0.0,
  "rating_reason": "",
  "estimated_rent_min_egp": 0,
  "estimated_rent_max_egp": 0,
  "rent_reason": "",
  "estimated_capacity": 1,
  "capacity_reason": "",
  "confidence": "متوسط",
  "images": [
    {{
      "image_name": "",
      "room_type": "",
      "confidence_level": "متوسط",
      "furniture": [],
      "colors": []
    }}
  ],
  "uncertainty": []
}}
"""

    contents = [prompt]

    for image_name, data in images.items():
        contents.append(f"Photo: {image_name}")
        contents.append(data["image"])

    raw = generate_json(client, contents)

    # ----- clean and validate -----
    def as_list(value, limit):
        if not isinstance(value, list):
            return []
        return [str(v).strip() for v in value if str(v).strip()][:limit]

    try:
        rating = float(raw.get("estimated_rating", 0))
    except (TypeError, ValueError):
        rating = 0.0

    rating = max(0.0, min(5.0, round(rating * 2) / 2))

    try:
        rent_min = int(float(raw.get("estimated_rent_min_egp", 0)))
        rent_max = int(float(raw.get("estimated_rent_max_egp", 0)))
    except (TypeError, ValueError):
        rent_min, rent_max = 0, 0

    if rent_min > rent_max:
        rent_min, rent_max = rent_max, rent_min

    try:
        capacity = max(1, int(float(raw.get("estimated_capacity", 1))))
    except (TypeError, ValueError):
        capacity = 1

    if rent_max > 0 and rent_min != rent_max:
        rent_text = f"من {rent_min:,} إلى {rent_max:,} جنيه مصري"
    elif rent_max > 0:
        rent_text = f"{rent_max:,} جنيه مصري"
    else:
        rent_text = "لم يتم التقدير"

    # The model gives 'high' confidence too easily:
    # lower it when it also listed things it could not confirm.
    confidence_raw = str(raw.get("confidence", "")).strip().lower()
    confidence_map = {
        "high": "مرتفع", "medium": "متوسط", "low": "منخفض",
        "مرتفع": "مرتفع", "متوسط": "متوسط", "منخفض": "منخفض"
    }
    confidence = confidence_map.get(confidence_raw, "متوسط")
    uncertainty_list = as_list(raw.get("uncertainty"), 3)

    if confidence == "مرتفع" and uncertainty_list:
        confidence = "متوسط"

    per_image = []

    for item in raw.get("images", []) if isinstance(raw.get("images"), list) else []:
        if isinstance(item, dict):
            per_image.append({
                "image_name": str(item.get("image_name", "")),
                "room_type": str(item.get("room_type", "")).strip() or "غير واضح",
                "confidence_level": str(item.get("confidence_level", "")).strip(),
                "furniture": as_list(item.get("furniture"), 8),
                "colors": as_list(item.get("colors"), 5),
            })

    return {
        "listing_name": str(raw.get("listing_name", "")).strip() or "إعلان شقة",
        "verified_description": str(raw.get("verified_description", "")).strip(),
        "highlights": as_list(raw.get("highlights"), 5),
        "estimated_rating": rating,
        "rating_reason": str(raw.get("rating_reason", "")).strip(),
        "estimated_rent_min_egp": rent_min,
        "estimated_rent_max_egp": rent_max,
        "estimated_rent_text": rent_text,
        "rent_reason": str(raw.get("rent_reason", "")).strip(),
        "estimated_capacity": capacity,
        "capacity_reason": str(raw.get("capacity_reason", "")).strip(),
        "confidence": confidence,
        "images": per_image,
        "uncertainty": uncertainty_list,
    }


# =====================================================
# 7. PDF REPORT  (designed page by page, RTL Arabic)
# =====================================================

def build_report_pdf(final_report, images):
    """Draws a designed, multi-section RTL Arabic PDF and returns its bytes."""

    from io import BytesIO
    from datetime import datetime
    import math

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase.pdfmetrics import stringWidth

    # ---------- Arabic libraries (must be imported BEFORE the font check) ----------
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError as font_error:
        raise RuntimeError(
            "ثبّت مكتبات دعم العربية في PDF بالأمر: "
            "python -m pip install arabic-reshaper python-bidi fonttools"
        ) from font_error

    # ---------- Fonts ----------
    # arabic_reshaper outputs "Arabic Presentation Forms" (U+FB50-FDFF / U+FE70-FEFF).
    # The font MUST contain those glyphs, otherwise they render as squares.
    # Amiri / Noto Naskh Arabic (static TTF) contain them. Readex Pro usually does not.
    # sample covers: ligature lam-alef, hamza, ta marbuta, comma, digits
    FONT_TEST_TEXT = arabic_reshaper.reshape(
        "حجرة الشقة تحليل الإيجار مُراجع بالذكاء الاصطناعي ، ؟ 0123456789"
    )

    def supports_arabic(path):
        """True only if the font has every glyph the reshaper will output."""
        try:
            from fontTools.ttLib import TTFont as FTFont
            cmap = FTFont(str(path), lazy=True).getBestCmap() or {}
        except ImportError:
            return True  # fonttools not installed: skip the check
        except Exception:
            return False
        return all(ord(ch) in cmap for ch in FONT_TEST_TEXT if not ch.isspace())

    def ensure_amiri():
        """Downloads Amiri into fonts/ if it is missing (e.g. on Streamlit Cloud)."""
        import urllib.request
        FONT_DIR.mkdir(exist_ok=True)
        base = "https://github.com/google/fonts/raw/main/ofl/amiri/"
        for name in ("Amiri-Regular.ttf", "Amiri-Bold.ttf"):
            target = FONT_DIR / name
            if not target.is_file():
                try:
                    urllib.request.urlretrieve(base + name, str(target))
                except Exception:
                    pass

    def candidates(is_bold):
        suffix = "Bold" if is_bold else "Regular"
        found = [
            FONT_DIR / f"Amiri-{suffix}.ttf",
            FONT_DIR / f"NotoNaskhArabic-{suffix}.ttf",
        ]
        if FONT_DIR.is_dir():
            found += sorted(FONT_DIR.glob(f"**/*{suffix}*.ttf"))
        found += [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if is_bold
                 else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("C:/Windows/Fonts/tahomabd.ttf" if is_bold else "C:/Windows/Fonts/tahoma.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf" if is_bold else "C:/Windows/Fonts/arial.ttf"),
        ]
        return found

    def pick(is_bold):
        return next((p for p in candidates(is_bold) if p.is_file() and supports_arabic(p)), None)

    reg, bold = pick(False), pick(True)

    if reg is None or bold is None:
        ensure_amiri()
        reg, bold = pick(False), pick(True)

    if reg is None or bold is None:
        existing = [p.name for p in FONT_DIR.glob("**/*.ttf")] if FONT_DIR.is_dir() else []
        raise RuntimeError(
            "لا يوجد خط عربي صالح. الملفات الموجودة في fonts: "
            + (", ".join(existing) or "لا شيء")
            + ". ارفع Amiri-Regular.ttf و Amiri-Bold.ttf (static TTF) إلى مجلد fonts في الريبو."
        )

    pdfmetrics.registerFont(TTFont("Body", str(reg), validate=1))
    pdfmetrics.registerFont(TTFont("Body-Bold", str(bold), validate=1))
    pdfmetrics.registerFontFamily(
        "Body", normal="Body", bold="Body-Bold",
        italic="Body", boldItalic="Body-Bold"
    )
    FONT, FONT_BOLD = "Body", "Body-Bold"

    def has_arabic(value):
        return any(
            ("\u0600" <= ch <= "\u08FF")
            or ("\uFB50" <= ch <= "\uFDFF")
            or ("\uFE70" <= ch <= "\uFEFF")
            for ch in str(value)
        )

    def shape(value):
        value = str(value if value is not None else "")
        if not has_arabic(value):
            return value
        return get_display(arabic_reshaper.reshape(value), base_dir="R")

    # ---------- Palette ----------
    NIGHT = HexColor("#08130F")
    GOLD = HexColor("#C9A24D")
    GOLD_LIGHT = HexColor("#EBD290")
    CHAMPAGNE = HexColor("#EFE7D2")
    MUTED_DARK = HexColor("#9BA89E")

    IVORY = HexColor("#F7F2E4")
    PAPER = HexColor("#FFFDF6")
    EMERALD_900 = HexColor("#0B3D2E")
    EMERALD_600 = HexColor("#11694F")
    MINT = HexColor("#E2EEE3")
    MINT_LINE = HexColor("#C9DFCF")
    BRASS = HexColor("#B58A3A")
    INK = HexColor("#17261F")
    MUTED = HexColor("#66776D")
    LINE = HexColor("#E3DBC4")
    STAR_OFF = HexColor("#D8D0B8")

    COLOR_MAP = {
        "white": "#F4F4F0", "beige": "#D9C7A3", "cream": "#EFE4C8",
        "brown": "#7A5230", "dark": "#3B3029", "wood": "#9A6B3F",
        "wooden": "#9A6B3F", "gray": "#9A9A9A", "grey": "#9A9A9A",
        "black": "#1C1C1C", "blue": "#3F6EA8", "navy": "#1F3558",
        "green": "#4C8A5B", "red": "#B04040", "pink": "#E3A2B1",
        "yellow": "#E3C55A", "orange": "#D98A3D", "gold": "#C9A24D",
        "silver": "#BFC3C7", "tan": "#C8A877", "ivory": "#F3EDD7",
        "purple": "#7C5BA6", "teal": "#2F8C8A", "turquoise": "#40B5B0",
        "light": "#E9E6DC", "terracotta": "#B9674A",
        "أبيض": "#F4F4F0", "بيج": "#D9C7A3", "كريمي": "#EFE4C8",
        "بني": "#7A5230", "خشبي": "#9A6B3F", "رمادي": "#9A9A9A",
        "أسود": "#1C1C1C", "أزرق": "#3F6EA8", "كحلي": "#1F3558",
        "أخضر": "#4C8A5B", "أحمر": "#B04040", "وردي": "#E3A2B1",
        "أصفر": "#E3C55A", "برتقالي": "#D98A3D", "ذهبي": "#C9A24D",
        "فضي": "#BFC3C7", "عاجي": "#F3EDD7", "بنفسجي": "#7C5BA6",
        "تركواز": "#40B5B0", "فاتح": "#E9E6DC",
    }

    PAGE_W, PAGE_H = A4
    MARGIN = 18 * mm
    CONTENT_W = PAGE_W - 2 * MARGIN
    BOTTOM = 24 * mm
    R = PAGE_W - MARGIN          # right edge of the content area (RTL anchor)

    info = final_report.get("listing_information", {})
    est = final_report.get("listing_estimate", {})
    per_image = {d.get("image_name"): d for d in est.get("images", [])}

    listing_name = info.get("name", "إعلان شقة")
    location = info.get("location", "")

    try:
        rating = float(info.get("demo_rating", 0))
    except (TypeError, ValueError):
        rating = 0.0

    try:
        people = max(int(info.get("required_people", 1)), 1)
        capacity = max(int(info.get("test_capacity", 1)), 1)
    except (TypeError, ValueError):
        people, capacity = 1, 1

    fit = min(100, round(capacity / people * 100))

    rent_text = info.get("monthly_rent", "لم يتم التقدير")
    rent_min = est.get("estimated_rent_min_egp", 0) or 0
    rent_max = est.get("estimated_rent_max_egp", 0) or 0

    buffer = BytesIO()
    c = rl_canvas.Canvas(buffer, pagesize=A4)
    c.setTitle(f"{listing_name} | حُجرة")
    c.setAuthor("حُجرة")

    state = {"y": 0.0, "page": 1}

    # ---------- Small drawing helpers ----------
    def text(x, y, value, font=None, size=10, color=INK, align="left"):

        c.setFont(font or FONT, size)
        c.setFillColor(color)
        value = shape(value)

        if align == "center":
            c.drawCentredString(x, y, value)
        elif align == "right":
            c.drawRightString(x, y, value)
        else:
            c.drawString(x, y, value)

    def fit_size(value, font, size, max_w, min_size=7.0):
        """Shrinks the font size until the shaped text fits in max_w."""
        shaped = shape(value)
        while size > min_size and stringWidth(shaped, font, size) > max_w:
            size -= 0.5
        return size

    def spaced(x_center, y, value, size, color, spacing=3):

        raw_value = str(value if value is not None else "")
        if has_arabic(raw_value):
            value = shape(raw_value)
            spacing = 0
        else:
            value = raw_value.upper()
        width = stringWidth(value, FONT_BOLD, size) + spacing * max(len(value) - 1, 0)

        t = c.beginText(x_center - width / 2, y)
        t.setFont(FONT_BOLD, size)
        t.setFillColor(color)
        t.setCharSpace(spacing)
        t.textOut(value)
        c.drawText(t)

    def wrap(value, font, size, width):
        """Wraps by measuring the SHAPED width (connected letters have
        different widths than isolated ones), then returns shaped lines."""

        words = str(value if value is not None else "").split()
        if not words:
            return []

        lines, cur = [], ""

        for word in words:
            test = (cur + " " + word).strip()
            if stringWidth(shape(test), font, size) <= width or not cur:
                cur = test
            else:
                lines.append(cur)
                cur = word

        if cur:
            lines.append(cur)

        return [shape(line) for line in lines]

    def arch_path(x, y, w, h):

        r = w / 2
        p = c.beginPath()
        p.moveTo(x, y)
        p.lineTo(x, y + h - r)
        p.arcTo(x, y + h - 2 * r, x + w, y + h, startAng=180, extent=-180)
        p.lineTo(x + w, y)
        return p

    def fitted(pil_image, w, h):
        return ImageReader(
            ImageOps.fit(
                pil_image,
                (max(int(w / mm * 9), 10), max(int(h / mm * 9), 10)),
                method=Image.LANCZOS
            )
        )

    def draw_photo(pil_image, x, y, w, h, radius=3 * mm):

        c.saveState()
        p = c.beginPath()
        p.roundRect(x, y, w, h, radius)
        c.clipPath(p, stroke=0, fill=0)
        c.drawImage(fitted(pil_image, w, h), x, y, w, h)
        c.restoreState()

        c.setStrokeColor(LINE)
        c.setLineWidth(0.8)
        c.roundRect(x, y, w, h, radius, fill=0, stroke=1)

    def draw_stars(x, y, size, value, on=BRASS, off=STAR_OFF):

        gap = size * 0.25
        r = size / 2

        def star(cx, cy):
            p = c.beginPath()
            for i in range(10):
                angle = math.pi / 2 + i * math.pi / 5
                rad = r if i % 2 == 0 else r * 0.42
                px, py = cx + rad * math.cos(angle), cy + rad * math.sin(angle)
                if i == 0:
                    p.moveTo(px, py)
                else:
                    p.lineTo(px, py)
            p.close()
            return p

        for i in range(5):

            cx = x + r + i * (size + gap)
            cy = y + r
            part = max(0.0, min(1.0, value - i))

            c.setFillColor(off)
            c.drawPath(star(cx, cy), fill=1, stroke=0)

            if part > 0:
                c.saveState()
                clip = c.beginPath()
                clip.rect(cx - r, y, 2 * r * part, size)
                c.clipPath(clip, stroke=0, fill=0)
                c.setFillColor(on)
                c.drawPath(star(cx, cy), fill=1, stroke=0)
                c.restoreState()

    def card(x, y_top, w, h, fill=PAPER, border=LINE):

        c.setFillColor(fill)
        c.setStrokeColor(border)
        c.setLineWidth(0.8)
        c.roundRect(x, y_top - h, w, h, 3 * mm, fill=1, stroke=1)

    def color_for(item):
        for word in str(item).lower().replace("-", " ").split():
            if word in COLOR_MAP:
                return COLOR_MAP[word]
            if word.startswith("ال") and word[2:] in COLOR_MAP:
                return COLOR_MAP[word[2:]]
        return None

    def chips(x_right, y_top, max_w, items, fill, border, color,
              swatch=False, draw=True, size=8.5):
        """RTL chip flow: starts at x_right and moves to the left."""

        if not items:
            return 0

        row_h = 6.4 * mm
        gap = 1.8 * mm
        pad = 2.6 * mm
        cx, cy = x_right, y_top          # cx = right edge of the next chip
        left_limit = x_right - max_w

        for item in items:

            label = shape(item)
            extra = 4 * mm if swatch else 0
            w = stringWidth(label, FONT, size) + 2 * pad + extra

            if cx - w < left_limit and cx < x_right:
                cx = x_right
                cy -= row_h + gap

            if draw:
                c.setFillColor(fill)
                c.setStrokeColor(border)
                c.setLineWidth(0.6)
                c.roundRect(cx - w, cy - row_h, w, row_h, row_h / 2, fill=1, stroke=1)

                tx = cx - pad

                if swatch:
                    hex_color = color_for(item)
                    if hex_color:
                        c.setFillColor(HexColor(hex_color))
                        c.setStrokeColor(LINE)
                        c.circle(cx - pad - 1.3 * mm, cy - row_h / 2, 1.5 * mm, fill=1, stroke=1)
                    tx -= extra

                text(tx, cy - row_h + 2.2 * mm, item, FONT, size, color, align="right")

            cx -= w + gap

        return y_top - (cy - row_h)

    # ---------- Page chrome ----------
    def body_page():

        c.setFillColor(IVORY)
        c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

        c.setFillColor(EMERALD_900)
        c.rect(0, PAGE_H - 14 * mm, PAGE_W, 14 * mm, fill=1, stroke=0)

        c.setFillColor(GOLD)
        c.rect(0, PAGE_H - 14 * mm - 0.8, PAGE_W, 0.8, fill=1, stroke=0)

        # RTL: brand on the right, listing name on the left
        text(R, PAGE_H - 9 * mm, "حُجرة", FONT_BOLD, 9, GOLD_LIGHT, align="right")
        text(MARGIN, PAGE_H - 9 * mm, listing_name, FONT, 9, HexColor("#BFD8C8"))

        c.setStrokeColor(LINE)
        c.setLineWidth(0.8)
        c.line(MARGIN, 16 * mm, PAGE_W - MARGIN, 16 * mm)

        text(R, 11 * mm,
             "حُجرة | تحليل بمساعدة الذكاء الاصطناعي. جميع الإعلانات تتطلب موافقة المالك.",
             FONT, 8, MUTED, align="right")
        text(MARGIN, 11 * mm, f"{state['page']}", FONT_BOLD, 8, EMERALD_900)

        state["y"] = PAGE_H - 28 * mm

    def new_page():
        c.showPage()
        state["page"] += 1
        body_page()

    def ensure(h):
        if state["y"] - h < BOTTOM:
            new_page()

    def section(number, title, need=22 * mm):
        """`need` = space to keep free under the title, so a title is never
        left alone at the bottom of a page."""

        ensure(need)

        y = state["y"]
        text(R, y, f"{number:02d}", FONT_BOLD, 9, BRASS, align="right")

        tx = R - 8 * mm
        text(tx, y, title, FONT_BOLD, 15, EMERALD_900, align="right")

        c.setFillColor(BRASS)
        c.rect(tx - 12 * mm, y - 2.6 * mm, 12 * mm, 0.9, fill=1, stroke=0)

        state["y"] -= 11 * mm

    # =====================================================
    # PAGE 1: COVER
    # =====================================================
    c.setFillColor(NIGHT)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    for i, alpha in enumerate([0.05, 0.07, 0.09, 0.11]):
        c.setFillColor(HexColor("#1F8A68"))
        c.setFillAlpha(alpha)
        c.circle(PAGE_W / 2, PAGE_H - 100 * mm, (120 - i * 20) * mm, fill=1, stroke=0)
    c.setFillAlpha(1)

    spaced(PAGE_W / 2, PAGE_H - 16 * mm, "حُجرة", 11, GOLD_LIGHT, spacing=5)

    arch_w, arch_h = 104 * mm, 140 * mm
    ax = (PAGE_W - arch_w) / 2
    ay = PAGE_H - 28 * mm - arch_h

    for i, alpha in enumerate([0.55, 0.35, 0.2]):
        d = (i + 1) * 6 * mm
        c.setStrokeColor(GOLD)
        c.setStrokeAlpha(alpha)
        c.setLineWidth(0.8)
        c.drawPath(arch_path(ax - d, ay, arch_w + 2 * d, arch_h + d), fill=0, stroke=1)
    c.setStrokeAlpha(1)

    first_image = list(images.values())[0]["image"]

    c.saveState()
    c.clipPath(arch_path(ax, ay, arch_w, arch_h), stroke=0, fill=0)
    c.drawImage(fitted(first_image, arch_w, arch_h), ax, ay, arch_w, arch_h)
    c.restoreState()

    c.setStrokeColor(GOLD)
    c.setLineWidth(1.8)
    c.drawPath(arch_path(ax, ay, arch_w, arch_h), fill=0, stroke=1)

    # verified pill (width follows the text)
    pill_text = "إعلان مُراجع بالذكاء الاصطناعي"
    pill_w = max(50 * mm, stringWidth(shape(pill_text), FONT_BOLD, 7.5) + 14 * mm)
    pill_h = 8 * mm
    px = (PAGE_W - pill_w) / 2
    py = ay - pill_h / 2
    c.setFillColor(GOLD)
    c.roundRect(px, py, pill_w, pill_h, pill_h / 2, fill=1, stroke=0)
    spaced(PAGE_W / 2, py + 2.7 * mm, pill_text, 7.5, NIGHT, spacing=0)

    title_lines = wrap(listing_name, FONT_BOLD, 28, PAGE_W - 40 * mm)[:2]
    ty = ay - 20 * mm

    for line in title_lines:
        c.setFont(FONT_BOLD, 28)
        c.setFillColor(CHAMPAGNE)
        c.drawCentredString(PAGE_W / 2, ty, line)
        ty -= 11 * mm

    text(PAGE_W / 2, ty - 1 * mm, location, FONT, 11, MUTED_DARK, align="center")

    c.setStrokeColor(GOLD)
    c.setStrokeAlpha(0.5)
    c.setLineWidth(0.8)
    c.line(PAGE_W / 2 - 14 * mm, ty - 8 * mm, PAGE_W / 2 + 14 * mm, ty - 8 * mm)
    c.setStrokeAlpha(1)

    # stats strip (RTL order: rent on the right, capacity on the left)
    col_w = (PAGE_W - 2 * MARGIN) / 3
    sy = 48 * mm

    for i in range(1, 3):
        c.setStrokeColor(GOLD)
        c.setStrokeAlpha(0.3)
        c.line(MARGIN + col_w * i, sy - 6 * mm, MARGIN + col_w * i, sy + 12 * mm)
    c.setStrokeAlpha(1)

    cx_rent = MARGIN + col_w * 2.5
    cx_rate = MARGIN + col_w * 1.5
    cx_cap = MARGIN + col_w * 0.5

    rent_size = fit_size(rent_text, FONT_BOLD, 13, col_w - 10 * mm, min_size=8)
    text(cx_rent, sy + 5 * mm, rent_text, FONT_BOLD, rent_size, GOLD_LIGHT, align="center")
    text(cx_rent, sy - 3 * mm, "الإيجار الشهري (تقديري)", FONT, 7, MUTED_DARK, align="center")

    star_size = 5 * mm
    star_w = 5 * star_size + 4 * star_size * 0.25
    draw_stars(cx_rate - star_w / 2, sy + 3 * mm, star_size, rating,
               on=GOLD_LIGHT, off=HexColor("#2B3A32"))
    text(cx_rate, sy - 3 * mm, f"{rating:.1f} من 5 - تقييم تقديري", FONT, 7,
         MUTED_DARK, align="center")

    text(cx_cap, sy + 5 * mm, f"{capacity} {'فرد' if capacity == 1 else 'أفراد'}",
         FONT_BOLD, 13, GOLD_LIGHT, align="center")
    text(cx_cap, sy - 3 * mm, "السعة التقديرية", FONT, 7, MUTED_DARK, align="center")

    text(PAGE_W / 2, 18 * mm,
         f"تاريخ إنشاء التقرير: {datetime.now().strftime('%d-%m-%Y')}",
         FONT, 8, MUTED_DARK, align="center")
    text(PAGE_W / 2, 13 * mm,
         "تحليل بمساعدة الذكاء الاصطناعي. جميع القيم تقديرية وتتطلب موافقة المالك.",
         FONT, 7.5, MUTED_DARK, align="center")

    # =====================================================
    # PAGE 2+: DETAILS
    # =====================================================
    c.showPage()
    state["page"] = 2
    body_page()

    # ---------- 01 Gallery ----------
    section(1, "الشقة", need=22 * mm + 42 * mm)

    pics = list(images.items())
    gap = 4 * mm
    pw = (CONTENT_W - 2 * gap) / 3
    ph = 42 * mm

    for start in range(0, len(pics), 3):

        ensure(ph + 14 * mm)

        row = pics[start:start + 3]
        top = state["y"]

        for i, (name, data) in enumerate(row):

            # RTL: first photo on the right
            x = R - (i + 1) * pw - i * gap
            draw_photo(data["image"], x, top - ph, pw, ph)

            room = per_image.get(name, {}).get("room_type", "") or "صورة"
            label_w = min(pw, stringWidth(shape(room), FONT_BOLD, 8) + 8 * mm)

            c.setFillColor(MINT)
            c.setStrokeColor(MINT_LINE)
            c.roundRect(x + (pw - label_w) / 2, top - ph - 8 * mm,
                        label_w, 6 * mm, 3 * mm, fill=1, stroke=1)
            text(x + pw / 2, top - ph - 6.1 * mm, room, FONT_BOLD, 8,
                 EMERALD_900, align="center")

        state["y"] = top - ph - 14 * mm

    state["y"] -= 2 * mm

    # ---------- 02 Description ----------
    description = get_verified_description(final_report)
    lines = wrap(description, FONT, 10.5, CONTENT_W - 16 * mm)
    leading = 5.8 * mm
    box_h = len(lines) * leading + 11 * mm

    section(2, "عن الشقة", need=min(box_h, 60 * mm) + 11 * mm)
    ensure(box_h)

    top = state["y"]
    card(MARGIN, top, CONTENT_W, box_h)

    c.setFillColor(BRASS)
    c.roundRect(R - 1.6 * mm, top - box_h, 1.6 * mm, box_h, 0.8 * mm, fill=1, stroke=0)

    ly = top - 8 * mm

    for line in lines:
        c.setFont(FONT, 10.5)
        c.setFillColor(INK)
        c.drawRightString(R - 8 * mm, ly, line)
        ly -= leading

    state["y"] = top - box_h - 10 * mm

    # ---------- 03 At a glance ----------
    section(3, "نظرة سريعة", need=11 * mm + 34 * mm + 5 * mm + 20 * mm)

    top = state["y"]
    cw = (CONTENT_W - 2 * gap) / 3
    ch = 34 * mm

    # Rent (rightmost)
    x = R - cw
    rx = x + cw - 5 * mm
    card(x, top, cw, ch)
    text(rx, top - 7 * mm, "الإيجار الشهري", FONT_BOLD, 7, MUTED, align="right")

    rent_size2 = fit_size(rent_text, FONT_BOLD, 12.5, cw - 10 * mm, min_size=8)
    text(rx, top - 15 * mm, rent_text, FONT_BOLD, rent_size2, EMERALD_600, align="right")

    bar_x, bar_w, bar_y = x + 5 * mm, cw - 10 * mm, top - 24 * mm

    c.setFillColor(MINT)
    c.roundRect(bar_x, bar_y, bar_w, 2.4 * mm, 1.2 * mm, fill=1, stroke=0)

    if rent_max > 0:
        scale = rent_max * 1.4
        # mirrored so the range grows from the right (RTL)
        hi = bar_x + bar_w * (1 - rent_min / scale)
        lo = bar_x + bar_w * (1 - rent_max / scale)
        c.setFillColor(BRASS)
        c.roundRect(lo, bar_y, max(hi - lo, 2 * mm), 2.4 * mm, 1.2 * mm, fill=1, stroke=0)

    text(rx, top - 30 * mm, "تقدير شهري بالذكاء الاصطناعي", FONT, 7.5, MUTED, align="right")

    # Rating (middle)
    x = R - 2 * cw - gap
    rx = x + cw - 5 * mm
    card(x, top, cw, ch)
    text(rx, top - 7 * mm, "التقييم", FONT_BOLD, 7, MUTED, align="right")

    num = f"{rating:.1f}"
    text(rx, top - 17 * mm, num, FONT_BOLD, 20, EMERALD_600, align="right")
    text(rx - stringWidth(num, FONT_BOLD, 20) - 1.5 * mm,
         top - 17 * mm, "/ 5", FONT, 9, MUTED, align="right")
    draw_stars(rx - (5 * 5 * mm + 4 * 5 * mm * 0.25), top - 26 * mm, 5 * mm, rating)
    text(rx, top - 31.5 * mm, "تقدير اعتمادًا على الصور", FONT, 7.5, MUTED, align="right")

    # Capacity (leftmost)
    x = R - 3 * cw - 2 * gap
    rx = x + cw - 5 * mm
    card(x, top, cw, ch)
    text(rx, top - 7 * mm, "السعة", FONT_BOLD, 7, MUTED, align="right")

    cap_str = str(capacity)
    text(rx, top - 17 * mm, cap_str, FONT_BOLD, 20, EMERALD_600, align="right")
    text(rx - stringWidth(cap_str, FONT_BOLD, 20) - 1.5 * mm,
         top - 17 * mm, "أفراد", FONT, 9, MUTED, align="right")

    shown = min(capacity, 8)

    for i in range(shown):
        ix = rx - 2 * mm - i * 5.6 * mm
        iy = top - 27 * mm
        c.setFillColor(BRASS)
        c.circle(ix, iy + 3.2 * mm, 1.2 * mm, fill=1, stroke=0)
        c.roundRect(ix - 1.8 * mm, iy - 1.6 * mm, 3.6 * mm, 3.8 * mm,
                    1.6 * mm, fill=1, stroke=0)

    if capacity > 8:
        text(rx - 2 * mm - 8 * 5.6 * mm, top - 27 * mm, f"+{capacity - 8}",
             FONT_BOLD, 8, BRASS, align="right")

    state["y"] = top - ch - 5 * mm

    # Capacity fit
    fh = 20 * mm
    ensure(fh)
    top = state["y"]

    card(MARGIN, top, CONTENT_W, fh)
    text(R - 6 * mm, top - 7.5 * mm,
         f"مدى ملاءمة السعة لعدد {people} من الأفراد المطلوبين",
         FONT_BOLD, 7.5, MUTED, align="right")
    text(MARGIN + 6 * mm, top - 9 * mm, f"{fit}%", FONT_BOLD, 16, EMERALD_600)

    track_w = CONTENT_W - 12 * mm
    track_x = MARGIN + 6 * mm
    fill_w = max(track_w * fit / 100, 3 * mm)

    c.setFillColor(MINT)
    c.roundRect(track_x, top - 16 * mm, track_w, 3 * mm, 1.5 * mm, fill=1, stroke=0)
    c.setFillColor(EMERALD_600 if fit >= 100 else BRASS)
    # fills from the right (RTL)
    c.roundRect(track_x + track_w - fill_w, top - 16 * mm, fill_w,
                3 * mm, 1.5 * mm, fill=1, stroke=0)

    state["y"] = top - fh - 10 * mm

    # ---------- 04 What the photos show ----------
    findings = list(images.items())

    if per_image:

        section(4, "ما توضحه الصور", need=11 * mm + 36 * mm)

        thumb_w, thumb_h = 34 * mm, 26 * mm
        inner_w = CONTENT_W - thumb_w - 14 * mm
        text_right = R - 4 * mm - thumb_w - 5 * mm

        for name, data in findings:

            info_i = per_image.get(name, {})
            furniture = info_i.get("furniture", [])
            colors = info_i.get("colors", [])

            h_f = chips(0, 0, inner_w, furniture, MINT, MINT_LINE, EMERALD_900, draw=False)
            h_c = chips(0, 0, inner_w, colors, PAPER, LINE, INK, swatch=True, draw=False)

            body_h = 8 * mm + (h_f + 2 * mm if h_f else 0) + (h_c if h_c else 0)
            card_h = max(thumb_h + 8 * mm, body_h + 8 * mm)

            ensure(card_h + 4 * mm)

            top = state["y"]
            card(MARGIN, top, CONTENT_W, card_h)

            draw_photo(data["image"], R - 4 * mm - thumb_w, top - 4 * mm - thumb_h,
                       thumb_w, thumb_h, radius=2 * mm)

            room = info_i.get("room_type", "") or "صورة"
            conf = info_i.get("confidence_level", "")

            text(text_right, top - 8 * mm, room, FONT_BOLD, 11, EMERALD_900, align="right")

            if conf:
                label = f"مستوى الثقة: {conf}"
                lw = stringWidth(shape(label), FONT, 7.5) + 5 * mm
                lx = MARGIN + 5 * mm
                c.setFillColor(MINT)
                c.setStrokeColor(MINT_LINE)
                c.roundRect(lx, top - 9.6 * mm, lw, 5 * mm, 2.5 * mm, fill=1, stroke=1)
                text(lx + lw / 2, top - 8.2 * mm, label, FONT, 7.5, EMERALD_900,
                     align="center")

            cy = top - 12 * mm

            if h_f:
                chips(text_right, cy, inner_w, furniture, MINT, MINT_LINE, EMERALD_900)
                cy -= h_f + 2 * mm

            if h_c:
                chips(text_right, cy, inner_w, colors, PAPER, LINE, INK, swatch=True)

            state["y"] = top - card_h - 4 * mm

        state["y"] -= 6 * mm

    # ---------- 05 Highlights ----------
    highlights = est.get("highlights", [])

    if highlights:

        section(5, "أبرز المميزات", need=11 * mm + 16 * mm)

        for index, item in enumerate(highlights, start=1):

            lines = wrap(item, FONT, 10.5, CONTENT_W - 16 * mm)
            row_h = max(8 * mm, len(lines) * 5.4 * mm + 2 * mm)

            ensure(row_h)

            top = state["y"]

            c.setFillColor(EMERALD_900)
            c.circle(R - 3.5 * mm, top - 2.2 * mm, 3.4 * mm, fill=1, stroke=0)
            text(R - 3.5 * mm, top - 3.6 * mm, str(index), FONT_BOLD, 8.5,
                 GOLD_LIGHT, align="center")

            ly = top - 3 * mm

            for line in lines:
                c.setFont(FONT, 10.5)
                c.setFillColor(INK)
                c.drawRightString(R - 11 * mm, ly, line)
                ly -= 5.4 * mm

            state["y"] = top - row_h - 1.5 * mm

        state["y"] -= 6 * mm

    # ---------- 06 How the AI estimated ----------
    notes = [
        ("التقييم", est.get("rating_reason", "")),
        ("الإيجار", est.get("rent_reason", "")),
        ("السعة", est.get("capacity_reason", "")),
    ]
    notes = [(a, b) for a, b in notes if b]

    if notes:

        section(6, "كيف قدّر الذكاء الاصطناعي هذه البيانات؟", need=11 * mm + 18 * mm)

        for label, reason in notes:

            lines = wrap(reason, FONT, 10, CONTENT_W - 30 * mm)
            row_h = max(8 * mm, len(lines) * 5.2 * mm + 3 * mm)

            ensure(row_h)

            top = state["y"]

            text(R, top - 3 * mm, label, FONT_BOLD, 8, BRASS, align="right")

            ly = top - 3 * mm

            for line in lines:
                c.setFont(FONT, 10)
                c.setFillColor(INK)
                c.drawRightString(R - 28 * mm, ly, line)
                ly -= 5.2 * mm

            c.setStrokeColor(LINE)
            c.setLineWidth(0.5)
            c.line(MARGIN, top - row_h + 1 * mm, PAGE_W - MARGIN, top - row_h + 1 * mm)

            state["y"] = top - row_h - 2 * mm

        # confidence meter
        confidence = (est.get("confidence") or "").strip()
        level = {"منخفض": 1, "متوسط": 2, "مرتفع": 3,
                 "low": 1, "medium": 2, "high": 3}.get(confidence.lower(), 0)

        if level:

            ensure(16 * mm)
            top = state["y"] - 4 * mm

            text(R, top - 3 * mm, "مستوى الثقة", FONT_BOLD, 8, BRASS, align="right")

            bar_w2, bar_gap = 20 * mm, 2 * mm
            first_x = R - 28 * mm          # right edge of the first bar

            for i in range(3):
                bx = first_x - (i + 1) * bar_w2 - i * bar_gap
                c.setFillColor(EMERALD_600 if i < level else MINT)
                c.roundRect(bx, top - 4.2 * mm, bar_w2, 3 * mm, 1.5 * mm, fill=1, stroke=0)

            text(first_x - 3 * bar_w2 - 2 * bar_gap - 2 * mm, top - 3.6 * mm,
                 confidence, FONT_BOLD, 9, EMERALD_900, align="right")

            state["y"] = top - 12 * mm

    # uncertainty + disclaimer
    uncertainty = est.get("uncertainty", [])

    note_lines = []

    if uncertainty:
        note_lines.append("ما لا يمكن تأكيده من الصور: " + "؛ ".join(uncertainty) + ".")

    note_lines.append(
        "هذه القيم تقديرات آلية مبنية على الصور المرفوعة فقط. "
        "يجب على المالك مراجعة كل التفاصيل والموافقة عليها قبل النشر."
    )

    wrapped = []

    for paragraph in note_lines:
        wrapped.extend(wrap(paragraph, FONT, 9, CONTENT_W - 14 * mm))
        wrapped.append("")

    wrapped = wrapped[:-1]

    box_h = len(wrapped) * 4.8 * mm + 9 * mm

    ensure(box_h + 4 * mm)

    top = state["y"] - 4 * mm
    card(MARGIN, top, CONTENT_W, box_h, fill=MINT, border=MINT_LINE)

    ly = top - 7 * mm

    for line in wrapped:
        c.setFont(FONT, 9)
        c.setFillColor(EMERALD_900)
        c.drawRightString(R - 7 * mm, ly, line)
        ly -= 4.8 * mm

    c.showPage()
    c.save()

    return buffer.getvalue()



# =====================================================
# 8. PROPERTY CARD (on screen)
# =====================================================

def display_property_card(
    images,
    listing_name,
    location,
    rent,
    rating,
    occupants,
    max_occupants
):

    st.markdown(
        '<div class="section-title">معاينة الإعلان</div>',
        unsafe_allow_html=True
    )

    image_values = list(images.values())
    n_images = len(image_values)

    for row_start in range(0, n_images, 3):

        cols = st.columns(3)

        for col, item in zip(cols, image_values[row_start:row_start + 3]):
            with col:
                st.image(make_thumb(item["image"]), use_container_width=True)

    st.markdown(
        f"""
        <div class="property-card">
            <div class="property-title">{safe_text(listing_name)}</div>
            <div class="property-location">📍 {safe_text(location)}</div>
            <div style="margin-bottom:12px;">
                <span class="tag">{n_images} صورة للشقة</span>
                <span class="tag">تحليل الصور بالذكاء الاصطناعي</span>
                <span class="tag">يتطلب موافقة المالك</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    stars_percent = max(0, min(100, rating / 5 * 100))

    with col1:
        st.markdown(
            f"""
            <div class="stars" style="--pct: {stars_percent:.0f}%;">★★★★★</div>
            <div class="rating-caption">{rating:.1f} من 5 (تقدير الذكاء الاصطناعي)</div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="price">{safe_text(rent)}</div>
            <div class="price-caption">الإيجار الشهري (تقدير الذكاء الاصطناعي)</div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        suitability = min(100, round((max_occupants / max(occupants, 1)) * 100)) if max_occupants > 0 else 0

        st.markdown(
            f"""
            <div class="pdi-box">
                <div class="pdi-number">{suitability}%</div>
                <div class="pdi-label">مدى ملاءمة السعة</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    st.markdown(
        '<div class="section-title">معلومات سريعة</div>',
        unsafe_allow_html=True
    )

    info_cols = st.columns(4)

    info_items = [
        (str(n_images), "الصور المرفوعة"),
        (str(occupants), "العدد المطلوب للأفراد"),
        (str(max_occupants), "السعة التقديرية"),
        ("ذكي", "تحليل الصور")
    ]

    for col, (number, label) in zip(info_cols, info_items):
        with col:
            st.markdown(
                f"""
                <div class="info-box">
                    <div class="info-number">{safe_text(number)}</div>
                    <div class="info-label">{safe_text(label)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


def display_full_details(final_report):

    st.markdown(
        '<div class="section-title">تفاصيل الشقة</div>',
        unsafe_allow_html=True
    )

    description = get_verified_description(final_report)
    estimate = final_report.get("listing_estimate", {})

    with st.expander("تفاصيل الشقة والوصف الكامل", expanded=False):

        st.markdown(
            '<div class="details-heading">الوصف الذي راجعه الذكاء الاصطناعي</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="description-box">{safe_text(description)}</div>',
            unsafe_allow_html=True
        )

        notes = [
            ("التقييم", estimate.get("rating_reason", "")),
            ("الإيجار", estimate.get("rent_reason", "")),
            ("السعة", estimate.get("capacity_reason", "")),
        ]

        notes = [(a, b) for a, b in notes if b]

        if notes:

            st.write("")

            st.markdown(
                '<div class="details-heading">كيف قدّر الذكاء الاصطناعي هذه البيانات؟</div>',
                unsafe_allow_html=True
            )

            for label, text_value in notes:
                st.markdown(f"**{label}:** {safe_text(text_value)}")

            confidence = estimate.get("confidence", "")

            if confidence:
                st.caption(
                    f"مستوى الثقة: {safe_text(confidence)}. "
                    "هذه تقديرات مبنية على الصور فقط."
                )


# =====================================================
# 9. MAIN PAGE
# =====================================================

st.markdown(
    '<div class="hero">'
    '<div class="hero-text">'
    '<h1>حُجرة</h1>'
    '<p>ارفع صور شقتك واحصل على إعلان سكن موصوف ومُراجع وجاهز لمراجعتك.</p>'
    '</div>'
    '<div class="arch" aria-hidden="true"><span>حجرة</span></div>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">أنشئ إعلان شقتك</div>',
    unsafe_allow_html=True
)

st.write(
    "ارفع صور شقتك ودع الذكاء الاصطناعي يحلل الأثاث والألوان والتفاصيل الظاهرة."
)

with st.expander("بيانات الإعلان", expanded=True):

    col1, col2 = st.columns(2)

    with col1:
        location = st.text_input("عنوان الشقة", value="الجيزة، مصر")

    with col2:
        occupants = st.number_input(
            "العدد المطلوب للأفراد",
            min_value=1,
            max_value=20,
            value=2
        )

    st.caption(
        "يقترح الذكاء الاصطناعي اسم الإعلان والتقييم والإيجار والسعة التقديرية بناءً على الصور."
    )

uploaded_files = st.file_uploader(
    f"ارفع من {MIN_IMAGES} إلى {MAX_IMAGES} صور للشقة",
    type=["jpg", "jpeg", "png", "webp", "heic", "heif"],
    accept_multiple_files=True
)

images = {}

if uploaded_files:

    if len(uploaded_files) > MAX_IMAGES:

        st.warning(
            f"الحد الأقصى {MAX_IMAGES} صور. لقد رفعت {len(uploaded_files)} صور."
        )

    else:

        failed = []

        for index, uploaded_file in enumerate(uploaded_files):
            try:
                images[f"image_{index + 1}"] = {
                    "image": prepare_image(uploaded_file),
                    "filename": uploaded_file.name
                }
            except Exception:
                failed.append(uploaded_file.name)

        if failed:
            st.warning("تعذر قراءة الصور التالية: " + ", ".join(failed))

        if images:

            st.success(f"تم رفع {len(images)} صورة بنجاح.")

            items = list(images.items())

            for row_start in range(0, len(items), 3):

                cols = st.columns(3)

                for col, (name, data) in zip(cols, items[row_start:row_start + 3]):
                    with col:
                        st.image(
                            make_thumb(data["image"]),
                            caption=data["filename"],
                            use_container_width=True
                        )


# =====================================================
# 10. START ANALYSIS (one calm loader, fast single call)
# =====================================================

if images:

    st.write("")

    if st.button("ابدأ تحليل الذكاء الاصطناعي", type="primary", use_container_width=True):

        loader = st.empty()
        loader.markdown(LOADER_HTML, unsafe_allow_html=True)

        try:

            # Optional Florence hints (only when it is fast: GPU)
            florence_results = []
            used_florence = False

            if florence_enabled():

                try:
                    processor, model, device, dtype = load_florence()

                    for image_name, data in images.items():
                        florence_results.append({
                            "image_name": image_name,
                            "description": str(
                                analyze_with_florence(
                                    data["image"], processor, model, device, dtype
                                )
                            )
                        })

                    used_florence = True

                except Exception:
                    florence_results = []

            client = create_gemini_client()

            listing_estimate = run_listing_analysis(
                client,
                images,
                location,
                florence_results
            )

            final_report = {
                "project": "Hujrah AI Apartment Analyzer",
                "model": "Florence-2-large + Gemini vision" if used_florence else "Gemini vision",
                "language_model": GEMINI_MODEL,
                "number_of_images": len(images),
                "listing_information": {
                    "name": listing_estimate["listing_name"],
                    "location": location,
                    "monthly_rent": listing_estimate["estimated_rent_text"],
                    "demo_rating": listing_estimate["estimated_rating"],
                    "required_people": occupants,
                    "test_capacity": listing_estimate["estimated_capacity"]
                },
                "listing_estimate": listing_estimate,
                "florence_analysis": florence_results,
                "final_description": listing_estimate["verified_description"],
                "owner_approval_required": True,
                "values_are_ai_estimates": True
            }

            save_json("Hujrah_Final_Report.json", final_report)

            pdf_bytes = None
            pdf_error = None

            try:
                pdf_bytes = build_report_pdf(final_report, images)
            except Exception as pdf_exc:
                pdf_error = f"{type(pdf_exc).__name__}: {pdf_exc}"

            st.session_state["final_report"] = final_report
            st.session_state["analysis_images"] = images
            st.session_state["pdf_bytes"] = pdf_bytes
            st.session_state["pdf_error"] = pdf_error
            st.session_state["autodl_done"] = False

            loader.empty()

        except Exception as exc:

            loader.empty()

            if is_network_error(exc):
                st.error(
                    "تعذر الاتصال بخدمة Gemini. تحقق من اتصال الإنترنت "
                    "ثم اضغط على زر ابدأ تحليل الذكاء الاصطناعي مرة أخرى."
                )
            else:
                st.error(f"فشل التحليل: {exc}")
                st.exception(exc)


# =====================================================
# 11. RESULTS
# =====================================================

if "final_report" in st.session_state:

    final_report = st.session_state["final_report"]
    report_images = st.session_state["analysis_images"]
    listing_info = final_report.get("listing_information", {})

    st.divider()

    st.markdown(
        '<div class="section-title">إعلان شقتك</div>',
        unsafe_allow_html=True
    )

    pdf_bytes = st.session_state.get("pdf_bytes")
    pdf_error = st.session_state.get("pdf_error")

    if pdf_bytes:

        st.download_button(
            "تحميل تقرير الإعلان (PDF)",
            data=pdf_bytes,
            file_name="Hujrah_Listing_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        if not st.session_state.get("autodl_done"):
            auto_download(pdf_bytes, "Hujrah_Listing_Report.pdf")
            st.session_state["autodl_done"] = True

    elif pdf_error:

        st.warning(
            f"تعذر إنشاء ملف PDF ({pdf_error}). "
            f"إذا كانت هناك مكتبة ناقصة، ثبّتها في بيئة Python المستخدمة "
            f"لتشغيل التطبيق:\n\n"
            f"{sys.executable} -m pip install reportlab pillow-heif "
            f"arabic-reshaper python-bidi fonttools"
        )

    display_property_card(
        images=report_images,
        listing_name=listing_info.get("name", "إعلان شقة"),
        location=listing_info.get("location", "لم يتم تحديد الموقع"),
        rent=listing_info.get("monthly_rent", "غير متاح"),
        rating=float(listing_info.get("demo_rating", 0)),
        occupants=int(listing_info.get("required_people", 1)),
        max_occupants=int(listing_info.get("test_capacity", 1))
    )

    st.write("")

    display_full_details(final_report)

    st.caption(
        "حُجرة. تحليل العقارات بمساعدة الذكاء الاصطناعي. "
        "جميع الإعلانات تتطلب موافقة المالك."
    )

else:

    st.info(
        "ارفع صور الشقة وابدأ التحليل لإنشاء إعلانك."
    )
