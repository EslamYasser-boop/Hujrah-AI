import os
import sys
import json
import importlib.util
import html
import streamlit as st
import torch

from pathlib import Path
from PIL import Image
from transformers import CLIPImageProcessor, BartTokenizerFast
from dotenv import load_dotenv
from google import genai
from google.genai import types


# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Florence-2-large"
RESULTS_DIR = BASE_DIR / "Results"

RESULTS_DIR.mkdir(exist_ok=True)

load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

st.set_page_config(
    page_title="Hujrah AI | Property Analyzer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =====================================================
# HTML FIX
# Streamlit's markdown treats lines indented by 4+ spaces (after a
# blank line) as a code block, which shows raw HTML on the page.
# This flattens every HTML snippet (except <style>) to avoid that.
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
# 2. DESIGN SYSTEM  (Emerald & Ivory)
# =====================================================
#
#   Ivory       #F7F2E4   page background
#   Paper       #FFFDF6   cards
#   Emerald 900 #0B3D2E   hero, headings, primary button
#   Emerald 600 #11694F   prices, numbers, links
#   Mint        #E2EEE3   tags, soft panels
#   Brass       #B58A3A   stars and the arch outline
#   Ink         #17261F   body text
#   Muted       #66776D   captions
#   Line        #E3DBC4   borders
#
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

/* ---------- Base + living background ---------- */
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

/* A slow band of light crossing the hero */
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

/* The arch: a doorway of light (hujrah = room) */
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
    animation: glow 5s ease-in-out infinite;
}

@keyframes glow {
    0%, 100% { box-shadow: 0 0 22px rgba(201, 162, 77, 0.18); }
    50%      { box-shadow: 0 0 52px rgba(201, 162, 77, 0.42); }
}

.arch span {
    font-size: 46px;
    font-weight: 500;
    line-height: 1;
    color: var(--gold-light);
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
.property-card {
    margin-top: 18px;
    padding: 28px 30px 16px 30px;
    background: linear-gradient(160deg, var(--forest), var(--night));
    border: 1px solid var(--line);
    border-radius: 20px;
    transition: border-color 0.25s ease, transform 0.25s ease;
}

.property-card:hover {
    border-color: rgba(235, 210, 144, 0.55);
    transform: translateY(-2px);
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

/* ---------- Tags ---------- */
.tag {
    display: inline-block;
    margin: 3px 6px 3px 0;
    padding: 6px 14px;
    border-radius: 999px;
    border: 1px solid var(--line);
    background: rgba(201, 162, 77, 0.08);
    color: var(--gold-light);
    font-size: 13px;
    font-weight: 400;
}

/* ---------- Rating ---------- */
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
}

/* ---------- PDI + info ---------- */
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

/* ---------- Details ---------- */
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

/* ---------- Buttons: gold, with a light sheen ---------- */
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
    letter-spacing: 0.3px;
    box-shadow: 0 8px 26px rgba(201, 162, 77, 0.18);
    transition: transform 0.2s ease, box-shadow 0.2s ease, background-position 0.5s ease;
}

.stButton > button::after,
.stDownloadButton > button::after {
    content: "";
    position: absolute;
    top: 0;
    bottom: 0;
    left: -70%;
    width: 45%;
    background: linear-gradient(100deg, transparent, rgba(255, 255, 255, 0.55), transparent);
    transform: skewX(-20deg);
    transition: left 0.6s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-3px);
    background-position: 100% 0;
    border-color: #F3DEA0;
    color: #1A1405;
    box-shadow: 0 14px 38px rgba(201, 162, 77, 0.38);
}

.stButton > button:hover::after,
.stDownloadButton > button:hover::after {
    left: 130%;
}

.stButton > button:active,
.stDownloadButton > button:active {
    transform: translateY(0) scale(0.985);
}

.stButton > button p,
.stDownloadButton > button p {
    color: inherit;
}

.stButton > button:focus-visible,
.stDownloadButton > button:focus-visible {
    outline: 3px solid var(--glow);
    outline-offset: 3px;
}

/* ---------- Inputs ---------- */
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 10px;
}

div[data-baseweb="input"] > div {
    border: 1px solid var(--line);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

div[data-baseweb="input"] > div:focus-within {
    border-color: var(--gold-light);
    box-shadow: 0 0 0 1px var(--gold-light), 0 0 18px rgba(201, 162, 77, 0.22);
}

/* ---------- Expanders, metrics, alerts ---------- */
div[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid var(--line);
    border-radius: 16px;
}

div[data-testid="stExpander"] summary {
    font-weight: 500;
    color: var(--champagne);
}

div[data-testid="stAlert"] {
    border-radius: 14px;
    border: 1px solid var(--line);
}

/* ---------- Upload area ---------- */
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
    transition: background 0.2s ease, color 0.2s ease;
}

div[data-testid="stFileUploader"] section button:hover {
    background: var(--gold);
    color: #1A1405;
}

/* ---------- Images: small, uniform thumbnails ---------- */
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

div[data-testid="stImage"] {
    overflow: visible;
}

hr {
    border-color: var(--line);
}

/* ---------- Mobile ---------- */
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

/* ---------- Visible motion ---------- */

/* Every block of the page rises in, one after another */
.block-container .element-container {
    animation: rise 0.8s cubic-bezier(.2, .8, .2, 1) both;
}
.block-container .element-container:nth-child(2) { animation-delay: 0.10s; }
.block-container .element-container:nth-child(3) { animation-delay: 0.20s; }
.block-container .element-container:nth-child(4) { animation-delay: 0.30s; }
.block-container .element-container:nth-child(5) { animation-delay: 0.40s; }
.block-container .element-container:nth-child(6) { animation-delay: 0.50s; }

/* The arch floats and glows */
.arch {
    animation: glow 4s ease-in-out infinite, float 6s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-12px); }
}

/* Main button breathes with a gold ring */
.stButton > button {
    animation: ring 2.6s ease-out infinite;
}

@keyframes ring {
    0%   { box-shadow: 0 0 0 0 rgba(235, 210, 144, 0.50), 0 8px 26px rgba(201, 162, 77, 0.18); }
    70%  { box-shadow: 0 0 0 16px rgba(235, 210, 144, 0), 0 8px 26px rgba(201, 162, 77, 0.18); }
    100% { box-shadow: 0 0 0 0 rgba(235, 210, 144, 0), 0 8px 26px rgba(201, 162, 77, 0.18); }
}

.stButton > button:hover {
    animation: none;
}

/* Listing card: a ray of gold runs around its border */
@property --ang {
    syntax: '<angle>';
    initial-value: 0deg;
    inherits: false;
}

.property-card {
    border: 1px solid transparent;
    background:
        linear-gradient(160deg, #12241C, #0C1813) padding-box,
        conic-gradient(from var(--ang), rgba(201, 162, 77, 0.15) 55%, #F3DEA0 78%, rgba(201, 162, 77, 0.15) 100%) border-box;
    animation: spin 6s linear infinite;
}

@keyframes spin {
    to { --ang: 360deg; }
}

/* Stars twinkle */
.stars {
    animation: twinkle 3s ease-in-out infinite;
}

@keyframes twinkle {
    50% { filter: brightness(1.4) drop-shadow(0 0 8px rgba(235, 210, 144, 0.65)); }
}

/* Upload area pulses softly while waiting */
div[data-testid="stFileUploader"] section {
    animation: wait 3.2s ease-in-out infinite;
}

@keyframes wait {
    50% { border-color: var(--gold-light); background: rgba(201, 162, 77, 0.09); }
}

/* ---------- Respect reduced motion ---------- */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation: none !important;
        transition: none !important;
    }
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. HELPER FUNCTIONS
# =====================================================

def save_json(filename, content):

    path = RESULTS_DIR / filename

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            content,
            file,
            ensure_ascii=False,
            indent=4
        )

    return path


def extract_json(response):

    response_text = response.text

    if not response_text:
        raise ValueError("Gemini returned an empty response.")

    text = response_text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")

    text = text.strip()

    try:
        return json.loads(text, strict=False)

    except json.JSONDecodeError:

        # Fallback: keep only the outermost JSON object
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end <= start:
            raise

        return json.loads(text[start:end + 1], strict=False)


def generate_json(client, contents, attempts=3):

    last_error = None

    for _ in range(attempts):

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.2,
                "max_output_tokens": 16384
            }
        )

        try:
            return extract_json(response)

        except (json.JSONDecodeError, ValueError) as error:
            # Gemini sometimes returns broken JSON: ask again
            last_error = error

    raise ValueError(
        f"Gemini returned invalid JSON {attempts} times: {last_error}"
    )


def prepare_image(uploaded_file):

    return Image.open(uploaded_file).convert("RGB")


def safe_text(value):

    return html.escape(str(value))


def make_thumb(image, size=(480, 360)):

    from PIL import ImageOps

    # Crop to a uniform 4:3 shape and shrink: small, tidy thumbnails
    return ImageOps.fit(image, size, method=Image.LANCZOS)


def get_verified_description(report):

    description = report.get("final_description", "")

    if not description:
        description = (
            "No verified description was generated."
        )

    return str(description)


# =====================================================
# 4. LOAD FLORENCE MODEL LOCALLY
# =====================================================

@st.cache_resource(show_spinner="Loading Florence-2-large...")

def load_florence():

    if not (MODEL_DIR / "config.json").exists():

        raise FileNotFoundError(
            f"Florence model folder not found: {MODEL_DIR}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    spec = importlib.util.spec_from_file_location(
        "florence_pkg",
        str(MODEL_DIR / "__init__.py"),
        submodule_search_locations=[str(MODEL_DIR)]
    )

    if spec is None or spec.loader is None:

        raise RuntimeError(
            "Could not load Florence local package."
        )

    pkg = importlib.util.module_from_spec(spec)

    sys.modules["florence_pkg"] = pkg

    spec.loader.exec_module(pkg)

    from florence_pkg.configuration_florence2 import Florence2Config

    from florence_pkg.modeling_florence2 import (
        Florence2ForConditionalGeneration
    )

    from florence_pkg.processing_florence2 import Florence2Processor

    config = Florence2Config.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True
    )

    image_processor = CLIPImageProcessor.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True
    )

    tokenizer = BartTokenizerFast.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True
    )

    processor = Florence2Processor(
        image_processor=image_processor,
        tokenizer=tokenizer
    )

    model = Florence2ForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        config=config,
        local_files_only=True,
        torch_dtype=torch.float32
    ).to(device)

    model.eval()

    return processor, model, device


# =====================================================
# 5. FLORENCE IMAGE ANALYSIS
# =====================================================

def analyze_with_florence(
    image,
    processor,
    model,
    device
):

    prompt = "<MORE_DETAILED_CAPTION>"

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        if hasattr(value, "to")
        else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=512,
            num_beams=3
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
# 6. GEMINI CLIENT
# =====================================================

def create_gemini_client():

    if not GEMINI_API_KEY:

        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to your .env file."
        )

    return genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(
            api_version="v1beta"
        )
    )


# =====================================================
# 7. STRUCTURED ANALYSIS
# =====================================================

def run_structured_analysis(
    client,
    florence_results
):

    prompt = f"""
You are a professional real estate visual analysis assistant.

Analyze the following image descriptions generated by Florence-2.

IMPORTANT:
- Do not invent objects or facilities.
- Use only information supported by the descriptions.
- Clearly identify uncertainty.
- Do not claim apartment size, rent, location, or quality.
- Do not assume different images represent different rooms.
- Keep each image analysis separate.

Return valid JSON only. Do not use double quotes inside text values (use single quotes). Keep every text value under 25 words.

Required structure:

{{
  "images": [
    {{
      "image_name": "",
      "room_identification": {{
        "predicted_room_type": "",
        "confidence_level": ""
      }},
      "furniture": [],
      "colors": [],
      "decorations": [],
      "visual_details": [],
      "uncertainty": []
    }}
  ],
  "final_description": ""
}}

Florence results:

{json.dumps(florence_results, ensure_ascii=False, indent=2)}
"""

    return generate_json(client, prompt)


# =====================================================
# 8. SEMANTIC COMPARISON
# =====================================================

def run_semantic_comparison(
    client,
    florence_results,
    structured_result
):

    prompt = f"""
You are an independent semantic comparison assistant.

Compare the original Florence-2 descriptions
with Gemini's structured analysis.

Do not assume that Gemini is always correct.

For every claim, determine whether it is:

- AGREEMENT
- PARTIAL_AGREEMENT
- NOT_MENTIONED
- CONTRADICTION

Return valid JSON only. Do not use double quotes inside text values (use single quotes). Keep every text value under 25 words.

Required structure:

{{
  "images": [
    {{
      "image_name": "",
      "comparisons": [
        {{
          "category": "",
          "claim": "",
          "florence_evidence": "",
          "status": "",
          "reason": ""
        }}
      ]
    }}
  ],
  "summary": ""
}}

Florence results:

{json.dumps(florence_results, ensure_ascii=False, indent=2)}

Gemini structured analysis:

{json.dumps(structured_result, ensure_ascii=False, indent=2)}
"""

    return generate_json(client, prompt)


# =====================================================
# 9. VISUAL VERIFICATION
# =====================================================

def run_visual_verification(
    client,
    images,
    structured_result,
    semantic_comparison
):

    verification_prompt = """
IMPORTANT INSTRUCTIONS:

1. Inspect the original images independently.
2. Do not blindly trust Florence or previous Gemini outputs.
3. Verify each claim against the corresponding original image.
4. Do not assume an object exists if it is not visible.
5. NOT_VISIBLE does not mean the object does not exist.
6. Do not infer room type unless visual evidence is sufficient.
7. Do not invent furniture, colors, decorations, or facilities.
8. Do not claim property size, location, rent, cleanliness,
   natural lighting, or quality without direct evidence.
9. Provide specific visual evidence for supported claims.
10. Avoid estimating room dimensions or number of occupants.
11. Do not describe the apartment as clean or well-maintained.
12. Do not treat phrases such as "no uncertainty" as visual claims.

Allowed statuses:
- VISUALLY_SUPPORTED
- UNCERTAIN
- NOT_VISIBLE
- CONTRADICTED

Return valid JSON only. Do not use double quotes inside text values (use single quotes). Keep every text value under 25 words.

Required structure:

{
  "images": [
    {
      "image_name": "",
      "verified_claims": [
        {
          "category": "",
          "claim": "",
          "status": "",
          "visual_evidence": "",
          "reason": ""
        }
      ]
    }
  ],
  "summary": {
    "visually_supported_count": 0,
    "uncertain_count": 0,
    "not_visible_count": 0,
    "contradicted_count": 0
  },
  "verified_description": ""
}

The verified_description must:
- Be professional English real estate text.
- Include only claims marked VISUALLY_SUPPORTED.
- Avoid unsupported marketing language.
- Avoid treating NOT_VISIBLE as proof of absence.
- Avoid assuming all photos show separate rooms.
- Describe rugs as covering only the visible portion of the floor
  when appropriate, not the entire floor.
"""

    verification_contents = [
        verification_prompt,

        "Previous structured analysis:",

        json.dumps(
            structured_result,
            ensure_ascii=False,
            indent=2
        ),

        "Previous semantic comparison:",

        json.dumps(
            semantic_comparison,
            ensure_ascii=False,
            indent=2
        )
    ]

    for image_name, data in images.items():

        verification_contents.append(
            f"Original apartment photo: {image_name}"
        )

        verification_contents.append(
            data["image"]
        )

    return generate_json(client, verification_contents)


# =====================================================
# 9B. LISTING ESTIMATION (done by Gemini from the photos)
# =====================================================

def run_listing_estimation(
    client,
    images,
    verification_result,
    location
):

    prompt = f"""
You are a real estate listing assistant for a student housing
platform in Egypt.

Using ONLY the apartment photos and the verified description below,
create the listing details that the owner would normally type by hand.

Apartment address (provided by the owner): {location}

Verified description:
{verification_result.get("verified_description", "")}

Rules:
- Return valid JSON only. Do not use double quotes inside text values (use single quotes). Keep every text value under 25 words.
- listing_name: a short, natural English title (max 7 words).
  Do not invent a location or facilities that are not visible.
- estimated_rating: a number from 0 to 5 in steps of 0.5, based only
  on what the photos show (furnishing, finishing, variety of rooms).
  Do not mention cleanliness. This is an AI estimate, not a review.
- estimated_rent_min_egp / estimated_rent_max_egp: a realistic
  monthly rent range in EGP for a student rental in that area of
  Egypt, based on the visible finishing and furnishing.
- estimated_capacity: how many people the visible sleeping spaces
  can reasonably host (whole number, at least 1).
- Each *_reason must be one short sentence in English.
- confidence: "low", "medium" or "high".

Required structure:

{{
  "listing_name": "",
  "estimated_rating": 0.0,
  "rating_reason": "",
  "estimated_rent_min_egp": 0,
  "estimated_rent_max_egp": 0,
  "rent_reason": "",
  "estimated_capacity": 1,
  "capacity_reason": "",
  "confidence": ""
}}
"""

    contents = [prompt]

    for image_name, data in images.items():

        contents.append(f"Apartment photo: {image_name}")
        contents.append(data["image"])

    raw = generate_json(client, contents)

    # Clean and validate the AI values
    try:
        rating_value = float(raw.get("estimated_rating", 0))
    except (TypeError, ValueError):
        rating_value = 0.0

    rating_value = max(0.0, min(5.0, round(rating_value * 2) / 2))

    try:
        rent_min = int(raw.get("estimated_rent_min_egp", 0))
        rent_max = int(raw.get("estimated_rent_max_egp", 0))
    except (TypeError, ValueError):
        rent_min, rent_max = 0, 0

    if rent_min > rent_max:
        rent_min, rent_max = rent_max, rent_min

    try:
        capacity_value = max(1, int(raw.get("estimated_capacity", 1)))
    except (TypeError, ValueError):
        capacity_value = 1

    if rent_max > 0 and rent_min != rent_max:
        rent_text = f"EGP {rent_min:,} to {rent_max:,}"
    elif rent_max > 0:
        rent_text = f"EGP {rent_max:,}"
    else:
        rent_text = "Not estimated"

    raw["estimated_rating"] = rating_value
    raw["estimated_capacity"] = capacity_value
    raw["estimated_rent_text"] = rent_text

    return raw


# =====================================================
# 10. PROPERTY CARD
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
        '<div class="section-title">Property Preview</div>',
        unsafe_allow_html=True
    )

    # Three small, equal photos
    image_values = list(images.values())

    gallery_cols = st.columns(3)

    for col, item in zip(gallery_cols, image_values[:3]):

        with col:

            st.image(
                make_thumb(item["image"]),
                use_container_width=True
            )

    st.markdown(
        f"""
        <div class="property-card">

            <div class="property-title">
                {safe_text(listing_name)}
            </div>

            <div class="property-location">
                📍 {safe_text(location)}
            </div>

            <div style="margin-bottom:12px;">
                <span class="tag">3 apartment photos</span>
                <span class="tag">AI image analysis</span>
                <span class="tag">Owner approval required</span>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # Rating, price, and occupancy fit
    col1, col2, col3 = st.columns(3)

    stars_percent = max(0, min(100, rating / 5 * 100))

    with col1:

        st.markdown(
            f"""
            <div class="stars" style="--pct: {stars_percent:.0f}%;">
                ★★★★★
            </div>
            <div class="rating-caption">
                {rating:.1f} out of 5 (AI estimate)
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="price">
                {safe_text(rent)}
            </div>
            <div class="price-caption">
                Monthly rent (AI estimate)
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        suitability = min(
            100,
            round((max_occupants / max(occupants, 1)) * 100)
        )

        st.markdown(
            f"""
            <div class="pdi-box">
                <div class="pdi-number">
                    {suitability}%
                </div>
                <div class="pdi-label">
                    Occupancy fit
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # Quick information
    st.markdown(
        '<div class="section-title">Quick Information</div>',
        unsafe_allow_html=True
    )

    info_cols = st.columns(4)

    info_items = [
        ("3", "Uploaded photos"),
        (str(occupants), "Required people"),
        (str(max_occupants), "Estimated capacity"),
        ("AI", "Visual analysis")
    ]

    for col, (number, label) in zip(info_cols, info_items):

        with col:

            st.markdown(
                f"""
                <div class="info-box">
                    <div class="info-number">
                        {safe_text(number)}
                    </div>
                    <div class="info-label">
                        {safe_text(label)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# =====================================================
# 10B. PDF REPORT
# =====================================================

def build_report_pdf(final_report, images):
    """Build a designed PDF listing report and return its bytes."""

    from io import BytesIO
    from datetime import datetime
    from xml.sax.saxutils import escape

    from PIL import ImageOps
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (
        BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
        Table, TableStyle, Image as RLImage, Flowable, KeepTogether,
        CondPageBreak
    )

    # ---------- Palette (same as the app) ----------
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

    PAGE_W, PAGE_H = A4
    MARGIN = 18 * mm
    CONTENT_W = PAGE_W - 2 * MARGIN
    HEADER_H = 40 * mm

    def clean(value):
        # Built-in PDF fonts support Latin text only
        text = str(value if value is not None else "")
        text = text.encode("cp1252", "ignore").decode("cp1252")
        return escape(text)

    # ---------- Styles ----------
    title_style = ParagraphStyle(
        "title", fontName="Helvetica-Bold", fontSize=24,
        leading=28, textColor=EMERALD_900
    )
    location_style = ParagraphStyle(
        "location", fontName="Helvetica", fontSize=11,
        leading=15, textColor=MUTED
    )
    section_style = ParagraphStyle(
        "section", fontName="Helvetica-Bold", fontSize=14,
        leading=18, textColor=EMERALD_900,
        spaceAfter=3 * mm
    )
    body_style = ParagraphStyle(
        "body", fontName="Helvetica", fontSize=10.5,
        leading=17, textColor=INK
    )
    note_label = ParagraphStyle(
        "note_label", fontName="Helvetica-Bold", fontSize=10,
        leading=15, textColor=EMERALD_900
    )
    note_text = ParagraphStyle(
        "note_text", fontName="Helvetica", fontSize=10,
        leading=15, textColor=INK
    )
    fact_value = ParagraphStyle(
        "fact_value", fontName="Helvetica-Bold", fontSize=15,
        leading=19, textColor=EMERALD_600
    )
    fact_label = ParagraphStyle(
        "fact_label", fontName="Helvetica", fontSize=8.5,
        leading=11, textColor=MUTED
    )
    small_style = ParagraphStyle(
        "small", fontName="Helvetica", fontSize=8.5,
        leading=12, textColor=MUTED
    )

    # ---------- Custom flowable: stars with partial fill ----------
    class Stars(Flowable):

        def __init__(self, rating, size=15):
            super().__init__()
            self.rating = max(0.0, min(5.0, float(rating)))
            self.size = size
            self.gap = 3
            self.width = 5 * size + 4 * self.gap
            self.height = size

        def _star_path(self, canv, cx, cy, r):
            import math
            path = canv.beginPath()
            for i in range(10):
                angle = math.pi / 2 + i * math.pi / 5
                radius = r if i % 2 == 0 else r * 0.42
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            path.close()
            return path

        def draw(self):
            canv = self.canv
            r = self.size / 2
            for i in range(5):
                cx = r + i * (self.size + self.gap)
                cy = r
                fill = max(0.0, min(1.0, self.rating - i))

                canv.setFillColor(STAR_OFF)
                canv.setStrokeColor(STAR_OFF)
                canv.drawPath(self._star_path(canv, cx, cy, r), fill=1, stroke=0)

                if fill > 0:
                    canv.saveState()
                    clip = canv.beginPath()
                    clip.rect(cx - r, 0, 2 * r * fill, self.size)
                    canv.clipPath(clip, stroke=0, fill=0)
                    canv.setFillColor(BRASS)
                    canv.drawPath(self._star_path(canv, cx, cy, r), fill=1, stroke=0)
                    canv.restoreState()

    # ---------- Page decoration ----------
    listing_info = final_report.get("listing_information", {})
    estimate = final_report.get("listing_estimate", {})

    def draw_page(canv, doc):
        canv.saveState()

        # Ivory page
        canv.setFillColor(IVORY)
        canv.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

        if doc.page == 1:
            # Emerald header band
            canv.setFillColor(EMERALD_900)
            canv.rect(0, PAGE_H - HEADER_H, PAGE_W, HEADER_H, fill=1, stroke=0)

            canv.setFillColor(IVORY)
            canv.setFont("Helvetica-Bold", 26)
            canv.drawString(MARGIN, PAGE_H - 22 * mm, "Hujrah AI")

            canv.setFillColor(HexColor("#BFD8C8"))
            canv.setFont("Helvetica", 11)
            canv.drawString(
                MARGIN, PAGE_H - 30 * mm,
                "Verified apartment listing report"
            )

            # Arch (a doorway into a room), sitting on the band's bottom edge
            arch_w = 24 * mm
            arch_h = 28 * mm
            ax = PAGE_W - MARGIN - arch_w
            ay = PAGE_H - HEADER_H
            r = arch_w / 2

            canv.setStrokeColor(BRASS)
            canv.setLineWidth(1.4)
            arch = canv.beginPath()
            arch.moveTo(ax, ay)
            arch.lineTo(ax, ay + arch_h - r)
            arch.arcTo(ax, ay + arch_h - 2 * r, ax + arch_w, ay + arch_h,
                       startAng=180, extent=-180)
            arch.lineTo(ax + arch_w, ay)
            canv.drawPath(arch, fill=0, stroke=1)

            # Door light inside the arch
            canv.setFillColor(HexColor("#134A38"))
            inner = canv.beginPath()
            pad = 2.2 * mm
            inner.moveTo(ax + pad, ay)
            inner.lineTo(ax + pad, ay + arch_h - r)
            inner.arcTo(ax + pad, ay + arch_h - 2 * r + pad,
                        ax + arch_w - pad, ay + arch_h - pad,
                        startAng=180, extent=-180)
            inner.lineTo(ax + arch_w - pad, ay)
            inner.close()
            canv.drawPath(inner, fill=1, stroke=0)

        # Footer
        canv.setStrokeColor(LINE)
        canv.setLineWidth(0.8)
        canv.line(MARGIN, 16 * mm, PAGE_W - MARGIN, 16 * mm)

        canv.setFillColor(MUTED)
        canv.setFont("Helvetica", 8)
        canv.drawString(
            MARGIN, 11 * mm,
            "Hujrah AI  |  AI-assisted property analysis. "
            "All listings require owner approval."
        )
        canv.drawRightString(
            PAGE_W - MARGIN, 11 * mm, f"Page {doc.page}"
        )

        canv.restoreState()

    # ---------- Content ----------
    story = []

    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph(clean(listing_info.get("name", "Apartment Listing")), title_style))
    story.append(Spacer(1, 1.5 * mm))
    story.append(Paragraph(clean(listing_info.get("location", "")), location_style))
    story.append(Spacer(1, 6 * mm))

    # Photos: one large, two stacked, all cropped to fixed shapes
    def prepared(pil_image, w_mm, h_mm):
        fitted = ImageOps.fit(
            pil_image, (int(w_mm * 10), int(h_mm * 10)),
            method=3, centering=(0.5, 0.5)
        )
        buffer = BytesIO()
        fitted.save(buffer, format="JPEG", quality=88)
        buffer.seek(0)
        return RLImage(buffer, width=w_mm * mm, height=h_mm * mm)

    gap_mm = 3
    big_w = 104
    small_w = CONTENT_W / mm - big_w - gap_mm
    small_h = 36
    big_h = small_h * 2 + gap_mm

    values = list(images.values())

    if len(values) >= 3:
        photo_table = Table(
            [
                [prepared(values[0]["image"], big_w, big_h),
                 prepared(values[1]["image"], small_w, small_h)],
                ["", prepared(values[2]["image"], small_w, small_h)],
            ],
            colWidths=[big_w * mm, CONTENT_W - big_w * mm],
            rowHeights=[(small_h + gap_mm / 2) * mm, (small_h + gap_mm / 2) * mm],
        )
        photo_table.setStyle(TableStyle([
            ("SPAN", (0, 0), (0, 1)),
            ("VALIGN", (0, 0), (0, 1), "TOP"),
            ("VALIGN", (1, 0), (1, 0), "TOP"),
            ("VALIGN", (1, 1), (1, 1), "BOTTOM"),
            ("LEFTPADDING", (0, 0), (0, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("LEFTPADDING", (1, 0), (1, -1), gap_mm * mm),
        ]))
        story.append(photo_table)

    story.append(Spacer(1, 7 * mm))

    # Key facts
    try:
        rating_value = float(listing_info.get("demo_rating", 0))
    except (TypeError, ValueError):
        rating_value = 0.0

    try:
        people = max(int(listing_info.get("required_people", 1)), 1)
        capacity = max(int(listing_info.get("test_capacity", 1)), 1)
    except (TypeError, ValueError):
        people, capacity = 1, 1

    fit = min(100, round(capacity / people * 100))

    def fact(value_flowable, label):
        return [value_flowable, Spacer(1, 1.5 * mm), Paragraph(label, fact_label)]

    facts = Table(
        [[
            fact(Paragraph(clean(listing_info.get("monthly_rent", "")), fact_value),
                 "Monthly rent (AI estimate)"),
            fact(Stars(rating_value), f"{rating_value:.1f} out of 5 (AI estimate)"),
            fact(Paragraph(str(capacity), fact_value), "Estimated capacity"),
            fact(Paragraph(f"{fit}%", fact_value),
                 f"Occupancy fit for {people} people"),
        ]],
        colWidths=[CONTENT_W * 0.31, CONTENT_W * 0.27, CONTENT_W * 0.18, CONTENT_W * 0.24],
    )
    facts.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PAPER),
        ("BOX", (0, 0), (-1, -1), 0.8, LINE),
        ("LINEAFTER", (0, 0), (-2, -1), 0.8, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(facts)

    story.append(Spacer(1, 9 * mm))

    # Description
    description = get_verified_description(final_report)

    story.append(CondPageBreak(45 * mm))
    story.append(Paragraph("About this apartment", section_style))

    # Split into small rows so a long description can flow across pages
    import re

    sentences = re.split(r"(?<=[.!?])\s+", str(description).strip())
    chunks = [
        " ".join(sentences[i:i + 2])
        for i in range(0, len(sentences), 2)
    ] or [""]

    desc_box = Table(
        [[Paragraph(clean(chunk), body_style)] for chunk in chunks],
        colWidths=[CONTENT_W],
        splitByRow=1,
    )
    last = len(chunks) - 1
    desc_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PAPER),
        ("BOX", (0, 0), (-1, -1), 0.8, LINE),
        ("LINEBEFORE", (0, 0), (0, -1), 3, BRASS),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, 0), 14),
        ("BOTTOMPADDING", (0, last), (-1, last), 14),
    ]))
    story.append(desc_box)

    # How the AI estimated the numbers
    notes = [
        ("Rating", estimate.get("rating_reason", "")),
        ("Rent", estimate.get("rent_reason", "")),
        ("Capacity", estimate.get("capacity_reason", "")),
    ]
    notes = [(label, text) for label, text in notes if text]

    if notes:

        rows = [
            [Paragraph(label, note_label), Paragraph(clean(text), note_text)]
            for label, text in notes
        ]

        notes_table = Table(rows, colWidths=[24 * mm, CONTENT_W - 24 * mm])
        notes_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LINEBELOW", (0, 0), (-1, -2), 0.5, LINE),
        ]))

        confidence = clean(estimate.get("confidence", ""))

        block = [
            Spacer(1, 9 * mm),
            Paragraph("How the AI estimated these values", section_style),
            notes_table,
        ]

        if confidence:
            block.append(Spacer(1, 3 * mm))
            block.append(Paragraph(
                f"AI confidence: {confidence}. Estimates are based on the "
                "photos only and should be reviewed by the owner.",
                small_style
            ))

        story.append(KeepTogether(block))

    # ---------- Build ----------
    buffer = BytesIO()

    doc = BaseDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=22 * mm,
        title=f"{listing_info.get('name', 'Apartment Listing')} | Hujrah AI",
        author="Hujrah AI",
    )

    first_frame = Frame(
        MARGIN, 22 * mm, CONTENT_W, PAGE_H - HEADER_H - 22 * mm - 8 * mm,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        id="first"
    )
    later_frame = Frame(
        MARGIN, 22 * mm, CONTENT_W, PAGE_H - MARGIN - 22 * mm,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        id="later"
    )

    doc.addPageTemplates([
        PageTemplate(id="First", frames=[first_frame], onPage=draw_page,
                     autoNextPageTemplate="Later"),
        PageTemplate(id="Later", frames=[later_frame], onPage=draw_page),
    ])

    doc.build(story)

    return buffer.getvalue()


# =====================================================
# 11. FULL PROPERTY DETAILS
# =====================================================

def display_full_details(final_report, images):

    st.markdown(
        '<div class="section-title">Apartment Details</div>',
        unsafe_allow_html=True
    )

    description = get_verified_description(final_report)

    estimate = final_report.get("listing_estimate", {})

    with st.expander(
        "تفاصيل الشقة والوصف الكامل",
        expanded=False
    ):

        st.markdown(
            '<div class="details-heading">AI verified description</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="description-box">
                {safe_text(description)}
            </div>
            """,
            unsafe_allow_html=True
        )

        notes = [
            ("Rating", estimate.get("rating_reason", "")),
            ("Rent", estimate.get("rent_reason", "")),
            ("Capacity", estimate.get("capacity_reason", "")),
        ]

        notes = [(label, text) for label, text in notes if text]

        if notes:

            st.write("")

            st.markdown(
                '<div class="details-heading">How the AI estimated these values</div>',
                unsafe_allow_html=True
            )

            for label, text in notes:

                st.markdown(f"**{label}:** {safe_text(text)}")

            confidence = estimate.get("confidence", "")

            if confidence:

                st.caption(
                    f"AI confidence: {safe_text(confidence)}. "
                    "These are estimates from the photos only."
                )

        st.write("")

        try:

            pdf_bytes = build_report_pdf(final_report, images)

            st.download_button(
                "Download listing report (PDF)",
                data=pdf_bytes,
                file_name="Hujrah_Listing_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        except ImportError:

            st.warning(
                "PDF export needs the reportlab package. "
                "Run: pip install reportlab"
            )


# =====================================================
# 12. MAIN HEADER
# =====================================================

st.markdown(
    '<div class="hero">'
    '<div class="hero-text">'
    '<h1>Hujrah AI</h1>'
    '<p>Three photos in. A verified, publish-ready apartment '
    'listing out.</p>'
    '</div>'
    '<div class="arch" aria-hidden="true"><span>حجرة</span></div>'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="section-title">Create your test listing</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload three apartment photos and let the AI analyze "
    "the visible furniture, colors, and apartment details."
)


# =====================================================
# 13. LISTING TEST INFORMATION
# =====================================================

with st.expander(
    "Listing Information",
    expanded=True
):

    col1, col2 = st.columns(2)

    with col1:

        location = st.text_input(
            "Apartment Address",
            value="Giza, Egypt"
        )

    with col2:

        occupants = st.number_input(
            "Required Number of People",
            min_value=1,
            max_value=20,
            value=2
        )

    st.caption(
        "The name, rating, rent and capacity are created by the AI "
        "from your photos."
    )


# =====================================================
# 14. UPLOAD IMAGES
# =====================================================

uploaded_files = st.file_uploader(
    "Upload exactly 3 apartment photos",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)


images = {}

if uploaded_files:

    if len(uploaded_files) != 3:

        st.warning(
            f"Please upload exactly 3 images. "
            f"You uploaded {len(uploaded_files)}."
        )

    else:

        for index, uploaded_file in enumerate(uploaded_files):

            image_name = f"image_{index + 1}"

            images[image_name] = {
                "image": prepare_image(uploaded_file),
                "filename": uploaded_file.name
            }

        st.success("Three apartment photos uploaded successfully.")

        preview_cols = st.columns(3)

        for index, (image_name, data) in enumerate(images.items()):

            with preview_cols[index]:

                st.image(
                    make_thumb(data["image"]),
                    caption=data["filename"],
                    use_container_width=True
                )


# =====================================================
# 15. START ANALYSIS
# =====================================================

if uploaded_files and len(uploaded_files) == 3:

    st.write("")

    if st.button(
        "Start AI Analysis",
        type="primary",
        use_container_width=True
    ):

        try:

            # =========================================
            # STAGE 1: FLORENCE ANALYSIS
            # =========================================

            with st.status(
                "Stage 1/5: Florence-2 image analysis...",
                expanded=True
            ) as status:

                processor, model, device = load_florence()

                florence_results = []

                for image_name, data in images.items():

                    st.write(f"Analyzing {image_name}...")

                    description = analyze_with_florence(
                        data["image"],
                        processor,
                        model,
                        device
                    )

                    florence_results.append({
                        "image_name": image_name,
                        "filename": data["filename"],
                        "description": str(description)
                    })

                save_json(
                    "florence_results.json",
                    florence_results
                )

                status.update(
                    label="Stage 1/5 complete",
                    state="complete"
                )

            # =========================================
            # GEMINI CLIENT
            # =========================================

            client = create_gemini_client()

            # =========================================
            # STAGE 2: STRUCTURED ANALYSIS
            # =========================================

            with st.status(
                "Stage 2/5: Gemini structured analysis...",
                expanded=True
            ) as status:

                structured_result = run_structured_analysis(
                    client,
                    florence_results
                )

                save_json(
                    "structured_analysis.json",
                    structured_result
                )

                status.update(
                    label="Stage 2/5 complete",
                    state="complete"
                )

            # =========================================
            # STAGE 3: SEMANTIC COMPARISON
            # =========================================

            with st.status(
                "Stage 3/5: Semantic comparison...",
                expanded=True
            ) as status:

                semantic_comparison = run_semantic_comparison(
                    client,
                    florence_results,
                    structured_result
                )

                save_json(
                    "semantic_comparison.json",
                    semantic_comparison
                )

                status.update(
                    label="Stage 3/5 complete",
                    state="complete"
                )

            # =========================================
            # STAGE 4: VISUAL VERIFICATION
            # =========================================

            with st.status(
                "Stage 4/5: Visual verification...",
                expanded=True
            ) as status:

                verification_result = run_visual_verification(
                    client,
                    images,
                    structured_result,
                    semantic_comparison
                )

                save_json(
                    "visual_verification.json",
                    verification_result
                )

                status.update(
                    label="Stage 4/5 complete",
                    state="complete"
                )

            # =========================================
            # STAGE 5: LISTING ESTIMATION
            # =========================================

            with st.status(
                "Stage 5/5: Creating listing details...",
                expanded=True
            ) as status:

                listing_estimate = run_listing_estimation(
                    client,
                    images,
                    verification_result,
                    location
                )

                save_json(
                    "listing_estimate.json",
                    listing_estimate
                )

                status.update(
                    label="Stage 5/5 complete",
                    state="complete"
                )

            # =========================================
            # FINAL REPORT
            # =========================================

            final_report = {

                "project": "Hujrah AI Apartment Analyzer",

                "model": "Florence-2-large",

                "language_model": GEMINI_MODEL,

                "number_of_images": len(images),

                "listing_information": {
                    "name": listing_estimate.get("listing_name", "Apartment Listing"),
                    "location": location,
                    "monthly_rent": listing_estimate.get("estimated_rent_text", "Not estimated"),
                    "demo_rating": listing_estimate.get("estimated_rating", 0),
                    "required_people": occupants,
                    "test_capacity": listing_estimate.get("estimated_capacity", 1)
                },

                "listing_estimate": listing_estimate,

                "florence_analysis": florence_results,

                "structured_analysis": structured_result,

                "semantic_comparison": semantic_comparison,

                "visual_verification": verification_result,

                "final_description": verification_result.get(
                    "verified_description",
                    ""
                ),

                "owner_approval_required": True,

                "values_are_ai_estimates": True,

            }

            save_json(
                "Hujrah_Final_Report.json",
                final_report
            )

            # Save results in Streamlit session
            st.session_state["final_report"] = final_report

            st.session_state["florence_results"] = florence_results

            st.session_state["structured_result"] = structured_result

            st.session_state["semantic_comparison"] = semantic_comparison

            st.session_state["verification_result"] = verification_result

            st.session_state["analysis_images"] = images

            st.success(
                "AI Analysis completed successfully. "
                "All JSON results were saved."
            )

        except Exception as exc:

            st.error(f"Analysis failed: {exc}")

            st.exception(exc)


# =====================================================
# 16. DISPLAY FINAL PROPERTY CARD
# =====================================================

if "final_report" in st.session_state:

    final_report = st.session_state["final_report"]

    report_images = st.session_state["analysis_images"]

    listing_info = final_report.get(
        "listing_information",
        {}
    )

    st.divider()

    st.markdown(
        '<div class="section-title">Your Apartment Listing</div>',
        unsafe_allow_html=True
    )

    display_property_card(
        images=report_images,
        listing_name=listing_info.get(
            "name",
            "Apartment Listing"
        ),
        location=listing_info.get(
            "location",
            "Location not provided"
        ),
        rent=listing_info.get(
            "monthly_rent",
            "Not provided"
        ),
        rating=float(
            listing_info.get("demo_rating", 0)
        ),
        occupants=int(
            listing_info.get("required_people", 1)
        ),
        max_occupants=int(
            listing_info.get("test_capacity", 1)
        )
    )

    st.write("")

    st.markdown(
        """
        <div class="section-title">
            Want to know more about this apartment?
        </div>
        """,
        unsafe_allow_html=True
    )

    display_full_details(final_report, report_images)

    st.caption(
        "Hujrah AI. AI-assisted property analysis. "
        "All listings require owner approval."
    )

else:

    st.info(
        "Upload exactly three photos and start the AI analysis "
        "to generate your apartment listing."
    )