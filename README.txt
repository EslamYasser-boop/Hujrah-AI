# Hujrah AI - Local Streamlit App

## Folder structure
Keep these files in the same folder as your existing `Florence-2-large` model folder:

Hujrah_AI/
- app.py
- requirements.txt
- .env
- run.bat
- Florence-2-large/
- Results/

## Setup (Windows CMD)
1. Open CMD in the Hujrah_AI folder.
2. Create a virtual environment (recommended):
   `python -m venv .venv`
3. Activate it:
   `.venv\\Scripts\\activate`
4. Install PyTorch using the official selector at https://pytorch.org/get-started/locally/
   (Choose the CPU build if you do not have a supported NVIDIA GPU.)
5. Install the remaining packages:
   `pip install -r requirements.txt`
6. Copy `.env.example` to `.env` and put your Gemini API key in `.env`.
7. Run `run.bat`, or use:
   `python -m streamlit run app.py`

The app expects exactly 3 images and saves JSON results in `Results/`.
The model is loaded locally from `Florence-2-large/`; no model download is performed.
