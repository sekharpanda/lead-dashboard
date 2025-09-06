## Setup notes (OCR.space + Streamlit Cloud)

This app uses OCR.space for image OCR (recommended on Streamlit Cloud because Tesseract binary cannot be installed there).

1. Get OCR.space API key: https://ocr.space/ocrapi (free tier available).
2. In Streamlit Cloud -> Manage app -> Settings -> Secrets, add:
   OCR_SPACE_API_KEY="your_api_key_here"
3. Commit `app.py` and `requirements.txt` to your repo and let Streamlit Cloud deploy.
4. Upload Leads (CSV/XLSX) and either a spend file (CSV/XLSX) or a spend image (JPG/PNG) to the app. Follow the editable steps:
   - Verify OCR-extracted spend rows
   - Confirm spend->lead campaign mapping
   - Click Apply mapping to merge and download outputs (merged Excel, aggregates, PDF report)

Notes:
- OCR accuracy depends on image clarity. Crop to table and use high-res images.
- If you run locally and want local OCR: install Tesseract (system binary) + `pytesseract` and the app will attempt local OCR first.
