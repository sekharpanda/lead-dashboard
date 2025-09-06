### OCR.space setup (for image → spend extraction on Streamlit Cloud)

This app calls OCR.space when a spend image is uploaded. OCR.space is used because Streamlit Cloud cannot install system Tesseract.

1. Get a free API key at https://ocr.space/ocrapi (free tier available).
2. In Streamlit Cloud:
   - Open your deployed app -> Manage app -> Settings -> Secrets.
   - Add a secret named: `OCR_SPACE_API_KEY` with your key value.
   - Example: `OCR_SPACE_API_KEY="your_api_key_here"`
3. Commit `requirements.txt` and `app.py` to the repo. Streamlit Cloud will install the listed packages automatically on deployment.

Notes:
- OCR accuracy depends on image clarity. Crop/rotate images to include only the table and ensure text is readable.
- If you prefer local OCR (Tesseract), run the app locally and install `pytesseract` + the Tesseract binary on your machine.
