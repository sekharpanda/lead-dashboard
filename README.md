# Lead Dashboard Generator

Upload your leads CSV/XLSX and generate:
- 3 bar charts (Leads, Spend, CPL)
- 1 pie chart (Sentiment)
- Highlights block
- Downloadable PNG

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Deploy (Streamlit Cloud)
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Create a new app, select this repo and `app.py` as entrypoint.
4. Deploy.
