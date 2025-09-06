# app.py
import os
import re
import io
import logging
from typing import Optional, List

import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# OCR libs
import pytesseract
import cv2

# ---------- CONFIG ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lead-dashboard-ocr")
DEFAULT_DATE_COL = "created_at"

st.set_page_config(page_title="Lead Dashboard (OCR spend)", layout="wide")

# ---------- HELPERS ----------
def ensure_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c not in df.columns]

def safe_parse_dates(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def parse_spend_string(s: str) -> float:
    """Normalize and parse a single spend string to float."""
    if not isinstance(s, str):
        s = str(s)
    t = re.sub(r"[₹$€£,]|AED|INR|usd|rs\.?", "", s, flags=re.IGNORECASE)
    t = re.sub(r"[^0-9\.\-]", "", t)
    t = re.sub(r"\.(?=.*\.)", "", t)
    try:
        return float(t) if t != "" else 0.0
    except Exception:
        return 0.0

# ---------- OCR IMAGE -> TABLE ----------
def preprocess_image_for_ocr(pil_img: Image.Image):
    """Convert to grayscale, denoise, threshold to help OCR."""
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Adaptive threshold helps with many table images
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)
    # optional median blur
    blur = cv2.medianBlur(th, 3)
    return blur

def ocr_image_to_text(pil_img: Image.Image) -> str:
    proc = preprocess_image_for_ocr(pil_img)
    # pytesseract accepts numpy arrays or pillow images
    txt = pytesseract.image_to_string(proc, lang='eng')
    return txt

def extract_table_from_text(raw_text: str):
    """
    Heuristic parser: looks for lines containing a campaign name + numeric fields.
    Returns list of dicts with keys: Campaign, Date (optional), Leads (opt), Spend.
    """
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    rows = []
    # Two main heuristics:
    # 1) Lines that contain a currency/number pair -> likely campaign + spend
    # 2) Multi-line tables broken into sections: look for campaign name lines and next lines with numbers
    # We'll attempt multiple passes.
    # Regex to capture numbers (with optional commas and decimals).
    num_re = r"[-+]?\d{1,3}(?:[,.\d]*\d)?(?:\.\d+)?"
    # common currency pattern
    currency_re = re.compile(r"(?:₹|\$|rs\.?|INR|AED)?\s*("+num_re+r")", re.I)

    # Pass 1: find "Campaign ... Spend" style lines
    for line in lines:
        # If line has both text and at least one number, treat last number as spend and rest as campaign
        numbers = re.findall(num_re, line.replace(',', ''))
        if numbers:
            # if line mostly numeric skip; else split
            non_numeric = re.sub(num_re, "|", line)
            # Check if there are alphabetic chars -> campaign likely present
            if re.search(r"[A-Za-z]", line):
                # We'll take last number as spend and attempt to extract leads if present
                # Extract all numeric tokens
                tokens = re.findall(num_re, line)
                if tokens:
                    spend_token = tokens[-1]
                    # attempt to find leads token (smaller integer) among tokens
                    leads_token = None
                    if len(tokens) >= 2:
                        # try first token as leads if small int
                        try:
                            candidate = int(float(tokens[0]))
                            if candidate < 2000:
                                leads_token = candidate
                        except Exception:
                            leads_token = None
                    # campaign name by removing numeric parts
                    campaign = re.sub(num_re, "", line)
                    campaign = re.sub(r"[^A-Za-z0-9\s&\-]", "", campaign).strip()
                    rows.append({
                        "Campaign": campaign or "unknown",
                        "Leads": int(leads_token) if leads_token is not None else np.nan,
                        "Spend": parse_spend_string(spend_token)
                    })
                    continue

    # Pass 2: block parsing: look for headings lines that seem like campaign titles (all letters) followed by a line with numbers
    for i, line in enumerate(lines[:-1]):
        # if current line mostly alphabetic and next line has numbers, combine
        if len(line) > 0 and re.search(r"[A-Za-z]", line) and not re.search(num_re, line):
            nxt = lines[i+1]
            tokens = re.findall(num_re, nxt)
            if tokens:
                spend_token = tokens[-1]
                leads_token = None
                if len(tokens) >= 1:
                    # choose first numeric token as leads if plausible
                    try:
                        first_num = int(float(tokens[0]))
                        if first_num < 2000:
                            leads_token = first_num
                    except Exception:
                        leads_token = None
                campaign = re.sub(r"[^A-Za-z0-9\s&\-]", "", line).strip()
                rows.append({
                    "Campaign": campaign or "unknown",
                    "Leads": int(leads_token) if leads_token is not None else np.nan,
                    "Spend": parse_spend_string(spend_token)
                })
    # Merge/clean rows: aggregate by campaign (sum spends if duplicates)
    if not rows:
        return pd.DataFrame(columns=["Campaign", "Leads", "Spend"])
    df = pd.DataFrame(rows)
    # clean campaign names
    df["Campaign"] = df["Campaign"].str.strip().replace("", "unknown")
    # group by campaign
    agg = df.groupby("Campaign", as_index=False).agg(
        Leads=("Leads", lambda s: int(np.nansum(s)) if s.notna().any() else 0),
        Spend=("Spend", "sum")
    )
    return agg

# ---------- SPEND MERGE / CPL ----------
def merge_spend_with_leads(leads_df: pd.DataFrame, spend_df: pd.DataFrame, lead_campaign_col: str = "campaign_mapped", spend_campaign_col: str = "Campaign"):
    leads = leads_df.copy()
    if lead_campaign_col not in leads.columns:
        # try to detect a campaign column
        possible = [c for c in leads.columns if re.search(r"campaign|source|utm", c, re.I)]
        if possible:
            leads["campaign_mapped"] = leads[possible[0]].astype(str).fillna("unknown")
        else:
            leads["campaign_mapped"] = "unknown"
    # normalize campaign keys to compare better
    leads["__cmp_key"] = leads["campaign_mapped"].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    spend_df = spend_df.copy()
    spend_df[spend_campaign_col] = spend_df[spend_campaign_col].astype(str)
    spend_df["__cmp_key"] = spend_df[spend_campaign_col].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    # join
    merged = leads.merge(spend_df[["__cmp_key", "Spend"]].rename(columns={"Spend": "campaign_spend"}), on="__cmp_key", how="left")
    # If spend is per campaign and repeated for each lead row, we need to divide spend over leads per campaign
    # compute leads per campaign
    campaign_counts = merged.groupby("__cmp_key").size().rename("leads_count").reset_index()
    merged = merged.merge(campaign_counts, on="__cmp_key", how="left")
    merged["campaign_spend_per_lead"] = merged.apply(lambda r: (r["campaign_spend"] / r["leads_count"]) if r["leads_count"]>0 and not pd.isna(r["campaign_spend"]) else 0.0, axis=1)
    return merged, spend_df

# ---------- UI & MAIN ----------
def main():
    st.title("Lead Dashboard — upload leads + spend image (OCR)")

    st.sidebar.header("Data inputs")
    uploaded_leads = st.sidebar.file_uploader("Upload Leads file (CSV / Excel)", type=["csv", "xlsx", "xls"])
    uploaded_spend_file = st.sidebar.file_uploader("OR upload spend file (CSV / Excel) — optional", type=["csv", "xlsx", "xls"])
    uploaded_spend_image = st.sidebar.file_uploader("OR upload spend image (jpg/png) — OCR will try to extract", type=["png", "jpg", "jpeg"])

    spend_source = None
    spend_df = pd.DataFrame(columns=["Campaign", "Leads", "Spend"])

    # Load leads first
    if uploaded_leads is None:
        st.info("Please upload your leads file (CSV or Excel) to continue. You can also upload a spend file or spend image to supply campaign spend.")
        st.stop()

    try:
        if uploaded_leads.name.lower().endswith(".csv"):
            leads_df = pd.read_csv(uploaded_leads)
        else:
            leads_df = pd.read_excel(uploaded_leads)
    except Exception as e:
        st.error(f"Failed to read leads file: {e}")
        st.stop()

    with st.expander("Leads sample and columns"):
        st.write("Rows:", len(leads_df))
        st.dataframe(leads_df.head(10))
        st.write("Columns:", list(leads_df.columns))

    # If a spend file was provided, prefer it
    if uploaded_spend_file is not None:
        try:
            if uploaded_spend_file.name.lower().endswith(".csv"):
                spend_df_loaded = pd.read_csv(uploaded_spend_file)
            else:
                spend_df_loaded = pd.read_excel(uploaded_spend_file)
            # try to map columns: prefer if columns 'Campaign' and 'Spend' exist
            # ask user to confirm mapping via UI
            st.sidebar.write("Spend file detected — confirm mapping")
            spend_campaign_col = st.sidebar.selectbox("Campaign column in spend file", options=list(spend_df_loaded.columns), index=0)
            spend_amount_col = st.sidebar.selectbox("Spend amount column in spend file", options=list(spend_df_loaded.columns), index=1 if len(spend_df_loaded.columns)>1 else 0)
            # build cleaned spend df
            spend_df = pd.DataFrame({
                "Campaign": spend_df_loaded[spend_campaign_col].astype(str).fillna("unknown"),
                "Spend": spend_df_loaded[spend_amount_col].apply(parse_spend_string)
            })
            spend_source = "file"
        except Exception as e:
            st.sidebar.error(f"Failed to load spend file: {e}")

    # Else if image provided, try OCR extraction
    if uploaded_spend_image is not None and spend_source is None:
        try:
            img = Image.open(uploaded_spend_image)
            st.sidebar.image(img, caption="Uploaded spend image (OCR source)")
            raw_text = ocr_image_to_text(img)
            st.sidebar.text_area("OCR raw text (preview)", raw_text[:5000], height=200)
            extracted_df = extract_table_from_text(raw_text)
            if extracted_df.empty:
                st.sidebar.warning("OCR did not find clear campaign spend rows. You may need to upload a spend file or correct manually below.")
            else:
                st.sidebar.success("OCR extracted spend table — please verify/edit and click 'Use extracted spend'.")
            # show editable table for corrections
            st.write("### OCR extracted spend table — review and edit if needed")
            if extracted_df.empty:
                st.info("No spend rows were automatically found. You can either upload a spend file or create/enter spend rows manually below.")
                extracted_df = pd.DataFrame(columns=["Campaign", "Leads", "Spend"])
            edited = st.data_editor(extracted_df, num_rows="dynamic", use_container_width=True)
            if st.button("Use extracted spend"):
                # Normalize numeric column
                edited["Spend"] = edited["Spend"].apply(parse_spend_string)
                spend_df = edited[["Campaign", "Leads", "Spend"]].copy()
                spend_source = "ocr"
        except Exception as e:
            st.sidebar.error(f"OCR failed: {e}")

    # If no spend file and no OCR result, give the option to manually create spend table
    if spend_source is None and uploaded_spend_file is None and uploaded_spend_image is None:
        st.sidebar.info("No spend input detected. You can upload a spend file or a spend image, or enter spend rows below.")
        manual = st.sidebar.checkbox("Enter spend rows manually")
        if manual:
            st.write("### Enter spend rows manually")
            manual_df = st.data_editor(pd.DataFrame([{"Campaign": "", "Leads": 0, "Spend": 0.0}]), num_rows="dynamic", use_container_width=True)
            if st.button("Use manual spend"):
                manual_df["Spend"] = manual_df["Spend"].apply(parse_spend_string)
                spend_df = manual_df.copy()
                spend_source = "manual"

    # If still none, default spend_df stays empty (all zeros)
    # Proceed to merge leads with spend (spend_df might be empty)
    merged, cleaned_spend_df = merge_spend_with_leads(leads_df, spend_df)

    # compute per-campaign aggregates and CPLs for display
    agg_leads = merged.groupby("campaign_mapped").agg(
        leads_count=("campaign_mapped","size"),
        spend_total=("campaign_spend", "first"),  # campaign_spend was attached per row
    ).reset_index()
    # If spend_total is NaN, replace with 0
    if "spend_total" in agg_leads.columns:
        agg_leads["spend_total"] = agg_leads["spend_total"].fillna(0.0)
    else:
        agg_leads["spend_total"] = 0.0
    agg_leads["CPL"] = agg_leads.apply(lambda r: (r["spend_total"]/r["leads_count"]) if r["leads_count"]>0 else 0.0, axis=1)

    # Show final outputs and charts
    st.header("Campaign performance (merged)")
    st.write("Spend source:", spend_source or "none (no spend supplied)")
    st.dataframe(agg_leads.sort_values("leads_count", ascending=False).reset_index(drop=True))

    # Charts
    st.subheader("Charts")
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(8,3))
        ax1.bar(agg_leads["campaign_mapped"], agg_leads["leads_count"])
        ax1.set_title("Leads per Campaign")
        ax1.set_ylabel("Leads")
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(8,3))
        ax2.bar(agg_leads["campaign_mapped"], agg_leads["spend_total"])
        ax2.set_title("Spend per Campaign")
        ax2.set_ylabel("Spend")
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(8,3))
    ax3.bar(agg_leads["campaign_mapped"], agg_leads["CPL"])
    ax3.set_title("CPL per Campaign")
    ax3.set_ylabel("CPL")
    ax3.tick_params(axis='x', rotation=45)
    st.pyplot(fig3)

    # sentiment quick heuristic (if notes exist)
    # attempt to find a notes/status column
    possible_notes = [c for c in leads_df.columns if re.search(r"note|status|remark|message|comment", c, re.I)]
    if possible_notes:
        notes_col = possible_notes[0]
        notes = leads_df[notes_col].astype(str).fillna("")
        pos_k = r"\b(good|positive|interested|converted|yes)\b"
        neg_k = r"\b(bad|not interested|no|complaint|angry)\b"
        neut_k = r"\b(follow up|follow-up|callback|maybe|pending|contacted|visited)\b"
        sent = []
        for t in notes:
            if re.search(pos_k, t, re.I):
                sent.append("Positive")
            elif re.search(neg_k, t, re.I):
                sent.append("Negative")
            elif re.search(neut_k, t, re.I):
                sent.append("Neutral")
            else:
                sent.append("Neutral")
        leads_df["sentiment_normalized"] = sent
        sent_counts = pd.Series(sent).value_counts()
        fig4, ax4 = plt.subplots(figsize=(4,4))
        ax4.pie(sent_counts.values, labels=sent_counts.index, autopct='%1.1f%%', startangle=140)
        ax4.set_title("Lead Sentiment Distribution")
        st.pyplot(fig4)
    else:
        st.info("No notes/status column found to compute quick sentiment. Upload notes in leads file for sentiment analysis.")

    # Downloads: campaign aggregates and merged leads
    st.subheader("Downloads")
    csv_agg = agg_leads.to_csv(index=False).encode('utf-8')
    st.download_button("Download campaign aggregates CSV", data=csv_agg, file_name="campaign_aggregates.csv", mime="text/csv")

    merged_csv = merged.to_csv(index=False).encode('utf-8')
    st.download_button("Download merged leads (with spend per lead)", data=merged_csv, file_name="leads_merged_with_spend.csv", mime="text/csv")

    st.info("If OCR results look off, upload the spend file (CSV/XLSX) or edit the extracted table above. OCR quality depends on image clarity; you can crop/rotate images for better results.")

if __name__ == "__main__":
    main()
