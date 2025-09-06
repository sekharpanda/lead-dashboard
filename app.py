# app.py
"""
Lead Dashboard (Cloud-friendly) with OCR.space fallback and PDF report export.

Features:
- Upload leads file (CSV / XLSX)
- Upload spend file (CSV / XLSX) OR upload spend image (JPG/PNG) -> OCR.space -> editable table
- Merge spend with leads via fuzzy key normalization, compute CPL per campaign
- Charts: Leads / Spend / CPL / Sentiment pie (simple heuristic)
- Download: campaign aggregates CSV + one-page PDF report (charts + bullets)

Requirements:
- Set OCR_SPACE_API_KEY in Streamlit Secrets (or environment) for OCR.
"""

import os
import io
import re
import logging
from typing import Optional, List
from datetime import datetime

import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

logger = logging.getLogger("lead-dashboard")
logging.basicConfig(level=logging.INFO)

# Config
st.set_page_config(page_title="Lead Dashboard (OCR.space)", layout="wide")
OCR_SPACE_API_KEY = st.secrets.get("OCR_SPACE_API_KEY") if "OCR_SPACE_API_KEY" in st.secrets else os.getenv("OCR_SPACE_API_KEY", "")
DEFAULT_DATE_COL = "created_at"

# --- Helpers -----------------------------------------------------------------
def parse_spend_string(s: str) -> float:
    if pd.isna(s):
        return 0.0
    try:
        t = str(s)
        t = re.sub(r"[₹$€£,]|AED|INR|usd|rs\.?", "", t, flags=re.IGNORECASE)
        t = re.sub(r"[^0-9\.\-]", "", t)
        t = re.sub(r"\.(?=.*\.)", "", t)
        return float(t) if t != "" else 0.0
    except Exception:
        return 0.0

def simple_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

# OCR.space call
def ocr_space_image_bytes(image_bytes: bytes, api_key: str, language: str = "eng") -> str:
    url = "https://api.ocr.space/parse/image"
    files = {"file": ("image.png", image_bytes, "image/png")}
    data = {"apikey": api_key, "language": language, "isOverlayRequired": False}
    resp = requests.post(url, files=files, data=data, timeout=60)
    resp.raise_for_status()
    j = resp.json()
    if j.get("IsErroredOnProcessing"):
        logger.warning("OCR.space error: %s", j.get("ErrorMessage"))
        return ""
    parsed = []
    for pr in j.get("ParsedResults", []):
        parsed.append(pr.get("ParsedText", ""))
    return "\n".join(parsed)

# Heuristic: convert OCR raw text to table (Campaign, Leads, Spend)
def extract_table_from_text(raw_text: str) -> pd.DataFrame:
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    rows = []
    num_re = r"[-+]?\d{1,3}(?:[,.\d]*\d)?(?:\.\d+)?"
    for i, line in enumerate(lines):
        # If a line looks like "CampaignName 9 699.00" or "CampaignName - Spend 699"
        numbers = re.findall(num_re, line.replace(",", ""))
        if numbers and re.search(r"[A-Za-z]", line):
            spend_token = numbers[-1]
            leads_token = None
            if len(numbers) >= 2:
                try:
                    candidate = int(float(numbers[0]))
                    if candidate < 2000:
                        leads_token = candidate
                except Exception:
                    leads_token = None
            campaign = re.sub(num_re, "", line)
            campaign = re.sub(r"[^A-Za-z0-9\s&\-]", "", campaign).strip()
            rows.append({"Campaign": campaign or "unknown", "Leads": int(leads_token) if leads_token is not None else np.nan, "Spend": parse_spend_string(spend_token)})
    # Block-style: title line then next line numbers
    for i in range(len(lines)-1):
        title = lines[i]
        nxt = lines[i+1]
        if re.search(r"[A-Za-z]", title) and re.search(num_re, nxt):
            nums = re.findall(num_re, nxt.replace(",", ""))
            if nums:
                spend_token = nums[-1]
                leads_token = None
                try:
                    first = int(float(nums[0]))
                    if first < 2000:
                        leads_token = first
                except Exception:
                    leads_token = None
                campaign = re.sub(r"[^A-Za-z0-9\s&\-]", "", title).strip()
                rows.append({"Campaign": campaign or "unknown", "Leads": int(leads_token) if leads_token is not None else np.nan, "Spend": parse_spend_string(spend_token)})
    if not rows:
        return pd.DataFrame(columns=["Campaign", "Leads", "Spend"])
    df = pd.DataFrame(rows)
    df["Campaign"] = df["Campaign"].str.strip().replace("", "unknown")
    agg = df.groupby("Campaign", as_index=False).agg(Leads=("Leads", lambda s: int(np.nansum(s)) if s.notna().any() else 0), Spend=("Spend", "sum"))
    return agg

# Merge spend with leads
def merge_spend_with_leads(leads_df: pd.DataFrame, spend_df: pd.DataFrame, lead_campaign_col: Optional[str] = None, spend_campaign_col: str = "Campaign"):
    leads = leads_df.copy()
    # detect lead campaign column if not provided
    if lead_campaign_col is None:
        candidates = [c for c in leads.columns if re.search(r"campaign|source|utm|ad", c, re.I)]
        lead_campaign_col = candidates[0] if candidates else None
    if lead_campaign_col is None:
        leads["campaign_mapped"] = "unknown"
    else:
        leads["campaign_mapped"] = leads[lead_campaign_col].astype(str).fillna("unknown")
    leads["__cmp_key"] = leads["campaign_mapped"].astype(str).apply(simple_key)
    spend = spend_df.copy()
    spend[spend_campaign_col] = spend[spend_campaign_col].astype(str).fillna("unknown")
    spend["__cmp_key"] = spend[spend_campaign_col].astype(str).apply(simple_key)
    merged = leads.merge(spend[["__cmp_key", "Spend"]].rename(columns={"Spend":"campaign_spend"}), on="__cmp_key", how="left")
    # compute leads count per campaign key and distribute spend to rows
    campaign_counts = merged.groupby("__cmp_key").size().rename("leads_count").reset_index()
    merged = merged.merge(campaign_counts, on="__cmp_key", how="left")
    merged["campaign_spend_per_lead"] = merged.apply(lambda r: (r["campaign_spend"] / r["leads_count"]) if r["leads_count"]>0 and not pd.isna(r["campaign_spend"]) else 0.0, axis=1)
    return merged, spend

# Generate charts and save images to bytes
def fig_to_image_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

def build_pdf_report(charts: dict, totals: dict, agg_table: pd.DataFrame, out_path: str):
    """
    charts: dict of PIL or file-like images for leads/spend/cpl/sent
    totals: dict of summary metrics
    agg_table: pandas DataFrame (campaign aggregates)
    out_path: path to write PDF
    """
    width, height = landscape(A4)
    c = canvas.Canvas(out_path, pagesize=landscape(A4))
    margin = 40
    # title
    c.setFont("Helvetica-Bold", 18)
    title = totals.get("report_title", f"Campaign Performance — {datetime.now().strftime('%d %b %Y')}")
    c.drawCentredString(width/2, height-30, title)

    # place 4 charts in grid (2x2)
    imgs = ["leads", "spend", "cpl", "sent"]
    img_w = (width - 3*margin) / 2
    img_h = (height - 4*margin - 60) / 2

    coords = [
        (margin, height - margin - img_h - 40),
        (margin*2 + img_w, height - margin - img_h - 40),
        (margin, height - 2*margin - 2*img_h - 40),
        (margin*2 + img_w, height - 2*margin - 2*img_h - 40)
    ]
    for k, coord in zip(imgs, coords):
        img_buf = charts.get(k)
        if img_buf:
            try:
                img = ImageReader(img_buf)
                c.drawImage(img, coord[0], coord[1], width=img_w, height=img_h, preserveAspectRatio=True, anchor='sw')
            except Exception:
                pass

    # Bullets: totals and top campaigns
    text_x = margin
    text_y = coords[-1][1] - 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(text_x, text_y, "Campaign Performance")
    text_y -= 14
    c.setFont("Helvetica", 10)
    bullets = [
        f"Total leads: {totals.get('total_leads', 0)}",
        f"Duplicate leads: {totals.get('duplicate_leads', 0)}",
        f"Unique valid leads: {totals.get('unique_leads', 0)}",
        f"Avg CPL (deduped): {totals.get('avg_cpl_unique', 0):.2f}",
        f"Total spend (merged): {totals.get('total_spend', 0):.2f}"
    ]
    for b in bullets:
        c.drawString(text_x+8, text_y, u"• " + b)
        text_y -= 12

    # top 5 campaigns
    text_y -= 6
    c.setFont("Helvetica-Bold", 12)
    c.drawString(text_x, text_y, "Top campaigns (by leads)")
    text_y -= 14
    c.setFont("Helvetica", 10)
    top_rows = agg_table.sort_values("leads_count", ascending=False).head(6)
    for _, r in top_rows.iterrows():
        c.drawString(text_x+8, text_y, f"- {r['campaign_mapped']}: leads={int(r['leads_count'])}, spend={r.get('spend_total',0):.2f}, CPL={r.get('CPL',0):.2f}")
        text_y -= 12
    c.showPage()
    c.save()

# --- Streamlit UI -----------------------------------------------------------
st.title("Lead Dashboard — Upload leads + spend image or file (Web OCR via OCR.space)")

st.sidebar.header("Inputs")
uploaded_leads = st.sidebar.file_uploader("Upload Leads (CSV or Excel)", type=["csv", "xlsx", "xls"])
uploaded_spend_file = st.sidebar.file_uploader("Optional: Spend file (CSV or Excel)", type=["csv", "xlsx", "xls"])
uploaded_spend_image = st.sidebar.file_uploader("Optional: Spend image (jpg/png) — OCR will try to extract", type=["png", "jpg", "jpeg"])

if uploaded_leads is None:
    st.info("Upload a leads file (CSV or Excel) first. You can also upload a spend file or spend image for spend data.")
    st.stop()

# Load leads
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

# Build spend_df from file or OCR
spend_df = pd.DataFrame(columns=["Campaign", "Leads", "Spend"])
spend_source = None

# Spend file path
if uploaded_spend_file is not None:
    try:
        if uploaded_spend_file.name.lower().endswith(".csv"):
            s_loaded = pd.read_csv(uploaded_spend_file)
        else:
            s_loaded = pd.read_excel(uploaded_spend_file)
        # ask mapping
        st.sidebar.write("Spend file detected — confirm mapping")
        s_campaign_col = st.sidebar.selectbox("Campaign column in spend file", options=list(s_loaded.columns), index=0)
        s_amount_col = st.sidebar.selectbox("Spend amount column in spend file", options=list(s_loaded.columns), index=1 if len(s_loaded.columns)>1 else 0)
        spend_df = pd.DataFrame({"Campaign": s_loaded[s_campaign_col].astype(str).fillna("unknown"), "Spend": s_loaded[s_amount_col].apply(parse_spend_string)})
        spend_source = "file"
    except Exception as e:
        st.sidebar.error(f"Failed to read spend file: {e}")

# If image provided and OCR key available, run OCR.space
if uploaded_spend_image is not None and spend_source is None:
    st.sidebar.image(uploaded_spend_image, caption="Spend image")
    st.sidebar.write("OCR attempt in progress (using OCR.space)..." if OCR_SPACE_API_KEY else "OCR disabled (no OCR_SPACE_API_KEY). Add the key in Streamlit secrets to enable OCR.space.")
    if OCR_SPACE_API_KEY:
        try:
            img_bytes = uploaded_spend_image.getvalue()
            raw_text = ocr_space_image_bytes(img_bytes, OCR_SPACE_API_KEY)
            st.sidebar.text_area("OCR raw text (preview)", raw_text[:8000], height=200)
            extracted = extract_table_from_text(raw_text)
            if extracted.empty:
                st.warning("OCR did not parse clear spend rows. You can edit manually below or upload spend file.")
                extracted = pd.DataFrame(columns=["Campaign", "Leads", "Spend"])
            st.write("### OCR extracted spend (edit if needed)")
            edited = st.data_editor(extracted, num_rows="dynamic", use_container_width=True)
            if st.button("Use extracted spend"):
                edited["Spend"] = edited["Spend"].apply(parse_spend_string)
                spend_df = edited[["Campaign", "Leads", "Spend"]].copy()
                spend_source = "ocr"
        except Exception as e:
            st.sidebar.error(f"OCR failed: {e}")

# Manual entry fallback
if spend_source is None and uploaded_spend_file is None and uploaded_spend_image is None:
    st.sidebar.info("No spend input detected. Optionally upload spend file/image or enter spend rows manually.")
    if st.sidebar.checkbox("Enter spend rows manually"):
        manual_df = st.data_editor(pd.DataFrame([{"Campaign": "", "Leads": 0, "Spend": 0.0}]), num_rows="dynamic", use_container_width=True)
        if st.button("Use manual spend"):
            manual_df["Spend"] = manual_df["Spend"].apply(parse_spend_string)
            spend_df = manual_df.copy()
            spend_source = "manual"

# If still none, spend_df remains empty (all zeros)
if spend_df.empty:
    spend_df = pd.DataFrame({"Campaign": [], "Leads": [], "Spend": []})

# Merge
# let user choose campaign column mapping if they want
possible_lead_campaigns = [c for c in leads_df.columns if re.search(r"campaign|source|utm|ad", c, re.I)]
chosen_campaign_col = None
if possible_lead_campaigns:
    chosen_campaign_col = st.selectbox("Map leads campaign column (detected)", options=["(none)"] + possible_lead_campaigns, index=1 if len(possible_lead_campaigns)>0 else 0)
    if chosen_campaign_col == "(none)":
        chosen_campaign_col = None

merged, cleaned_spend = merge_spend_with_leads(leads_df, spend_df, lead_campaign_col=chosen_campaign_col)
# aggregated view
agg_leads = merged.groupby("campaign_mapped").agg(leads_count=("campaign_mapped","size"), spend_total=("campaign_spend","first")).reset_index()
if "spend_total" in agg_leads.columns:
    agg_leads["spend_total"] = agg_leads["spend_total"].fillna(0.0)
else:
    agg_leads["spend_total"] = 0.0
agg_leads["CPL"] = agg_leads.apply(lambda r: (r["spend_total"]/r["leads_count"]) if r["leads_count"]>0 else 0.0, axis=1)

# Totals and diagnostics
total_leads = len(merged)
duplicate_detected = merged.get("duplicate", pd.Series([False]*len(merged))).astype(bool).sum()
unique_leads = int(total_leads - int(duplicate_detected))
total_spend = float(spend_df["Spend"].sum()) if not spend_df.empty else float(merged["campaign_spend"].sum())
avg_cpl_unique = (total_spend / unique_leads) if unique_leads>0 else 0.0

st.header("Campaign performance (merged)")
st.write("Spend source:", spend_source or "none")
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

fig3, ax3 = plt.subplots(figsize=(10,3))
ax3.bar(agg_leads["campaign_mapped"], agg_leads["CPL"])
ax3.set_title("CPL per Campaign")
ax3.set_ylabel("CPL")
ax3.tick_params(axis='x', rotation=45)
st.pyplot(fig3)

# Sentiment quick heuristic
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
    st.info("No notes/status column found to compute quick sentiment. Upload notes/remarks in leads file for sentiment analysis.")

# Downloads and PDF report
st.subheader("Downloads & Report")
csv_agg = agg_leads.to_csv(index=False).encode('utf-8')
st.download_button("Download campaign aggregates CSV", data=csv_agg, file_name="campaign_aggregates.csv", mime="text/csv")

merged_csv = merged.to_csv(index=False).encode('utf-8')
st.download_button("Download merged leads (with spend per lead)", data=merged_csv, file_name="leads_merged_with_spend.csv", mime="text/csv")

# prepare chart buffers for PDF
leads_buf = fig_to_image_bytes(fig1)
spend_buf = fig_to_image_bytes(fig2)
cpl_buf = fig_to_image_bytes(fig3)
sent_buf = None
if possible_notes:
    sent_buf = fig_to_image_bytes(fig4)

pdf_path = "/tmp/lead_report.pdf"
totals = {"total_leads": total_leads, "duplicate_leads": int(duplicate_detected), "unique_leads": unique_leads, "avg_cpl_unique": avg_cpl_unique, "total_spend": total_spend, "report_title": f"Campaign Performance ({datetime.now().strftime('%d %b %Y')})"}
charts = {"leads": leads_buf, "spend": spend_buf, "cpl": cpl_buf, "sent": sent_buf}
try:
    build_pdf_report(charts, totals, agg_leads, pdf_path)
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    st.download_button("Download PDF report", data=pdf_bytes, file_name="campaign_report.pdf", mime="application/pdf")
except Exception as e:
    st.warning(f"PDF generation failed: {e}. CSV downloads still available.")

st.success("Done. If OCR missed rows, edit the extracted table and re-run merge (use the 'Use extracted spend' action).")
