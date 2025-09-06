# app.py
"""
Lead Analysis Dashboard (Cloud-ready)
- Upload Leads master (CSV/XLSX)
- Upload Spend file (CSV/XLSX) OR Spend image (JPG/PNG)
- OCR via pytesseract (if available) OR OCR.space API (recommended on Streamlit Cloud)
- Editable extraction + mapping UI -> merge -> duplicate detection -> allocate spend
- Outputs: merged leads Excel/CSV, campaign aggregates, charts, PDF one-pager
"""
import os
import io
import re
import logging
from datetime import datetime
from difflib import get_close_matches
from typing import Optional, List

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import requests

# PDF generator
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

logger = logging.getLogger("lead-dashboard")
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Lead Dashboard (OCR.space)", layout="wide")
OCR_SPACE_API_KEY = st.secrets.get("OCR_SPACE_API_KEY") if "OCR_SPACE_API_KEY" in st.secrets else os.getenv("OCR_SPACE_API_KEY", "")

# Try to import pytesseract for local OCR if available (no crash if absent)
USE_PYTESSERACT = False
try:
    import pytesseract  # type: ignore
    USE_PYTESSERACT = True
    logger.info("pytesseract available - local OCR enabled.")
except Exception:
    logger.info("pytesseract not available - will use OCR.space when needed.")

# ---------- helper functions ----------
def parse_spend_string(s):
    if pd.isna(s):
        return 0.0
    t = str(s)
    t = re.sub(r"[₹$€£,]|AED|INR|usd|rs\.?", "", t, flags=re.IGNORECASE)
    t = re.sub(r"[^0-9\.\-]", "", t)
    t = re.sub(r"\.(?=.*\.)", "", t)
    try:
        return float(t) if t != "" else 0.0
    except Exception:
        return 0.0

def simple_key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower()) if pd.notna(s) else ""

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

def ocr_with_pytesseract(pil_img: Image.Image) -> str:
    # simple local OCR call (if installed)
    try:
        return pytesseract.image_to_string(pil_img, lang="eng")
    except Exception as e:
        logger.exception("pytesseract error: %s", e)
        return ""

def extract_table_from_text(raw_text: str) -> pd.DataFrame:
    # Heuristic parser to turn OCR text into Campaign, Leads, CPL, Spend rows
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    rows = []
    num_re = r"[-+]?\d{1,3}(?:[,.\d]*\d)?(?:\.\d+)?"
    for i, line in enumerate(lines):
        if not re.search(r"[A-Za-z]", line):
            continue
        numbers = re.findall(num_re, line.replace(",", ""))
        if numbers:
            spend_token = numbers[-1]
            leads_token = None
            if len(numbers) >= 2:
                try:
                    cand = int(float(numbers[0]))
                    if cand < 10000:
                        leads_token = cand
                except Exception:
                    leads_token = None
            campaign = re.sub(num_re, "", line)
            campaign = re.sub(r"[^A-Za-z0-9\s&\-]", "", campaign).strip()
            rows.append({"Campaign": campaign or "unknown", "Leads": int(leads_token) if leads_token is not None else np.nan, "CPL": None, "Spend": parse_spend_string(spend_token)})
    # Block style: title line then a numeric line
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
                    if first < 10000:
                        leads_token = first
                except Exception:
                    leads_token = None
                campaign = re.sub(r"[^A-Za-z0-9\s&\-]", "", title).strip()
                rows.append({"Campaign": campaign or "unknown", "Leads": int(leads_token) if leads_token is not None else np.nan, "CPL": None, "Spend": parse_spend_string(spend_token)})
    if not rows:
        return pd.DataFrame(columns=["Campaign", "Leads", "CPL", "Spend"])
    df = pd.DataFrame(rows)
    df["Campaign"] = df["Campaign"].str.strip().replace("", "unknown")
    agg = df.groupby("Campaign", as_index=False).agg(Leads=("Leads", lambda s: int(np.nansum(s)) if s.notna().any() else 0), CPL=("CPL", lambda s: None), Spend=("Spend", "sum"))
    return agg

def fig_to_image_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

def build_pdf_report(charts: dict, totals: dict, agg_table: pd.DataFrame, out_path: str):
    width, height = landscape(A4)
    c = canvas.Canvas(out_path, pagesize=landscape(A4))
    margin = 36
    c.setFont("Helvetica-Bold", 18)
    title = totals.get("report_title", f"Campaign Performance — {datetime.now().strftime('%d %b %Y')}")
    c.drawCentredString(width/2, height-30, title)

    # place 4 charts in grid
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

    # Bulleted summary
    text_x = margin
    text_y = coords[-1][1] - 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(text_x, text_y, "Campaign Performance")
    text_y -= 14
    c.setFont("Helvetica", 10)
    bullets = [
        f"Total leads: {totals.get('total_leads', 0)}",
        f"Duplicate leads (internal): {totals.get('internal_duplicates', 0)}",
        f"External mismatch (reported - in master): {totals.get('external_mismatch_total', 0)}",
        f"Unique valid leads (est.): {totals.get('unique_leads_estimated', 0)}",
        f"Avg CPL (deduped): {totals.get('avg_cpl_unique', 0):.2f}"
    ]
    for b in bullets:
        c.drawString(text_x+8, text_y, u"• " + b)
        text_y -= 12

    # top campaigns
    text_y -= 6
    c.setFont("Helvetica-Bold", 12)
    c.drawString(text_x, text_y, "Top campaigns (by leads)")
    text_y -= 14
    c.setFont("Helvetica", 10)
    top_rows = agg_table.sort_values("leads_rows_count", ascending=False).head(6)
    for _, r in top_rows.iterrows():
        c.drawString(text_x+8, text_y, f"- {r.get('campaign_mapped','')}: leads={int(r.get('leads_rows_count',0))}, spend={r.get('spend_total',0):.2f}, CPL={r.get('CPL_computed',0):.2f}")
        text_y -= 12

    c.showPage()
    c.save()

# ---------- main UI ----------
def main():
    st.title("Lead Dashboard — Upload leads + spend (image or file)")

    st.sidebar.header("Data inputs")
    uploaded_leads = st.sidebar.file_uploader("Upload Leads file (CSV / Excel)", type=["csv", "xlsx", "xls"])
    uploaded_spend_file = st.sidebar.file_uploader("Optional: Spend file (CSV / Excel)", type=["csv", "xlsx", "xls"])
    uploaded_spend_image = st.sidebar.file_uploader("Optional: Spend image (jpg/png)", type=["png", "jpg", "jpeg"])

    if uploaded_leads is None:
        st.info("Please upload Leads master (CSV or Excel).")
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

    with st.expander("Leads preview"):
        st.write("Rows:", len(leads_df))
        st.dataframe(leads_df.head(10))
        st.write("Columns:", list(leads_df.columns))

    # Normalize leads: detect campaign column
    possible_campaign_cols = [c for c in leads_df.columns if re.search(r"campaign|source|utm|ad", c, re.I)]
    campaign_col = st.sidebar.selectbox("Map lead campaign column (if detected)", options=["(none)"] + possible_campaign_cols, index=1 if possible_campaign_cols else 0)
    if campaign_col == "(none)":
        campaign_col = None

    if campaign_col is None:
        leads_df["campaign_mapped"] = "unknown"
    else:
        leads_df["campaign_mapped"] = leads_df[campaign_col].astype(str).fillna("unknown")

    if "lead_id" not in leads_df.columns:
        leads_df["_lead_row_id"] = range(1, len(leads_df)+1)
        lead_id_col = "_lead_row_id"
    else:
        lead_id_col = "lead_id"

    leads_df["__cmp_key"] = leads_df["campaign_mapped"].astype(str).apply(simple_key)

    # Internal duplicates by phone/email if present
    phone_col = next((c for c in leads_df.columns if re.search(r"phone|mobile|contact", c, re.I)), None)
    email_col = next((c for c in leads_df.columns if re.search(r"email", c, re.I)), None)
    leads_df["_duplicate_internal"] = False
    leads_df["_duplicate_internal_reason"] = ""

    if phone_col:
        dup_phone = leads_df.duplicated(subset=[phone_col], keep=False)
        leads_df.loc[dup_phone, "_duplicate_internal"] = True
        leads_df.loc[dup_phone, "_duplicate_internal_reason"] += f"dup_phone({phone_col});"
    if email_col:
        dup_email = leads_df.duplicated(subset=[email_col], keep=False)
        leads_df.loc[dup_email, "_duplicate_internal"] = True
        leads_df.loc[dup_email, "_duplicate_internal_reason"] += f"dup_email({email_col});"

    # Build spend_df from file or OCR image
    spend_df = pd.DataFrame(columns=["Campaign", "Leads", "CPL", "Spend"])
    spend_source = None

    if uploaded_spend_file is not None:
        try:
            if uploaded_spend_file.name.lower().endswith(".csv"):
                s_loaded = pd.read_csv(uploaded_spend_file)
            else:
                s_loaded = pd.read_excel(uploaded_spend_file)
            st.sidebar.write("Confirm spend file mapping")
            s_campaign_col = st.sidebar.selectbox("Campaign column in spend file", options=list(s_loaded.columns), index=0)
            s_amount_col = st.sidebar.selectbox("Spend amount column", options=list(s_loaded.columns), index=1 if len(s_loaded.columns)>1 else 0)
            s_leads_col = st.sidebar.selectbox("Leads count column (optional)", options=["(none)"] + list(s_loaded.columns), index=0)
            spend_df = pd.DataFrame({"Campaign": s_loaded[s_campaign_col].astype(str).fillna("unknown"), "Leads": (s_loaded[s_leads_col] if s_leads_col!="(none)" else 0), "CPL": None, "Spend": s_loaded[s_amount_col].apply(parse_spend_string)}) if s_leads_col!="(none)" else pd.DataFrame({"Campaign": s_loaded[s_campaign_col].astype(str).fillna("unknown"), "Leads": 0, "CPL": None, "Spend": s_loaded[s_amount_col].apply(parse_spend_string)})
            spend_source = "file"
        except Exception as e:
            st.sidebar.error(f"Failed to load spend file: {e}")

    # If image uploaded and no spend file, try OCR
    if uploaded_spend_image is not None and spend_source is None:
        st.sidebar.image(uploaded_spend_image, caption="Spend image (OCR source)")
        raw_text = ""
        if USE_PYTESSERACT:
            raw_text = ocr_with_pytesseract(Image.open(uploaded_spend_image))
        if (not raw_text.strip()) and OCR_SPACE_API_KEY:
            try:
                raw_text = ocr_space_image_bytes(uploaded_spend_image.getvalue(), OCR_SPACE_API_KEY)
            except Exception as e:
                st.sidebar.error(f"OCR.space error: {e}")
        st.sidebar.text_area("OCR raw text (preview)", raw_text[:8000], height=200)
        extracted = extract_table_from_text(raw_text)
        if extracted.empty:
            st.warning("OCR didn't extract clear rows. You can enter spend manually or upload a spend file.")
            extracted = pd.DataFrame(columns=["Campaign", "Leads", "CPL", "Spend"])
        st.write("### OCR-extracted spend (edit if needed)")
        edited = st.data_editor(extracted, num_rows="dynamic", use_container_width=True)
        if st.button("Use extracted spend"):
            edited["Spend"] = edited["Spend"].apply(parse_spend_string)
            edited["Leads"] = pd.to_numeric(edited["Leads"], errors="coerce").fillna(0).astype(int)
            spend_df = edited.copy()
            spend_source = "ocr"

    # Manual spend entry fallback
    if spend_source is None and uploaded_spend_file is None and uploaded_spend_image is None:
        st.sidebar.info("No spend input detected. You can upload a spend file/image or enter spend rows manually.")
        if st.sidebar.checkbox("Enter spend rows manually"):
            manual = st.data_editor(pd.DataFrame([{"Campaign":"", "Leads":0, "CPL":None, "Spend":0.0}]), num_rows="dynamic", use_container_width=True)
            if st.button("Use manual spend"):
                manual["Spend"] = manual["Spend"].apply(parse_spend_string)
                manual["Leads"] = pd.to_numeric(manual["Leads"], errors="coerce").fillna(0).astype(int)
                spend_df = manual.copy()
                spend_source = "manual"

    # Ensure spend_df columns
    if "Campaign" not in spend_df.columns:
        spend_df["Campaign"] = spend_df.iloc[:,0].astype(str) if spend_df.shape[1]>0 else ""
    if "Spend" not in spend_df.columns:
        spend_df["Spend"] = 0.0
    if "Leads" not in spend_df.columns:
        spend_df["Leads"] = 0
    spend_df["Spend"] = spend_df["Spend"].apply(parse_spend_string)
    spend_df["Leads"] = pd.to_numeric(spend_df["Leads"], errors="coerce").fillna(0).astype(int)
    spend_df["__cmp_key"] = spend_df["Campaign"].astype(str).apply(simple_key)

    # Build mapping suggestions
    lead_keys = leads_df["__cmp_key"].unique().tolist()
    suggestions = []
    for sc in spend_df["Campaign"].unique().tolist():
        sk = simple_key(sc)
        if sk in lead_keys:
            suggested = leads_df[leads_df["__cmp_key"]==sk]["campaign_mapped"].iloc[0]
        else:
            matches = get_close_matches(sk, lead_keys, n=1, cutoff=0.4)
            suggested = leads_df[leads_df["__cmp_key"]==matches[0]]["campaign_mapped"].iloc[0] if matches else None
        suggestions.append({"spend_campaign": sc, "suggested_lead_campaign": suggested})
    sugg_df = pd.DataFrame(suggestions)

    st.write("### Mapping suggestions (spend campaign -> suggested lead campaign). Edit if needed and click Apply mapping.")
    mapping_editor = st.data_editor(sugg_df, num_rows="dynamic", use_container_width=True)
    if st.button("Apply mapping"):
        # build mapping dict from user edits
        mapping_dict = {}
        for _, row in mapping_editor.iterrows():
            sk = simple_key(row["spend_campaign"])
            if pd.notna(row.get("suggested_lead_campaign")) and row.get("suggested_lead_campaign") not in [None, "None", ""]:
                mapping_dict[sk] = simple_key(row.get("suggested_lead_campaign"))
            else:
                mapping_dict[sk] = None

        # map spend rows to matched lead keys
        mapped = []
        for _, r in spend_df.iterrows():
            sk = r["__cmp_key"]
            target = mapping_dict.get(sk)
            mapped.append({"__cmp_key": target, "spend_campaign_label": r["Campaign"], "reported_leads": int(r["Leads"]), "spend_total": float(r["Spend"])})
        spend_mapped_df = pd.DataFrame(mapped)
        # create lookup
        spend_lookup = spend_mapped_df.groupby("__cmp_key").agg(reported_leads=("reported_leads","sum"), spend_total=("spend_total","sum")).reset_index()
        # merge to leads by __cmp_key
        merged = leads_df.merge(spend_lookup, on="__cmp_key", how="left")
        merged["spend_total"] = merged["spend_total"].fillna(0.0)
        merged["reported_leads"] = merged["reported_leads"].fillna(0).astype(int)
        # allocate spend per campaign rows
        campaign_row_counts = merged.groupby("__cmp_key").size().rename("campaign_row_count").reset_index()
        merged = merged.merge(campaign_row_counts, on="__cmp_key", how="left")
        merged["spend_per_lead"] = merged.apply(lambda r: (r["spend_total"]/r["campaign_row_count"]) if r["campaign_row_count"]>0 else 0.0, axis=1)

        # external mismatch detection
        lead_counts = leads_df.groupby("__cmp_key").size().reset_index(name="lead_rows_count")
        external_rows = []
        total_external_mismatch = 0
        for _, r in spend_mapped_df.iterrows():
            mk = r["__cmp_key"]
            reported = r["reported_leads"]
            actual = int(lead_counts[lead_counts["__cmp_key"]==mk]["lead_rows_count"].iloc[0]) if mk in lead_counts["__cmp_key"].values else 0
            diff = reported - actual
            external_rows.append({"spend_campaign_label": r["spend_campaign_label"], "matched_lead_key": mk, "reported_leads": reported, "actual_leads": actual, "diff": diff})
            if diff > 0:
                total_external_mismatch += diff
        external_mismatch_df = pd.DataFrame(external_rows)

        # mark external mismatch campaigns
        mismatch_keys = external_mismatch_df[external_mismatch_df["diff"]>0]["matched_lead_key"].tolist()
        merged["_external_mismatch_campaign"] = merged["__cmp_key"].isin(mismatch_keys)

        # build aggregates
        agg = merged.groupby("campaign_mapped").agg(
            leads_rows_count=("campaign_row_count", "first"),
            spend_total=("spend_total", "sum"),
            spend_per_lead_avg=("spend_per_lead", "mean"),
            reported_leads=("reported_leads", "first")
        ).reset_index()

        # attach diffs
        def get_diff_for_campaign(cm):
            key = simple_key(cm)
            row = external_mismatch_df[external_mismatch_df["matched_lead_key"]==key]
            if not row.empty:
                return int(row["reported_leads"].iloc[0]), int(row["actual_leads"].iloc[0]), int(row["diff"].iloc[0])
            return 0, 0, 0
        rep_list, act_list, diff_list = [], [], []
        for _, r in agg.iterrows():
            rep, act, diff = get_diff_for_campaign(r["campaign_mapped"])
            rep_list.append(rep); act_list.append(act); diff_list.append(diff)
        agg["reported_leads_from_spend"] = rep_list
        agg["actual_leads_in_master"] = act_list
        agg["external_diff"] = diff_list
        agg["CPL_computed"] = agg.apply(lambda r: (r["spend_total"]/r["leads_rows_count"]) if r["leads_rows_count"]>0 else 0.0, axis=1)

        # totals
        total_leads = len(leads_df)
        internal_dupes = int(leads_df["_duplicate_internal"].sum())
        totals = {
            "total_leads": int(total_leads),
            "internal_duplicates": internal_dupes,
            "external_mismatch_total": int(total_external_mismatch),
            "unique_leads_estimated": int(total_leads - internal_dupes - total_external_mismatch),
            "total_spend_reported": float(spend_df["Spend"].sum()),
            "avg_cpl_unique": float( (agg["spend_total"].sum() / (total_leads-internal_dupes)) if (total_leads-internal_dupes)>0 else 0.0)
        }

        # charts
        st.header("Campaign performance (merged)")
        st.write("Spend source:", spend_source or "(from OCR/manual/file)")
        st.dataframe(agg.sort_values("leads_rows_count", ascending=False).reset_index(drop=True))

        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots(figsize=(8,3))
            ax1.bar(agg["campaign_mapped"], agg["leads_rows_count"])
            ax1.set_title("Leads per Campaign")
            ax1.set_ylabel("Leads")
            ax1.tick_params(axis='x', rotation=45)
            st.pyplot(fig1)
        with col2:
            fig2, ax2 = plt.subplots(figsize=(8,3))
            ax2.bar(agg["campaign_mapped"], agg["spend_total"])
            ax2.set_title("Spend per Campaign")
            ax2.set_ylabel("Spend")
            ax2.tick_params(axis='x', rotation=45)
            st.pyplot(fig2)

        fig3, ax3 = plt.subplots(figsize=(10,3))
        ax3.bar(agg["campaign_mapped"], agg["CPL_computed"])
        ax3.set_title("CPL per Campaign")
        ax3.set_ylabel("CPL")
        ax3.tick_params(axis='x', rotation=45)
        st.pyplot(fig3)

        # sentiment quick heuristic from notes/status if present
        possible_notes = [c for c in leads_df.columns if re.search(r"note|status|remark|message|comment", c, re.I)]
        sent_buf = None
        if possible_notes:
            notes_col = possible_notes[0]
            notes = leads_df[notes_col].astype(str).fillna("")
            pos_k = r"\b(good|positive|interested|converted|yes)\b"
            neg_k = r"\b(bad|not interested|no|complaint|angry)\b"
            sent = []
            for t in notes:
                if re.search(pos_k, t, re.I):
                    sent.append("Positive")
                elif re.search(neg_k, t, re.I):
                    sent.append("Negative")
                else:
                    sent.append("Neutral")
            leads_df["sentiment_normalized"] = sent
            sent_counts = pd.Series(sent).value_counts()
            fig4, ax4 = plt.subplots(figsize=(4,4))
            ax4.pie(sent_counts.values, labels=sent_counts.index, autopct='%1.1f%%', startangle=140)
            st.pyplot(fig4)
            sent_buf = fig_to_image_bytes(fig4)

        # downloads
        st.subheader("Downloads")
        merged_path = "/tmp/merged_leads_with_spend_and_flags.xlsx"
        agg_path = "/tmp/campaign_aggregates_with_mismatch.xlsx"
        merged.to_excel(merged_path, index=False)
        agg.to_excel(agg_path, index=False)
        with open(merged_path, "rb") as f:
            st.download_button("Download merged leads Excel", f.read(), file_name="merged_leads_with_spend_and_flags.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with open(agg_path, "rb") as f:
            st.download_button("Download campaign aggregates", f.read(), file_name="campaign_aggregates_with_mismatch.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # PDF
        try:
            leads_buf = fig_to_image_bytes(fig1)
            spend_buf = fig_to_image_bytes(fig2)
            cpl_buf = fig_to_image_bytes(fig3)
            charts = {"leads": leads_buf, "spend": spend_buf, "cpl": cpl_buf, "sent": sent_buf}
            pdf_path = "/tmp/lead_report.pdf"
            build_pdf_report(charts, totals, agg, pdf_path)
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button("Download PDF report", pdf_bytes, file_name="campaign_report.pdf", mime="application/pdf")
        except Exception as e:
            st.warning(f"PDF generation failed: {e}")

        st.success("Merge done. Review downloads. If mapping or OCR looks off, edit and re-apply mapping.")

if __name__ == "__main__":
    main()
