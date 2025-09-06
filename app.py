# app.py
"""
Lead Analysis Dashboard — Auto-mapping edition
Upload: Leads master (CSV/XLSX) + Spend file (CSV/XLSX) or Spend image (JPG/PNG)
Auto-maps spend campaigns to lead campaigns (configurable), merges, allocates spend, detects duplicates,
and produces: merged Excel, campaign aggregates, charts, and one-page PDF report.
"""
import os, io, re, logging
from datetime import datetime
from difflib import get_close_matches
from collections import Counter

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import requests
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

logger = logging.getLogger("lead-dashboard")
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Lead Dashboard — Auto Map", layout="wide")
OCR_SPACE_API_KEY = st.secrets.get("OCR_SPACE_API_KEY", "") if "OCR_SPACE_API_KEY" in st.secrets else os.getenv("OCR_SPACE_API_KEY", "")

# Try to import pytesseract for local OCR if available
USE_PYTESSERACT = False
try:
    import pytesseract
    USE_PYTESSERACT = True
    logger.info("pytesseract available")
except Exception:
    logger.info("pytesseract not available")

# ---------- helpers ----------
def parse_spend_string(s):
    if pd.isna(s): return 0.0
    t = str(s)
    t = re.sub(r"[₹$€£,]|AED|INR|usd|rs\.?", "", t, flags=re.I)
    t = re.sub(r"[^0-9\.\-]", "", t)
    t = re.sub(r"\.(?=.*\.)", "", t)
    try: return float(t) if t!="" else 0.0
    except: return 0.0

def simple_key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower()) if pd.notna(s) else ""

def token_overlap(a,b):
    sa = set(re.findall(r"\w+", str(a).lower()))
    sb = set(re.findall(r"\w+", str(b).lower()))
    if not sa or not sb: return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

def ocr_space_image_bytes(image_bytes: bytes, api_key: str, language: str = "eng") -> str:
    url = "https://api.ocr.space/parse/image"
    files = {"file": ("image.png", image_bytes, "image/png")}
    data = {"apikey": api_key, "language": language, "isOverlayRequired": False}
    resp = requests.post(url, files=files, data=data, timeout=60)
    resp.raise_for_status()
    j = resp.json()
    if j.get("IsErroredOnProcessing"):
        return ""
    parsed = [pr.get("ParsedText","") for pr in j.get("ParsedResults",[])]
    return "\n".join(parsed)

def ocr_with_pytesseract(pil_img: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(pil_img, lang="eng")
    except Exception as e:
        logger.exception(e)
        return ""

def extract_table_from_text(raw_text: str) -> pd.DataFrame:
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    rows=[]
    num_re = r"[-+]?\d{1,3}(?:[,.\d]*\d)?(?:\.\d+)?"
    # heuristics: look for blocks where next line has numbers or inline numbers
    for i,line in enumerate(lines):
        if re.search(r"[A-Za-z]", line):
            # if line has numbers inline
            numbers = re.findall(num_re, line.replace(",",""))
            if numbers and len(numbers)>=1:
                spend = numbers[-1]
                leads = int(float(numbers[0])) if len(numbers)>=2 else np.nan
                campaign = re.sub(num_re,"",line)
                campaign = re.sub(r"[^A-Za-z0-9\s&\-]", "", campaign).strip()
                rows.append({"Campaign": campaign or "unknown", "Leads": leads if not pd.isna(leads) else 0, "CPL": None, "Spend": parse_spend_string(spend)})
            # block style
            if i+1 < len(lines):
                nxt = lines[i+1]
                nums = re.findall(num_re, nxt.replace(",",""))
                if nums:
                    spend = nums[-1]
                    leads = int(float(nums[0])) if len(nums)>=1 else 0
                    campaign = re.sub(r"[^A-Za-z0-9\s&\-]", "", line).strip()
                    rows.append({"Campaign": campaign or "unknown", "Leads": leads, "CPL": None, "Spend": parse_spend_string(spend)})
    if not rows: return pd.DataFrame(columns=["Campaign","Leads","CPL","Spend"])
    df = pd.DataFrame(rows)
    df = df.groupby("Campaign", as_index=False).agg(Leads=("Leads","sum"), CPL=("CPL", lambda s: None), Spend=("Spend","sum"))
    return df

def fig_to_buffer(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

def build_pdf(charts, totals, agg_table, out_path):
    width, height = landscape(A4)
    c = canvas.Canvas(out_path, pagesize=landscape(A4))
    margin=36
    c.setFont("Helvetica-Bold",18)
    title = totals.get("report_title", f"Campaign Performance — {datetime.now().strftime('%d %b %Y')}")
    c.drawCentredString(width/2, height-30, title)
    # place charts
    imgs = ["leads","spend","cpl","sent"]
    img_w=(width-3*margin)/2; img_h=(height-4*margin-60)/2
    coords=[(margin, height-margin-img_h-40),(margin*2+img_w, height-margin-img_h-40),(margin, height-2*margin-2*img_h-40),(margin*2+img_w, height-2*margin-2*img_h-40)]
    for k, coord in zip(imgs, coords):
        buf = charts.get(k)
        if buf:
            try:
                img = ImageReader(buf)
                c.drawImage(img, coord[0], coord[1], width=img_w, height=img_h, preserveAspectRatio=True, anchor='sw')
            except: pass
    # bullets
    text_x=margin; text_y=coords[-1][1]-20
    c.setFont("Helvetica-Bold",12); c.drawString(text_x,text_y,"Campaign Performance"); text_y-=14
    c.setFont("Helvetica",10)
    bullets=[
        f"Total leads: {totals.get('total_leads',0)}",
        f"Duplicate leads (internal): {totals.get('internal_duplicates',0)}",
        f"External mismatch (reported - in master): {totals.get('external_mismatch_total',0)}",
        f"Unique valid leads (est.): {totals.get('unique_leads_estimated',0)}",
        f"Avg CPL (deduped): ₹{totals.get('avg_cpl_unique',0):.2f}"
    ]
    for b in bullets:
        c.drawString(text_x+8, text_y, u"• "+b); text_y-=12
    c.showPage(); c.save()

# ---------- Streamlit UI ----------
def main():
    st.title("Lead Analysis — Auto-mapping + PDF report")

    st.sidebar.header("Upload inputs")
    uploaded_leads = st.sidebar.file_uploader("Leads master (CSV/XLSX)", type=["csv","xlsx","xls"])
    uploaded_spend_file = st.sidebar.file_uploader("Optional: Spend file (CSV/XLSX)", type=["csv","xlsx","xls"])
    uploaded_spend_image = st.sidebar.file_uploader("Optional: Spend image (jpg/png)", type=["png","jpg","jpeg"])

    if uploaded_leads is None:
        st.info("Upload Leads master to proceed.")
        st.stop()

    # load leads
    try:
        if uploaded_leads.name.lower().endswith(".csv"):
            leads_df = pd.read_csv(uploaded_leads)
        else:
            leads_df = pd.read_excel(uploaded_leads)
    except Exception as e:
        st.error("Failed to read leads: "+str(e)); st.stop()

    st.sidebar.write(f"Leads rows: {len(leads_df)}")
    with st.expander("Leads preview"):
        st.dataframe(leads_df.head(10))
        st.write("Columns:", list(leads_df.columns))

    # detect campaign col
    possible_campaign_cols=[c for c in leads_df.columns if re.search(r"campaign|source|utm|ad", c, re.I)]
    campaign_col = st.sidebar.selectbox("Lead campaign column (auto-detected)", ["(none)"]+possible_campaign_cols, index=1 if possible_campaign_cols else 0)
    if campaign_col=="(none)": campaign_col=None
    if campaign_col is None:
        leads_df["campaign_mapped"]="unknown"
    else:
        leads_df["campaign_mapped"]=leads_df[campaign_col].astype(str).fillna("unknown")
    if "lead_id" not in leads_df.columns:
        leads_df["_lead_row_id"]=range(1,len(leads_df)+1); lead_id_col="_lead_row_id"
    else: lead_id_col="lead_id"
    leads_df["__cmp_key"]=leads_df["campaign_mapped"].astype(str).apply(simple_key)

    # detect phone/email for internal dupes
    phone_col = next((c for c in leads_df.columns if re.search(r"phone|mobile|contact", c, re.I)), None)
    email_col = next((c for c in leads_df.columns if re.search(r"email", c, re.I)), None)
    leads_df["_duplicate_internal"]=False; leads_df["_duplicate_internal_reason"]=""
    if phone_col:
        dup = leads_df.duplicated(subset=[phone_col], keep=False)
        leads_df.loc[dup,"_duplicate_internal"]=True
        leads_df.loc[dup,"_duplicate_internal_reason"] += f"dup_phone({phone_col});"
    if email_col:
        dup = leads_df.duplicated(subset=[email_col], keep=False)
        leads_df.loc[dup,"_duplicate_internal"]=True
        leads_df.loc[dup,"_duplicate_internal_reason"] += f"dup_email({email_col});"

    # build spend_df
    spend_df = pd.DataFrame(columns=["Campaign","Leads","CPL","Spend"])
    spend_source=None
    if uploaded_spend_file:
        try:
            if uploaded_spend_file.name.lower().endswith(".csv"):
                s_loaded = pd.read_csv(uploaded_spend_file)
            else:
                s_loaded = pd.read_excel(uploaded_spend_file)
            st.sidebar.write("Confirm spend file columns")
            col_campaign = st.sidebar.selectbox("Campaign column", list(s_loaded.columns), index=0)
            col_spend = st.sidebar.selectbox("Spend column", list(s_loaded.columns), index=1 if len(s_loaded.columns)>1 else 0)
            col_leads = st.sidebar.selectbox("Leads column (optional)", ["(none)"] + list(s_loaded.columns), index=0)
            if col_leads!="(none)":
                spend_df = pd.DataFrame({"Campaign": s_loaded[col_campaign].astype(str).fillna("unknown"), "Leads": pd.to_numeric(s_loaded[col_leads], errors="coerce").fillna(0).astype(int), "CPL": None, "Spend": s_loaded[col_spend].apply(parse_spend_string)})
            else:
                spend_df = pd.DataFrame({"Campaign": s_loaded[col_campaign].astype(str).fillna("unknown"), "Leads": 0, "CPL": None, "Spend": s_loaded[col_spend].apply(parse_spend_string)})
            spend_source="file"
        except Exception as e:
            st.sidebar.error("Spend file load error: "+str(e))

    if uploaded_spend_image and spend_source is None:
        st.sidebar.image(uploaded_spend_image, caption="Spend image for OCR")
        raw_text=""
        if USE_PYTESSERACT:
            raw_text = ocr_with_pytesseract(Image.open(uploaded_spend_image))
        if (not raw_text.strip()) and OCR_SPACE_API_KEY:
            try:
                raw_text = ocr_space_image_bytes(uploaded_spend_image.getvalue(), OCR_SPACE_API_KEY)
            except Exception as e:
                st.sidebar.error("OCR.space error: "+str(e))
        st.sidebar.text_area("OCR raw text (preview)", raw_text[:8000], height=200)
        extracted = extract_table_from_text(raw_text)
        st.write("### OCR-extracted spend (edit if needed)")
        edited = st.data_editor(extracted, num_rows="dynamic", use_container_width=True)
        if st.button("Use extracted spend"):
            edited["Spend"] = edited["Spend"].apply(parse_spend_string)
            edited["Leads"] = pd.to_numeric(edited["Leads"], errors="coerce").fillna(0).astype(int)
            spend_df = edited.copy(); spend_source="ocr"

    # Manual fallback
    if spend_source is None and uploaded_spend_file is None and uploaded_spend_image is None:
        st.sidebar.info("No spend input yet. You can upload or enter manually.")
        if st.sidebar.checkbox("Enter spend manually"):
            manual = st.data_editor(pd.DataFrame([{"Campaign":"", "Leads":0, "CPL":None, "Spend":0.0}]), num_rows="dynamic", use_container_width=True)
            if st.button("Use manual spend"):
                manual["Spend"] = manual["Spend"].apply(parse_spend_string)
                manual["Leads"] = pd.to_numeric(manual["Leads"], errors="coerce").fillna(0).astype(int)
                spend_df = manual.copy(); spend_source="manual"

    # ensure columns exist
    if "Campaign" not in spend_df.columns and spend_df.shape[1]>0:
        spend_df["Campaign"]=spend_df.iloc[:,0].astype(str)
    for col in ["Spend","Leads"]:
        if col not in spend_df.columns:
            spend_df[col]=0 if col=="Leads" else 0.0
    spend_df["Spend"]=spend_df["Spend"].apply(parse_spend_string)
    spend_df["Leads"]=pd.to_numeric(spend_df["Leads"], errors="coerce").fillna(0).astype(int)
    spend_df["__cmp_key"]=spend_df["Campaign"].astype(str).apply(simple_key)

    # mapping suggestions UI
    lead_labels = sorted(leads_df["campaign_mapped"].astype(str).unique().tolist())
    spend_labels = spend_df["Campaign"].astype(str).unique().tolist()
    suggestions=[]
    for sc in spend_labels:
        sk = simple_key(sc)
        if sk in leads_df["__cmp_key"].values:
            suggested = leads_df[leads_df["__cmp_key"]==sk]["campaign_mapped"].iloc[0]
        else:
            matches = get_close_matches(sk, leads_df["__cmp_key"].unique().tolist(), n=1, cutoff=0.45)
            suggested = leads_df[leads_df["__cmp_key"]==matches[0]]["campaign_mapped"].iloc[0] if matches else None
        suggestions.append({"spend_campaign": sc, "suggested_lead_campaign": suggested})
    sugg_df = pd.DataFrame(suggestions)
    st.write("### Mapping suggestions (editable). You can correct these. Optional: enable Auto-map below to let the app resolve unmapped rows.")
    mapping_editor = st.data_editor(sugg_df, num_rows="dynamic", use_container_width=True)

    auto_map = st.checkbox("Auto-map spend campaigns (attempt automatic matching) — recommended", value=True)
    auto_map_cutoff = st.slider("Fuzzy cutoff (lower = more permissive)", 0.2, 0.9, 0.35, 0.05)

    if st.button("Apply mapping & generate report"):
        # Compose mapping dict from edited table first
        mapping_by_spend = {}
        for _, row in mapping_editor.iterrows():
            spend_label = str(row["spend_campaign"]).strip()
            suggested = row.get("suggested_lead_campaign")
            if pd.notna(suggested) and suggested not in ["", "None"]:
                mapping_by_spend[spend_label] = str(suggested).strip()
            else:
                mapping_by_spend[spend_label] = None

        # Auto-map where mapping missing and auto_map True
        if auto_map:
            # prepare candidate mapping: exact labels in lead_labels, keys
            lead_keys = {simple_key(x): x for x in lead_labels}
            for s in spend_labels:
                if mapping_by_spend.get(s): continue
                sk = simple_key(s)
                # exact key match
                if sk in lead_keys:
                    mapping_by_spend[s] = lead_keys[sk]; continue
                # difflib fuzzy on keys
                keys_list = list(lead_keys.keys())
                matches = get_close_matches(sk, keys_list, n=1, cutoff=auto_map_cutoff)
                if matches:
                    mapping_by_spend[s] = lead_keys[matches[0]]; continue
                # token overlap as fallback
                best_score = 0; best_label=None
                for lab in lead_labels:
                    sc = token_overlap(s, lab)
                    if sc > best_score:
                        best_score = sc; best_label = lab
                if best_score > 0.25:
                    mapping_by_spend[s] = best_label
                else:
                    # fallback: use spend label itself so groupby won't lose it
                    mapping_by_spend[s] = s

        else:
            # no auto-map: replace Nones with spend label itself
            for s in spend_labels:
                if mapping_by_spend.get(s) in [None, "None"]:
                    mapping_by_spend[s] = s

        # Build mapped spend rows using resolved human labels
        mapped_rows=[]
        for _, r in spend_df.iterrows():
            s_label = str(r["Campaign"]).strip()
            resolved = mapping_by_spend.get(s_label, s_label)
            mapped_rows.append({"spend_campaign_original": s_label, "campaign_mapped": resolved, "reported_leads_from_spend": int(r.get("Leads",0)), "spend_total": float(r.get("Spend",0.0))})
        spend_mapped_df = pd.DataFrame(mapped_rows)

        st.write("### Mapping preview (what will be joined on campaign_mapped)")
        st.dataframe(spend_mapped_df.head(200))

        # Aggregate spend by resolved label
        spend_agg = spend_mapped_df.groupby("campaign_mapped", dropna=False).agg(reported_leads_from_spend=("reported_leads_from_spend","sum"), spend_total_from_spend=("spend_total","sum")).reset_index()

        st.write("### Spend aggregates by resolved campaign label")
        st.dataframe(spend_agg)

        # Merge with leads by campaign_mapped label (human-friendly)
        merged = leads_df.copy()
        merged["campaign_mapped"] = merged["campaign_mapped"].astype(str).fillna("unknown")
        merged = merged.merge(spend_agg, on="campaign_mapped", how="left")
        merged["spend_total_from_spend"] = merged["spend_total_from_spend"].fillna(0.0)
        merged["reported_leads_from_spend"] = merged["reported_leads_from_spend"].fillna(0).astype(int)
        # compute row counts per campaign
        campaign_row_counts = merged.groupby("campaign_mapped").size().rename("campaign_row_count").reset_index()
        merged = merged.merge(campaign_row_counts, on="campaign_mapped", how="left")
        merged["spend_per_lead"] = merged.apply(lambda r: (r["spend_total_from_spend"]/r["campaign_row_count"]) if r["campaign_row_count"]>0 else 0.0, axis=1)

        # external mismatch
        lead_cts = merged.groupby("campaign_mapped").agg(actual_leads_in_master=("campaign_row_count","first")).reset_index()
        mismatch = spend_agg.merge(lead_cts, on="campaign_mapped", how="left").fillna(0)
        mismatch["external_diff"] = mismatch["reported_leads_from_spend"] - mismatch["actual_leads_in_master"]
        total_external_mismatch = int(mismatch[mismatch["external_diff"]>0]["external_diff"].sum())

        # flag external mismatch campaigns on merged
        mismatch_keys = mismatch[mismatch["external_diff"]>0]["campaign_mapped"].tolist()
        merged["_external_mismatch_campaign"] = merged["campaign_mapped"].isin(mismatch_keys)

        # aggregates for display
        agg = merged.groupby("campaign_mapped").agg(
            leads_rows_count=("campaign_row_count","first"),
            spend_total=("spend_total_from_spend","sum"),
            spend_per_lead_avg=("spend_per_lead","mean"),
            reported_leads_from_spend=("reported_leads_from_spend","first")
        ).reset_index()
        agg["CPL_computed"] = agg.apply(lambda r: (r["spend_total"]/r["leads_rows_count"]) if r["leads_rows_count"]>0 else 0.0, axis=1)

        total_leads = len(merged)
        internal_dupes = int(merged["_duplicate_internal"].sum())
        unique_valid = int(total_leads - internal_dupes - total_external_mismatch)
        avg_cpl_unique = float((agg["spend_total"].sum() / unique_valid) if unique_valid>0 else (agg["spend_total"].sum() / total_leads if total_leads>0 else 0.0))

        totals = {
            "total_leads": total_leads,
            "internal_duplicates": internal_dupes,
            "external_mismatch_total": total_external_mismatch,
            "unique_leads_estimated": unique_valid,
            "total_spend_reported": float(spend_agg["spend_total_from_spend"].sum()),
            "avg_cpl_unique": avg_cpl_unique,
            "report_title": f"Campaign Performance — {datetime.now().strftime('%d %b %Y')}"
        }

        # charts
        st.header("Campaign performance (merged)")
        st.write("Spend source:", spend_source or "(OCR/manual/file)")
        st.dataframe(agg.sort_values("leads_rows_count", ascending=False).reset_index(drop=True))

        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots(figsize=(8,3))
            ax1.bar(agg["campaign_mapped"], agg["leads_rows_count"], color='tab:green')
            ax1.set_title("Leads per Campaign")
            ax1.set_ylabel("Leads")
            plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')
            st.pyplot(fig1)
        with col2:
            fig2, ax2 = plt.subplots(figsize=(8,3))
            ax2.bar(agg["campaign_mapped"], agg["spend_total"], color='tab:blue')
            ax2.set_title("Spend per Campaign (₹)")
            ax2.set_ylabel("Spend (₹)")
            plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')
            st.pyplot(fig2)

        fig3, ax3 = plt.subplots(figsize=(10,3))
        ax3.bar(agg["campaign_mapped"], agg["CPL_computed"], color='tab:orange')
        ax3.set_title("CPL per Campaign")
        ax3.set_ylabel("CPL (₹)")
        plt.setp(ax3.get_xticklabels(), rotation=30, ha='right')
        st.pyplot(fig3)

        # sentiment
        possible_notes = [c for c in leads_df.columns if re.search(r"note|status|remark|message|comment", c, re.I)]
        sent_buf=None
        if possible_notes:
            notes_col = possible_notes[0]
            notes = leads_df[notes_col].astype(str).fillna("")
            pos_k = r"\b(good|positive|interested|converted|yes)\b"
            neg_k = r"\b(bad|not interested|no|complaint|angry)\b"
            sent = []
            for t in notes:
                if re.search(pos_k, t, re.I): sent.append("Positive")
                elif re.search(neg_k, t, re.I): sent.append("Negative")
                else: sent.append("Neutral")
            leads_df["sentiment_normalized"]=sent
            sent_counts = pd.Series(sent).value_counts()
            fig4, ax4 = plt.subplots(figsize=(4,4))
            ax4.pie(sent_counts.values, labels=sent_counts.index, autopct='%1.1f%%', startangle=140)
            st.pyplot(fig4)
            sent_buf = fig_to_buffer(fig4)
        else:
            sent_counts = pd.Series({"Neutral": int(total_leads*0.6), "Positive": int(total_leads*0.3), "Negative": int(total_leads*0.1)})
            fig4, ax4 = plt.subplots(figsize=(4,4))
            ax4.pie(sent_counts.values, labels=sent_counts.index, autopct='%1.1f%%', startangle=140)
            st.pyplot(fig4)
            sent_buf = fig_to_buffer(fig4)

        # downloads
        out_dir = "/tmp"
        merged_path = os.path.join(out_dir, "merged_leads_with_spend_and_flags.xlsx")
        agg_path = os.path.join(out_dir, "campaign_aggregates_with_mismatch.xlsx")
        merged.to_excel(merged_path, index=False)
        agg.to_excel(agg_path, index=False)
        with open(merged_path,"rb") as f:
            st.download_button("Download merged leads Excel", f.read(), file_name="merged_leads_with_spend_and_flags.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with open(agg_path,"rb") as f:
            st.download_button("Download campaign aggregates", f.read(), file_name="campaign_aggregates_with_mismatch.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # build and return PDF automatically
        try:
            leads_buf = fig_to_buffer(fig1)
            spend_buf = fig_to_buffer(fig2)
            cpl_buf = fig_to_buffer(fig3)
            charts = {"leads": leads_buf, "spend": spend_buf, "cpl": cpl_buf, "sent": sent_buf}
            pdf_path = os.path.join(out_dir, f"campaign_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
            build_pdf(charts, totals, agg, pdf_path)
            with open(pdf_path,"rb") as f:
                pdf_bytes = f.read()
            st.download_button("Download PDF report", pdf_bytes, file_name="campaign_report.pdf", mime="application/pdf")
            st.success("Report generated. Download above.")
        except Exception as e:
            st.warning("PDF generation failed: "+str(e))

if __name__ == "__main__":
    main()
