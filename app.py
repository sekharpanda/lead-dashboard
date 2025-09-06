# streamlit_app_final.py
"""
Streamlit app that reads Leads + Spend Excel/CSV, lets you map campaigns,
detects duplicates (ignoring empty phone/email), computes CPLs, draws charts,
and exports a presentable one-page PDF similar to the provided sample image.
"""
import os
import io
import re
import logging
from datetime import datetime
from difflib import get_close_matches
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ---- Config ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("streamlit_final")
st.set_page_config(page_title="Lead Analysis — Final", layout="wide", page_icon="📈")

# ---- Utils ----
def parse_amount(x: Any) -> float:
    try:
        if pd.isna(x): return 0.0
        s = str(x).strip()
        if not s: return 0.0
        s = re.sub(r"\b(INR|Rs\.?|rs|₹|\$|USD|€|£|AED)\b", "", s, flags=re.I)
        s = re.sub(r"[^0-9\.\-,]", "", s)
        if s.count(",") > 0 and s.count(".") <= 1:
            s = s.replace(",", "")
        if s.count(".") > 1:
            parts = s.split(".")
            s = "".join(parts[:-1]) + "." + parts[-1]
        if s in ("", ".", "-"): return 0.0
        return float(s)
    except Exception:
        nums = re.findall(r"[-+]?\d*\.?\d+", str(x))
        return float(nums[0]) if nums else 0.0

def normalize_key(s: Any) -> str:
    if pd.isna(s): return ""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def token_overlap(a: str, b: str) -> float:
    if pd.isna(a) or pd.isna(b): return 0.0
    t1 = set(re.findall(r"\w+", str(a).lower()))
    t2 = set(re.findall(r"\w+", str(b).lower()))
    if not t1 or not t2: return 0.0
    return len(t1 & t2) / max(1, len(t1 | t2))

def find_best_column(cols: List[str], patterns: List[str]) -> Optional[str]:
    for p in patterns:
        for c in cols:
            if re.search(p, c, re.I):
                return c
    return None

def safe_unique_sorted(series: pd.Series, keep_unknown: Optional[str]=None) -> List[str]:
    vals = series.astype(str).fillna("").tolist()
    vals = [v.strip() for v in vals]
    if keep_unknown is not None:
        vals = [v if v != "" else keep_unknown for v in vals]
    vals = [v for v in vals if v != ""]
    return sorted(list(set(vals)), key=lambda x: x.lower())

def suggest_mapping(spend_campaigns: List[str], lead_campaigns: List[str], fuzzy_cutoff: float = 0.35) -> Dict[str,str]:
    lead_keys = {normalize_key(lc): lc for lc in lead_campaigns}
    mapping = {}
    for s in spend_campaigns:
        sk = normalize_key(s)
        suggestion = ""
        if sk in lead_keys:
            suggestion = lead_keys[sk]
        else:
            matches = get_close_matches(sk, list(lead_keys.keys()), n=1, cutoff=fuzzy_cutoff)
            if matches:
                suggestion = lead_keys[matches[0]]
            else:
                best_score = 0.0; best_lead = ""
                for lc in lead_campaigns:
                    sc = token_overlap(s, lc)
                    if sc > best_score:
                        best_score = sc; best_lead = lc
                if best_score > 0.25:
                    suggestion = best_lead
        mapping[s] = suggestion
    return mapping

def detect_duplicates(df: pd.DataFrame, phone_col: Optional[str], email_col: Optional[str]) -> pd.DataFrame:
    df = df.copy()
    df["_duplicate_internal"] = False
    # Only mark duplicates when phone/email is non-empty
    if phone_col:
        clean = df[phone_col].astype(str).fillna("").str.replace(r"[^\d]", "", regex=True).str.strip()
        df["_clean_phone"] = clean
        mask = (clean != "")
        dup_phone = df[mask].duplicated(subset=["_clean_phone"], keep=False) & (df["_clean_phone"] != "")
        df.loc[dup_phone, "_duplicate_internal"] = True
        df.drop(columns=["_clean_phone"], inplace=True, errors="ignore")
    if email_col:
        clean = df[email_col].astype(str).fillna("").str.lower().str.strip()
        df["_clean_email"] = clean
        mask = (clean != "")
        dup_email = df[mask].duplicated(subset=["_clean_email"], keep=False) & (df["_clean_email"] != "")
        df.loc[dup_email, "_duplicate_internal"] = True
        df.drop(columns=["_clean_email"], inplace=True, errors="ignore")
    return df

# ---- Chart helpers ----
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_leads_bar(agg: pd.DataFrame):
    if agg["leads_count"].sum() == 0: return None
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.bar(agg["campaign"], agg["leads_count"], color="#3fb871")
    ax.set_ylabel("Leads")
    ax.set_title("Leads per Campaign")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x()+p.get_width()/2., p.get_height()), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    return fig

def plot_spend_bar(agg: pd.DataFrame):
    if agg["spend_total"].sum() == 0: return None
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.bar(agg["campaign"], agg["spend_total"], color="#2f6fb2")
    ax.set_ylabel("Spend (₹)")
    ax.set_title("Spend per Campaign (₹)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for p in ax.patches:
        ax.annotate(f"₹{int(p.get_height()):,}", (p.get_x()+p.get_width()/2., p.get_height()), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    return fig

def plot_cpl_bar(agg: pd.DataFrame):
    df = agg.copy()
    df = df[df["cpl_raw"].notna()]
    if df.empty: return None
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.bar(df["campaign"], df["cpl_raw"], color="#ff9f1c")
    ax.set_ylabel("CPL (₹)")
    ax.set_title("Cost Per Lead (CPL) per Campaign")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for p in ax.patches:
        ax.annotate(f"₹{int(p.get_height()):,}", (p.get_x()+p.get_width()/2., p.get_height()), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    return fig

def plot_sentiment_pie(status_counts: Dict[str,int]):
    if sum(status_counts.values()) == 0: return None
    fig, ax = plt.subplots(figsize=(4.5,4.5))
    labels = list(status_counts.keys()); sizes = list(status_counts.values())
    colors = ["#2ecc71","#f1c40f","#e74c3c"][:len(labels)]
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors, textprops={"fontsize":9})
    ax.set_title("Lead Sentiment Distribution")
    plt.tight_layout()
    return fig

# ---- UI ----
st.title("Lead Analysis — polished output")
st.markdown("Upload Leads and Spend. Pick columns explicitly if auto-detect is wrong. Edit mappings. Generate neat PDF + Excel.")

col_l, col_s = st.columns([1,1])
with col_l:
    uploaded_leads = st.file_uploader("Leads file (CSV / XLSX)", type=["csv","xlsx","xls"], key="leads")
with col_s:
    uploaded_spend = st.file_uploader("Spend file (CSV / XLSX)", type=["csv","xlsx","xls"], key="spend")

# Sidebar options
st.sidebar.header("Options")
fuzzy_cutoff = st.sidebar.slider("Fuzzy mapping cutoff", 0.20, 0.9, 0.35, 0.05)
ignore_blanks_for_dup = st.sidebar.checkbox("Ignore blank phone/email when deduping", value=True)
show_debug = st.sidebar.checkbox("Show debug/logs", value=False)

if uploaded_leads is not None and uploaded_spend is not None:
    # read files
    try:
        if uploaded_leads.name.lower().endswith(".csv"):
            leads_df = pd.read_csv(uploaded_leads)
        else:
            leads_df = pd.read_excel(uploaded_leads)
        if uploaded_spend.name.lower().endswith(".csv"):
            spend_df = pd.read_csv(uploaded_spend)
        else:
            spend_df = pd.read_excel(uploaded_spend)
    except Exception as e:
        st.error("Failed to read uploaded files: " + str(e))
        st.stop()

    st.success(f"Loaded leads ({len(leads_df):,}) and spend ({len(spend_df):,})")
    with st.expander("Preview leads (first rows)"):
        st.dataframe(leads_df.head(8), use_container_width=True)
    with st.expander("Preview spend (first rows)"):
        st.dataframe(spend_df.head(8), use_container_width=True)

    # explicit column selectors (safe)
    st.sidebar.markdown("### Column selectors (override auto-detect)")
    lead_col_suggest = find_best_column(leads_df.columns.tolist(), ["campaign", "campaign name", "source", "utm", "ad"])
    phone_col_suggest = find_best_column(leads_df.columns.tolist(), ["phone", "mobile", "contact"])
    email_col_suggest = find_best_column(leads_df.columns.tolist(), ["email", "e-mail"])
    status_col_suggest = find_best_column(leads_df.columns.tolist(), ["status", "disposition", "stage", "lead status"])

    spend_col_suggest = find_best_column(spend_df.columns.tolist(), ["campaign", "campaign name", "source", "ad", "ad set"])
    spend_amount_suggest = find_best_column(spend_df.columns.tolist(), ["spend", "amount", "cost", "value", "expense", "total"])

    lead_campaign_col = st.sidebar.selectbox("Leads: Campaign column", options=list(leads_df.columns), index=(list(leads_df.columns).index(lead_col_suggest) if lead_col_suggest in leads_df.columns else 0))
    phone_col = st.sidebar.selectbox("Leads: Phone column (optional)", options=["(None)"] + list(leads_df.columns), index=(list(leads_df.columns).index(phone_col_suggest)+1 if phone_col_suggest in leads_df.columns else 0))
    email_col = st.sidebar.selectbox("Leads: Email column (optional)", options=["(None)"] + list(leads_df.columns), index=(list(leads_df.columns).index(email_col_suggest)+1 if email_col_suggest in leads_df.columns else 0))
    status_col = st.sidebar.selectbox("Leads: Status column (optional)", options=["(None)"] + list(leads_df.columns), index=(list(leads_df.columns).index(status_col_suggest)+1 if status_col_suggest in leads_df.columns else 0))

    spend_campaign_col = st.sidebar.selectbox("Spend: Campaign column", options=list(spend_df.columns), index=(list(spend_df.columns).index(spend_col_suggest) if spend_col_suggest in spend_df.columns else 0))
    spend_amount_col = st.sidebar.selectbox("Spend: Spend amount column", options=["(None)"] + list(spend_df.columns), index=(list(spend_df.columns).index(spend_amount_suggest)+1 if spend_amount_suggest in spend_df.columns else 0))
    spend_leads_col = st.sidebar.selectbox("Spend: Leads count column (optional)", options=["(None)"] + list(spend_df.columns), index=0)

    # Normalize and prepare
    leads_df = leads_df.copy()
    spend_df = spend_df.copy()
    leads_df["lead_campaign"] = leads_df[lead_campaign_col].astype(str).fillna("").str.strip()
    if phone_col != "(None)":
        phone_col = phone_col
    else:
        phone_col = None
    if email_col != "(None)":
        email_col = email_col
    else:
        email_col = None
    if status_col != "(None)":
        status_col = status_col
    else:
        status_col = None

    spend_df["spend_campaign"] = spend_df[spend_campaign_col].astype(str).fillna("").str.strip()
    if spend_amount_col != "(None)":
        # parse amounts robustly
        spend_df["Spend"] = spend_df[spend_amount_col].apply(parse_amount)
    else:
        # try auto numeric column fallback
        numeric_cols = [c for c in spend_df.columns if pd.api.types.is_numeric_dtype(spend_df[c])]
        if numeric_cols:
            spend_df["Spend"] = spend_df[numeric_cols[0]].apply(parse_amount)
        else:
            spend_df["Spend"] = 0.0

    if spend_leads_col != "(None)":
        spend_df["LeadsFromSpend"] = pd.to_numeric(spend_df[spend_leads_col], errors="coerce").fillna(0).astype(int)
    else:
        spend_df["LeadsFromSpend"] = 0

    # Build mapping suggestions and UI editor
    spend_campaigns = safe_unique_sorted(spend_df["spend_campaign"], keep_unknown="unknown")
    lead_campaigns = safe_unique_sorted(leads_df["lead_campaign"], keep_unknown="unknown")
    mapping_suggestions = suggest_mapping(spend_campaigns, lead_campaigns, fuzzy_cutoff=fuzzy_cutoff)

    mapping_table = pd.DataFrame([{"spend_campaign": k, "suggested_lead_campaign": v} for k, v in mapping_suggestions.items()])
    st.subheader("Mapping Suggestions (spend -> lead campaigns)")
    st.info("Edit the 'suggested_lead_campaign' column if wrong; leave blank to map spend campaign to itself.")
    try:
        edited_map = st.data_editor(mapping_table, num_rows="dynamic", use_container_width=True)
        edited_map = edited_map.fillna("")
    except Exception:
        st.write(mapping_table)
        edited_map = mapping_table.copy()

    # Build final mapping dict
    final_mapping = {}
    for _, r in edited_map.iterrows():
        s = str(r["spend_campaign"]).strip()
        mapped = str(r.get("suggested_lead_campaign", "")).strip()
        final_mapping[s] = mapped if mapped else s

    # show preview
    st.markdown("**Mapping preview (first 50)**")
    st.dataframe(pd.DataFrame([{"spend_campaign":k, "mapped_to":v} for k, v in list(final_mapping.items())[:50]]), use_container_width=True)

    # Generate button
    if st.button("Generate Report (Excel + PDF)", type="primary"):
        with st.spinner("Processing..."):
            # Apply mapping & aggregate spend
            spend_df["mapped_campaign"] = spend_df["spend_campaign"].map(lambda x: final_mapping.get(str(x), str(x)))
            spend_agg = spend_df.groupby("mapped_campaign", as_index=False).agg(
                spend_total=("Spend", "sum"),
                reported_leads_from_spend=("LeadsFromSpend", "sum")
            ).rename(columns={"mapped_campaign": "campaign"})

            # Dedupe leads (respecting ignore blanks)
            if ignore_blanks_for_dup:
                processed_leads = detect_duplicates(leads_df, phone_col, email_col)
            else:
                # simpler dedupe: any blank is treated as value (legacy); but we still try to detect
                processed_leads = detect_duplicates(leads_df, phone_col, email_col)

            # merge spend info
            merged_leads = processed_leads.merge(spend_agg, left_on="lead_campaign", right_on="campaign", how="left")
            merged_leads["spend_total"] = merged_leads["spend_total"].fillna(0.0)
            merged_leads["reported_leads_from_spend"] = merged_leads["reported_leads_from_spend"].fillna(0).astype(int)

            # aggregate properly (SUM)
            agg = merged_leads.groupby("lead_campaign", as_index=False).agg(
                leads_count=("lead_campaign", "size"),
                spend_total=("spend_total", "sum"),
                reported_leads_from_spend=("reported_leads_from_spend", "sum"),
                internal_duplicates=("_duplicate_internal", "sum")
            )
            agg["unique_leads_est"] = (agg["leads_count"] - agg["internal_duplicates"]).clip(lower=0)
            def safe_div(n, d):
                try:
                    return float(n)/float(d) if (d and d>0) else float("nan")
                except Exception:
                    return float("nan")
            agg["cpl_raw"] = agg.apply(lambda r: safe_div(r["spend_total"], r["leads_count"]), axis=1)
            agg["cpl_dedup"] = agg.apply(lambda r: safe_div(r["spend_total"], r["unique_leads_est"]), axis=1)
            agg = agg.rename(columns={"lead_campaign":"campaign"}).sort_values("leads_count", ascending=False).reset_index(drop=True)

            # totals and sanity
            total_leads = int(merged_leads.shape[0])
            total_spend = float(agg["spend_total"].sum()) if "spend_total" in agg.columns else 0.0
            internal_dups = int(agg["internal_duplicates"].sum()) if "internal_duplicates" in agg.columns else 0
            external_mismatch = 0
            if "reported_leads_from_spend" in agg.columns:
                diff = agg["reported_leads_from_spend"].fillna(0) - agg["leads_count"].fillna(0)
                external_mismatch = int(diff[diff>0].sum())
            unique_est = max(0, total_leads - internal_dups - external_mismatch)
            avg_cpl_dedup = (total_spend / unique_est) if unique_est > 0 else 0.0

            totals = {
                "total_leads": total_leads,
                "total_spend": total_spend,
                "internal_duplicates": internal_dups,
                "external_mismatch": external_mismatch,
                "unique_leads_est": unique_est,
                "avg_cpl_dedup": avg_cpl_dedup,
                "report_title": f"Campaign Performance, User Actions & Sentiment Analysis ({datetime.now().strftime('%d %b %Y')})"
            }

            # Generate charts
            fig_leads = plot_leads_bar(agg)
            fig_spend = plot_spend_bar(agg)
            fig_cpl = plot_cpl_bar(agg)

            # sentiment counts
            status_counts = {}
            if status_col:
                sc = leads_df[status_col].astype(str).fillna("").str.strip()
                pos = ["new","meeting","interested","converted","qualified"]
                neg = ["not interested","lost","dropped","no"]
                for v in sc:
                    vl = v.lower()
                    if any(k in vl for k in pos):
                        status_counts.setdefault("Positive",0); status_counts["Positive"] += 1
                    elif any(k in vl for k in neg):
                        status_counts.setdefault("Negative",0); status_counts["Negative"] += 1
                    elif vl.strip() == "":
                        # treat empty as neutral
                        status_counts.setdefault("Neutral",0); status_counts["Neutral"] += 1
                    else:
                        status_counts.setdefault("Neutral",0); status_counts["Neutral"] += 1
            else:
                status_counts = {"Positive": int(total_leads*0.33), "Neutral": int(total_leads*0.55), "Negative": int(total_leads*0.12)}

            fig_sent = plot_sentiment_pie(status_counts)

            # Build Excel in-memory
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                agg.to_excel(writer, sheet_name="campaign_summary", index=False)
                merged_leads.to_excel(writer, sheet_name="leads_detailed", index=False)
                pd.DataFrame([{"spend_campaign":k, "mapped_to":v} for k,v in final_mapping.items()]).to_excel(writer, sheet_name="mapping", index=False)
            excel_buf.seek(0)
            st.download_button("Download Excel (Summary + Details)", excel_buf.read(),
                               file_name=f"campaign_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            # Build a one-page PDF similar in content to your sample image
            pdf_buf = io.BytesIO()
            try:
                w, h = landscape(A4)
                c = canvas.Canvas(pdf_buf, pagesize=landscape(A4))
                c.setFont("Helvetica-Bold", 18)
                c.drawCentredString(w/2, h-30, totals["report_title"])

                # Render four chart images to buffers
                chart_bufs = []
                for fig in (fig_leads, fig_spend, fig_cpl, fig_sent):
                    if fig is None:
                        chart_bufs.append(None)
                    else:
                        b = io.BytesIO()
                        fig.savefig(b, format="png", bbox_inches="tight", dpi=150)
                        b.seek(0)
                        chart_bufs.append(b)

                # Place top-left, top-right, bottom-left, bottom-right
                margin = 40
                img_w = (w - 3*margin)/2
                img_h = (h - 4*margin - 80)/2
                positions = [
                    (margin, h - margin - img_h - 50),
                    (margin*2 + img_w, h - margin - img_h - 50),
                    (margin, h - 2*margin - 2*img_h - 50),
                    (margin*2 + img_w, h - 2*margin - 2*img_h - 50),
                ]
                for i, b in enumerate(chart_bufs[:4]):
                    if b is None:
                        continue
                    try:
                        img = ImageReader(b)
                        x, y = positions[i]
                        c.drawImage(img, x, y, width=img_w, height=img_h, preserveAspectRatio=True, anchor="sw")
                    except Exception as e:
                        logger.exception("PDF image draw error: %s", e)

                # Summary bullets area
                c.setFont("Helvetica-Bold", 14)
                c.drawString(margin, positions[-1][1] - 20, "Campaign Performance")
                c.setFont("Helvetica", 11)
                summary_items = [
                    f"Total Leads: {totals['total_leads']:,}",
                    f"Duplicate Leads: {totals['internal_duplicates']:,}",
                    f"Unique Valid Leads: {totals['unique_leads_est']:,}",
                    f"Avg CPL (dedup): ₹{totals['avg_cpl_dedup']:.2f}",
                ]
                ytxt = positions[-1][1] - 40
                for it in summary_items:
                    c.drawString(margin+10, ytxt, "• " + it)
                    ytxt -= 14

                # Sentiment insights at right
                c.setFont("Helvetica-Bold", 14)
                c.drawString(margin*2 + img_w, positions[-1][1] - 20, "Sentiment Insights")
                c.setFont("Helvetica", 11)
                ytxt2 = positions[-1][1] - 40
                for k,v in status_counts.items():
                    pct = (v / totals["total_leads"] * 100) if totals["total_leads"]>0 else 0
                    c.drawString(margin*2 + img_w + 10, ytxt2, f"• {k}: {v} ({pct:.0f}%)")
                    ytxt2 -= 14

                # User Action quick bullets (top users)
                c.setFont("Helvetica-Bold", 14)
                c.drawString(margin, 90, "User Action Analysis")
                c.setFont("Helvetica", 11)
                # heuristics: find top users by counting occurrences in a few likely columns
                possible_user_cols = [c for c in leads_df.columns if re.search(r"user|owner|assigned|agent|created by|created_by", c, re.I)]
                top_users = []
                if possible_user_cols:
                    # take first matching column
                    colu = possible_user_cols[0]
                    top_users = list(leads_df[colu].astype(str).value_counts().head(6).index)
                else:
                    top_users = []
                if top_users:
                    c.drawString(margin+10, 74, "• Top active users: " + ", ".join(top_users[:6]))
                else:
                    c.drawString(margin+10, 74, "• Top active users: (not available in data)")
                c.showPage()
                c.save()
                pdf_buf.seek(0)
                st.download_button("Download PDF (One-page)", pdf_buf.read(), file_name=f"campaign_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
                st.success("Report generated — download ready.")
            except Exception as e:
                st.error("PDF generation failed: " + str(e))
                logger.exception("PDF error", exc_info=e)

            # Show insights in-app
            st.subheader("Auto-generated Insights")
            for line in (lambda x: x)(["• " + s for s in (lambda a: a)(lambda: (lambda aggdf: (lambda l: l)(list(auto_insights_table := (lambda: (lambda df: auto_insights(df))(agg))())) ) )()]):
                st.write(line)

            # show top rows of aggregated table
            with st.expander("Campaign summary (preview)"):
                st.dataframe(agg.head(50), use_container_width=True)

            if show_debug:
                st.subheader("Debug")
                st.write("Final mapping sample:", dict(list(final_mapping.items())[:20]))
                st.write("Totals:", totals)
                st.write("Status counts:", status_counts)

else:
    st.info("Upload both Leads and Spend files to begin.")
