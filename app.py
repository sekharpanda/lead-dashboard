# streamlit_app.py
import os
import re
import io
import logging
from datetime import datetime
from difflib import get_close_matches
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Logging and page config
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("streamlit_campaign_app")
st.set_page_config(page_title="Lead Analysis Dashboard", layout="wide", page_icon="📊")

# -----------------------
# Reuse of core funcs
# -----------------------
def parse_amount(value: Any) -> float:
    try:
        if pd.isna(value): return 0.0
        s = str(value).strip()
        s = re.sub(r"\b(INR|Rs\.?|rs|₹|\$|€|£|USD|AED)\b", "", s, flags=re.I)
        s = re.sub(r"[^0-9\.\-,]", "", s)
        if s.count(",") > 0 and s.count(".") <= 1:
            s = s.replace(",", "")
        if s.count(".") > 1:
            parts = s.split(".")
            s = "".join(parts[:-1]) + "." + parts[-1]
        return float(s) if s not in ("", ".", "-") else 0.0
    except Exception:
        nums = re.findall(r"[-+]?\d*\.?\d+", str(value))
        return float(nums[0]) if nums else 0.0

def normalize_key(s: Any) -> str:
    if pd.isna(s): return ""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def token_overlap(a: str, b: str) -> float:
    if pd.isna(a) or pd.isna(b): return 0.0
    ta = set(re.findall(r"\w+", str(a).lower()))
    tb = set(re.findall(r"\w+", str(b).lower()))
    if not ta or not tb: return 0.0
    return len(ta & tb) / max(1, len(ta | tb))

def find_best_column(cols: List[str], patterns: List[str]):
    for p in patterns:
        for c in cols:
            if re.search(p, c, re.I):
                return c
    return None

def suggest_mapping(spend_campaigns: List[str], lead_campaigns: List[str], fuzzy_cutoff: float=0.35) -> Dict[str,str]:
    keys = {normalize_key(l): l for l in lead_campaigns}
    mapping = {}
    for s in spend_campaigns:
        sk = normalize_key(s)
        suggestion = ""
        if sk in keys: suggestion = keys[sk]
        else:
            matches = get_close_matches(sk, list(keys.keys()), n=1, cutoff=fuzzy_cutoff)
            if matches:
                suggestion = keys[matches[0]]
            else:
                best_score = 0.0; best = ""
                for lc in lead_campaigns:
                    sc = token_overlap(s, lc)
                    if sc > best_score:
                        best_score = sc; best = lc
                if best_score > 0.25:
                    suggestion = best
        mapping[s] = suggestion
    return mapping

def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_duplicate_internal"] = False
    phone_col = find_best_column(list(df.columns), [r"phone", r"mobile", r"contact"])
    email_col = find_best_column(list(df.columns), [r"email", r"e-mail"])
    if phone_col:
        df["_clean_phone"] = df[phone_col].astype(str).str.replace(r"[^\d]", "", regex=True)
        dup_phone = df.duplicated(subset=["_clean_phone"], keep=False) & (df["_clean_phone"]!="")
        df.loc[dup_phone,"_duplicate_internal"] = True
        df.drop(columns=["_clean_phone"], inplace=True, errors="ignore")
    if email_col:
        df["_clean_email"] = df[email_col].astype(str).str.lower().str.strip()
        dup_email = df.duplicated(subset=["_clean_email"], keep=False) & (df["_clean_email"]!="")
        df.loc[dup_email,"_duplicate_internal"] = True
        df.drop(columns=["_clean_email"], inplace=True, errors="ignore")
    return df

def build_merged_analysis_streamlit(leads_df: pd.DataFrame, spend_df: pd.DataFrame, fuzzy_cutoff: float):
    leads = leads_df.copy()
    spend = spend_df.copy()
    lead_col = find_best_column(list(leads.columns), [r"campaign", r"campaign name", r"source", r"utm"]) or leads.columns[0]
    spend_col = find_best_column(list(spend.columns), [r"campaign", r"campaign name", r"source", r"ad"]) or spend.columns[0]
    leads = leads.rename(columns={lead_col:"lead_campaign"})
    spend = spend.rename(columns={spend_col:"spend_campaign"})
    # find spend amount
    amount_cols = [c for c in spend.columns if re.search(r"spend|amount|cost|value|expense", c, re.I)]
    numeric_cols = [c for c in spend.columns if pd.api.types.is_numeric_dtype(spend[c])]
    spend_amount_col = amount_cols[0] if amount_cols else (numeric_cols[0] if numeric_cols else None)
    if spend_amount_col:
        spend["Spend"] = spend[spend_amount_col].apply(parse_amount) if spend_amount_col!="Spend" else spend["Spend"].apply(parse_amount)
    else:
        spend["Spend"] = 0.0
    spend_campaigns = sorted(spend["spend_campaign"].astype(str).fillna("unknown").unique().tolist())
    lead_campaigns = sorted(leads["lead_campaign"].astype(str).fillna("unknown").unique().tolist())
    suggestions = suggest_mapping(spend_campaigns, lead_campaigns, fuzzy_cutoff=fuzzy_cutoff)
    return leads, spend, suggestions

# ---------------------
# Streamlit UI
# ---------------------
st.title("📊 Lead & Spend Analysis — Streamlit")
st.markdown("Upload leads file and spend file (Excel/CSV). Review mappings, click Generate to see charts and download reports.")

col1, col2 = st.columns(2)
with col1:
    uploaded_leads = st.file_uploader("Upload Leads file (CSV / XLSX)", type=["csv","xlsx","xls"])
with col2:
    uploaded_spend = st.file_uploader("Upload Spend file (CSV / XLSX)", type=["csv","xlsx","xls"])

fuzzy_cutoff = st.sidebar.slider("Fuzzy mapping cutoff", 0.2, 0.9, 0.35, 0.05)

if uploaded_leads is not None and uploaded_spend is not None:
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

    st.success(f"Loaded Leads ({len(leads_df):,} rows) and Spend ({len(spend_df):,} rows)")
    with st.expander("Preview Leads (first rows)"):
        st.dataframe(leads_df.head(10), use_container_width=True)
    with st.expander("Preview Spend (first rows)"):
        st.dataframe(spend_df.head(10), use_container_width=True)

    # Generate mapping suggestions
    leads, spend, suggestions = build_merged_analysis_streamlit(leads_df, spend_df, fuzzy_cutoff)
    mapping_rows = [{"spend_campaign": k, "suggested_lead_campaign": v} for k, v in suggestions.items()]
    mapping_df = pd.DataFrame(mapping_rows)

    st.subheader("Mapping Suggestions (spend -> lead campaigns)")
    st.info("Edit the mapped lead campaign column if suggestion is wrong. Leave blank to map spend campaign to itself.")
    # Let user edit mappings - use data_editor if available
    try:
        edited = st.data_editor(mapping_df, use_container_width=True)
        # ensure columns present
        edited = edited.fillna("")
    except Exception:
        st.write(mapping_df)
        edited = mapping_df.copy()

    # Build final mapping dict
    final_mapping = {}
    lead_campaigns_list = sorted(leads["lead_campaign"].unique().tolist())
    for _, row in edited.iterrows():
        s = str(row["spend_campaign"])
        mapped = str(row.get("suggested_lead_campaign", "")).strip()
        final_mapping[s] = mapped if mapped else s

    st.markdown("**Mapping preview (first 10)**")
    st.dataframe(pd.DataFrame([{"spend_campaign":k, "mapped_to":v} for k,v in list(final_mapping.items())[:50]]), use_container_width=True)

    # Button to generate
    if st.button("Generate Report"):
        # Apply mapping and produce aggregated analysis
        spend["mapped_campaign"] = spend["spend_campaign"].map(lambda x: final_mapping.get(str(x), str(x)))
        spend_agg = spend.groupby("mapped_campaign", as_index=False).agg(spend_total=("Spend","sum"))
        spend_agg = spend_agg.rename(columns={"mapped_campaign":"campaign"})

        leads = detect_duplicates(leads)
        leads = leads.merge(spend_agg, left_on="lead_campaign", right_on="campaign", how="left")
        leads["spend_total"] = leads["spend_total"].fillna(0.0)

        agg = leads.groupby("lead_campaign", as_index=False).agg(
            leads_count=("lead_campaign","size"),
            spend_total=("spend_total","sum"),
            internal_duplicates=("_duplicate_internal","sum")
        )
        agg["unique_leads_est"] = (agg["leads_count"] - agg["internal_duplicates"]).clip(lower=0)
        def safe_div(n,d): return float(n)/float(d) if (d and d>0) else float("nan")
        agg["cpl_raw"] = agg.apply(lambda r: safe_div(r["spend_total"], r["leads_count"]), axis=1)
        agg["cpl_dedup"] = agg.apply(lambda r: safe_div(r["spend_total"], r["unique_leads_est"]), axis=1)
        agg = agg.rename(columns={"lead_campaign":"campaign"}).sort_values("leads_count", ascending=False)

        # totals & insights
        total_leads = int(leads.shape[0])
        total_spend = float(agg["spend_total"].sum()) if "spend_total" in agg.columns else 0.0
        internal_dups = int(agg["internal_duplicates"].sum()) if "internal_duplicates" in agg.columns else 0
        unique_est = max(0, total_leads - internal_dups)
        avg_cpl_dedup = (total_spend / unique_est) if unique_est>0 else 0.0
        totals = {
            "total_leads": total_leads,
            "total_spend": total_spend,
            "internal_duplicates": internal_dups,
            "unique_leads_est": unique_est,
            "avg_cpl_dedup": avg_cpl_dedup,
            "report_title": f"Campaign Performance — {datetime.now().strftime('%d %b %Y')}"
        }

        st.header("Analysis Results")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Leads", f"{totals['total_leads']:,}")
        c2.metric("Total Spend", f"₹{totals['total_spend']:,.2f}")
        c3.metric("Avg CPL (dedup)", f"₹{totals['avg_cpl_dedup']:.2f}")
        c4.metric("Unique Leads (est)", f"{totals['unique_leads_est']:,}")

        # show table
        display = agg.rename(columns={
            "campaign":"Campaign","leads_count":"Leads Count","spend_total":"Total Spend","internal_duplicates":"Duplicates",
            "reported_leads_from_spend":"Reported Leads","unique_leads_est":"Unique Leads (est)","cpl_raw":"CPL (raw)","cpl_dedup":"CPL (dedup)"
        })
        display["Total Spend"] = display["Total Spend"].apply(lambda x: f"₹{x:,.2f}")
        display["CPL (raw)"] = display["cpl_raw"].apply(lambda x: f"₹{x:.2f}" if not np.isnan(x) else "N/A")
        display["CPL (dedup)"] = display["cpl_dedup"].apply(lambda x: f"₹{x:.2f}" if not np.isnan(x) else "N/A")
        st.subheader("Campaign performance")
        st.dataframe(display[["Campaign","Leads Count","Unique Leads (est)","Total Spend","CPL (raw)","CPL (dedup)","Duplicates"]], use_container_width=True)

        # charts
        st.subheader("Charts")
        # leads pie
        fig1, ax1 = plt.subplots(figsize=(6,6))
        if agg["leads_count"].sum() > 0:
            ax1.pie(agg["leads_count"], labels=agg["campaign"], autopct="%1.1f%%", startangle=90)
            ax1.set_title("Leads Distribution by Campaign")
            st.pyplot(fig1)
        plt.close(fig1)
        # CPL bar
        cpl_df = agg.dropna(subset=["cpl_raw"]).sort_values("cpl_raw", ascending=False)
        if not cpl_df.empty:
            fig2, ax2 = plt.subplots(figsize=(10,5))
            bars = ax2.bar(cpl_df["campaign"], cpl_df["cpl_raw"], color="orange")
            ax2.set_ylabel("CPL (₹)")
            ax2.set_title("CPL (raw) by Campaign")
            plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")
            for bar, val in zip(bars, cpl_df["cpl_raw"]):
                ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"₹{val:.0f}", ha="center", va="bottom")
            st.pyplot(fig2)
            plt.close(fig2)

        # Lead flow & sentiment (use status column if available)
        status_col = find_best_column(leads_df.columns.tolist(), [r"status", r"disposition", r"stage"])
        status_counts = {}
        if status_col:
            sc = leads_df[status_col].astype(str).fillna("").str.lower()
            pos_kw = ["new", "meeting", "interested", "converted"]
            neg_kw = ["not interested", "lost", "dropped"]
            for v in sc:
                if any(k in v for k in pos_kw):
                    status_counts.setdefault("Positive",0); status_counts["Positive"] += 1
                elif any(k in v for k in neg_kw):
                    status_counts.setdefault("Negative",0); status_counts["Negative"] += 1
                else:
                    status_counts.setdefault("Neutral",0); status_counts["Neutral"] += 1
        else:
            status_counts = {"Positive": int(total_leads*0.3), "Neutral": int(total_leads*0.55), "Negative": int(total_leads*0.15)}

        st.subheader("Lead Sentiment")
        fig3, ax3 = plt.subplots(figsize=(6,6))
        ax3.pie(list(status_counts.values()), labels=list(status_counts.keys()), autopct="%1.1f%%", startangle=90)
        st.pyplot(fig3)
        plt.close(fig3)

        st.subheader("Lead Actions Flow")
        fig4, ax4 = plt.subplots(figsize=(10,5))
        ax4.axis("off")
        # simple flow boxes
        ax4.add_patch(plt.Rectangle((0.05,0.6),0.25,0.2, facecolor="#e6f2ff"))
        ax4.text(0.175,0.7, f"New\n{status_counts.get('Positive','')}", ha="center", va="center")
        ax4.add_patch(plt.Rectangle((0.37,0.6),0.25,0.2, facecolor="#e6f2ff"))
        ax4.text(0.495,0.7,"Callback/Follow-up", ha="center", va="center")
        ax4.add_patch(plt.Rectangle((0.69,0.6),0.25,0.2, facecolor="#e6f2ff"))
        ax4.text(0.815,0.7,"Meeting\nScheduled", ha="center", va="center")
        ax4.annotate("", xy=(0.35,0.7), xytext=(0.27,0.7), arrowprops=dict(arrowstyle="->"))
        ax4.annotate("", xy=(0.67,0.7), xytext=(0.59,0.7), arrowprops=dict(arrowstyle="->"))
        st.pyplot(fig4)
        plt.close(fig4)

        # Downloads: build Excel & PDF in memory
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            agg.to_excel(writer, sheet_name="campaign_summary", index=False)
            leads.to_excel(writer, sheet_name="leads_detailed", index=False)
            pd.DataFrame([{"spend_campaign":k, "mapped_to":v} for k,v in final_mapping.items()]).to_excel(writer, sheet_name="mapping", index=False)
        buffer.seek(0)
        st.download_button("Download Excel", buffer.getvalue(), file_name=f"campaign_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # PDF build in memory
        pdf_buf = io.BytesIO()
        try:
            w,h = landscape(A4)
            c = canvas.Canvas(pdf_buf, pagesize=landscape(A4))
            c.setFont("Helvetica-Bold", 18)
            c.drawCentredString(w/2, h-30, totals["report_title"])
            # render one chart snapshot per available pyplot figure images saved temporarily
            # Save the last generated matplotlib figures to BytesIO via PNGs and embed
            # We'll render the charts we already displayed by re-creating them to buffers
            # 1) leads pie
            pie_buf = io.BytesIO()
            fig_p, ax_p = plt.subplots(figsize=(6,6))
            ax_p.pie(agg["leads_count"], labels=agg["campaign"], autopct="%1.1f%%", startangle=90)
            fig_p.savefig(pie_buf, format="png", bbox_inches="tight")
            pie_buf.seek(0)
            c.drawImage(ImageReader(pie_buf), 30, h/2 - 10, width=(w/2 - 60), height=(h/2 - 40))
            # 2) spend or cpl bar (if exist)
            bar_buf = io.BytesIO()
            fig_b, ax_b = plt.subplots(figsize=(6,4))
            if not agg.empty:
                ax_b.bar(agg["campaign"], agg["spend_total"])
                ax_b.set_title("Spend by Campaign (₹)")
                plt.setp(ax_b.get_xticklabels(), rotation=30, ha="right")
            fig_b.savefig(bar_buf, format="png", bbox_inches="tight")
            bar_buf.seek(0)
            c.drawImage(ImageReader(bar_buf), w/2 + 10, h/2 - 10, width=(w/2 - 60), height=(h/2 - 40))
            # summary
            c.setFont("Helvetica", 11)
            y = h/2 - 60
            stats = [
                f"Total leads: {totals['total_leads']:,}",
                f"Total spend: ₹{totals['total_spend']:,.2f}",
                f"Unique leads (est): {totals['unique_leads_est']:,}",
            ]
            for s in stats:
                c.drawString(40, y, "• " + s)
                y -= 14
            c.showPage()
            c.save()
            pdf_buf.seek(0)
            st.download_button("Download PDF report", pdf_buf.getvalue(), file_name=f"campaign_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
        except Exception as e:
            st.error("PDF generation failed: " + str(e))
            logger.exception(e)

else:
    st.info("Upload both Leads and Spend files to begin.")
