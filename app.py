# app.py
import os
import logging
import re
from typing import Optional, List

import pandas as pd
import streamlit as st

# ---------- CONFIG ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lead-dashboard")

DEFAULT_DATE_COL = os.getenv("DEFAULT_DATE_COL", "created_at")

# ---------- UTILITIES ----------
def ensure_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c not in df.columns]

def safe_parse_dates(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

# ---------- SPEND PARSING & CPL LOGIC ----------
def build_dashboard_and_highlights(df: pd.DataFrame, spend_col: Optional[str], include_duplicates: bool):
    if df is None or len(df) == 0:
        st.warning("Empty dataframe passed to build_dashboard_and_highlights.")
        return pd.DataFrame(), {}

    if "duplicate" not in df.columns:
        df["duplicate"] = False
    if "campaign_mapped" not in df.columns:
        if "campaign" in df.columns:
            df["campaign_mapped"] = df["campaign"].astype(str).fillna("unknown")
        else:
            df["campaign_mapped"] = "unknown"

    def parse_spend_series(series):
        s = series.astype(str).fillna("").str.strip()
        s = s.str.replace(r"[₹$€£,]|AED|INR|usd|rs\.?", "", regex=True, flags=re.IGNORECASE)
        s = s.str.replace(r"[^0-9\.\-]", "", regex=True)
        s = s.str.replace(r"\.(?=.*\.)", "", regex=True)
        return pd.to_numeric(s, errors="coerce").fillna(0.0)

    if spend_col and spend_col in df.columns:
        df["spend_num"] = parse_spend_series(df[spend_col])
    else:
        df["spend_num"] = 0.0

    spend_sum = float(df["spend_num"].sum())
    zero_frac = float((df["spend_num"] == 0).mean()) if len(df) else 1.0
    if spend_sum == 0 or zero_frac > 0.9:
        candidate_cols = [c for c in df.columns if re.search(r"total.*spend|campaign.*spend|ad.*spend|spend.*total|total.*cost", c, re.I)]
        if candidate_cols:
            df["spend_num"] = parse_spend_series(df[candidate_cols[0]])
        else:
            text_cols = [c for c in df.columns if df[c].dtype == object]
            for col in text_cols:
                example = df[col].astype(str).str.extract(r"([\d\.,]+\s*(?:AED|INR|USD|rs|₹)?)", expand=False).fillna("")
                parsed = parse_spend_series(example)
                if parsed.sum() > 0:
                    df["spend_num"] = parsed
                    break

    df_deduped = df[~df["duplicate"].astype(bool)].copy()
    df_raw = df.copy()

    agg_raw = (
        df_raw.groupby("campaign_mapped")
        .agg(leads_raw=("campaign_mapped", "size"), spend_raw=("spend_num", "sum"))
        .reset_index()
    )
    agg_dedup = (
        df_deduped.groupby("campaign_mapped")
        .agg(leads_unique=("campaign_mapped", "size"), spend_unique=("spend_num", "sum"))
        .reset_index()
    )
    agg = pd.merge(agg_raw, agg_dedup, on="campaign_mapped", how="left").fillna(0)

    agg["CPL_raw"] = agg.apply(lambda r: r["spend_raw"]/r["leads_raw"] if r["leads_raw"] > 0 else 0, axis=1)
    agg["CPL_unique"] = agg.apply(lambda r: r["spend_unique"]/r["leads_unique"] if r["leads_unique"] > 0 else 0, axis=1)

    agg["CPL_display"] = agg["CPL_raw"] if include_duplicates else agg["CPL_unique"]

    totals = {
        "total_spend_raw": float(agg["spend_raw"].sum()),
        "total_spend_unique": float(agg["spend_unique"].sum()),
        "total_leads_raw": int(len(df_raw)),
        "total_leads_unique": int(len(df_deduped)),
        "avg_cpl_raw": float((agg["spend_raw"].sum()/len(df_raw)) if len(df_raw) else 0),
        "avg_cpl_unique": float((agg["spend_unique"].sum()/len(df_deduped)) if len(df_deduped) else 0),
    }
    return agg, totals

# ---------- APP ----------
def main():
    st.set_page_config(page_title="Lead Dashboard", layout="wide")
    st.title("Lead Analysis Dashboard — CSV + Excel support")

    st.sidebar.header("Upload your leads file")
    uploaded_file = st.sidebar.file_uploader("Choose CSV or Excel", type=["csv", "xlsx", "xls"])

    spend_col_input = st.sidebar.text_input("Spend column name (optional)", "")
    date_col_input = st.sidebar.text_input("Date column", DEFAULT_DATE_COL)
    include_duplicates = st.sidebar.checkbox("Include duplicate leads in CPL (use CPL_raw)", value=False)

    if uploaded_file is None:
        st.info("Please upload a CSV or Excel file to continue.")
        st.stop()

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        st.stop()

    with st.expander("Data sample and schema"):
        st.write("Rows:", len(df))
        st.dataframe(df.head(20))
        st.write("Columns:", list(df.columns))

    missing = ensure_columns(df, ["lead_id", date_col_input])
    if missing:
        st.warning(f"Missing required columns: {missing}. Some KPIs will be limited.")

    df = safe_parse_dates(df, [date_col_input])

    agg, totals = build_dashboard_and_highlights(df, spend_col_input.strip() or None, include_duplicates)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total leads (raw)", totals.get("total_leads_raw", 0))
    k2.metric("Total leads (unique)", totals.get("total_leads_unique", 0))
    k3.metric("Total spend (unique)", f"{totals.get('total_spend_unique', 0):.2f}")
    k4.metric("Avg CPL (unique)", f"{totals.get('avg_cpl_unique', 0):.2f}")

    st.subheader("Campaign CPLs")
    if not agg.empty:
        st.dataframe(agg.sort_values("CPL_display").reset_index(drop=True).head(200))
        try:
            top_by_leads = agg.sort_values("leads_unique", ascending=False).head(10)
            st.bar_chart(top_by_leads.set_index("campaign_mapped")["leads_unique"])
        except Exception as e:
            logger.warning("Failed to draw campaign chart: %s", e)
    else:
        st.info("No campaign data to display.")

    with st.expander("Download campaign aggregates"):
        csv = agg.to_csv(index=False)
        st.download_button("Download CSV", csv, file_name="campaign_aggregates.csv", mime="text/csv")

    with st.expander("Diagnostics / Notes"):
        st.write("Applied spend column:", spend_col_input or "(auto-detected)")
        st.write("Include duplicates for CPL:", include_duplicates)
        st.write("Totals:", totals)
        st.write("Sample aggregated campaigns:", agg.head(10))

if __name__ == "__main__":
    main()
