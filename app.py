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

DATA_SOURCE = os.getenv("LEAD_DATA_SOURCE", "data/leads.csv")
DEFAULT_DATE_COL = os.getenv("DEFAULT_DATE_COL", "created_at")

# ---------- UTILITIES ----------
def safe_select(df: pd.DataFrame, col: str, default=None):
    if col in df.columns:
        return df[col]
    else:
        logger.warning("Column missing: %s", col)
        return pd.Series([default] * len(df), index=df.index)

def ensure_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    missing = [c for c in cols if c not in df.columns]
    return missing

@st.cache_data(ttl=300)
def load_csv(path: str) -> pd.DataFrame:
    logger.info("Loading CSV from %s", path)
    try:
        df = pd.read_csv(path)
        logger.info("Loaded %d rows", len(df))
        return df
    except FileNotFoundError:
        logger.exception("CSV file not found: %s", path)
        raise

@st.cache_data(ttl=300)
def load_data(source: str) -> pd.DataFrame:
    if source.startswith("db:"):
        conn_str = source.split("db:", 1)[1]
        logger.info("Would connect to DB with %s", conn_str)
        raise NotImplementedError("DB source not yet implemented")
    if source.startswith("api:"):
        endpoint = source.split("api:", 1)[1]
        logger.info("Would call API %s", endpoint)
        raise NotImplementedError("API source not yet implemented")
    return load_csv(source)

def safe_parse_dates(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

# ---------- New spend parsing + CPL logic wrapped as function ----------
def build_dashboard_and_highlights(df: pd.DataFrame, spend_col: Optional[str], include_duplicates: bool):
    """
    This function ingests df and computes per-campaign CPLs with robust spend parsing.
    It returns:
      - agg (DataFrame) with campaign_mapped and CPL_display
      - totals dict with total_spend_raw/unique, total_leads_raw/unique, avg cpls
    """
    # defensive defaults
    if df is None or len(df) == 0:
        st.warning("Empty dataframe passed to build_dashboard_and_highlights.")
        empty_agg = pd.DataFrame(columns=[
            "campaign_mapped", "leads_raw", "spend_raw", "leads_unique", "spend_unique",
            "CPL_raw", "CPL_unique", "CPL_display"
        ])
        totals = {
            "total_spend_raw": 0.0,
            "total_spend_unique": 0.0,
            "total_leads_raw": 0,
            "total_leads_unique": 0,
            "avg_cpl_raw": 0.0,
            "avg_cpl_unique": 0.0
        }
        return empty_agg, totals

    # ensure some helpful columns exist
    if "duplicate" not in df.columns:
        # assume no duplicates unless flagged
        df["duplicate"] = False

    if "campaign_mapped" not in df.columns:
        # fallback to naive campaign mapping
        if "campaign" in df.columns:
            df["campaign_mapped"] = df["campaign"].astype(str).fillna("unknown")
        else:
            df["campaign_mapped"] = "unknown"

    # ---------- Paste your improved spend parsing + CPL code (with slight safety edits) ----------
    # 1) Robust spend parsing per row
    def parse_spend_series(series):
        # series: pd.Series of raw spend strings/numbers
        s = series.astype(str).fillna("").str.strip()
        # remove common currency symbols and words, keep digits and dots and commas and minus
        s = s.str.replace(r"[₹$€£,]|AED|INR|usd|rs\.?|AED", "", regex=True, flags=re.IGNORECASE)
        # replace any non digit/dot/minus with empty
        s = s.str.replace(r"[^0-9\.\-]", "", regex=True)
        # fix multiple dots (keep the last dot as decimal separator)
        s = s.str.replace(r"\.(?=.*\.)", "", regex=True)
        return pd.to_numeric(s, errors="coerce").fillna(0.0)

    if spend_col and spend_col in df.columns:
        # attempt numeric parse
        df["spend_num"] = parse_spend_series(df[spend_col])
    else:
        # no per-row spend column detected -> default to 0
        df["spend_num"] = 0.0

    # 2) If per-row spend parsed is mostly zeros, try to detect campaign-level spend columns
    try:
        spend_sum = float(df["spend_num"].sum())
        zero_frac = float((df["spend_num"] == 0).mean())
    except Exception:
        spend_sum = 0.0
        zero_frac = 1.0

    if spend_sum == 0 or zero_frac > 0.9:
        # look for campaign-level spend columns (common labels)
        candidate_cols = [c for c in df.columns if re.search(r"total.*spend|campaign.*spend|ad.*spend|spend.*total|total.*cost", c, re.I)]
        if candidate_cols:
            df["spend_num"] = parse_spend_series(df[candidate_cols[0]])
            logger.info("Used campaign-level spend column: %s", candidate_cols[0])
        else:
            # fallback: attempt to extract spend from "Notes" or "Message" columns if they contain "spend: 1234"
            text_cols = [c for c in df.columns if df[c].dtype == object]
            found = False
            for col in text_cols:
                example = df[col].astype(str).str.extract(r"([\d\.,]+\s*(?:AED|INR|USD|rs|₹)?)", expand=False).fillna("")
                parsed = parse_spend_series(example)
                if parsed.sum() > 0:
                    df["spend_num"] = parsed
                    found = True
                    logger.info("Extracted spend from text column: %s", col)
                    break
            # if still nothing, keep zeros

    # 3) Now compute aggregates correctly
    # choose whether to dedupe BEFORE or AFTER spend aggregation. We will compute both so you can choose.
    try:
        df_deduped = df[~df["duplicate"].astype(bool)].copy()
    except Exception:
        df_deduped = df.copy()

    df_raw = df.copy()

    # per-campaign spend aggregation (sum of spend_num)
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

    # merge to one table
    agg = pd.merge(agg_raw, agg_dedup, on="campaign_mapped", how="left").fillna(0)

    # Calculate CPLs explicitly with protective checks
    agg["CPL_raw"] = agg.apply(lambda r: (r["spend_raw"] / r["leads_raw"]) if r["leads_raw"] > 0 else 0, axis=1)
    agg["CPL_unique"] = agg.apply(lambda r: (r["spend_unique"] / r["leads_unique"]) if r["leads_unique"] > 0 else 0, axis=1)

    # For dashboard choose CPL_unique (deduped) unless include_duplicates True
    if include_duplicates:
        agg["CPL_display"] = agg["CPL_raw"]
    else:
        agg["CPL_display"] = agg["CPL_unique"]

    # Totals (show both raw and deduped)
    total_spend_raw = float(agg["spend_raw"].sum())
    total_spend_unique = float(agg["spend_unique"].sum())
    total_leads_raw = int(len(df_raw))
    total_leads_unique = int(len(df_deduped))
    avg_cpl_raw = float((total_spend_raw / total_leads_raw) if total_leads_raw > 0 else 0)
    avg_cpl_unique = float((total_spend_unique / total_leads_unique) if total_leads_unique > 0 else 0)

    totals = {
        "total_spend_raw": total_spend_raw,
        "total_spend_unique": total_spend_unique,
        "total_leads_raw": total_leads_raw,
        "total_leads_unique": total_leads_unique,
        "avg_cpl_raw": avg_cpl_raw,
        "avg_cpl_unique": avg_cpl_unique,
    }

    # Now the rest of the code should use agg["CPL_display"] and totals for highlights
    return agg, totals

# ---------- KPI HELPERS ----------
def compute_kpis(df: pd.DataFrame):
    total_leads = df["lead_id"].nunique() if "lead_id" in df.columns else len(df)
    created = df[DEFAULT_DATE_COL].notna().sum() if DEFAULT_DATE_COL in df.columns else total_leads
    spend = df.get("_spend_col", pd.Series([0])).sum() if isinstance(df, pd.DataFrame) else 0
    return {
        "total_leads": int(total_leads),
        "created_count": int(created),
        "total_spend": float(spend),
        "avg_spend_per_lead": float(spend / total_leads) if total_leads else 0,
    }

# ---------- STREAMLIT APP ----------
def main():
    st.set_page_config(page_title="Lead Dashboard", layout="wide")
    st.title("Lead Analysis Dashboard — improved app.py")

    st.sidebar.header("Settings")
    source = st.sidebar.text_input("Data source (CSV path / db:... / api:...)", DATA_SOURCE)
    spend_col_input = st.sidebar.text_input("Spend column name (optional)", "")
    date_col_input = st.sidebar.text_input("Date column", DEFAULT_DATE_COL)
    include_duplicates = st.sidebar.checkbox("Include duplicate leads in CPL (use CPL_raw)", value=False)

    try:
        df = load_data(source)
    except Exception as e:
        st.error(f"Failed to load data from `{source}`: {e}")
        st.stop()

    with st.expander("Data sample and schema"):
        st.write("Rows:", len(df))
        st.dataframe(df.head(20))
        st.write("Columns:", list(df.columns))

    required_example = ["lead_id", date_col_input]
    missing = ensure_columns(df, required_example)
    if missing:
        st.warning(f"Missing required columns: {missing}. Some KPIs will be limited.")

    df = safe_parse_dates(df, [date_col_input])

    # attach a pre-spend internal column name for legacy code compatibility
    # compute spend parsing and CPLs with our function
    agg, totals = build_dashboard_and_highlights(df, spend_col_input.strip() or None, include_duplicates)

    # show highlights (totals)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total leads (raw)", totals["total_leads_raw"])
    k2.metric("Total leads (unique)", totals["total_leads_unique"])
    k3.metric("Total spend (unique)", f"{totals['total_spend_unique']:.2f}")
    k4.metric("Avg CPL (unique)", f"{totals['avg_cpl_unique']:.2f}")

    st.subheader("Campaign CPLs")
    st.dataframe(agg.sort_values("CPL_display").reset_index(drop=True).head(200))

    # simple visual: top campaigns by leads
    if not agg.empty:
        try:
            top_by_leads = agg.sort_values("leads_unique", ascending=False).head(10)
            st.bar_chart(top_by_leads.set_index("campaign_mapped")["leads_unique"])
        except Exception as e:
            logger.warning("Failed to draw campaign chart: %s", e)

    with st.expander("Download campaign aggregates"):
        csv = agg.to_csv(index=False)
        st.download_button("Download campaign aggregates CSV", csv, file_name="campaign_aggregates.csv", mime="text/csv")

    with st.expander("Diagnostics / Notes"):
        st.write("Applied spend column:", spend_col_input or "(auto-detected)")
        st.write("Include duplicates for CPL:", include_duplicates)
        st.write("Totals:", totals)
        st.write("Sample aggregated campaigns:", agg.head(10))

    st.markdown("### Add your custom transforms and visuals below")
    st.code(
        """
# Example placeholder - replace with your own computations
# df['is_qualified'] = df['status'].isin(['Qualified','SQL'])
# qualified_count = df[df['is_qualified']].lead_id.nunique()
""",
        language="python",
    )

if __name__ == "__main__":
    main()
