# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import re

st.set_page_config(layout="wide", page_title="Lead Dashboard Generator")

def normalize_phone(p):
    if pd.isna(p) or p=="":
        return ""
    s = re.sub(r"\D", "", str(p))
    return s.lstrip("0")

def sentiment_from_status(s):
    s = str(s).strip().lower()
    if s in ["new", "meeting scheduled", "meeting fixed", "booked", "booked under name"]:
        return "Positive"
    if any(k in s for k in ["callback", "follow", "busy", "not answered", "need more info", "not reachable", "pending"]):
        return "Neutral"
    if any(k in s for k in ["not interested", "dropped", "dnp", "no interest"]):
        return "Negative"
    return "Neutral"

def build_dashboard(df):
    df = df.copy()

    def find_col(possible):
        for c in possible:
            if c in df.columns:
                return c
        return None

    phone_col = find_col(["Contact No","Contact","Phone","phone","Mobile"])
    email_col = find_col(["Email","email","E-mail"])
    campaign_col = find_col(["Campaign Names","Campaign","campaign names","campaign"])
    status_col = find_col(["Status","status"])
    spend_col = find_col(["Spend","spend","Amount","amount","Cost"])

    if phone_col:
        df["phone_norm"] = df[phone_col].apply(normalize_phone)
    else:
        df["phone_norm"] = ""

    if email_col:
        df["email_norm"] = df[email_col].str.lower().fillna("")
    else:
        df["email_norm"] = ""

    df["campaign"] = df[campaign_col].fillna("Unknown") if campaign_col else "Unknown"
    df["status_norm"] = df[status_col].fillna("") if status_col else ""
    df["sentiment"] = df["status_norm"].apply(sentiment_from_status)

    if spend_col:
        df["spend"] = pd.to_numeric(df[spend_col].replace("", 0).fillna(0), errors="coerce").fillna(0)
    else:
        df["spend"] = 0.0

    df["dup_phone"] = df["phone_norm"].duplicated(keep="first") & (df["phone_norm"] != "")
    df["dup_email"] = df["email_norm"].duplicated(keep="first") & (df["email_norm"] != "")
    df["duplicate"] = df["dup_phone"] | df["dup_email"]

    agg_raw = df.groupby("campaign").agg(leads_raw=("campaign","size"), spend_raw=("spend","sum")).reset_index()
    agg_unique = df[~df["duplicate"]].groupby("campaign").agg(leads_unique=("campaign","size"), spend_unique=("spend","sum")).reset_index()
    agg = pd.merge(agg_raw, agg_unique, on="campaign", how="left").fillna(0)
    agg["cpl_unique"] = agg.apply(lambda r: (r["spend_unique"] / r["leads_unique"]) if r["leads_unique"]>0 else 0, axis=1)
    agg = agg.sort_values("leads_raw", ascending=False)

    total_leads = int(df.shape[0])
    duplicate_count = int(df["duplicate"].sum())
    unique_leads = total_leads - duplicate_count
    total_spend = float(df["spend"].sum())
    avg_cpl = total_spend / unique_leads if unique_leads>0 else 0

    campaigns = agg["campaign"].tolist()
    leads_vals = agg["leads_raw"].astype(int).tolist()
    spend_vals = agg["spend_raw"].astype(float).tolist()
    cpl_vals = agg["cpl_unique"].astype(float).tolist()

    fig, axes = plt.subplots(2,2, figsize=(14,10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    axes[0,0].bar(campaigns, leads_vals, color="#2ca02c")
    axes[0,0].set_title("Leads per Campaign", fontsize=12, fontweight="bold")
    axes[0,0].tick_params(axis='x', rotation=30)

    axes[0,1].bar(campaigns, spend_vals, color="#1f77b4")
    axes[0,1].set_title("Spend per Campaign (₹)", fontsize=12, fontweight="bold")
    axes[0,1].tick_params(axis='x', rotation=30)

    axes[1,0].bar(campaigns, cpl_vals, color="#ff7f0e")
    axes[1,0].set_title("CPL per Campaign (deduped)", fontsize=12, fontweight="bold")
    axes[1,0].tick_params(axis='x', rotation=30)

    sent_counts = df["sentiment"].value_counts()
    sizes = [sent_counts.get("Positive",0), sent_counts.get("Neutral",0), sent_counts.get("Negative",0)]
    axes[1,1].pie(sizes, labels=["Positive","Neutral","Negative"], autopct='%1.1f%%', colors=["#4daf4a","#ffbf00","#e41a1c"])
    axes[1,1].set_title("Lead Sentiment Distribution", fontsize=12, fontweight="bold")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close(fig)

    highlights = {
        "total_leads": total_leads,
        "duplicate_count": duplicate_count,
        "duplicate_pct": round((duplicate_count/total_leads*100) if total_leads>0 else 0,1),
        "unique_leads": unique_leads,
        "total_spend": total_spend,
        "avg_cpl": avg_cpl,
        "top_campaigns": agg.head(5).to_dict(orient="records"),
        "top_users": df['Assigned User Name'].value_counts().head(5).to_dict() if 'Assigned User Name' in df.columns else {}
    }
    return buf, highlights

st.title("Lead Dashboard Generator — Photo-style template")
uploaded = st.file_uploader("Upload leads CSV / XLSX", type=["csv","xlsx"])
if uploaded:
    if uploaded.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)
    st.dataframe(df.head())

    buf, highlights = build_dashboard(df)
    st.image(buf, use_column_width=True)

    st.markdown("### Highlights")
    st.write(f"**Total Leads (raw):** {highlights['total_leads']}")
    st.write(f"**Duplicate Leads:** {highlights['duplicate_count']} ({highlights['duplicate_pct']}%)")
    st.write(f"**Unique Leads:** {highlights['unique_leads']}")
    st.write(f"**Total Spend:** ₹{highlights['total_spend']:.2f}")
    st.write(f"**Avg CPL:** ₹{highlights['avg_cpl']:.2f}")
