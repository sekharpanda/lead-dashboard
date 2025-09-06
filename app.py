# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import re

st.set_page_config(layout="wide", page_title="Lead Dashboard Generator")

# ----------------- helpers -----------------
def normalize_phone(p):
    if pd.isna(p) or p == "": 
        return ""
    s = re.sub(r"\D", "", str(p))
    return s.lstrip("0")

def sentiment_from_status(s):
    s = str(s).strip().lower()
    if any(x in s for x in ["new","meeting scheduled","meeting fixed","booked","meeting fixed","booked under name"]):
        return "Positive"
    if any(x in s for x in ["callback","follow","busy","not answered","need more info","not reachable","pending","follow up"]):
        return "Neutral"
    if any(x in s for x in ["not interested","dropped","dnp","no interest"]):
        return "Negative"
    return "Neutral"

def read_uploaded_file(uploaded):
    """
    Read an uploaded CSV/XLSX. Handles two layouts:
    - Normal table: header row + rows
    - Vertical export: first column contains field labels and each subsequent column is a lead.
    """
    try:
        if uploaded.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded, engine="openpyxl")
        else:
            df = pd.read_csv(uploaded)
    except Exception:
        # fallback: read with no header and inspect first column for label tokens
        if uploaded.name.endswith(".xlsx"):
            raw = pd.read_excel(uploaded, engine="openpyxl", header=None)
        else:
            raw = pd.read_csv(uploaded, header=None)
        first_col = raw.iloc[:,0].astype(str).str.lower()
        label_score = first_col.str.contains(r"name|email|contact|status|campaign|created|assigned").sum()
        if label_score >= 6:
            # vertical layout: build records from columns
            keys = raw[0].astype(str).str.rstrip(":").str.strip().tolist()
            records = []
            for col in raw.columns[1:]:
                rec = {k: v for k,v in zip(keys, raw[col].tolist())}
                records.append(rec)
            df = pd.DataFrame(records)
            df.columns = [str(c).strip() for c in df.columns]
        else:
            # try guessing header row offsets
            df = None
            for h in range(0,6):
                try:
                    if uploaded.name.endswith(".xlsx"):
                        cand = pd.read_excel(uploaded, engine="openpyxl", header=h)
                    else:
                        cand = pd.read_csv(uploaded, header=h)
                    cols_text = " ".join([str(x).lower() for x in cand.columns])
                    if any(k in cols_text for k in ["name","status","contact","campaign"]):
                        df = cand
                        break
                except Exception:
                    continue
            if df is None:
                # last resort: read with header=0
                if uploaded.name.endswith(".xlsx"):
                    df = pd.read_excel(uploaded, engine="openpyxl", header=0)
                else:
                    df = pd.read_csv(uploaded, header=0)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def map_campaigns(df):
    """
    Create campaign_mapped column:
      - prefer Facebook Campaign Name when Lead Source contains 'facebook'
      - prefer Sub Source when Lead Source mentions Google / search / ad
      - else fallback to Campaign / Campaign Names
    """
    def find(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    lead_source_col = find(["Lead Source","Source","lead source","Source "])
    fb_col = find(["Facebook Campaign Name","Facebook Campaign Nam","Facebook Campaign","Facebook Campaign Name "])
    google_col = find(["Sub Source","SubSource","Sub Source " ,"Sub Source(s)","Sub Source(s) "])
    camp_col = find(["Campaign Names","Campaign","campaign names","Campaign Names "])

    df["campaign_base"] = df[camp_col] if camp_col else ""
    df["lead_source_norm"] = df[lead_source_col].astype(str).str.lower() if lead_source_col else ""

    def pick_campaign(row):
        ls = str(row.get("lead_source_norm","")).lower()
        if "facebook" in ls and fb_col and pd.notna(row.get(fb_col,"")) and str(row.get(fb_col)).strip() != "":
            return row.get(fb_col)
        if any(x in ls for x in ["google","search","ad","adword","ad words","gclid"]) and google_col and pd.notna(row.get(google_col,"")) and str(row.get(google_col)).strip() != "":
            return row.get(google_col)
        if pd.notna(row.get("campaign_base","")) and str(row.get("campaign_base")).strip() != "":
            return row.get("campaign_base")
        return "Unknown"

    df["campaign_mapped"] = df.apply(pick_campaign, axis=1)
    return df

def build_dashboard_and_highlights(df, include_duplicates=False, date_from=None, date_to=None):
    # detect common columns
    phone_col = next((c for c in df.columns if re.search(r"contact no|contact|phone|mobile", c, re.I)), None)
    email_col = next((c for c in df.columns if re.search(r"email", c, re.I)), None)
    status_col = next((c for c in df.columns if re.search(r"status", c, re.I)), None)
    assigned_col = next((c for c in df.columns if re.search(r"assigned", c, re.I)), None)
    spend_col = next((c for c in df.columns if re.search(r"spend|amount|cost", c, re.I)), None)
    received_col = next((c for c in df.columns if re.search(r"received|created|date", c, re.I)), None)

    # normalize
    df["phone_norm"] = df[phone_col].apply(normalize_phone) if phone_col else ""
    df["email_norm"] = df[email_col].str.lower().fillna("") if email_col else ""
    df["status_norm"] = df[status_col].astype(str).fillna("") if status_col else ""
    df["assigned_user"] = df[assigned_col].fillna("Unassigned") if assigned_col else "Unassigned"
    if spend_col:
        df["spend_num"] = pd.to_numeric(df[spend_col].astype(str).replace('[^0-9.]','', regex=True), errors="coerce").fillna(0)
    else:
        df["spend_num"] = 0.0

    # dedupe
    df["dup_phone"] = df["phone_norm"].duplicated(keep="first") & (df["phone_norm"] != "")
    df["dup_email"] = df["email_norm"].duplicated(keep="first") & (df["email_norm"] != "")
    df["duplicate"] = df["dup_phone"] | df["dup_email"]

    # filter date range
    if received_col:
        df["received_dt"] = pd.to_datetime(df[received_col], errors="coerce")
        if date_from:
            df = df[df["received_dt"] >= pd.to_datetime(date_from)]
        if date_to:
            df = df[df["received_dt"] <= pd.to_datetime(date_to)]

    # sentiment
    df["sentiment"] = df["status_norm"].apply(sentiment_from_status)

    # map campaigns
    df = map_campaigns(df)

    # choose deduped or raw for aggregation
    df_for_agg = df if include_duplicates else df[~df["duplicate"]]

    # aggregates
    agg = df_for_agg.groupby("campaign_mapped").agg(
        leads = ("campaign_mapped","size"),
        spend = ("spend_num","sum")
    ).reset_index().sort_values("leads", ascending=False)

    agg["cpl"] = agg.apply(lambda r: (r["spend"]/r["leads"]) if r["leads"]>0 else 0, axis=1)

    # totals
    total_leads_raw = len(df)
    dup_count = int(df["duplicate"].sum())
    unique_leads = total_leads_raw - dup_count
    total_spend = float(df["spend_num"].sum())
    avg_cpl = total_spend / unique_leads if unique_leads>0 else 0

    # highlights
    highlights = {
        "total_leads_raw": total_leads_raw,
        "duplicate_count": dup_count,
        "duplicate_pct": round(dup_count/total_leads_raw*100,1) if total_leads_raw>0 else 0,
        "unique_leads": unique_leads,
        "total_spend": total_spend,
        "avg_cpl": round(avg_cpl,2),
        "top_campaigns": agg.head(6).to_dict(orient="records"),
        "top_facebook": agg[agg["campaign_mapped"].str.contains("facebook", case=False, na=False)].head(3).to_dict(orient="records"),
        "top_google": agg[agg["campaign_mapped"].str.contains("google|search|ad|adword|gclid", case=False, na=False)].head(3).to_dict(orient="records"),
        "top_users": df["assigned_user"].value_counts().head(6).to_dict()
    }

    # plotting
    fig, axes = plt.subplots(2,2, figsize=(14,10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    campaigns = agg["campaign_mapped"].astype(str).tolist()
    axes[0,0].bar(campaigns, agg["leads"], color="#2ca02c")
    axes[0,0].set_title("Leads per Campaign")
    axes[0,0].tick_params(axis='x', rotation=30)

    axes[0,1].bar(campaigns, agg["spend"], color="#1f77b4")
    axes[0,1].set_title("Spend per Campaign (₹)")
    axes[0,1].tick_params(axis='x', rotation=30)

    axes[1,0].bar(campaigns, agg["cpl"], color="#ff7f0e")
    axes[1,0].set_title("CPL per Campaign (deduped)")
    axes[1,0].tick_params(axis='x', rotation=30)

    sent_counts = df["sentiment"].value_counts()
    sizes = [sent_counts.get("Positive",0), sent_counts.get("Neutral",0), sent_counts.get("Negative",0)]
    axes[1,1].pie(sizes, labels=["Positive","Neutral","Negative"], autopct='%1.1f%%', colors=["#4daf4a","#ffbf00","#e41a1c"])
    axes[1,1].set_title("Lead Sentiment Distribution")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close(fig)

    return buf, highlights, df, agg

# ----------------- UI: uploader + options + chat + tasks -----------------
st.title("Lead Dashboard Generator — Photo-style template")

col1, col2 = st.columns([3,1])
with col1:
    uploaded = st.file_uploader("Upload leads CSV / XLSX", type=["csv","xlsx"])
with col2:
    include_dup = st.checkbox("Include duplicates in counts", value=False)
    # date input: allow either single date or range; Streamlit returns a single date or tuple
    date_range = st.date_input("Filter by date range (optional)", value=None)

if uploaded:
    df_raw = read_uploaded_file(uploaded)
    st.markdown("**Preview (first 6 rows)**")
    st.dataframe(df_raw.head(6))

    # map date range input
    date_from, date_to = (None, None)
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        date_from, date_to = date_range
    # build dashboard
    buf, highlights, df_clean, agg = build_dashboard_and_highlights(df_raw, include_duplicates=include_dup, date_from=date_from, date_to=date_to)
    st.image(buf, use_column_width=True)

    # Highlights box (3 columns like your template)
    st.markdown("### ✨ Highlights")
    hcol1, hcol2, hcol3 = st.columns(3)
    with hcol1:
        st.markdown("**📊 Campaign Performance**")
        st.write(f"- Total Leads (raw): **{highlights['total_leads_raw']}**")
        st.write(f"- Duplicate Leads: **{highlights['duplicate_count']}**  ({highlights['duplicate_pct']}%)")
        st.write(f"- Unique Leads: **{highlights['unique_leads']}**")
        st.write(f"- Total Spend: **₹{highlights['total_spend']:.2f}**")
        st.write(f"- Avg CPL (deduped): **₹{highlights['avg_cpl']}**")
        st.write("")
        st.write("Top campaigns:")
        for r in highlights["top_campaigns"][:5]:
            st.write(f"• {r['campaign_mapped']}: {int(r['leads'])} leads | CPL ₹{r['cpl']:.0f}")

    with hcol2:
        st.markdown("**😊 Sentiment Insights**")
        st.write(f"- Positive: **{int((df_clean['sentiment']=='Positive').sum())}**")
        st.write(f"- Neutral: **{int((df_clean['sentiment']=='Neutral').sum())}**")
        st.write(f"- Negative: **{int((df_clean['sentiment']=='Negative').sum())}**")
        st.write("")
        st.write("Notes:")
        st.write("- Most leads are Neutral → prioritize callbacks & follow-ups")
        st.write("- Check high-CPL campaigns for optimization")

    with hcol3:
        st.markdown("**👥 User Action Analysis**")
        st.write("Top assigned users:")
        for u,c in highlights["top_users"].items():
            st.write(f"• {u}: {c} leads")
        st.write("")
        st.write("Top Facebook campaigns:")
        for r in highlights["top_facebook"]:
            st.write(f"• {r['campaign_mapped']}: {int(r['leads'])} leads | CPL ₹{r['cpl']:.0f}")
        st.write("")
        st.write("Top Google campaigns:")
        for r in highlights["top_google"]:
            st.write(f"• {r['campaign_mapped']}: {int(r['leads'])} leads | CPL ₹{r['cpl']:.0f}")

    # ---------- Simple Chatbox ----------
    st.markdown("---")
    st.markdown("### 💬 Quick Chat / Notes")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for msg in st.session_state.chat_history:
        st.markdown(msg)
    chat_input = st.text_input("Type a quick note or question (press Enter to add)", key="chat_input")
    if chat_input:
        st.session_state.chat_history.append(f"- {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}  {chat_input}")
        st.experimental_rerun()

    # ---------- Tasks textarea + copy/download ----------
    st.markdown("---")
    st.markdown("### ✍️ Quick Tasks (paste tasks here; download or copy)")
    tasks = st.text_area("Paste tasks (one per line):", height=120, key="tasks_area")
    cols = st.columns([1,1,1])
    if cols[0].button("Show formatted"):
        lines = [l.strip() for l in tasks.splitlines() if l.strip()]
        out = "\n".join([f"• {i+1}. {l}" for i,l in enumerate(lines)])
        st.code(out)
    if cols[1].button("Download as .txt"):
        st.download_button("Download tasks file", data=tasks or "", file_name="tasks.txt", mime="text/plain")
    if cols[2].button("Clear tasks"):
        st.session_state.tasks_area = ""
        st.experimental_rerun()
