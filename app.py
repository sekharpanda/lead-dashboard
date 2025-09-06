# ----- Paste this inside build_dashboard_and_highlights, replacing the current spend parsing + CPL part -----

# 1) Robust spend parsing per row
# `spend_col` detection earlier stays the same. Now parse safely:
def parse_spend_series(series):
    # series: pd.Series of raw spend strings/numbers
    s = series.astype(str).fillna("").str.strip()
    # remove common currency symbols and words, keep digits and dots and commas and minus
    s = s.str.replace(r"[₹$€£,]|AED|INR|usd|rs\.?|AED", "", regex=True, flags=re.IGNORECASE)
    # replace any non digit/dot/minus with empty
    s = s.str.replace(r"[^0-9\.\-]", "", regex=True)
    # fix multiple dots
    s = s.str.replace(r"\.(?=.*\.)", "", regex=True)
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

if spend_col:
    # attempt numeric parse
    df["spend_num"] = parse_spend_series(df[spend_col])
else:
    # no per-row spend column detected -> default to 0
    df["spend_num"] = 0.0

# 2) If per-row spend parsed is mostly zeros, try to detect campaign-level spend columns
if df["spend_num"].sum() == 0 or (df["spend_num"]==0).mean() > 0.9:
    # look for campaign-level spend columns (common labels)
    candidate_cols = [c for c in df.columns if re.search(r"total.*spend|campaign.*spend|ad.*spend|spend.*total|total.*cost", c, re.I)]
    if candidate_cols:
        # choose first sensible candidate and try parse
        df["spend_num"] = parse_spend_series(df[candidate_cols[0]])
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
                break
        # if still nothing, keep zeros

# 3) Now compute aggregates correctly
# choose whether to dedupe BEFORE or AFTER spend aggregation. We will compute both so you can choose.
df_deduped = df[~df["duplicate"]]
df_raw = df

# per-campaign spend aggregation (sum of spend_num)
agg_raw = df_raw.groupby("campaign_mapped").agg(
    leads_raw=("campaign_mapped","size"),
    spend_raw=("spend_num","sum")
).reset_index()

agg_dedup = df_deduped.groupby("campaign_mapped").agg(
    leads_unique=("campaign_mapped","size"),
    spend_unique=("spend_num","sum")
).reset_index()

# merge to one table
agg = pd.merge(agg_raw, agg_dedup, on="campaign_mapped", how="left").fillna(0)

# Calculate CPLs explicitly with protective checks
# CPL_raw = spend_raw / leads_raw ; CPL_unique = spend_unique / leads_unique
agg["CPL_raw"] = agg.apply(lambda r: (r["spend_raw"] / r["leads_raw"]) if r["leads_raw"]>0 else 0, axis=1)
agg["CPL_unique"] = agg.apply(lambda r: (r["spend_unique"] / r["leads_unique"]) if r["leads_unique"]>0 else 0, axis=1)

# For dashboard choose CPL_unique (deduped) unless include_duplicates True
if include_duplicates:
    agg["CPL_display"] = agg["CPL_raw"]
else:
    agg["CPL_display"] = agg["CPL_unique"]

# Totals (show both raw and deduped)
total_spend_raw = agg["spend_raw"].sum()
total_spend_unique = agg["spend_unique"].sum()
total_leads_raw = len(df_raw)
total_leads_unique = len(df_deduped)
avg_cpl_raw = (total_spend_raw / total_leads_raw) if total_leads_raw>0 else 0
avg_cpl_unique = (total_spend_unique / total_leads_unique) if total_leads_unique>0 else 0

# Now the rest of the code should use agg["CPL_display"] and total_* variables for highlights
# -------------------------------------------------------------------------------------------------------
