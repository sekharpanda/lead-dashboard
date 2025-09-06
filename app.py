            # ---- START PATCHED AGGREGATION & MAPPING LOGIC ----

            # Ensure mapping_editor exists (when no spend campaigns, avoid NameError)
            if not spend_campaigns:
                mapping_editor = pd.DataFrame(columns=["spend_campaign", "suggested_lead_campaign"])

            # Build mapping dictionary from editor (if present)
            mapping_dict = {}
            if not mapping_editor.empty:
                for _, row in mapping_editor.iterrows():
                    spend_camp = str(row.get("spend_campaign", "")).strip()
                    mapped_camp = row.get("suggested_lead_campaign")
                    if mapped_camp and str(mapped_camp).strip() != "":
                        mapping_dict[spend_camp] = str(mapped_camp).strip()

            # Apply auto-mapping if requested (if spend_campaigns exist)
            if spend_campaigns and 'auto_map' in locals() and auto_map:
                mapping_dict = self.mapper.apply_auto_mapping(
                    mapping_dict, spend_campaigns, lead_campaigns, fuzzy_cutoff
                )

            # Map spend rows to lead campaigns (if spend data present)
            if not spend_df.empty:
                spend_mapped_rows = []
                for _, row in spend_df.iterrows():
                    spend_label = str(row.get("Campaign", "")).strip()
                    mapped_label = mapping_dict.get(spend_label, spend_label)
                    spend_mapped_rows.append({
                        "spend_campaign_original": spend_label,
                        "campaign_mapped": mapped_label,
                        "reported_leads_from_spend": int(row.get("Leads", 0) or 0),
                        "spend_total": float(row.get("Spend", 0.0) or 0.0)
                    })

                spend_mapped_df = pd.DataFrame(spend_mapped_rows)

                # Aggregate spend and reported leads by mapped campaign (sum, not first)
                spend_agg = spend_mapped_df.groupby("campaign_mapped", dropna=False).agg(
                    reported_leads_from_spend=("reported_leads_from_spend", "sum"),
                    spend_total_from_spend=("spend_total", "sum")
                ).reset_index()
            else:
                spend_agg = pd.DataFrame(columns=["campaign_mapped", "reported_leads_from_spend", "spend_total_from_spend"])

            # Merge spend_agg into leads master so every lead has campaign-level spend info
            leads_processed = leads_df.copy()
            if not spend_agg.empty:
                leads_processed = leads_processed.merge(spend_agg, on="campaign_mapped", how="left")
                leads_processed["spend_total_from_spend"] = leads_processed["spend_total_from_spend"].fillna(0.0)
                leads_processed["reported_leads_from_spend"] = leads_processed["reported_leads_from_spend"].fillna(0).astype(int)
            else:
                leads_processed["spend_total_from_spend"] = 0.0
                leads_processed["reported_leads_from_spend"] = 0

            # Per-campaign aggregation (this is where we compute inside-campaign stats properly)
            # Use groupby size for total rows and aggregate sums for spend/duplicates etc.
            agg_df = leads_processed.groupby("campaign_mapped", dropna=False).agg(
                leads_rows_count=("campaign_mapped", "size"),
                spend_total=("spend_total_from_spend", "first"),  # spend_total is same across rows after merge
                reported_leads_from_spend=("reported_leads_from_spend", "first"),
                internal_duplicates=("_duplicate_internal", "sum")
            ).reset_index()

            # If spend_total was merged as zero for some rows but spend_agg has value, prefer the spend_agg sum
            if not spend_agg.empty:
                spend_lookup = spend_agg.set_index("campaign_mapped")["spend_total_from_spend"].to_dict()
                agg_df["spend_total"] = agg_df["campaign_mapped"].map(lambda x: spend_lookup.get(x, agg_df.loc[agg_df['campaign_mapped'] == x, 'spend_total'].iat[0]))

            # Compute unique leads per campaign (deduplicated), clamp at zero
            agg_df["unique_leads_est"] = (agg_df["leads_rows_count"] - agg_df["internal_duplicates"]).clip(lower=0)

            # Compute CPLs:
            # - raw CPL using rows_count (if spend and rows exist)
            # - dedup CPL using unique_leads_est
            def safe_div(n, d):
                return (n / d) if d and d > 0 else np.nan

            agg_df["CPL_computed"] = agg_df.apply(lambda r: safe_div(r["spend_total"], r["leads_rows_count"]), axis=1)
            agg_df["CPL_dedup"] = agg_df.apply(lambda r: safe_div(r["spend_total"], r["unique_leads_est"]), axis=1)

            # Totals & cross-campaign mismatch calculation
            total_leads = int(len(leads_processed))
            internal_duplicates = int(leads_processed["_duplicate_internal"].sum())
            total_spend = float(agg_df["spend_total"].sum())

            external_mismatch = 0
            if not spend_agg.empty:
                # Compare reported leads from spend vs actual master leads per mapped campaign
                mismatch_df = spend_agg.merge(
                    agg_df[["campaign_mapped", "leads_rows_count"]],
                    on="campaign_mapped",
                    how="left"
                ).fillna(0)
                mismatch_df["external_diff"] = mismatch_df["reported_leads_from_spend"] - mismatch_df["leads_rows_count"]
                # Count only positive mismatches (spend source claims more leads than master)
                external_mismatch = int(mismatch_df[mismatch_df["external_diff"] > 0]["external_diff"].sum())

            unique_leads_estimated = max(0, total_leads - internal_duplicates - external_mismatch)
            avg_cpl_unique = (total_spend / unique_leads_estimated) if unique_leads_estimated > 0 else 0.0

            totals = {
                "total_leads": total_leads,
                "internal_duplicates": internal_duplicates,
                "external_mismatch_total": external_mismatch,
                "unique_leads_estimated": unique_leads_estimated,
                "total_spend_reported": total_spend,
                "avg_cpl_unique": avg_cpl_unique,
                "report_title": f"Campaign Performance Analysis — {datetime.now().strftime('%d %b %Y')}"
            }

            # ---- END PATCH ----
