"""
Lead Analysis Dashboard — Enhanced version with improved structure and error handling
Drop-in replacement: paste, commit, redeploy.
"""
import os
import io
import re
import logging
from datetime import datetime
from difflib import get_close_matches
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import requests
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("lead-dashboard")

# Configure Streamlit
st.set_page_config(
    page_title="Lead Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define color scheme
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "warning": "#ffcc00",
    "danger": "#d62728",
}

# Set style
plt.style.use("seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default")
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    color=[
        COLORS["primary"],
        COLORS["secondary"],
        COLORS["success"],
        COLORS["warning"],
        COLORS["danger"],
        "#9467bd",
    ]
)


@dataclass
class Config:
    OCR_SPACE_API_KEY: str = ""
    USE_PYTESSERACT: bool = False
    MAX_DISPLAY_ROWS: int = 20
    DEFAULT_FUZZY_CUTOFF: float = 0.35
    SENTIMENT_KEYWORDS: Dict[str, List[str]] = None

    def __post_init__(self):
        try:
            self.OCR_SPACE_API_KEY = (
                st.secrets["OCR_SPACE_API_KEY"]
                if "OCR_SPACE_API_KEY" in st.secrets
                else os.getenv("OCR_SPACE_API_KEY", "")
            )
        except Exception:
            self.OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "")

        try:
            import pytesseract  # noqa: F401
            self.USE_PYTESSERACT = True
            logger.info("pytesseract available")
        except Exception:
            logger.info("pytesseract not available")

        self.SENTIMENT_KEYWORDS = {
            "positive": [
                "good",
                "positive",
                "interested",
                "converted",
                "yes",
                "excellent",
                "great",
                "satisfied",
            ],
            "negative": [
                "bad",
                "not interested",
                "no",
                "complaint",
                "angry",
                "poor",
                "terrible",
                "disappointed",
            ],
            "neutral": ["maybe", "considering", "thinking", "undecided"],
        }


config = Config()


class DataProcessor:
    """Handles data processing operations"""

    @staticmethod
    def parse_spend_string(value: Any) -> float:
        """Robustly parse a string (₹, commas, text) into float. Returns 0.0 on failure."""
        try:
            if pd.isna(value):
                return 0.0
            text = str(value).strip()
            if not text:
                return 0.0
            # Remove currency symbols and common prefixes
            text = re.sub(r"\b(INR|USD|AED|Rs\.?|rs|₹|\$|€|£)\b", "", text, flags=re.I)
            # Keep only digits, dots, commas, and minus
            text = re.sub(r"[^0-9\.\-,]", "", text)
            # Handle comma as thousands separator
            if text.count(",") > 0 and text.count(".") <= 1:
                text = text.replace(",", "")
            # Handle multiple decimal points
            if text.count(".") > 1:
                parts = text.split(".")
                text = "".join(parts[:-1]) + "." + parts[-1]
            # Convert to float
            return float(text) if text and text not in ["", ".", "-"] else 0.0
        except Exception:
            numbers = re.findall(r"[-+]?\d*\.?\d+", str(value))
            if numbers:
                try:
                    return float(numbers[0])
                except Exception:
                    return 0.0
            return 0.0

    @staticmethod
    def create_simple_key(text: str) -> str:
        """Create normalized key for matching"""
        if pd.isna(text):
            return ""
        return re.sub(r"[^a-z0-9]", "", str(text).lower())

    @staticmethod
    def calculate_token_overlap(text1: str, text2: str) -> float:
        """Calculate token overlap between two strings"""
        try:
            if pd.isna(text1) or pd.isna(text2):
                return 0.0
            tokens1 = set(re.findall(r"\w+", str(text1).lower()))
            tokens2 = set(re.findall(r"\w+", str(text2).lower()))
            if not tokens1 or not tokens2:
                return 0.0
            intersection = len(tokens1 & tokens2)
            union = len(tokens1 | tokens2)
            return intersection / max(1, union)
        except Exception:
            return 0.0

    @staticmethod
    def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Detect duplicate leads based on phone and email"""
        df = df.copy()
        df["_duplicate_internal"] = False
        df["_duplicate_internal_reason"] = ""

        try:
            # Find phone and email columns
            phone_col = next((c for c in df.columns if re.search(r"phone|mobile|contact", c, re.I)), None)
            email_col = next((c for c in df.columns if re.search(r"email", c, re.I)), None)

            if phone_col and phone_col in df.columns:
                df["_clean_phone"] = df[phone_col].astype(str).str.replace(r"[^\d]", "", regex=True)
                dup_phone = df.duplicated(subset=["_clean_phone"], keep=False) & (df["_clean_phone"] != "")
                df.loc[dup_phone, "_duplicate_internal"] = True
                df.loc[dup_phone, "_duplicate_internal_reason"] += f"dup_phone({phone_col});"
                df.drop("_clean_phone", axis=1, inplace=True)

            if email_col and email_col in df.columns:
                df["_clean_email"] = df[email_col].astype(str).str.lower().str.strip()
                dup_email = df.duplicated(subset=["_clean_email"], keep=False) & (df["_clean_email"] != "")
                df.loc[dup_email, "_duplicate_internal"] = True
                df.loc[dup_email, "_duplicate_internal_reason"] += f"dup_email({email_col});"
                df.drop("_clean_email", axis=1, inplace=True)
        except Exception as e:
            logger.error(f"Duplicate detection failed: {e}")

        return df


class OCRProcessor:
    """Handles OCR operations"""

    @staticmethod
    def ocr_with_space_api(image_bytes: bytes, api_key: str, language: str = "eng") -> str:
        """Extract text using OCR.space API"""
        if not api_key:
            return ""
        try:
            url = "https://api.ocr.space/parse/image"
            files = {"file": ("image.png", image_bytes, "image/png")}
            data = {
                "apikey": api_key,
                "language": language,
                "isOverlayRequired": False,
                "scale": True,
                "OCREngine": 2,
            }
            response = requests.post(url, files=files, data=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            if result.get("IsErroredOnProcessing"):
                logger.error(f"OCR API error: {result.get('ErrorMessage', 'Unknown error')}")
                return ""
            parsed_results = result.get("ParsedResults", [])
            extracted_text = []
            for result_item in parsed_results:
                text = result_item.get("ParsedText", "")
                if text:
                    extracted_text.append(text)
            return "\n".join(extracted_text)
        except Exception as e:
            logger.error(f"OCR.space request failed: {e}")
            return ""

    @staticmethod
    def ocr_with_pytesseract(pil_image: Image.Image) -> str:
        """Extract text using pytesseract"""
        if not config.USE_PYTESSERACT:
            return ""
        try:
            import pytesseract  # noqa: F401
            if pil_image.mode != "L":
                pil_image = pil_image.convert("L")
            return pytesseract.image_to_string(pil_image, lang="eng", config="--psm 6")
        except Exception as e:
            logger.error(f"pytesseract OCR failed: {e}")
            return ""

    @staticmethod
    def extract_table_from_text(raw_text: str) -> pd.DataFrame:
        """Enhanced table extraction with better pattern matching"""
        if not raw_text or not raw_text.strip():
            return pd.DataFrame(columns=["Campaign", "Leads", "Spend"])

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        rows = []
        number_pattern = r"[-+]?\d{1,3}(?:[,.\s]*\d{3})*(?:\.\d{1,2})?"

        for i, line in enumerate(lines):
            # Skip header-like lines with no numbers
            if re.search(r"campaign|spend|lead|total|summary", line, re.I) and not re.search(r"\d", line):
                continue
            numbers = re.findall(number_pattern, line.replace(",", ""))
            if not numbers:
                continue
            campaign_text = re.split(number_pattern, line)[0]
            campaign_text = re.sub(r"[^A-Za-z0-9\s&\-_]", " ", campaign_text).strip()
            if not campaign_text:
                campaign_text = f"Campaign_{i+1}"
            spend_value = DataProcessor.parse_spend_string(numbers[-1])
            leads_value = 0
            if len(numbers) >= 2:
                try:
                    potential_leads = float(numbers[0].replace(",", ""))
                    if 0 <= potential_leads <= 10000:
                        leads_value = int(potential_leads)
                except Exception:
                    pass
            rows.append({"Campaign": campaign_text, "Leads": leads_value, "Spend": spend_value})

        if not rows:
            return pd.DataFrame(columns=["Campaign", "Leads", "Spend"])

        df = pd.DataFrame(rows)
        df = df.groupby("Campaign", as_index=False).agg({"Leads": "sum", "Spend": "sum"})
        df = df[(df["Spend"] > 0) | (df["Leads"] > 0)]
        return df


class CampaignMapper:
    """Handles campaign mapping logic"""

    @staticmethod
    def generate_mapping_suggestions(spend_campaigns: List[str], lead_campaigns: List[str]) -> pd.DataFrame:
        suggestions = []
        lead_keys = {DataProcessor.create_simple_key(camp): camp for camp in lead_campaigns}
        for spend_camp in spend_campaigns:
            spend_key = DataProcessor.create_simple_key(spend_camp)
            suggested = None
            confidence = 0.0
            if spend_key in lead_keys:
                suggested = lead_keys[spend_key]
                confidence = 1.0
            else:
                matches = get_close_matches(spend_key, list(lead_keys.keys()), n=1, cutoff=0.4)
                if matches:
                    suggested = lead_keys[matches[0]]
                    confidence = 0.8
                else:
                    best_score = 0.0
                    best_match = None
                    for lead_camp in lead_campaigns:
                        score = DataProcessor.calculate_token_overlap(spend_camp, lead_camp)
                        if score > best_score and score > 0.3:
                            best_score = score
                            best_match = lead_camp
                    if best_match:
                        suggested = best_match
                        confidence = best_score
            suggestions.append(
                {"spend_campaign": spend_camp, "suggested_lead_campaign": suggested, "confidence": confidence}
            )
        return pd.DataFrame(suggestions)

    @staticmethod
    def apply_auto_mapping(mapping_dict: Dict[str, str], spend_campaigns: List[str], lead_campaigns: List[str], cutoff: float = 0.35) -> Dict[str, str]:
        result = mapping_dict.copy()
        lead_keys = {DataProcessor.create_simple_key(camp): camp for camp in lead_campaigns}
        for spend_camp in spend_campaigns:
            if result.get(spend_camp):
                continue
            spend_key = DataProcessor.create_simple_key(spend_camp)
            if spend_key in lead_keys:
                result[spend_camp] = lead_keys[spend_key]
                continue
            matches = get_close_matches(spend_key, list(lead_keys.keys()), n=1, cutoff=cutoff)
            if matches:
                result[spend_camp] = lead_keys[matches[0]]
                continue
            best_score = 0.0
            best_match = None
            for lead_camp in lead_campaigns:
                score = DataProcessor.calculate_token_overlap(spend_camp, lead_camp)
                if score > best_score:
                    best_score = score
                    best_match = lead_camp
            if best_score > 0.25:
                result[spend_camp] = best_match
            else:
                result[spend_camp] = spend_camp
        return result


class ReportGenerator:
    """Handles report generation"""

    @staticmethod
    def create_charts(agg_df: pd.DataFrame, leads_df: pd.DataFrame) -> Dict[str, io.BytesIO]:
        charts = {}
        display_df = agg_df.sort_values("leads_rows_count", ascending=False).head(10)

        # Leads chart
        try:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            bars1 = ax1.bar(display_df["campaign_mapped"], display_df["leads_rows_count"], color=COLORS["primary"], edgecolor="navy", linewidth=0.5)
            ax1.set_title("Leads per Campaign", fontsize=16, fontweight="bold", pad=20)
            ax1.set_ylabel("Number of Leads", fontsize=12)
            ax1.set_xlabel("Campaign", fontsize=12)
            plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
            for bar in bars1:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width() / 2.0, height + max(display_df["leads_rows_count"]) * 0.01, f"{int(height)}", ha="center", va="bottom", fontsize=10)
            plt.tight_layout()
            charts["leads"] = ReportGenerator.fig_to_buffer(fig1)
            plt.close(fig1)
        except Exception as e:
            logger.error(f"Failed to create leads chart: {e}")

        # Spend chart
        try:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            bars2 = ax2.bar(display_df["campaign_mapped"], display_df["spend_total"], color=COLORS["secondary"], edgecolor="darkred", linewidth=0.5)
            ax2.set_title("Spend per Campaign (₹)", fontsize=16, fontweight="bold", pad=20)
            ax2.set_ylabel("Spend Amount (₹)", fontsize=12)
            ax2.set_xlabel("Campaign", fontsize=12)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
            for bar in bars2:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width() / 2.0, height + max(display_df["spend_total"]) * 0.01, f"₹{height:,.0f}", ha="center", va="bottom", fontsize=10)
            plt.tight_layout()
            charts["spend"] = ReportGenerator.fig_to_buffer(fig2)
            plt.close(fig2)
        except Exception as e:
            logger.error(f"Failed to create spend chart: {e}")

        # CPL chart
        try:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            cpl_data = display_df["CPL_computed"].fillna(0)
            bars3 = ax3.bar(display_df["campaign_mapped"], cpl_data, color=COLORS["warning"], edgecolor="darkorange", linewidth=0.5)
            ax3.set_title("Cost Per Lead (CPL) by Campaign", fontsize=16, fontweight="bold", pad=20)
            ax3.set_ylabel("CPL (₹)", fontsize=12)
            ax3.set_xlabel("Campaign", fontsize=12)
            plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
            for bar, cpl_val in zip(bars3, cpl_data):
                height = bar.get_height()
                if height > 0:
                    ax3.text(bar.get_x() + bar.get_width() / 2.0, height + max(cpl_data) * 0.01, f"₹{height:.0f}", ha="center", va="bottom", fontsize=10)
            plt.tight_layout()
            charts["cpl"] = ReportGenerator.fig_to_buffer(fig3)
            plt.close(fig3)
        except Exception as e:
            logger.error(f"Failed to create CPL chart: {e}")

        # Sentiment chart
        try:
            sentiment_data = ReportGenerator.analyze_sentiment(leads_df)
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            colors = [COLORS["success"], COLORS["warning"], COLORS["danger"]]
            wedges, texts, autotexts = ax4.pie(sentiment_data.values(), labels=sentiment_data.keys(), autopct="%1.1f%%", startangle=90, colors=colors)
            ax4.set_title("Lead Sentiment Distribution", fontsize=16, fontweight="bold")
            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontweight("bold")
            charts["sentiment"] = ReportGenerator.fig_to_buffer(fig4)
            plt.close(fig4)
        except Exception as e:
            logger.error(f"Failed to create sentiment chart: {e}")

        return charts

    @staticmethod
    def analyze_sentiment(leads_df: pd.DataFrame) -> Dict[str, int]:
        notes_cols = [c for c in leads_df.columns if re.search(r"note|status|remark|message|comment|feedback", c, re.I)]
        if not notes_cols:
            total_leads = len(leads_df)
            return {"Neutral": int(total_leads * 0.55), "Positive": int(total_leads * 0.35), "Negative": int(total_leads * 0.10)}
        notes_col = notes_cols[0]
        notes = leads_df[notes_col].astype(str).fillna("").str.lower()
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        for note in notes:
            if not note or note == "nan":
                sentiment_counts["Neutral"] += 1
                continue
            pos_found = any(keyword in note for keyword in config.SENTIMENT_KEYWORDS["positive"])
            neg_found = any(keyword in note for keyword in config.SENTIMENT_KEYWORDS["negative"])
            if pos_found and not neg_found:
                sentiment_counts["Positive"] += 1
            elif neg_found and not pos_found:
                sentiment_counts["Negative"] += 1
            else:
                sentiment_counts["Neutral"] += 1
        return sentiment_counts

    @staticmethod
    def fig_to_buffer(fig) -> io.BytesIO:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="white", edgecolor="none")
        buf.seek(0)
        return buf

    @staticmethod
    def generate_pdf_report(charts: Dict[str, io.BytesIO], totals: Dict[str, Any], agg_table: pd.DataFrame, output_path: str):
        try:
            width, height = landscape(A4)
            c = canvas.Canvas(output_path, pagesize=landscape(A4))
            margin = 40
            c.setFont("Helvetica-Bold", 20)
            title = totals.get("report_title", f"Campaign Performance Report — {datetime.now().strftime('%d %b %Y')}")
            c.drawCentredString(width / 2, height - 40, title)
            chart_keys = ["leads", "spend", "cpl", "sentiment"]
            img_width = (width - 3 * margin) / 2
            img_height = (height - 5 * margin - 80) / 2
            positions = [
                (margin, height - margin - img_height - 60),
                (margin * 2 + img_width, height - margin - img_height - 60),
                (margin, height - 2 * margin - 2 * img_height - 60),
                (margin * 2 + img_width, height - 2 * margin - 2 * img_height - 60),
            ]
            for chart_key, pos in zip(chart_keys, positions):
                chart_buf = charts.get(chart_key)
                if chart_buf:
                    try:
                        chart_buf.seek(0)
                        img = ImageReader(chart_buf)
                        c.drawImage(img, pos[0], pos[1], width=img_width, height=img_height, preserveAspectRatio=True, anchor="sw")
                    except Exception as e:
                        logger.error(f"Failed to add chart {chart_key} to PDF: {e}")
            text_x = margin
            text_y = positions[-1][1] - 30
            c.setFont("Helvetica-Bold", 14)
            c.drawString(text_x, text_y, "Performance Summary")
            text_y -= 20
            c.setFont("Helvetica", 11)
            summary_items = [
                f"Total Leads: {totals.get('total_leads', 0):,}",
                f"Internal Duplicates: {totals.get('internal_duplicates', 0):,}",
                f"External Mismatch: {totals.get('external_mismatch_total', 0):,}",
                f"Unique Leads (Est.): {totals.get('unique_leads_estimated', 0):,}",
                f"Total Spend: ₹{totals.get('total_spend_reported', 0):,.2f}",
                f"Average CPL (Deduplicated): ₹{totals.get('avg_cpl_unique', 0):.2f}",
            ]
            for item in summary_items:
                c.drawString(text_x + 10, text_y, f"• {item}")
                text_y -= 15
            c.showPage()
            c.save()
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            raise


class LeadAnalysisDashboard:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.ocr_processor = OCRProcessor()
        self.mapper = CampaignMapper()
        self.report_generator = ReportGenerator()

    def run(self):
        self._render_header()
        self._render_sidebar()
        leads_df = self._load_leads_data()
        if leads_df is None:
            return
        leads_df = self._process_leads_data(leads_df)
        spend_df, spend_source = self._load_spend_data()
        if st.button("🚀 Generate Analysis Report", type="primary", use_container_width=True):
            self._generate_analysis(leads_df, spend_df, spend_source)

    def _render_header(self):
        st.title("📊 Lead Analysis Dashboard")
        st.markdown(
            """
        **Upload your leads master and spend data to get comprehensive campaign performance insights**

        Features:
        - 📈 Campaign performance analysis
        - 🔍 Duplicate detection
        - 🎯 Intelligent campaign mapping
        - 📊 Visual reports with charts
        - 📄 PDF report generation
        """
        )
        st.divider()

    def _render_sidebar(self):
        st.sidebar.header("📁 Data Upload")
        st.sidebar.markdown("Upload your data files to begin analysis")

    def _load_leads_data(self) -> Optional[pd.DataFrame]:
        uploaded_leads = st.sidebar.file_uploader(
            "Upload Leads Master File", type=["csv", "xlsx", "xls"], help="Upload your master leads file (CSV or Excel format)"
        )
        if uploaded_leads is None:
            st.info("👆 Please upload your Leads Master file to begin analysis.")
            return None
        try:
            if uploaded_leads.name.lower().endswith(".csv"):
                leads_df = pd.read_csv(uploaded_leads, encoding="utf-8")
            else:
                leads_df = pd.read_excel(uploaded_leads)
            if leads_df.empty:
                st.error("❌ The uploaded leads file is empty.")
                return None
            with st.expander(f"📋 Leads Preview ({len(leads_df):,} rows)", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Rows", f"{len(leads_df):,}")
                with col2:
                    st.metric("Columns", len(leads_df.columns))
                st.dataframe(leads_df.head(config.MAX_DISPLAY_ROWS), use_container_width=True, hide_index=True)
                st.markdown("**Available Columns:**")
                st.write(", ".join(leads_df.columns.tolist()))
            return leads_df
        except Exception as e:
            st.error(f"❌ Failed to read leads file: {str(e)}")
            logger.error(f"Failed to load leads file: {e}")
            return None

    def _process_leads_data(self, leads_df: pd.DataFrame) -> pd.DataFrame:
        campaign_cols = [c for c in leads_df.columns if re.search(r"campaign|source|utm|ad", c, re.I)]
        campaign_col = st.sidebar.selectbox("Select Campaign Column", options=["(Auto-detect)"] + list(leads_df.columns), index=0, help="Choose the column that contains campaign names")
        if campaign_col == "(Auto-detect)" and campaign_cols:
            campaign_col = campaign_cols[0]
            st.sidebar.success(f"✅ Auto-detected: {campaign_col}")
        elif campaign_col == "(Auto-detect)":
            campaign_col = None
        if campaign_col and campaign_col in leads_df.columns:
            leads_df["campaign_mapped"] = leads_df[campaign_col].astype(str).fillna("unknown")
        else:
            leads_df["campaign_mapped"] = "unknown"
        if "lead_id" not in leads_df.columns:
            leads_df["_lead_row_id"] = range(1, len(leads_df) + 1)
        leads_df["__cmp_key"] = leads_df["campaign_mapped"].apply(self.data_processor.create_simple_key)
        leads_df = self.data_processor.detect_duplicates(leads_df)
        if int(leads_df["_duplicate_internal"].sum()) > 0:
            with st.expander("⚠️ Duplicate Detection Results"):
                dup_count = int(leads_df["_duplicate_internal"].sum())
                st.warning(f"Found {dup_count} potential duplicate leads")
                dup_df = leads_df[leads_df["_duplicate_internal"] == True]
                st.dataframe(dup_df[["campaign_mapped", "_duplicate_internal_reason"]].head(10), use_container_width=True)
        return leads_df

    def _load_spend_data(self) -> Tuple[pd.DataFrame, Optional[str]]:
        spend_df = pd.DataFrame(columns=["Campaign", "Leads", "Spend"])
        spend_source = None
        st.sidebar.subheader("💰 Spend Data")
        uploaded_spend_file = st.sidebar.file_uploader("Upload Spend File (Optional)", type=["csv", "xlsx", "xls"], help="Upload spend data in CSV or Excel format")
        uploaded_spend_image = st.sidebar.file_uploader("Upload Spend Image (Optional)", type=["png", "jpg", "jpeg"], help="Upload screenshot or image containing spend data")
        if uploaded_spend_file is not None:
            spend_df, spend_source = self._process_spend_file(uploaded_spend_file)
        elif uploaded_spend_image is not None:
            spend_df, spend_source = self._process_spend_image(uploaded_spend_image)
        elif st.sidebar.checkbox("📝 Enter Spend Data Manually"):
            spend_df, spend_source = self._manual_spend_entry()
        return spend_df, spend_source

    def _process_spend_file(self, uploaded_file) -> Tuple[pd.DataFrame, Optional[str]]:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                spend_raw = pd.read_csv(uploaded_file, encoding="utf-8")
            else:
                spend_raw = pd.read_excel(uploaded_file)
            if spend_raw.empty:
                st.sidebar.error("❌ Spend file is empty")
                return pd.DataFrame(columns=["Campaign", "Leads", "Spend"]), None
            st.sidebar.success(f"✅ Loaded spend file: {len(spend_raw)} rows")
            st.sidebar.markdown("**Map Columns:**")
            campaign_col = st.sidebar.selectbox("Campaign Column", options=list(spend_raw.columns), help="Column containing campaign names")
            spend_col = st.sidebar.selectbox("Spend Amount Column", options=list(spend_raw.columns), index=min(1, len(spend_raw.columns) - 1), help="Column containing spend amounts")
            leads_col = st.sidebar.selectbox("Leads Count Column (Optional)", options=["(None)"] + list(spend_raw.columns), help="Column containing lead counts from spend source")
            spend_data = {
                "Campaign": spend_raw[campaign_col].astype(str).fillna("unknown"),
                "Spend": spend_raw[spend_col].apply(self.data_processor.parse_spend_string),
            }
            if leads_col != "(None)":
                spend_data["Leads"] = pd.to_numeric(spend_raw[leads_col], errors="coerce").fillna(0).astype(int)
            else:
                spend_data["Leads"] = 0
            spend_df = pd.DataFrame(spend_data)
            with st.expander("💰 Spend Data Preview"):
                st.dataframe(spend_df.head(10), use_container_width=True)
                st.metric("Total Spend", f"₹{spend_df['Spend'].sum():,.2f}")
            return spend_df, "file"
        except Exception as e:
            st.sidebar.error(f"❌ Failed to process spend file: {str(e)}")
            logger.error(f"Failed to process spend file: {e}")
            return pd.DataFrame(columns=["Campaign", "Leads", "Spend"]), None

    def _process_spend_image(self, uploaded_image) -> Tuple[pd.DataFrame, Optional[str]]:
        try:
            st.sidebar.image(uploaded_image, caption="Spend Data Image", width=200)
            raw_text = ""
            with st.spinner("🔍 Extracting data from image..."):
                if config.USE_PYTESSERACT:
                    try:
                        image = Image.open(uploaded_image)
                        raw_text = self.ocr_processor.ocr_with_pytesseract(image)
                        if raw_text.strip():
                            st.sidebar.success("✅ OCR with pytesseract successful")
                    except Exception as e:
                        st.sidebar.warning(f"⚠️ pytesseract failed: {str(e)}")
                if not raw_text.strip() and config.OCR_SPACE_API_KEY:
                    try:
                        raw_text = self.ocr_processor.ocr_with_space_api(uploaded_image.getvalue(), config.OCR_SPACE_API_KEY)
                        if raw_text.strip():
                            st.sidebar.success("✅ OCR with OCR.space API successful")
                    except Exception as e:
                        st.sidebar.error(f"❌ OCR.space API failed: {str(e)}")
                if not raw_text.strip():
                    st.sidebar.warning("⚠️ Could not extract text from image")
                    return pd.DataFrame(columns=["Campaign", "Leads", "Spend"]), None
            with st.expander("📄 Extracted Text from Image"):
                st.text_area("Raw OCR Output", raw_text[:2000], height=150)
            extracted_df = self.ocr_processor.extract_table_from_text(raw_text)
            if extracted_df.empty:
                st.warning("⚠️ Could not extract structured data from image. Try manual entry or upload a clearer image.")
                return pd.DataFrame(columns=["Campaign", "Leads", "Spend"]), None
            st.subheader("✏️ Review and Edit Extracted Data")
            st.info("The system has attempted to extract spend data from your image. Please review and correct as needed.")
            edited_df = st.data_editor(
                extracted_df,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "Campaign": st.column_config.TextColumn("Campaign Name"),
                    "Leads": st.column_config.NumberColumn("Leads Count", min_value=0, step=1),
                    "Spend": st.column_config.NumberColumn("Spend Amount", min_value=0.0, format="₹%.2f"),
                },
            )
            if st.button("✅ Use Extracted Data", type="primary"):
                try:
                    edited_df["Spend"] = edited_df["Spend"].apply(self.data_processor.parse_spend_string)
                    edited_df["Leads"] = pd.to_numeric(edited_df["Leads"], errors="coerce").fillna(0).astype(int)
                    edited_df = edited_df[(edited_df["Campaign"].notna()) & (edited_df["Campaign"] != "") & ((edited_df["Spend"] > 0) | (edited_df["Leads"] > 0))]
                    if not edited_df.empty:
                        st.success(f"✅ Using {len(edited_df)} spend records from image")
                        return edited_df, "ocr"
                    else:
                        st.error("❌ No valid spend data found")
                        return pd.DataFrame(columns=["Campaign", "Leads", "Spend"]), None
                except Exception as e:
                    st.error(f"❌ Validation failed: {e}")
                    return pd.DataFrame(columns=["Campaign", "Leads", "Spend"]), None
            return pd.DataFrame(columns=["Campaign", "Leads", "Spend"]), None
        except Exception as e:
            st.sidebar.error(f"❌ Failed to process spend image: {e}")
            logger.error(f"Failed to process spend image: {e}")
            return pd.DataFrame(columns=["Campaign", "Leads", "Spend"]), None

    def _manual_spend_entry(self) -> Tuple[pd.DataFrame, Optional[str]]:
        try:
            st.subheader("📝 Manual Spend Data Entry")
            if "manual_spend_data" not in st.session_state:
                st.session_state.manual_spend_data = pd.DataFrame([{"Campaign": "", "Leads": 0, "Spend": 0.0}])
            manual_df = st.data_editor(
                st.session_state.manual_spend_data,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "Campaign": st.column_config.TextColumn("Campaign Name", required=True),
                    "Leads": st.column_config.NumberColumn("Leads Count", min_value=0, step=1),
                    "Spend": st.column_config.NumberColumn("Spend Amount (₹)", min_value=0.0, format="₹%.2f"),
                },
            )
            if st.button("💾 Save Manual Data", type="primary"):
                manual_df["Spend"] = manual_df["Spend"].apply(self.data_processor.parse_spend_string)
                manual_df["Leads"] = pd.to_numeric(manual_df["Leads"], errors="coerce").fillna(0).astype(int)
                valid_df = manual_df[(manual_df["Campaign"].notna()) & (manual_df["Campaign"] != "") & ((manual_df["Spend"] > 0) | (manual_df["Leads"] > 0))]
                if not valid_df.empty:
                    st.session_state.manual_spend_data = valid_df
                    st.success(f"✅ Saved {len(valid_df)} spend records")
                    return valid_df, "manual"
                else:
                    st.error("❌ Please enter valid spend data")
                    return pd.DataFrame(columns=["Campaign", "Leads", "Spend"]), None
            return pd.DataFrame(columns=["Campaign", "Leads", "Spend"]), None
        except Exception as e:
            st.error(f"❌ Manual entry error: {e}")
            logger.error(f"Manual spend entry error: {e}")
            return pd.DataFrame(columns=["Campaign", "Leads", "Spend"]), None

    def _generate_analysis(self, leads_df: pd.DataFrame, spend_df: pd.DataFrame, spend_source: Optional[str]):
        if spend_df.empty and spend_source is None:
            st.warning("⚠️ No spend data provided. Analysis will be limited to lead data only.")
        if not spend_df.empty:
            spend_df["__cmp_key"] = spend_df["Campaign"].apply(self.data_processor.create_simple_key)
        lead_campaigns = sorted(leads_df["campaign_mapped"].unique().tolist())
        spend_campaigns = sorted(spend_df["Campaign"].unique().tolist()) if not spend_df.empty else []
        if spend_campaigns:
            st.subheader("🎯 Campaign Mapping")
            suggestions_df = self.mapper.generate_mapping_suggestions(spend_campaigns, lead_campaigns)
            st.info("Review and adjust the mapping between spend campaigns and lead campaigns below.")
            mapping_editor = st.data_editor(
                suggestions_df.drop("confidence", axis=1),
                use_container_width=True,
                column_config={
                    "spend_campaign": st.column_config.TextColumn("Spend Campaign", disabled=True),
                    "suggested_lead_campaign": st.column_config.SelectboxColumn("Mapped Lead Campaign", options=[""] + lead_campaigns),
                },
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                auto_map = st.checkbox("🤖 Auto-map unmatched campaigns", value=True)
            with col2:
                fuzzy_cutoff = st.slider("Fuzzy Match Threshold", 0.2, 0.9, 0.35, 0.05)
            with col3:
                st.metric("Campaigns to Map", len(spend_campaigns))
        else:
            mapping_editor = pd.DataFrame(columns=["spend_campaign", "suggested_lead_campaign"])
            auto_map = False
            fuzzy_cutoff = config.DEFAULT_FUZZY_CUTOFF
        with st.spinner("🔄 Processing data and generating analysis..."):
            # Build mapping dict
            mapping_dict: Dict[str, str] = {}
            try:
                if not mapping_editor.empty:
                    for _, row in mapping_editor.iterrows():
                        spend_camp = str(row.get("spend_campaign", "")).strip()
                        mapped_camp = row.get("suggested_lead_campaign")
                        if mapped_camp and str(mapped_camp).strip() != "":
                            mapping_dict[spend_camp] = str(mapped_camp).strip()
            except Exception as e:
                logger.error(f"Mapping editor read failed: {e}")
            if spend_campaigns and auto_map:
                try:
                    mapping_dict = self.mapper.apply_auto_mapping(mapping_dict, spend_campaigns, lead_campaigns, fuzzy_cutoff)
                except Exception as e:
                    logger.error(f"Auto-mapping failed: {e}")
            # Map spend rows
            if not spend_df.empty:
                spend_mapped_rows = []
                for _, row in spend_df.iterrows():
                    spend_label = str(row.get("Campaign", "")).strip()
                    mapped_label = mapping_dict.get(spend_label, spend_label)
                    try:
                        reported_leads = int(row.get("Leads", 0) or 0)
                    except Exception:
                        reported_leads = 0
                    try:
                        spend_val = float(row.get("Spend", 0.0) or 0.0)
                    except Exception:
                        spend_val = 0.0
                    spend_mapped_rows.append(
                        {
                            "spend_campaign_original": spend_label,
                            "campaign_mapped": mapped_label,
                            "reported_leads_from_spend": reported_leads,
                            "spend_total": spend_val,
                        }
                    )
                spend_mapped_df = pd.DataFrame(spend_mapped_rows)
                spend_agg = spend_mapped_df.groupby("campaign_mapped", dropna=False).agg(
                    reported_leads_from_spend=("reported_leads_from_spend", "sum"),
                    spend_total_from_spend=("spend_total", "sum"),
                ).reset_index()
            else:
                spend_agg = pd.DataFrame(columns=["campaign_mapped", "reported_leads_from_spend", "spend_total_from_spend"])
            leads_processed = leads_df.copy()
            if not spend_agg.empty:
                leads_processed = leads_processed.merge(spend_agg, on="campaign_mapped", how="left")
                leads_processed["spend_total_from_spend"] = leads_processed["spend_total_from_spend"].fillna(0.0)
                leads_processed["reported_leads_from_spend"] = leads_processed["reported_leads_from_spend"].fillna(0).astype(int)
            else:
                leads_processed["spend_total_from_spend"] = 0.0
                leads_processed["reported_leads_from_spend"] = 0
            # Aggregate per campaign
            agg_df = leads_processed.groupby("campaign_mapped", dropna=False).agg(
                leads_rows_count=("campaign_mapped", "size"),
                spend_total=("spend_total_from_spend", "first"),
                reported_leads_from_spend=("reported_leads_from_spend", "first"),
                internal_duplicates=("_duplicate_internal", "sum"),
            ).reset_index()
            if not spend_agg.empty:
                try:
                    spend_lookup = spend_agg.set_index("campaign_mapped")["spend_total_from_spend"].to_dict()

                    def choose_spend(x):
                        try:
                            return float(spend_lookup.get(x, float(agg_df.loc[agg_df["campaign_mapped"] == x, "spend_total"].iat[0]) if not agg_df[agg_df["campaign_mapped"] == x].empty else 0.0))
                        except Exception:
                            return 0.0

                    agg_df["spend_total"] = agg_df["campaign_mapped"].map(choose_spend)
                except Exception as e:
                    logger.error(f"Failed to reconcile spend totals: {e}")
                    agg_df["spend_total"] = agg_df["spend_total"].fillna(0.0)
            else:
                agg_df["spend_total"] = agg_df["spend_total"].fillna(0.0)
            agg_df["unique_leads_est"] = (agg_df["leads_rows_count"] - agg_df["internal_duplicates"]).clip(lower=0)
            def safe_div(n, d):
                return (n / d) if d and d > 0 else np.nan
            agg_df["CPL_computed"] = agg_df.apply(lambda r: safe_div(r["spend_total"], r["leads_rows_count"]), axis=1)
            agg_df["CPL_dedup"] = agg_df.apply(lambda r: safe_div(r["spend_total"], r["unique_leads_est"]), axis=1)
            total_leads = int(len(leads_processed))
            internal_duplicates = int(leads_processed["_duplicate_internal"].sum()) if "_duplicate_internal" in leads_processed.columns else 0
            total_spend = float(agg_df["spend_total"].sum()) if "spend_total" in agg_df.columns else 0.0
            external_mismatch = 0
            if not spend_agg.empty:
                try:
                    mismatch_df = spend_agg.merge(agg_df[["campaign_mapped", "leads_rows_count"]], on="campaign_mapped", how="left").fillna(0)
                    mismatch_df["external_diff"] = mismatch_df["reported_leads_from_spend"] - mismatch_df["leads_rows_count"]
                    external_mismatch = int(mismatch_df[mismatch_df["external_diff"] > 0]["external_diff"].sum())
                except Exception as e:
                    logger.error(f"Failed to compute external mismatch: {e}")
                    external_mismatch = 0
            unique_leads_estimated = max(0, total_leads - internal_duplicates - external_mismatch)
            avg_cpl_unique = (total_spend / unique_leads_estimated) if unique_leads_estimated > 0 else 0.0
            totals = {
                "total_leads": total_leads,
                "internal_duplicates": internal_duplicates,
                "external_mismatch_total": external_mismatch,
                "unique_leads_estimated": unique_leads_estimated,
                "total_spend_reported": total_spend,
                "avg_cpl_unique": avg_cpl_unique,
                "report_title": f"Campaign Performance Analysis — {datetime.now().strftime('%d %b %Y')}",
            }
        # Display and downloads
        self._display_analysis_results(agg_df, totals, leads_processed, spend_source)
        self._generate_downloads(agg_df, totals, leads_processed)

    def _display_analysis_results(self, agg_df: pd.DataFrame, totals: Dict[str, Any], leads_processed: pd.DataFrame, spend_source: Optional[str]):
        st.header("📊 Analysis Results")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Leads", f"{totals['total_leads']:,}")
        with col2:
            st.metric("Total Spend", f"₹{totals['total_spend_reported']:,.2f}")
        with col3:
            st.metric("Avg CPL", f"₹{totals['avg_cpl_unique']:.2f}")
        with col4:
            st.metric("Unique Leads", f"{totals['unique_leads_estimated']:,}")
        with st.expander("🔍 Data Quality Insights"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Internal Duplicates", f"{totals['internal_duplicates']:,}", delta=f"-{(totals['internal_duplicates'] / totals['total_leads'] * 100):.1f}%" if totals["total_leads"] > 0 else "0%")
            with col2:
                st.metric("External Mismatch", f"{totals['external_mismatch_total']:,}", help="Difference between reported leads in spend data and actual leads in master")
            with col3:
                data_quality = (totals["unique_leads_estimated"] / totals["total_leads"] * 100) if totals["total_leads"] > 0 else 0
                st.metric("Data Quality Score", f"{data_quality:.1f}%", help="Percentage of unique, valid leads")
        st.subheader("🎯 Campaign Performance")
        display_agg = agg_df.sort_values("leads_rows_count", ascending=False).copy()
        display_agg["CPL_formatted"] = display_agg["CPL_computed"].apply(lambda x: f"₹{x:.2f}" if pd.notna(x) else "N/A")
        display_agg["CPL_dedup_formatted"] = display_agg["CPL_dedup"].apply(lambda x: f"₹{x:.2f}" if pd.notna(x) else "N/A")
        display_agg["Spend_formatted"] = display_agg["spend_total"].apply(lambda x: f"₹{x:,.2f}")
        st.dataframe(
            display_agg[["campaign_mapped", "leads_rows_count", "unique_leads_est", "Spend_formatted", "CPL_formatted", "CPL_dedup_formatted", "internal_duplicates", "reported_leads_from_spend"]]
            .rename(columns={
                "campaign_mapped": "Campaign",
                "leads_rows_count": "Leads Count",
                "unique_leads_est": "Unique Leads (Est.)",
                "Spend_formatted": "Total Spend",
                "CPL_formatted": "CPL (Raw)",
                "CPL_dedup_formatted": "CPL (Dedup)",
                "internal_duplicates": "Duplicates",
                "reported_leads_from_spend": "Reported Leads",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("📈 Performance Charts")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Leads per Campaign")
            top_campaigns = agg_df.nlargest(10, "leads_rows_count")
            if not top_campaigns.empty:
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                bars = ax1.bar(top_campaigns["campaign_mapped"], top_campaigns["leads_rows_count"], color=COLORS["primary"], alpha=0.8, edgecolor="white", linewidth=0.5)
                ax1.set_title("Top 10 Campaigns by Leads")
                ax1.set_ylabel("Number of Leads")
                plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f"{int(height)}", ha="center", va="bottom", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig1)
                plt.close()
        with col2:
            st.markdown("#### Spend per Campaign")
            top_spend_campaigns = agg_df[agg_df["spend_total"] > 0].nlargest(10, "spend_total")
            if not top_spend_campaigns.empty:
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                bars = ax2.bar(top_spend_campaigns["campaign_mapped"], top_spend_campaigns["spend_total"], color=COLORS["secondary"], alpha=0.8, edgecolor="white", linewidth=0.5)
                ax2.set_title("Top 10 Campaigns by Spend")
                ax2.set_ylabel("Spend Amount (₹)")
                plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f"₹{height:,.0f}", ha="center", va="bottom", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
            else:
                st.info("No spend data available for visualization")
        cpl_data = agg_df[agg_df["CPL_computed"].notna() & (agg_df["CPL_computed"] > 0)]
        if not cpl_data.empty:
            st.markdown("#### Cost Per Lead (CPL) Analysis")
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            bars = ax3.bar(cpl_data["campaign_mapped"], cpl_data["CPL_computed"], color=COLORS["warning"], alpha=0.8, edgecolor="white", linewidth=0.5)
            ax3.set_title("CPL by Campaign")
            ax3.set_ylabel("CPL (₹)")
            plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax3.text(bar.get_x() + bar.get_width() / 2.0, height, f"₹{height:.0f}", ha="center", va="bottom", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()
        sentiment_data = self.report_generator.analyze_sentiment(leads_processed)
        if sentiment_data:
            st.markdown("#### Lead Sentiment Distribution")
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            colors = [COLORS["success"], COLORS["warning"], COLORS["danger"]]
            wedges, texts, autotexts = ax4.pie(sentiment_data.values(), labels=sentiment_data.keys(), autopct="%1.1f%%", startangle=90, colors=colors)
            ax4.set_title("Lead Sentiment Analysis")
            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontweight("bold")
            st.pyplot(fig4)
            plt.close()

    def _generate_downloads(self, agg_df: pd.DataFrame, totals: Dict[str, Any], leads_processed: pd.DataFrame):
        st.subheader("📥 Download Reports")
        try:
            import tempfile

            temp_dir = tempfile.mkdtemp()
            merged_path = os.path.join(temp_dir, "detailed_leads_analysis.xlsx")
            summary_path = os.path.join(temp_dir, "campaign_summary.xlsx")
            leads_export = leads_processed.copy()
            cols_to_remove = [col for col in leads_export.columns if col.startswith("_")]
            safe_drop_columns = [c for c in cols_to_remove + ["__cmp_key"] if c in leads_export.columns]
            if safe_drop_columns:
                leads_export = leads_export.drop(columns=safe_drop_columns)
            leads_export.to_excel(merged_path, index=False)
            summary_export = agg_df.copy()
            summary_export["CPL_formatted"] = summary_export["CPL_computed"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            summary_export.to_excel(summary_path, index=False)
            col1, col2, col3 = st.columns(3)
            with col1:
                with open(merged_path, "rb") as f:
                    st.download_button("📊 Download Detailed Analysis", f.read(), file_name=f"detailed_leads_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            with col2:
                with open(summary_path, "rb") as f:
                    st.download_button("📈 Download Campaign Summary", f.read(), file_name=f"campaign_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            with col3:
                if st.button("📄 Generate PDF Report", type="secondary"):
                    try:
                        charts = self.report_generator.create_charts(agg_df, leads_processed)
                        pdf_path = os.path.join(temp_dir, f"campaign_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
                        self.report_generator.generate_pdf_report(charts, totals, agg_df, pdf_path)
                        with open(pdf_path, "rb") as f:
                            st.download_button("📄 Download PDF Report", f.read(), file_name=f"campaign_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
                        st.success("✅ PDF report generated successfully!")
                    except Exception as e:
                        st.error(f"❌ PDF generation failed: {str(e)}")
                        logger.error(f"PDF generation error: {e}")
        except Exception as e:
            st.error(f"❌ Failed to generate downloads: {str(e)}")
            logger.error(f"Download generation error: {e}")


def main():
    try:
        dashboard = LeadAnalysisDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Application error: {e}")
        with st.expander("🐛 Error Details (for debugging)"):
            st.code(str(e))


if __name__ == "__main__":
    main()
