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
from collections import Counter
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'warning': '#ffcc00',
    'danger': '#d62728'
}

# Set style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
# Set a nice color palette for matplotlib
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
    COLORS['primary'], COLORS['secondary'], COLORS['success'],
    COLORS['warning'], COLORS['danger'], '#9467bd'
])

# Configuration
@dataclass
class Config:
    OCR_SPACE_API_KEY: str = ""
    USE_PYTESSERACT: bool = False
    MAX_DISPLAY_ROWS: int = 20
    DEFAULT_FUZZY_CUTOFF: float = 0.35
    SENTIMENT_KEYWORDS: Dict[str, List[str]] = None

    def __post_init__(self):
        self.OCR_SPACE_API_KEY = st.secrets.get("OCR_SPACE_API_KEY", "") if "OCR_SPACE_API_KEY" in st.secrets else os.getenv("OCR_SPACE_API_KEY", "")

        try:
            import pytesseract  # type: ignore
            self.USE_PYTESSERACT = True
            logger.info("pytesseract available")
        except Exception:
            logger.info("pytesseract not available")

        self.SENTIMENT_KEYWORDS = {
            'positive': ['good', 'positive', 'interested', 'converted', 'yes', 'excellent', 'great', 'satisfied'],
            'negative': ['bad', 'not interested', 'no', 'complaint', 'angry', 'poor', 'terrible', 'disappointed'],
            'neutral': ['maybe', 'considering', 'thinking', 'undecided']
        }

config = Config()

class DataProcessor:
    """Handles data processing operations"""

    @staticmethod
    def parse_spend_string(value: Any) -> float:
        """Robustly parse a string (₹, commas, text) into float. Returns 0.0 on failure."""
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
        if text.count(',') > 0 and text.count('.') <= 1:
            text = text.replace(",", "")

        # Handle multiple decimal points
        if text.count(".") > 1:
            parts = text.split(".")
            text = "".join(parts[:-1]) + "." + parts[-1]

        # Convert to float
        try:
            return float(text) if text and text not in ["", ".", "-"] else 0.0
        except ValueError:
            # Try to extract first number
            numbers = re.findall(r"[-+]?\d*\.?\d+", text)
            if numbers:
                try:
                    return float(numbers[0])
                except ValueError:
                    pass
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
        if pd.isna(text1) or pd.isna(text2):
            return 0.0

        tokens1 = set(re.findall(r"\w+", str(text1).lower()))
        tokens2 = set(re.findall(r"\w+", str(text2).lower()))

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / max(1, union)

    @staticmethod
    def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Detect duplicate leads based on phone and email"""
        df = df.copy()
        df["_duplicate_internal"] = False
        df["_duplicate_internal_reason"] = ""

        # Find phone and email columns
        phone_col = next((c for c in df.columns if re.search(r"phone|mobile|contact", c, re.I)), None)
        email_col = next((c for c in df.columns if re.search(r"email", c, re.I)), None)

        if phone_col and phone_col in df.columns:
            # Clean phone numbers for better matching
            df['_clean_phone'] = df[phone_col].astype(str).str.replace(r"[^\d]", "", regex=True)
            dup_phone = df.duplicated(subset=['_clean_phone'], keep=False) & (df['_clean_phone'] != "")
            df.loc[dup_phone, "_duplicate_internal"] = True
            df.loc[dup_phone, "_duplicate_internal_reason"] += f"dup_phone({phone_col});"
            df.drop('_clean_phone', axis=1, inplace=True)

        if email_col and email_col in df.columns:
            # Clean emails for better matching
            df['_clean_email'] = df[email_col].astype(str).str.lower().str.strip()
            dup_email = df.duplicated(subset=['_clean_email'], keep=False) & (df['_clean_email'] != "")
            df.loc[dup_email, "_duplicate_internal"] = True
            df.loc[dup_email, "_duplicate_internal_reason"] += f"dup_email({email_col});"
            df.drop('_clean_email', axis=1, inplace=True)

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
                "OCREngine": 2  # Use engine 2 for better accuracy
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

        except requests.RequestException as e:
            logger.error(f"OCR API request failed: {e}")
            return ""
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return ""

    @staticmethod
    def ocr_with_pytesseract(pil_image: Image.Image) -> str:
        """Extract text using pytesseract"""
        if not config.USE_PYTESSERACT:
            return ""

        try:
            import pytesseract  # type: ignore
            # Enhance image for better OCR
            # Convert to grayscale and increase contrast
            if pil_image.mode != 'L':
                pil_image = pil_image.convert('L')

            return pytesseract.image_to_string(pil_image, lang="eng", config='--psm 6')
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

        # Enhanced number regex
        number_pattern = r"[-+]?\d{1,3}(?:[,.\s]*\d{3})*(?:\.\d{1,2})?"

        for i, line in enumerate(lines):
            # Skip header-like lines
            if re.search(r"campaign|spend|lead|total|summary", line, re.I) and not re.search(r"\d", line):
                continue

            numbers = re.findall(number_pattern, line.replace(",", ""))
            if not numbers:
                continue

            # Extract campaign name (text before numbers)
            campaign_text = re.split(number_pattern, line)[0]
            campaign_text = re.sub(r"[^A-Za-z0-9\s&\-_]", " ", campaign_text).strip()

            if not campaign_text:
                campaign_text = f"Campaign_{i+1}"

            # Parse numbers - assume last is spend, first might be leads
            spend_value = DataProcessor.parse_spend_string(numbers[-1])
            leads_value = 0

            if len(numbers) >= 2:
                try:
                    potential_leads = float(numbers[0].replace(",", ""))
                    # Reasonable lead count check
                    if 0 <= potential_leads <= 10000:
                        leads_value = int(potential_leads)
                except (ValueError, TypeError):
                    pass

            rows.append({
                "Campaign": campaign_text,
                "Leads": leads_value,
                "Spend": spend_value
            })

        if not rows:
            return pd.DataFrame(columns=["Campaign", "Leads", "Spend"])

        df = pd.DataFrame(rows)

        # Consolidate similar campaigns
        df = df.groupby("Campaign", as_index=False).agg({
            "Leads": "sum",
            "Spend": "sum"
        })

        # Filter out invalid rows
        df = df[(df["Spend"] > 0) | (df["Leads"] > 0)]

        return df

class CampaignMapper:
    """Handles campaign mapping logic"""

    @staticmethod
    def generate_mapping_suggestions(spend_campaigns: List[str], lead_campaigns: List[str]) -> pd.DataFrame:
        """Generate intelligent mapping suggestions"""
        suggestions = []

        # Create lookup dictionaries
        lead_keys = {DataProcessor.create_simple_key(camp): camp for camp in lead_campaigns}

        for spend_camp in spend_campaigns:
            spend_key = DataProcessor.create_simple_key(spend_camp)
            suggested = None
            confidence = 0.0

            # Exact key match
            if spend_key in lead_keys:
                suggested = lead_keys[spend_key]
                confidence = 1.0
            else:
                # Fuzzy matching
                matches = get_close_matches(spend_key, list(lead_keys.keys()), n=1, cutoff=0.4)
                if matches:
                    suggested = lead_keys[matches[0]]
                    confidence = 0.8
                else:
                    # Token overlap matching
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

            suggestions.append({
                "spend_campaign": spend_camp,
                "suggested_lead_campaign": suggested,
                "confidence": confidence
            })

        return pd.DataFrame(suggestions)

    @staticmethod
    def apply_auto_mapping(mapping_dict: Dict[str, str], spend_campaigns: List[str],
                          lead_campaigns: List[str], cutoff: float = 0.35) -> Dict[str, str]:
        """Apply automatic mapping with fallbacks"""
        result = mapping_dict.copy()
        lead_keys = {DataProcessor.create_simple_key(camp): camp for camp in lead_campaigns}

        for spend_camp in spend_campaigns:
            if result.get(spend_camp):
                continue

            spend_key = DataProcessor.create_simple_key(spend_camp)

            # Try exact match
            if spend_key in lead_keys:
                result[spend_camp] = lead_keys[spend_key]
                continue

            # Try fuzzy match
            matches = get_close_matches(spend_key, list(lead_keys.keys()), n=1, cutoff=cutoff)
            if matches:
                result[spend_camp] = lead_keys[matches[0]]
                continue

            # Try token overlap
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
                result[spend_camp] = spend_camp  # Preserve original

        return result

class ReportGenerator:
    """Handles report generation"""

    @staticmethod
    def create_charts(agg_df: pd.DataFrame, leads_df: pd.DataFrame) -> Dict[str, io.BytesIO]:
        """Create all charts for the report"""
        charts = {}

        # Sort data for consistent display
        display_df = agg_df.sort_values("leads_rows_count", ascending=False).head(10)

        # 1. Leads per Campaign
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        bars1 = ax1.bar(display_df["campaign_mapped"], display_df["leads_rows_count"],
                        color=COLORS['primary'], edgecolor='navy', linewidth=0.5)
        ax1.set_title("Leads per Campaign", fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel("Number of Leads", fontsize=12)
        ax1.set_xlabel("Campaign", fontsize=12)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2., height + max(display_df["leads_rows_count"]) * 0.01,
                         f'{int(height)}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        charts['leads'] = ReportGenerator.fig_to_buffer(fig1)
        plt.close(fig1)

        # 2. Spend per Campaign
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        bars2 = ax2.bar(display_df["campaign_mapped"], display_df["spend_total"],
                        color=COLORS['secondary'], edgecolor='darkred', linewidth=0.5)
        ax2.set_title("Spend per Campaign (₹)", fontsize=16, fontweight='bold', pad=20)
        ax2.set_ylabel("Spend Amount (₹)", fontsize=12)
        ax2.set_xlabel("Campaign", fontsize=12)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2., height + max(display_df["spend_total"]) * 0.01,
                         f'₹{height:,.0f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        charts['spend'] = ReportGenerator.fig_to_buffer(fig2)
        plt.close(fig2)

        # 3. CPL per Campaign
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        cpl_data = display_df["CPL_computed"].fillna(0)
        bars3 = ax3.bar(display_df["campaign_mapped"], cpl_data,
                        color=COLORS['warning'], edgecolor='darkorange', linewidth=0.5)
        ax3.set_title("Cost Per Lead (CPL) by Campaign", fontsize=16, fontweight='bold', pad=20)
        ax3.set_ylabel("CPL (₹)", fontsize=12)
        ax3.set_xlabel("Campaign", fontsize=12)
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

        # Add value labels
        for bar, cpl_val in zip(bars3, cpl_data):
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width() / 2., height + max(cpl_data) * 0.01,
                         f'₹{height:.0f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        charts['cpl'] = ReportGenerator.fig_to_buffer(fig3)
        plt.close(fig3)

        # 4. Sentiment Analysis
        sentiment_data = ReportGenerator.analyze_sentiment(leads_df)
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        colors = [COLORS['success'], COLORS['warning'], COLORS['danger']]  # Green, Gold, Red
        wedges, texts, autotexts = ax4.pie(sentiment_data.values(), labels=sentiment_data.keys(),
                                          autopct='%1.1f%%', startangle=90, colors=colors)
        ax4.set_title("Lead Sentiment Distribution", fontsize=16, fontweight='bold')

        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        charts['sentiment'] = ReportGenerator.fig_to_buffer(fig4)
        plt.close(fig4)

        return charts

    @staticmethod
    def analyze_sentiment(leads_df: pd.DataFrame) -> Dict[str, int]:
        """Analyze sentiment from lead notes/comments"""
        # Find potential notes/comments columns
        notes_cols = [c for c in leads_df.columns
                      if re.search(r"note|status|remark|message|comment|feedback", c, re.I)]

        if not notes_cols:
            # Return estimated distribution
            total_leads = len(leads_df)
            return {
                "Neutral": int(total_leads * 0.55),
                "Positive": int(total_leads * 0.35),
                "Negative": int(total_leads * 0.10)
            }

        # Use first notes column
        notes_col = notes_cols[0]
        notes = leads_df[notes_col].astype(str).fillna("").str.lower()

        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

        for note in notes:
            if not note or note == "nan":
                sentiment_counts["Neutral"] += 1
                continue

            # Check for positive keywords
            pos_found = any(keyword in note for keyword in config.SENTIMENT_KEYWORDS['positive'])
            neg_found = any(keyword in note for keyword in config.SENTIMENT_KEYWORDS['negative'])

            if pos_found and not neg_found:
                sentiment_counts["Positive"] += 1
            elif neg_found and not pos_found:
                sentiment_counts["Negative"] += 1
            else:
                sentiment_counts["Neutral"] += 1

        return sentiment_counts

    @staticmethod
    def fig_to_buffer(fig) -> io.BytesIO:
        """Convert matplotlib figure to buffer"""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150,
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        return buf

    @staticmethod
    def generate_pdf_report(charts: Dict[str, io.BytesIO], totals: Dict[str, Any],
                           agg_table: pd.DataFrame, output_path: str):
        """Generate comprehensive PDF report"""
        try:
            width, height = landscape(A4)
            c = canvas.Canvas(output_path, pagesize=landscape(A4))
            margin = 40

            # Title
            c.setFont("Helvetica-Bold", 20)
            title = totals.get("report_title", f"Campaign Performance Report — {datetime.now().strftime('%d %b %Y')}")
            c.drawCentredString(width / 2, height - 40, title)

            # Layout charts in 2x2 grid
            chart_keys = ["leads", "spend", "cpl", "sentiment"]
            img_width = (width - 3 * margin) / 2
            img_height = (height - 5 * margin - 80) / 2

            positions = [
                (margin, height - margin - img_height - 60),
                (margin * 2 + img_width, height - margin - img_height - 60),
                (margin, height - 2 * margin - 2 * img_height - 60),
                (margin * 2 + img_width, height - 2 * margin - 2 * img_height - 60)
            ]

            for chart_key, pos in zip(chart_keys, positions):
                chart_buf = charts.get(chart_key)
                if chart_buf:
                    try:
                        chart_buf.seek(0)
                        img = ImageReader(chart_buf)
                        c.drawImage(img, pos[0], pos[1], width=img_width, height=img_height,
                                    preserveAspectRatio=True, anchor='sw')
                    except Exception as e:
                        logger.error(f"Failed to add chart {chart_key} to PDF: {e}")

            # Summary statistics
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
                f"Average CPL (Deduplicated): ₹{totals.get('avg_cpl_unique', 0):.2f}"
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
    """Main dashboard class"""

    def __init__(self):
        self.data_processor = DataProcessor()
        self.ocr_processor = OCRProcessor()
        self.mapper = CampaignMapper()
        self.report_generator = ReportGenerator()

    def run(self):
        """Main application entry point"""
        self._render_header()
        self._render_sidebar()

        # Load leads data
        leads_df = self._load_leads_data()
        if leads_df is None:
            return

        # Process leads data
        leads_df = self._process_leads_data(leads_df)

        # Load spend data
        spend_df, spend_source = self._load_spend_data()

        # Generate mapping and analysis
        if st.button("🚀 Generate Analysis Report", type="primary", use_container_width=True):
            self._generate_analysis(leads_df, spend_df, spend_source)

    def _render_header(self):
        """Render application header"""
        st.title("📊 Lead Analysis Dashboard")
        st.markdown("""
        **Upload your leads master and spend data to get comprehensive campaign performance insights**

        Features:
        - 📈 Campaign performance analysis
        - 🔍 Duplicate detection
        - 🎯 Intelligent campaign mapping
        - 📊 Visual reports with charts
        - 📄 PDF report generation
        """)
        st.divider()

    def _render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("📁 Data Upload")
        st.sidebar.markdown("Upload your data files to begin analysis")

    def _load_leads_data(self) -> Optional[pd.DataFrame]:
        """Load and validate leads data"""
        uploaded_leads = st.sidebar.file_uploader(
            "Upload Leads Master File",
            type=["csv", "xlsx", "xls"],
            help="Upload your master leads file (CSV or Excel format)"
        )

        if uploaded_leads is None:
            st.info("👆 Please upload your Leads Master file to begin analysis.")
            return None

        try:
            # Load data based on file type
            if uploaded_leads.name.lower().endswith(".csv"):
                leads_df = pd.read_csv(uploaded_leads, encoding='utf-8')
            else:
                leads_df = pd.read_excel(uploaded_leads)

            # Validate data
            if leads_df.empty:
                st.error("❌ The uploaded leads file is empty.")
                return None

            # Display preview
            with st.expander(f"📋 Leads Preview ({len(leads_df):,} rows)", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Rows", f"{len(leads_df):,}")
                with col2:
                    st.metric("Columns", len(leads_df.columns))

                st.dataframe(
                    leads_df.head(config.MAX_DISPLAY_ROWS),
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("**Available Columns:**")
                st.write(", ".join(leads_df.columns.tolist()))

            return leads_df

        except Exception as e:
            st.error(f"❌ Failed to read leads file: {str(e)}")
            return None

    def _process_leads_data(self, leads_df: pd.DataFrame) -> pd.DataFrame:
        """Process leads data with campaign mapping and duplicate detection"""
        # Detect campaign column
        campaign_cols = [c for c in leads_df.columns
                         if re.search(r"campaign|source|utm|ad", c, re.I)]

        campaign_col = st.sidebar.selectbox(
            "Select Campaign Column",
            options=["(Auto-detect)"] + list(leads_df.columns),
            index=0,
            help="Choose the column that contains campaign names"
        )

        if campaign_col == "(Auto-detect)" and campaign_cols:
            campaign_col = campaign_cols[0]
            st.sidebar.success(f"✅ Auto-detected: {campaign_col}")
        elif campaign_col == "(Auto-detect)":
            campaign_col = None

        # Map campaign column
        if campaign_col and campaign_col in leads_df.columns:
            leads_df["campaign_mapped"] = leads_df[campaign_col].astype(str).fillna("unknown")
        else:
            leads_df["campaign_mapped"] = "unknown"

        # Ensure lead ID column
        if "lead_id" not in leads_df.columns:
            leads_df["_lead_row_id"] = range(1, len(leads_df) + 1)

        # Create comparison keys
        leads_df["__cmp_key"] = leads_df["campaign_mapped"].apply(self.data_processor.create_simple_key)

        # Detect duplicates
        leads_df = self.data_processor.detect_duplicates(leads_df)

        # Show duplicate summary
        if leads_df["_duplicate_internal"].sum() > 0:
            with st.expander("⚠️ Duplicate Detection Results"):
                dup_count = int(leads_df["_duplicate_internal"].sum())
                st.warning(f"Found {dup_count} potential duplicate leads")

                dup_df = leads_df[leads_df["_duplicate_internal"] == True]
                st.dataframe(
                    dup_df[["campaign_mapped", "_duplicate_internal_reason"]].head(10),
                    use_container_width=True
                )

        return leads_df

    def _load_spend_data(self) -> Tuple[pd.DataFrame, Optional[str]]:
        """Load spend data from file or image"""
        spend_df = pd.DataFrame(columns=["Campaign", "Leads", "Spend"])
        spend_source = None

        st.sidebar.subheader("💰 Spend Data")

        # File upload
        uploaded_spend_file = st.sidebar.file_uploader(
            "Upload Spend File (Optional)",
            type=["csv", "xlsx", "xls"],
            help="Upload spend data in CSV or Excel format"
        )

        # Image upload for OCR
        uploaded_spend_image = st.sidebar.file_uploader(
            "Upload Spend Image (Optional)",
            type=["png", "jpg", "jpeg"],
            help="Upload screenshot or image containing spend data"
        )

        # Process file upload
        if uploaded_spend_file is not None:
            spend_df, spend_source = self._process_spend_file(uploaded_spend_file)

        # Process image upload (if no file was uploaded)
        elif uploaded_spend_image is not None:
            spend_df, spend_source = self._process_spend_image(uploaded_spend_image)

        # Manual entry option
        elif st.sidebar.checkbox("📝 Enter Spend Data Manually"):
            spend_df, spend_source = self._manual_spend_entry()

        return spend_df, spend_source

    def _process_spend_file(self, uploaded_file) -> Tuple[pd.DataFrame, str]:
        """Process uploaded spend file"""
        try:
            # Load the file
            if uploaded_file.name.lower().endswith(".csv"):
                spend_raw = pd.read_csv(uploaded_file, encoding='utf-8')
            else:
                spend_raw = pd.read_excel(uploaded_file)

            if spend_raw.empty:
                st.sidebar.error("❌ Spend file is empty")
                return pd.DataFrame(columns=["Campaign", "Leads", "Spend"]), None

            st.sidebar.success(f"✅ Loaded spend file: {len(spend_raw)} rows")
