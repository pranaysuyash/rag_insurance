import os
from tempfile import NamedTemporaryFile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import pytesseract
from openai import OpenAI

import pikepdf
import pdfplumber
import pandas as pd
import json
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, CohereEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline

# ─── Load environment & initialize clients ────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

st.set_page_config(page_title="Policy QA Agent", layout="wide")
st.title("📄 Policy QA Agent")

# ─── Constants ────────────────────────────────────────────────
UPLOAD_LIMIT_MB = int(os.getenv("UPLOAD_LIMIT_MB", 10))
INDEX_DIR = Path(os.getenv("INDEX_DIR", "faiss_index"))

# ─── Embedding model registry ─────────────────────────────────
EMBED_OPTIONS = {
    "OpenAI • text-embedding-ada-002": lambda: OpenAIEmbeddings(model="text-embedding-ada-002"),
    "Cohere • embed-english-v2.0": lambda: CohereEmbeddings(
        model="embed-english-v2.0", cohere_api_key=os.getenv("COHERE_API_KEY", ""),
    ),
    "HF • multi-qa-mpnet-base-dot-v1": lambda: HuggingFaceEmbeddings(
        model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"
    ),
    "HF • all-MiniLM-L12-v2": lambda: HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2"
    ),
}

# ─── Fetch available OpenAI GPT models ────────────────────────
@st.cache_resource(show_spinner=False)
def get_available_openai_models() -> list[str]:
    data = client.models.list().data
    return sorted(
        m.id for m in data
        if isinstance(m.id, str) and (m.id.startswith("gpt-") or m.id.startswith("gpt4"))
    )

OPENAI_MODELS = get_available_openai_models()

# ─── Sidebar: model selection ─────────────────────────────────
st.sidebar.header("⚙️ Settings")
embed_choice = st.sidebar.selectbox("Embedding model", list(EMBED_OPTIONS.keys()), index=0)
llm_choice = st.sidebar.selectbox("LLM model", OPENAI_MODELS + ["Local • GPT-2"], index=0)

# ─── Enable debugging toggle ───────────────────────────────────
enable_verbose_debug = st.sidebar.checkbox("Enable verbose debugging", value=False)

# ─── Insured name input ────────────────────────────────────────
insured_name = st.sidebar.text_input("Insured Name", value="Pranay Suyash")

# ─── Instantiate embedder & LLM ──────────────────────────────
embedder = EMBED_OPTIONS[embed_choice]()
if llm_choice == "Local • GPT-2":
    hf_pipe = pipeline(
        "text-generation",
        model=AutoModelForCausalLM.from_pretrained("gpt2"),
        tokenizer=AutoTokenizer.from_pretrained("gpt2"),
        max_new_tokens=512,
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)
else:
    llm = ChatOpenAI(model_name=llm_choice, temperature=0)

# ─── Helper: sanitize filenames ───────────────────────────────
def sanitize(name: str) -> str:
    return Path(name).stem.replace(" ", "_")

# ─── Helper: sanitize PDF via pikepdf ─────────────────────────
def sanitize_pdf(input_path: str) -> str:
    try:
        with pikepdf.Pdf.open(input_path) as pdf:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                pdf.save(tmp.name)
                return tmp.name
    except Exception:
        return input_path

# ─── Debug utility ───────────────────────────────────────────
def debug_print(message, obj=None):
    if enable_verbose_debug:
        if obj is not None:
            st.sidebar.write(message)
            st.sidebar.write(obj)
        else:
            st.sidebar.write(message)
    print(message)
    if obj is not None:
        print(obj)

# ─── Unified PDF extractor (text + tables) ────────────────────
def extract_doc_data(pdf_path: str) -> tuple[str, list[tuple[int, pd.DataFrame]], list[dict]]:
    clean_path = sanitize_pdf(pdf_path)
    
    # Add structured metadata extraction
    metadata = []

    # 1) Extract text via PyPDFLoader with better error handling
    try:
        loader = PyPDFLoader(clean_path)
        pages = loader.load()
        text = "\n\n".join(page.page_content for page in pages)
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        text = ""
        pages = []
    
    # Extract key metadata patterns
    policy_number_pattern = r'(?i)policy\s+(?:no|number|#)[:\.\s]*([A-Za-z0-9\-\/]{5,20})'
    insured_name_pattern = r'(?i)(?:name\s+of\s+(?:the\s+)?insured|insured\s+name)[:\.\s]*([A-Za-z\s\.]{3,50})'
    premium_pattern = r'(?i)(?:premium|total\s+premium)[:\.\s]*(?:Rs\.?|INR|₹)?\s*([0-9,\.]+)'
    date_pattern = r'(?i)(?:policy\s+(?:commencement|issue)\s+date|date\s+of\s+issue)[:\.\s]*([0-9]{1,2}[-/\.][0-9]{1,2}[-/\.][0-9]{2,4})'
    
    for pattern_name, regex in [
        ("policy_number", policy_number_pattern),
        ("insured_name", insured_name_pattern),
        ("premium", premium_pattern),
        ("date", date_pattern)
    ]:
        matches = re.findall(regex, text)
        if matches:
            metadata.append({
                "type": pattern_name,
                "value": matches[0].strip(),
                "source": "text_extraction",
                "confidence": "high"
            })

    # 2) Extract tables via pdfplumber with improved error handling and column normalization
    tables: list[tuple[int, pd.DataFrame]] = []
    with pdfplumber.open(clean_path) as pdf:
        for page in pdf.pages:
            try:
                raw_tables = page.extract_tables()
            except Exception as e:
                st.warning(f"Skipping table extraction on page {page.page_number}: {e}")
                continue

            for raw in raw_tables:
                if not raw or len(raw) < 2:
                    continue

                # Build unique, non-empty headers with more aggressive normalization
                seen: dict[str, int] = {}
                headers: list[str] = []
                for i, h in enumerate(raw[0]):
                    # Standardize header cleaning
                    if isinstance(h, str) and h.strip():
                        base = h.strip().lower()
                        # Normalize column names aggressively
                        base = re.sub(r'[^\w\s]', '', base)  # Remove special characters
                        base = base.replace(" ", "_").replace("__", "_").strip("_")
                        
                        # Special handling for common insurance column formats
                        if any(term in base for term in ["name", "insured", "policy_holder"]):
                            base = "insured_name"
                        elif any(term in base for term in ["sum", "assured", "coverage"]):
                            base = "sum_assured"
                        elif any(term in base for term in ["premium", "amount"]):
                            base = "premium"
                        elif any(term in base for term in ["start", "commencement", "date_of_issue", "policy_date"]):
                            base = "commencement_date"
                        elif any(term in base for term in ["policy", "no", "number", "id"]):
                            base = "policy_number"
                    else:
                        base = f"col_{i}"
                        
                    count = seen.get(base, 0)
                    name = f"{base}_{count}" if count else base
                    seen[base] = count + 1
                    headers.append(name)

                # Normalize rows and extract values
                rows = [[cell.strip() if isinstance(cell, str) else "" for cell in row] for row in raw[1:]]
                df = pd.DataFrame(rows, columns=headers).dropna(how="all", subset=headers)
                
                if not df.empty:
                    tables.append((page.page_number, df))
                    debug_print(f"Extracted table on page {page.page_number}, columns: {headers}")
                    
                    # Extract metadata from tables
                    for col in df.columns:
                        col_lower = col.lower()
                        if any(key in col_lower for key in ["policy", "number", "no"]) and "policy_number" not in [m["type"] for m in metadata]:
                            if not df[col].iloc[0].isspace() and df[col].iloc[0]:
                                metadata.append({
                                    "type": "policy_number",
                                    "value": df[col].iloc[0],
                                    "source": f"table_p{page.page_number}",
                                    "confidence": "high"
                                })
                        
                        elif any(key in col_lower for key in ["insured", "name", "policy_holder"]) and "insured_name" not in [m["type"] for m in metadata]:
                            if not df[col].iloc[0].isspace() and df[col].iloc[0]:
                                metadata.append({
                                    "type": "insured_name",
                                    "value": df[col].iloc[0],
                                    "source": f"table_p{page.page_number}",
                                    "confidence": "high"
                                })
                        
                        elif any(key in col_lower for key in ["premium", "amount"]) and "premium" not in [m["type"] for m in metadata]:
                            if not df[col].iloc[0].isspace() and df[col].iloc[0]:
                                metadata.append({
                                    "type": "premium",
                                    "value": df[col].iloc[0],
                                    "source": f"table_p{page.page_number}",
                                    "confidence": "high"
                                })
                        
                        elif any(key in col_lower for key in ["commencement", "start", "date", "from", "period"]) and "date" not in [m["type"] for m in metadata]:
                            if not df[col].iloc[0].isspace() and df[col].iloc[0] and str(df[col].iloc[0]).lower() != "none":
                                # If the column appears to be period related with a date range, extract just the start date
                                value = str(df[col].iloc[0])
                                if " to " in value.lower() or "-" in value or "–" in value:
                                    # Try to extract just the first date from a range (e.g., "12/01/2023 to 11/30/2024")
                                    parts = re.split(r'\s+to\s+|-|–', value)
                                    if parts and parts[0].strip():
                                        value = parts[0].strip()
                                
                                metadata.append({
                                    "type": "date",
                                    "value": value,
                                    "source": f"table_p{page.page_number}",
                                    "confidence": "high"
                                })

    return text, tables, metadata

# ─── OCR for images ───────────────────────────────────────────
def extract_text_from_image(uploaded_file) -> str:
    try:
        img = Image.open(uploaded_file)
        # Improve OCR with preprocessing
        img = img.convert('L')  # Convert to grayscale
        return pytesseract.image_to_string(img, config='--psm 6')  # Assume single block of text
    except Exception as e:
        st.error(f"Error in OCR processing: {e}")
        return ""

# ─── Policy validation via OpenAI client ─────────────────────
def validate_policy(text: str, model_name: str) -> bool:
    if not text:
        return False
        
    prompt = (
        "You are a policy validation assistant.\n"
        "Reply 'YES' if the document is a medical or health‑insurance policy, otherwise 'NO'.\n\n"
        f"DOCUMENT (first 2,000 chars):\n{text[:2000]}"
    )
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        verdict = resp.choices[0].message.content.strip().upper()
        return verdict.startswith("Y")
    except Exception as e:
        st.error(f"Error validating policy: {e}")
        return False

# ─── Direct answer from tables function ────────────────────────
def direct_answer_from_tables(question: str, tables: list[tuple[int, pd.DataFrame]], 
                            insured_name: str = "Pranay Suyash") -> tuple[str | None, list[str]]:
    """
    Extract answers from table data using direct pattern matching for common insurance queries.
    Returns a tuple of (formatted answer string or None, list of table sources used).
    """
    q = question.lower().strip()
    insured_name_lower = insured_name.lower()
    name_parts = insured_name_lower.split()
    
    # Enable debugging
    debug_info = []
    debug_info.append(f"Direct method - Processing query: '{q}'")
    debug_info.append(f"Direct method - Looking for insured name: '{insured_name}'")
    table_sources = []
    
    # Define patterns for common questions with expanded patterns
    patterns = {
        "since": {
            "keywords": ["since", "how long", "tie up", "tied up", "join", "first", "start", "begin", 
                        "when did", "commenced", "enrolled", "member since", "how long have", "duration"],
            "col_patterns": ["since", "start", "date", "from", "join", "tie", "commence", "enroll", 
                            "insured_with_niva_bupa_since", "member_since", "duration", "period"],
            # Primary columns to check first
            "primary_cols": ["since", "insured_with_niva_bupa_since", "member_since"],
            # Fallback columns to check if primary columns don't have valid values
            "fallback_cols": ["date", "start_date", "from_date", "commencement_date", "effective_date"],
            "format": lambda val, pg: f"You have been insured since **{val}** (page {pg})."
        },
        "commencement_date": {
            "keywords": ["commencement", "policy start", "effective date", "current policy", "begin date", 
                        "policy commencement", "start date", "from date", "when did the policy start", 
                        "policy period", "renewal date", "when is my policy from"],
            "col_patterns": ["commencement", "effective", "start", "date", "policy_start", 
                            "policy_commencement", "commencement_date", "effective_date", 
                            "commencement_date_and_time", "from_date", "policy_date", "renewal", "period"],
            # Primary columns to check first
            "primary_cols": ["commencement_date", "policy_commencement_date", "effective_date", "policy_start_date", "period_from", "policy_period_from"],
            # Fallback columns to check if primary columns don't have valid values
            "fallback_cols": ["start_date", "date", "from_date", "policy_date", "period", "policy_period"],
            "format": lambda val, pg: f"The current policy commencement date is **{val}** (page {pg})."
        },
        "premium": {
            "keywords": ["premium", "cost", "pay", "payment", "fee", "amount", "price", "how much do i pay",
                        "annual premium", "monthly premium", "premium amount", "how much is the premium"],
            "col_patterns": ["premium", "cost", "payment", "fee", "amount", "price", "annual_premium", 
                            "monthly_premium", "premium_amount", "total_premium"],
            "format": lambda val, pg: f"Your premium is **{val}** (page {pg})."
        },
        "coverage": {
            "keywords": ["cover", "coverage", "limit", "benefit", "sum", "assured", "covered for", "maximum", 
                        "coverage amount", "how much am i covered for", "sum insured", "sum assured"],
            "col_patterns": ["cover", "limit", "sum", "benefit", "assured", "amount", "coverage", "sum_insured", 
                            "sum_assured", "coverage_amount", "max_coverage"],
            "format": lambda val, pg: f"Your coverage is **{val}** (page {pg})."
        },
        "policy_id": {
            "keywords": ["policy number", "policy id", "policy no", "policy identification", "reference number",
                        "what is my policy number", "policy reference", "policy #"],
            "col_patterns": ["policy", "id", "number", "no", "policy_id", "policy_number", "policy_no", 
                            "reference", "ref_number"],
            "format": lambda val, pg: f"Your policy number is **{val}** (page {pg})."
        },
        "expiry_date": {
            "keywords": ["expiry", "expiration", "valid until", "end date", "termination", "when does my policy expire",
                        "renewal due", "policy end", "validity", "valid till"],
            "col_patterns": ["expiry", "expiration", "end_date", "valid_until", "valid_till", "termination_date",
                            "end", "to_date", "renewal_due"],
            "format": lambda val, pg: f"Your policy expires on **{val}** (page {pg})."
        }
    }
    
    # Check if question matches any pattern
    matched_types = []
    for p_type, config in patterns.items():
        if any(kw in q for kw in config["keywords"]):
            matched_types.append(p_type)
            debug_info.append(f"Direct method - Matched type: {p_type}")
    
    if not matched_types:
        debug_info.append("Direct method - No pattern matched, falling back")
        debug_print("\n".join(debug_info))
        return None, []
    
    # Process each table for the matched patterns
    for page_no, df in tables:
        debug_info.append(f"\nDirect method - Checking table on page {page_no}")
        debug_info.append(f"Direct method - Columns: {df.columns.tolist()}")
        
        # Try each matched pattern
        for matched_type in matched_types:
            config = patterns[matched_type]
            debug_info.append(f"Direct method - Processing pattern '{matched_type}' for table on page {page_no}")
            
            # Find primary and fallback columns matching the pattern
            primary_cols = []
            fallback_cols = []
            other_cols = []
            
            for col in df.columns:
                col_lower = str(col).lower()
                
                # Check if this column is in primary columns list
                if "primary_cols" in config and any(primary_col.lower() in col_lower for primary_col in config["primary_cols"]):
                    primary_cols.append(col)
                # Check if this column is in fallback columns list
                elif "fallback_cols" in config and any(fallback_col.lower() in col_lower for fallback_col in config["fallback_cols"]):
                    fallback_cols.append(col)
                # Otherwise check general pattern match
                elif any(pattern in col_lower for pattern in config["col_patterns"]):
                    other_cols.append(col)
            
            # Order columns by priority: primary first, then fallback, then others
            rel_cols = primary_cols + fallback_cols + other_cols
            
            debug_info.append(f"Direct method - Primary columns: {primary_cols}")
            debug_info.append(f"Direct method - Fallback columns: {fallback_cols}")
            debug_info.append(f"Direct method - Other relevant columns: {other_cols}")
            debug_info.append(f"Direct method - All relevant columns (ordered): {rel_cols}")
            
            if not rel_cols:
                debug_info.append("Direct method - No relevant columns in this table, skipping")
                continue
            
            # Find name columns for filtering
            name_cols = []
            for col in df.columns:
                col_lower = str(col).lower()
                if any(term in col_lower for term in ["name", "insured", "person", "holder"]):
                    name_cols.append(col)
            
            debug_info.append(f"Direct method - Name columns: {name_cols}")
            
            # Process each relevant column
            for rel_col in rel_cols:
                try:
                    # Validate column data - filter out None values and empty strings
                    valid_vals = df[rel_col].dropna().astype(str).str.strip()
                    valid_vals = valid_vals[valid_vals.str.lower() != "none"]  # Filter out "None" string values
                    if valid_vals.empty:
                        debug_info.append(f"Direct method - No valid values in column {rel_col}, skipping")
                        continue
                    
                    if name_cols:
                        # Filter by name if possible
                        for name_col in name_cols:
                            try:
                                # Try full name match
                                mask = df[name_col].astype(str).str.lower().str.contains(insured_name_lower, na=False)
                                
                                # If full name doesn't match, try with name parts
                                if not mask.any() and len(name_parts) > 1:
                                    mask = pd.Series(True, index=df.index)
                                    for part in name_parts:
                                        if len(part) > 2:
                                            mask = mask & df[name_col].astype(str).str.lower().str.contains(part, na=False)
                                
                                if mask.any():
                                    debug_info.append(f"Direct method - Found match for '{insured_name}' in {name_col}")
                                    val = df.loc[mask, rel_col].iloc[0]
                                    debug_info.append(f"Direct method - Value: {val}")
                                    
                                    # If value is a date range, extract just the start date for date-related patterns
                                    if "commencement" in matched_type or "since" in matched_type or matched_type == "expiry_date":
                                        value_str = str(val)
                                        if " to " in value_str.lower() or "-" in value_str or "–" in value_str:
                                            # Try to extract just the first date from a range (e.g., "12/01/2023 to 11/30/2024")
                                            parts = re.split(r'\s+to\s+|-|–', value_str)
                                            if parts and parts[0].strip():
                                                val = parts[0].strip()
                                                debug_info.append(f"Direct method - Extracted start date from range: {val}")
                                    
                                    if pd.notna(val) and str(val).strip() and str(val).lower() != "none":
                                        answer = config["format"](val, page_no)
                                        table_sources.append(f"Table on page {page_no} with columns: {', '.join(df.columns.tolist())}")
                                        debug_info.append(f"Direct method - Answer: {answer}")
                                        debug_print("\n".join(debug_info))
                                        return answer, table_sources
                            except Exception as e:
                                debug_info.append(f"Direct method - Error in name filtering for {name_col}: {e}")
                    else:
                        # No name columns, use first valid value for small tables
                        if len(df) <= 2:
                            val = valid_vals.iloc[0]
                            debug_info.append(f"Direct method - Single/small table, value: {val}")
                            
                            # If value is a date range, extract just the start date for date-related patterns
                            if "commencement" in matched_type or "since" in matched_type or matched_type == "expiry_date":
                                value_str = str(val)
                                if " to " in value_str.lower() or "-" in value_str or "–" in value_str:
                                    # Try to extract just the first date from a range (e.g., "12/01/2023 to 11/30/2024")
                                    parts = re.split(r'\s+to\s+|-|–', value_str)
                                    if parts and parts[0].strip():
                                        val = parts[0].strip()
                                        debug_info.append(f"Direct method - Extracted start date from range: {val}")
                            
                            if pd.notna(val) and str(val).strip() and str(val).lower() != "none":
                                answer = config["format"](val, page_no)
                                table_sources.append(f"Table on page {page_no} with columns: {', '.join(df.columns.tolist())}")
                                debug_info.append(f"Direct method - Answer: {answer}")
                                debug_print("\n".join(debug_info))
                                return answer, table_sources
                        else:
                            debug_info.append(f"Direct method - Multi-row table without name column, skipping to avoid ambiguity")
                except Exception as e:
                    debug_info.append(f"Direct method - Error processing column {rel_col}: {e}")
    
    debug_info.append("Direct method - No answer found in tables")
    debug_print("\n".join(debug_info))
    return None, []

# ─── AI-powered table query analyzer ─────────────────────────
def analyze_query_intent(question: str, tables: list[tuple[int, pd.DataFrame]]) -> dict:
    """
    Use OpenAI to analyze the query intent and identify relevant table columns.
    Returns a dict with query intent analysis.
    """
    # Build a description of all tables
    tables_description = []
    for page_no, df in tables:
        cols = df.columns.tolist()
        # Add sample data from first row for context
        sample_data = {}
        if not df.empty:
            sample_data = {col: str(df[col].iloc[0]) for col in df.columns if not pd.isna(df[col].iloc[0])}
        
        tables_description.append(
            f"Table on page {page_no} with columns: {', '.join(cols)}\n"
            f"Sample data: {json.dumps(sample_data, ensure_ascii=False)}"
        )
    
    tables_info = "\n".join(tables_description)
    
    # Enhanced prompt with explicit guidance for common queries
    prompt = f"""
    You are a query intent analyzer for insurance policy documents.
    
    Analyze this user question and determine if it can be answered from structured table data.
    Map queries to specific intents based on keywords and context:
    - "tie up," "join," "start," "since," "commenced," "enrolled": Ask about the initial association 
      with the insurer (e.g., Niva Bupa). Map to columns like "since," "start_date," 
      "insured_with_niva_bupa_since."
    - "commencement," "effective date," "policy start," "current policy": Ask about the current 
      policy's start date. Map to columns like "commencement," "effective_date," "policy_start," 
      "commencement_date_and_time," "from_date."
    - "premium," "cost," "payment": Ask about policy cost. Map to columns like "premium," "amount."
    - "coverage," "limit," "sum assured": Ask about coverage details. Map to columns like 
      "coverage," "limit," "sum."
    - "expiry," "termination," "valid until": Ask about when policy ends. Map to columns like
      "expiry_date," "valid_until," "to_date."
    
    User question: "{question}"
    
    Available tables:
    {tables_info}
    
    Return a JSON with the following structure:
    {{
        "can_answer_from_table": true/false,
        "intent_category": "<category>",  // e.g., "since_when", "commencement_date", "premium"
        "relevant_columns": ["column1", "column2"],  // columns that might contain the answer
        "primary_columns": ["best_col1", "best_col2"],  // highest priority columns to check first
        "fallback_columns": ["fallback_col1", "fallback_col2"],  // columns to check if primary ones don't work
        "person_filter": "<name>",  // name of person if query asks about someone specific
        "response_template": "<template>"  // e.g., "You have been insured since {{value}} (page {{page}})."
    }}
    
    Examples:
    - For "When did I tie up with Niva Bupa?":
      - can_answer_from_table: true
      - intent_category: "since_when"
      - relevant_columns: ["insured_with_niva_bupa_since", "since", "start_date", "date"]
      - primary_columns: ["insured_with_niva_bupa_since", "since", "member_since"]
      - fallback_columns: ["start_date", "from_date", "commencement_date"]
      - person_filter: "Pranay Suyash"
      - response_template: "You have been insured since {{value}} (page {{page}})."
    - For "When is the current policy commencement date?":
      - can_answer_from_table: true
      - intent_category: "commencement_date"
      - relevant_columns: ["policy_commencement_date", "commencement", "effective_date", 
                          "commencement_date_and_time", "from_date", "policy_date"]
      - primary_columns: ["policy_commencement_date", "commencement_date", "effective_date"]
      - fallback_columns: ["policy_date", "from_date", "start_date"]
      - person_filter: "Pranay Suyash"
      - response_template: "The current policy commencement date is {{value}} (page {{page}})."
    
    Only return the JSON with no explanations or other text.
    """
    
    debug_print("AI Intent Analysis - Prompt:", prompt)
    
    try:
        response = client.chat.completions.create(
            model=llm_choice,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        result_text = response.choices[0].message.content
        debug_print("AI Intent Analysis - Raw response:", result_text)
        
        try:
            result = json.loads(result_text)
            debug_print("AI Intent Analysis - Parsed result:", result)
            return result
        except Exception as e:
            debug_print(f"Error parsing intent analysis: {e}")
    except Exception as e:
        debug_print(f"Error calling OpenAI API: {e}")
    
    # Fallback for critical queries
    if any(kw in question.lower() for kw in ["tie up", "tied up", "since", "join", "commenced"]):
        debug_print("Fallback: Detected 'tie up' or similar in query, using hardcoded response")
        return {
            "can_answer_from_table": True,
            "intent_category": "since_when",
            "relevant_columns": ["insured_with_niva_bupa_since", "since", "start", "date", "from_date", "commencement_date"],
            "primary_columns": ["since", "insured_with_niva_bupa_since", "member_since"],
            "fallback_columns": ["date", "start_date", "from_date", "commencement_date", "effective_date"],
            "person_filter": insured_name,
            "response_template": "You have been insured since {value} (page {page})."
        }
    if any(kw in question.lower() for kw in ["commencement", "effective date", "policy start", 
                                            "current policy", "from date"]):
        debug_print("Fallback: Detected 'commencement' or similar in query, using hardcoded response")
        return {
            "can_answer_from_table": True,
            "intent_category": "commencement_date",
            "relevant_columns": ["policy_commencement_date", "commencement", "effective_date", 
                                "start_date", "commencement_date_and_time", "from_date", "policy_date"],
            "primary_columns": ["commencement_date", "policy_commencement_date", "effective_date", "policy_start_date"],
            "fallback_columns": ["start_date", "date", "from_date", "policy_date"],
            "person_filter": insured_name,
            "response_template": "The current policy commencement date is {value} (page {page})."
        }
    return {
        "can_answer_from_table": False,
        "intent_category": "unknown",
        "relevant_columns": [],
        "primary_columns": [],
        "fallback_columns": [],
        "person_filter": insured_name,
        "response_template": "The answer is {value} (page {page})."
    }

# ─── Enhanced answer from tables function with AI intent analysis ─────────────
def answer_from_tables(question: str, tables: list[tuple[int, pd.DataFrame]], metadata: list[dict]) -> tuple[str | None, list[str]]:
    """
    Extract answers from table data using AI to understand query intent.
    Returns a tuple of (formatted answer string or None, list of table sources used).
    """
    # Enable debugging
    debug_info = []
    debug_info.append(f"AI method - Processing query: '{question}'")
    table_sources = []
    
    # Check if we can answer from metadata first
    meta_answer = answer_from_metadata(question, metadata)
    if meta_answer:
        debug_info.append("AI method - Found answer in extracted metadata")
        debug_print("\n".join(debug_info))
        return meta_answer, ["Extracted document metadata"]
    
    # Try direct pattern matching first (faster and more reliable for common queries)
    direct_answer, direct_sources = direct_answer_from_tables(question, tables, insured_name)
    if direct_answer:
        debug_info.append("AI method - Found answer using direct pattern matching")
        table_sources.extend(direct_sources)
        debug_print("\n".join(debug_info))
        return direct_answer, table_sources
    
    # Fall back to AI intent analysis if direct method doesn't find an answer
    debug_info.append("AI method - Direct pattern matching failed, trying AI intent analysis")
    
    # Use AI to analyze the query intent
    intent_analysis = analyze_query_intent(question, tables)
    debug_info.append(f"AI method - Intent analysis: {intent_analysis}")
    
    # If the AI thinks this can't be answered from tables, return None
    if not intent_analysis.get("can_answer_from_table", False):
        debug_info.append("AI method - AI determined query cannot be answered from tables, falling back to LLM")
        debug_print("\n".join(debug_info))
        return None, []
    
    # Extract relevant information from intent analysis
    intent_category = intent_analysis.get("intent_category", "unknown")
    relevant_columns = intent_analysis.get("relevant_columns", [])
    primary_columns = intent_analysis.get("primary_columns", [])
    fallback_columns = intent_analysis.get("fallback_columns", [])
    person_filter = intent_analysis.get("person_filter", insured_name)
    response_template = intent_analysis.get("response_template", "The answer is {value} (page {page}).")
    
    if not relevant_columns and not primary_columns and not fallback_columns:
        debug_info.append("AI method - No relevant columns identified, falling back to LLM")
        debug_print("\n".join(debug_info))
        return None, []
    
    answers = []
    
    # Process each table looking for matches
    for page_no, df in tables:
        debug_info.append(f"\nAI method - Checking table on page {page_no}")
        debug_info.append(f"AI method - Columns: {df.columns.tolist()}")
        
        # Find primary, fallback, and other matching columns in this table
        primary_matching_cols = []
        fallback_matching_cols = []
        other_matching_cols = []
        
        # Process primary columns first
        for rel_col in primary_columns:
            # Try exact matches
            if rel_col in df.columns:
                primary_matching_cols.append(rel_col)
                continue
                
            # Try fuzzy matches
            for col in df.columns:
                col_lower = str(col).lower()
                rel_col_lower = str(rel_col).lower()
                if rel_col_lower in col_lower or any(part.lower() in col_lower 
                                                  for part in rel_col_lower.split('_') if len(part) > 2):
                    primary_matching_cols.append(col)
                    break
        
        # Process fallback columns next
        for rel_col in fallback_columns:
            # Skip if already a match in primary columns
            if any(rel_col.lower() in col.lower() for col in primary_matching_cols):
                continue
                
            # Try exact matches
            if rel_col in df.columns:
                fallback_matching_cols.append(rel_col)
                continue
                
            # Try fuzzy matches
            for col in df.columns:
                col_lower = str(col).lower()
                rel_col_lower = str(rel_col).lower()
                if rel_col_lower in col_lower or any(part.lower() in col_lower 
                                                  for part in rel_col_lower.split('_') if len(part) > 2):
                    fallback_matching_cols.append(col)
                    break
        
        # Process any remaining relevant columns
        for rel_col in relevant_columns:
            # Skip if already a match in primary or fallback columns
            if any(rel_col.lower() in col.lower() for col in primary_matching_cols + fallback_matching_cols):
                continue
                
            # Try exact matches
            if rel_col in df.columns:
                other_matching_cols.append(rel_col)
                continue
                
            # Try fuzzy matches
            for col in df.columns:
                col_lower = str(col).lower()
                rel_col_lower = str(rel_col).lower()
                if rel_col_lower in col_lower or any(part.lower() in col_lower 
                                                  for part in rel_col_lower.split('_') if len(part) > 2):
                    other_matching_cols.append(col)
                    break
        
        # Combine all columns with priority ordering
        matching_cols = primary_matching_cols + fallback_matching_cols + other_matching_cols
        
        debug_info.append(f"AI method - Primary matching columns: {primary_matching_cols}")
        debug_info.append(f"AI method - Fallback matching columns: {fallback_matching_cols}")
        debug_info.append(f"AI method - Other matching columns: {other_matching_cols}")
        debug_info.append(f"AI method - All matching columns (ordered): {matching_cols}")
        
        if not matching_cols:
            debug_info.append("AI method - No matching columns in this table, skipping")
            continue
        
        # Find name columns for filtering by person
        name_cols = []
        for col in df.columns:
            col_str = str(col).lower()
            if any(term in col_str for term in ["name", "insured", "person", "holder"]):
                name_cols.append(col)
        
        debug_info.append(f"AI method - Name columns found: {name_cols}")
        
        # Process each matching column
        for target_col in matching_cols:
            debug_info.append(f"AI method - Checking column: {target_col}")
            
            try:
                # Validate column data - filter out None values and empty strings
                valid_vals = df[target_col].dropna().astype(str).str.strip()
                valid_vals = valid_vals[valid_vals.str.lower() != "none"]  # Filter out "None" string values
                if valid_vals.empty:
                    debug_info.append(f"AI method - No valid values in column {target_col}, skipping")
                    continue
                
                # Try to filter by person name if name columns exist
                if name_cols and person_filter:
                    person_filter_lower = person_filter.lower()
                    person_parts = person_filter_lower.split()
                    
                    for name_col in name_cols:
                        try:
                            debug_info.append(f"AI method - Looking for '{person_filter}' in column: {name_col}")
                            
                            # First try contains match
                            mask = df[name_col].astype(str).str.lower().str.contains(person_filter_lower, na=False)
                            
                            if not mask.any() and len(person_parts) > 1:
                                debug_info.append("AI method - Full name not found, trying with parts")
                                mask = pd.Series(True, index=df.index)
                                for part in person_parts:
                                    if len(part) > 2:
                                        mask = mask & df[name_col].astype(str).str.lower().str.contains(part, na=False)
                            
                            if mask.any():
                                debug_info.append(f"AI method - Found matching row(s): {mask.sum()}")
                                val = df.loc[mask, target_col].iloc[0]
                                debug_info.append(f"AI method - Value found: {val}")
                                
                                # If value is a date range, extract just the start date for date-related intents
                                if intent_category in ["since_when", "commencement_date", "expiry_date"] or "date" in intent_category:
                                    value_str = str(val)
                                    if " to " in value_str.lower() or "-" in value_str or "–" in value_str:
                                        # Try to extract just the first date from a range (e.g., "12/01/2023 to 11/30/2024")
                                        parts = re.split(r'\s+to\s+|-|–', value_str)
                                        if parts and parts[0].strip():
                                            val = parts[0].strip()
                                            debug_info.append(f"AI method - Extracted start date from range: {val}")
                                
                                if pd.notna(val) and str(val).strip() and str(val).lower() != "none":
                                    try:
                                        answer = response_template.format(value=val, page=page_no)
                                    except Exception as e:
                                        debug_info.append(f"AI method - Error formatting template: {e}")
                                        answer = f"The answer is **{val}** (page {page_no})."
                                    
                                    answers.append(answer)
                                    table_sources.append(f"Table on page {page_no} with columns: {', '.join(df.columns.tolist())}")
                                    debug_info.append(f"AI method - Added answer: {answer}")
                                    break
                        except Exception as e:
                            debug_info.append(f"AI method - Error processing name column {name_col}: {e}")
                else:
                    debug_info.append("AI method - No name columns found or no person filter, using first valid row")
                    
                    if len(df) <= 2:
                        val = valid_vals.iloc[0]
                        debug_info.append(f"AI method - Single/small table, value: {val}")
                        
                        # If value is a date range, extract just the start date for date-related intents
                        if intent_category in ["since_when", "commencement_date", "expiry_date"] or "date" in intent_category:
                            value_str = str(val)
                            if " to " in value_str.lower() or "-" in value_str or "–" in value_str:
                                # Try to extract just the first date from a range (e.g., "12/01/2023 to 11/30/2024")
                                parts = re.split(r'\s+to\s+|-|–', value_str)
                                if parts and parts[0].strip():
                                    val = parts[0].strip()
                                    debug_info.append(f"AI method - Extracted start date from range: {val}")
                        
                        if pd.notna(val) and str(val).strip() and str(val).lower() != "none":
                            try:
                                answer = response_template.format(value=val, page=page_no)
                            except Exception as e:
                                debug_info.append(f"AI method - Error formatting template: {e}")
                                answer = f"The answer is **{val}** (page {page_no})."
                            
                            answers.append(answer)
                            table_sources.append(f"Table on page {page_no} with columns: {', '.join(df.columns.tolist())}")
                            debug_info.append(f"AI method - Added answer: {answer}")
                    else:
                        debug_info.append(f"AI method - Multi-row table without name column, skipping to avoid ambiguity")
            except Exception as e:
                debug_info.append(f"AI method - Error processing column {target_col}: {e}")
    
    debug_print("\n".join(debug_info))
    
    if answers:
        return "\n\n".join(answers), table_sources
    
    debug_info.append("AI method - No answers found in tables, falling back to LLM")
    debug_print("\n".join(debug_info))
    return None, []

# ─── Answer from extracted metadata ─────────────────────────────
def answer_from_metadata(question: str, metadata: list[dict]) -> str | None:
    """
    Check if the question can be answered from extracted metadata.
    Returns formatted answer string or None.
    """
    q = question.lower().strip()
    
    # Define patterns for metadata matching
    patterns = {
        "policy_number": {
            "keywords": ["policy number", "policy id", "policy no", "reference number"],
            "format": lambda val: f"Your policy number is **{val}**."
        },
        "insured_name": {
            "keywords": ["insured name", "policy holder", "who is insured", "my name"],
            "format": lambda val: f"The insured name is **{val}**."
        },
        "premium": {
            "keywords": ["premium", "cost", "pay", "how much", "fee"],
            "format": lambda val: f"Your premium is **{val}**."
        },
        "date": {
            "keywords": ["date", "commencement", "start", "begin", "issue date"],
            "format": lambda val: f"The policy commencement date is **{val}**."
        }
    }
    
    # Check for matches
    for meta_item in metadata:
        meta_type = meta_item["type"]
        if meta_type in patterns:
            pattern = patterns[meta_type]
            if any(kw in q for kw in pattern["keywords"]):
                return pattern["format"](meta_item["value"])
    
    return None

# ─── Build or load FAISS vector store ─────────────────────────
@st.cache_resource
def build_vector_store(_docs: list[Document]) -> FAISS:
    if INDEX_DIR.exists():
        try:
            return FAISS.load_local(str(INDEX_DIR), embedder, allow_dangerous_deserialization=True)
        except Exception as e:
            st.warning(f"Could not load existing index, rebuilding: {e}")
            INDEX_DIR.mkdir(exist_ok=True)
    
    vs = FAISS.from_documents(_docs, embedder)
    vs.save_local(str(INDEX_DIR))
    return vs

# ─── File upload & processing ────────────────────────────────
uploads = st.file_uploader(
    "Upload policy PDF or image(s)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploads:
    full_text = ""
    all_tables: list[tuple[int, pd.DataFrame]] = []
    all_metadata: list[dict] = []

    for f in uploads:
        if f.size > UPLOAD_LIMIT_MB * 1024**2:
            st.error(f"{sanitize(f.name)} exceeds {UPLOAD_LIMIT_MB} MB limit.")
            st.stop()

        if f.type == "application/pdf":
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                tmp.flush()
                pdf_path = tmp.name

            text, tables, metadata = extract_doc_data(pdf_path)
            full_text += text
            all_tables.extend(tables)
            all_metadata.extend(metadata)
        else:
            full_text += extract_text_from_image(f)

        full_text += "\n\n---DOC_BREAK---\n\n"

    # Display metadata if available
    if all_metadata:
        with st.expander("Extracted Document Metadata", expanded=True):
            for item in all_metadata:
                st.markdown(f"**{item['type']}**: {item['value']} (source: {item['source']})")

    # Debug: show extracted tables in a collapsible section
    with st.expander("Extracted Tables", expanded=False):
        st.subheader("Tables extracted from document")
        for pg, df in all_tables:
            st.markdown(f"**Table on page {pg}:**")
            st.dataframe(df)

    # Debug: Print all table columns to help diagnosis
    debug_print("All extracted tables:")
    for page_no, df in all_tables:
        debug_print(f"Table on page {page_no}, columns: {df.columns.tolist()}")
        if enable_verbose_debug:
            debug_print(f"Table data (full): {df.to_dict()}")

    # Validate policy
    if not validate_policy(full_text, llm_choice):
        st.error("⚠️ This doesn't look like a health‑insurance policy.")
        st.stop()
    st.success("✅ Policy validated. Building vector index…")

    # Create summaries of tables to include in context
    table_summaries = []
    for page_no, df in all_tables:
        summary = f"Table on page {page_no} contains columns: {', '.join(df.columns.tolist())}"
        if len(df) == 1:
            try:
                name_cols = [c for c in df.columns if any(term in str(c).lower() for term in ["name", "insured", "person", "holder"])]
                if name_cols:
                    summary += f" with data for {df.iloc[0][name_cols[0]]}"
                else:
                    summary += " with a single row of data"
            except:
                summary += " with a single row of data"
        elif len(df) > 1:
            summary += f" with data for {len(df)} records"
        table_summaries.append(summary)
    
    if table_summaries:
        table_context = "TABLES IN DOCUMENT:\n" + "\n".join(table_summaries)
        full_text += "\n\n" + table_context
    
    # Add metadata as context
    if all_metadata:
        metadata_context = "DOCUMENT METADATA:\n"
        for item in all_metadata:
            metadata_context += f"- {item['type']}: {item['value']}\n"
        full_text += "\n\n" + metadata_context

    # Chunk & index with improved chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    docs = splitter.split_documents([Document(page_content=full_text)])
    vs = build_vector_store(docs)

    # QA chain with enhanced prompt
    system_prompt = (
        "You are analyzing a health insurance policy document. The document contains both text and tables. "
        "Answer the question based on the provided context. "
        "If the answer likely resides in a table (e.g., commencement date, premium, or initial tie-up date), "
        "indicate that the answer may be in structured table data not fully represented in this context. "
        f"The insured name is {insured_name}. When answering, use this name in your responses. "
        "If you don't know the answer, say so and do not make up information. "
        "For example, for 'When is the current policy commencement date?', if not in context, respond: "
        "'The commencement date information may be in a table. "
        "Please check the structured table data.'"
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vs.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        verbose=enable_verbose_debug
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    # Handle user query
    query = st.text_input("Ask questions about your policy:")
    if query:
        # First try to get answer from tables or metadata
        table_ans, table_sources = answer_from_tables(query, all_tables, all_metadata)
        
        if table_ans:
            st.markdown(table_ans)
            # Show table sources
            with st.expander("Sources Used", expanded=True):
                st.subheader("Table Sources")
                if table_sources:
                    for i, source in enumerate(table_sources):
                        st.markdown(f"**Source {i+1}:**")
                        st.text(source)
                else:
                    st.text("No specific table sources identified.")
        else:
            with st.spinner("🔍 Thinking…"):
                # Enhance the query with the insured name for better context
                enhanced_query = (f"{query} (Note: The insured name is {insured_name})" 
                                if not insured_name.lower() in query.lower() else query)
                
                result = qa_chain({
                    "question": enhanced_query,
                    "chat_history": st.session_state.history
                })
                st.markdown("**Answer:**")
                st.write(result["answer"])
                
                # Show text sources
                with st.expander("Sources Used", expanded=True):
                    st.subheader("Retrieved Document Sources")
                    sources = result.get("source_documents", [])
                    if sources:
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Source {i+1}:**")
                            st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    else:
                        st.text("No document sources retrieved.")
                
                st.session_state.history.append((query, result["answer"]))