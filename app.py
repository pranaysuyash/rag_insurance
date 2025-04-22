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
INDEX_DIR       = Path(os.getenv("INDEX_DIR", "faiss_index"))

# ─── Embedding model registry ─────────────────────────────────
EMBED_OPTIONS = {
    "OpenAI • text-embedding-ada-002": lambda: OpenAIEmbeddings(model="text-embedding-ada-002"),
    "Cohere • embed-english-v2.0":    lambda: CohereEmbeddings(
        model="embed-english-v2.0", cohere_api_key=os.getenv("COHERE_API_KEY", ""),
    ),
    "HF • multi-qa-mpnet-base-dot-v1": lambda: HuggingFaceEmbeddings(
        model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"
    ),
    "HF • all-MiniLM-L12-v2":        lambda: HuggingFaceEmbeddings(
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
llm_choice   = st.sidebar.selectbox("LLM model", OPENAI_MODELS + ["Local • GPT-2"], index=0)

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

# ─── Unified PDF extractor (text + tables) ────────────────────
def extract_doc_data(pdf_path: str) -> tuple[str, list[tuple[int, pd.DataFrame]]]:
    clean_path = sanitize_pdf(pdf_path)

    # 1) Extract text via PyPDFLoader
    loader = PyPDFLoader(clean_path)
    pages = loader.load()
    text = "\n\n".join(page.page_content for page in pages)

    # 2) Extract tables via pdfplumber with unique headers
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

                # Build unique, non-empty headers
                seen: dict[str, int] = {}
                headers: list[str] = []
                for i, h in enumerate(raw[0]):
                    base = (h.strip().lower() if isinstance(h, str) and h.strip() else f"col_{i}")
                    base = base.replace(" ", "_").replace("(", "").replace(")", "")
                    count = seen.get(base, 0)
                    name = f"{base}_{count}" if count else base
                    seen[base] = count + 1
                    headers.append(name)

                # Normalize rows
                rows = [[cell if isinstance(cell, str) else "" for cell in row] for row in raw[1:]]
                df = pd.DataFrame(rows, columns=headers).dropna(how="all", subset=headers)
                if not df.empty:
                    tables.append((page.page_number, df))

    return text, tables

# ─── OCR for images ───────────────────────────────────────────
def extract_text_from_image(uploaded_file) -> str:
    img = Image.open(uploaded_file)
    return pytesseract.image_to_string(img)

# ─── Policy validation via OpenAI client ─────────────────────
def validate_policy(text: str, model_name: str) -> bool:
    prompt = (
        "You are a policy validation assistant.\n"
        "Reply 'YES' if the document is a medical or health‑insurance policy, otherwise 'NO'.\n\n"
        f"DOCUMENT (first 1,500 chars):\n{text[:1500]}"
    )
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role":"user", "content":prompt}],
        temperature=0,
    )
    verdict = resp.choices[0].message.content.strip().upper()
    return verdict.startswith("Y")

# ─── Enhanced answer from tables function ───────────────────────
def answer_from_tables(question: str, tables: list[tuple[int, pd.DataFrame]], 
                       insured_name: str = "Pranay Suyash") -> str | None:
    """
    Extract answers from table data based on query intent with comprehensive debugging.
    Returns a formatted answer string if found in tables, or None to fall back to LLM.
    """
    q = question.lower().strip()
    insured_name_lower = insured_name.lower()
    
    # Enable debugging
    debug_info = []
    debug_info.append(f"Processing query: '{q}'")
    debug_info.append(f"Looking for insured name: '{insured_name}'")
    
    # Define keyword mappings for common query types
    query_types = {
        "since": {
            "keywords": ["since", "how long", "start", "commenced", "began", "from when", 
                        "coverage start", "member since", "joined", "enrollment"],
            "column_patterns": ["since", "start", "commenced", "begin", "date", "enroll", "join"],
            "format": lambda val, pg: f"You have been insured since **{val}** (page {pg}).",
            "validate": lambda v: bool(v and str(v).strip())
        },
        "premium": {
            "keywords": ["premium", "cost", "payment", "fee", "amount", "pay", "price", 
                        "how much", "charge"],
            "column_patterns": ["premium", "cost", "fee", "amount", "payment", "price", "charge"],
            "format": lambda val, pg: f"Your premium is **{val}** (page {pg}).",
            "validate": lambda v: bool(v and str(v).strip())
        },
        "coverage": {
            "keywords": ["coverage", "covered", "benefit", "limit", "policy term", "maximum", 
                        "sum assured", "insured amount"],
            "column_patterns": ["coverage", "benefit", "limit", "term", "details", "max", 
                              "sum", "assured", "amount"],
            "format": lambda val, pg: f"Coverage details: **{val}** (page {pg}).",
            "validate": lambda v: bool(v and str(v).strip())
        },
        "policy_number": {
            "keywords": ["policy number", "policy id", "policy no", "contract number"],
            "column_patterns": ["policy", "policy_no", "policy_id", "contract", "number", "id"],
            "format": lambda val, pg: f"Your policy number is **{val}** (page {pg}).",
            "validate": lambda v: bool(v and str(v).strip())
        },
        "insured_name": {
            "keywords": ["name", "insured person", "policyholder", "who is insured", 
                        "whose name", "under whose name"],
            "column_patterns": ["name", "insured", "policyholder", "person", "holder", "client"],
            "format": lambda val, pg: f"The insured person is **{val}** (page {pg}).",
            "validate": lambda v: bool(v and str(v).strip())
        },
        "deductible": {
            "keywords": ["deductible", "excess", "co-pay", "out of pocket", "self payment"],
            "column_patterns": ["deductible", "excess", "co-pay", "out_of_pocket", "self_pay"],
            "format": lambda val, pg: f"Your deductible amount is **{val}** (page {pg}).",
            "validate": lambda v: bool(v and str(v).strip())
        },
        "renewal": {
            "keywords": ["renewal", "expiry", "valid until", "expiration", "renew date"],
            "column_patterns": ["renewal", "expiry", "valid", "expiration", "end"],
            "format": lambda val, pg: f"Your policy renewal date is **{val}** (page {pg}).",
            "validate": lambda v: bool(v and str(v).strip())
        }
    }

    # Find matching query types (can match multiple)
    matched_types = []
    for q_type, config in query_types.items():
        if any(kw in q for kw in config["keywords"]):
            matched_types.append(q_type)
            debug_info.append(f"Query matched type: {q_type}")
    
    if not matched_types:
        debug_info.append("No query type matched, falling back to LLM")
        print("\n".join(debug_info))
        return None  # No table-relevant query; fall back to LLM
    
    answers = []
    
    # Process each table for the matched query types
    for page_no, df in tables:
        debug_info.append(f"\nChecking table on page {page_no}")
        debug_info.append(f"Columns: {df.columns.tolist()}")
        
        # Sample data preview for debugging
        try:
            preview = df.head(2).to_string()
            debug_info.append(f"Data preview:\n{preview}")
        except Exception as e:
            debug_info.append(f"Error showing preview: {e}")
        
        # Try to answer each matched query type from this table
        for matched_type in matched_types:
            config = query_types[matched_type]
            debug_info.append(f"\nProcessing query type '{matched_type}' for table on page {page_no}")
            
            try:
                # Find columns matching the query type
                relevant_cols = []
                for col in df.columns:
                    col_str = str(col).lower()
                    if any(pattern in col_str for pattern in config["column_patterns"]):
                        relevant_cols.append(col)
                
                debug_info.append(f"Relevant columns found: {relevant_cols}")
                
                if not relevant_cols:
                    debug_info.append("No relevant columns in this table, skipping")
                    continue
                
                # Find name columns for filtering by insured person
                name_cols = []
                for col in df.columns:
                    col_str = str(col).lower()
                    if any(pattern in col_str for pattern in ["name", "insured", "person", "holder"]):
                        name_cols.append(col)
                
                debug_info.append(f"Name columns found: {name_cols}")
                
                # Process each relevant column
                for target_col in relevant_cols:
                    debug_info.append(f"Checking column: {target_col}")
                    
                    # Try to filter by insured name if name columns exist
                    if name_cols:
                        for name_col in name_cols:
                            try:
                                debug_info.append(f"Looking for '{insured_name}' in column: {name_col}")
                                
                                # First try exact match
                                mask = df[name_col].astype(str).str.lower().str.contains(insured_name_lower, na=False)
                                
                                if not mask.any():
                                    # Try partial matches - split name and check for each part
                                    name_parts = insured_name_lower.split()
                                    if len(name_parts) > 1:
                                        debug_info.append("Exact match not found, trying partial name matches")
                                        mask = df[name_col].astype(str).str.lower()
                                        for part in name_parts:
                                            mask = mask.str.contains(part, na=False)
                                
                                if mask.any():
                                    debug_info.append(f"Found matching row(s): {mask.sum()}")
                                    val = df.loc[mask, target_col].iloc[0]
                                    debug_info.append(f"Value found: {val}")
                                    
                                    if config["validate"](val):
                                        answer = config["format"](val, page_no)
                                        answers.append(answer)
                                        debug_info.append(f"Added answer: {answer}")
                                        break  # Found an answer for this name column
                                else:
                                    debug_info.append("No matching name found in column")
                            except Exception as e:
                                debug_info.append(f"Error processing name column {name_col}: {e}")
                    else:
                        # No name column - use first row with valid data
                        debug_info.append("No name columns found, using first valid row")
                        
                        if not df.empty:
                            try:
                                # For single-row tables, assume it's about the current policy
                                if len(df) == 1:
                                    val = df[target_col].iloc[0]
                                    debug_info.append(f"Single row table, value: {val}")
                                    
                                    if config["validate"](val):
                                        answer = config["format"](val, page_no)
                                        answers.append(answer)
                                        debug_info.append(f"Added answer: {answer}")
                                # For multi-row tables without name columns, look for value that's not NaN
                                else:
                                    valid_vals = df[target_col].dropna()
                                    if not valid_vals.empty:
                                        val = valid_vals.iloc[0]
                                        debug_info.append(f"Using first non-empty value: {val}")
                                        
                                        if config["validate"](val):
                                            answer = config["format"](val, page_no)
                                            answers.append(answer)
                                            debug_info.append(f"Added answer: {answer}")
                            except Exception as e:
                                debug_info.append(f"Error extracting value: {e}")
            except Exception as e:
                debug_info.append(f"Error processing table: {e}")
    
    # Print debug info to console
    print("\n".join(debug_info))
    
    # Return combined answers or None if no answers found
    if answers:
        return "\n\n".join(answers)
    debug_info.append("No answers found in tables, falling back to LLM")
    return None

# ─── Build or load FAISS vector store ─────────────────────────
@st.cache_resource
def build_vector_store(_docs: list[Document]) -> FAISS:
    if INDEX_DIR.exists():
        return FAISS.load_local(str(INDEX_DIR), embedder, allow_dangerous_deserialization=True)
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

    for f in uploads:
        if f.size > UPLOAD_LIMIT_MB * 1024**2:
            st.error(f"{sanitize(f.name)} exceeds {UPLOAD_LIMIT_MB} MB limit.")
            st.stop()

        if f.type == "application/pdf":
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                tmp.flush()
                pdf_path = tmp.name

            text, tables = extract_doc_data(pdf_path)
            full_text += text
            all_tables.extend(tables)
        else:
            full_text += extract_text_from_image(f)

        full_text += "\n\n---DOC_BREAK---\n\n"

    # Debug: show extracted tables in a collapsible section
    with st.expander("Debug: Extracted Tables", expanded=False):
        st.subheader("Tables extracted from document")
        for pg, df in all_tables:
            st.markdown(f"**Table on page {pg}:**")
            st.dataframe(df)

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

    # Chunk & index
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs     = splitter.split_documents([Document(page_content=full_text)])
    vs       = build_vector_store(docs)

    # QA chain with enhanced prompt
    system_prompt = (
        "You are analyzing a health insurance policy document. The document contains both text and tables. "
        "Answer the question based on the provided context. "
        "If you don't know the answer, say you don't know and don't try to make up an answer. "
        "If the answer should come from a table, look for table data in the retrieved documents "
        "but note that some detailed tables may not be fully represented in your context."
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vs.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        verbose=True
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    # Handle user query
    query = st.text_input("Ask questions about your policy:")
    if query:
        # First try to get answer from tables
        table_ans = answer_from_tables(query, all_tables)
        
        if table_ans:
            st.markdown(table_ans)
        else:
            with st.spinner("🔍 Thinking…"):
                result = qa_chain({
                    "question": query,
                    "chat_history": st.session_state.history
                })
                st.markdown("**Answer:**")
                st.write(result["answer"])
                
                # Debug: Show sources used by LLM
                with st.expander("Sources Used", expanded=False):
                    for i, doc in enumerate(result["source_documents"]):
                        st.markdown(f"**Source {i+1}:**")
                        st.text(doc.page_content[:300] + "...")
                
                st.session_state.history.append((query, result["answer"]))