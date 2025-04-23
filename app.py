"""
policy_rag_hybrid.py   –   Health-insurance QA (generic, GPT-4-class)

Run:
    streamlit run policy_rag_hybrid.py

Flow:
    • Upload one or more policy PDFs.
    • Ask a free-form question, e.g. “since when am I with Niva Bupa”.
"""

import os, re, json, hashlib, time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Tuple, Dict, Any
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
import pdfplumber, pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter            import RecursiveCharacterTextSplitter
from langchain_community.embeddings     import OpenAIEmbeddings
from langchain_community.vectorstores   import FAISS
from langchain_community.chat_models    import ChatOpenAI
from langchain.schema                   import Document
from langchain.chains                   import ConversationalRetrievalChain
from langchain.prompts                  import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# ───────────────────────── env / model prefs ─────────────────────────
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found in .env"); st.stop()

primary_model   = os.getenv("PRIMARY_MODEL",   "gpt-4o-mini")
validator_model = os.getenv("VALIDATOR_MODEL", "gpt-4o-mini")
rescue_model    = os.getenv("RESCUE_MODEL",    "gpt-4o-mini")

# ───────────────────────── constants ─────────────────────────
upload_limit_mb = 25
index_dir = Path("faiss_hybrid"); index_dir.mkdir(exist_ok=True)
date_rgx = r"[0-9]{1,2}[\/\-.][0-9]{1,2}[\/\-.][0-9]{2,4}"

# ───────────────────────── small helpers ─────────────────────────
def clean_header(cell: Any, idx: int) -> str:
    cell = re.sub(r"\W+", "_", str(cell).strip().lower()) or f"col_{idx}"
    return re.sub(r"_+", "_", cell).strip("_")

def is_scanned(pdf_path: str) -> bool:
    try:
        return all("/Font" not in p.get("/Resources", {}) for p in PdfReader(pdf_path).pages)
    except Exception:
        return False

def ocr_pdf(pdf_path: str) -> str:
    imgs = convert_from_path(pdf_path, dpi=300, fmt="png")
    return "\n".join(pytesseract.image_to_string(i) for i in imgs)

def sha1(files) -> str:
    h = hashlib.sha1()
    for f in files:
        h.update(f.name.encode()); h.update(str(f.size).encode())
    return h.hexdigest()

# ───────────────────────── pdf → text + rows ─────────────────────────
def parse_pdf(pdf_path: str, prog) -> Tuple[str, List[str]]:
    text = ocr_pdf(pdf_path) if is_scanned(pdf_path) else \
           "\n\n".join(p.page_content for p in PyPDFLoader(pdf_path).load())

    rows: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages, 1):
            try:
                tables = page.extract_tables() or []
            except Exception:
                prog.progress(i / total); continue
            for t_idx, tbl in enumerate(tables):
                if not tbl or len(tbl) < 2:
                    continue
                hdrs = [clean_header(h, j) for j, h in enumerate(tbl[0])]
                df   = pd.DataFrame(tbl[1:], columns=hdrs)
                for _, r in df.iterrows():
                    cells = {c: str(r[c]).strip()
                             for c in df.columns
                             if str(r[c]).strip() not in ("", "None", "none")}
                    if cells:
                        rows.append(json.dumps(
                            {"row_page": i, "table": t_idx, "cells": cells},
                            ensure_ascii=False
                        ))
            prog.progress(i / total)
    return text, rows

# ───────────────────────── metadata shortcuts ─────────────────────────
def most_recent(raw: List[str]) -> str:
    out = []
    for d in raw:
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
                    "%d/%m/%y", "%d-%m-%y", "%d.%m.%y"):
            try:
                out.append(datetime.strptime(d, fmt)); break
            except ValueError:
                continue
    return max(out).strftime("%d/%m/%Y") if out else raw[0]

def extract_metadata(text: str, rows: List[str]) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    # free-text commencement
    dates = re.findall(r"policy\s+commencement[^0-9]*" + date_rgx, text, re.I)
    if dates:
        meta["commencement_date"] = most_recent(dates)

    # insured … since (rows, fuzzy)
    for r in rows:
        try:
            obj = json.loads(r)
            for k, v in obj.get("cells", {}).items():
                kl = k.lower()
                if "insured" in kl and "since" in kl:
                    if re.search(date_rgx, v):
                        meta["insured_since"] = re.search(date_rgx, v).group(0)
                        raise StopIteration
        except Exception:
            continue
    return meta

def meta_answer(q: str, meta: Dict[str, str]) -> str | None:
    ql = q.lower()
    if ("commencement" in ql or "start" in ql or "effective" in ql) and meta.get("commencement_date"):
        return f"The current policy commencement date is **{meta['commencement_date']}**."
    if ("since" in ql or "member since" in ql) and meta.get("insured_since"):
        return f"You have been insured with Niva Bupa since **{meta['insured_since']}**."
    return None

# ───────────────────────── vectorstore ─────────────────────────
def build_vs(digest: str, docs: List[Document], embedder):
    path = index_dir / digest
    if path.exists():
        return FAISS.load_local(str(path), embedder,
                                allow_dangerous_deserialization=True)
    vs = FAISS.from_documents(docs, embedder)
    vs.save_local(str(path))
    return vs

# ───────────────────────── rescue (full reread) ─────────────────────────
def rescue_answer(full_text: str, q: str) -> str:
    llm = ChatOpenAI(model_name=rescue_model, temperature=0)
    prompt = (
        "You are an expert assistant.\n"
        "Given the full plain-text of an insurance policy below, answer the user's question. "
        "If the answer is not explicitly present, reply exactly “I don’t know.”\n\n"
        f"--- POLICY START ---\n{full_text}\n--- POLICY END ---\n\n"
        f"QUESTION: {q}\nANSWER:"
    )
    return llm.invoke(prompt).content.strip()

# ───────────────────────── Streamlit UI ─────────────────────────
st.set_page_config(page_title="Policy QA – hybrid RAG", layout="wide")
st.title("📄 Policy QA (hybrid RAG)")

files = st.file_uploader("Upload policy PDF(s)", type=["pdf"], accept_multiple_files=True)
if not files:
    st.stop()

for f in files:
    if f.size > upload_limit_mb * 1024**2:
        st.error(f"{f.name} exceeds {upload_limit_mb} MB limit"); st.stop()

embedder   = OpenAIEmbeddings(model="text-embedding-ada-002")
primary_llm = ChatOpenAI(model_name=primary_model, temperature=0)

raw_text, flat_rows = "", []
with st.spinner("Parsing PDFs …"):
    for up in files:
        bar = st.progress(0.0, text=f"⇢ {up.name}")
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(up.read()); tmp.flush()
            txt, rows = parse_pdf(tmp.name, bar)
        raw_text += txt + "\n\n---DOC_BREAK---\n\n"
        flat_rows.extend(rows)
        bar.empty()

metadata = extract_metadata(raw_text, flat_rows)

query = st.text_input("💬 Ask a question about your policy:")
if not query:
    st.stop()

# ─── stage 1 : metadata fast-path
if ans := meta_answer(query, metadata):
    st.markdown("### 📝 Answer"); st.write(ans)
    st.info("✅ Answered instantly from metadata")
    st.stop()

# ─── stage 2 : vector RAG
digest = sha1(files)
docs   = [Document(page_content=raw_text)] + \
         [Document(page_content=r) for r in flat_rows]
chunks = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)\
           .split_documents(docs)
vs = build_vs(digest, chunks, embedder)

system_msg = (
    "You answer questions about insurance policies. "
    "Some context chunks are flattened table-rows in JSON like "
    '{{"row_page":3,"table":0,"cells":{{"insured_with_niva_bupa_since":"07/08/2017"}}}}. '
    "Quote exact values and append “(row_page_X)” if the value came from such a row. "
    'If you can’t find the answer, reply exactly “I don’t know.”'
)

# *** FIX: inject context as a string, not a MessagesPlaceholder ***
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        system_msg + "\n\nRelevant context:\n{context}"
    ),
    HumanMessagePromptTemplate.from_template("{question}")
])

qa_chain = ConversationalRetrievalChain.from_llm(
    llm          = primary_llm,
    retriever    = vs.as_retriever(search_kwargs={"k": 10}),
    combine_docs_chain_kwargs = {"prompt": prompt},
    return_source_documents   = True
)

with st.spinner("🔍 RAG is thinking …"):
    result = qa_chain({"question": query, "chat_history": []})

rag_answer = result["answer"].strip()

# ─── stage 3 : rescue reread, if needed
if rag_answer.lower().startswith("i don’t know"):
    with st.spinner("🤔 Running rescue pass …"):
        rag_answer = rescue_answer(raw_text, query)

st.markdown("### 📝 Answer"); st.write(rag_answer)

with st.expander("🔎 Source chunks (RAG)"):
    for i, doc in enumerate(result.get("source_documents", [])):
        snippet = doc.page_content.replace("\n", " ")[:400]
        st.markdown(f"**{i+1}.** {snippet}…")
