import os
from tempfile import NamedTemporaryFile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import pytesseract
from openai import OpenAI

# ─── LangChain Community & Core imports ─────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, CohereEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document

# ─── Hugging Face pipeline imports ─────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline

# ─── Load environment & configure Streamlit ─────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Policy QA Agent", layout="wide")
st.title("📄 Policy QA Agent")

# ─── Constants ─────────────────────────────────────────────
UPLOAD_LIMIT_MB = int(os.getenv("UPLOAD_LIMIT_MB", 10))
INDEX_DIR = Path(os.getenv("INDEX_DIR", "faiss_index"))

# ─── Embedding model options ─────────────────────────────────
EMBED_OPTIONS = {
    "OpenAI • text-embedding-ada-002": lambda: OpenAIEmbeddings(model="text-embedding-ada-002"),
    "Cohere • embed-english-v2.0":    lambda: CohereEmbeddings(
        model="embed-english-v2.0",
        cohere_api_key=os.getenv("COHERE_API_KEY", ""),
    ),
    "HF • multi-qa-mpnet-base-dot-v1": lambda: HuggingFaceEmbeddings(
        model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"
    ),
    "HF • all-MiniLM-L12-v2":        lambda: HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2"
    ),
}

# ─── Dynamically list available OpenAI chat models ────────────
@st.cache_resource(show_spinner=False)
def get_available_openai_models() -> list[str]:
    data = client.models.list().data
    return sorted(
        m.id
        for m in data
        if isinstance(m.id, str) and (m.id.startswith("gpt-") or m.id.startswith("gpt4"))
    )

OPENAI_MODELS = get_available_openai_models()

# ─── Sidebar UI ──────────────────────────────────────────────
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

# ─── Helper functions ────────────────────────────────────────
def sanitize(name: str) -> str:
    return Path(name).stem.replace(" ", "_")

def extract_text_from_pdf(uploaded_file) -> str:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp.flush()
        loader = PyPDFLoader(str(tmp.name))
        pages = loader.load()
    return "\n\n".join(page.page_content for page in pages)

def extract_text_from_image(uploaded_file) -> str:
    img = Image.open(uploaded_file)
    return pytesseract.image_to_string(img)

def validate_policy(text: str, model_name: str) -> bool:
    prompt = (
        "You are a policy validation assistant.\n"
        "Reply 'YES' if the document is a medical or health‑insurance policy, otherwise 'NO'.\n\n"
        f"DOCUMENT (first 1,500 chars):\n{text[:1500]}"
    )
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role":"user","content":prompt}],
        temperature=0,
    )
    verdict = resp.choices[0].message.content.strip().upper()
    return verdict.startswith("Y")

@st.cache_resource
def build_vector_store(_docs: list[Document]) -> FAISS:
    if INDEX_DIR.exists():
        return FAISS.load_local(str(INDEX_DIR), embedder)
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
    for f in uploads:
        if f.size > UPLOAD_LIMIT_MB * 1024**2:
            st.error(f"{sanitize(f.name)} exceeds {UPLOAD_LIMIT_MB} MB limit.")
            st.stop()
        if f.type == "application/pdf":
            full_text += extract_text_from_pdf(f)
        else:
            full_text += extract_text_from_image(f)
        full_text += "\n\n---DOC_BREAK---\n\n"

    if not validate_policy(full_text, llm_choice):
        st.error("⚠️ This doesn’t look like a health‑insurance policy.")
        st.stop()

    st.success("✅ Policy validated. Building vector index…")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs     = splitter.split_documents([Document(page_content=full_text)])
    vs       = build_vector_store(docs)

    # ← use ConversationalRetrievalChain + custom retriever k
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vs.as_retriever(search_kwargs={"k": 4}),
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    query = st.text_input("Ask questions about your policy:")
    if query:
        with st.spinner("🔍 Thinking…"):
            result = qa_chain({
                "question": query,
                "chat_history": st.session_state.history
            })
        st.markdown("**Answer:**")
        st.write(result["answer"])
        st.session_state.history.append((query, result["answer"]))
