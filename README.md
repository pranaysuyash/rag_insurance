# Policy QA Agent

**Policy QA Agent** is a Streamlit application that validates, indexes and answers questions about health or medical insurance policies. It combines classic metadata extraction with hybrid **RAG** (Retrieval‑Augmented Generation) so users can upload a policy PDF or scan and immediately start chatting with it.

---

## ✨ What’s new (April 2025)

- **Prompt‑fix release – no more `ValueError: variable context should be a list …`**  
  The RAG prompt now injects the retrieved context as a plain string instead of a `MessagesPlaceholder`, eliminating the runtime crash.
- **Instant answers from metadata** (e.g. commencement date, _insured‑since_ rows) before invoking the vector store – lightning‑fast for common queries.
- **Single‑file app ➡ `policy_rag_hybrid.py`** – clearer naming, easier `streamlit run`.

---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [File Structure](#file-structure)
7. [Dependencies](#dependencies)
8. [Contributing](#contributing)
9. [License](#license)

---

## 1  Features

| Category               | Description                                                                                                             |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Smart validation**   | Verifies the upload is a health/medical insurance document using GPT.                                                   |
| **OCR + PDF parsing**  | Extracts text from native PDFs _or_ scanned images via `pytesseract`.                                                   |
| **Metadata fast‑path** | Detects policy commencement / _insured since_ dates and answers certain questions instantly.                            |
| **Hybrid RAG**         | Splits text + flattened table rows, embeds with OpenAI Ada, stores in FAISS and feeds top‑K chunks to GPT‑4‑class LLMs. |
| **Conversational QA**  | Natural chat interface in Streamlit with source‑chunk disclosure.                                                       |
| **Model swap**         | Drop‑in support for alternative embedding or LLM providers (Cohere, Hugging Face, local GGUF).                          |
| **Caching**            | Parsed text, OCR output and FAISS indexes are cached for repeat uploads.                                                |

---

## 2  Architecture

```text
┌──────────────────┐  upload    ┌───────────────┐
│  Streamlit UI    │ ─────────▶ │  PDF / Image  │
└──────────────────┘            │ Extraction    │
                                └───────────────┘
                                       │ plain‑text + rows
                                       ▼
                        ┌────────────────────────┐
                        │ Split, Embed (Ada‑002) │
                        └────────────────────────┘
                                       │
                                       ▼
                        ┌────────────────────────┐
                        │   FAISS Vector Store   │
                        └────────────────────────┘
                                       │
                               question│answer
                                       ▼
                        ┌────────────────────────┐
                        │  Conversational RAG    │
                        └────────────────────────┘
                                       │ rescue‑LLM (optional)
                                       ▼
                                Streamlit UI
```

---

## 3  Installation

```bash
# 1. Clone
$ git clone https://github.com/your‑repo/policy‑qa‑agent.git
$ cd policy‑qa‑agent

# 2. Virtual env
$ python3 -m venv venv && source venv/bin/activate

# 3. Dependencies
$ pip install -r requirements.txt

# 4. Tesseract (if you need OCR)
# macOS:  brew install tesseract
# Ubuntu: sudo apt-get install tesseract-ocr
```

---

## 4  Configuration

1. Copy `.env.example` ➜ `.env`
2. Fill in keys / tweak limits:

```dotenv
OPENAI_API_KEY=your_openai_key
UPLOAD_LIMIT_MB=25
INDEX_DIR=faiss_hybrid
PRIMARY_MODEL=gpt-4o-mini
```

---

## 5  Usage

```bash
streamlit run policy_rag_hybrid.py
```

1. Visit `http://localhost:8501`.
2. Drag‑and‑drop one or more policy PDFs/scans.
3. Ask questions like:
   - “Since when am I with Niva Bupa?”
   - “What is the waiting period for cataract surgery?”
4. The answer panel shows the reply + expandable source snippets.

---

## 6  File Structure

```text
├── policy_rag_hybrid.py   # Main Streamlit app (this repo)
├── requirements.txt       # Python deps
├── .env.example           # Env template
├── faiss_hybrid/          # Persisted FAISS indexes (auto‑created)
└── README.md              # You are here
```

---

## 7  Dependencies (pip)

- **streamlit** / **python‑dotenv**
- **openai** (& `langchain-openai`)
- **langchain‑community**
- **faiss‑cpu**
- **pypdf**, **pdfplumber**, **pdf2image**
- **pillow**, **pytesseract**
- **transformers**, **torch** (optional local LLMs)
- **cohere** (optional embeddings)

---

## 8  Contributing

```bash
git checkout -b feat/my-awesome‑thing
# hack…
git commit -m "feat: add my awesome thing"
git push origin feat/my-awesome‑thing
```

Then open a Pull Request 🎉. Please match the existing code style and include tests if the feature is logic‑heavy.

---

## 9  License

[MIT](LICENSE)
