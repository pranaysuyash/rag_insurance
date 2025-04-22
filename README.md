# Policy QA Agent

**Policy QA Agent** is a Streamlit application that validates and analyzes health or medical insurance policies. It leverages state-of-the-art LLMs and embeddings to:

- **Validate** whether an uploaded document is a health insurance policy
- **Extract** text from PDFs and images using OCR and PDF parsing
- **Index** content in a FAISS vector store for fast retrieval
- **Answer** user queries conversationally about policy details

---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [File Structure](#file-structure)
7. [Dependencies](#dependencies)
8. [Contributing](#contributing)
9. [License](#license)

---

## Features

- **Policy Validation**: Uses OpenAI GPT models to verify that an uploaded document is a health/medical insurance policy.
- **Document Processing**:
  - **PDF**: Parses with `langchain_community.document_loaders.PyPDFLoader`.
  - **Image**: Extracts text via OCR (`pytesseract`).
- **Vector Store Indexing**: Splits content into fixed-size chunks and indexes with FAISS for similarity search.
- **Conversational QA**: Retrieves relevant chunks and answers free-form questions in a chat-like interface.
- **Customizable Models**:
  - **Embeddings**: OpenAI, Cohere, or HuggingFace sentence-transformers.
  - **LLMs**: OpenAI GPT-3.5/4 family or local HuggingFace pipelines (e.g., GPT-2).
- **Caching**: Uses Streamlit’s caching to avoid reloading models or rebuilding indexes on every interaction.

---

## Architecture

```
┌──────────────────┐   upload    ┌───────────────┐
│  Streamlit UI    │ ──────────▶ │ PDF / Image   │
└──────────────────┘            │ Extraction    │
                                └───────────────┘
                                       │
                                       ▼
                        ┌────────────────────────┐
                        │ Text Split & Embedding │
                        └────────────────────────┘
                                       │
                                       ▼
                        ┌────────────────────────┐
                        │  FAISS Vector Store    │
                        └────────────────────────┘
                                       │
                               question│answer
                                       ▼
                        ┌────────────────────────┐
                        │ Conversational Chain   │
                        └────────────────────────┘
                                       │
                                       ▼
                                Streamlit UI
```

---

## Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/your-repo/policy-qa-agent.git
   cd policy-qa-agent
   ```

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR engine**
   - **macOS (Homebrew)**: `brew install tesseract`
   - **Ubuntu**: `sudo apt-get install tesseract-ocr`

---

## Configuration

1. Copy `.env.example` to `.env`
2. Populate environment variables:
   ```dotenv
   OPENAI_API_KEY=your_openai_api_key
   COHERE_API_KEY=your_cohere_api_key       # if using Cohere embeddings
   UPLOAD_LIMIT_MB=10
   INDEX_DIR=faiss_index
   ```

---

## Usage

```bash
streamlit run app.py
```

1. Open `http://localhost:8501` in your browser.
2. Upload one or more PDF/image files (insurance policies).
3. Validate the policy automatically.
4. Ask questions about coverage, clauses, premiums, etc.

---

## File Structure

```
├── app.py             # Main Streamlit application
├── requirements.txt   # Python dependencies
├── .env.example       # Environment variable template
├── faiss_index/       # Directory for persisted FAISS indexes
└── README.md          # This file
```

---

## Dependencies

- **Streamlit**: UI framework
- **python-dotenv**: Load `.env` configs
- **openai**: OpenAI v1.x client
- **langchain-community**: Document loaders, embeddings, chains
- **langchain-openai**: Updated OpenAIEmbedding classes
- **pypdf**: PDF parsing
- **transformers**, **torch**: Local HuggingFace models
- **faiss-cpu**: Vector store
- **pillow**, **pytesseract**: OCR
- **cohere**: Cohere embeddings (optional)

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m "Add awesome feature"`)
4. Push to your fork (`git push origin feature/YourFeature`)
5. Open a Pull Request

Please follow the existing code style and include tests where applicable.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
