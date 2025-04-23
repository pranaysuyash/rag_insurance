# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run app: `streamlit run app.py`
- Install dependencies: `pip install -r requirements.txt`
- Virtual environment: `python -m venv venv && source venv/bin/activate`
- Install Tesseract OCR: macOS `brew install tesseract`, Ubuntu `sudo apt-get install tesseract-ocr`

## Code Style
- **Imports**: Group imports by standard library, third-party, and local
- **Formatting**: Use section separators `# ─── Section Name ─────────`
- **Types**: Use type annotations throughout (e.g., `def func(x: str) -> bool:`)
- **Naming**: Use snake_case for functions/variables, PascalCase for classes
- **Error Handling**: Use try/except with specific exceptions and helpful error messages
- **Docstrings**: Include for functions, especially utility functions
- **Debug Utilities**: Use debug_print() with optional obj parameter
- **Patterns**: Follow pattern matching approach in direct_answer_from_tables

## Project Structure
- Single app.py file with modular function organization
- FAISS vector store persisted in faiss_index/
- Environment variables in .env file (API keys, config)