# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-functional Streamlit web application that integrates Google's Gemini AI API for image and text analysis with a specialized NGWord detection system for marketing compliance in Japan. The app serves as a comprehensive content analysis tool with focus on cosmetics and pharmaceutical advertising compliance.

## Key Commands

### Development Setup
```bash
# Activate Python 3.13 environment (managed by pyenv)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application locally
streamlit run app.py
```

### NGWord Database Management
```bash
# Setup/rebuild the NGWord vector database
python setup_ngword_db_simple.py

# Test NGWord search functionality standalone
streamlit run search_ngwords_simple.py
```

### Testing and Validation
```bash
# Run the app and check all three tabs
streamlit run app.py
# Then test:
# 1. Image analysis with JSON output and automatic NGWord detection
# 2. File analysis with Gemini
# 3. NGWord search (single and batch)
```

## Architecture

### Core Components

1. **app.py** - Main application with three functional tabs:
   - Image Analysis Tab: Uploads images, analyzes with Gemini using structured prompts, outputs JSON with automatic NGWord checking on query_strings
   - File Analysis Tab: Analyzes text files with Gemini using customizable prompts
   - NGWord Search Tab: Vector similarity search for prohibited marketing terms

2. **NGWord System** - TF-IDF based vector search for compliance:
   - `setup_ngword_db_simple.py`: Creates ChromaDB with TF-IDF vectorization from input.csv
   - `search_ngwords_simple.py`: Standalone search interface
   - `input.csv`: 100+ NGWords with replacements, reasons, and risk levels
   - `chroma_db/`: Persistent vector database storage
   - `tfidf_vectorizer.pkl`: Serialized TF-IDF model

### Data Flow

1. **Image Analysis Pipeline**:
   - Image upload → Gemini API with fixed JSON prompt → Parse JSON response
   - Extract query_strings from JSON → Automatic NGWord search
   - Display results with risk indicators (⚠️ for NGWords detected)

2. **NGWord Detection**:
   - Text input → TF-IDF vectorization → ChromaDB similarity search
   - Returns matches with similarity scores, risk levels (high/mid/low), and replacement suggestions
   - Batch processing capability for multiple text checks

### Authentication & Security

- Basic authentication using environment variables (AUTH_USERNAME, AUTH_PASSWORD)
- API keys stored in .env locally or Streamlit Secrets for deployment
- Password-protected access with session state management

## Environment Configuration

### Required Environment Variables
```bash
GEMINI_API_KEY=<your-gemini-api-key>
AUTH_USERNAME=<username>  # Optional, defaults provided
AUTH_PASSWORD=<password>  # Optional, defaults provided
```

### Python Version
- Python 3.13.7 (specified in .python-version and runtime.txt)
- Virtual environment at `venv/`

## JSON Output Format

The image analysis produces structured JSON:
```json
[
  {
    "id": 1,
    "object": [{"label": "...", "category": "人の顔/化粧品容器/その他"}],
    "text": ["extracted", "texts"],
    "source": "filename",
    "query_string": ["searchable", "strings"]
  }
]
```

## Key Implementation Details

- **SimpleJapaneseVectorizer class**: Custom TF-IDF wrapper optimized for Japanese text (2-4 character n-grams)
- **check_ngwords_in_query_strings()**: Automatically checks all query_strings against NGWord database
- **Cached resources**: Vectorizer and database loaded once using @st.cache_resource
- **Session state**: Manages authentication, model selection, and analysis history

## Deployment Notes

- Streamlit Community Cloud compatible
- Requires runtime.txt with Python version
- ChromaDB and pickle files must be committed to repository
- Consider memory limits when processing large images or files