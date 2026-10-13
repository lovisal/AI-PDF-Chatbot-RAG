
# AI PDF Chatbot — RAG Project

An AI-powered PDF question-answering application built using Retrieval-Augmented Generation (RAG).

## Features
- Upload PDF documents
- Extract and split PDF text into chunks
- Generate semantic embeddings using SentenceTransformers
- Retrieve relevant passages using FAISS
- Generate contextual answers using the Gemini API
- Display source page references
- Interactive Streamlit chat interface

## Technologies Used
- Python
- Streamlit
- FAISS
- SentenceTransformers
- Google Gemini API
- PyPDF

## Setup

1. Clone or download this repository.
2. Install the dependencies:

   `pip install -r requirements.txt`

3. Create a `.env` file in the project folder and add:

   `GEMINI_API_KEY=your_api_key_here`

4. Run the application:

   `python -m streamlit run app.py`

## Security
Never upload your real API key or confidential PDF documents.
