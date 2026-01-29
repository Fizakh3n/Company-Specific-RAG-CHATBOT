# Document Chatbot (RAG + Streamlit)

A Retrieval-Augmented Generation (RAG) chatbot for organizational documents. Ask questions and get precise answers from your PDF documents.

Note: Currently designed for internal/company-specific use. LLM is integrated but deployment is for private use. Future work: public deployment and optimized multi-document support.

## Features

Extracts text from PDF documents

Splits documents into smart chunks for efficient retrieval

Uses vector store (FAISS) for semantic search

Generates answers using a pre-trained LLM (google/flan-t5-base)

Simple Streamlit interface for easy interaction

## 1. Clone the repository
```bash 
git clone <https://github.com/Fizakh3n/Company-Specific-RAG-CHATBOT>
cd rag_streamlit
```
## 2. Create and activate a virtual environment
``` bash
python -m venv venv
```
## Windows
``` bash
venv\Scripts\activate
```

## 3. Install dependencies
``` bash
pip install -r requirements.txt
```
## 4. Run the Streamlit app
``` bash
streamlit run app.py
```

## Future Improvements

Public deployment for multiple documents

Support for multiple PDFs at once

Optimize vector store and LLM usage for faster responses

Add richer context-aware answer generation

## License

This project is free to use for personal or organizational purposes.
