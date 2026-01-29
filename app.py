import os
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

st.set_page_config(page_title="RAG Chatbot", page_icon="📄")

# ---------- Functions ----------

@st.cache_data(show_spinner=False)
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

@st.cache_data(show_spinner=False)
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(text)

@st.cache_resource(show_spinner=True)
def load_vector_store(pdf_path="sample.pdf"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists("faiss_index"):
        st.info("🔹 Loading saved vector store...")
        return FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

    st.info("🔹 Creating new vector store...")
    text = load_pdf(pdf_path)
    chunks = chunk_text(text)
    store = FAISS.from_texts(chunks, embeddings)
    store.save_local("faiss_index")
    st.success("✅ Vector store created and saved locally!")
    return store

@st.cache_resource(show_spinner=True)
def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=200
    )

def generate_answer(llm, question, docs):
    context = "\n\n".join(d.page_content for d in docs)
    prompt = f"""
Answer strictly using the context.
If not found, say: Not found in the document.

Context:
{context}

Question:
{question}

Answer:
"""
    return llm(prompt)[0]["generated_text"].strip()


# ---------- App UI ----------

st.title("📄 RAG Chatbot -Company Policies")
st.write("Ask questions about your document and get precise answers!")

# Initialize vector store and LLM once
vector_store = load_vector_store("sample.pdf")
llm = load_llm()

# Question input
query = st.text_input("Type your question here:")

if query:
    with st.spinner(" Fetching answer..."):
        docs = vector_store.similarity_search(query, k=5)
        answer = generate_answer(llm, query, docs)
    st.markdown("**Answer:**")
    st.write(answer)
