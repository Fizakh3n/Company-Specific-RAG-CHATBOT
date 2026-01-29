from fastapi import FastAPI
from pydantic import BaseModel
from app import vector_store, llm, generate_answer

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(q: Query):
    docs = vector_store.similarity_search(q.question, k=5)
    answer = generate_answer(llm, q.question, docs)
    return {"answer": answer}
