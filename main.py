from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from generate import answer_question

app = FastAPI(title="RAG Document Q&A")


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/ask")
def ask(payload: QuestionRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    return answer_question(payload.question, top_k=payload.top_k)