# Local RAG Document Q&A

A retrieval-augmented generation (RAG) system that answers questions about your own PDFs and text files. It runs entirely on your machine with free, open-source tools: no API keys and no per-request cost.

Every answer comes back with the source documents it was drawn from, and the system says "I don't know" when the documents don't cover the question.

## How it works

```
Documents -> extract text -> chunk -> embed -> FAISS index
Question  -> embed -> FAISS search -> top chunks -> grounded prompt -> Ollama -> answer + sources
```

1. **Ingest:** PDFs are loaded, whitespace is cleaned up, and the text is split into overlapping chunks of about 200 tokens
2. **Embed:** each chunk becomes a 384-dimension vector using `sentence-transformers` (`all-MiniLM-L6-v2`)
3. **Index:** vectors are normalized and stored in a FAISS index; chunk text and source filenames are saved alongside it
4. **Retrieve:** a question is embedded and the closest chunks are found by cosine similarity. If the question names a document (like `HCC_3`), the search is restricted to that file
5. **Generate:** the retrieved chunks are passed to a local `llama3.2` model with an instruction to answer only from that context
6. **Guardrails:** chunks below a similarity threshold are discarded, and if nothing relevant is found the system returns "I don't know" with no sources

## Tech stack

Python, FastAPI, sentence-transformers, FAISS, Ollama (llama3.2), pypdf, tiktoken

## Setup

1. Install [Ollama](https://ollama.com/download) and pull the model:
```bash
   ollama pull llama3.2
```
2. Create a virtual environment and install dependencies:
```bash
   python -m venv venv
   venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
   pip install -r requirements.txt
```
3. Create a `documents/` folder and add your own PDFs or `.txt` files
4. Build the index:
```bash
   python index.py
```
5. Start the API:
```bash
   uvicorn main:app --reload
```
6. Open `http://localhost:8000/docs` and try `POST /ask`

## Example

Request:
```json
{ "question": "What is the capital of France?", "top_k": 5 }
```
Response when the documents don't cover it:
```json
{ "answer": "I don't know.", "sources": [] }
```

## Limitations

- **Vague questions retrieve poorly.** "What did I conclude in my reflection?" gives retrieval nothing specific to match
- **The 3B model sometimes hedges** on broad definition questions even when the context contains a definition
- **The document filter is simple.** It only matches filenames shaped like `HCC_3`
- **Sources list every retrieved document,** not only the ones the answer actually used

## Possible improvements

- Reranking with a cross-encoder
- Query rewriting for vague questions
- A larger local model
- A small regression test set of questions
- Docker Compose deployment (FastAPI plus Ollama)
