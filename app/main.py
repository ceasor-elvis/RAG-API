from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, Request
from typing import Union, List
from pydantic import BaseModel
from funcs import Rag

app = FastAPI()
templates = Jinja2Templates(directory="templates")

########## Endpoints for site views #########
@app.get("/", response_class=HTMLResponse, tags=["Site views"])  
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

########## Document ingestion and management ##########
@app.post("/documents", tags=["Document ingestion and management"])
async def insert_docs(document: str, metadata:Union[dict, None] = None):
    """Upload or add a new document to the knowledge base or vector store"""
    try:
        rag = Rag()
        rag.store_documents(document)
        return {
            "status": "success",
            "message": "Document added successfully."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/documents", tags=["Document ingestion and management"])
async def get_docs(size: int = 5):
    return Rag().vector_store.get()

@app.get("/documents/{id}", tags=["Document ingestion and management"])
async def get_doc_by_id(id: int):
    return {}

@app.delete("/documents/{id}", tags=["Document ingestion and management"])
async def delete_doc_by_id(id: int):
    return {}

########## Retrieval and Generation Endpoints ##########
@app.get("/query", tags=["Retrieval and Generation"])
async def retrieve_doc(query: str, top_k: Union[int, None] = None):
    """Retrieves documents or data points based on a query."""
    return {
        "query": "What is RAG?",
        "results": [
            {
                "id": "1",
                "title": "Document title",
                "snippet": "Relevant content snippet",
                "score": 0.95
            }
        ]
    }

@app.post("/generate", tags=["Retrieval and Generation"])
async def generate(query: str, retrieved_context: List[str]):
    """Generates an output based on retrieved documents and query"""
    return {
        "query":"What ia RAG",
        "generated_response": "RAG stands for Retrieval-Agumented Generation..."
    }

@app.post("/query-and-generate", tags=["Retrieval and Generation"])
async def query_and_generate(query: str, top_k: Union[int, None] = None):
    """Combines retrieval and generation """
    retrieved_docs = retrieve_doc(query, top_k)
    generated_response = generate(query, retrieved_docs)
    return {
        "query": query,
        "retrieved_context": retrieved_docs,
        "generated_response": generated_response["generated_response"]
    }

@app.get("/suggest", tags=["Retrieval and Generation"])
async def suggestions(prefix: str):
    """Provides or auto-complete options based on a partial query."""
    return {"suggestions": ["How to...", "What is...", "Why does..."]}


########## Maintanance ##########
@app.get("/health", tags=["Health and Maintenance"])
async def health_check():
    """Check if the service is running correctly"""
    return {
        "status": "ok",
        "retriever": "active",
        "generator": "active"
    }

@app.post("/feedback", tags=["Health and Maintenance"])
async def feedback(query: str, feedback: str, comments: str):
    return {
    }