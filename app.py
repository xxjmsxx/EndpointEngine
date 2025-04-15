from fastapi import FastAPI
from pydantic import BaseModel
from main import run_pipeline
import config
import os
import pickle
import faiss
import pandas as pd
from src.database.neo4j_client import get_graph_connection, fetch_variable_and_value_nodes
from src.embeddings.vector_index import get_embedding_model, create_faiss_index
from src.retrieval.node_retrieval import build_entries
from src.generation.gemini_client import initialize_gemini

from fastapi import FastAPI
from contextlib import asynccontextmanager
from threading import Thread

# Shared resources
resources = {}
initialized = False

class Query(BaseModel):
    query: str

def init_all():
    global resources, initialized
    print("⚙️ Initializing resources...")

    graph = get_graph_connection()
    embed_model = get_embedding_model()
    llm_model = initialize_gemini()

    df = pd.read_excel(config.EXCEL_PATH, engine='openpyxl')
    actual_columns = list(df.columns)
    column_context = "\n".join(f"- {col}" for col in actual_columns)

    CACHE_DIR = "cache"
    ENTRIES_CACHE = os.path.join(CACHE_DIR, "kg_entries.pkl")
    FAISS_INDEX_CACHE = os.path.join(CACHE_DIR, "faiss_index.faiss")

    with open(ENTRIES_CACHE, 'rb') as f:
        all_entries = pickle.load(f)

    faiss_index = faiss.read_index(FAISS_INDEX_CACHE)

    resources = {
        "graph": graph,
        "embed_model": embed_model,
        "llm_model": llm_model,
        "df": df,
        "all_entries": all_entries,
        "faiss_index": faiss_index,
        "column_context": column_context,
    }
    initialized = True
    print("✅ Initialization complete")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background initialization without blocking server startup
    Thread(target=init_all).start()
    yield
    # Cleanup operations can go here (if needed)
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

@app.post("/analyze")
def analyze(q: Query):
    if not initialized:
        return {"status": "initializing, please try again shortly"}
    try:
        return run_pipeline(
            q.query,
            **resources
        )
    except Exception as e:
        import traceback
        print("❌ ERROR:", str(e))
        traceback.print_exc()
        return {"error": str(e)}
