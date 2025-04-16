from fastapi import FastAPI
from pydantic import BaseModel
from main import run_pipeline
import config
import os
import uvicorn
import pickle
import faiss
import dask.dataframe as dd
import time
import traceback
from src.database.neo4j_client import get_graph_connection, fetch_variable_and_value_nodes
from src.embeddings.vector_index import get_embedding_model, create_faiss_index
from src.retrieval.node_retrieval import build_entries
from src.generation.gemini_client import initialize_gemini

from fastapi import FastAPI
from contextlib import asynccontextmanager
from threading import Thread

# Shared resources and status tracking
resources = {}
initialized = False
init_error = None
init_stage = "Not started"
init_start_time = None

class Query(BaseModel):
    query: str

class StatusResponse(BaseModel):
    initialized: bool
    stage: str
    error: str = None
    elapsed_time: float = None
    progress: list = None

def init_all():
    global resources, initialized, init_error, init_stage, init_start_time

    init_start_time = time.time()
    progress_steps = []

    try:
        init_stage = "Connecting to Neo4j graph database"
        print(f"⚙️ {init_stage}...")
        graph = get_graph_connection()
        progress_steps.append(f"✅ {init_stage} - Complete")

        init_stage = "Loading embedding model"
        print(f"⚙️ {init_stage}...")
        embed_model = get_embedding_model()
        progress_steps.append(f"✅ {init_stage} - Complete")

        init_stage = "Initializing Gemini LLM"
        print(f"⚙️ {init_stage}...")
        llm_model = initialize_gemini()
        progress_steps.append(f"✅ {init_stage} - Complete")

        init_stage = "Loading Excel data (converted CSV via Dask)"
        print(f"⚙️ {init_stage}...")
        df = dd.read_csv(config.EXCEL_PATH.replace('.xlsx', '.csv'), blocksize="10MB")
        actual_columns = list(df.columns)
        column_context = "\n".join(f"- {col}" for col in actual_columns)
        progress_steps.append(f"✅ {init_stage} - Complete")

        init_stage = "Loading cached knowledge graph entries"
        print(f"⚙️ {init_stage}...")
        CACHE_DIR = "cache"
        ENTRIES_CACHE = os.path.join(CACHE_DIR, "kg_entries.pkl")
        with open(ENTRIES_CACHE, 'rb') as f:
            all_entries = pickle.load(f)
        progress_steps.append(f"✅ {init_stage} - Complete")

        init_stage = "Loading FAISS vector index"
        print(f"⚙️ {init_stage}...")
        FAISS_INDEX_CACHE = os.path.join(CACHE_DIR, "faiss_index.faiss")
        faiss_index = faiss.read_index(FAISS_INDEX_CACHE)
        progress_steps.append(f"✅ {init_stage} - Complete")

        resources = {
            "graph": graph,
            "embed_model": embed_model,
            "llm_model": llm_model,
            "df": df,
            "all_entries": all_entries,
            "faiss_index": faiss_index,
            "column_context": column_context,
            "progress_steps": progress_steps
        }

        init_stage = "Initialization complete"
        initialized = True
        print(f"✅ {init_stage} in {time.time() - init_start_time:.2f} seconds")

    except Exception as e:
        error_msg = f"❌ ERROR during {init_stage}: {str(e)}"
        init_error = error_msg
        print(error_msg)
        traceback.print_exc()
        resources["progress_steps"] = progress_steps

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background initialization without blocking server startup
    init_thread = Thread(target=init_all)
    init_thread.start()
    yield
    # Cleanup operations can go here (if needed)
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

@app.get("/status")
def get_status():
    """Get the current initialization status of the system"""
    global init_start_time

    elapsed = None
    if init_start_time:
        elapsed = time.time() - init_start_time

    progress_steps = resources.get("progress_steps", [])

    return StatusResponse(
        initialized=initialized,
        stage=init_stage,
        error=init_error,
        elapsed_time=elapsed,
        progress=progress_steps
    )

@app.post("/analyze")
def analyze(q: Query):
    if not initialized:
        elapsed = "unknown"
        if init_start_time:
            elapsed = f"{time.time() - init_start_time:.2f} seconds"

        if init_error:
            return {
                "status": "initialization_failed",
                "error": init_error,
                "elapsed": elapsed
            }
        else:
            return {
                "status": "initializing",
                "current_stage": init_stage,
                "elapsed": elapsed,
                "progress": resources.get("progress_steps", [])
            }

    try:
        return run_pipeline(
            q.query,
            **{k: v for k, v in resources.items() if k != "progress_steps"}
        )
    except Exception as e:
        error_msg = str(e)
        print("❌ ERROR during query processing:", error_msg)
        traceback.print_exc()
        return {"error": error_msg}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
