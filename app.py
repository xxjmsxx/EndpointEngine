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

# Initialization – only runs once
graph = get_graph_connection()
embed_model = get_embedding_model()
llm_model = initialize_gemini()

df = pd.read_excel(config.EXCEL_PATH, engine='openpyxl')
actual_columns = list(df.columns)
column_context = "\n".join(f"- {col}" for col in actual_columns)

# Define cache file paths
CACHE_DIR = "cache"
ENTRIES_CACHE = os.path.join(CACHE_DIR, "kg_entries.pkl")
FAISS_INDEX_CACHE = os.path.join(CACHE_DIR, "faiss_index.faiss")

with open(ENTRIES_CACHE, 'rb') as f:
    all_entries = pickle.load(f)

faiss_index = faiss.read_index(FAISS_INDEX_CACHE)


# Then pass these as arguments to your run_pipeline function
def run_pipeline_cached(user_input: str):
    return run_pipeline(
        user_input,
        graph=graph,
        embed_model=embed_model,
        llm_model=llm_model,
        df=df,
        all_entries=all_entries,
        faiss_index=faiss_index,
        column_context=column_context
    )

app = FastAPI()

class Query(BaseModel):
    query: str

@app.post("/analyze")
def analyze(q: Query):
    try:
        return run_pipeline_cached(q.query)
    except Exception as e:
        import traceback
        print("❌ ERROR:", str(e))
        traceback.print_exc()
        return {"error": str(e)}
