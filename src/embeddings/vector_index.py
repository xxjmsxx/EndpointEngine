import faiss
from sentence_transformers import SentenceTransformer
import config
from numpy import dot
from numpy.linalg import norm

def get_embedding_model():
    """Initialize and return the sentence transformer model"""
    return SentenceTransformer(config.EMBED_MODEL_NAME)

def create_faiss_index(entries, embed_model):
    """Create a FAISS index from entries"""
    texts = [e['text'] for e in entries]
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def compute_cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))
