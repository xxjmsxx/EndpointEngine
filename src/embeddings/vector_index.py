import faiss
import numpy as np
import os
from dotenv import load_dotenv
import config
from numpy import dot
from numpy.linalg import norm
from huggingface_hub import InferenceClient

load_dotenv()

def get_embedding_model():
    """Return a function that uses HuggingFace's InferenceClient for embeddings"""
    api_key = os.getenv("HF_API_KEY")
    model_name = os.getenv("HF_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

    # Initialize the client
    client = InferenceClient(
        model=model_name,
        api_key=api_key
    )

    def get_embeddings(texts):
        """Get embeddings using HuggingFace InferenceClient"""
        if not isinstance(texts, list):
            is_single = True
            texts = [texts]
        else:
            is_single = False

        # Process in smaller batches for API stability
        batch_size = 20
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            # Get embeddings for the batch
            batch_embeddings = []
            for text in batch:
                embedding = client.feature_extraction(text)
                batch_embeddings.append(embedding)

            all_embeddings.extend(batch_embeddings)

        # Convert to numpy array with the right shape
        embeddings_array = np.array(all_embeddings)

        # Return single vector if input was single text
        if is_single:
            return embeddings_array[0]
        return embeddings_array

    return get_embeddings

def create_faiss_index(entries, embed_model):
    """Create a FAISS index from entries using external API"""
    texts = [e['text'] for e in entries]

    # Process in batches to avoid API limits
    batch_size = 20
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = embed_model(batch_texts)

        if i == 0:
            # Initialize array with correct dimensions based on first batch
            dim = batch_embeddings.shape[1]
            all_embeddings = np.zeros((len(texts), dim))

        all_embeddings[i:i+len(batch_texts)] = batch_embeddings

    # Create FAISS index
    dim = all_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(all_embeddings)

    return index, all_embeddings

def compute_cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))
