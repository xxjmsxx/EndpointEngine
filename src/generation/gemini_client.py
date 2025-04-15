import google.generativeai as genai
import config
import os
from dotenv import load_dotenv

load_dotenv()

def initialize_gemini():
    """Initialize and return the Gemini model"""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-2.0-flash")

def summarize_expansions_with_llm(llm_model, user_query, expansions, chunk_size=None):
    """Summarize expansions using LLM"""
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE

    from src.retrieval.node_retrieval import chunk_results

    final_selection = []
    for chunk in chunk_results(expansions, chunk_size=chunk_size):
        chunk_text = "\n".join(f"- {c[0]['text']}" for c in chunk)

        prompt = f"""
A user asked: "{user_query}"

We have the following potential expansions:
{chunk_text}

Please list the lines (by exact text) that you believe are relevant to the query
(or say 'none' if none are relevant).
        """
        reflection_response = llm_model.generate_content(prompt).text.strip().splitlines()
        selected_lines = [line.strip("- ").strip() for line in reflection_response if line.strip()]

        for c in chunk:
            if any(c[0]['text'] in s for s in selected_lines):
                final_selection.append(c)
    return final_selection
