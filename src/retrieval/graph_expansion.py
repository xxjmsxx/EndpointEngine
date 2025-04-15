import config

def expand_graph_from_variable_filtered(graph, var_name, user_query, embed_model, similarity_threshold=None):
    """Expand graph from a variable with filtering by relevance"""
    if similarity_threshold is None:
        similarity_threshold = config.SIMILARITY_THRESHOLD

    query_embedding = embed_model.encode([user_query], convert_to_numpy=True)[0]

    cypher = """
    MATCH (v:Variable {name: $var_name})
    OPTIONAL MATCH (v)-[:HAS_VALUE]->(val:Value)
    OPTIONAL MATCH (v)-[r]->(related:Variable)
    OPTIONAL MATCH (related)-[:HAS_VALUE]->(val2:Value)
    RETURN DISTINCT related.name AS related_var,
                    related.description AS related_desc,
                    val2.label AS related_val,
                    labels(related) AS labels
    """
    data = graph.run(cypher, var_name=var_name).data()
    expansions = []

    from src.embeddings.vector_index import compute_cosine_similarity

    for row in data:
        related_var = row['related_var']
        related_val = row['related_val']
        if related_var and related_val:
            text = f"Value: {related_val} (from {related_var} - expanded)"
            text_embed = embed_model.encode([text], convert_to_numpy=True)[0]
            sim = compute_cosine_similarity(query_embedding, text_embed)
            if sim >= similarity_threshold:
                expansions.append((
                    {
                        "text": text,
                        "type": "value",
                        "parent_var": related_var,
                        "label": related_val,
                        "category": "unknown"
                    },
                    1.0
                ))
    return expansions
