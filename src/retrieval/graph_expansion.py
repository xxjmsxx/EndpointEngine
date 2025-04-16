import config

def expand_graph_from_variable_filtered(graph, var_name, user_query, embed_model, similarity_threshold=None):
    """Expand graph from a variable with filtering by relevance"""
    if similarity_threshold is None:
        similarity_threshold = config.SIMILARITY_THRESHOLD

    # Get query embedding using the API function
    query_embedding = embed_model(user_query)

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

    # Process in batches to avoid API limits
    batch_size = 20
    texts = []
    row_map = []

    for row in data:
        related_var = row['related_var']
        related_val = row['related_val']
        if related_var and related_val:
            text = f"Value: {related_val} (from {related_var} - expanded)"
            texts.append(text)
            row_map.append((related_var, related_val))

    if not texts:
        return expansions

    # Get embeddings for all texts in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = embed_model(batch_texts)

        for j, text_embed in enumerate(batch_embeddings):
            sim = compute_cosine_similarity(query_embedding, text_embed)
            if sim >= similarity_threshold:
                related_var, related_val = row_map[i+j]
                expansions.append((
                    {
                        "text": texts[i+j],
                        "type": "value",
                        "parent_var": related_var,
                        "label": related_val,
                        "category": "unknown"
                    },
                    1.0
                ))

    return expansions
