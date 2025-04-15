import config

def build_entries(raw_nodes):
    """Build entries from raw nodes"""
    entries = []
    seen = set()

    for row in raw_nodes:
        var_key = row['var_name']
        var_desc = row['var_description']
        category = row['category']
        val = row['value_label']

        if val and (var_key, val) not in seen:
            text = f"Value: {val} (from {var_key} - {category})"
            entries.append({
                "text": text,
                "type": "value",
                "parent_var": var_key,
                "category": category,
                "label": val
            })
            seen.add((var_key, val))
        elif not val and var_key not in seen:
            text = f"Variable: {var_key} - {var_desc} ({category})"
            entries.append({
                "text": text,
                "type": "variable",
                "var_name": var_key,
                "description": var_desc,
                "category": category
            })
            seen.add(var_key)

    return entries

def retrieve_nodes(user_query, entries, index, embed_model, top_k=None):
    """Retrieve relevant nodes based on query"""
    if top_k is None:
        top_k = config.TOP_K

    query_embedding = embed_model.encode([user_query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for rank, i in enumerate(indices[0]):
        results.append((entries[i], float(distances[0][rank])))
    return results

def merge_results(existing, new):
    """Merge two sets of results, avoiding duplicates"""
    combined = existing[:]
    existing_texts = set(e['text'].lower() for (e, _) in existing)
    for e, d in new:
        if e['text'].lower() not in existing_texts:
            combined.append((e, d))
            existing_texts.add(e['text'].lower())
    return combined

def chunk_results(results, chunk_size=None):
    """Split results into chunks"""
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE

    for i in range(0, len(results), chunk_size):
        yield results[i:i+chunk_size]
