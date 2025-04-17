def format_context(results, driver):
    """Format context for presentation to LLM"""
    context_blocks = []

    for entry, dist in results:
        if entry['type'] == 'value':
            cypher_val = """
            MATCH (v:Variable {name: $var_name})-[:HAS_VALUE]->(val:Value {label: $val})
            OPTIONAL MATCH (v)-[r]->(connected)
            RETURN type(r) as rel_type, connected.name as connected_name, labels(connected) as labels
            """
            with driver.session() as session:
                result = session.run(cypher_val, {"var_name": entry["parent_var"], "val": entry["label"]})
                records = list(result)  # ✅ Cache before reuse
                rels = [record.data() for record in records]

            rel_txt = "\n".join([
                f"➪ [{r['rel_type']}] → {r['connected_name']} ({', '.join(r['labels'])})"
                for r in rels if r['connected_name']
            ])

            block = (f"Value '{entry['label']}' from Variable '{entry['parent_var']}' "
                     f"(Category: {entry.get('category','?')}, score={dist:.2f})\n{rel_txt}")

        else:  # 'variable'
            cypher_var = """
            MATCH (v:Variable {name: $var_name})
            OPTIONAL MATCH (v)-[r]->(connected)
            RETURN type(r) as rel_type, connected.name as connected_name, labels(connected) as labels
            """
            with driver.session() as session:
                result = session.run(cypher_var, {"var_name": entry["var_name"]})
                records = list(result)  # ✅ Cache before reuse
                rels = [record.data() for record in records]

            rel_txt = "\n".join([
                f"➪ [{r['rel_type']}] → {r['connected_name']} ({', '.join(r['labels'])})"
                for r in rels if r['connected_name']
            ])

            block = (f"Variable '{entry['var_name']}' - {entry['description']} "
                     f"(Category: {entry['category']}, score={dist:.2f})\n{rel_txt}")

        context_blocks.append(block)

    return "\n\n".join(context_blocks)

def reflection_loop(llm_model, user_query, current_results, entries, index, embed_model, graph, column_context, steps=2):
    """Perform reflection loop to refine results"""
    from src.retrieval.node_retrieval import retrieve_nodes, merge_results

    for _ in range(steps):
        context_text = format_context(current_results, graph)
        prompt = f"""
A user asked: "{user_query}"

Current context from the knowledge graph:
{context_text}

Column Context:
{column_context}

Based on this context and the list of column names, which variables or values might still be missing to fully answer the question?
List just their names, one per line. If you're unsure, list none.
        """
        reflection = llm_model.generate_content(prompt).text.strip().splitlines()
        new_terms = [line.strip() for line in reflection if line.strip()]

        extra_results = []
        for term in new_terms:
            # Avoid re-retrieving if it obviously overlaps existing
            if not any(term.lower() in e['text'].lower() for e, _ in current_results):
                retrieved = retrieve_nodes(term, entries, index, embed_model, top_k=5)
                extra_results = merge_results(extra_results, retrieved)

        current_results = merge_results(current_results, extra_results)
    return current_results

def generate_answer(llm_model, user_query, context, column_context, mode="default", picot=None):
    """Generate final answer based on context"""
    picot_text = ""
    if picot:
        picot_text = f"""
(PICOT format)
- Population: {picot.get("population")}
- Intervention: {picot.get("intervention")}
- Control: {picot.get("control")}
- Outcome: {picot.get("outcome")}
- Timeframe: {picot.get("timeframe")}
"""

    prompt = f"""
You are a biomedical (or general) assistant using a knowledge graph to help answer user queries.

User's question: "{user_query}"

Mode: {mode}

{picot_text}

Relevant variables/values from the graph and connections:
{context}

--- DataFrame Column Context (Use these exactly) ---
{column_context}

Please:
1) Identify which variables/values are relevant to the query.
2) Group them logically.
3) Summarize how we might filter or join data to answer the question.
4) Remember that only the number is the key (e.g., "Cardiaccomorbidity1 == 1" NOT "Cardiaccomorbidity1 == 1 - Coronary Artery Disease")
5) Use the column context to change to correct spelling (e.g., "Cardiaccomorbidity1" = "cardiaccomorbidity1" )
6) The last thing in the output should be an array of all variables that will be used during the steps. Confirm they have the same spelling as the data column context

Provide a concise yet complete explanation.
"""
    response = llm_model.generate_content(prompt)
    return response.text

def create_response_json(llm_model, final_response, user_query):
    """Create a simplified JSON response"""
    prompt = f"""
You are a biomedical assistant. Based on the following user query and the final detailed response, summarize the final conclusion into a single, clear sentence.

User query:
"{user_query}"

Final synthesized answer:
{final_response}

Respond with only the final answer as a natural-language sentence. Do not include any extra commentary.
"""

    short_answer = llm_model.generate_content(prompt).text.strip()

    return {"answer": short_answer}
