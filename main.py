import json
import pandas as pd
import sys


import numpy as np

# Add the project root to sys.path
sys.path.append('.')

import config

# Import necessary modules
from src.retrieval.node_retrieval import retrieve_nodes, merge_results
from src.retrieval.graph_expansion import expand_graph_from_variable_filtered
from src.generation.gemini_client import summarize_expansions_with_llm
from src.generation.answer_generation import format_context, reflection_loop, generate_answer, create_response_json
from src.execution.plan_generation import generate_plan
from src.execution.plan_execution import execute_plan
from src.database.neo4j_client import get_all_values_for_variables, extract_variable_array_from_text


def run_pipeline(
    user_input: str,
    graph,
    embed_model,
    llm_model,
    df,
    all_entries,
    faiss_index,
    column_context
):
    try:
        parsed = json.loads(user_input)
        user_query = parsed.get("fullQuestion") or "No question provided"
        mode = parsed.get("mode", "default")
        picot = parsed.get("picot", {})
    except json.JSONDecodeError:
        user_query = user_input
        mode = "default"
        picot = {}

    # 2) Initial retrieval
    results = retrieve_nodes(user_query, all_entries, faiss_index, embed_model, top_k=config.TOP_K)
    print("\n🔍 Initial Concepts (from FAISS vector search):")
    for (entry, dist) in results:
        if entry['type'] == 'value':
            print(f"  ✔ Value '{entry['label']}' from '{entry['parent_var']}' (score={dist:.2f})")
        else:
            print(f"  ✔ Variable '{entry['var_name']}' (score={dist:.2f})")

    # 3) Reflection Loop
    reflected_results = reflection_loop(llm_model, user_query, results, all_entries, faiss_index, embed_model, graph, column_context, steps=2)

    # 4) Graph expansion
    expansions = []
    for (entry, dist) in reflected_results:
        if entry['type'] == 'variable':
            expansions_for_var = expand_graph_from_variable_filtered(graph, entry['var_name'], user_query, embed_model)
            expansions = merge_results(expansions, expansions_for_var)
    expansions = summarize_expansions_with_llm(llm_model, user_query, expansions)
    full_results = merge_results(reflected_results, expansions)

    # 5) Final graph-based context
    print("\n⟲ Final Expanded Results (Reflections + Expansions):")
    for (entry, _) in full_results:
        if entry['type'] == 'value':
            print(f"  ✔ Value '{entry['label']}' from '{entry['parent_var']}'")
        else:
            print(f"  ✔ Variable '{entry['var_name']}'")

    final_context = format_context(full_results, graph)

    # 6) Gemini LLM generates reasoning over KG
    final_answer = generate_answer(llm_model, user_query, final_context, column_context, mode=mode, picot=picot)
    print("\n📝 Final Answer (Knowledge Graph Synthesis):\n")
    print(final_answer)


    # 🧵 Step 6.5 - Extract variables from LLM answer
    variable_names = extract_variable_array_from_text(final_answer)
    print("\n🔢 Extracted Variable Names:", variable_names)

    # 🧠 Step 6.6 - Get all value labels for those variables from Neo4j
    value_dict = get_all_values_for_variables(graph, variable_names)
    print(value_dict)

    print("\n📊 Matched Values from Neo4j by Variable:")
    for var, values in value_dict.items():
        print(f"✔ {var}:")
        for v in values:
            print(f"   - {v}")

    # 7) Ask Gemini to turn explanation into a structured execution plan
    print("\n🧩 Creating Agentic Plan from Gemini...\n")

    plan = generate_plan(llm_model, final_answer, column_context, user_query, value_dict)
    print(plan)
    if not plan:
        print("❌ No plan generated. Skipping agentic execution.")
        return

    print("📋 Plan Steps:")
    for step in plan:
        print(f"  - {step['name']}: {step['description']}")
        print(f"    Instruction: {step['instruction']}")

    # 8) ReAct-style supervised execution
    final_response, react_log = execute_plan(
        initial_df=df,
        plan_steps=plan,
        user_query=user_query,
        llm_model=llm_model,
        max_retries=1,
        verbose=True
    )

    print("\n🤖 Final Synthesized Answer:\n")
    print(final_response)

    response = create_response_json(llm_model, final_response, user_query)

    print(f"FINAL ANSWER: {response}")
    return {"answer": response, "debug": final_response}
