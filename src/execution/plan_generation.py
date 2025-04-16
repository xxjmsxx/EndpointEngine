import re
import json
import ast

def extract_clean_json_array(text):
    """Attempt to cleanly extract a JSON array from LLM output"""
    match = re.search(r"\[\s*{.*}\s*\]", text, re.DOTALL)
    if match:
        json_block = match.group(0)
        try:
            return json.loads(json_block)
        except Exception:
            try:
                return ast.literal_eval(json_block)
            except Exception as e:
                print("⚠️ Could not parse even with literal_eval:", e)
    return []

def generate_plan(llm_model, answer_text, column_context, question, value_dict):
    """Generate a structured execution plan from a reasoning answer"""
    plan_prompt = f"""
You are an assistant generating a structured plan to analyze biomedical patient data in a pandas DataFrame.

Below is:
1. A reasoning analysis of how to answer the user's question.
2. The actual column names available in the dataset.

--- Reasoning Analysis ---
{answer_text}

--- DataFrame Column Context (Use these exactly) ---
{column_context}

--- User Query  ---
{question}

Your task:
Please output a structured JSON array of steps to be executed on a DataFrame.
Each step must include:
- "name": short identifier
- "description": what the step does
- "instruction": natural language instruction describing the query/filter logic
- The instruction should contain a list of the variables that are needed to perform said step (e.g. [operativedeath, bmi, bilobectomylobectomyqualifier])
- The instructions variables should ALL match the spelling from the --- DataFrame Column Context (Use these exactly) ---
- Each step should only talk about that step and not the previous or following sten UNLESS the information is relevant
- Be explicit in your filters and use the key(number) for the value (e.g., "Cardiaccomorbidity1 == 1" NOT "Cardiaccomorbidity1 == 1 - Coronary Artery Disease")
- The above point applies to ALL COLUMN - VALUE pairs. it will never be "1 - valuename" it will always just be a number that is the key
- Compare and match the KG info against the column names above
- The aim of the steps should be resolving the user query
- If the instruction requires comparing cohorts (e.g., VATS vs. thoracotomy), assign them to named DataFrames like `df_vats` or `df_thoracotomy`.
- I am using dask so the final step needs to be using compute so you actually get numbers back. Please make the steps lead up to a final compute step.


⚠️ Do not return any explanation. Respond ONLY with the JSON array.
"""
    response = llm_model.generate_content(plan_prompt).text.strip()
    return extract_clean_json_array(response)
