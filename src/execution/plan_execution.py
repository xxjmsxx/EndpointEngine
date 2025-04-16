import pandas as pd
import numpy as np
import json

def execute_plan(initial_df, plan_steps, user_query, llm_model, max_retries=2, verbose=True):
    state = {"df": initial_df.copy()}
    completed_steps, failed_steps = [], []
    react_log = {}

    for i, step in enumerate(plan_steps):
        if i > 0 and plan_steps[i - 1]["name"] in failed_steps:
            if verbose:
                print(f"⚠️ Skipping step {i + 1} ({step['name']}) due to previous failure.")
            continue
        success = _execute_step(step, state, user_query, llm_model, completed_steps, failed_steps, react_log, max_retries, verbose)
        if not success and verbose:
            print(f"⚠️ Halting execution at step '{step['name']}' due to repeated failure.")

    final_check = _run_final_check(llm_model, user_query, completed_steps, failed_steps, react_log)
    if verbose:
        print(final_check)

    final_response = _synthesize_final_response(llm_model, user_query, completed_steps, failed_steps, react_log)
    return final_response, react_log

def _execute_step(step, state, user_query, llm_model, completed_steps, failed_steps, react_log, max_retries, verbose, attempt=1):
    step_name, instruction = step["name"], step["instruction"]
    _print_step_header(step_name, len(completed_steps) + 1, attempt if attempt > 1 else None, verbose)
    state_description = _get_state_description(state)

    thought = llm_model.generate_content(_thought_prompt(user_query, step_name, instruction)).text.strip()
    code = llm_model.generate_content(_code_prompt(user_query, thought, instruction, state_description)).text.strip()
    code = code.replace("```python", "").replace("```", "").strip()

    try:
        result = _run_code(code, state)

        reflection = llm_model.generate_content(
            _reflection_prompt(user_query, step_name, instruction, code, "execution complete")).text.strip()
        _log_step(react_log, step_name, thought, instruction, code, str(result), reflection, verbose)

        completed_steps.append(step_name)
        return True

    except Exception as e:
        if verbose:
            print(f"❌ Error in step '{step_name}': {str(e)}")

        if attempt < max_retries:
            recovery_code = llm_model.generate_content(
                _recovery_prompt(str(e), code, state_description, instruction)).text.strip()
            recovery_code = recovery_code.replace("```python", "").replace("```", "").strip()
            try:
                _run_code(recovery_code, state)

                reflection = llm_model.generate_content(
                    _reflection_prompt(user_query, step_name, instruction, recovery_code, "execution complete")).text.strip()
                _log_step(react_log, step_name, thought, instruction, recovery_code, "(not captured)", reflection, verbose)
                completed_steps.append(step_name)
                return True

            except Exception as e2:
                if verbose:
                    print(f"❌ Recovery also failed: {str(e2)}")
                reflection = f"Step failed with error: {str(e)}. Recovery attempt also failed: {str(e2)}"
                _log_step(react_log, step_name, thought, instruction, code, f"ERROR: {str(e)}", reflection, verbose)
                failed_steps.append(step_name)
                return False
        else:
            reflection = f"Step failed with error: {str(e)}. Maximum retries exceeded."
            _log_step(react_log, step_name, thought, instruction, code, f"ERROR: {str(e)}", reflection, verbose)
            failed_steps.append(step_name)
            return False

def _run_code(code, state):
    local_scope = dict(state)
    exec(code, {"pd": pd, "np": np}, local_scope)
    for var_name, var_value in local_scope.items():
        if var_name != "__builtins__" and var_name not in state:
            state[var_name] = var_value
    return local_scope.get("result")

def _print_step_header(step_name, step_num, attempt, verbose):
    if verbose:
        print(f"\n--- Step {step_num}: {step_name}{f' (Attempt {attempt})' if attempt else ''} ---")

def _log_step(log, step_name, thought, instruction, code, result, reflection, verbose):
    log[step_name] = {
        "thought": thought,
        "instruction": instruction,
        "code": code,
        "result": result,
        "reflection": reflection
    }
    if verbose:
        print(f"\U0001f4ad Thought: {thought}\n🧾 Instruction: {instruction}\n🧠 Code:\n{code}")
        print(f"📈 Result: {result}")
        print(f"🔎 Reflection: {reflection}")

def _get_state_description(state):
    descriptions = []
    for name, value in state.items():
        if isinstance(value, pd.DataFrame):
            descriptions.append(f"- '{name}': DataFrame with {value.shape[0]} rows, {value.shape[1]} columns")
        elif isinstance(value, (int, float)):
            descriptions.append(f"- '{name}': {value}")
        elif isinstance(value, (list, dict)):
            descriptions.append(f"- '{name}': {type(value).__name__} with {len(value)} elements")
        else:
            descriptions.append(f"- '{name}': {type(value).__name__}")
    return "\n".join(descriptions)

def _thought_prompt(user_query, step_name, instruction):
    return f"""
You are a biomedical data analyst. Review this instruction to understand its purpose:

USER QUERY: {user_query}
STEP: {step_name}
INSTRUCTION: {instruction}

Write a concise paragraph explaining what this step is trying to achieve in the context of the user's query.
"""

def _code_prompt(user_query, thought, instruction, state_description):
    return f"""
You are a Python expert who writes correct pandas code to analyze biomedical data.

USER QUERY: {user_query}
STEP PURPOSE: {thought}
INSTRUCTION: {instruction}

CURRENT STATE:
{state_description}

Write Python code that:
1. Uses the variables from the current state - no need to redefine them
2. Performs the analysis described in the instruction
3. Result should be in a variable called 'result'
4. If creating a new DataFrame named in the instruction (e.g., "Create a DataFrame called df_filtered"), define it as indicated and also assign it to a variable with that exact name
5. Follow ONLY what is in the instructions. No extra steps as it might mess up following steps.

You can import things only if you are 100% certain they exist.
Do not add print statements or comments.
Focus only on what the instruction asks for.

Return ONLY executable Python code with no markdown formatting or explanation.
"""

def _recovery_prompt(error_msg, code, state_description, instruction):
    return f"""
You are a Python expert fixing code that failed with this error: {error_msg}

The failed code was:
{code}

Current state:
{state_description}

Current instruction:
{instruction}

Only use the variables / values as per the current instruction.
If it is due to an import error remove what ever is being imported and try another way of doing it.
Fix ONLY the code to address the error while maintaining the same goal.
Do NOT add any explanation or comments.
Return ONLY the corrected code.
"""

def _reflection_prompt(user_query, step_name, instruction, code, result):
    return f"""
You are a data scientist reviewing code execution results.

USER QUERY: {user_query}
STEP: {step_name}
INSTRUCTION: {instruction}
CODE EXECUTED:
{code}

RESULT SUMMARY:
{str(result)[:2000] + '...' if len(str(result)) > 2000 else str(result)}

Provide a reflection covering:
1. Did the code accomplish what was intended?
2. Are there any issues or unexpected results?
3. What logical next steps should follow?

Be concise but thorough.
"""

def _run_final_check(llm_model, user_query, completed_steps, failed_steps, react_log):
    prompt = f"""
You are a biomedical data analyst making a list of variables that need a dictionary before the final analysis steps.

USER QUERY: {user_query}

EXECUTION SUMMARY:
- Completed steps: {', '.join(completed_steps) if completed_steps else 'None'}
- Failed steps: {', '.join(failed_steps) if failed_steps else 'None'}

DETAILED EXECUTION LOG:
{json.dumps(react_log, indent=2)}

Provide a comprehensive summary that includes:
1. Any variables that are needed to provide a useful answer. Make an array of missing variables names. (Example: [operativedeath, bmi, complication1])

ONLY provide a list of variables where further clarification is needed.
ONLY return an array of variables.
DO not explain anything or add extra text.
"""
    return llm_model.generate_content(prompt).text.strip()

def _synthesize_final_response(llm_model, user_query, completed_steps, failed_steps, react_log):
    prompt = f"""
You are a biomedical data analyst synthesizing results from multiple analysis steps.

USER QUERY: {user_query}

EXECUTION SUMMARY:
- Completed steps: {', '.join(completed_steps) if completed_steps else 'None'}
- Failed steps: {', '.join(failed_steps) if failed_steps else 'None'}

DETAILED EXECUTION LOG:
{json.dumps(react_log, indent=2)}

Provide a comprehensive summary that includes:
1. What steps were performed and their purpose
2. Key findings and results discovered
3. Any limitations or errors encountered
4. A final answer to the user's original query based on available results

Be thorough but focus on answering the user's original question.
"""
    return llm_model.generate_content(prompt).text.strip()
