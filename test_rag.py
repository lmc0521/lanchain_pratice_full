from langchain_community.llms.ollama import Ollama
from CRAG_module import query_crag
EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

def test_monopoly_rules():
    assert query_and_validate(
        question={"question":"How much total money does a player start with in Monopoly? (Answer with the number only)","steps":[]},
        expected_response="$1500",
    )
def test_agent_knowledges():
    assert query_and_validate(
        question={"question":"What is an AI agent?","steps":[]},
        expected_response="agents are a system with complex reasoning capabilities, memory, and the means to execute tasks.",
    )

def query_and_validate(question: dict, expected_response: str):
    response_text = query_crag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="llama3")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )