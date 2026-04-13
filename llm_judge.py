#!/usr/bin/env python3
import json
from enum import IntEnum
from pathlib import Path
from qa_item import QAItem
from openai_client.openai_client import MyOpenAIClient


JUDGE_PROMPT_VERSION = "v1"
JUDGE_PROMPT_DIR = Path(f"./prompts_llm_judge/{JUDGE_PROMPT_VERSION}")
QA_FOLDER = Path("./qa_items/v1")
MODEL = "gpt-4.1-nano"


class Result(IntEnum):
    FAIL = 0
    PASS = 1


def _load_prompt(name: str) -> str:
    path = JUDGE_PROMPT_DIR / f"{name}.prompt"
    return path.read_text().strip()


def _ask_judge(client: MyOpenAIClient, system_prompt: str, qa_item: QAItem) -> Result:
    """Send the QAItem JSON + system prompt to the LLM and parse PASS/FAIL from the response."""
    item_json = qa_item.model_dump_json(indent=2)
    input_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": item_json},
    ]
    response = client.query(input=input_messages)
    verdict = response.output_text.strip().upper()
    return Result.PASS if "PASS" in verdict else Result.FAIL


# ---------------------------------------------------------------------------
# Failure mode checks
# ---------------------------------------------------------------------------

def check_incomplete_answer(client: MyOpenAIClient, qa_item: QAItem) -> Result:
    return _ask_judge(client, _load_prompt("incomplete_answer"), qa_item)


def check_safety_violations(client: MyOpenAIClient, qa_item: QAItem) -> Result:
    return _ask_judge(client, _load_prompt("safety_violations"), qa_item)


def check_unrealistic_tools(client: MyOpenAIClient, qa_item: QAItem) -> Result:
    return _ask_judge(client, _load_prompt("unrealistic_tools"), qa_item)


def check_overcomplicated_solution(client: MyOpenAIClient, qa_item: QAItem) -> Result:
    return _ask_judge(client, _load_prompt("overcomplicated_solution"), qa_item)


def check_missing_context(client: MyOpenAIClient, qa_item: QAItem) -> Result:
    return _ask_judge(client, _load_prompt("missing_context"), qa_item)


def check_poor_quality_tips(client: MyOpenAIClient, qa_item: QAItem) -> Result:
    return _ask_judge(client, _load_prompt("poor_quality_tips"), qa_item)


# ---------------------------------------------------------------------------
# Quality dimension checks
# ---------------------------------------------------------------------------

def check_answer_coherence(client: MyOpenAIClient, qa_item: QAItem) -> Result:
    return _ask_judge(client, _load_prompt("answer_coherence"), qa_item)


def check_step_actionability(client: MyOpenAIClient, qa_item: QAItem) -> Result:
    return _ask_judge(client, _load_prompt("step_actionability"), qa_item)


def check_tool_realism(client: MyOpenAIClient, qa_item: QAItem) -> Result:
    return _ask_judge(client, _load_prompt("tool_realism"), qa_item)


def check_safety_specificity(client: MyOpenAIClient, qa_item: QAItem) -> Result:
    if len(qa_item.safety_info) < 80:
        return Result.FAIL
    return _ask_judge(client, _load_prompt("safety_specificity"), qa_item)


def check_tip_usefulness(client: MyOpenAIClient, qa_item: QAItem) -> Result:
    return _ask_judge(client, _load_prompt("tip_usefulness"), qa_item)


def check_problem_answer_alignment(client: MyOpenAIClient, qa_item: QAItem) -> Result:
    return _ask_judge(client, _load_prompt("problem_answer_alignment"), qa_item)


def check_appropriate_scope(client: MyOpenAIClient, qa_item: QAItem) -> Result:
    return _ask_judge(client, _load_prompt("appropriate_scope"), qa_item)


def check_category_accuracy(client: MyOpenAIClient, qa_item: QAItem) -> Result:
    return _ask_judge(client, _load_prompt("category_accuracy"), qa_item)


# ---------------------------------------------------------------------------
# Per-item evaluation
# ---------------------------------------------------------------------------

def evaluate_qa_item(client: MyOpenAIClient, trace_id: str, qa_item: QAItem) -> dict:
    incomplete_answer      = check_incomplete_answer(client, qa_item)
    safety_violations      = check_safety_violations(client, qa_item)
    unrealistic_tools      = check_unrealistic_tools(client, qa_item)
    overcomplicated        = check_overcomplicated_solution(client, qa_item)
    missing_context        = check_missing_context(client, qa_item)
    poor_quality_tips      = check_poor_quality_tips(client, qa_item)

    overall_failure = any(
        score == Result.FAIL
        for score in [
            incomplete_answer, safety_violations, unrealistic_tools,
            overcomplicated, missing_context, poor_quality_tips,
        ]
    )

    answer_coherence       = check_answer_coherence(client, qa_item)
    step_actionability     = check_step_actionability(client, qa_item)
    tool_realism           = check_tool_realism(client, qa_item)
    safety_specificity     = check_safety_specificity(client, qa_item)
    tip_usefulness         = check_tip_usefulness(client, qa_item)
    problem_answer_align   = check_problem_answer_alignment(client, qa_item)
    appropriate_scope      = check_appropriate_scope(client, qa_item)
    category_accuracy      = check_category_accuracy(client, qa_item)

    quality_scores = {
        "answer_coherence":        int(answer_coherence),
        "step_actionability":      int(step_actionability),
        "tool_realism":            int(tool_realism),
        "safety_specificity":      int(safety_specificity),
        "tip_usefulness":          int(tip_usefulness),
        "problem_answer_alignment": int(problem_answer_align),
        "appropriate_scope":       int(appropriate_scope),
        "category_accuracy":       int(category_accuracy),
    }

    quality_pass = all(v == Result.PASS for v in quality_scores.values())

    return {
        "trace_id":               trace_id,
        "incomplete_answer":      int(incomplete_answer),
        "safety_violations":      int(safety_violations),
        "unrealistic_tools":      int(unrealistic_tools),
        "overcomplicated_solution": int(overcomplicated),
        "missing_context":        int(missing_context),
        "poor_quality_tips":      int(poor_quality_tips),
        "overall_failure":        overall_failure,
        "quality_scores":         quality_scores,
        "quality_pass":           quality_pass,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    client = MyOpenAIClient(model=MODEL)

    qa_files = sorted(QA_FOLDER.glob("*.qa"))
    if not qa_files:
        print("No .qa files found.")
        return

    results = []
    for qa_file in qa_files:
        trace_id = qa_file.stem
        print(f"Evaluating: {qa_file.name}")
        content = qa_file.read_text()
        try:
            qa_item = QAItem.model_validate_json(content)
        except Exception as e:
            print(f"  Parse error: {e}")
            continue

        result = evaluate_qa_item(client, trace_id, qa_item)
        results.append(result)
        print(json.dumps(result, indent=2))

    print(f"\nEvaluated {len(results)} item(s).")


if __name__ == "__main__":
    main()
