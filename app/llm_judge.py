#!/usr/bin/env python3
import argparse
import json
import sys
from enum import IntEnum
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qa_item import QAItem
from openai_client.openai_client import MyOpenAIClient


JUDGE_PROMPT_VERSION = "v1"
JUDGE_PROMPT_DIR = PROJECT_ROOT / f"prompts_llm_judge/{JUDGE_PROMPT_VERSION}"
QA_FOLDER = PROJECT_ROOT / "qa_items/v1"
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
# Training set evaluation
# ---------------------------------------------------------------------------

EVAL_JSONL = PROJECT_ROOT / "eval.jsonl"
_QA_ITEM_FIELDS = set(QAItem.model_fields.keys())


def evaluate_training_set(client: MyOpenAIClient, eval_path: Path = EVAL_JSONL) -> list[dict]:
    results = []
    with eval_path.open() as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Line {line_num}: JSON parse error — {e}")
                continue

            trace_id = raw.get("id", f"line_{line_num}")
            qa_fields = {k: v for k, v in raw.items() if k in _QA_ITEM_FIELDS}
            try:
                qa_item = QAItem.model_validate(qa_fields)
            except Exception as e:
                print(f"  {trace_id}: validation error — {e}")
                continue

            print(f"Evaluating: {trace_id}")
            result = evaluate_qa_item(client, trace_id, qa_item)
            results.append(result)
            print(json.dumps(result, indent=2))

    print(f"\nEvaluated {len(results)} item(s) from {eval_path}.")
    return results


# ---------------------------------------------------------------------------
# Evaluation dispatch helpers
# ---------------------------------------------------------------------------

def evaluate_qa_folder(client: MyOpenAIClient, folder: Path) -> list[dict]:
    qa_files = sorted(folder.glob("*.qa"))
    if not qa_files:
        print(f"No .qa files found in {folder}.")
        return []
    results = []
    for qa_file in qa_files:
        trace_id = qa_file.stem
        print(f"Evaluating: {qa_file.name}")
        try:
            qa_item = QAItem.model_validate_json(qa_file.read_text())
        except Exception as e:
            print(f"  Parse error: {e}")
            continue
        result = evaluate_qa_item(client, trace_id, qa_item)
        results.append(result)
        print(json.dumps(result, indent=2))
    print(f"\nEvaluated {len(results)} item(s) from {folder}.")
    return results


def evaluate_single_qa_file(client: MyOpenAIClient, path: Path) -> list[dict]:
    print(f"Evaluating: {path.name}")
    try:
        qa_item = QAItem.model_validate_json(path.read_text())
    except Exception as e:
        print(f"  Parse error: {e}")
        return []
    result = evaluate_qa_item(client, path.stem, qa_item)
    print(json.dumps(result, indent=2))
    return [result]


def _resolve_target(client: MyOpenAIClient, target: Path) -> list[dict]:
    if target.is_dir():
        return evaluate_qa_folder(client, target)
    if target.suffix == ".jsonl":
        return evaluate_training_set(client, target)
    if target.suffix == ".qa":
        return evaluate_single_qa_file(client, target)
    print(f"Unsupported file type: {target.suffix}. Expected a directory, .jsonl, or .qa file.")
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM judge for DIY repair QA items.")
    parser.add_argument(
        "--evaluate",
        metavar="OBJECT",
        help="Path to evaluate: a directory of .qa files, a .jsonl file, or a single .qa file.",
    )
    args = parser.parse_args()

    client = MyOpenAIClient(model=MODEL)

    if args.evaluate:
        target = Path(args.evaluate)
        if not target.exists():
            print(f"Error: path not found — {target}")
            return
        _resolve_target(client, target)
        return

    # Interactive prompt
    print("What would you like to evaluate?")
    print("  1) Training dataset (eval.jsonl)")
    print(f"  2) Generated QA items ({QA_FOLDER})")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        evaluate_training_set(client)
    elif choice == "2":
        evaluate_qa_folder(client, QA_FOLDER)
    else:
        print(f"Unknown choice: {choice!r}. Please enter 1 or 2.")


if __name__ == "__main__":
    main()
