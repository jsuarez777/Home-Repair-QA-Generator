#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qa_item import QAItem
from openai_client.openai_client import MyOpenAIClient


PROMPTS_ROOT = PROJECT_ROOT / "prompts_llm_judge"
MODEL = "gpt-4.1-nano"

DIMENSIONS = [
    "answer_completeness",
    "safety_specificity",
    "tool_realism",
    "scope_appropriateness",
    "context_clarity",
    "tip_usefulness",
]


def _select_prompt_version() -> Path:
    versions = sorted(
        (p for p in PROMPTS_ROOT.iterdir() if p.is_dir()),
        key=lambda p: int(p.name.lstrip("v")) if p.name.lstrip("v").isdigit() else 0,
    )
    if not versions:
        raise RuntimeError(f"No prompt version folders found in {PROMPTS_ROOT}")
    if len(versions) == 1:
        return versions[0]
    print("\nAvailable judge prompt versions:")
    for i, v in enumerate(versions, start=1):
        prompts = list(v.glob("*.prompt"))
        print(f"  {i}) {v.name}  [{len(prompts)} prompt(s)]")
    choice = input(f"Select prompt version (1-{len(versions)}): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(versions):
        return versions[int(choice) - 1]
    print(f"Invalid choice {choice!r}, defaulting to {versions[-1].name}.")
    return versions[-1]


def _load_mono_prompt(prompt_dir: Path) -> str:
    return (prompt_dir / "mono.prompt").read_text().strip()


# ---------------------------------------------------------------------------
# Per-item evaluation
# ---------------------------------------------------------------------------

def evaluate_qa_item(client: MyOpenAIClient, mono_prompt: str, trace_id: str, qa_item: QAItem) -> dict:
    item_json = qa_item.model_dump_json(indent=2)
    messages = [
        {"role": "system", "content": mono_prompt},
        {"role": "user", "content": item_json},
    ]
    response = client.query(input=messages)
    raw_text = response.output_text.strip()

    try:
        scores = json.loads(raw_text)
    except json.JSONDecodeError:
        # Strip markdown code fences if present
        cleaned = raw_text.strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
        scores = json.loads(cleaned)

    result = {"trace_id": trace_id}
    for dim in DIMENSIONS:
        result[dim] = int(scores.get(dim, 0))
    result["overall_pass"] = all(result[d] == 1 for d in DIMENSIONS)
    return result


# ---------------------------------------------------------------------------
# Training set evaluation
# ---------------------------------------------------------------------------

EVAL_JSONL = PROJECT_ROOT / "eval.jsonl"
_QA_ITEM_FIELDS = set(QAItem.model_fields.keys())


def evaluate_training_set(client: MyOpenAIClient, mono_prompt: str, eval_path: Path = EVAL_JSONL) -> list[dict]:
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
            result = evaluate_qa_item(client, mono_prompt, trace_id, qa_item)
            results.append(result)
            print(json.dumps(result, indent=2))

    print(f"\nEvaluated {len(results)} item(s) from {eval_path}.")
    _write_eval_results(results, eval_path.parent)
    return results


# ---------------------------------------------------------------------------
# Evaluation dispatch helpers
# ---------------------------------------------------------------------------

def _write_eval_results(results: list[dict], folder: Path) -> None:
    if not results:
        return
    out_path = folder / "QA_llm_eval.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results written to {out_path}")


def evaluate_qa_folder(client: MyOpenAIClient, mono_prompt: str, folder: Path) -> list[dict]:
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
        result = evaluate_qa_item(client, mono_prompt, trace_id, qa_item)
        results.append(result)
        print(json.dumps(result, indent=2))
    print(f"\nEvaluated {len(results)} item(s) from {folder}.")
    _write_eval_results(results, folder)
    return results


def evaluate_single_qa_file(client: MyOpenAIClient, mono_prompt: str, path: Path) -> list[dict]:
    print(f"Evaluating: {path.name}")
    try:
        qa_item = QAItem.model_validate_json(path.read_text())
    except Exception as e:
        print(f"  Parse error: {e}")
        return []
    result = evaluate_qa_item(client, mono_prompt, path.stem, qa_item)
    print(json.dumps(result, indent=2))
    _write_eval_results([result], path.parent)
    return [result]


def _resolve_target(client: MyOpenAIClient, mono_prompt: str, target: Path) -> list[dict]:
    if target.is_dir():
        return evaluate_qa_folder(client, mono_prompt, target)
    if target.suffix == ".jsonl":
        return evaluate_training_set(client, mono_prompt, target)
    if target.suffix == ".qa":
        return evaluate_single_qa_file(client, mono_prompt, target)
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
    prompt_dir = _select_prompt_version()
    mono_prompt = _load_mono_prompt(prompt_dir)
    print(f"Using judge prompts from: {prompt_dir.name}\n")

    if args.evaluate:
        target = Path(args.evaluate)
        if not target.exists():
            print(f"Error: path not found — {target}")
            return
        _resolve_target(client, mono_prompt, target)
        return

    # Interactive prompt
    qa_items_root = PROJECT_ROOT / "qa_items"
    versions = sorted(
        (p for p in qa_items_root.iterdir() if p.is_dir()),
        key=lambda p: int(p.name.lstrip("v")) if p.name.lstrip("v").isdigit() else 0,
    )

    print("What would you like to evaluate?")
    print("  1) Training dataset (eval.jsonl)")
    for i, folder in enumerate(versions, start=2):
        qa_count = len(list(folder.glob("*.qa")))
        human_eval = folder / "QA_human_eval.json"
        human_count = 0
        if human_eval.exists():
            try:
                human_count = len(json.load(human_eval.open()))
            except Exception:
                pass
        human_note = f", {human_count} human judge item(s)"
        print(f"  {i}) {folder.name}  [{qa_count} QA item(s){human_note}]")

    max_choice = len(versions) + 1
    choice = input(f"Enter 1-{max_choice}: ").strip()

    if choice == "1":
        evaluate_training_set(client, mono_prompt)
    elif choice.isdigit() and 2 <= int(choice) <= max_choice:
        evaluate_qa_folder(client, mono_prompt, versions[int(choice) - 2])
    else:
        print(f"Unknown choice: {choice!r}. Please enter a number between 1 and {max_choice}.")


if __name__ == "__main__":
    main()
