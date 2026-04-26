#!/usr/bin/env python3
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qa_item import QAItem
from openai_client.openai_client import MyOpenAIClient


PROMPTS_ROOT = PROJECT_ROOT / "prompts_llm_judge"
DEFAULT_MODEL = "gpt-4.1-nano"
MAX_PARALLEL = 50
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0

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


def _extract_trace_id(stem: str) -> str:
    m = re.match(r"QA[^_]*", stem)
    return m.group() if m else stem


def _load_mono_prompt(prompt_dir: Path) -> str:
    return (prompt_dir / "mono.prompt").read_text().strip()


def _next_eval_version(folder: Path) -> str:
    nums = []
    for f in folder.glob("QA_llm_eval_v*.json"):
        m = re.search(r"QA_llm_eval_v(\d+)\.json", f.name)
        if m:
            nums.append(int(m.group(1)))
    return f"v{max(nums) + 1}" if nums else "v1"


# ---------------------------------------------------------------------------
# Per-item evaluation
# ---------------------------------------------------------------------------

def evaluate_qa_item(client: MyOpenAIClient, mono_prompt: str, trace_id: str, qa_item: QAItem) -> dict:
    from openai import RateLimitError
    item_json = qa_item.model_dump_json(indent=2)
    messages = [
        {"role": "system", "content": mono_prompt},
        {"role": "user", "content": item_json},
    ]
    delay = RETRY_BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.query(input=messages)
            raw_text = response.output_text.strip()

            try:
                scores = json.loads(raw_text)
            except json.JSONDecodeError:
                cleaned = raw_text.strip("`").strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                scores = json.loads(cleaned)

            missing = [d for d in DIMENSIONS if d not in scores]
            if missing:
                raise ValueError(f"LLM response missing dimensions: {missing}")

            llm_trace_id = scores.get("trace_id")
            if llm_trace_id != "[NA]":
                print(f"  [warning] {trace_id}: expected trace_id '[NA]' from LLM, got {llm_trace_id!r}")

            result = {"trace_id": trace_id}
            for dim in DIMENSIONS:
                result[dim] = int(scores[dim])
            result["overall_pass"] = all(result[d] == 1 for d in DIMENSIONS)
            return result

        except RateLimitError:
            if attempt == MAX_RETRIES:
                raise
            print(f"  [rate limit] {trace_id}: attempt {attempt}/{MAX_RETRIES}, retrying in {delay:.1f}s...")
            time.sleep(delay)
            delay *= 2
        except (json.JSONDecodeError, ValueError) as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"  [parse error] {trace_id}: attempt {attempt}/{MAX_RETRIES}: {e}")
            time.sleep(delay)
            delay *= 2


# ---------------------------------------------------------------------------
# Training set evaluation
# ---------------------------------------------------------------------------

EVAL_JSONL = PROJECT_ROOT / "eval.jsonl"
_QA_ITEM_FIELDS = set(QAItem.model_fields.keys())


def evaluate_training_set(
    client: MyOpenAIClient, mono_prompt: str, prompt_dir: Path, eval_path: Path = EVAL_JSONL,
) -> list[dict]:
    items: list[tuple[str, QAItem]] = []
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
            trace_id = _extract_trace_id(raw.get("id", f"line_{line_num}"))
            qa_fields = {k: v for k, v in raw.items() if k in _QA_ITEM_FIELDS}
            try:
                items.append((trace_id, QAItem.model_validate(qa_fields)))
            except Exception as e:
                print(f"  {trace_id}: validation error — {e}")

    results = _evaluate_parallel(client, mono_prompt, items)
    print(f"\nEvaluated {len(results)} item(s) from {eval_path}.")
    _write_eval_results(results, eval_path.parent, prompt_dir, mono_prompt, client.model)
    return results


# ---------------------------------------------------------------------------
# Evaluation dispatch helpers
# ---------------------------------------------------------------------------

def _evaluate_parallel(client: MyOpenAIClient, mono_prompt: str, items: list[tuple[str, QAItem]]) -> list[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        pending = {
            executor.submit(evaluate_qa_item, client, mono_prompt, trace_id, qa_item): trace_id
            for trace_id, qa_item in items
        }
        while pending:
            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                trace_id = pending.pop(future)
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Evaluated: {trace_id}")
                    print(json.dumps(result, indent=2))
                except Exception as e:
                    print(f"  Error evaluating {trace_id}: {e}")
    return results


def _write_eval_results(results: list[dict], folder: Path, prompt_dir: Path, mono_prompt: str, model: str) -> None:
    if not results:
        return
    eval_version = _next_eval_version(folder)
    out_path = folder / f"QA_llm_eval_{eval_version}.json"
    payload = {
        "eval_version": eval_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "judge_prompt_version": prompt_dir.name,
        "judge_prompt": mono_prompt,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Results written to {out_path}")


def evaluate_qa_folder(client: MyOpenAIClient, mono_prompt: str, prompt_dir: Path, folder: Path) -> list[dict]:
    qa_files = sorted(folder.glob("*.qa"))
    if not qa_files:
        print(f"No .qa files found in {folder}.")
        return []
    items: list[tuple[str, QAItem]] = []
    for qa_file in qa_files:
        try:
            items.append((_extract_trace_id(qa_file.stem), QAItem.model_validate_json(qa_file.read_text())))
        except Exception as e:
            print(f"  Parse error {qa_file.name}: {e}")
    results = _evaluate_parallel(client, mono_prompt, items)
    print(f"\nEvaluated {len(results)} item(s) from {folder}.")
    _write_eval_results(results, folder, prompt_dir, mono_prompt, client.model)
    return results


def evaluate_single_qa_file(client: MyOpenAIClient, mono_prompt: str, prompt_dir: Path, path: Path) -> list[dict]:
    print(f"Evaluating: {path.name}")
    try:
        qa_item = QAItem.model_validate_json(path.read_text())
    except Exception as e:
        print(f"  Parse error: {e}")
        return []
    result = evaluate_qa_item(client, mono_prompt, _extract_trace_id(path.stem), qa_item)
    print(json.dumps(result, indent=2))
    _write_eval_results([result], path.parent, prompt_dir, mono_prompt, client.model)
    return [result]


def _resolve_target(client: MyOpenAIClient, mono_prompt: str, prompt_dir: Path, target: Path) -> list[dict]:
    if target.is_dir():
        return evaluate_qa_folder(client, mono_prompt, prompt_dir, target)
    if target.suffix == ".jsonl":
        return evaluate_training_set(client, mono_prompt, prompt_dir, target)
    if target.suffix == ".qa":
        return evaluate_single_qa_file(client, mono_prompt, prompt_dir, target)
    print(f"Unsupported file type: {target.suffix}. Expected a directory, .jsonl, or .qa file.")
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _select_model() -> str:
    catalog = MyOpenAIClient.available_models()
    models = list(catalog.keys())
    default_idx = models.index(DEFAULT_MODEL) + 1 if DEFAULT_MODEL in models else 1

    print("\nAvailable models:")
    print(f"  {'#':<4} {'Model':<30} {'Input/1M':>10} {'Cached/1M':>10} {'Output/1M':>10}")
    print(f"  {'─' * 68}")
    for i, m in enumerate(models, start=1):
        p = catalog[m]
        cached = f"${p['cached_input']:.4f}" if p["cached_input"] is not None else "—"
        marker = " *" if m == DEFAULT_MODEL else ""
        print(f"  {i:<4} {m + marker:<30} ${p['input']:>9.4f} {cached:>10} ${p['output']:>9.4f}")
    print(f"  * = default")

    choice = input(f"\nSelect model (1-{len(models)}) [{default_idx}]: ").strip()
    if choice == "":
        return models[default_idx - 1]
    if choice.isdigit() and 1 <= int(choice) <= len(models):
        return models[int(choice) - 1]
    print(f"Invalid choice {choice!r}, using default ({DEFAULT_MODEL}).")
    return DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]


def main():
    parser = argparse.ArgumentParser(description="LLM judge for DIY repair QA items.")
    parser.add_argument(
        "--evaluate",
        metavar="OBJECT",
        help="Path to evaluate: a directory of .qa files, a .jsonl file, or a single .qa file.",
    )
    args = parser.parse_args()

    model = _select_model()
    client = MyOpenAIClient(model=model)
    prompt_dir = _select_prompt_version()
    mono_prompt = _load_mono_prompt(prompt_dir)
    print(f"Using judge prompts from: {prompt_dir.name}\n")

    if args.evaluate:
        target = Path(args.evaluate)
        if not target.exists():
            print(f"Error: path not found — {target}")
            return
        _resolve_target(client, mono_prompt, prompt_dir, target)
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
        llm_eval_count = len(list(folder.glob("QA_llm_eval_v*.json")))
        llm_note = f", {llm_eval_count} llm eval version(s)" if llm_eval_count else ""
        human_note = f", {human_count} human judge item(s)" if human_count else ""
        print(f"  {i}) {folder.name}  [{qa_count} QA item(s){human_note}{llm_note}]")

    max_choice = len(versions) + 1
    choice = input(f"Enter 1-{max_choice}: ").strip()

    if choice == "1":
        evaluate_training_set(client, mono_prompt, prompt_dir)
    elif choice.isdigit() and 2 <= int(choice) <= max_choice:
        evaluate_qa_folder(client, mono_prompt, prompt_dir, versions[int(choice) - 2])
    else:
        print(f"Unknown choice: {choice!r}. Please enter a number between 1 and {max_choice}.")


if __name__ == "__main__":
    main()
