#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_LOGS_DIR = PROJECT_ROOT / "logs"
_LOGS_DIR.mkdir(exist_ok=True)
_log_file = _LOGS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_llm_judge.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(_log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)
if os.getenv("LOG_HTTP") != "1":
    logging.getLogger("httpx").setLevel(logging.WARNING)
log.info(f"Logging to {_log_file}")

from app.qa_item import QAItem
from openai_client.openai_client import MyOpenAIClient


PROMPTS_ROOT = PROJECT_ROOT / "prompts_llm_judge"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_MAX_PARALLEL = 20
MAX_RETRIES = 6
RETRY_DELAYS = [2, 8, 16, 32, 64, 100]  # seconds for each retry attempt

# Dynamic rate limit throttling
_rate_limit_lock = threading.Lock()
_dynamic_max_workers = None  # Will be set if we hit a 429 error
_rate_limit_hit = False  # Flag to pause new submissions until all existing threads complete

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
    log.info("\nAvailable judge prompt versions:")
    for i, v in enumerate(versions, start=1):
        prompts = list(v.glob("*.prompt"))
        marker = " *" if i == len(versions) else ""
        log.info(f"  {i}) {v.name}  [{len(prompts)} prompt(s)]{marker}")
    log.info("  * = default")
    choice = input(f"Select prompt version (1-{len(versions)}) [{len(versions)}]: ").strip()
    if choice == "":
        return versions[-1]
    if choice.isdigit() and 1 <= int(choice) <= len(versions):
        return versions[int(choice) - 1]
    log.info(f"Invalid choice {choice!r}, defaulting to {versions[-1].name}.")
    return versions[-1]


def _extract_trace_id(stem: str) -> str:
    m = re.match(r"QA[^_]*", stem)
    return m.group() if m else stem


def _load_mono_prompt(prompt_dir: Path) -> str:
    return (prompt_dir / "mono.prompt").read_text().strip()


def _next_eval_version(folder: Path) -> str:
    return datetime.now().strftime("%y%m%d_%H%M%S")


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

            if isinstance(scores, list):
                if len(scores) == 0:
                    raise ValueError("LLM response is empty list")
                if len(scores) > 1:
                    raise ValueError(f"LLM response contains {len(scores)} objects, expected exactly 1")
                scores = scores[0]

            missing = [d for d in DIMENSIONS if d not in scores]
            if missing:
                raise ValueError(f"LLM response missing dimensions: {missing}")

            llm_trace_id = scores.get("trace_id")
            if llm_trace_id != "[NA]":
                log.warning(f"  [warning] {trace_id}: expected trace_id '[NA]' from LLM, got {llm_trace_id!r}")

            result = {"trace_id": trace_id}
            for dim in DIMENSIONS:
                result[dim] = int(scores[dim])
            result["overall_pass"] = all(result[d] == 1 for d in DIMENSIONS)
            return result

        except RateLimitError as e:
            if attempt == MAX_RETRIES:
                raise
            # Try to extract server-recommended wait time and token limits from response body
            retry_delay = RETRY_DELAYS[attempt - 1]  # attempt-1 because attempt is 1-indexed
            rate_limit_info = ""
            if hasattr(e, 'response') and e.response is not None:
                if hasattr(e.response, 'text'):
                    try:
                        body = json.loads(e.response.text)
                        if 'error' in body and 'message' in body['error']:
                            msg = body['error']['message']
                            # Extract "Please try again in X.XXXs" from the message
                            match = re.search(r'Please try again in ([\d.]+)s', msg)
                            if match:
                                server_wait_time = float(match.group(1))
                                rate_limit_info = f" (server says: {server_wait_time:.2f}s)"
                            # Extract token limits to throttle parallelism
                            limit_match = re.search(r'Limit (\d+)', msg)
                            requested_match = re.search(r'Requested (\d+)', msg)
                            if limit_match and requested_match:
                                limit = int(limit_match.group(1))
                                requested = int(requested_match.group(1))
                                calculated_max = max(1, (limit // requested) // 2)  # half the calculated ratio
                                with _rate_limit_lock:
                                    global _dynamic_max_workers, _rate_limit_hit
                                    _dynamic_max_workers = calculated_max
                                    _rate_limit_hit = True
                                log.warning(f"  [rate limit] {trace_id}: Pausing new submissions until all existing threads complete. Will resume at {calculated_max} workers (Limit: {limit}, Requested: {requested})")
                    except (json.JSONDecodeError, ValueError, AttributeError):
                        pass

            # Check if wait time is excessive (>180 seconds)
            if retry_delay > 180:
                log.error(f"  [ERROR] {trace_id}: Excessive rate limit wait time! Waiting {retry_delay}s ({int(retry_delay/60)}m {int(retry_delay%60)}s). This indicates severe API rate limiting.")

            log.warning(f"  [rate limit] {trace_id}: attempt {attempt}/{MAX_RETRIES}, retrying in {retry_delay}s...{rate_limit_info}")
            time.sleep(retry_delay)
        except (json.JSONDecodeError, ValueError) as e:
            if attempt == MAX_RETRIES:
                raise
            retry_delay = RETRY_DELAYS[attempt - 1]
            log.warning(f"  [parse error] {trace_id}: attempt {attempt}/{MAX_RETRIES}: {e}")
            log.warning(f"    Raw response received: {raw_text[:500]}")
            time.sleep(retry_delay)


# ---------------------------------------------------------------------------
# Training set evaluation
# ---------------------------------------------------------------------------

EVAL_JSONL = PROJECT_ROOT / "eval.jsonl"
_QA_ITEM_FIELDS = set(QAItem.model_fields.keys())


def evaluate_training_set(
    client: MyOpenAIClient, mono_prompt: str, prompt_dir: Path, eval_path: Path = EVAL_JSONL, max_workers: int = DEFAULT_MAX_PARALLEL, limit: int = None,
) -> list[dict]:
    items: list[tuple[str, QAItem]] = []
    with eval_path.open() as f:
        for line_num, line in enumerate(f, start=1):
            if limit and line_num > limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"  Line {line_num}: JSON parse error — {e}")
                continue
            trace_id = _extract_trace_id(raw.get("id", f"line_{line_num}"))
            qa_fields = {k: v for k, v in raw.items() if k in _QA_ITEM_FIELDS}
            try:
                items.append((trace_id, QAItem.model_validate(qa_fields)))
            except Exception as e:
                log.warning(f"  {trace_id}: validation error — {e}")

    results = _evaluate_parallel(client, mono_prompt, prompt_dir.name, items, max_workers=max_workers)
    log.info(f"\nEvaluated {len(results)} item(s) from {eval_path}.")
    source_path = str(eval_path.relative_to(PROJECT_ROOT))
    _write_eval_results(results, eval_path.parent, prompt_dir, mono_prompt, client.model, source=source_path)
    return results


# ---------------------------------------------------------------------------
# Evaluation dispatch helpers
# ---------------------------------------------------------------------------

def _log_result(result: dict, prompt_version: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    dims = "  ".join(f"{d}={'PASS' if result[d] == 1 else 'FAIL'}" for d in DIMENSIONS)
    overall = "PASS" if result["overall_pass"] else "FAIL"
    log.info(f"ts={ts} trace_id={result['trace_id']} prompt_version={prompt_version} | {dims} | overall={overall}")


def _evaluate_parallel(client: MyOpenAIClient, mono_prompt: str, prompt_version: str, items: list[tuple[str, QAItem]], max_workers: int = DEFAULT_MAX_PARALLEL) -> list[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit initial batch of tasks
        pending = {}
        items_queue = list(items)

        # Submit up to max_workers tasks initially
        while len(pending) < max_workers and items_queue:
            trace_id, qa_item = items_queue.pop(0)
            future = executor.submit(evaluate_qa_item, client, mono_prompt, trace_id, qa_item)
            pending[future] = trace_id

        # Process completed tasks and submit new ones
        while pending:
            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                trace_id = pending.pop(future)
                try:
                    result = future.result()
                    results.append(result)
                    _log_result(result, prompt_version)
                except Exception as e:
                    log.warning(f"  Error evaluating {trace_id}: {e}")

            # Check if we should throttle based on dynamic rate limit
            with _rate_limit_lock:
                rate_limit_hit = _rate_limit_hit
                effective_max = _dynamic_max_workers if _dynamic_max_workers is not None else max_workers

            # If rate limit was hit, pause new submissions until all existing threads complete
            if rate_limit_hit:
                if not pending and items_queue:
                    # All threads completed, resume submissions at reduced rate
                    log.info(f"Resuming with {effective_max} parallel workers")
                    with _rate_limit_lock:
                        globals()['_rate_limit_hit'] = False
                    # Submit new tasks at the reduced rate
                    while len(pending) < effective_max and items_queue:
                        trace_id, qa_item = items_queue.pop(0)
                        future = executor.submit(evaluate_qa_item, client, mono_prompt, trace_id, qa_item)
                        pending[future] = trace_id
            else:
                # Normal operation: submit new tasks respecting the effective max workers limit
                while len(pending) < effective_max and items_queue:
                    trace_id, qa_item = items_queue.pop(0)
                    future = executor.submit(evaluate_qa_item, client, mono_prompt, trace_id, qa_item)
                    pending[future] = trace_id

    return results


def _write_eval_results(results: list[dict], folder: Path, prompt_dir: Path, mono_prompt: str, model: str, source: str = None) -> None:
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
    if source:
        payload["evaluated_source"] = source
    out_path.write_text(json.dumps(payload, indent=2))
    log.info(f"Results written to {out_path}")


def evaluate_qa_folder(client: MyOpenAIClient, mono_prompt: str, prompt_dir: Path, folder: Path, max_workers: int = DEFAULT_MAX_PARALLEL) -> list[dict]:
    qa_files = sorted(folder.glob("*.qa"))
    if not qa_files:
        log.info(f"No .qa files found in {folder}.")
        return []
    items: list[tuple[str, QAItem]] = []
    for qa_file in qa_files:
        try:
            items.append((_extract_trace_id(qa_file.stem), QAItem.model_validate_json(qa_file.read_text())))
        except Exception as e:
            log.warning(f"  Parse error {qa_file.name}: {e}")
    results = _evaluate_parallel(client, mono_prompt, prompt_dir.name, items, max_workers=max_workers)
    log.info(f"\nEvaluated {len(results)} item(s) from {folder}.")
    source_path = str(folder.relative_to(PROJECT_ROOT))
    _write_eval_results(results, folder, prompt_dir, mono_prompt, client.model, source=source_path)
    return results


def evaluate_single_qa_file(client: MyOpenAIClient, mono_prompt: str, prompt_dir: Path, path: Path) -> list[dict]:
    log.info(f"Evaluating: {path.name}")
    try:
        qa_item = QAItem.model_validate_json(path.read_text())
    except Exception as e:
        log.warning(f"  Parse error: {e}")
        return []
    result = evaluate_qa_item(client, mono_prompt, _extract_trace_id(path.stem), qa_item)
    _log_result(result, prompt_dir.name)
    source_path = str(path.parent.relative_to(PROJECT_ROOT))
    _write_eval_results([result], path.parent, prompt_dir, mono_prompt, client.model, source=source_path)
    return [result]


def _resolve_target(client: MyOpenAIClient, mono_prompt: str, prompt_dir: Path, target: Path, max_workers: int = DEFAULT_MAX_PARALLEL) -> list[dict]:
    if target.is_dir():
        return evaluate_qa_folder(client, mono_prompt, prompt_dir, target, max_workers=max_workers)
    if target.suffix == ".jsonl":
        return evaluate_training_set(client, mono_prompt, prompt_dir, target, max_workers=max_workers)
    if target.suffix == ".qa":
        return evaluate_single_qa_file(client, mono_prompt, prompt_dir, target)
    log.warning(f"Unsupported file type: {target.suffix}. Expected a directory, .jsonl, or .qa file.")
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _select_model() -> str:
    catalog = MyOpenAIClient.available_models()
    models = list(catalog.keys())
    default_idx = models.index(DEFAULT_MODEL) + 1 if DEFAULT_MODEL in models else 1

    log.info("\nAvailable models:")
    log.info(f"  {'#':<4} {'Model':<30} {'Input/1M':>10} {'Cached/1M':>10} {'Output/1M':>10}")
    log.info(f"  {'─' * 68}")
    for i, m in enumerate(models, start=1):
        p = catalog[m]
        cached = f"${p['cached_input']:.4f}" if p["cached_input"] is not None else "—"
        marker = " *" if m == DEFAULT_MODEL else ""
        log.info(f"  {i:<4} {m + marker:<30} ${p['input']:>9.4f} {cached:>10} ${p['output']:>9.4f}")
    log.info("  * = default")

    choice = input(f"\nSelect model (1-{len(models)}) [{default_idx}]: ").strip()
    if choice == "":
        return models[default_idx - 1]
    if choice.isdigit() and 1 <= int(choice) <= len(models):
        return models[int(choice) - 1]
    log.info(f"Invalid choice {choice!r}, using default ({DEFAULT_MODEL}).")
    return DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]


def _select_max_parallel() -> int:
    presets = [10, 20, 50, 100]
    log.info("\nParallel workers:")
    for i, n in enumerate(presets, start=1):
        marker = " *" if n == DEFAULT_MAX_PARALLEL else ""
        log.info(f"  {i}) {n} workers{marker}")
    log.info("  5) Custom value")
    log.info("  * = default")

    choice = input(f"\nSelect (1-5) [{presets.index(DEFAULT_MAX_PARALLEL) + 1}]: ").strip()
    if choice == "":
        return DEFAULT_MAX_PARALLEL
    if choice.isdigit() and 1 <= int(choice) <= 4:
        return presets[int(choice) - 1]
    if choice == "5":
        custom = input("Enter number of parallel workers: ").strip()
        if custom.isdigit():
            val = int(custom)
            if val > 0:
                log.info(f"Using {val} parallel workers.")
                return val
        log.info(f"Invalid value, using default ({DEFAULT_MAX_PARALLEL}).")
        return DEFAULT_MAX_PARALLEL
    log.info(f"Invalid choice {choice!r}, using default ({DEFAULT_MAX_PARALLEL}).")
    return DEFAULT_MAX_PARALLEL


def main():
    parser = argparse.ArgumentParser(description="LLM judge for DIY repair QA items.")
    parser.add_argument(
        "--evaluate",
        metavar="OBJECT",
        help="Path to evaluate: a directory of .qa files, a .jsonl file, or a single .qa file.",
    )
    parser.add_argument("--model", metavar="MODEL", help="LLM model to use for judging.")
    parser.add_argument("--prompt-version", metavar="VERSION", help="Judge prompt version folder (e.g. v1, v2).")
    parser.add_argument("--max-parallel", type=int, metavar="N", default=DEFAULT_MAX_PARALLEL, help=f"Maximum parallel workers (default: {DEFAULT_MAX_PARALLEL}).")
    args = parser.parse_args()

    model = args.model if args.model else _select_model()
    client = MyOpenAIClient(model=model, temperature=0.1)
    client.validate_api_key()

    if args.prompt_version:
        prompt_dir = PROMPTS_ROOT / args.prompt_version
        if not prompt_dir.exists():
            log.error(f"Error: prompt version not found — {prompt_dir}")
            sys.exit(1)
    else:
        prompt_dir = _select_prompt_version()

    mono_prompt = _load_mono_prompt(prompt_dir)
    log.info(f"Using judge prompts from: {prompt_dir.name}\n")

    max_parallel = args.max_parallel if args.max_parallel != DEFAULT_MAX_PARALLEL else _select_max_parallel()

    if args.evaluate:
        target = Path(args.evaluate)
        if not target.exists():
            log.info(f"Error: path not found — {target}")
            return
        _resolve_target(client, mono_prompt, prompt_dir, target, max_workers=max_parallel)
        return

    # Interactive prompt
    qa_items_root = PROJECT_ROOT / "qa_items"
    versions = sorted(
        (p for p in qa_items_root.iterdir() if p.is_dir()),
        key=lambda p: int(p.name.lstrip("v")) if p.name.lstrip("v").isdigit() else 0,
    )

    # Discover JSONL files and count items
    def count_jsonl_items(path: Path) -> int:
        count = 0
        try:
            with path.open() as f:
                for line in f:
                    if line.strip():
                        count += 1
        except Exception:
            return 0
        return count

    jsonl_files = []
    jsonl_item_counts = {}
    for name in ["eval.jsonl", "train.jsonl"]:
        path = PROJECT_ROOT / name
        if path.exists():
            item_count = count_jsonl_items(path)
            jsonl_files.append((name, path, item_count))
            jsonl_item_counts[name] = item_count

    log.info("What would you like to evaluate?")
    choice_idx = 1
    jsonl_choices = {}
    for name, path, count in jsonl_files:
        jsonl_choices[choice_idx] = (name, path, count)
        marker = " *" if choice_idx == len(jsonl_files) else ""
        log.info(f"  {choice_idx}) {name}  [{count} items]{marker}")
        choice_idx += 1

    version_start_idx = choice_idx
    for i, folder in enumerate(versions, start=version_start_idx):
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
        marker = " *" if i == version_start_idx + len(versions) - 1 else ""
        log.info(f"  {i}) {folder.name}  [{qa_count} QA item(s){human_note}{llm_note}]{marker}")

    max_choice = version_start_idx + len(versions) - 1
    default_choice = max_choice
    log.info("  * = default")
    choice = input(f"Enter 1-{max_choice} [{default_choice}]: ").strip()

    if choice == "":
        selected_choice = default_choice
    elif choice.isdigit():
        selected_choice = int(choice)
    else:
        log.warning(f"Unknown choice: {choice!r}. Please enter a number between 1 and {max_choice}.")
        return

    if selected_choice in jsonl_choices:
        name, path, count = jsonl_choices[selected_choice]
        limit = None
        if count > 100:
            limit_prompt = input(f"This dataset has {count} items. Enter number to limit (default 200, or 0 for all): ").strip()
            if limit_prompt.isdigit():
                limit = int(limit_prompt)
                if limit == 0:
                    limit = None
                if limit:
                    log.info(f"Processing first {limit} items from {name}.")
            else:
                limit = 200
                log.info(f"Processing first {limit} items from {name} (default).")
        evaluate_training_set(client, mono_prompt, prompt_dir, eval_path=path, max_workers=max_parallel, limit=limit)
    elif version_start_idx <= selected_choice <= max_choice:
        version_idx = selected_choice - version_start_idx
        evaluate_qa_folder(client, mono_prompt, prompt_dir, versions[version_idx], max_workers=max_parallel)
    else:
        log.warning(f"Invalid choice: {selected_choice}. Please enter a number between 1 and {max_choice}.")


if __name__ == "__main__":
    main()
