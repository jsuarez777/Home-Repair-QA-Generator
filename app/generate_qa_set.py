#!/usr/bin/env python3
import argparse
import json
import logging
import os
import shutil
import sys
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from random import randint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openai_client.openai_client import MyOpenAIClient

DEFAULT_MODEL = "gpt-5.4-mini"

_LOGS_DIR = PROJECT_ROOT / "logs"
_LOGS_DIR.mkdir(exist_ok=True)
_log_file = _LOGS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_generate_qa_set.log"

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


def _read_file_content(file_path: str) -> str:
    try:
        with open(file_path) as f:
            content = f.read().strip()
            if not content:
                raise ValueError(f"File {file_path} is empty.")
            return content
    except Exception as e:
        log.info(f"Error reading file {file_path}: {e}")
        raise


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


def _pick_version() -> str:
    prompts_dir = PROJECT_ROOT / "prompts"
    versions = sorted(
        (p.name for p in prompts_dir.iterdir() if p.is_dir()),
        key=lambda v: int(v.lstrip("v")) if v.lstrip("v").isdigit() else 0,
    )
    if not versions:
        raise RuntimeError(f"No version folders found in {prompts_dir}")
    default = versions[-1]
    if len(versions) == 1:
        log.info(f"Using prompt version: {default}")
        return default
    log.info("Available prompt versions:")
    for i, v in enumerate(versions, start=1):
        log.info(f"  {i}) {v}")
    while True:
        choice = input(f"Select version (1-{len(versions)}) [{default}]: ").strip()
        if choice == "":
            return default
        if choice.isdigit() and 1 <= int(choice) <= len(versions):
            return versions[int(choice) - 1]
        log.info(f"  Please enter a number between 1 and {len(versions)}.")



DEFAULT_MAX_PARALLEL = 50
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0  # seconds; doubles on each attempt


def _generate_one(client: MyOpenAIClient, cat: str, prompt: str) -> tuple[str, str, str]:
    from openai import RateLimitError
    delay = RETRY_BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.query(input=prompt)
            raw_response = response.output_text
            try:
                response_text = raw_response.strip(' `\n')
                brace = response_text.find("{")
                bracket = response_text.find("[")
                json_start = min(brace, bracket) if brace != -1 and bracket != -1 else max(brace, bracket)
                if json_start == -1:
                    raise ValueError(f"No valid JSON start character ('{{' or '[') found in response:\n{response_text}")
                response_text = response_text[json_start:]
                if response_text.startswith("["):
                    parsed = json.loads(response_text)
                    if not isinstance(parsed, list) or len(parsed) != 1:
                        raise ValueError(f"Expected a JSON array with exactly 1 item, got {len(parsed) if isinstance(parsed, list) else 'non-list'}")
                    response_text = json.dumps(parsed[0])
                json.loads(response_text)
            except Exception as e:
                sep = "=" * 80
                log.warning(f"\n  [parse error] {cat}: {e}\n  {sep}\n{response_text}\n  {sep}\n")
                raise
            return cat, response_text, raw_response
        except RateLimitError as e:
            if attempt == MAX_RETRIES:
                raise
            log.warning(f"  [rate limit] {cat}: attempt {attempt}/{MAX_RETRIES}, retrying in {delay:.1f}s...")
            time.sleep(delay)
            delay *= 2


def main():
    parser = argparse.ArgumentParser(description="Generate QA items.")
    parser.add_argument("--version", metavar="VERSION", help="Prompt version folder (e.g. v1, v2).")
    parser.add_argument("--model", metavar="MODEL", help="LLM model to use for generation.")
    parser.add_argument("--num-items", type=int, metavar="NUM", help="Number of items to generate (1-1000).")
    parser.add_argument("--temperature", type=float, metavar="TEMP", default=1.8, help="Temperature for generation (default: 1.8).")
    parser.add_argument("--max-parallel", type=int, metavar="N", default=DEFAULT_MAX_PARALLEL, help=f"Maximum parallel workers (default: {DEFAULT_MAX_PARALLEL}).")
    args = parser.parse_args()

    model = args.model if args.model else _select_model()
    prompt_version = args.version if args.version else _pick_version()
    prompt_output_format_suffix_file = PROJECT_ROOT / f"prompts/{prompt_version}/gen_qa_output_format_suffix.prompt"
    prompt_output_format_suffix = _read_file_content(str(prompt_output_format_suffix_file))
    qa_folder = PROJECT_ROOT / f"qa_items/{prompt_version}"
    qa_folder.mkdir(parents=True, exist_ok=True)

    # Archive any existing .qa files (and duplicates/) before generating new ones
    items_to_archive = [
        f for f in qa_folder.iterdir()
        if not f.name.startswith(".") and f.name != "archive"
    ] if qa_folder.exists() else []
    if items_to_archive:
        archive_base = qa_folder / "archive"
        archive_base.mkdir(parents=True, exist_ok=True)
        existing_counts = [
            int(p.name) for p in archive_base.iterdir()
            if p.is_dir() and p.name.isdigit()
        ]
        next_count = max(existing_counts, default=0) + 1
        archive_dest = archive_base / str(next_count)
        archive_dest.mkdir()
        for item in items_to_archive:
            shutil.move(str(item), str(archive_dest / item.name))

    if args.num_items:
        if 1 <= args.num_items <= 1000:
            items_to_generate = args.num_items
        else:
            log.error("Number of items must be between 1 and 1000")
            sys.exit(1)
    else:
        while True:
            raw = input("How many items to generate? [50]: ").strip()
            if raw == "":
                items_to_generate = 50
                break
            if raw.isdigit() and 1 <= int(raw) <= 1000:
                items_to_generate = int(raw)
                break
            log.info("  Please enter a number between 1 and 1000.")

    # Setup prompts for each category
    categories_file = PROJECT_ROOT / f"prompts/{prompt_version}/categories.yml"
    with open(categories_file) as f:
        category_defs = yaml.safe_load(f)
    categories = list(category_defs.keys())

    template_file = str(PROJECT_ROOT / f"prompts/{prompt_version}/gen_qa_items.template")
    template = _read_file_content(template_file)
    prompts = {}
    for cat, vars in category_defs.items():
        prompt = template.format_map(vars)
        prompt += "\n\n" + prompt_output_format_suffix
        prompts[cat] = prompt

    sep = "=" * 80
    log.info(f"\n{sep}\nTEMPLATE ({template_file}):\n{sep}\n{template}")
    log.info(f"\n{sep}\nOUTPUT FORMAT SUFFIX ({prompt_output_format_suffix_file}):\n{sep}\n{prompt_output_format_suffix}")
    log.info(f"\n{sep}\nCATEGORIES ({categories_file}):\n{sep}")
    with open(categories_file) as f:
        log.info(f.read())
    log.info(sep)

    # Initialize the OpenAI client and generate QA items in parallel
    my_ai_client = MyOpenAIClient(model=model, temperature=args.temperature)
    total_count = 0
    count_per_category = {cat: 0 for cat in categories}

    def _pick_cat():
        return categories[randint(0, len(categories) - 1)]

    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        pending = {}  # future -> cat
        for _ in range(min(args.max_parallel, items_to_generate)):
            cat = _pick_cat()
            pending[executor.submit(_generate_one, my_ai_client, cat, prompts[cat])] = cat

        while total_count < items_to_generate and pending:
            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                cat = pending.pop(future)
                try:
                    result_cat, response_text, raw_response = future.result()
                    if total_count < items_to_generate:
                        total_count += 1
                        count_per_category[result_cat] += 1
                        output_file = qa_folder / f"QA{total_count}_{result_cat}{count_per_category[result_cat]}.qa"
                        output_file.write_text(response_text)
                        log.info(f"[{total_count}/{items_to_generate}] Saved {output_file.name} (category: {result_cat})")
                        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
                        trace_id = output_file.stem.split("_")[0]
                        vars = category_defs[result_cat]
                        examples_str = " ".join(
                            f'category={v}' if k == "repair_type" else f'{k}="{v}"'
                            for k, v in vars.items()
                        )
                        escaped_raw = raw_response.replace('\n', '\\n')
                        log.info(f"  ts={ts} trace_id={trace_id} filename={output_file.name} model={my_ai_client.model} {examples_str} raw_response={escaped_raw}")
                except Exception as e:
                    log.warning(f"============================={cat}: {e}")

                # Submit a replacement if we still need more items
                still_needed = items_to_generate - total_count - len(pending)
                if still_needed > 0:
                    new_cat = _pick_cat()
                    pending[executor.submit(_generate_one, my_ai_client, new_cat, prompts[new_cat])] = new_cat


if __name__ == "__main__":
	main()
