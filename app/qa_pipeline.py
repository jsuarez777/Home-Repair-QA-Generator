#!/usr/bin/env python3
import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_LOGS_DIR = PROJECT_ROOT / "logs"
_LOGS_DIR.mkdir(exist_ok=True)
_log_file = _LOGS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_qa_pipeline.log"

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

from openai_client.openai_client import MyOpenAIClient


def _select_model(prompt: str = "Select model") -> str:
    catalog = MyOpenAIClient.available_models()
    models = list(catalog.keys())
    if not models:
        raise RuntimeError("No available models found")

    log.info(f"\n{prompt}:")
    log.info(f"  {'#':<4} {'Model':<30} {'Input/1M':>10} {'Cached/1M':>10} {'Output/1M':>10}")
    log.info(f"  {'─' * 68}")
    for i, m in enumerate(models, start=1):
        p = catalog[m]
        cached = f"${p['cached_input']:.4f}" if p["cached_input"] is not None else "—"
        log.info(f"  {i:<4} {m:<30} ${p['input']:>9.4f} {cached:>10} ${p['output']:>9.4f}")

    while True:
        choice = input(f"\nEnter model number (1-{len(models)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        log.info(f"Please enter a number between 1 and {len(models)}.")


def _pick_version(root_dir: Path, name: str = "prompt") -> str:
    versions = sorted(
        (p for p in root_dir.iterdir() if p.is_dir()),
        key=lambda p: int(p.name.lstrip("v")) if p.name.lstrip("v").isdigit() else 0,
    )
    if not versions:
        raise RuntimeError(f"No {name} version folders found in {root_dir}")
    default = versions[-1].name
    if len(versions) == 1:
        log.info(f"Using {name} version: {default}")
        return default

    log.info(f"\nAvailable {name} versions:")
    for i, v in enumerate(versions, start=1):
        log.info(f"  {i}) {v.name}")

    while True:
        choice = input(f"Select {name} version (1-{len(versions)}) [{default}]: ").strip()
        if choice == "":
            return default
        if choice.isdigit() and 1 <= int(choice) <= len(versions):
            return versions[int(choice) - 1].name
        log.info(f"Please enter a number between 1 and {len(versions)}.")


def _run_command(cmd: list[str], description: str) -> bool:
    log.info(f"\n{'=' * 80}")
    log.info(f"Running: {description}")
    log.info(f"Command: {' '.join(cmd)}")
    log.info(f"{'=' * 80}\n")

    try:
        result = subprocess.run(cmd, check=True)
        log.info(f"\n✓ {description} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"\n✗ {description} failed with exit code {e.returncode}\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="QA generation and evaluation pipeline.")
    parser.add_argument("--gen-model", metavar="MODEL", help="LLM model for QA generation.")
    parser.add_argument("--gen-version", metavar="VERSION", help="Prompt version for QA generation.")
    parser.add_argument("--num-items", type=int, metavar="NUM", help="Number of items to generate (1-1000).")
    parser.add_argument("--temperature", type=float, metavar="TEMP", default=1.0, help="Temperature for QA generation (default: 1.0).")
    parser.add_argument("--max-parallel", type=int, metavar="N", help="Maximum parallel workers (default: 50).")
    parser.add_argument("--judge-model", metavar="MODEL", help="LLM model for judging.")
    parser.add_argument("--judge-prompt-version", metavar="VERSION", help="Judge prompt version.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip data validation step.")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM judging step.")
    args = parser.parse_args()

    log.info("QA Generation and Evaluation Pipeline")
    log.info("=" * 80)

    # Collect inputs
    gen_model = args.gen_model if args.gen_model else _select_model("Select model for QA generation")
    log.info(f"Using generation model: {gen_model}")

    gen_version = args.gen_version if args.gen_version else _pick_version(
        PROJECT_ROOT / "prompts", "generation prompt"
    )
    log.info(f"Using generation prompt version: {gen_version}")

    if args.num_items:
        num_items = args.num_items
        log.info(f"Generating {num_items} items")
    else:
        while True:
            raw = input("\nHow many items to generate? [50]: ").strip()
            if raw == "":
                num_items = 50
                break
            if raw.isdigit() and 1 <= int(raw) <= 1000:
                num_items = int(raw)
                break
            log.info("Please enter a number between 1 and 1000.")
        log.info(f"Generating {num_items} items")

    judge_model = None
    judge_prompt_version = None
    if not args.skip_judge:
        judge_model = args.judge_model if args.judge_model else _select_model("Select model for judging")
        log.info(f"Using judge model: {judge_model}")

        judge_prompt_version = args.judge_prompt_version if args.judge_prompt_version else _pick_version(
            PROJECT_ROOT / "prompts_llm_judge", "judge prompt"
        )
        log.info(f"Using judge prompt version: {judge_prompt_version}")

    # Step 1: Generate QA items
    gen_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "app" / "generate_qa_set.py"),
        "--model", gen_model,
        "--version", gen_version,
        "--num-items", str(num_items),
    ]
    if args.temperature is not None:
        gen_cmd.extend(["--temperature", str(args.temperature)])
    if args.max_parallel is not None:
        gen_cmd.extend(["--max-parallel", str(args.max_parallel)])
    if not _run_command(gen_cmd, "QA Generation"):
        return

    # Step 2: Validate QA items
    if not args.skip_validation:
        val_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "app" / "data_validation_checks.py"),
            "--version", gen_version,
        ]
        if not _run_command(val_cmd, "Data Validation"):
            log.warning("Validation failed, but continuing to judge step...\n")

    # Step 3: Judge QA items
    if not args.skip_judge:
        judge_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "app" / "llm_judge.py"),
            "--model", judge_model,
            "--prompt-version", judge_prompt_version,
        ]
        if args.max_parallel is not None:
            judge_cmd.extend(["--max-parallel", str(args.max_parallel)])
        if not _run_command(judge_cmd, "LLM Judging"):
            log.error("Judging failed")
            return

    log.info("=" * 80)
    log.info("✓ Pipeline completed successfully!")
    log.info(f"See logs at: {_log_file}")


if __name__ == "__main__":
    main()
