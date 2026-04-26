#!/usr/bin/env python3
"""
Corrupt QA items in a version folder for LLM judge testing.

Randomly selects (with overlap allowed) four groups of N items and:
  - Group 1: replaces safety_info with an irrelevant string
  - Group 2: replaces tips with an irrelevant tip
  - Group 3: appends one expensive (>$100) tool to tools_required
  - Group 4: inserts 1-3 irrelevant steps into steps (and answer)

Before corrupting, all .qa files are preserved under:
  <folder>/preserved/<YYYYMMDD_HHMMSS>/

Usage:
    python corrupt_qa_items_for_testing.py            # interactive version picker
    python corrupt_qa_items_for_testing.py qa_items/v5
"""

import argparse
import json
import logging
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QA_ITEMS_ROOT = PROJECT_ROOT / "qa_items"

_LOGS_DIR = PROJECT_ROOT / "logs"
_LOGS_DIR.mkdir(exist_ok=True)
_log_file = _LOGS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_corrupt_qa_items_for_testing.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(_log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)
log.info(f"Logging to {_log_file}")


BAD_SAFETY = (
    "Always wear sunscreen when working outdoors on sunny days and stay hydrated "
    "to avoid heat exhaustion during yard work."
)

BAD_TIPS = [
    "Remember to water your houseplants regularly and keep them away from direct "
    "sunlight to prevent leaf burn."
]

EXPENSIVE_TOOLS = [
    "thermal imaging camera",
    "professional pipe inspection camera",
    "industrial wet/dry vacuum",
    "laser level kit",
    "cordless rotary hammer drill",
]

IRRELEVANT_STEPS = [
    "Apply a slow-release fertilizer to the lawn and water it in thoroughly.",
    "Check tire pressure on all four tires and inflate to the manufacturer's recommended PSI.",
    "Descale the espresso machine by running a citric acid solution through the brew cycle.",
    "Reprogram the sprinkler controller for the new seasonal watering schedule.",
    "Replace the cabin air filter in the vehicle's HVAC compartment.",
    "Trim any overhanging tree branches that may be shading the garden bed.",
    "Calibrate the oven temperature using a standalone thermometer and adjust the offset setting.",
    "Apply a coat of car wax to all painted exterior panels and buff to a shine.",
    "Aerate the lawn using a core aerator and overseed any bare patches.",
    "Clean and condition the leather seats with a pH-balanced leather conditioner.",
    "Top-dress garden beds with two inches of compost and work it in lightly.",
    "Flush the radiator coolant and refill with a 50/50 antifreeze-water mix.",
    "Sharpen the lawn mower blade using a metal file or bench grinder.",
    "Rotate the mattress 180 degrees and flip it if it is a double-sided model.",
    "Run a dishwasher cleaning tablet through an empty hot-water cycle to remove buildup.",
]


# ---------------------------------------------------------------------------
# Version picker
# ---------------------------------------------------------------------------

def _select_version_folder() -> Path:
    versions = sorted(
        (p for p in QA_ITEMS_ROOT.iterdir() if p.is_dir()),
        key=lambda p: int(p.name.lstrip("v")) if p.name.lstrip("v").isdigit() else 0,
    )
    if not versions:
        raise RuntimeError(f"No version folders found in {QA_ITEMS_ROOT}")
    log.info("Select a QA items version to corrupt:")
    for i, v in enumerate(versions, start=1):
        qa_count = len(list(v.glob("*.qa")))
        log.info(f"  {i}) {v.name}  [{qa_count} QA item(s)]")
    choice = input(f"Enter 1-{len(versions)}: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(versions):
        return versions[int(choice) - 1]
    log.info(f"Invalid choice {choice!r}, defaulting to {versions[-1].name}.")
    return versions[-1]


# ---------------------------------------------------------------------------
# Preservation
# ---------------------------------------------------------------------------

def _preserve(folder: Path, files: list[Path]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = folder / "preserved" / stamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, backup_dir / f.name)
    log.info(f"Preserved {len(files)} file(s) to {backup_dir}\n")
    return backup_dir


# ---------------------------------------------------------------------------
# Answer field helpers
# ---------------------------------------------------------------------------

def _rebuild_answer_steps(answer: str, original_steps: list[str], new_steps: list[str]) -> str:
    """Replace the numbered steps block in answer with new_steps.

    If the steps are not embedded in the answer as numbered lines, appends them.
    """
    if not original_steps:
        return answer

    # Look for "1. <beginning of first step>"
    first_snippet = original_steps[0][:50]
    start_idx = answer.find("1. " + first_snippet)

    new_block = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(new_steps))

    if start_idx == -1:
        # Steps not embedded — append them
        return answer + "\n" + new_block

    # Find end of last original step in answer
    last_snippet = original_steps[-1]
    last_idx = answer.rfind(last_snippet)
    if last_idx == -1:
        return answer + "\n" + new_block

    end_idx = last_idx + len(last_snippet)
    return answer[:start_idx] + new_block + answer[end_idx:]


# ---------------------------------------------------------------------------
# Corruption helpers
# ---------------------------------------------------------------------------

def corrupt_safety_info(files: list[Path]) -> None:
    for f in files:
        data = json.loads(f.read_text())
        data["safety_info"] = BAD_SAFETY
        f.write_text(json.dumps(data, indent=2))
        log.info(f"  [safety_info] {f.name}")


def corrupt_tips(files: list[Path]) -> None:
    for f in files:
        data = json.loads(f.read_text())
        data["tips"] = BAD_TIPS
        f.write_text(json.dumps(data, indent=2))
        log.info(f"  [tips] {f.name}")


def corrupt_tools(files: list[Path]) -> None:
    for f in files:
        data = json.loads(f.read_text())
        data["tools_required"] = data["tools_required"] + [random.choice(EXPENSIVE_TOOLS)]
        f.write_text(json.dumps(data, indent=2))
        log.info(f"  [tools_required] {f.name} -> {data['tools_required'][-1]}")


def _steps_embedded_in_answer(data: dict) -> bool:
    """Return True only if every step in the steps field appears in the answer field."""
    answer = data.get("answer", "")
    return all(step in answer for step in data.get("steps", []))


def corrupt_steps(selected: list[Path], pool: list[Path]) -> None:
    remaining_pool = [f for f in pool if f not in selected]
    targets = list(selected)
    i = 0
    corrupted = 0
    while i < len(targets) and corrupted < len(selected):
        f = targets[i]
        data = json.loads(f.read_text())
        if not _steps_embedded_in_answer(data):
            log.info(f"  [steps] SKIP {f.name}  (steps not fully present in answer field)")
            if remaining_pool:
                replacement = random.choice(remaining_pool)
                remaining_pool.remove(replacement)
                targets.append(replacement)
            i += 1
            continue

        original_steps: list[str] = list(data["steps"])
        bad = random.sample(IRRELEVANT_STEPS, k=random.randint(1, 3))
        new_steps = list(original_steps)
        for step in bad:
            pos = random.randint(0, len(new_steps))
            new_steps.insert(pos, step)

        data["steps"] = new_steps
        data["answer"] = _rebuild_answer_steps(data["answer"], original_steps, new_steps)
        f.write_text(json.dumps(data, indent=2))
        log.info(f"  [steps] {f.name}  (+{len(bad)} irrelevant step(s) at random positions)")
        corrupted += 1
        i += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _prompt_percentage(total: int) -> int:
    """Ask the user for a corruption percentage and return the item count (min 1)."""
    while True:
        raw = input(f"What percentage of items to corrupt per category? ({total} items total): ").strip().rstrip("%")
        try:
            pct = float(raw)
            if not (0 < pct <= 100):
                raise ValueError
            n = max(1, round(total * pct / 100))
            log.info(f"  → {pct}% of {total} = {n} item(s) per category")
            return n
        except ValueError:
            log.info("  Please enter a number between 1 and 100.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrupt QA items for LLM judge testing.")
    parser.add_argument("folder", nargs="?", help="Path to a QA version folder (omit for interactive picker)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--pct", type=float, default=None, help="Percentage of items to corrupt per category (e.g. 10)")
    args = parser.parse_args()

    if args.folder:
        folder = Path(args.folder)
        if not folder.is_dir():
            log.error(f"Error: {folder} is not a directory.")
            return
    else:
        folder = _select_version_folder()

    files = sorted(folder.glob("*.qa"))
    if not files:
        log.error(f"Error: no .qa files found in {folder}.")
        return

    n = max(1, round(len(files) * args.pct / 100)) if args.pct is not None else _prompt_percentage(len(files))

    _preserve(folder, files)

    random.seed(args.seed)
    group_safety = random.sample(files, n)
    group_tips   = random.sample(files, n)
    group_tools  = random.sample(files, n)
    group_steps  = random.sample(files, n)

    log.info(f"\nCorrupting {n} item(s) per category in {folder.name} (seed={args.seed})\n")

    corrupt_safety_info(group_safety)
    log.info("")
    corrupt_tips(group_tips)
    log.info("")
    corrupt_tools(group_tools)
    log.info("")
    corrupt_steps(group_steps, files)
    log.info("\nDone.")


if __name__ == "__main__":
    main()
