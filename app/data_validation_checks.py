#!/usr/bin/env python3
import argparse
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_LOGS_DIR = PROJECT_ROOT / "logs"
_LOGS_DIR.mkdir(exist_ok=True)
_log_file = _LOGS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_data_validation_checks.log"

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

from pydantic import ValidationError
from qa_item import QAItem

JACCARD_THRESHOLD = 0.5
VAGUE_PHRASE_MAX_RATIO = 0.15

_STOPWORDS = {
    "a","an","the","and","or","but","if","in","on","at","to","for","of","with",
    "my","i","is","it","its","are","was","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","can","may","might","not",
    "no","so","how","what","why","when","where","which","who","that","this",
    "there","here","then","than","by","from","up","out","after","before","about",
    "first","just","also","still","even","already","any","all","both","each",
    "won","don","s","t",
}


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z']+", text.lower())
    return {w.strip("'") for w in words if w.strip("'") not in _STOPWORDS and len(w.strip("'")) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

_VAGUE_PHRASES = {"be careful", "good luck", "take your time", "stay safe", "use caution"}


def _matching_vague_phrase(text: str) -> str | None:
    """Returns a vague phrase if it appears in text and makes up more than 15% of its length."""
    text_lower = text.strip().lower()
    for phrase in _VAGUE_PHRASES:
        if phrase in text_lower and len(phrase) / len(text_lower) > VAGUE_PHRASE_MAX_RATIO:
            return phrase
    return None


def _validate_text(text: str) -> tuple[list[str], dict, list[str]]:
    """Returns (results[7], failures_detail, errors) for a raw QA item string."""
    results = ["PASS"] * 7
    failures_detail = {}
    errors = []

    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        results[0] = "FAIL"
        failures_detail["json_valid"] = True
        for i in range(1, 7):
            results[i] = "----"
        errors.append(f"invalid JSON: {e}")
        return results, failures_detail, errors

    try:
        qa_item = QAItem.model_validate_json(text)
    except ValidationError as e:
        for err in e.errors():
            t = err["type"]
            field = err.get("loc", (None,))[0]
            if t == "missing":
                results[1] = "FAIL"
                failures_detail.setdefault("all_fields_present", []).append(field)
            elif t == "string_too_short":
                results[2] = "FAIL"
                failures_detail.setdefault("non_empty_strings", []).append(field)
            elif t == "too_short":
                if field == "steps":
                    results[3] = "FAIL"
                    failures_detail["sufficient_steps"] = True
                elif field == "tools_required":
                    results[4] = "FAIL"
                    failures_detail["tools_present"] = True
                elif field == "tips":
                    results[5] = "FAIL"
                    failures_detail["tips_present"] = True
        errors.append(f"schema validation: {e}")
        return results, failures_detail, errors

    try:
        dim_sanity_check(qa_item)
    except ValueError as e:
        results[6] = "FAIL"
        failures_detail["no_vague_phrases"] = True
        errors.append(str(e))

    return results, failures_detail, errors


def dim_sanity_check(qa_item: QAItem) -> None:
    problems = []
    for tip in qa_item.tips:
        match = _matching_vague_phrase(tip)
        if match:
            problems.append(f"vague tip ('{match}' is >{VAGUE_PHRASE_MAX_RATIO:.0%} of tip): '{tip}'")
    match = _matching_vague_phrase(qa_item.safety_info)
    if match:
        problems.append(f"vague safety_info ('{match}' is >{VAGUE_PHRASE_MAX_RATIO:.0%} of field): '{qa_item.safety_info}'")
    if problems:
        raise ValueError("Sanity check failed:\n" + "\n".join(f"  - {p}" for p in problems))


def batch_dedup_check(qa_folder: Path) -> None:
    qa_files = sorted(qa_folder.glob("*.qa"), key=lambda f: int(re.search(r"QA(\d+)", f.name).group(1)) if re.search(r"QA(\d+)", f.name) else 0)
    if len(qa_files) < 2:
        return

    items = []
    for f in qa_files:
        try:
            items.append((f, QAItem.model_validate_json(f.read_text())))
        except Exception as e:
            log.warning(f"  [dedup] Skipping {f.name}: {e}")
            continue

    to_remove: set[Path] = set()

    if len(items) >= 2:
        flagged: set[int] = set()
        for i in range(len(items)):
            if i in flagged:
                continue
            words_i = _content_words(items[i][1].question)
            for j in range(i + 1, len(items)):
                if j in flagged:
                    continue
                sim = _jaccard(words_i, _content_words(items[j][1].question))
                if sim >= JACCARD_THRESHOLD:
                    f_j = items[j][0]
                    to_remove.add(f_j)
                    flagged.add(j)
                    log.info(f"  [dedup] Near-duplicate (jaccard={sim:.3f}): {f_j.name} ~ {items[i][0].name}")
                    log.info(f"    kept:    {items[i][1].question}")
                    log.info(f"    removed: {items[j][1].question}")

    if to_remove:
        dupes_dir = qa_folder / "duplicates"
        dupes_dir.mkdir(exist_ok=True)
        for f in to_remove:
            shutil.move(str(f), str(dupes_dir / f.name))
        log.info(f"  [dedup] Moved {len(to_remove)} duplicate(s) to {dupes_dir}.")
    else:
        log.info("  [dedup] No duplicates found.")


def _pick_version() -> str:
    qa_items_root = PROJECT_ROOT / "qa_items"
    versions = sorted(
        (p for p in qa_items_root.iterdir() if p.is_dir()),
        key=lambda p: int(p.name.lstrip("v")) if p.name.lstrip("v").isdigit() else 0,
    )
    if not versions:
        raise RuntimeError(f"No version folders found in {qa_items_root}")
    default = versions[-1]
    if len(versions) == 1:
        log.info(f"Using QA items version: {default.name}")
        return default.name
    log.info("Available QA item versions:")
    for i, v in enumerate(versions, start=1):
        count = len(list(v.glob("*.qa")))
        log.info(f"  {i}) {v.name}  [{count} item(s)]")
    while True:
        choice = input(f"Select version (1-{len(versions)}) [{default.name}]: ").strip()
        if choice == "":
            return default.name
        if choice.isdigit() and 1 <= int(choice) <= len(versions):
            return versions[int(choice) - 1].name
        log.info(f"  Please enter a number between 1 and {len(versions)}.")


def main():
    parser = argparse.ArgumentParser(description="Validate QA items in a version folder.")
    parser.add_argument("--version", metavar="VERSION", help="QA items version folder (e.g. v1, v2).")
    args = parser.parse_args()

    version = args.version if args.version else _pick_version()
    qa_folder = PROJECT_ROOT / "qa_items" / version
    if not qa_folder.exists():
        log.error(f"Error: folder not found — {qa_folder}")
        sys.exit(1)

    qa_files = sorted(qa_folder.glob("*.qa"))
    if not qa_files:
        log.info(f"No .qa files found in {qa_folder}")
        sys.exit(0)

    log.info(f"Validating {len(qa_files)} item(s) in qa_items/{version}...\n")
    log.info("Checks performed:")
    log.info("  1. json_valid          — raw file content parses as valid JSON")
    log.info("  2. all_fields_present  — all 7 fields present (question, answer, equipment_problem,")
    log.info("                           tools_required, steps, safety_info, tips)")
    log.info("  3. non_empty_strings   — string fields meet minimum lengths (question>=20, answer>=100,")
    log.info("                           equipment_problem>=10, safety_info>=80)")
    log.info("  4. sufficient_steps    — steps list has >= 3 items")
    log.info("  5. tools_present       — tools_required list has >= 1 item")
    log.info("  6. tips_present        — tips list has >= 1 item")
    log.info("  7. no_vague_phrases    — tips and safety_info do not contain vague filler phrases")
    log.info("                           (e.g. 'be careful', 'good luck') that dominate the field")
    log.info("")

    failed_dir = qa_folder / "failed_checks"

    passed = 0
    failed = 0
    for f in qa_files:
        text = f.read_text().strip()
        results, failures_detail, errors = _validate_text(text)
        item_failed = any(r == "FAIL" for r in results)

        result_line = ",".join(results)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        trace_id = f.stem.split("_")[0]
        failures_json = json.dumps(failures_detail) if failures_detail else "None"
        if item_failed:
            failed += 1
            failed_dir.mkdir(exist_ok=True)
            shutil.move(str(f), str(failed_dir / f.name))
            log.info(f"[ts={ts}] [trace_id={trace_id}] [status=FAIL] [filename={f.name}]  [checks={result_line}]  [FAILURES={failures_json}]")
            for err in errors:
                log.info(f"      {err}")
        else:
            passed += 1
            log.info(f"[ts={ts}] [trace_id={trace_id}] [status=ok]   [filename={f.name}]  [checks={result_line}]  [FAILURES={failures_json}]")

    log.info(f"\n{passed} passed, {failed} failed out of {len(qa_files)} item(s).")

    log.info("\nRunning dedup check on passing items...")
    batch_dedup_check(qa_folder)

    remaining = sorted(qa_folder.glob("*.qa"))
    if remaining:
        dist: dict[str, int] = {}
        for f in remaining:
            cat = re.sub(r"\d+$", "", f.stem.split("_")[1]) if "_" in f.stem else "unknown"
            dist[cat] = dist.get(cat, 0) + 1
        total = len(remaining)
        log.info(f"\nFinal item distribution by category ({total} of {len(qa_files)} original):")
        for cat, count in sorted(dist.items()):
            pct = count / total * 100
            log.info(f"  {cat}: {pct:.1f}% ({count})")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
