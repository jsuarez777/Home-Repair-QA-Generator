#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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


def dim_sanity_check(qa_item: QAItem) -> None:
    problems = []
    if len(qa_item.safety_info) < 80:
        problems.append(f"safety_info too short ({len(qa_item.safety_info)} chars, min 80)")
    if not qa_item.steps:
        problems.append("steps list is empty")
    if not qa_item.tools_required:
        problems.append("tools_required list is empty")
    _UNREALISTIC_TOOL_PHRASES = {"professional-grade", "trade-only"}
    for tool in qa_item.tools_required:
        if len(tool.strip()) < 3:
            problems.append(f"tool name too short: '{tool}'")
        for phrase in _UNREALISTIC_TOOL_PHRASES:
            if phrase in tool.strip().lower():
                problems.append(f"unrealistic tool detected ('{phrase}'): '{tool}'")
    if not qa_item.tips:
        problems.append("tips list is empty")
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
            print(f"  [dedup] Skipping {f.name}: {e}")
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
                    print(f"  [dedup] Near-duplicate (jaccard={sim:.3f}): {f_j.name} ~ {items[i][0].name}")
                    print(f"    kept:    {items[i][1].question}")
                    print(f"    removed: {items[j][1].question}")

    if to_remove:
        dupes_dir = qa_folder / "duplicates"
        dupes_dir.mkdir(exist_ok=True)
        for f in to_remove:
            shutil.move(str(f), str(dupes_dir / f.name))
        print(f"  [dedup] Moved {len(to_remove)} duplicate(s) to {dupes_dir}.")
    else:
        print("  [dedup] No duplicates found.")


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
        print(f"Using QA items version: {default.name}")
        return default.name
    print("Available QA item versions:")
    for i, v in enumerate(versions, start=1):
        count = len(list(v.glob("*.qa")))
        print(f"  {i}) {v.name}  [{count} item(s)]")
    while True:
        choice = input(f"Select version (1-{len(versions)}) [{default.name}]: ").strip()
        if choice == "":
            return default.name
        if choice.isdigit() and 1 <= int(choice) <= len(versions):
            return versions[int(choice) - 1].name
        print(f"  Please enter a number between 1 and {len(versions)}.")


def main():
    parser = argparse.ArgumentParser(description="Validate QA items in a version folder.")
    parser.add_argument("--version", metavar="VERSION", help="QA items version folder (e.g. v1, v2).")
    args = parser.parse_args()

    version = args.version if args.version else _pick_version()
    qa_folder = PROJECT_ROOT / "qa_items" / version
    if not qa_folder.exists():
        print(f"Error: folder not found — {qa_folder}")
        sys.exit(1)

    qa_files = sorted(qa_folder.glob("*.qa"))
    if not qa_files:
        print(f"No .qa files found in {qa_folder}")
        sys.exit(0)

    print(f"Validating {len(qa_files)} item(s) in qa_items/{version}...\n")

    passed = 0
    failed = 0
    for f in qa_files:
        errors = []
        try:
            text = f.read_text().strip()

            try:
                json.loads(text)
            except json.JSONDecodeError as e:
                errors.append(f"invalid JSON: {e}")
                raise

            try:
                qa_item = QAItem.model_validate_json(text)
            except Exception as e:
                errors.append(f"schema validation: {e}")
                raise

            try:
                dim_sanity_check(qa_item)
            except ValueError as e:
                errors.append(str(e))
                raise

        except Exception:
            failed += 1
            print(f"FAIL  {f.name}")
            for err in errors:
                print(f"      {err}")
            continue

        passed += 1
        print(f"ok    {f.name}")

    print(f"\n{passed} passed, {failed} failed out of {len(qa_files)} item(s).")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
