#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from huggingface_hub import try_to_load_from_cache
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if try_to_load_from_cache("sentence-transformers/all-MiniLM-L6-v2", "config.json"):
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

from qa_item import QAItem

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.88


def semantic_dedup_check(qa_folder: Path, threshold: float = SIMILARITY_THRESHOLD) -> None:
    qa_files = sorted(
        qa_folder.glob("*.qa"),
        key=lambda f: int(re.search(r"QA(\d+)", f.name).group(1)) if re.search(r"QA(\d+)", f.name) else 0,
    )
    if len(qa_files) < 2:
        print("  [dedup] Not enough items to compare.")
        return

    items = []
    for f in qa_files:
        try:
            items.append((f, QAItem.model_validate_json(f.read_text())))
        except Exception as e:
            print(f"  [dedup] Skipping {f.name}: {e}")

    if len(items) < 2:
        print("  [dedup] Not enough valid items to compare.")
        return

    to_remove: set[Path] = set()

    questions = [qa_item.question for _, qa_item in items]
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(questions, convert_to_tensor=True)

    flagged: set[int] = set()
    for i in range(len(items)):
        if i in flagged:
            continue
        for j in range(i + 1, len(items)):
            if j in flagged:
                continue
            sim = float(cos_sim(embeddings[i], embeddings[j]))
            if sim >= threshold:
                f_j = items[j][0]
                to_remove.add(f_j)
                flagged.add(j)
                print(f"  [dedup] Near-duplicate (sim={sim:.3f}): {f_j.name} ~ {items[i][0].name}")
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
    parser = argparse.ArgumentParser(description="Semantic dedup using cosine similarity on question embeddings.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--version", metavar="VERSION", help="QA items version folder (e.g. v1, v2).")
    group.add_argument("--path", metavar="PATH", help="Full path to a QA items folder.")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD,
                        help=f"Cosine similarity threshold (default: {SIMILARITY_THRESHOLD})")
    args = parser.parse_args()

    threshold = args.threshold
    if args.path:
        qa_folder = Path(args.path)
    else:
        version = args.version if args.version else _pick_version()
        qa_folder = PROJECT_ROOT / "qa_items" / version
    if not qa_folder.exists():
        print(f"Error: folder not found — {qa_folder}")
        sys.exit(1)

    print(f"Running semantic dedup on {qa_folder} (threshold={threshold})...\n")
    t0 = time.perf_counter()
    semantic_dedup_check(qa_folder, threshold=threshold)
    elapsed = time.perf_counter() - t0
    count = len(list(qa_folder.glob("*.qa")))
    print(f"\n  [perf] {count} item(s) processed in {elapsed:.2f}s ({elapsed / max(count, 1) * 1000:.1f}ms/item)")


if __name__ == "__main__":
    main()
