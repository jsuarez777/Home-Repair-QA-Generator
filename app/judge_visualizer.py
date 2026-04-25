#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

QA_ITEMS_DIR = Path(__file__).parent.parent / "qa_items"
HUMAN_FILE = "QA_human_eval.json"
LLM_FILE_PATTERN = "QA_llm_eval_*.json"
DIMENSIONS = [
    "answer_completeness",
    "context_clarity",
    "tool_realism",
    "scope_appropriateness",
    "safety_specificity",
    "tip_usefulness",
]


def discover_versions():
    versions = []
    for d in sorted(QA_ITEMS_DIR.iterdir()):
        if not d.is_dir():
            continue
        has_human = (d / HUMAN_FILE).exists()
        llm_count = len(list(d.glob(LLM_FILE_PATTERN)))
        if has_human or llm_count > 0:
            versions.append((d.name, d, has_human, llm_count))
    return versions


def load_eval(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("results", [])


def load_llm_evals(version_dir: Path) -> dict[str, list[dict]]:
    evals = {}
    for f in sorted(version_dir.glob(LLM_FILE_PATTERN)):
        key = f.stem.replace("QA_llm_eval_", "")
        try:
            evals[key] = load_eval(f)
        except Exception as e:
            print(f"  Warning: could not load {f.name}: {e}")
    return evals


def pass_rate(records: list[dict]) -> float:
    passes = [r["overall_pass"] for r in records if r.get("overall_pass") is not None]
    return sum(passes) / len(passes) * 100 if passes else 0.0


def dim_score(records: list[dict], dim: str) -> float:
    vals = [r[dim] for r in records if r.get(dim) is not None]
    return sum(vals) / len(vals) * 100 if vals else float("nan")


def bar(value: float, width: int = 30) -> str:
    if value != value:  # nan
        return "[" + "?" * width + "]"
    filled = round(value / 100 * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def print_section(title: str, records: list[dict], labeler_label: str):
    n = len(records)
    pr = pass_rate(records)
    print(f"\n  {labeler_label}  ({n} items)")
    print(f"  {'─' * 50}")
    print(f"  Overall pass rate : {bar(pr)} {pr:5.1f}%")
    print()
    for dim in DIMENSIONS:
        score = dim_score(records, dim)
        label = f"{dim:<25}"
        if score != score:
            print(f"  {label}: {'n/a':>6}")
        else:
            print(f"  {label}: {bar(score, 20)} {score:5.1f}%")


def compare_section(human: list[dict], llm: list[dict], label: str = "HUMAN vs LLM AGREEMENT"):
    human_map = {r["trace_id"]: r for r in human}
    llm_map = {r["trace_id"]: r for r in llm}
    shared = set(human_map) & set(llm_map)
    if not shared:
        print("\n  No overlapping trace_ids — cannot compare directly.")
        return
    print(f"\n  {label}  ({len(shared)} shared traces)")
    print(f"  {'─' * 50}")
    for dim in DIMENSIONS + ["overall_pass"]:
        agreements = 0
        total = 0
        for tid in shared:
            hv = human_map[tid].get(dim)
            lv = llm_map[tid].get(dim)
            if hv is None or lv is None:
                continue
            total += 1
            if hv == lv:
                agreements += 1
        if total:
            pct = agreements / total * 100
            print(f"  {dim:<25}: {bar(pct, 20)} {pct:5.1f}% agree  ({agreements}/{total})")
        else:
            print(f"  {dim:<25}: n/a")


def visualize(version_name: str, version_dir: Path, has_human: bool, llm_count: int):
    print(f"\n{'═' * 60}")
    print(f"  Version: {version_name}")
    print(f"{'═' * 60}")

    human_records = load_eval(version_dir / HUMAN_FILE) if has_human else None
    llm_evals = load_llm_evals(version_dir)  # dict[str, list[dict]]

    if human_records:
        print_section("HUMAN EVAL", human_records, "Human Judge")

    for llm_ver, llm_records in llm_evals.items():
        print_section(f"LLM EVAL ({llm_ver})", llm_records, f"LLM Judge {llm_ver}")
        if human_records:
            print(f"\n{'─' * 60}")
            compare_section(human_records, llm_records, f"Human vs LLM {llm_ver} Agreement")

    print()


def main():
    versions = discover_versions()
    if not versions:
        print(f"No eval files found under {QA_ITEMS_DIR}")
        sys.exit(1)

    print("\nAvailable versions with eval data:\n")
    print(f"  {'#':<4} {'Version':<12} {'Human Eval':<14} {'LLM Eval Versions'}")
    print(f"  {'─' * 50}")
    for i, (name, _, has_h, llm_count) in enumerate(versions, 1):
        h_mark = "present" if has_h else "—"
        l_mark = f"{llm_count} version{'s' if llm_count != 1 else ''}" if llm_count > 0 else "—"
        print(f"  {i:<4} {name:<12} {h_mark:<14} {l_mark}")

    print()
    try:
        choice = input("Enter version number (or name) to visualize: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(0)

    selected = None
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(versions):
            selected = versions[idx]
    else:
        for v in versions:
            if v[0] == choice:
                selected = v
                break

    if selected is None:
        print(f"Invalid selection: '{choice}'")
        sys.exit(1)

    visualize(*selected)


if __name__ == "__main__":
    main()
