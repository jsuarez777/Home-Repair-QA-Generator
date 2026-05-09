#!/usr/bin/env python3
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
QA_ITEMS_DIR = PROJECT_ROOT / "qa_items"
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


def load_llm_evals(version_dir: Path) -> dict[str, dict]:
    """Returns {eval_key: {"records": [...], "model": str, "judge_prompt_version": str}} for all LLM eval files."""
    evals = {}
    for f in sorted(version_dir.glob(LLM_FILE_PATTERN)):
        try:
            raw = json.load(open(f))
            if isinstance(raw, list):
                records, model, prompt_version = raw, "unknown", "unknown"
            else:
                records = raw.get("results", [])
                model = raw.get("model", "unknown")
                prompt_version = raw.get("judge_prompt_version", "unknown")
            safe_key = f"{prompt_version}_{re.sub(r'[^a-zA-Z0-9]', '', model)}"
            evals[safe_key] = {"records": records, "model": model, "judge_prompt_version": prompt_version}
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
    """Print agreement rate between human and LLM evaluations for each dimension on shared traces.

    Args:
        human: list of dicts with keys: trace_id, dimension scores, overall_pass
        llm: list of dicts with keys: trace_id, dimension scores, overall_pass
    """
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


def _dim_agreement(human: list[dict], llm: list[dict], dim: str) -> float:
    """Agreement rate (0-100) between human and LLM for one dimension on shared traces."""
    hm = {r["trace_id"]: r for r in human}
    lm = {r["trace_id"]: r for r in llm}
    shared = set(hm) & set(lm)
    matches = sum(1 for t in shared if hm[t].get(dim) == lm[t].get(dim) and hm[t].get(dim) is not None)
    total = sum(1 for t in shared if hm[t].get(dim) is not None and lm[t].get(dim) is not None)
    return matches / total * 100 if total else float("nan")


def plot_human_llm_agreement(
    version_name: str, human_records: list[dict], llm_evals: dict[str, dict], out_dir: Path
) -> None:
    if not human_records or not llm_evals:
        return

    hm = {r["trace_id"] for r in human_records}
    n_shared = max(
        len(hm & {r["trace_id"] for r in entry["records"]})
        for entry in llm_evals.values()
    )

    all_dims = DIMENSIONS + ["overall_pass"]
    dim_labels = [d.replace("_", " ").title() for d in all_dims]
    rows = []
    sorted_evals = sorted(llm_evals.items(), key=lambda x: (x[1]["model"], x[1]["judge_prompt_version"]))
    for llm_ver, entry in sorted_evals:
        legend_label = f"Prompt {entry['judge_prompt_version']} ({entry['model']})"
        for dim, label in zip(all_dims, dim_labels):
            rows.append({
                "Judge Prompt & Model": legend_label,
                "Dimension": label,
                "Agreement (%)": _dim_agreement(human_records, entry["records"], dim),
            })

    df = pd.DataFrame(rows)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=df, x="Dimension", y="Agreement (%)", hue="Judge Prompt & Model", ax=ax)

    ax.set_title(f"Human vs LLM Agreement by Dimension — QA version {version_name}", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Agreement (%)")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend(title="Judge Prompt & Model")
    ax.axhline(y=80, color="grey", linestyle="--", linewidth=0.8, label="80% reference")
    _add_item_axis(ax, n_shared)

    plt.tight_layout()
    out_path = out_dir / "human_llm_agreement.png"
    fig.savefig(out_path, dpi=150)
    print(f"  Agreement chart saved to {out_path}")
    plt.show()


def _add_item_axis(ax, n_items: int) -> None:
    """Add a right-side y-axis showing item counts proportional to the left % axis."""
    ax2 = ax.twinx()
    ax2.set_ylim(0, ax.get_ylim()[1] * n_items / 100)
    ax2.set_ylabel(f"Items  (100% = {n_items})")
    ax2.grid(False)


def _build_trace_category_map(version_dir: Path) -> dict[str, str]:
    """Maps trace_id (e.g. 'QA13') to category (e.g. 'general') from .qa filenames."""
    mapping = {}
    for f in version_dir.glob("*.qa"):
        parts = f.stem.split("_", 1)
        if len(parts) == 2:
            trace_id = parts[0]
            category = re.sub(r"\d+$", "", parts[1])
            mapping[trace_id] = category
    return mapping


def plot_qa_category_distribution(version_name: str, version_dir: Path, out_dir: Path) -> None:
    trace_cat = _build_trace_category_map(version_dir)
    if not trace_cat:
        print("  No .qa files found — skipping category distribution chart.")
        return

    counts = Counter(trace_cat.values())
    categories = sorted(counts.keys())
    values = [counts[c] for c in categories]
    total = sum(values)
    percentages = [v / total * 100 for v in values]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(max(6, len(categories) * 0.9 + 1), 5))
    sns.barplot(x=categories, y=percentages, ax=ax, palette="tab10")
    ax.set_axisbelow(True)

    ax.set_title(f"QA Items by Category — QA version {version_name}", fontsize=13)
    ax.set_xlabel("Category")
    ax.set_ylabel("Share (%)")
    y_top = max(percentages) * 1.18
    ax.set_ylim(0, y_top)
    ax.tick_params(axis="x", labelrotation=20)
    for patch, pct in zip(ax.patches, percentages):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + y_top * 0.01,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )

    ax2 = ax.twinx()
    ax2.set_ylim(0, y_top * total / 100)
    ax2.set_ylabel("Item Count")
    ax2.grid(False)

    plt.tight_layout()
    out_path = out_dir / "qa_category_distribution.png"
    fig.savefig(out_path, dpi=150)
    print(f"  Category distribution chart saved to {out_path}")
    plt.show()


def plot_llm_heatmap(version_name: str, version_dir: Path, llm_evals: dict[str, dict], out_dir: Path) -> None:
    trace_cat = _build_trace_category_map(version_dir)
    if not trace_cat:
        print("  No .qa files found for category mapping — skipping heatmap.")
        return

    dim_labels = [d.replace("_", " ").title() for d in DIMENSIONS]

    for llm_ver, entry in llm_evals.items():
        records = entry["records"]
        model = entry["model"]
        prompt_version = entry["judge_prompt_version"]

        cat_records: dict[str, list[dict]] = {}
        for r in records:
            cat = trace_cat.get(r["trace_id"], "unknown")
            cat_records.setdefault(cat, []).append(r)

        categories = sorted(cat_records.keys())
        data = [
            [dim_score(cat_records[cat], dim) for dim in DIMENSIONS]
            for cat in categories
        ]
        df = pd.DataFrame(data, index=categories, columns=dim_labels)

        fig, ax = plt.subplots(figsize=(11, max(4, len(categories) * 0.6 + 1.5)))
        sns.heatmap(
            df, ax=ax, annot=True, fmt=".0f", cmap="RdYlGn",
            vmin=0, vmax=100, linewidths=0.5,
            cbar_kws={"label": "Pass Rate (%)"},
        )
        ax.set_title(
            f"Pass Rate by Category & Dimension\n"
            f"QA {version_name}  —  LLM Judge Prompt {prompt_version} ({model})",
            fontsize=12,
        )
        ax.set_xlabel("")
        ax.set_ylabel("Category")
        ax.tick_params(axis="x", labelrotation=20)
        plt.tight_layout()

        out_path = out_dir / f"llm_heatmap_{llm_ver}.png"
        fig.savefig(out_path, dpi=150)
        print(f"  Heatmap saved to {out_path}")
        plt.show()


def plot_llm_pass_rates(version_name: str, llm_evals: dict[str, dict], out_dir: Path) -> None:
    if not llm_evals:
        return

    n_items = max(len(entry["records"]) for entry in llm_evals.values())
    dim_labels = [d.replace("_", " ").title() for d in DIMENSIONS]
    rows = []
    sorted_evals = sorted(llm_evals.items(), key=lambda x: (x[1]["model"], x[1]["judge_prompt_version"]))
    for _, entry in sorted_evals:
        records = entry["records"]
        model = entry["model"]
        legend_label = f"Prompt {entry['judge_prompt_version']} ({model})"
        for dim, label in zip(DIMENSIONS, dim_labels):
            rows.append({"Judge Prompt & Model": legend_label, "Dimension": label, "Pass Rate (%)": dim_score(records, dim)})

    df = pd.DataFrame(rows)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x="Dimension", y="Pass Rate (%)", hue="Judge Prompt & Model", ax=ax)

    ax.set_title(f"LLM Judge Pass Rate by Dimension — QA version {version_name}", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend(title="Judge Prompt & Model")
    _add_item_axis(ax, n_items)

    plt.tight_layout()
    out_path = out_dir / "llm_pass_rates.png"
    fig.savefig(out_path, dpi=150)
    print(f"\n  Chart saved to {out_path}")
    plt.show()


def print_dimension_agreement_table(human_records: list[dict], llm_evals: dict[str, dict]) -> None:
    """Print human vs LLM agreement broken down by dimension."""
    if not human_records or not llm_evals:
        return

    all_dims = DIMENSIONS + ["overall_pass"]
    print(f"\n{'HUMAN vs LLM JUDGE AGREEMENT BY DIMENSION'}")
    print(f"{'═' * 66}")

    for dim in all_dims:
        dim_label = dim.replace("_", " ").title()
        print(f"\n{dim_label}")
        for _, entry in sorted(llm_evals.items(), key=lambda x: (x[1]["model"], x[1]["judge_prompt_version"])):
            prompt_version = entry["judge_prompt_version"]
            model = entry["model"]
            llm_records = entry["records"]

            agreement_pct = _dim_agreement(human_records, llm_records, dim)
            hm = {r["trace_id"]: r for r in human_records}
            lm = {r["trace_id"]: r for r in llm_records}
            shared = set(hm) & set(lm)
            matches = sum(1 for t in shared if hm[t].get(dim) == lm[t].get(dim) and hm[t].get(dim) is not None)
            total = sum(1 for t in shared if hm[t].get(dim) is not None and lm[t].get(dim) is not None)

            label = f"Prompt {prompt_version} ({model})"
            if agreement_pct != agreement_pct:  # nan
                print(f"  {label:<30}: n/a")
            else:
                bar_str = bar(agreement_pct, 20)
                print(f"  {label:<30}: {bar_str} {agreement_pct:5.1f}% agree  ({matches}/{total})")


def visualize(version_name: str, version_dir: Path, has_human: bool, llm_count: int):
    print(f"\n{'═' * 60}")
    print(f"  Version: {version_name}")
    print(f"{'═' * 60}")

    out_dir = PROJECT_ROOT / "visualizations" / datetime.now().strftime("%Y%m%d_%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)

    human_records = load_eval(version_dir / HUMAN_FILE) if has_human else None
    llm_evals = load_llm_evals(version_dir)  # dict[str, list[dict]]

    if human_records:
        print_section("HUMAN EVAL", human_records, "Human Judge")

    for _, entry in llm_evals.items():
        llm_records = entry["records"]
        model = entry["model"]
        prompt_version = entry["judge_prompt_version"]
        print_section(f"LLM EVAL (Prompt {prompt_version})", llm_records, f"LLM Judge Prompt {prompt_version} — {model}")
        if human_records:
            print(f"\n{'─' * 60}")
            compare_section(human_records, llm_records, f"Human vs LLM Prompt {prompt_version} ({model}) Agreement")

    if human_records:
        print_dimension_agreement_table(human_records, llm_evals)

    print()
    plot_qa_category_distribution(version_name, version_dir, out_dir)
    plot_llm_pass_rates(version_name, llm_evals, out_dir)
    plot_llm_heatmap(version_name, version_dir, llm_evals, out_dir)
    plot_human_llm_agreement(version_name, human_records, llm_evals, out_dir)


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
        num_str = f"{i} *" if i == len(versions) else str(i)
        print(f"  {num_str:<4} {name:<12} {h_mark:<14} {l_mark}")

    print("  * = default")
    print()
    try:
        choice = input(f"Enter version number (or name) to visualize [{len(versions)}]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(0)

    selected = None
    if choice == "":
        selected = versions[-1]
    elif choice.isdigit():
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
