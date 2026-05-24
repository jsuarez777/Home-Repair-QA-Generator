#!/usr/bin/env python3
import argparse
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
        model = entry["model"]
        prompt_version = entry["judge_prompt_version"]
        for dim, label in zip(all_dims, dim_labels):
            rows.append({
                "Dimension": label,
                "Model": model,
                "Prompt Version": prompt_version,
                "Agreement (%)": _dim_agreement(human_records, entry["records"], dim),
            })

    df = pd.DataFrame(rows)

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.grid(axis="y", alpha=0.3)

    dimensions = dim_labels
    n_dims = len(dimensions)

    # Get all (model, version) pairs in order
    eval_pairs = [(row["Model"], row["Prompt Version"]) for _, row in df.drop_duplicates(subset=["Model", "Prompt Version"]).iterrows()]
    eval_pairs = sorted(set(eval_pairs), key=lambda x: (x[0], x[1]))

    n_evals = len(eval_pairs)
    bar_width = 0.8 / n_evals if n_evals > 1 else 0.6

    # Assign unique colors to each (model, version) pair
    palette = sns.color_palette("husl", n_evals)
    eval_colors = {pair: palette[idx] for idx, pair in enumerate(eval_pairs)}

    # Track where models change for dividers
    model_change_indices = [0]
    for eval_idx in range(1, len(eval_pairs)):
        if eval_pairs[eval_idx][0] != eval_pairs[eval_idx-1][0]:
            model_change_indices.append(eval_idx)
    model_change_indices.append(len(eval_pairs))

    # Plot bars for each (model, version) pair
    for eval_idx, (model, version) in enumerate(eval_pairs):
        agreements = []

        for dim in dimensions:
            dim_data = df[(df["Dimension"] == dim) & (df["Model"] == model) & (df["Prompt Version"] == version)]
            if len(dim_data) > 0:
                agreements.append(float(dim_data.iloc[0]["Agreement (%)"]))
            else:
                agreements.append(0.0)

        # Calculate x positions for this eval's bars
        if n_evals > 1:
            x_offset = (eval_idx - n_evals / 2 + 0.5) * bar_width
        else:
            x_offset = 0
        x_positions = [float(i) + x_offset for i in range(n_dims)]

        # Use (model, version) for label and color
        label = f"{model} ({version})"
        bars = ax.bar(x_positions, agreements, bar_width, label=label, color=eval_colors[(model, version)])

    # Add vertical dividers between model groups for each dimension
    for model_boundary_idx in model_change_indices[1:-1]:
        # Calculate the x position between two model groups
        x_before = (model_boundary_idx - 1 - n_evals / 2 + 0.5) * bar_width + bar_width / 2
        x_after = (model_boundary_idx - n_evals / 2 + 0.5) * bar_width - bar_width / 2
        x_divider = (x_before + x_after) / 2

        for dim_idx in range(n_dims):
            x_pos = dim_idx + x_divider
            ax.axvline(x=x_pos, color="gray", linestyle="-", linewidth=1.5, alpha=0.4)

    ax.set_xlabel("")
    ax.set_ylabel("Agreement (%)")
    ax.set_title(f"Human vs LLM Agreement by Dimension — QA version {version_name}", fontsize=13)
    ax.set_xticks(range(n_dims))
    ax.set_xticklabels(dimensions, rotation=20)
    ax.set_ylim(0, 105)
    ax.legend(title="Model & Version", loc="lower right")
    ax.axhline(y=80, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)

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


def plot_llm_heatmap_from_records(dataset_name: str, records: list[dict], model: str, prompt_version: str, out_dir: Path, jsonl_file: Path = None) -> None:
    """Heatmap for JSONL data using category field from source JSONL."""
    # Build category mapping from source JSONL file
    cat_map = {}
    if jsonl_file and jsonl_file.exists():
        try:
            with open(jsonl_file) as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        trace_id = item.get("id")
                        category = item.get("category", "unknown")
                        if trace_id:
                            cat_map[trace_id] = category
        except Exception as e:
            print(f"  Warning: could not load categories from {jsonl_file}: {e}")

    # Group records by category
    cat_records: dict[str, list[dict]] = {}
    for r in records:
        trace_id = r.get("trace_id")
        cat = cat_map.get(trace_id, "unknown") if trace_id else "unknown"
        cat_records.setdefault(cat, []).append(r)

    if not cat_records or all(k == "unknown" for k in cat_records.keys()):
        print("  No category information found — skipping heatmap.")
        return

    dim_labels = [d.replace("_", " ").title() for d in DIMENSIONS]
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
        f"Pass Rate by Category & Dimension — {dataset_name}\n"
        f"LLM Judge Prompt {prompt_version} ({model})",
        fontsize=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Category")
    ax.tick_params(axis="x", labelrotation=20)
    plt.tight_layout()

    out_path = out_dir / "llm_heatmap.png"
    fig.savefig(out_path, dpi=150)
    print(f"  Heatmap saved to {out_path}")
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


def visualize(version_name: str, version_dir: Path, has_human: bool, llm_count: int, charts: list[str] = None):
    if charts is None:
        charts = ["all"]

    charts = [c.lower() for c in charts]
    generate_all = "all" in charts

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
    if generate_all or "category" in charts:
        plot_qa_category_distribution(version_name, version_dir, out_dir)
    if generate_all or "pass-rates" in charts:
        plot_llm_pass_rates(version_name, llm_evals, out_dir)
    if generate_all or "heatmap" in charts:
        plot_llm_heatmap(version_name, version_dir, llm_evals, out_dir)
    if generate_all or "agreement" in charts:
        plot_human_llm_agreement(version_name, human_records, llm_evals, out_dir)


def discover_jsonl_evals():
    """Find eval files for eval.jsonl and train.jsonl in PROJECT_ROOT by reading evaluated_source field."""
    jsonl_evals = {}
    eval_files = sorted(PROJECT_ROOT.glob("QA_llm_eval_*.json"), reverse=True)

    # Group eval files by their evaluated_source
    eval_by_source = {}
    for eval_file in eval_files:
        try:
            with open(eval_file) as f:
                data = json.load(f)
            source = data.get("evaluated_source", "unknown")
            if source not in eval_by_source:
                eval_by_source[source] = []
            eval_by_source[source].append(eval_file)
        except Exception as e:
            print(f"  Warning: could not read {eval_file.name}: {e}")

    # Extract eval files for eval.jsonl and train.jsonl
    for source in ["eval.jsonl", "train.jsonl"]:
        if source in eval_by_source:
            files = eval_by_source[source]
            jsonl_evals[source] = (len(files), files[0])

    return jsonl_evals


def main():
    parser = argparse.ArgumentParser(description="Visualize QA evaluation results")
    parser.add_argument(
        "--chart",
        nargs="+",
        choices=["all", "category", "pass-rates", "heatmap", "agreement"],
        default=["all"],
        help="Specific chart(s) to generate: 'category' (category distribution), 'pass-rates' (LLM pass rates), 'heatmap' (by category & dimension), 'agreement' (human vs LLM), or 'all' (default)"
    )
    args = parser.parse_args()

    versions = discover_versions()
    jsonl_evals = discover_jsonl_evals()

    if not versions and not jsonl_evals:
        print(f"No eval files found under {QA_ITEMS_DIR} or in PROJECT_ROOT")
        sys.exit(1)

    print("\nAvailable eval data:\n")

    choice_idx = 1
    jsonl_choices = {}
    version_choices = {}

    total_items = len(jsonl_evals) + len(versions)

    # JSONL files first
    if jsonl_evals:
        print(f"  {'#':<4} {'Dataset':<12} {'LLM Eval Versions'}")
        print(f"  {'─' * 50}")
        for name in ["eval.jsonl", "train.jsonl"]:
            if name in jsonl_evals:
                llm_count, _ = jsonl_evals[name]
                l_mark = f"{llm_count} version{'s' if llm_count != 1 else ''}"
                marker = " *" if choice_idx == total_items else ""
                print(f"  {choice_idx:<4} {name:<12} {l_mark}{marker}")
                jsonl_choices[choice_idx] = name
                choice_idx += 1
        print()

    # QA versions
    if versions:
        print(f"  {'#':<4} {'Version':<12} {'Human Eval':<14} {'LLM Eval Versions'}")
        print(f"  {'─' * 50}")
        for name, _, has_h, llm_count in versions:
            h_mark = "present" if has_h else "—"
            l_mark = f"{llm_count} version{'s' if llm_count != 1 else ''}" if llm_count > 0 else "—"
            marker = " *" if choice_idx == total_items else ""
            num_str = f"{choice_idx}{marker}"
            print(f"  {num_str:<4} {name:<12} {h_mark:<14} {l_mark}")
            version_choices[choice_idx] = (name, _, has_h, llm_count)
            choice_idx += 1

    max_choice = total_items
    default_choice = max_choice
    print("  * = default")
    print()
    try:
        choice = input(f"Enter number to visualize [{default_choice}]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(0)

    selected_choice = None
    if choice == "":
        selected_choice = default_choice
    elif choice.isdigit() and 1 <= int(choice) <= max_choice:
        selected_choice = int(choice)
    else:
        print(f"Invalid selection: '{choice}'")
        sys.exit(1)

    # Handle JSONL selection
    if selected_choice in jsonl_choices:
        jsonl_name = jsonl_choices[selected_choice]
        llm_count, latest_eval = jsonl_evals[jsonl_name]
        with open(latest_eval) as f:
            eval_data = json.load(f)
        results = eval_data.get("results", [])
        model = eval_data.get("model", "unknown")
        prompt_version = eval_data.get("judge_prompt_version", "unknown")

        print(f"\n{'═' * 60}")
        print(f"  Dataset: {jsonl_name}")
        print(f"  LLM Eval File: {latest_eval.name}")
        print(f"  Model: {model}")
        print(f"{'═' * 60}\n")

        print_section("LLM EVAL", results, f"LLM Judge ({model})")

        # Generate charts for JSONL data
        out_dir = PROJECT_ROOT / "visualizations" / datetime.now().strftime("%Y%m%d_%H%M")
        out_dir.mkdir(parents=True, exist_ok=True)

        charts = args.chart
        charts = [c.lower() for c in charts]
        generate_all = "all" in charts

        llm_evals_for_chart = {f"{prompt_version}_{model}": {"records": results, "model": model, "judge_prompt_version": prompt_version}}
        if generate_all or "pass-rates" in charts:
            plot_llm_pass_rates(jsonl_name.replace(".jsonl", ""), llm_evals_for_chart, out_dir)
        if generate_all or "heatmap" in charts:
            jsonl_path = PROJECT_ROOT / jsonl_name
            plot_llm_heatmap_from_records(jsonl_name.replace(".jsonl", ""), results, model, prompt_version, out_dir, jsonl_path)
        print()
        return

    # Handle version selection
    if selected_choice in version_choices:
        selected = version_choices[selected_choice]
        visualize(*selected, charts=args.chart)
    else:
        print(f"Invalid selection: {selected_choice}")
        sys.exit(1)


if __name__ == "__main__":
    main()
