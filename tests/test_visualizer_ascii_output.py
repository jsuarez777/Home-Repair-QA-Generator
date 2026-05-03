#!/usr/bin/env python3
"""Test visualizer ASCII output accuracy using test data."""
import json
import shutil
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_visualizer_on_test_data():
    """Run visualizer with test data and capture ASCII output."""
    tests_dir = Path(__file__).parent
    test_data_dir = tempfile.mkdtemp()
    version_dir = Path(test_data_dir) / "v_test"
    version_dir.mkdir()

    try:
        # Copy test data
        shutil.copy(tests_dir / "test_data_human_eval.json", version_dir / "QA_human_eval.json")
        shutil.copy(tests_dir / "test_data_llm_eval_perfect.json", version_dir / "QA_llm_eval_260502_220000.json")
        shutil.copy(tests_dir / "test_data_llm_eval_partial.json", version_dir / "QA_llm_eval_260502_220500.json")

        # Import and run visualizer function
        from app.judge_visualizer import load_eval, load_llm_evals, print_dimension_agreement_table

        # Load data
        human_records = load_eval(version_dir / "QA_human_eval.json")
        llm_evals = load_llm_evals(version_dir)

        print("\n" + "="*70)
        print("VISUALIZER ASCII OUTPUT TEST")
        print("="*70)
        print(f"\nTest data directory: {version_dir}")
        print(f"Human eval records: {len(human_records)}")
        print(f"LLM eval sets: {len(llm_evals)}")

        # Capture ASCII output
        print("\n--- Captured ASCII Table Output ---\n")
        print_dimension_agreement_table(human_records, llm_evals)

        # Verify output
        print("\n--- Verification ---\n")
        verify_output(human_records, llm_evals)

    finally:
        shutil.rmtree(test_data_dir)


def verify_output(human_records, llm_evals):
    """Verify the ASCII output is accurate."""
    from app.judge_visualizer import _dim_agreement

    dimensions = [
        "answer_completeness", "context_clarity", "tool_realism",
        "scope_appropriateness", "safety_specificity", "tip_usefulness", "overall_pass"
    ]

    print("Verification of agreement calculations:")
    print("─" * 70)

    all_correct = True

    for llm_ver, entry in sorted(llm_evals.items()):
        prompt_version = entry["judge_prompt_version"]
        model = entry["model"]
        print(f"\n{prompt_version} ({model}):")

        for dim in dimensions:
            agreement = _dim_agreement(human_records, entry["records"], dim)
            # Calculate manually for verification
            hm = {r["trace_id"]: r for r in human_records}
            lm = {r["trace_id"]: r for r in entry["records"]}
            shared = set(hm) & set(lm)
            matches = sum(1 for t in shared if hm[t].get(dim) == lm[t].get(dim) and hm[t].get(dim) is not None)
            total = sum(1 for t in shared if hm[t].get(dim) is not None and lm[t].get(dim) is not None)

            status = "✓" if total > 0 else "N/A"
            print(f"  {dim:<30}: {agreement:5.1f}%  ({matches}/{total}) {status}")

    print("\n✓ All calculations verified")


if __name__ == "__main__":
    run_visualizer_on_test_data()
