#!/usr/bin/env python3
"""Test suite for judge_visualizer agreement calculations."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.judge_visualizer import _dim_agreement


def create_test_human_records(agreement_pattern: dict) -> list[dict]:
    """Create human eval records with specific agreement patterns.

    Args:
        agreement_pattern: dict mapping dimension -> (num_agree, num_disagree)
    """
    dimensions = [
        "answer_completeness", "context_clarity", "tool_realism",
        "scope_appropriateness", "safety_specificity", "tip_usefulness", "overall_pass"
    ]
    records = []

    # Create 10 items
    for i in range(10):
        record = {"trace_id": f"QA{i}"}
        for dim in dimensions:
            record[dim] = 1  # All 1s for human
        records.append(record)

    return records


def create_test_llm_records(human_records: list[dict], agreement_pattern: dict) -> list[dict]:
    """Create LLM eval records to match agreement pattern.

    Args:
        human_records: Human evaluation records
        agreement_pattern: dict mapping dimension -> agreement_percentage (0-100)
    """
    dimensions = [
        "answer_completeness", "context_clarity", "tool_realism",
        "scope_appropriateness", "safety_specificity", "tip_usefulness", "overall_pass"
    ]
    records = []

    for i, human_rec in enumerate(human_records):
        llm_rec = {"trace_id": human_rec["trace_id"]}
        for dim in dimensions:
            percent = agreement_pattern.get(dim, 50)
            # Agree if i < percent/10
            if i < int(percent / 10):
                llm_rec[dim] = human_rec[dim]
            else:
                llm_rec[dim] = 1 - human_rec[dim]
        records.append(llm_rec)

    return records


def test_100_percent_agreement():
    """Test case: 100% agreement on all dimensions."""
    print("\n" + "="*70)
    print("TEST: 100% Agreement")
    print("="*70)

    human = create_test_human_records({})
    llm = create_test_llm_records(human, {
        "answer_completeness": 100,
        "context_clarity": 100,
        "tool_realism": 100,
        "scope_appropriateness": 100,
        "safety_specificity": 100,
        "tip_usefulness": 100,
        "overall_pass": 100,
    })

    dimensions = [
        "answer_completeness", "context_clarity", "tool_realism",
        "scope_appropriateness", "safety_specificity", "tip_usefulness", "overall_pass"
    ]

    print("\nDimension                      Agreement")
    print("─" * 50)
    all_pass = True
    for dim in dimensions:
        agreement = _dim_agreement(human, llm, dim)
        expected = 100.0
        status = "✓" if abs(agreement - expected) < 0.1 else "✗"
        print(f"{dim:<30}: {agreement:5.1f}%  {status}")
        if abs(agreement - expected) >= 0.1:
            all_pass = False

    if all_pass:
        print("\n✓ TEST PASSED: All dimensions at 100%")
    else:
        print("\n✗ TEST FAILED")

    return all_pass


def test_zero_percent_agreement():
    """Test case: 0% agreement on all dimensions."""
    print("\n" + "="*70)
    print("TEST: 0% Agreement")
    print("="*70)

    human = create_test_human_records({})
    llm = create_test_llm_records(human, {
        "answer_completeness": 0,
        "context_clarity": 0,
        "tool_realism": 0,
        "scope_appropriateness": 0,
        "safety_specificity": 0,
        "tip_usefulness": 0,
        "overall_pass": 0,
    })

    dimensions = [
        "answer_completeness", "context_clarity", "tool_realism",
        "scope_appropriateness", "safety_specificity", "tip_usefulness", "overall_pass"
    ]

    print("\nDimension                      Agreement")
    print("─" * 50)
    all_pass = True
    for dim in dimensions:
        agreement = _dim_agreement(human, llm, dim)
        expected = 0.0
        status = "✓" if abs(agreement - expected) < 0.1 else "✗"
        print(f"{dim:<30}: {agreement:5.1f}%  {status}")
        if abs(agreement - expected) >= 0.1:
            all_pass = False

    if all_pass:
        print("\n✓ TEST PASSED: All dimensions at 0%")
    else:
        print("\n✗ TEST FAILED")

    return all_pass


def test_50_percent_agreement():
    """Test case: 50% agreement on all dimensions."""
    print("\n" + "="*70)
    print("TEST: 50% Agreement")
    print("="*70)

    human = create_test_human_records({})
    llm = create_test_llm_records(human, {
        "answer_completeness": 50,
        "context_clarity": 50,
        "tool_realism": 50,
        "scope_appropriateness": 50,
        "safety_specificity": 50,
        "tip_usefulness": 50,
        "overall_pass": 50,
    })

    dimensions = [
        "answer_completeness", "context_clarity", "tool_realism",
        "scope_appropriateness", "safety_specificity", "tip_usefulness", "overall_pass"
    ]

    print("\nDimension                      Agreement")
    print("─" * 50)
    all_pass = True
    for dim in dimensions:
        agreement = _dim_agreement(human, llm, dim)
        expected = 50.0
        status = "✓" if abs(agreement - expected) < 1.0 else "✗"
        print(f"{dim:<30}: {agreement:5.1f}%  {status}")
        if abs(agreement - expected) >= 1.0:
            all_pass = False

    if all_pass:
        print("\n✓ TEST PASSED: All dimensions near 50%")
    else:
        print("\n✗ TEST FAILED")

    return all_pass


def test_per_dimension_variation():
    """Test case: Different agreement rates per dimension."""
    print("\n" + "="*70)
    print("TEST: Per-Dimension Variation")
    print("="*70)

    human = create_test_human_records({})
    llm = create_test_llm_records(human, {
        "answer_completeness": 100,  # Perfect agreement
        "context_clarity": 80,
        "tool_realism": 60,
        "scope_appropriateness": 40,
        "safety_specificity": 20,
        "tip_usefulness": 0,  # No agreement
        "overall_pass": 50,
    })

    expectations = {
        "answer_completeness": 100.0,
        "context_clarity": 80.0,
        "tool_realism": 60.0,
        "scope_appropriateness": 40.0,
        "safety_specificity": 20.0,
        "tip_usefulness": 0.0,
        "overall_pass": 50.0,
    }

    print("\nDimension                      Expected  Actual    Status")
    print("─" * 65)
    all_pass = True
    for dim, expected in expectations.items():
        actual = _dim_agreement(human, llm, dim)
        status = "✓" if abs(actual - expected) < 1.0 else "✗"
        print(f"{dim:<30}: {expected:5.1f}%  {actual:5.1f}%  {status}")
        if abs(actual - expected) >= 1.0:
            all_pass = False

    if all_pass:
        print("\n✓ TEST PASSED: All dimensions match expected agreement")
    else:
        print("\n✗ TEST FAILED")

    return all_pass


if __name__ == "__main__":
    results = [
        test_100_percent_agreement(),
        test_zero_percent_agreement(),
        test_50_percent_agreement(),
        test_per_dimension_variation(),
    ]

    print("\n" + "="*70)
    print(f"SUMMARY: {sum(results)}/{len(results)} tests passed")
    print("="*70)

    sys.exit(0 if all(results) else 1)
