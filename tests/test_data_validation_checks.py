import json
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from data_validation_checks import _validate_text, dim_sanity_check
from qa_item import QAItem

VALID = {
    "question": "What should I do when the hydraulic pump fails?",
    "answer": "A" * 100,
    "equipment_problem": "Hydraulic pump failure",
    "tools_required": ["wrench", "oil"],
    "steps": ["Step one", "Step two", "Step three"],
    "safety_info": "W" * 80,
    "tips": ["Check oil level regularly"],
}


def make(**overrides) -> str:
    return json.dumps({**VALID, **overrides})


# ---------------------------------------------------------------------------
# Check 1 — json_valid
# ---------------------------------------------------------------------------

def test_check1_invalid_json():
    results, failures, _ = _validate_text("not valid json {{")
    assert results[0] == "FAIL"
    assert all(r == "----" for r in results[1:])
    assert failures == {"json_valid": True}


def test_check1_pass():
    results, failures, _ = _validate_text(make())
    assert results[0] == "PASS"
    assert "json_valid" not in failures


# ---------------------------------------------------------------------------
# Check 2 — all_fields_present
# ---------------------------------------------------------------------------

def test_check2_one_missing_field():
    data = {**VALID}
    del data["safety_info"]
    results, failures, _ = _validate_text(json.dumps(data))
    assert results[1] == "FAIL"
    assert failures.get("all_fields_present") == ["safety_info"]


def test_check2_multiple_missing_fields():
    data = {**VALID}
    del data["safety_info"]
    del data["answer"]
    results, failures, _ = _validate_text(json.dumps(data))
    assert results[1] == "FAIL"
    assert set(failures.get("all_fields_present", [])) == {"safety_info", "answer"}


# ---------------------------------------------------------------------------
# Check 3 — non_empty_strings
# ---------------------------------------------------------------------------

def test_check3_one_string_too_short():
    results, failures, _ = _validate_text(make(answer="too short"))
    assert results[2] == "FAIL"
    assert failures.get("non_empty_strings") == ["answer"]


def test_check3_multiple_strings_too_short():
    results, failures, _ = _validate_text(make(answer="too short", safety_info="also short"))
    assert results[2] == "FAIL"
    assert set(failures.get("non_empty_strings", [])) == {"answer", "safety_info"}


# ---------------------------------------------------------------------------
# Check 4 — sufficient_steps
# ---------------------------------------------------------------------------

def test_check4_too_few_steps():
    results, failures, _ = _validate_text(make(steps=["only one", "only two"]))
    assert results[3] == "FAIL"
    assert failures.get("sufficient_steps") is True


def test_check4_empty_steps():
    results, failures, _ = _validate_text(make(steps=[]))
    assert results[3] == "FAIL"
    assert failures.get("sufficient_steps") is True


# ---------------------------------------------------------------------------
# Check 5 — tools_present
# ---------------------------------------------------------------------------

def test_check5_no_tools():
    results, failures, _ = _validate_text(make(tools_required=[]))
    assert results[4] == "FAIL"
    assert failures.get("tools_present") is True


# ---------------------------------------------------------------------------
# Check 6 — tips_present
# ---------------------------------------------------------------------------

def test_check6_no_tips():
    results, failures, _ = _validate_text(make(tips=[]))
    assert results[5] == "FAIL"
    assert failures.get("tips_present") is True


# ---------------------------------------------------------------------------
# Check 7 — no_vague_phrases
# ---------------------------------------------------------------------------

def test_check7_vague_phrase_in_tip():
    results, failures, _ = _validate_text(make(tips=["be careful"]))
    assert results[6] == "FAIL"
    assert failures.get("no_vague_phrases") is True


def test_check7_vague_phrase_in_safety_info():
    results, failures, _ = _validate_text(make(safety_info="stay safe" + " " * 80))
    assert results[6] == "FAIL"
    assert failures.get("no_vague_phrases") is True


# ---------------------------------------------------------------------------
# Multiple failures
# ---------------------------------------------------------------------------

def test_multiple_pydantic_failures():
    results, failures, _ = _validate_text(make(
        steps=["only one"],
        tools_required=[],
        tips=[],
    ))
    assert results[3] == "FAIL"
    assert results[4] == "FAIL"
    assert results[5] == "FAIL"
    assert failures.get("sufficient_steps") is True
    assert failures.get("tools_present") is True
    assert failures.get("tips_present") is True


def test_multiple_string_and_list_failures():
    results, failures, _ = _validate_text(make(
        answer="short",
        safety_info="short",
        steps=["only one", "only two"],
    ))
    assert results[2] == "FAIL"
    assert results[3] == "FAIL"
    assert set(failures.get("non_empty_strings", [])) == {"answer", "safety_info"}
    assert failures.get("sufficient_steps") is True


def test_missing_and_string_failures():
    data = {**VALID, "answer": "short"}
    del data["safety_info"]
    results, failures, _ = _validate_text(json.dumps(data))
    assert results[1] == "FAIL"
    assert results[2] == "FAIL"
    assert "safety_info" in failures.get("all_fields_present", [])
    assert "answer" in failures.get("non_empty_strings", [])


# ---------------------------------------------------------------------------
# All passing
# ---------------------------------------------------------------------------

def test_all_checks_pass():
    results, failures, errors = _validate_text(make())
    assert all(r == "PASS" for r in results)
    assert failures == {}
    assert errors == []
