import pytest
from pydantic import ValidationError
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.qa_item import QAItem

VALID = {
    "question": "What should I do when the hydraulic pump fails?",
    "answer": "A" * 100,
    "equipment_problem": "Hydraulic failure",
    "tools_required": ["wrench"],
    "steps": ["Step one", "Step two", "Step three"],
    "safety_info": "W" * 80,
    "tips": ["Check oil level"],
}


def make(**overrides):
    return {**VALID, **overrides}


def test_valid_item():
    item = QAItem(**VALID)
    assert item.question == VALID["question"]
    assert item.answer == VALID["answer"]
    assert item.equipment_problem == VALID["equipment_problem"]
    assert item.tools_required == VALID["tools_required"]
    assert item.steps == VALID["steps"]
    assert item.safety_info == VALID["safety_info"]
    assert item.tips == VALID["tips"]


# --- question: min length 20 ---

def test_question_too_short():
    with pytest.raises(ValidationError):
        QAItem(**make(question="Too short"))


def test_question_exact_min():
    QAItem(**make(question="Q" * 20))


# --- answer: min length 100 ---

def test_answer_too_short():
    with pytest.raises(ValidationError):
        QAItem(**make(answer="Too short"))


def test_answer_exact_min():
    QAItem(**make(answer="A" * 100))


# --- equipment_problem: min length 10 ---

def test_equipment_problem_too_short():
    with pytest.raises(ValidationError):
        QAItem(**make(equipment_problem="Short"))


def test_equipment_problem_exact_min():
    QAItem(**make(equipment_problem="E" * 10))


# --- tools_required: min 1 item ---

def test_tools_required_empty():
    with pytest.raises(ValidationError):
        QAItem(**make(tools_required=[]))


def test_tools_required_one_item():
    QAItem(**make(tools_required=["hammer"]))


# --- steps: min 3 items ---

def test_steps_too_few():
    with pytest.raises(ValidationError):
        QAItem(**make(steps=["only one", "only two"]))


def test_steps_exact_min():
    QAItem(**make(steps=["step 1", "step 2", "step 3"]))


# --- safety_info: min length 80 ---

def test_safety_info_too_short():
    with pytest.raises(ValidationError):
        QAItem(**make(safety_info="Too short"))


def test_safety_info_exact_min():
    QAItem(**make(safety_info="S" * 80))


# --- tips: min 1 item ---

def test_tips_empty():
    with pytest.raises(ValidationError):
        QAItem(**make(tips=[]))


def test_tips_one_item():
    QAItem(**make(tips=["Wear gloves"]))
