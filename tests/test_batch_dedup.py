import json
import sys
import os
import pytest

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.data_validation_checks import batch_dedup_check

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_BASE = {
    "question": "What should I do when the hydraulic pump fails?",
    "answer": "A" * 100,
    "equipment_problem": "Hydraulic failure",
    "tools_required": ["wrench"],
    "steps": ["Step one", "Step two", "Step three"],
    "safety_info": "W" * 80,
    "tips": ["Check oil level"],
}

NEAR_DUPLICATE_QUESTION = "What should I do when my hydraulic pump fails?"
DISTINCT_QUESTION = "How do I unclog a kitchen sink drain that drains very slowly?"


def _write_qa(tmp_path, filename, overrides=None):
    data = {**VALID_BASE, **(overrides or {})}
    f = tmp_path / filename
    f.write_text(json.dumps(data))
    return f


# ---------------------------------------------------------------------------
# Tests for Jaccard-based deduplication
# ---------------------------------------------------------------------------

def test_no_files_does_nothing(tmp_path):
    """Folder with fewer than 2 files returns immediately without error."""
    batch_dedup_check(tmp_path)
    assert list(tmp_path.glob("*.qa")) == []


def test_single_file_does_nothing(tmp_path):
    """Single .qa file is never removed."""
    _write_qa(tmp_path, "QA1_hvac1.qa")
    batch_dedup_check(tmp_path)
    assert len(list(tmp_path.glob("*.qa"))) == 1


def test_distinct_items_are_kept(tmp_path):
    """Two semantically different questions are both kept."""
    _write_qa(tmp_path, "QA1_hvac1.qa")
    _write_qa(tmp_path, "QA2_plumbing1.qa", {"question": DISTINCT_QUESTION})

    batch_dedup_check(tmp_path)
    assert len(list(tmp_path.glob("*.qa"))) == 2


def test_near_duplicate_is_moved(tmp_path):
    """Two near-identical questions: the second file is moved to duplicates/."""
    _write_qa(tmp_path, "QA1_hvac1.qa")
    _write_qa(tmp_path, "QA2_hvac2.qa", {"question": NEAR_DUPLICATE_QUESTION})

    batch_dedup_check(tmp_path)

    assert [f.name for f in tmp_path.glob("*.qa")] == ["QA1_hvac1.qa"]
    assert (tmp_path / "duplicates" / "QA2_hvac2.qa").exists()


def test_first_occurrence_is_kept(tmp_path):
    """When multiple duplicates exist, the first file is preserved."""
    _write_qa(tmp_path, "QA1_hvac1.qa")
    _write_qa(tmp_path, "QA2_hvac2.qa", {"question": NEAR_DUPLICATE_QUESTION})
    _write_qa(tmp_path, "QA3_hvac3.qa", {"question": NEAR_DUPLICATE_QUESTION})

    batch_dedup_check(tmp_path)

    assert [f.name for f in tmp_path.glob("*.qa")] == ["QA1_hvac1.qa"]
    assert (tmp_path / "duplicates" / "QA2_hvac2.qa").exists()
    assert (tmp_path / "duplicates" / "QA3_hvac3.qa").exists()


def test_invalid_qa_file_is_skipped(tmp_path):
    """A malformed .qa file is skipped without crashing."""
    _write_qa(tmp_path, "QA1_hvac1.qa")
    (tmp_path / "QA2_bad.qa").write_text("not valid json{{{{")
    _write_qa(tmp_path, "QA3_hvac2.qa", {"question": NEAR_DUPLICATE_QUESTION})

    batch_dedup_check(tmp_path)  # must not raise


def test_distinct_third_item_survives(tmp_path):
    """Two near-dupes collapse to one; a distinct third item is kept."""
    _write_qa(tmp_path, "QA1_hvac1.qa")
    _write_qa(tmp_path, "QA2_hvac2.qa", {"question": NEAR_DUPLICATE_QUESTION})
    _write_qa(tmp_path, "QA3_plumbing1.qa", {"question": DISTINCT_QUESTION})

    batch_dedup_check(tmp_path)

    remaining = {f.name for f in tmp_path.glob("*.qa")}
    assert "QA1_hvac1.qa" in remaining
    assert "QA3_plumbing1.qa" in remaining
    assert "QA2_hvac2.qa" not in remaining
    assert (tmp_path / "duplicates" / "QA2_hvac2.qa").exists()
