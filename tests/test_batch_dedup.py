import json
import sys
import os
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from generate_qa_set import batch_dedup_check

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


# Embeddings where index 0 and 1 are near-identical (sim ~1.0),
# and index 2 is distinct (orthogonal).
_SIMILAR   = [1.0, 0.0, 0.0]
_SIMILAR2  = [0.99, 0.1, 0.0]   # cosine ~0.995 with _SIMILAR
_DISTINCT  = [0.0, 0.0, 1.0]    # cosine ~0.0 with _SIMILAR


def _make_mock_st(embeddings: list[list[float]]):
    """Return a patched SentenceTransformer that returns the given embeddings."""
    import math

    def _cos_sim(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        mag = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b))
        return dot / mag if mag else 0.0

    mock_model = MagicMock()
    mock_model.encode.return_value = embeddings

    mock_cos_sim = MagicMock(side_effect=lambda a, b: _cos_sim(a, b))

    return mock_model, mock_cos_sim


def _patch(mock_model, mock_cos_sim):
    return [
        patch("generate_qa_set.SentenceTransformer", return_value=mock_model),
        patch("generate_qa_set.cos_sim", mock_cos_sim),
    ]


# ---------------------------------------------------------------------------
# Tests that don't reach the embedding path
# ---------------------------------------------------------------------------

def test_no_files_does_nothing(tmp_path):
    """Folder with fewer than 2 files returns immediately without error."""
    batch_dedup_check(tmp_path, client=None)
    assert list(tmp_path.glob("*.qa")) == []


def test_single_file_does_nothing(tmp_path):
    """Single .qa file is never removed."""
    _write_qa(tmp_path, "QA1_hvac1.qa")
    batch_dedup_check(tmp_path, client=None)
    assert len(list(tmp_path.glob("*.qa"))) == 1


# ---------------------------------------------------------------------------
# Tests that exercise the embedding path (sentence_transformers mocked)
# ---------------------------------------------------------------------------

def test_distinct_items_are_kept(tmp_path):
    """Two semantically different questions are both kept."""
    _write_qa(tmp_path, "QA1_hvac1.qa")
    _write_qa(tmp_path, "QA2_plumbing1.qa", {"question": DISTINCT_QUESTION})

    mock_model, mock_cos_sim = _make_mock_st([_SIMILAR, _DISTINCT])
    with patch("generate_qa_set.SentenceTransformer", return_value=mock_model), \
         patch("generate_qa_set.cos_sim", mock_cos_sim):
        batch_dedup_check(tmp_path, client=None)

    assert len(list(tmp_path.glob("*.qa"))) == 2


def test_near_duplicate_is_moved(tmp_path):
    """Two near-identical questions: the second file is moved to duplicates/."""
    _write_qa(tmp_path, "QA1_hvac1.qa")
    _write_qa(tmp_path, "QA2_hvac2.qa", {"question": NEAR_DUPLICATE_QUESTION})

    mock_model, mock_cos_sim = _make_mock_st([_SIMILAR, _SIMILAR2])
    with patch("generate_qa_set.SentenceTransformer", return_value=mock_model), \
         patch("generate_qa_set.cos_sim", mock_cos_sim):
        batch_dedup_check(tmp_path, client=None)

    assert [f.name for f in tmp_path.glob("*.qa")] == ["QA1_hvac1.qa"]
    assert (tmp_path / "duplicates" / "QA2_hvac2.qa").exists()


def test_first_occurrence_is_kept(tmp_path):
    """When multiple duplicates exist, the first file is preserved."""
    _write_qa(tmp_path, "QA1_hvac1.qa")
    _write_qa(tmp_path, "QA2_hvac2.qa", {"question": NEAR_DUPLICATE_QUESTION})
    _write_qa(tmp_path, "QA3_hvac3.qa", {"question": NEAR_DUPLICATE_QUESTION})

    mock_model, mock_cos_sim = _make_mock_st([_SIMILAR, _SIMILAR2, _SIMILAR2])
    with patch("generate_qa_set.SentenceTransformer", return_value=mock_model), \
         patch("generate_qa_set.cos_sim", mock_cos_sim):
        batch_dedup_check(tmp_path, client=None)

    assert [f.name for f in tmp_path.glob("*.qa")] == ["QA1_hvac1.qa"]
    assert (tmp_path / "duplicates" / "QA2_hvac2.qa").exists()
    assert (tmp_path / "duplicates" / "QA3_hvac3.qa").exists()


def test_invalid_qa_file_is_skipped(tmp_path):
    """A malformed .qa file is skipped without crashing."""
    _write_qa(tmp_path, "QA1_hvac1.qa")
    (tmp_path / "QA2_bad.qa").write_text("not valid json{{{{")
    _write_qa(tmp_path, "QA3_hvac2.qa", {"question": NEAR_DUPLICATE_QUESTION})

    # Only QA1 and QA3 are valid; embeddings for those two
    mock_model, mock_cos_sim = _make_mock_st([_SIMILAR, _SIMILAR2])
    with patch("generate_qa_set.SentenceTransformer", return_value=mock_model), \
         patch("generate_qa_set.cos_sim", mock_cos_sim):
        batch_dedup_check(tmp_path, client=None)  # must not raise


def test_distinct_third_item_survives(tmp_path):
    """Two near-dupes collapse to one; a distinct third item is kept."""
    _write_qa(tmp_path, "QA1_hvac1.qa")
    _write_qa(tmp_path, "QA2_hvac2.qa", {"question": NEAR_DUPLICATE_QUESTION})
    _write_qa(tmp_path, "QA3_plumbing1.qa", {"question": DISTINCT_QUESTION})

    mock_model, mock_cos_sim = _make_mock_st([_SIMILAR, _SIMILAR2, _DISTINCT])
    with patch("generate_qa_set.SentenceTransformer", return_value=mock_model), \
         patch("generate_qa_set.cos_sim", mock_cos_sim):
        batch_dedup_check(tmp_path, client=None)

    remaining = {f.name for f in tmp_path.glob("*.qa")}
    assert "QA1_hvac1.qa" in remaining
    assert "QA3_plumbing1.qa" in remaining
    assert "QA2_hvac2.qa" not in remaining
    assert (tmp_path / "duplicates" / "QA2_hvac2.qa").exists()
