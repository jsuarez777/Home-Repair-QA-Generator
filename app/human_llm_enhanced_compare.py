#!/usr/bin/env python3
"""Compare human and enhanced LLM judge evaluations side-by-side."""

import json
import logging
import sys
import time
import webbrowser
from pathlib import Path
from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QA_ITEMS_ROOT = PROJECT_ROOT / "qa_items"
HUMAN_FILE = "QA_human_eval.json"
ENHANCED_LLM_FILE_PATTERN = "QA_llm_enhanced_eval_*.json"

_LOGS_DIR = PROJECT_ROOT / "logs"
_LOGS_DIR.mkdir(exist_ok=True)
_log_file = _LOGS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_human_llm_compare.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(_log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

DIMENSIONS = [
    "answer_completeness",
    "safety_specificity",
    "tool_realism",
    "scope_appropriateness",
    "context_clarity",
    "tip_usefulness",
]

app = Flask(__name__)

state: dict = {
    "qa_folder": None,
    "qa_files": [],
    "human_evals": {},
    "llm_evals": {},
    "shared_trace_ids": [],
}


def _trace_id(qa_file: Path) -> str:
    """QA1_hvac1.qa  ->  QA1"""
    return qa_file.stem.split("_")[0]


def _category(qa_file: Path) -> str:
    """QA1_hvac1.qa  ->  hvac"""
    import re
    parts = qa_file.stem.split("_", 1)
    return re.sub(r"\d+$", "", parts[1]) if len(parts) > 1 else "unknown"


def _load_human_eval(folder: Path) -> dict:
    eval_path = folder / HUMAN_FILE
    if not eval_path.exists():
        return {}
    try:
        items = json.loads(eval_path.read_text())
        if isinstance(items, list):
            return {e["trace_id"]: e for e in items if "trace_id" in e}
    except Exception:
        pass
    return {}


def _load_enhanced_llm_eval(folder: Path) -> dict:
    """Load the most recent enhanced LLM eval file."""
    files = sorted(folder.glob(ENHANCED_LLM_FILE_PATTERN), reverse=True)
    if not files:
        return {}
    return _load_enhanced_llm_eval_file(folder, files[0].name)


def _load_enhanced_llm_eval_file(folder: Path, filename: str) -> dict:
    """Load a specific enhanced LLM eval file."""
    filepath = folder / filename
    if not filepath.exists():
        return {}
    try:
        with open(filepath) as f:
            data = json.load(f)
        results = data.get("results", []) if isinstance(data, dict) else data
        return {r["trace_id"]: r for r in results if "trace_id" in r}
    except Exception:
        pass
    return {}


def _extract_eval_metadata(folder: Path, filename: str) -> dict:
    """Extract metadata from enhanced eval file."""
    filepath = folder / filename
    if not filepath.exists():
        return {}

    # Parse timestamp from filename: QA_llm_enhanced_eval_260503_163726.json (YYMMDD_HHMMSS)
    try:
        stem = filepath.stem  # QA_llm_enhanced_eval_260503_163726
        parts = stem.split("_")
        if len(parts) >= 5:
            date_part = parts[-2]  # 260503 (YYMMDD)
            time_part = parts[-1]  # 163726 (HHMMSS)
            # Parse YYMMDD
            yy = date_part[:2]  # 26
            mm = date_part[2:4]  # 05
            dd = date_part[4:6]  # 03
            yyyy = f"20{yy}"  # 2026
            date_str = f"{yyyy}-{mm}-{dd}"
            # Parse HHMMSS
            hh = time_part[:2]  # 16
            min_part = time_part[2:4]  # 37
            ss = time_part[4:6]  # 26
            time_str = f"{hh}:{min_part}:{ss}"
        else:
            date_str = "unknown"
            time_str = "unknown"
    except Exception:
        date_str = "unknown"
        time_str = "unknown"

    metadata = {
        "filename": filename,
        "date": date_str,
        "time": time_str,
        "model": "unknown",
        "prompt_version": "unknown",
    }

    # Try to extract model and prompt version from file content
    try:
        with open(filepath) as f:
            data = json.load(f)

        # Check if metadata is stored in the file
        if isinstance(data, dict):
            if "model" in data:
                metadata["model"] = data["model"]
            if "judge_prompt_version" in data:
                metadata["prompt_version"] = data["judge_prompt_version"]
    except:
        pass

    return metadata


def _detect_comparable_versions() -> list[dict]:
    """Return list of version dicts with both human and enhanced LLM evals."""
    if not QA_ITEMS_ROOT.exists():
        return []
    versions = []
    for d in sorted(QA_ITEMS_ROOT.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        has_human = (d / HUMAN_FILE).exists()
        if not has_human:
            continue

        enhanced_files = sorted(d.glob(ENHANCED_LLM_FILE_PATTERN), reverse=True)
        if not enhanced_files:
            continue

        qa_count = len(list(d.glob("*.qa")))

        # Create an entry for each enhanced eval file
        for enhanced_file in enhanced_files:
            metadata = _extract_eval_metadata(d, enhanced_file.name)
            versions.append({
                "version": d.name,
                "qa_count": qa_count,
                "enhanced_file": enhanced_file.name,
                "metadata": metadata,
            })
    return versions


@app.route("/")
def index():
    return LANDING_HTML


@app.route("/compare")
def compare():
    if not state["qa_folder"]:
        return '<meta http-equiv="refresh" content="0;url=/">'
    return COMPARE_HTML


@app.route("/api/versions")
def get_versions():
    return jsonify(_detect_comparable_versions())


@app.route("/api/select-version", methods=["POST"])
def select_version():
    version = (request.json or {}).get("version", "").strip()
    enhanced_file = (request.json or {}).get("enhanced_file", "").strip()
    folder = QA_ITEMS_ROOT / version
    if not folder.is_dir():
        return jsonify({"error": f"Version '{version}' not found"}), 404

    human_evals = _load_human_eval(folder)
    if not human_evals:
        return jsonify({"error": "Missing human eval data"}), 404

    # Load specific enhanced eval file if provided, otherwise load most recent
    llm_evals = _load_enhanced_llm_eval_file(folder, enhanced_file) if enhanced_file else _load_enhanced_llm_eval(folder)

    if not llm_evals:
        return jsonify({"error": "Missing enhanced LLM eval data"}), 404

    shared = set(human_evals.keys()) & set(llm_evals.keys())
    if not shared:
        return jsonify({"error": "No overlapping trace IDs"}), 404

    files = sorted(folder.glob("*.qa"), key=lambda p: int(p.stem.split("_")[0][2:] or 0))
    files = [f for f in files if _trace_id(f) in shared]

    if not files:
        return jsonify({"error": "No shared QA files"}), 404

    state["qa_folder"] = folder
    state["qa_files"] = files
    state["human_evals"] = human_evals
    state["llm_evals"] = llm_evals
    state["shared_trace_ids"] = sorted(shared)

    return jsonify({"ok": True, "version": version, "qa_count": len(files), "enhanced_file": enhanced_file})


@app.route("/api/qa/<int:idx>")
def get_qa(idx: int):
    files = state["qa_files"]
    if idx >= len(files):
        return jsonify({"done": True, "total": len(files)})

    qa_file = files[idx]
    trace_id = _trace_id(qa_file)

    try:
        qa_data = json.loads(qa_file.read_text())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    human_eval = state["human_evals"].get(trace_id, {})
    llm_eval = state["llm_evals"].get(trace_id, {})

    # Extract reasoning from the second object (if present)
    llm_reasoning = {}
    if isinstance(llm_eval, dict) and "reasoning" in llm_eval:
        llm_reasoning = llm_eval["reasoning"]

    return jsonify({
        "index": idx,
        "total": len(files),
        "trace_id": trace_id,
        "filename": qa_file.name,
        "qa": qa_data,
        "human_eval": human_eval,
        "llm_eval": llm_eval,
        "llm_reasoning": llm_reasoning,
    })


@app.route("/api/qa/list")
def list_qa():
    items = []
    for i, qa_file in enumerate(state["qa_files"]):
        try:
            qa_data = json.loads(qa_file.read_text())
            question = qa_data.get("question", "")
        except Exception:
            question = ""
        trace_id = _trace_id(qa_file)

        # Check agreement status
        human_eval = state["human_evals"].get(trace_id, {})
        llm_eval = state["llm_evals"].get(trace_id, {})

        agree_all = True
        if human_eval and llm_eval:
            for dim in DIMENSIONS:
                if human_eval.get(dim) != llm_eval.get(dim):
                    agree_all = False
                    break
        else:
            agree_all = False

        items.append({
            "index": i,
            "trace_id": trace_id,
            "category": _category(qa_file),
            "filename": qa_file.name,
            "question": question,
            "agree_all": agree_all,
        })
    return jsonify(items)


# ── Landing page ─────────────────────────────────────────────────────────────

LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Human ↔ LLM Enhanced Compare</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html { font-size: 18px; }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      background: #f5f5f5;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100vh;
      color: #1e293b;
      padding: 40px 20px;
    }
    .card {
      background: #ffffff;
      border: 2px solid #cbd5e1;
      border-radius: 16px;
      padding: 60px 60px;
      min-width: 500px;
      max-width: 600px;
      text-align: center;
      box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }
    h1 {
      font-size: 2rem;
      font-weight: 800;
      letter-spacing: 0.05em;
      color: #0f172a;
      margin-bottom: 20px;
      line-height: 1.2;
    }
    .subtitle {
      font-size: 1.1rem;
      color: #475569;
      margin-bottom: 48px;
      line-height: 1.6;
    }
    #version-list { display: flex; flex-direction: column; gap: 20px; }
    .version-btn {
      background: #f8fafc;
      border: 2px solid #cbd5e1;
      border-radius: 12px;
      padding: 28px 28px;
      cursor: pointer;
      display: block;
      width: 100%;
      text-align: left;
      transition: border-color 0.15s, background 0.15s;
    }
    .version-btn:hover { border-color: #3b82f6; background: #eff6ff; }
    .version-name { font-size: 1.4rem; font-weight: 800; color: #0f172a; letter-spacing: 0.06em; margin-bottom: 12px; }
    .version-meta { font-size: 1rem; color: #64748b; line-height: 1.8; margin-bottom: 8px; }
    .version-badges { display: flex; gap: 8px; align-items: center; justify-content: center; margin-top: 12px; }
    .badge {
      font-size: 0.9rem; font-weight: 700; padding: 6px 12px;
      border-radius: 6px; text-transform: uppercase; letter-spacing: 0.06em;
    }
    .badge-both  { background: #d1fae5; color: #065f46; }
    #error-msg { color: #dc2626; font-size: 1rem; margin-top: 24px; display: none; line-height: 1.6; }
    #loading   { color: #64748b; font-size: 1rem; }
  </style>
</head>
<body>
<div class="card">
  <h1>HUMAN ↔ LLM COMPARE</h1>
  <p class="subtitle">Select a version with both evaluations</p>
  <div id="version-list"><span id="loading">Scanning for versions…</span></div>
  <div id="error-msg"></div>
</div>
<script>
function loadVersions() {
  fetch("/api/versions")
    .then(r => r.json())
    .then(versions => {
      const list = document.getElementById("version-list");
      if (!versions.length) {
        list.innerHTML = '<span style="color:#f87171">No comparable versions found</span>';
        return;
      }
      list.innerHTML = "";

      // Group versions by version name
      const grouped = {};
      versions.forEach(v => {
        if (!grouped[v.version]) {
          grouped[v.version] = [];
        }
        grouped[v.version].push(v);
      });

      Object.keys(grouped).sort().forEach(versionName => {
        const versionGroup = grouped[versionName];
        const v = versionGroup[0]; // Use first item for qa_count (same for all in group)

        if (versionGroup.length === 1) {
          // Single enhanced eval file - direct button
          const meta = versionGroup[0].metadata || {};
          const btn = document.createElement("button");
          btn.className = "version-btn";
          btn.innerHTML = `
            <div>
              <div class="version-name">${versionName}</div>
              <div class="version-meta">
                <div>${v.qa_count} QA item${v.qa_count !== 1 ? "s" : ""}</div>
                <div style="font-size:0.75rem;color:#94a3b8;margin-top:4px;">
                  <strong>${meta.date} ${meta.time}</strong> • Model: ${meta.model} • Prompt: ${meta.prompt_version}
                </div>
                <div style="font-size:0.7rem;color:#64748b;font-family:monospace;margin-top:3px;">
                  ${meta.filename}
                </div>
              </div>
            </div>
            <div class="version-badges">
              <span class="badge badge-both">Human + Enhanced LLM</span>
            </div>`;
          btn.onclick = () => selectVersion(versionName, versionGroup[0].enhanced_file);
          list.appendChild(btn);
        } else {
          // Multiple enhanced eval files - expandable list
          const container = document.createElement("div");
          container.style.marginBottom = "8px";

          const mainBtn = document.createElement("button");
          mainBtn.className = "version-btn";
          mainBtn.style.cursor = "pointer";
          mainBtn.innerHTML = `
            <div>
              <div class="version-name">${versionName}</div>
              <div class="version-meta">${v.qa_count} QA item${v.qa_count !== 1 ? "s" : ""} • ${versionGroup.length} LLM eval${versionGroup.length !== 1 ? "s" : ""}</div>
            </div>
            <div class="version-badges">
              <span class="badge badge-both">Expand ▼</span>
            </div>`;

          const subList = document.createElement("div");
          subList.style.display = "none";
          subList.style.marginTop = "4px";
          subList.style.paddingLeft = "12px";
          subList.style.borderLeft = "2px solid #334155";

          versionGroup.forEach((v, idx) => {
            const meta = v.metadata || {};
            const subBtn = document.createElement("button");
            subBtn.className = "version-btn";
            subBtn.style.marginBottom = "4px";
            subBtn.style.fontSize = "0.9rem";
            subBtn.innerHTML = `
              <div>
                <div class="version-name" style="font-size: 0.95rem;">LLM Eval ${idx + 1}</div>
                <div class="version-meta">
                  <div style="font-size:0.75rem;color:#94a3b8;margin-bottom:3px;">
                    <strong>${meta.date} ${meta.time}</strong> • Model: ${meta.model} • Prompt: ${meta.prompt_version}
                  </div>
                  <div style="font-size:0.7rem;color:#64748b;font-family:monospace;">
                    ${meta.filename}
                  </div>
                </div>
              </div>`;
            subBtn.onclick = () => selectVersion(versionName, v.enhanced_file);
            subList.appendChild(subBtn);
          });

          mainBtn.onclick = (e) => {
            e.preventDefault();
            subList.style.display = subList.style.display === "none" ? "block" : "none";
            mainBtn.innerHTML = `
              <div>
                <div class="version-name">${versionName}</div>
                <div class="version-meta">${v.qa_count} QA item${v.qa_count !== 1 ? "s" : ""} • ${versionGroup.length} LLM eval${versionGroup.length !== 1 ? "s" : ""}</div>
              </div>
              <div class="version-badges">
                <span class="badge badge-both">${subList.style.display === "none" ? "Expand ▼" : "Collapse ▲"}</span>
              </div>`;
          };

          container.appendChild(mainBtn);
          container.appendChild(subList);
          list.appendChild(container);
        }
      });
    })
    .catch(() => {
      document.getElementById("version-list").innerHTML =
        '<span style="color:#f87171">Error scanning versions.</span>';
    });
}

function selectVersion(version, enhancedFile) {
  fetch("/api/select-version", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ version, enhanced_file: enhancedFile }),
  })
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        const el = document.getElementById("error-msg");
        el.textContent = data.error;
        el.style.display = "block";
      } else {
        window.location.href = "/compare";
      }
    });
}

loadVersions();
</script>
</body>
</html>
"""

# ── Compare page ──────────────────────────────────────────────────────────────

COMPARE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Human ↔ LLM Enhanced Compare</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html { font-size: 16px; }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      background: #f1f5f9;
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }

    #header {
      background: #0f172a;
      color: white;
      padding: 10px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
      gap: 16px;
    }
    #header h1 { font-size: 1rem; font-weight: 700; letter-spacing: 0.05em; color: #e2e8f0; }
    #trace-id { font-size: 1rem; font-weight: 700; color: #f1f5f9; }
    #progress { font-size: 0.85rem; color: #94a3b8; }

    #font-controls {
      display: flex;
      align-items: center;
      gap: 4px;
      white-space: nowrap;
    }
    .font-btn {
      background: #334155;
      color: #e2e8f0;
      border: none;
      border-radius: 5px;
      padding: 3px 9px;
      cursor: pointer;
      font-weight: 700;
      line-height: 1;
      transition: background 0.12s;
    }
    .font-btn:hover { background: #475569; }
    #font-size-label { font-size: 0.72rem; color: #64748b; min-width: 32px; text-align: center; }

    #app {
      flex: 1;
      display: flex;
      overflow: hidden;
    }

    /* Sidebar */
    #sidebar {
      width: 230px;
      background: #1e293b;
      display: flex;
      flex-direction: column;
      border-right: 2px solid #0f172a;
      overflow: hidden;
      position: relative;
      transition: width 0.15s ease;
    }
    #sidebar-resize-handle {
      position: absolute;
      right: 0;
      top: 0;
      bottom: 0;
      width: 8px;
      cursor: col-resize;
      background: transparent;
      transition: background 0.1s;
      z-index: 100;
    }
    #sidebar-resize-handle:hover {
      background: #3b82f6;
    }
    #sidebar-header {
      padding: 10px 12px;
      background: #0f172a;
      flex-shrink: 0;
      border-bottom: 1px solid #334155;
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      color: #64748b;
    }
    #sidebar-list {
      flex: 1;
      overflow-y: auto;
      padding: 6px 0;
    }
    #sidebar-list::-webkit-scrollbar { width: 5px; }
    #sidebar-list::-webkit-scrollbar-track { background: #1e293b; }
    #sidebar-list::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

    .sidebar-item {
      padding: 10px 12px;
      cursor: pointer;
      border-bottom: 1px solid #0f172a;
      transition: background 0.1s;
    }
    .sidebar-item:hover { background: #273548; }
    .sidebar-item.active { background: #1d4ed8; }
    .sidebar-item.agree { background: #dcfce7; }
    .sidebar-item.agree:hover { background: #bbf7d0; }
    .sidebar-item.disagree { background: #fee2e2; }
    .sidebar-item.disagree:hover { background: #fecaca; }
    .sidebar-item.agree .item-trace,
    .sidebar-item.agree .item-question { color: #166534; }
    .sidebar-item.disagree .item-trace,
    .sidebar-item.disagree .item-question { color: #991b1b; }

    .item-trace {
      font-size: 0.78rem;
      font-weight: 800;
      color: #f1f5f9;
    }
    .item-question {
      font-size: 0.75rem;
      color: #94a3b8;
      margin-top: 4px;
      overflow: hidden;
      display: -webkit-box;
      -webkit-box-orient: vertical;
      -webkit-line-clamp: 2;
    }

    /* Main content */
    #content-area {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
      background: white;
    }

    .qa-section {
      margin-bottom: 16px;
    }
    .qa-section.full-width {
      padding-bottom: 16px;
      border-bottom: 1px solid #e2e8f0;
      margin-bottom: 20px;
    }
    .two-column-sections {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 20px;
      padding-bottom: 16px;
      border-bottom: 1px solid #e2e8f0;
    }
    .two-column-sections .qa-section {
      margin-bottom: 0;
      border-bottom: none;
      padding-bottom: 0;
    }
    .qa-section h3 {
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #64748b;
      margin-bottom: 6px;
    }
    .qa-section .content {
      font-size: 0.9rem;
      color: #1e293b;
      line-height: 1.6;
    }
    .qa-section ul { padding-left: 20px; }

    .inline-fields {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px 16px;
      margin-bottom: 20px;
      padding-bottom: 16px;
      border-bottom: 1px solid #e2e8f0;
    }
    .inline-field {
      display: flex;
      gap: 6px;
      align-items: flex-start;
    }
    .inline-field h3 {
      font-size: 0.85rem;
      font-weight: 700;
      color: #64748b;
      white-space: nowrap;
      margin: 0;
      flex-shrink: 0;
    }
    .inline-field h3::after {
      content: ":";
    }
    .inline-field .content {
      font-size: 0.85rem;
      color: #1e293b;
      line-height: 1.4;
      margin: 0;
      flex: 1;
    }
    .inline-field .content ul {
      margin: 0;
      padding-left: 40px;
    }
    .inline-field .content li {
      margin-bottom: 3px;
    }

    .dimensions-section {
      margin-top: 30px;
      padding-top: 20px;
      border-top: 3px solid #e2e8f0;
    }
    .dimensions-section h2 {
      font-size: 0.95rem;
      font-weight: 700;
      color: #0f172a;
      margin-bottom: 16px;
    }

    .dimension-row {
      margin-bottom: 16px;
      padding-bottom: 16px;
      border-bottom: 1px solid #e2e8f0;
    }
    .dimension-row:last-child { border-bottom: none; }

    .dim-title {
      font-size: 0.85rem;
      font-weight: 700;
      color: #1e293b;
      margin-bottom: 8px;
      text-transform: capitalize;
    }

    .dim-table {
      display: inline-block;
      border-collapse: collapse;
      font-size: 0.85rem;
    }
    .dim-table th, .dim-table td {
      padding: 6px 12px;
      text-align: center;
      border: 1px solid #cbd5e1;
    }
    .dim-table th {
      background: #f1f5f9;
      font-weight: 700;
      color: #475569;
    }
    .dim-table td {
      font-weight: 700;
    }

    .agree { background: #dcfce7; color: #166534; }
    .disagree { background: #fee2e2; color: #991b1b; }

    .llm-reasoning {
      font-size: 0.8rem;
      color: #64748b;
      margin-top: 8px;
      padding: 8px 12px;
      background: #f8fafc;
      border-left: 3px solid #f87171;
      border-radius: 4px;
      font-style: italic;
    }

    /* ── Dimension tooltip ── */
    .has-tooltip {
      position: relative;
      cursor: help;
      text-decoration: underline dotted #475569;
    }
    .has-tooltip::after {
      content: attr(data-tooltip);
      position: absolute;
      top: calc(100% + 8px);
      left: 50%;
      transform: translateX(-50%);
      background: #0f172a;
      color: #e2e8f0;
      padding: 10px 14px;
      border-radius: 7px;
      font-size: 0.78rem;
      font-weight: 400;
      line-height: 1.5;
      white-space: normal;
      width: 300px;
      text-align: left;
      text-transform: none;
      letter-spacing: 0;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.15s;
      z-index: 200;
      box-shadow: 0 4px 16px rgba(0,0,0,0.4);
      border: 1px solid #334155;
    }
    .has-tooltip:hover::after { opacity: 1; }

    /* Footer */
    #footer {
      background: #0f172a;
      padding: 10px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
    }
    .footer-btns { display: flex; gap: 8px; }

    .nav-btn {
      color: white;
      border: none;
      padding: 8px 22px;
      border-radius: 7px;
      font-size: 0.85rem;
      font-weight: 700;
      cursor: pointer;
      transition: background 0.12s;
    }
    .nav-btn:hover:not(:disabled) { background: #64748b; }
    .nav-btn:disabled { background: #1e293b; color: #334155; cursor: not-allowed; }
    #btn-prev { background: #475569; }
    #btn-next { background: #3b82f6; }
    #btn-next:hover:not(:disabled) { background: #2563eb; }
  </style>
</head>
<body>

<div id="header">
  <h1>HUMAN ↔ LLM ENHANCED</h1>
  <span id="trace-id">—</span>
  <span id="progress">— / —</span>
  <div id="font-controls">
    <button class="font-btn" onclick="changeFontSize(-1)">A−</button>
    <span id="font-size-label">16px</span>
    <button class="font-btn" onclick="changeFontSize(+1)">A+</button>
  </div>
</div>

<div id="app">
  <div id="sidebar">
    <div id="sidebar-header">QA Items</div>
    <div id="sidebar-list"></div>
    <div id="sidebar-resize-handle"></div>
  </div>

  <div id="content-area">
    <div id="qa-content"></div>
  </div>
</div>

<div id="footer">
  <span></span>
  <div class="footer-btns">
    <button id="btn-prev" class="nav-btn" onclick="prevQA()">‹ PREV</button>
    <button id="btn-next" class="nav-btn" onclick="nextQA()">NEXT ›</button>
  </div>
</div>

<script>
const DIMENSIONS = [
  "answer_completeness",
  "safety_specificity",
  "tool_realism",
  "scope_appropriateness",
  "context_clarity",
  "tip_usefulness",
];

const DIM_LABELS = {
  answer_completeness: "Answer Completeness",
  safety_specificity: "Safety Specificity",
  tool_realism: "Tool Realism",
  scope_appropriateness: "Scope Appropriateness",
  context_clarity: "Context Clarity",
  tip_usefulness: "Tip Usefulness",
};

const CRITERIA = {
  answer_completeness:   "D1 · Answer Completeness — The answer contains enough detail for a homeowner to actually complete the repair end to end (tools, concrete steps, safety, a useful tip). Answers that stop short or omit key stages fail.",
  safety_specificity:    "D2 · Safety Specificity — safety_info names the specific hazard of this repair and the specific precaution to take. Generic phrases ('be careful', 'use caution', 'stay safe') fail.",
  tool_realism:          "D3 · Tool Realism — Every item in tools_required is something a typical homeowner already owns or could buy at a general hardware store for under $50. No professional, specialty, or trade-only tools.",
  scope_appropriateness: "D4 · Scope Appropriateness — The repair is within realistic DIY capability. If professional help is genuinely needed (e.g., gas lines, panel work), the answer says so clearly rather than giving amateur instructions.",
  context_clarity:       "D5 · Context Clarity — question and answer contain enough context to understand the problem, and the answer directly addresses the specific equipment_problem.",
  tip_usefulness:        "D6 · Tip Usefulness — tips provide non-obvious, task-specific advice that adds value beyond the steps. Tips that merely restate a step or offer generic encouragement fail.",
};

let currentIndex = 0;
let sidebarItems = [];

function esc(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function loadSidebar() {
  fetch("/api/qa/list")
    .then(r => r.json())
    .then(items => {
      sidebarItems = items;
      renderSidebar();
    })
    .catch(err => {
      console.error("Error loading sidebar:", err);
      document.getElementById("sidebar-list").innerHTML = '<div style="color:#f87171;padding:10px">Error loading items</div>';
    });
}

function renderSidebar() {
  const list = document.getElementById("sidebar-list");
  list.innerHTML = "";
  sidebarItems.forEach(item => {
    const div = document.createElement("div");
    let className = "sidebar-item" + (item.index === currentIndex ? " active" : "");
    if (item.agree_all) {
      className += " agree";
    } else {
      className += " disagree";
    }
    div.className = className;
    div.dataset.index = item.index;
    div.onclick = () => loadQA(item.index);
    div.innerHTML = `
      <div class="item-trace">${esc(item.trace_id)}</div>
      <div class="item-question">${esc(item.question)}</div>`;
    list.appendChild(div);
  });
}

function updateSidebarActive() {
  document.querySelectorAll(".sidebar-item").forEach(el => {
    el.classList.toggle("active", parseInt(el.dataset.index) === currentIndex);
  });
  const active = document.querySelector(".sidebar-item.active");
  if (active) active.scrollIntoView({ block: "nearest" });
}

function formatValue(val) {
  if (Array.isArray(val)) {
    if (!val.length) return "<em>None</em>";
    return "<ul>" + val.map(v => `<li>${esc(v)}</li>`).join("") + "</ul>";
  }
  return esc(String(val));
}

function renderQA(data) {
  document.getElementById("trace-id").textContent = data.trace_id;
  document.getElementById("progress").textContent = `${data.index + 1} / ${data.total}`;
  document.getElementById("btn-prev").disabled = data.index === 0;
  document.getElementById("btn-next").disabled = data.index + 1 >= data.total;

  const container = document.getElementById("qa-content");
  container.innerHTML = "";

  // Inline fields (2-column grid)
  const inlineFields = [
    { key: "question", label: "Question" },
    { key: "equipment_problem", label: "Equipment Problem" },
    { key: "safety_info", label: "Safety Info" },
    { key: "tips", label: "Tips" },
  ];

  const inlineContainer = document.createElement("div");
  inlineContainer.className = "inline-fields";
  inlineFields.forEach(field => {
    const div = document.createElement("div");
    div.className = "inline-field";
    const label = document.createElement("h3");
    label.textContent = field.label;
    const content = document.createElement("div");
    content.className = "content";
    content.innerHTML = formatValue(data.qa[field.key]);
    div.appendChild(label);
    div.appendChild(content);
    inlineContainer.appendChild(div);
  });
  container.appendChild(inlineContainer);

  // Answer section (full-width)
  const answerDiv = document.createElement("div");
  answerDiv.className = "qa-section full-width";
  answerDiv.innerHTML = `
    <h3>Answer</h3>
    <div class="content">${formatValue(data.qa.answer)}</div>`;
  container.appendChild(answerDiv);

  // Tools Required and Steps (side-by-side)
  const twoColContainer = document.createElement("div");
  twoColContainer.className = "two-column-sections";

  const toolsDiv = document.createElement("div");
  toolsDiv.className = "qa-section";
  toolsDiv.innerHTML = `
    <h3>Tools Required</h3>
    <div class="content">${formatValue(data.qa.tools_required)}</div>`;
  twoColContainer.appendChild(toolsDiv);

  const stepsDiv = document.createElement("div");
  stepsDiv.className = "qa-section";
  stepsDiv.innerHTML = `
    <h3>Steps</h3>
    <div class="content">${formatValue(data.qa.steps)}</div>`;
  twoColContainer.appendChild(stepsDiv);

  container.appendChild(twoColContainer);

  // Dimensions comparison
  const dimDiv = document.createElement("div");
  dimDiv.className = "dimensions-section";
  dimDiv.innerHTML = "<h2>Dimension Scores</h2>";

  const human = data.human_eval || {};
  const llm = data.llm_eval || {};
  const reasoning = data.llm_reasoning || {};

  DIMENSIONS.forEach(dim => {
    const hVal = human[dim];
    const lVal = llm[dim];
    const agree = hVal === lVal;
    const reason = reasoning[dim] || "";

    const dimRow = document.createElement("div");
    dimRow.className = "dimension-row";

    const title = document.createElement("div");
    title.className = "dim-title has-tooltip";
    title.textContent = DIM_LABELS[dim] || dim;
    title.setAttribute("data-tooltip", CRITERIA[dim] || "");
    dimRow.appendChild(title);

    const table = document.createElement("table");
    table.className = "dim-table";
    table.innerHTML = `
      <tr>
        <th>Human</th>
        <th>LLM</th>
      </tr>
      <tr class="${agree ? 'agree' : 'disagree'}">
        <td>${hVal === 1 ? 'PASS' : hVal === 0 ? 'FAIL' : '—'}</td>
        <td>${lVal === 1 ? 'PASS' : lVal === 0 ? 'FAIL' : '—'}</td>
      </tr>`;
    dimRow.appendChild(table);

    if (!agree && reason) {
      const reasonDiv = document.createElement("div");
      reasonDiv.className = "llm-reasoning";
      reasonDiv.textContent = "LLM Reasoning: " + reason;
      dimRow.appendChild(reasonDiv);
    }

    dimDiv.appendChild(dimRow);
  });

  container.appendChild(dimDiv);
  updateSidebarActive();
  container.scrollTop = 0;
}

function loadQA(idx) {
  fetch(`/api/qa/${idx}`)
    .then(r => r.json())
    .then(data => {
      if (data.done) return;
      currentIndex = data.index;
      renderQA(data);
    })
    .catch(err => {
      console.error("Error loading QA:", err);
      document.getElementById("qa-content").innerHTML = '<div style="color:#f87171;padding:20px">Error loading QA item</div>';
    });
}

function nextQA() { loadQA(currentIndex + 1); }
function prevQA() { loadQA(currentIndex - 1); }

let baseFontSize = 16;
function changeFontSize(delta) {
  baseFontSize = Math.min(26, Math.max(11, baseFontSize + delta));
  document.documentElement.style.fontSize = baseFontSize + "px";
  document.getElementById("font-size-label").textContent = baseFontSize + "px";
}

// Sidebar resize
let isResizing = false;
const sidebar = document.getElementById("sidebar");
const resizeHandle = document.getElementById("sidebar-resize-handle");

resizeHandle.addEventListener("mousedown", () => {
  isResizing = true;
  document.body.style.userSelect = "none";
});

document.addEventListener("mousemove", (e) => {
  if (!isResizing) return;
  const appDiv = document.getElementById("app");
  const appRect = appDiv.getBoundingClientRect();
  let newWidth = e.clientX - appRect.left;
  newWidth = Math.max(150, Math.min(600, newWidth));
  sidebar.style.width = newWidth + "px";
});

document.addEventListener("mouseup", () => {
  isResizing = false;
  document.body.style.userSelect = "";
});

loadSidebar();
loadQA(0);
</script>
</body>
</html>
"""

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info(f"Logging to {_log_file}")
    versions = _detect_comparable_versions()
    if not versions:
        log.error(f"No comparable versions found in {QA_ITEMS_ROOT}")
        sys.exit(1)
    log.info(f"Found {len(versions)} comparable version(s): {[v['version'] for v in versions]}")
    webbrowser.open("http://127.0.0.1:5002")
    app.run(host="127.0.0.1", port=5002, debug=False)
