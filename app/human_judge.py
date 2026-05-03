#!/usr/bin/env python3
"""Human judge Flask web app for evaluating QA items."""

import json
import logging
import sys
import time
import webbrowser
from pathlib import Path
from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QA_ITEMS_ROOT = PROJECT_ROOT / "qa_items"
EVAL_FILENAME = "QA_human_eval.json"

_LOGS_DIR = PROJECT_ROOT / "logs"
_LOGS_DIR.mkdir(exist_ok=True)
_log_file = _LOGS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_human_judge.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(_log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# (qa_field_key, dimension_key, display_label)
# dimension_key is None for fields that are display-only (no PASS/FAIL)
FIELDS = [
    ("question",          None,                    "Question"),
    ("answer",            "answer_completeness",   "Answer"),
    ("equipment_problem", "context_clarity",       "Equipment Problem"),
    ("tools_required",    "tool_realism",          "Tools Required"),
    ("steps",             "scope_appropriateness", "Steps"),
    ("safety_info",       "safety_specificity",    "Safety Info"),
    ("tips",              "tip_usefulness",        "Tips"),
]

DIMENSIONS = [dim for _, dim, _ in FIELDS if dim]

app = Flask(__name__)

state: dict = {
    "qa_folder":   None,   # set when user picks a version
    "qa_files":    [],
    "evaluations": {},
}


def _trace_id(qa_file: Path) -> str:
    """QA1_hvac1.qa  ->  QA1"""
    return qa_file.stem.split("_")[0]


def _category(qa_file: Path) -> str:
    """QA1_hvac1.qa  ->  hvac"""
    import re
    parts = qa_file.stem.split("_", 1)
    return re.sub(r"\d+$", "", parts[1]) if len(parts) > 1 else "unknown"


def _load_existing_evals(folder: Path) -> dict:
    eval_path = folder / EVAL_FILENAME
    if not eval_path.exists():
        return {}
    try:
        items = json.loads(eval_path.read_text())
        if isinstance(items, list):
            return {e["trace_id"]: e for e in items if "trace_id" in e}
    except Exception:
        pass
    return {}



def _has_any_vote(e: dict) -> bool:
    return any(e.get(dim) is not None for dim in DIMENSIONS)


def _vote_state(e: dict) -> str:
    voted = sum(1 for dim in DIMENSIONS if e.get(dim) is not None)
    if voted == 0:
        return "none"
    return "full" if voted == len(DIMENSIONS) else "partial"


def _detect_versions() -> list[dict]:
    """Return sorted list of version dicts found under qa_items/."""
    if not QA_ITEMS_ROOT.exists():
        return []
    versions = []
    for d in sorted(QA_ITEMS_ROOT.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            qa_count = len(list(d.glob("*.qa")))
            has_eval = (d / EVAL_FILENAME).exists()
            versions.append({"version": d.name, "qa_count": qa_count, "has_eval": has_eval})
    return versions


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return LANDING_HTML


@app.route("/judge")
def judge():
    if not state["qa_folder"]:
        return '<meta http-equiv="refresh" content="0;url=/">'
    return JUDGE_HTML


@app.route("/api/versions")
def get_versions():
    return jsonify(_detect_versions())


@app.route("/api/select-version", methods=["POST"])
def select_version():
    version = (request.json or {}).get("version", "").strip()
    folder = QA_ITEMS_ROOT / version
    if not folder.is_dir():
        return jsonify({"error": f"Version '{version}' not found"}), 404
    files = sorted(folder.glob("*.qa"), key=lambda p: int(p.stem.split("_")[0][2:] or 0))
    if not files:
        return jsonify({"error": f"No .qa files in {folder}"}), 404
    state["qa_folder"]   = folder
    state["qa_files"]    = files
    state["evaluations"] = _load_existing_evals(folder)
    return jsonify({"ok": True, "version": version, "qa_count": len(files)})


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
    existing = state["evaluations"].get(trace_id, {})
    return jsonify({
        "index": idx,
        "total": len(files),
        "trace_id": trace_id,
        "filename": qa_file.name,
        "qa": qa_data,
        "eval": existing,
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
        items.append({
            "index":    i,
            "trace_id": trace_id,
            "category": _category(qa_file),
            "filename": qa_file.name,
            "question": question,
            "vote_state": _vote_state(state["evaluations"].get(trace_id, {})),
        })
    return jsonify(items)


@app.route("/api/eval", methods=["POST"])
def post_eval():
    data = request.json
    trace_id = data.get("trace_id")
    if not trace_id:
        return jsonify({"error": "missing trace_id"}), 400
    state["evaluations"][trace_id] = data
    return jsonify({"ok": True})


@app.route("/api/save", methods=["POST"])
def save_file():
    evals = [e for e in state["evaluations"].values() if _has_any_vote(e)]
    out = state["qa_folder"] / EVAL_FILENAME
    out.write_text(json.dumps(evals, indent=2))
    return jsonify({"ok": True, "path": str(out), "count": len(evals)})


# ── Landing page ─────────────────────────────────────────────────────────────

LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>QA Human Judge — Select Version</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html { font-size: 16px; }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      background: #0f172a;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100vh;
      color: #e2e8f0;
    }
    .card {
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 40px 48px;
      min-width: 420px;
      max-width: 520px;
      text-align: center;
    }
    h1 { font-size: 1.4rem; font-weight: 800; letter-spacing: 0.05em; color: #f1f5f9; margin-bottom: 6px; }
    .subtitle { font-size: 0.85rem; color: #64748b; margin-bottom: 32px; }
    #version-list { display: flex; flex-direction: column; gap: 12px; }
    .version-btn {
      background: #0f172a;
      border: 2px solid #334155;
      border-radius: 8px;
      padding: 16px 20px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: space-between;
      transition: border-color 0.15s, background 0.15s;
      text-align: left;
    }
    .version-btn:hover { border-color: #3b82f6; background: #1a2744; }
    .version-name { font-size: 1.1rem; font-weight: 800; color: #f1f5f9; letter-spacing: 0.06em; }
    .version-meta { font-size: 0.78rem; color: #64748b; margin-top: 3px; }
    .version-badges { display: flex; gap: 8px; align-items: center; }
    .badge {
      font-size: 0.68rem; font-weight: 700; padding: 3px 8px;
      border-radius: 4px; text-transform: uppercase; letter-spacing: 0.06em;
    }
    .badge-qa    { background: #1e3a5f; color: #93c5fd; }
    .badge-eval  { background: #14532d; color: #86efac; }
    #error-msg { color: #f87171; font-size: 0.82rem; margin-top: 16px; display: none; }
    #loading   { color: #64748b; font-size: 0.85rem; }
  </style>
</head>
<body>
<div class="card">
  <h1>QA HUMAN JUDGE</h1>
  <p class="subtitle">Select a version to evaluate</p>
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
        list.innerHTML = '<span style="color:#f87171">No versions found in qa_items/</span>';
        return;
      }
      list.innerHTML = "";
      versions.forEach(v => {
        const btn = document.createElement("button");
        btn.className = "version-btn";
        btn.innerHTML = `
          <div>
            <div class="version-name">${v.version}</div>
            <div class="version-meta">${v.qa_count} QA item${v.qa_count !== 1 ? "s" : ""}</div>
          </div>
          <div class="version-badges">
            <span class="badge badge-qa">${v.qa_count} items</span>
            ${v.has_eval ? '<span class="badge badge-eval">eval saved</span>' : ""}
          </div>`;
        btn.onclick = () => selectVersion(v.version);
        list.appendChild(btn);
      });
    })
    .catch(() => {
      document.getElementById("version-list").innerHTML =
        '<span style="color:#f87171">Error scanning versions.</span>';
    });
}

function selectVersion(version) {
  fetch("/api/select-version", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ version }),
  })
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        const el = document.getElementById("error-msg");
        el.textContent = data.error;
        el.style.display = "block";
      } else {
        window.location.href = "/judge";
      }
    });
}

loadVersions();
</script>
</body>
</html>
"""

# ── Judge app ─────────────────────────────────────────────────────────────────

JUDGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>QA Human Judge</title>
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

    /* ── Header ── */
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
    #header h1   { font-size: 1rem; font-weight: 700; letter-spacing: 0.05em; color: #e2e8f0; white-space: nowrap; }
    #trace-id    { font-size: 1rem; font-weight: 700; color: #f1f5f9; letter-spacing: 0.06em; white-space: nowrap; }
    #filename    { font-family: monospace; font-size: 0.78rem; color: #64748b; flex: 1; text-align: center; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    #progress    { font-size: 0.85rem; color: #94a3b8; white-space: nowrap; }

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

    /* ── Body: sidebar + survey ── */
    #app {
      flex: 1;
      display: flex;
      overflow: hidden;
    }

    /* ── Sidebar ── */
    #sidebar {
      width: 230px;
      min-width: 230px;
      background: #1e293b;
      display: flex;
      flex-direction: column;
      border-right: 2px solid #0f172a;
      transition: width 0.2s ease, min-width 0.2s ease;
      overflow: hidden;
    }
    #sidebar.expanded {
      width: 400px;
      min-width: 400px;
    }

    #sidebar-header {
      padding: 10px 12px;
      background: #0f172a;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
      border-bottom: 1px solid #334155;
    }
    #sidebar-header span {
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      color: #64748b;
    }
    #btn-expand {
      background: #334155;
      color: #94a3b8;
      border: none;
      padding: 4px 10px;
      border-radius: 5px;
      font-size: 0.7rem;
      font-weight: 700;
      cursor: pointer;
      letter-spacing: 0.05em;
      transition: background 0.12s;
    }
    #btn-expand:hover { background: #475569; color: #e2e8f0; }

    #sidebar-list {
      flex: 1;
      overflow-y: auto;
      padding: 6px 0;
    }
    #sidebar-list::-webkit-scrollbar { width: 5px; }
    #sidebar-list::-webkit-scrollbar-track { background: #1e293b; }
    #sidebar-list::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

    .sidebar-item {
      position: relative;
      padding: 10px 12px;
      cursor: pointer;
      border-bottom: 1px solid #0f172a;
      transition: background 0.1s;
    }
    .sidebar-item:hover   { background: #273548; }
    .sidebar-item.active  { background: #1d4ed8; }

    .item-check {
      position: absolute;
      top: 8px;
      right: 8px;
      font-size: 0.82rem;
      font-weight: 900;
      line-height: 1;
      pointer-events: none;
    }
    .item-check.checked, .item-check.partial { color: #4ade80; }
    .item-check.unchecked { color: #334155; }

    .item-meta {
      display: flex;
      align-items: center;
      gap: 6px;
      margin-bottom: 4px;
    }
    .item-trace {
      font-size: 0.78rem;
      font-weight: 800;
      color: #f1f5f9;
      letter-spacing: 0.04em;
    }
    .item-cat {
      font-size: 0.65rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      padding: 1px 6px;
      border-radius: 4px;
    }
    .cat-hvac        { background: #1e40af; color: #bfdbfe; }
    .cat-appliance   { background: #92400e; color: #fde68a; }
    .cat-electrical  { background: #713f12; color: #fef08a; }
    .cat-plumbing    { background: #134e4a; color: #99f6e4; }
    .cat-general     { background: #374151; color: #d1d5db; }
    .cat-unknown     { background: #374151; color: #d1d5db; }

    .item-question {
      font-size: 0.75rem;
      color: #94a3b8;
      line-height: 1.4;
      overflow: hidden;
      display: -webkit-box;
      -webkit-box-orient: vertical;
      -webkit-line-clamp: 2;
    }
    #sidebar.expanded .item-question {
      -webkit-line-clamp: unset;
      display: block;
    }
    .sidebar-item.active .item-question { color: #bfdbfe; }
    .sidebar-item.active .item-trace    { color: white; }

    /* ── Survey pane ── */
    #survey-area {
      flex: 1;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
    }

    /* ── QA rows ── */
    .qa-row {
      display: flex;
      border-bottom: 1px solid #e2e8f0;
      background: white;
    }
    .qa-row:nth-child(even) { background: #f8fafc; }

    .qa-field {
      flex: 0 0 75%;
      padding: 14px 22px;
      border-right: 2px solid #e2e8f0;
    }
    .qa-field h3 {
      font-size: 0.68rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      color: #94a3b8;
      margin-bottom: 6px;
    }
    .qa-field .content {
      font-size: 0.88rem;
      color: #1e293b;
      line-height: 1.6;
    }
    .qa-field ul { padding-left: 20px; }
    .qa-field li { margin-bottom: 3px; }

    .qa-buttons {
      flex: 0 0 25%;
      padding: 14px 16px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }
    .dim-label {
      font-size: 0.68rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      color: #94a3b8;
      text-align: center;
    }
    .btn-row { display: flex; gap: 8px; }

    /* ── PASS / FAIL buttons ── */
    .btn {
      padding: 7px 18px;
      border: 2px solid transparent;
      border-radius: 6px;
      font-size: 0.78rem;
      font-weight: 800;
      cursor: pointer;
      letter-spacing: 0.06em;
      transition: all 0.12s ease;
    }
    .btn-pass { background: #dcfce7; color: #166534; border-color: #86efac; }
    .btn-pass:hover { background: #bbf7d0; border-color: #4ade80; }
    .btn-pass.active { background: #16a34a; color: white; border-color: #15803d; }
    .btn-fail { background: #fee2e2; color: #991b1b; border-color: #fca5a5; }
    .btn-fail:hover { background: #fecaca; border-color: #f87171; }
    .btn-fail.active { background: #dc2626; color: white; border-color: #b91c1c; }

    /* ── Done screen ── */
    #done-msg {
      display: none;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      flex: 1;
      gap: 10px;
      color: #64748b;
      font-size: 1rem;
    }
    #done-msg h2 { color: #1e293b; font-size: 1.4rem; }

    /* ── Footer ── */
    #footer {
      background: #0f172a;
      padding: 10px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
    }
    #eval-status { color: #64748b; font-size: 0.82rem; }
    .footer-btns { display: flex; gap: 8px; }

    .nav-btn {
      color: white;
      border: none;
      padding: 8px 22px;
      border-radius: 7px;
      font-size: 0.85rem;
      font-weight: 700;
      cursor: pointer;
      letter-spacing: 0.05em;
      transition: background 0.12s;
    }
    #btn-back { background: #475569; }
    #btn-back:hover:not(:disabled) { background: #64748b; }
    #btn-back:disabled { background: #1e293b; color: #334155; cursor: not-allowed; }
    #btn-next { background: #3b82f6; }
    #btn-next:hover:not(:disabled) { background: #2563eb; }
    #btn-next:disabled { background: #1e293b; color: #334155; cursor: not-allowed; }
    #btn-save { background: #059669; }
    #btn-save:hover { background: #047857; }

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

    /* ── Toast ── */
    #toast {
      position: fixed;
      bottom: 60px;
      left: 50%;
      transform: translateX(-50%);
      background: #1e293b;
      color: #e2e8f0;
      padding: 9px 20px;
      border-radius: 8px;
      font-size: 0.82rem;
      opacity: 0;
      transition: opacity 0.25s;
      pointer-events: none;
      white-space: nowrap;
      z-index: 999;
    }
    #toast.show { opacity: 1; }
  </style>
</head>
<body>

<div id="header">
  <h1>QA HUMAN JUDGE</h1>
  <span id="trace-id">—</span>
  <span id="filename">Loading…</span>
  <div id="font-controls">
    <button class="font-btn" onclick="changeFontSize(-1)">A−</button>
    <span id="font-size-label">16px</span>
    <button class="font-btn" onclick="changeFontSize(+1)">A+</button>
  </div>
  <span id="progress">— / —</span>
</div>

<div id="app">

  <!-- ── Sidebar ── -->
  <div id="sidebar">
    <div id="sidebar-header">
      <span>QA Items</span>
      <button id="btn-expand" onclick="toggleExpand()">Expand</button>
    </div>
    <div id="sidebar-list"></div>
  </div>

  <!-- ── Survey pane ── -->
  <div id="survey-area">
    <div id="qa-rows"></div>
    <div id="done-msg">
      <h2>All items reviewed!</h2>
      <p>Click SAVE to write results to a file.</p>
    </div>
  </div>

</div>

<div id="footer">
  <span id="eval-status">—</span>
  <div class="footer-btns">
    <button id="btn-back" class="nav-btn" onclick="prevQA()">‹ BACK</button>
    <button id="btn-next" class="nav-btn" onclick="nextQA()">NEXT ›</button>
    <button id="btn-save" class="nav-btn" onclick="saveEvals()">SAVE</button>
  </div>
</div>

<div id="toast"></div>

<script>
const FIELDS = [
  { key: "question",          dim: null,                    label: "Question" },
  { key: "answer",            dim: "answer_completeness",   label: "Answer" },
  { key: "equipment_problem", dim: "context_clarity",       label: "Equipment Problem" },
  { key: "tools_required",    dim: "tool_realism",          label: "Tools Required" },
  { key: "steps",             dim: "scope_appropriateness", label: "Steps" },
  { key: "safety_info",       dim: "safety_specificity",    label: "Safety Info" },
  { key: "tips",              dim: "tip_usefulness",        label: "Tips" },
];
const DIMENSIONS = FIELDS.filter(f => f.dim).map(f => f.dim);

const CRITERIA = {
  answer_completeness:   "D1 · Answer Completeness — The answer contains enough detail for a homeowner to actually complete the repair end to end (tools, concrete steps, safety, a useful tip). Answers that stop short or omit key stages fail.",
  safety_specificity:    "D2 · Safety Specificity — safety_info names the specific hazard of this repair and the specific precaution to take. Generic phrases (\\"be careful\\", \\"use caution\\", \\"stay safe\\") fail.",
  tool_realism:          "D3 · Tool Realism — Every item in tools_required is something a typical homeowner already owns or could buy at a general hardware store for under $50. No professional, specialty, or trade-only tools.",
  scope_appropriateness: "D4 · Scope Appropriateness — The repair is within realistic DIY capability. If professional help is genuinely needed (e.g., gas lines, panel work), the answer says so clearly rather than giving amateur instructions.",
  context_clarity:       "D5 · Context Clarity — question and answer contain enough context to understand the problem, and the answer directly addresses the specific equipment_problem.",
  tip_usefulness:        "D6 · Tip Usefulness — tips provide non-obvious, task-specific advice that adds value beyond the steps. Tips that merely restate a step or offer generic encouragement fail.",
};

let currentIndex  = 0;
let currentTraceId = null;
let currentEval   = {};
let sidebarItems  = [];   // full list from /api/qa/list

// ── Helpers ──────────────────────────────────────────────────────────────────

function esc(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function formatValue(val) {
  if (Array.isArray(val)) {
    if (!val.length) return "<em>None</em>";
    return "<ul>" + val.map(v => `<li>${esc(v)}</li>`).join("") + "</ul>";
  }
  return esc(String(val));
}

function formatAnswer(val, steps) {
  let text = esc(String(val));
  const stepSet = new Set((steps || []).map(s => esc(String(s)).trim()));

  // Section headers on their own line (bold)
  text = text.replace(/\\s*(plan\\s*:)/gi, "<br><strong>$1</strong>");
  text = text.replace(/\\s*(safety(?:\\s+info)?\\s*:)/gi, "<br><strong>$1</strong>");
  text = text.replace(/\\s*(tools?(?:\\s+required)?\\s*:)/gi, "<br><strong>$1</strong>");
  text = text.replace(/\\s*(tips?\\s*:)/gi, "<br><strong>$1</strong>");
  text = text.replace(/\\s*(steps?\\s*:)/gi, "<br><strong>$1</strong>");

  // Highlight numbered steps only inside the Steps: section
  const HBREAK = '<br><strong>';
  const parts = text.split(HBREAK);
  for (let i = 1; i < parts.length; i++) {
    if (/^steps?\\s*:/i.test(parts[i])) {
      const closeTag = '</strong>';
      const closeIdx = parts[i].indexOf(closeTag);
      if (closeIdx !== -1) {
        const headerInner = parts[i].slice(0, closeIdx + closeTag.length);
        let sectionContent = parts[i].slice(closeIdx + closeTag.length);
        sectionContent = sectionContent.replace(/(?:<br>|\\s+)(\\d+[.)]\\s)([^]*?)(?=<br>|\\s+\\d+[.)]\\s|$)/g, function(_, num, content) {
          const stepText = content.trim();
          const bg = stepSet.has(stepText) ? '#dcfce7' : '#fee2e2';
          return '<br><span style="display:block;padding-left:1em;background:' + bg + '">' + num + stepText + '</span>';
        });
        parts[i] = headerInner + sectionContent;
      }
    }
  }
  text = parts.join(HBREAK);

  // Tip sentence not already caught by header (e.g. "Tip," or "Tip —")
  text = text.replace(/([.!?])\\s+([Tt]ip\\b(?!\\s*:))/g, "$1<br>$2");

  return text;
}

function dimTitle(dim) {
  return dim.replace(/_/g," ").replace(/\\b\\w/g, c => c.toUpperCase());
}

function countFullyEvaluated() {
  return sidebarItems.filter(item => item.vote_state === 'full').length;
}

function updateProgressBar() {
  const currentData = document.getElementById("progress");
  if (currentData.textContent === "— / —") return; // Not yet loaded
  const match = currentData.textContent.match(/^(\d+) \/ (\d+)/);
  if (!match) return;
  const fullyEvaluated = countFullyEvaluated();
  currentData.textContent = `${match[1]} / ${match[2]}  [${fullyEvaluated} fully evaluated]`;
}

// ── Sidebar ───────────────────────────────────────────────────────────────────

function loadSidebar() {
  fetch("/api/qa/list")
    .then(r => r.json())
    .then(items => {
      sidebarItems = items;
      renderSidebar();
    });
}

function renderSidebar() {
  const list = document.getElementById("sidebar-list");
  list.innerHTML = "";
  sidebarItems.forEach(item => {
    const div = document.createElement("div");
    div.className = "sidebar-item" + (item.index === currentIndex ? " active" : "");
    div.dataset.index = item.index;
    div.onclick = () => loadQA(item.index);

    const catClass = "cat-" + (item.category || "unknown");
    div.innerHTML = `
      <div class="item-meta">
        <span class="item-trace">${esc(item.trace_id)}</span>
        <span class="item-cat ${catClass}">${esc(item.category)}</span>
      </div>
      <div class="item-question">${esc(item.question)}</div>
      <span class="item-check ${item.vote_state === 'full' ? 'checked' : item.vote_state === 'partial' ? 'partial' : 'unchecked'}">${item.vote_state === 'full' ? '✓' : item.vote_state === 'partial' ? '⧄' : '☐'}</span>`;

    list.appendChild(div);
  });
  updateProgressBar();
}

function updateSidebarActive() {
  document.querySelectorAll(".sidebar-item").forEach(el => {
    el.classList.toggle("active", parseInt(el.dataset.index) === currentIndex);
  });
  // Scroll the active item into view
  const active = document.querySelector(".sidebar-item.active");
  if (active) active.scrollIntoView({ block: "nearest" });
}

function toggleExpand() {
  const sidebar = document.getElementById("sidebar");
  const btn = document.getElementById("btn-expand");
  sidebar.classList.toggle("expanded");
  btn.textContent = sidebar.classList.contains("expanded") ? "Collapse" : "Expand";
}

// ── Render survey ─────────────────────────────────────────────────────────────

function renderQA(data) {
  currentTraceId = data.trace_id;
  currentEval = {};

  const saved = data.eval || {};
  for (const dim of DIMENSIONS) {
    if (saved[dim] !== undefined && saved[dim] !== null) {
      currentEval[dim] = saved[dim];
    }
  }

  document.getElementById("trace-id").textContent  = data.trace_id;
  document.getElementById("filename").textContent   = data.filename;
  document.getElementById("progress").textContent   = `${data.index + 1} / ${data.total}`;
  updateProgressBar();
  document.getElementById("btn-back").disabled      = data.index === 0;
  document.getElementById("btn-next").disabled      = data.index + 1 >= data.total;
  document.getElementById("done-msg").style.display = "none";

  const container = document.getElementById("qa-rows");
  container.style.display = "";
  container.innerHTML = "";

  for (const field of FIELDS) {
    const row = document.createElement("div");
    row.className = "qa-row";

    const fieldDiv = document.createElement("div");
    fieldDiv.className = "qa-field";
    const content = field.key === "answer"
      ? formatAnswer(data.qa[field.key], data.qa.steps)
      : formatValue(data.qa[field.key]);
    fieldDiv.innerHTML = `<h3>${esc(field.label)}</h3><div class="content">${content}</div>`;

    const btnsDiv = document.createElement("div");
    btnsDiv.className = "qa-buttons";

    if (field.dim) {
      const cur = currentEval[field.dim];
      const passActive = cur === 1 ? "active" : "";
      const failActive = cur === 0 ? "active" : "";
      btnsDiv.innerHTML = `
        <span class="dim-label has-tooltip" data-tooltip="${esc(CRITERIA[field.dim] || '')}">${dimTitle(field.dim)}</span>
        <div class="btn-row">
          <button class="btn btn-pass ${passActive}" data-dim="${field.dim}"
                  onclick="setVote('${field.dim}', 1, this)">PASS</button>
          <button class="btn btn-fail ${failActive}" data-dim="${field.dim}"
                  onclick="setVote('${field.dim}', 0, this)">FAIL</button>
        </div>`;
    }

    row.appendChild(fieldDiv);
    row.appendChild(btnsDiv);
    container.appendChild(row);
  }

  updateStatus();
  updateSidebarActive();
  document.getElementById("survey-area").scrollTop = 0;
}

// ── Vote handling ─────────────────────────────────────────────────────────────

function setVote(dim, val, clickedBtn) {
  currentEval[dim] = val;
  document.querySelectorAll(`[data-dim="${dim}"]`).forEach(b => b.classList.remove("active"));
  clickedBtn.classList.add("active");
  updateStatus();
  autoSave();
}

function updateStatus() {
  const voted = DIMENSIONS.filter(d => currentEval[d] !== undefined).length;
  document.getElementById("eval-status").textContent =
    `${voted} / ${DIMENSIONS.length} dimensions rated`;
}

function autoSave() {
  if (!currentTraceId) return;
  fetch("/api/eval", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildPayload()),
  }).then(() => {
    const voted = DIMENSIONS.filter(d => currentEval[d] !== undefined).length;
    const state = voted === 0 ? 'none' : voted < DIMENSIONS.length ? 'partial' : 'full';
    const el = document.querySelector(`.sidebar-item[data-index="${currentIndex}"]`);
    if (el) {
      const check = el.querySelector('.item-check');
      if (check) {
        check.textContent = state === 'full' ? '✓' : state === 'partial' ? '⧄' : '☐';
        check.className = 'item-check ' + (state === 'full' ? 'checked' : state === 'partial' ? 'partial' : 'unchecked');
      }
    }
    // Update the sidebar state in memory and refresh progress bar
    const item = sidebarItems.find(i => i.index === currentIndex);
    if (item) {
      item.vote_state = state;
      updateProgressBar();
    }
  });
}

function buildPayload() {
  const dims = {};
  for (const dim of DIMENSIONS) {
    dims[dim] = currentEval[dim] !== undefined ? currentEval[dim] : null;
  }
  const allSet  = DIMENSIONS.every(d => dims[d] !== null);
  const allPass = allSet && DIMENSIONS.every(d => dims[d] === 1);
  return { trace_id: currentTraceId, labeler: "human", ...dims, overall_pass: allPass };
}

// ── Navigation ────────────────────────────────────────────────────────────────

function loadQA(idx) {
  fetch(`/api/qa/${idx}`)
    .then(r => r.json())
    .then(data => {
      if (data.done) { showDone(data.total); return; }
      currentIndex = data.index;
      renderQA(data);
    })
    .catch(err => showToast("Error: " + err));
}

function nextQA() { loadQA(currentIndex + 1); }
function prevQA() { loadQA(currentIndex - 1); }

function showDone(total) {
  document.getElementById("qa-rows").style.display = "none";
  document.getElementById("done-msg").style.display = "flex";
  document.getElementById("filename").textContent   = "All done";
  document.getElementById("btn-next").disabled      = true;
  document.getElementById("eval-status").textContent = `${total} items reviewed`;
}

// ── Save ──────────────────────────────────────────────────────────────────────

function saveEvals() {
  fetch("/api/save", { method: "POST" })
    .then(r => r.json())
    .then(data => {
      if (data.error) showToast("Save error: " + data.error);
      else showToast(`Saved ${data.count} evaluation(s) to ${data.path}`);
    })
    .catch(err => showToast("Save error: " + err));
}

// ── Toast ─────────────────────────────────────────────────────────────────────

let toastTimer = null;
function showToast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("show");
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 4000);
}

// ── Font size ─────────────────────────────────────────────────────────────────

let baseFontSize = 16;
function changeFontSize(delta) {
  baseFontSize = Math.min(26, Math.max(11, baseFontSize + delta));
  document.documentElement.style.fontSize = baseFontSize + "px";
  document.getElementById("font-size-label").textContent = baseFontSize + "px";
}

// ── Boot ──────────────────────────────────────────────────────────────────────
loadSidebar();
loadQA(0);
</script>
</body>
</html>
"""

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info(f"Logging to {_log_file}")
    versions = _detect_versions()
    if not versions:
        log.error(f"No version folders found in {QA_ITEMS_ROOT}")
        sys.exit(1)
    log.info(f"Found {len(versions)} version(s): {[v['version'] for v in versions]}")
    webbrowser.open("http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=False)
