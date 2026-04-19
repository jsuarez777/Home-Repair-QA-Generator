#!/usr/bin/env python3
"""Human judge Flask web app for evaluating QA items."""

import json
import sys
import webbrowser
from pathlib import Path
from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QA_VERSION = "v1"
QA_FOLDER = PROJECT_ROOT / f"qa_items/{QA_VERSION}"
EVAL_FILENAME = "QA_human_eval.json"

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
    "qa_files": [],
    "evaluations": {},   # trace_id -> eval dict
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


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML


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
        items.append({
            "index":    i,
            "trace_id": _trace_id(qa_file),
            "category": _category(qa_file),
            "filename": qa_file.name,
            "question": question,
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
    out = QA_FOLDER / EVAL_FILENAME
    out.write_text(json.dumps(evals, indent=2))
    return jsonify({"ok": True, "path": str(out), "count": len(evals)})


# ── HTML/CSS/JS ───────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
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
      padding: 10px 12px;
      cursor: pointer;
      border-bottom: 1px solid #0f172a;
      transition: background 0.1s;
    }
    .sidebar-item:hover   { background: #273548; }
    .sidebar-item.active  { background: #1d4ed8; }

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

function dimTitle(dim) {
  return dim.replace(/_/g," ").replace(/\\b\\w/g, c => c.toUpperCase());
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
      <div class="item-question">${esc(item.question)}</div>`;
    list.appendChild(div);
  });
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
    fieldDiv.innerHTML = `<h3>${esc(field.label)}</h3><div class="content">${formatValue(data.qa[field.key])}</div>`;

    const btnsDiv = document.createElement("div");
    btnsDiv.className = "qa-buttons";

    if (field.dim) {
      const cur = currentEval[field.dim];
      const passActive = cur === 1 ? "active" : "";
      const failActive = cur === 0 ? "active" : "";
      btnsDiv.innerHTML = `
        <span class="dim-label">${dimTitle(field.dim)}</span>
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
    state["qa_files"] = sorted(QA_FOLDER.glob("*.qa"))
    if not state["qa_files"]:
        print(f"No .qa files found in {QA_FOLDER}", file=sys.stderr)
        sys.exit(1)

    state["evaluations"] = _load_existing_evals(QA_FOLDER)

    print(f"Loaded {len(state['qa_files'])} QA file(s).")
    if state["evaluations"]:
        print(f"Resumed {len(state['evaluations'])} existing evaluation(s).")

    webbrowser.open("http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
