"""
server.py
---------
Author : Learner transitioning into Agentic AI Engineering
Purpose: Web server + autonomous folder watcher (Option 4).

Two systems running simultaneously
------------------------------------

1. FLASK WEB SERVER (interactive)
   Serves the browser UI and handles user-initiated requests.
   Routes: /upload, /ask, /analyse, /status, /reset, /reports

2. FOLDER WATCHER (autonomous, Option 4)
   Uses the watchdog library to monitor the watched/ directory.
   When a new .xlsx, .xls, or .csv file appears:
     a. The file is automatically parsed
     b. The agent runs a full autonomous analysis
     c. The report is printed to terminal with a timestamp
     d. The report is saved as a .txt file in reports/
   No human trigger required. This is what "autonomous" means in practice.

How this relates to agentic AI engineering
-------------------------------------------
The folder watcher is a simple form of an "event-driven agent trigger."
Production agentic systems use similar patterns with cloud storage events
(S3 triggers, GCS Pub/Sub) or message queues (Kafka, RabbitMQ) instead
of a local folder. The agent logic is identical — only the trigger changes.

Setup
------
1. Get a free Groq key: https://console.groq.com/keys
2. pip install groq flask pandas openpyxl chromadb watchdog
3. export GROQ_API_KEY=gsk_...
4. python server.py
5. Open http://localhost:5000

To test the folder watcher
---------------------------
With the server running, copy or move any financial Excel or CSV file
into the watched/ directory. The server will detect it automatically
and produce a report within seconds.
"""

import os
import time
import threading
import traceback
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

WATCHED_DIR = Path("watched")
REPORTS_DIR = Path("reports")
UPLOAD_DIR  = Path("uploads")

for d in (WATCHED_DIR, REPORTS_DIR, UPLOAD_DIR):
    d.mkdir(exist_ok=True)

SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv"}

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

# Shared in-memory log of auto-generated reports
# (accessible via /reports endpoint for the UI)
_auto_reports: list = []
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            return None
        from analyst_agent import CorporateAnalystAgent
        _agent = CorporateAnalystAgent(api_key=api_key)
    return _agent


# ---------------------------------------------------------------------------
# Autonomous report generation
# Called by both the folder watcher and the /analyse route
# ---------------------------------------------------------------------------

def generate_and_save_report(filepath: str, triggered_by: str = "watcher") -> dict:
    """
    Parse a financial file, run a full autonomous analysis,
    print the report to terminal, and save it to reports/.

    Returns a dict with report metadata and the full text.
    """
    agent = get_agent()
    if not agent:
        msg = "Cannot generate report: GROQ_API_KEY not set."
        print(f"\n[WATCHER] {msg}")
        return {"error": msg}

    filename  = Path(filepath).name
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts_file   = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*65}")
    print(f"  AUTONOMOUS ANALYSIS TRIGGERED")
    print(f"  File      : {filename}")
    print(f"  Triggered : {triggered_by}")
    print(f"  Time      : {timestamp}")
    print(f"{'='*65}")

    try:
        # Parse the file
        summary = agent.load_file(filepath)
        detected = ", ".join(summary.get("detected", []))
        print(f"  Detected  : {detected}")
        print(f"  Running full analysis via Groq LLaMA 3.3 70B...")

        # Run autonomous analysis (agent decides what to analyse)
        result = agent.full_analysis()
        answer = result.get("answer", "No answer returned.")
        tools  = result.get("tools_used", [])
        topics = result.get("rag_topics", [])

        # Print full report to terminal
        print(f"\n{'─'*65}")
        print(f"  ANALYSIS REPORT: {filename}")
        print(f"{'─'*65}\n")
        print(answer)
        print(f"\n{'─'*65}")
        print(f"  Tools used  : {', '.join(set(tools)) if tools else 'none'}")
        print(f"  RAG topics  : {', '.join(topics[:3]) if topics else 'none'}")
        print(f"{'─'*65}\n")

        # Save report as .txt file
        report_filename = f"{ts_file}_{Path(filename).stem}_report.txt"
        report_path     = REPORTS_DIR / report_filename

        report_content = (
            f"CORPORATE FINANCIAL ANALYST -- AUTONOMOUS REPORT\n"
            f"{'='*65}\n"
            f"File          : {filename}\n"
            f"Triggered by  : {triggered_by}\n"
            f"Generated at  : {timestamp}\n"
            f"Statements    : {detected}\n"
            f"Tools used    : {', '.join(set(tools)) if tools else 'none'}\n"
            f"RAG topics    : {', '.join(topics) if topics else 'none'}\n"
            f"{'='*65}\n\n"
            f"{answer}\n"
        )

        report_path.write_text(report_content, encoding="utf-8")
        print(f"  Report saved: {report_path.resolve()}")

        report_meta = {
            "filename":       filename,
            "report_file":    report_filename,
            "generated_at":   timestamp,
            "triggered_by":   triggered_by,
            "detected":       summary.get("detected", []),
            "tools_used":     list(set(tools)),
            "rag_topics":     topics,
            "answer":         answer,
        }
        _auto_reports.append(report_meta)
        return report_meta

    except Exception as exc:
        error_msg = f"Analysis failed: {exc}"
        print(f"\n[ERROR] {error_msg}")
        print(traceback.format_exc())
        return {"error": error_msg, "filename": filename}


# ---------------------------------------------------------------------------
# Folder watcher (Option 4 -- autonomous trigger)
# ---------------------------------------------------------------------------

class FinancialFileHandler(FileSystemEventHandler):
    """
    Watchdog event handler for the watched/ directory.

    Learning note: watchdog uses the observer pattern. When a file system
    event occurs, watchdog calls the appropriate on_* method. We override
    on_created and on_moved to catch new files appearing in the directory.
    """

    def __init__(self):
        super().__init__()
        # Track recently processed files to avoid duplicate triggers
        self._processed: set = set()
        self._lock = threading.Lock()

    def _handle(self, path: str):
        ext = Path(path).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return

        with self._lock:
            if path in self._processed:
                return
            self._processed.add(path)

        # Small delay to ensure the file is fully written before reading
        time.sleep(1.5)

        if not Path(path).exists():
            return

        # Run analysis in a background thread so the watcher is not blocked
        thread = threading.Thread(
            target=generate_and_save_report,
            args=(path, "folder watcher (automated)"),
            daemon=True,
        )
        thread.start()

    def on_created(self, event):
        if not event.is_directory:
            self._handle(event.src_path)

    def on_moved(self, event):
        # Handles files moved/copied into the watched folder
        if not event.is_directory:
            self._handle(event.dest_path)


def start_folder_watcher():
    """
    Start the watchdog observer in a background daemon thread.
    The observer monitors WATCHED_DIR for new financial files.
    """
    handler  = FinancialFileHandler()
    observer = Observer()
    observer.schedule(handler, str(WATCHED_DIR), recursive=False)
    observer.start()
    print(f"  Folder watcher active: {WATCHED_DIR.resolve()}")
    print(f"  Drop a financial file there for automatic analysis.")
    return observer


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Accept a file upload from the browser, parse it, return detected statements."""
    agent = get_agent()
    if not agent:
        return jsonify({
            "error": (
                "GROQ_API_KEY is not set.\n\n"
                "1. Get a free key at https://console.groq.com/keys\n"
                "2. Set it:  export GROQ_API_KEY=gsk_...\n"
                "3. Restart: python server.py"
            )
        }), 500

    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    f   = request.files["file"]
    ext = Path(f.filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type '{ext}'. Use .xlsx, .xls, or .csv."}), 400

    save_path = UPLOAD_DIR / f.filename
    f.save(str(save_path))

    try:
        summary = agent.load_file(str(save_path))
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/ask", methods=["POST"])
def ask():
    """Send a question to the agent, return analysis + tools used + RAG topics."""
    agent = get_agent()
    if not agent:
        return jsonify({"error": "GROQ_API_KEY not set."}), 500

    body     = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided."}), 400

    try:
        result = agent.ask(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/analyse", methods=["POST"])
def auto_analyse():
    """
    Trigger a full autonomous analysis from the browser.
    Also saves a report file (same as folder watcher behaviour).
    """
    agent = get_agent()
    if not agent:
        return jsonify({"error": "GROQ_API_KEY not set."}), 500
    if not agent.file_info():
        return jsonify({"error": "No file loaded."}), 400

    # Run analysis and save report
    filepath = str(UPLOAD_DIR / agent.filename)
    if not Path(filepath).exists():
        # Fall back to just running the analysis without saving
        try:
            result = agent.full_analysis()
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    report = generate_and_save_report(filepath, triggered_by="browser (manual)")
    if "error" in report:
        return jsonify(report), 500

    return jsonify({
        "answer":      report["answer"],
        "tools_used":  report.get("tools_used", []),
        "rag_topics":  report.get("rag_topics", []),
        "report_file": report.get("report_file"),
    })


@app.route("/status")
def status():
    agent = get_agent()
    if not agent:
        return jsonify({"ready": False, "reason": "GROQ_API_KEY not set"})
    return jsonify({
        "ready":        agent.file_info() is not None,
        "file":         agent.file_info(),
        "model":        GROQ_MODEL,
        "rag_stats":    agent.rag_stats(),
        "reports_count": len(_auto_reports),
        "watched_dir":  str(WATCHED_DIR.resolve()),
        "reports_dir":  str(REPORTS_DIR.resolve()),
    })


@app.route("/reports")
def list_reports():
    """Return metadata for all auto-generated reports (for the UI reports panel)."""
    # Include reports from both sources: in-memory list and reports/ directory
    report_files = sorted(REPORTS_DIR.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    file_reports = []
    for rp in report_files:
        file_reports.append({
            "report_file":  rp.name,
            "generated_at": datetime.fromtimestamp(rp.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "size_kb":      round(rp.stat().st_size / 1024, 1),
        })
    return jsonify({"reports": file_reports, "count": len(file_reports)})


@app.route("/reports/<filename>")
def get_report(filename: str):
    """Return the full text of a specific report file."""
    report_path = REPORTS_DIR / filename
    if not report_path.exists() or report_path.suffix != ".txt":
        return jsonify({"error": "Report not found."}), 404
    content = report_path.read_text(encoding="utf-8")
    return jsonify({"filename": filename, "content": content})


@app.route("/reset", methods=["POST"])
def reset():
    agent = get_agent()
    if agent:
        agent.reset_conversation()
    return jsonify({"ok": True})


GROQ_MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    api_key = os.environ.get("GROQ_API_KEY", "").strip()

    print("")
    print("  Corporate Financial Analyst -- AI System")
    print("  Option 3: RAG-Augmented Analysis")
    print("  Option 4: Autonomous Folder Watcher + Scheduled Reporting")
    print("  " + "-" * 52)

    if not api_key:
        print("")
        print("  WARNING: GROQ_API_KEY is not set.")
        print("")
        print("  Steps:")
        print("    1. Visit https://console.groq.com/keys")
        print("    2. Create a free account and generate a key")
        print("    3. export GROQ_API_KEY=gsk_...")
        print("    4. Restart: python server.py")
        print("")
    else:
        key_preview = api_key[:14] + "..." if len(api_key) > 14 else api_key
        print(f"  Groq API key : {key_preview}")
        print(f"  Model        : llama-3.3-70b-versatile")
        print(f"  RAG database : ./rag_db  (auto-seeded on first run)")
        print(f"  Watch folder : {WATCHED_DIR.resolve()}")
        print(f"  Reports dir  : {REPORTS_DIR.resolve()}")
        print(f"  Browser UI   : http://localhost:5000")
        print("")

    # Start the autonomous folder watcher
    observer = start_folder_watcher()

    try:
        app.run(debug=False, host="0.0.0.0", port=5000)
    finally:
        observer.stop()
        observer.join()