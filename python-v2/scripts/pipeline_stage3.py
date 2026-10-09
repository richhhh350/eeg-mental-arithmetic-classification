"""One-shot local pipeline: resume frozen training, audit, export and report.

Run from python-v2 after resolving any interrupted partial fold directories.
The process can run independently of an interactive terminal; no scheduling,
GitHub publishing, notifications or new experiment selection is performed.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs"
    lock = output / "stage3-pipeline.lock"
    status_path = output / "stage3-pipeline-status.json"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    status = {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "completed_steps": []}
    os.write(descriptor, json.dumps(status).encode("utf-8"))
    os.close(descriptor)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    steps = [
        ("training", ["-m", "eeg_v2.stage3", "run", "--prepared", "outputs/prepared-stage3", "--output", "outputs/runs-stage3", "--workers", "4", "--resume"]),
        ("independent_audit", ["scripts/audit_stage3.py"]),
        ("export_and_report", ["scripts/report_stage3.py"]),
    ]
    def save():
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        temporary.replace(status_path)
    try:
        for step, arguments in steps:
            status["step"] = step
            save()
            print(f"Starting {step}", flush=True)
            subprocess.run([sys.executable, *arguments], cwd=root, env=env, check=True)
            status["completed_steps"].append(step)
        status.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        print("Stage3 training, independent audit and report complete.", flush=True)
    except BaseException as error:
        status.update(status="failed", error=repr(error), traceback=traceback.format_exc(), finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
