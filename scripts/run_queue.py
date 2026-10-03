"""Run several training jobs, a few at a time (the overnight queue).

    python scripts/run_queue.py                     # default: main run + the 4 ablations, 2 at a time
    python scripts/run_queue.py --parallel 2 configs/base.yaml:A0_full configs/ablations/A1_no_curriculum.yaml:A1

Two runs side by side give ~1.5x the total env-steps/s of one run on the M1 Pro (one run's
gradient updates overlap the other's simulation), so the queue keeps `--parallel` jobs going.
Each job logs to runs/<name>.log. Keeps the Mac awake (caffeinate) until the queue is done.
Ctrl-C / kill stops all jobs; each one still saves its model (see train.py).
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_JOBS = [
    "configs/base.yaml:A0_full",
    "configs/ablations/A1_no_curriculum.yaml:A1_no_curriculum",
    "configs/ablations/A2_no_home.yaml:A2_no_home",
    "configs/ablations/A3_no_posture.yaml:A3_no_posture",
    "configs/ablations/A4_no_randomization.yaml:A4_no_randomization",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="*", default=DEFAULT_JOBS, help="config.yaml:run_name pairs, run in order")
    ap.add_argument("--parallel", type=int, default=2)
    args = ap.parse_args()

    pending = [j.split(":", 1) for j in args.jobs]
    for cfg, name in pending:
        if not (REPO / cfg).exists():
            sys.exit(f"missing config {cfg}")
        if (REPO / "runs" / name).exists():
            sys.exit(f"runs/{name} already exists; remove it or rename the job")

    def _interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _interrupt)   # also when started in the background with `&`
    signal.signal(signal.SIGTERM, _interrupt)

    awake = subprocess.Popen(["caffeinate", "-i"]) if sys.platform == "darwin" else None
    running: list[tuple[str, subprocess.Popen]] = []
    try:
        while pending or running:
            while pending and len(running) < args.parallel:
                cfg, name = pending.pop(0)
                log = open(REPO / "runs" / f"{name}.log", "w")
                cmd = [sys.executable, "scripts/train.py", "--config", cfg, "--name", name]
                running.append((name, subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)))
                print(f"{time.strftime('%H:%M:%S')} started {name} ({cfg})", flush=True)
            time.sleep(10)
            for name, p in list(running):
                if p.poll() is not None:
                    running.remove((name, p))
                    print(f"{time.strftime('%H:%M:%S')} finished {name} (exit {p.returncode})", flush=True)
    except KeyboardInterrupt:
        print("stopping all jobs (each saves its model)", flush=True)
        for _, p in running:
            p.terminate()
        for _, p in running:
            p.wait()
    finally:
        if awake:
            awake.terminate()


if __name__ == "__main__":
    main()
