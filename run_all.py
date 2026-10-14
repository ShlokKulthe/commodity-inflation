"""
run_all.py - run the whole commodity-inflation pipeline (Phases 1-7) in order.

Usage (from the project folder):
    python run_all.py                  # run phases 1-7
    python run_all.py --skip-download  # reuse existing data, run phases 2-7
    python run_all.py --from 4         # run phases 4-7
    python run_all.py --only 5         # run a single phase
    python run_all.py --from 3 --to 5  # run phases 3-5

Script discovery: for phase N the runner looks for 0N_*.py, then PhaseN.py / phaseN.py,
so both the original file names and renamed ones work.

FRED key: set FRED_API_KEY as an environment variable, or put it in a file named .env
next to this script:   FRED_API_KEY=your32characterkey
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"

PHASES = {
    1: "Data collection",
    2: "EDA and stationarity tests",
    3: "Preprocessing",
    4: "Factor extraction",
    5: "Forecasting models",
    6: "Evaluation and robustness",
    7: "Interpretation and report",
}

# Files that must exist after a phase finishes successfully
EXPECTED = {
    1: ["data/processed/monthly_merged.csv"],
    2: ["data/processed/stationarity_report.csv"],
    3: ["data/processed/X_train_scaled.csv", "data/processed/targets.csv"],
    4: ["data/processed/factors_all.csv", "data/processed/factor_diagnostics.json"],
    5: ["data/processed/results_metrics.csv", "data/processed/forecasts_h1.csv"],
    6: ["data/processed/results_eval_all.csv"],
    7: ["reports/results_summary.md"],
}


def find_script(n: int):
    for pattern in (f"0{n}_*.py", f"Phase{n}.py", f"phase{n}.py", f"Phase_{n}.py"):
        hits = [p for p in sorted(ROOT.glob(pattern)) if p.name != Path(__file__).name]
        if hits:
            return hits[0]
    return None


def load_env_file():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def fred_key_ok(script: Path) -> bool:
    key = os.environ.get("FRED_API_KEY", "")
    if re.fullmatch(r"[a-z0-9]{32}", key):
        return True
    # maybe the key is pasted directly inside the script
    text = script.read_text(encoding="utf-8", errors="ignore")
    return bool(re.search(r"FRED_API_KEY\s*=.*[\"']([a-z0-9]{32})[\"']", text))


def run_phase(n: int, script: Path) -> bool:
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"phase{n}.log"
    env = os.environ.copy()
    env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"})
    start = time.time()
    print(f"\n{'=' * 70}\nPHASE {n}: {PHASES[n]}   ({script.name})\n{'=' * 70}")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, "-u", str(script)], cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        proc.wait()
    secs = time.time() - start
    if proc.returncode != 0:
        print(f"\n[FAILED] Phase {n} exited with code {proc.returncode}. See {log_path}")
        return False
    missing = [f for f in EXPECTED[n] if not (ROOT / f).exists()]
    if missing:
        print(f"\n[FAILED] Phase {n} finished but did not create: {missing}")
        return False
    print(f"\n[OK] Phase {n} finished in {secs:.0f}s")
    return True


def main():
    ap = argparse.ArgumentParser(description="Run the commodity inflation pipeline")
    ap.add_argument("--from", dest="start", type=int, default=1, help="first phase (1-7)")
    ap.add_argument("--to", dest="end", type=int, default=7, help="last phase (1-7)")
    ap.add_argument("--only", type=int, help="run only this phase")
    ap.add_argument("--skip-download", action="store_true", help="skip phase 1 (reuse data)")
    args = ap.parse_args()

    start, end = (args.only, args.only) if args.only else (args.start, args.end)
    if args.skip_download and start == 1:
        start = 2
    phases = [p for p in range(start, end + 1) if p in PHASES]
    if not phases:
        sys.exit("Nothing to run. Phases are numbered 1-7.")

    load_env_file()
    scripts = {}
    for n in phases:
        s = find_script(n)
        if s is None:
            sys.exit(f"Could not find a script for phase {n} (looked for 0{n}_*.py / Phase{n}.py) in {ROOT}")
        scripts[n] = s

    if 1 in phases and not fred_key_ok(scripts[1]):
        sys.exit("No valid FRED API key found.\n"
                 "  Get a free key at https://fred.stlouisfed.org (My Account -> API Keys), then either\n"
                 "  - create a file named .env next to run_all.py containing: FRED_API_KEY=your32characterkey\n"
                 "  - or set the FRED_API_KEY environment variable,\n"
                 "  - or run with --skip-download if data/processed/monthly_merged.csv already exists.")

    results, t0 = {}, time.time()
    for n in phases:
        ok = run_phase(n, scripts[n])
        results[n] = ok
        if not ok:
            break

    print(f"\n{'=' * 70}\nSUMMARY (total {time.time() - t0:.0f}s)\n{'=' * 70}")
    for n in phases:
        status = {True: "OK", False: "FAILED"}.get(results.get(n), "not run")
        print(f"  Phase {n}: {PHASES[n]:<32} {status}")
    if all(results.get(n) for n in phases):
        print("\nAll requested phases completed.")
        if 7 in phases:
            print("  Report   : reports/results_summary.md")
            print("  Figures  : reports/figures/")
            print("  Dashboard: streamlit run app.py")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
