"""
Unified runner for the full J-Lens Safety Study.

Phases:
  1. Safety Gap Analysis — does J-Lens readout degrade on safety inputs?
  2. Deception Detection — can trajectory patterns identify deceptive intent?
  3. Cross-Architecture — is the proximity artifact general?

Usage:
  python run_full_study.py [phase]
    phase 1   — Safety gap only
    phase 2   — Deception detection only
    phase 3   — Cross-architecture only
    all       — Run all phases
    (empty)   — Run all phases
"""

import sys, os, time
from pathlib import Path


def _run_phase(mod_name, title):
    print("\n" + "#" * 70)
    print(f"# {title}")
    print("#" * 70)
    try:
        mod = __import__(mod_name, fromlist=["main"])
        phase_main = mod.main
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"Failed to load {mod_name}.main: {e}")
    phase_main()


def run_phase1():
    _run_phase("safety_gap", "PHASE 1: Safety-Specific Readout Validation")


def run_phase2():
    _run_phase("deception_detector", "PHASE 2: Deception Detection via Trajectory Analysis")


def run_phase3():
    _run_phase("cross_arch", "PHASE 3: Cross-Architecture Scaling")


def main():
    phase = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    t0 = time.time()
    print("=" * 70)
    print("J-LENS SAFETY STUDY — FULL PIPELINE")
    print("=" * 70)
    print(f"Phase: {phase}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    runners = {"1": run_phase1, "2": run_phase2, "3": run_phase3,
                 "phase1": run_phase1, "phase2": run_phase2, "phase3": run_phase3,
                 "safety_gap": run_phase1, "deception": run_phase2, "cross_arch": run_phase3}
    if phase == "all":
        todo = ["1", "2", "3"]
    elif phase in runners:
        todo = [phase]
    else:
        print(f"Unknown phase: {phase}")
        print("Usage: python run_full_study.py [1|2|3|all]")
        sys.exit(1)

    ran_phases, failed = [], []
    for p in todo:
        key = p if p in ("1", "2", "3") else {"phase1": "1", "phase2": "2", "phase3": "3"}.get(p, p)
        try:
            runners[p](); ran_phases.append(key)
        except Exception as e:
            print(f"\n*** Phase {key} FAILED: {e} ***")
            failed.append(key)
            # in 'all' mode keep going so remaining phases still run

    dt = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"STUDY {'COMPLETE' if not failed else 'FINISHED WITH FAILURES'} — {dt:.0f}s ({dt/60:.1f}min)")
    print(f"{'=' * 70}")
    phase_dirs = {"1": "phase1_safety_gap", "2": "phase2_deception", "3": "phase3_cross_arch"}
    for p in ran_phases:
        print(f"  results/{phase_dirs.get(p, p)}/")
    if failed:
        print(f"  FAILED: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
