#!/usr/bin/env python3
"""
Benchmark & Metrics Tracker for TechJam Conversational Search Agent.
- Default: Runs fast unit tests (< 0.05s) after every code change.
- With --eval / -e: Runs unit tests + executes 200-session local evaluator + renders human-readable report.
- With --report / -r: Renders human-readable report of the latest results.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CATALOG_PATH = DATA_DIR / "catalog.jsonl"
PUBLIC_SET_PATH = DATA_DIR / "public_set.jsonl"
RESULTS_PATH = ROOT_DIR / "results.json"
BASELINE_PATH = ROOT_DIR / "docs" / "baseline_results.json"
HISTORY_PATH = ROOT_DIR / ".benchmark_history.json"


def run_unit_tests() -> bool:
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                    1. RUNNING UNIT TESTS                    │")
    print("└─────────────────────────────────────────────────────────────┘")
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    result = subprocess.run(cmd, cwd=str(ROOT_DIR))
    if result.returncode != 0:
        print("\n❌ Unit tests failed! Fix failing tests before proceeding.")
        return False
    print("\n✅ All unit tests passed successfully!")
    return True


def run_local_evaluator() -> dict | None:
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│         2. EXECUTING LOCAL EVALUATOR (200 Sessions)         │")
    print("└─────────────────────────────────────────────────────────────┘")

    if not CATALOG_PATH.exists():
        print(f"\n⚠️  Catalog file not found at: {CATALOG_PATH}")
        print("To enable full 200-session evaluation:")
        print("  1. Download `catalog.jsonl.gz` from the GitHub release")
        print("  2. Run: gzip -dk catalog.jsonl.gz && mv catalog.jsonl data/catalog.jsonl\n")
        return None

    print("Evaluating 200 public sessions... (this may take a few seconds)")
    cmd = [
        sys.executable,
        "-m",
        "evaluator.local_evaluator",
        "--catalog",
        str(CATALOG_PATH),
        "--dataset",
        str(PUBLIC_SET_PATH),
        "--output",
        str(RESULTS_PATH),
    ]
    result = subprocess.run(cmd, cwd=str(ROOT_DIR), stdout=subprocess.DEVNULL)
    if result.returncode != 0:
        print("\n❌ Local evaluator execution failed!")
        return None

    if not RESULTS_PATH.exists():
        print("\n❌ results.json was not found.")
        return None

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _progress_bar(value: float, total: float = 1.0, length: int = 20) -> str:
    fraction = max(0.0, min(1.0, value / total if total > 0 else 0.0))
    filled = int(round(fraction * length))
    bar = "█" * filled + "░" * (length - filled)
    return f"[{bar}] {fraction * 100:>5.1f}%"


def format_human_report(current: dict) -> None:
    baseline = {}
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH, "r", encoding="utf-8") as f:
            baseline = json.load(f)

    prev = {}
    if HISTORY_PATH.exists():
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                prev = json.load(f)
        except Exception:
            pass

    hit_rate = float(current.get("hit_rate_at_10", 0.0))
    mrr = float(current.get("mrr", 0.0))
    mttc = float(current.get("mttc", 0.0))
    efficiency = float(current.get("efficiency", 0.0))
    tech_score = float(current.get("recommended_technical_score", 0.0))
    sample_count = current.get("sample_count", 200)

    print("\n" + "=" * 65)
    print("🏆  TECHJAM EVALUATION SCORECARD")
    print("=" * 65)
    print(f" Total Sessions Evaluated: {sample_count}")
    print(f" Overall Technical Score : {tech_score:.4f} / 1.0000  {_progress_bar(tech_score)}")
    print("=" * 65)

    print("\n📊 CORE METRICS SUMMARY")
    print("┌───────────────────────────┬──────────┬──────────┬──────────┬─────────────────┐")
    print("│ Metric                    │ Baseline │ Previous │ Current  │ Delta (vs Base) │")
    print("├───────────────────────────┼──────────┼──────────┼──────────┼─────────────────┤")

    metric_rows = [
        ("Hit Rate@10 (Recall)", "hit_rate_at_10", hit_rate, baseline.get("hit_rate_at_10"), prev.get("hit_rate_at_10"), True, True),
        ("MRR (Precision Rank 1)", "mrr", mrr, baseline.get("mrr"), prev.get("mrr"), False, True),
        ("MTTC (Turns to Convert)", "mttc", mttc, baseline.get("mttc"), prev.get("mttc"), False, False),
        ("Efficiency Score", "efficiency", efficiency, baseline.get("efficiency"), prev.get("efficiency"), True, True),
        ("Final Technical Score", "recommended_technical_score", tech_score, baseline.get("technical_score"), prev.get("recommended_technical_score"), False, True),
    ]

    for label, key, curr_v, base_v, prev_v, as_pct, higher_better in metric_rows:
        def fmt(v):
            if v is None:
                return "   —    "
            return f"{v * 100:>6.1f}% " if as_pct else f"{v:>7.4f} "

        delta_str = "      —       "
        if base_v is not None:
            delta = curr_v - float(base_v)
            sign = "+" if delta > 0 else ""
            d_formatted = f"{sign}{delta * 100:.1f}%" if as_pct else f"{sign}{delta:.4f}"
            is_good = (delta > 0 and higher_better) or (delta < 0 and not higher_better)
            icon = "🚀" if is_good else ("⚠️" if delta != 0 else "  ")
            delta_str = f"{d_formatted:>9} {icon}"

        print(f"│ {label:<25} │ {fmt(base_v)} │ {fmt(prev_v)} │ {fmt(curr_v)} │ {delta_str:<15} │")

    print("└───────────────────────────┴──────────┴──────────┴──────────┴─────────────────┘")

    # Scenario Breakdown
    scenarios = current.get("scenario_metrics", {})
    if scenarios:
        print("\n🎭 SCENARIO-BY-SCENARIO BREAKDOWN")
        print("┌──────────────────┬─────────┬──────────────┬────────────┬─────────────┐")
        print("│ Scenario Type    │ Samples │ Hit Rate@10  │ MRR        │ Avg Turns   │")
        print("├──────────────────┼─────────┼──────────────┼────────────┼─────────────┤")
        scenario_labels = {
            "buying": "Buying (40%)",
            "browsing": "Browsing (40%)",
            "intent_override": "Override (15%)",
            "boundary": "Boundary (5%)",
        }
        for sc_key in ["buying", "browsing", "intent_override", "boundary"]:
            sc_data = scenarios.get(sc_key)
            if not sc_data:
                continue
            name = scenario_labels.get(sc_key, sc_key)
            cnt = sc_data.get("sample_count", 0)
            s_hit = sc_data.get("hit_rate_at_10", 0.0)
            s_mrr = sc_data.get("mrr", 0.0)
            s_mttc = sc_data.get("mttc", 0.0)
            print(f"│ {name:<16} │ {cnt:>7} │ {s_hit * 100:>10.1f}%  │ {s_mrr:>10.4f} │ {s_mttc:>8.2f} trn │")
        print("└──────────────────┴─────────┴──────────────┴────────────┴─────────────┘")

    # Token Usage & Feasibility
    tokens = current.get("reported_token_usage", {})
    total_tokens = tokens.get("total_tokens", 0)
    print(f"\n💡 Execution Stats: In-memory deterministic | Reported LLM Tokens: {total_tokens}")
    print("=" * 65)

    # Save to history
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="TechJam Testing & Benchmark Suite")
    parser.add_argument(
        "--eval",
        "-e",
        action="store_true",
        help="Run unit tests AND full 200-session local evaluation",
    )
    parser.add_argument(
        "--report",
        "-r",
        action="store_true",
        help="Display human-readable scorecard from existing results.json without re-running",
    )
    args = parser.parse_args()

    if args.report:
        if not RESULTS_PATH.exists():
            print(f"❌ No existing results.json found at {RESULTS_PATH}. Run with --eval first.")
            sys.exit(1)
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            format_human_report(json.load(f))
        return

    # 1. Always run unit tests
    if not run_unit_tests():
        sys.exit(1)

    # 2. Only run evaluator if explicitly requested via --eval / -e
    if args.eval:
        eval_result = run_local_evaluator()
        if eval_result:
            format_human_report(eval_result)
    else:
        print("\n💡 Unit tests passed! (Run with `python3 scripts/benchmark.py --eval` when ready for full 200-session evaluation)")


if __name__ == "__main__":
    main()
