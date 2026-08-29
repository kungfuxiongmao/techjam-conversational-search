---
name: eval-benchmark
description: Automates unit testing, local evaluation on the 200-session public set, and scenario-by-scenario metrics reporting for the TechJam conversational search agent. Use after making code changes or enhancements to verify score gains and prevent regressions.
---

# Eval & Benchmark Skill

Use this skill to automate testing, incremental verification, and human-readable scorecards for the conversational shopping agent.

## Workflow Modes

### 1. Fast Unit Test (Default - After Every Code Change)
Runs the lightweight unit test suite in $< 0.05$ seconds:
```bash
python3 scripts/benchmark.py
```
* Use this immediately after any code edit to ensure no syntax or logic breaks.

### 2. Full Local Evaluation (After Completing an Increment / Feature)
Runs unit tests + executes the full 200-session local evaluator + renders the human-readable scorecard:
```bash
python3 scripts/benchmark.py --eval
```
*(or short form: `python3 scripts/benchmark.py -e`)*

### 3. View Latest Scorecard (Without Re-running)
Formats and displays the last generated `results.json`:
```bash
python3 scripts/benchmark.py --report
```

## Scorecard Structure

The human-readable output presents:
1. **Executive Summary**: Technical Score out of 1.0000 with visual progress bar.
2. **Core Metrics Comparison Table**: Hit Rate@10, MRR, MTTC, and Efficiency compared against Baseline and Previous run with deltas (`+82.0% 🚀`).
3. **Scenario Breakdown**: Buying, Browsing, Intent Override, and Boundary session hit rates and mean turns.
4. **Feasibility Stats**: Token usage and execution latency.
