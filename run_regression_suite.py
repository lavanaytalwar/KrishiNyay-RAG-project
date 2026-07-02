"""
Run the KrishiNyay local regression gates in one command.

This wrapper is intentionally simple: each validator remains independently
usable, while this script provides the Phase 12 production-readiness gate.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Gate:
    name: str
    command: list[str]


def run_gate(gate: Gate, env: dict[str, str]) -> tuple[bool, float]:
    print("=" * 78)
    print(f"RUN: {gate.name}")
    print("CMD:", " ".join(gate.command))
    start = time.perf_counter()
    completed = subprocess.run(gate.command, cwd=ROOT, env=env)
    elapsed = time.perf_counter() - start
    status = "PASS" if completed.returncode == 0 else "FAIL"
    print(f"{status}: {gate.name} ({elapsed:.1f}s)")
    return completed.returncode == 0, elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-generation", action="store_true", help="Skip the Ollama generation gate")
    args = parser.parse_args()

    python = sys.executable
    env = os.environ.copy()
    env.setdefault("PYTHONPYCACHEPREFIX", "/private/tmp/krishinyay_pycache")
    runtime_chroma = Path(tempfile.gettempdir()) / "krishinyay_regression_chroma"
    if runtime_chroma.exists():
        shutil.rmtree(runtime_chroma)
    env.setdefault("CHROMA_RUNTIME_COPY", "true")
    env.setdefault("CHROMA_RUNTIME_PATH", str(runtime_chroma))
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")

    gates = [
        Gate("setup readiness", [python, "validate_setup_readiness.py"]),
        Gate("Python syntax", [python, "-m", "py_compile", "app.py", "rag_chain.py", "vector_store.py", "workflow.py", "live_data.py", "language_policy.py", "ocr_utils.py", "validate_corpus.py", "validate_farmer_eval.py", "validate_phase5_retrieval.py", "validate_ocr_pipeline.py", "validate_phase7_live.py", "validate_phase8_workflows.py", "validate_answer_quality.py", "validate_setup_readiness.py", "validate_public_demo.py"]),
        Gate("public demo dry-run", [python, "validate_public_demo.py"]),
        Gate("farmer eval", [python, "validate_farmer_eval.py"]),
        Gate("farmer eval spot-check", [python, "validate_farmer_eval.py", "--spot-check"]),
        Gate("hybrid retrieval", [python, "validate_phase5_retrieval.py"]),
        Gate("OCR pipeline", [python, "validate_ocr_pipeline.py"]),
        Gate("live data", [python, "validate_phase7_live.py"]),
        Gate("workflow", [python, "validate_phase8_workflows.py"]),
        Gate("answer quality", [python, "validate_answer_quality.py"]),
        Gate("corpus smoke", [python, "validate_corpus.py"]),
        Gate("frontend syntax", ["node", "--check", "web/app.js"]),
        Gate("diff whitespace", ["git", "diff", "--check"]),
    ]
    if not args.skip_generation:
        gates.insert(9, Gate("Ollama generation", [python, "validate_generation.py", "--provider", "ollama"]))

    failures = []
    total_time = 0.0
    for gate in gates:
        passed, elapsed = run_gate(gate, env)
        total_time += elapsed
        if not passed:
            failures.append(gate.name)

    print("=" * 78)
    print(f"Regression suite finished in {total_time:.1f}s")
    if failures:
        print("FAILED GATES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("OK: all regression gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
