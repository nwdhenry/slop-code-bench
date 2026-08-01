"""Collect combined SCBench and fault-typed harness results."""

from __future__ import annotations

import argparse
from pathlib import Path

from slop_code.fault_typed_results import collect_fault_typed_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scbench_run_dir", type=Path)
    arguments = parser.parse_args()
    combined = collect_fault_typed_results(arguments.scbench_run_dir)
    session = combined["session"]
    print(
        "Collected "
        f"{session['checkpoint_count']} checkpoints; "
        f"{session['checkpoints_passed']} externally passed."
    )


if __name__ == "__main__":
    main()
