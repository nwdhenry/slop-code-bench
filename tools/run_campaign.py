"""Run a fixed fault-typed harness SCBench campaign."""

from __future__ import annotations

import argparse

from slop_code.fault_typed_campaign import run_campaign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--repetitions", required=True, type=int)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--model", required=True)
    arguments = parser.parse_args()
    result = run_campaign(
        campaign=arguments.campaign,
        problem=arguments.problem,
        repetitions=arguments.repetitions,
        agent=arguments.agent,
        model=arguments.model,
    )
    summary = result["summary"]
    print(
        f"Campaign completed: {summary['successful_full_sessions']}/"
        f"{summary['sessions_requested']} full clean sessions"
    )


if __name__ == "__main__":
    main()
