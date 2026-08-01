"""Rebuild a fault-typed campaign scorecard from persisted evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from slop_code.fault_typed_campaign import recollect_campaign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_root", type=Path)
    arguments = parser.parse_args()
    campaign = recollect_campaign(arguments.campaign_root)
    summary = campaign["summary"]
    print(
        f"Recollected {summary['sessions_with_results']}/"
        f"{summary['sessions_requested']} sessions."
    )


if __name__ == "__main__":
    main()
