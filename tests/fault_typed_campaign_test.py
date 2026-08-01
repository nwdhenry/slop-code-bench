from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from slop_code import fault_typed_campaign
from slop_code.fault_typed_campaign import _campaign_summary
from slop_code.fault_typed_campaign import _scbench_command
from slop_code.fault_typed_campaign import recollect_campaign

if TYPE_CHECKING:
    import pytest


def _combined(*, all_passed: bool, fault_type: str | None) -> dict:
    terminal_fault = None if fault_type is None else {"fault_type": fault_type}
    return {
        "session": {
            "checkpoint_count": 1,
            "checkpoints_passed": int(all_passed),
            "all_checkpoints_passed": all_passed,
            "furthest_checkpoint": int(all_passed),
        },
        "integrity_hard_failure": False,
        "checkpoints": [
            {
                "external": {
                    "regression_total": 2,
                    "regression_passed": 1 if not all_passed else 2,
                },
                "internal": {
                    "harness_terminal_status": "COMPLETE"
                    if all_passed
                    else "BLOCKED",
                    "review_verdict": "APPROVE" if all_passed else None,
                    "open_fault_count": int(not all_passed),
                    "recovery_attempt_count": int(not all_passed),
                    "successful_recovery_count": 0,
                    "prompt_tokens": 100,
                    "generated_tokens": 10,
                    "model_seconds": 2.0,
                    "terminal_fault": terminal_fault,
                },
                "faults": {
                    "by_fault_type": {}
                    if fault_type is None
                    else {fault_type: 1}
                },
            }
        ],
    }


def test_scbench_command_is_local_fixed_and_has_no_timeout(
    tmp_path: Path,
) -> None:
    command = _scbench_command(
        session_root=tmp_path,
        problem="file_backup",
        seed=7,
        agent="fault_typed_harness",
        model="qwen-local",
    )

    assert command[1:4] == ["--seed", "7", "run"]
    assert "--problem" in command and "file_backup" in command
    assert "--agent" in command and "fault-typed-harness" in command
    assert "local_llama_cpp/qwen-local" in command
    assert not any("timeout" in argument.lower() for argument in command)


def test_campaign_summary_retains_successes_and_failures(
    tmp_path: Path,
) -> None:
    first = tmp_path / "r1.json"
    second = tmp_path / "r2.json"
    first.write_text(json.dumps(_combined(all_passed=True, fault_type=None)))
    second.write_text(
        json.dumps(_combined(all_passed=False, fault_type="PATCH_REJECTED"))
    )
    manifest = {
        "repetitions": 2,
        "sessions": [
            {"result": first.name},
            {"result": second.name},
        ],
    }

    summary = _campaign_summary(tmp_path, manifest)

    assert summary["sessions_with_results"] == 2
    assert summary["successful_full_sessions"] == 1
    assert summary["checkpoint_pass_rate"] == 0.5
    assert summary["checkpoint_survival_curve"] == {"1": 1}
    assert summary["regression_rate"] == 0.25
    assert summary["clean_harness_completion_rate"] == 0.5
    assert summary["fault_distribution"] == {"PATCH_REJECTED": 1}
    assert summary["terminal_failure_clusters"] == {"PATCH_REJECTED": 1}
    assert summary["tokens_per_checkpoint"] == 110
    assert summary["tokens_per_successful_session"] == 110


def test_recollect_campaign_recovers_prior_collection_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = "campaign-r01-seed7"
    manifest: dict[str, Any] = {
        "campaign_id": "campaign",
        "problem": "file_backup",
        "repetitions": 1,
        "treatment": {"fingerprint": "frozen"},
        "sessions": [
            {
                "session_id": session_id,
                "exit_code": 0,
                "error": "missing adapter result",
            }
        ],
    }
    (tmp_path / "campaign.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    def fake_collect(problem_dir: Path) -> dict[str, Any]:
        combined = _combined(all_passed=False, fault_type="INFRA")
        problem_dir.mkdir(parents=True)
        (problem_dir / "combined-results.json").write_text(
            json.dumps(combined), encoding="utf-8"
        )
        return combined

    monkeypatch.setattr(
        fault_typed_campaign, "collect_fault_typed_results", fake_collect
    )

    result = recollect_campaign(tmp_path)

    session = result["sessions"][0]
    assert session["error"] is None
    assert session["prior_collection_error"] == "missing adapter result"
    assert session["result"].endswith("combined-results.json")
    assert result["summary"]["sessions_with_results"] == 1
