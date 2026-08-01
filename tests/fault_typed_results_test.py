from __future__ import annotations

import hashlib
import json
from pathlib import Path

from slop_code.fault_typed_results import collect_fault_typed_results


def _write_checkpoint(
    root: Path,
    index: int,
    *,
    core_passed: int,
    core_total: int,
    regression_passed: int = 0,
    regression_total: int = 0,
    prompt_fingerprint: str | None = "prompt-1",
) -> None:
    checkpoint = root / f"checkpoint_{index}"
    harness = checkpoint / "agent" / "harness-run"
    harness.mkdir(parents=True)
    evaluation = {
        "pass_counts": {
            "Core": core_passed,
            "Regression": regression_passed,
            "Error": 0,
            "Functionality": 0,
        },
        "total_counts": {
            "Core": core_total,
            "Regression": regression_total,
            "Error": 0,
            "Functionality": 0,
        },
        "infrastructure_failure": False,
    }
    (checkpoint / "evaluation.json").write_text(
        json.dumps(evaluation), encoding="utf-8"
    )
    adapter = {
        "task_sha256": "task-hash",
        "harness_run_id": f"run-{index}",
        "harness_status": "COMPLETE",
        "harness_state": "COMPLETE",
        "accepted_checkpoint": f"checkpoint-{index}",
        "review_verdict": "APPROVE",
        "open_fault_count": 0,
        "model_calls": 1,
        "prompt_tokens": 100,
        "generated_tokens": 10,
        "model_seconds": 2.5,
        "wall_seconds": 3.0,
        "workspace_manifest_after": [],
        "run_dir": "harness-run",
    }
    (checkpoint / "agent" / "adapter-result.json").write_text(
        json.dumps(adapter), encoding="utf-8"
    )
    fault_records = {}
    if index == 2:
        fault_records["fault-1"] = {
            "fault_type": "PATCH_REJECTED",
            "fault_signature": "patch:anchor",
            "occurrence_count": 2,
            "status": "CLOSED",
        }
    report = {
        "run_id": f"run-{index}",
        "reentry": {"fault_records": fault_records},
    }
    (harness / "run-report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    payload = {
        "telemetry": {
            "prompt_tokens": 100,
            "generated_tokens": 10,
            "client_duration_seconds": 2.5,
        }
    }
    raw = json.dumps(payload).encode()
    digest = hashlib.sha256(raw).hexdigest()
    artifact = harness / "artifacts" / "sha256" / digest[:2] / digest[2:]
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(raw)
    events = [
        {
            "event_type": "ATTEMPT_STARTED",
            "stage": "IMPLEMENT",
        },
        {
            "event_type": "MODEL_RESPONSE",
            "stage": "IMPLEMENT",
            "prompt_fingerprint": prompt_fingerprint,
            "payload_ref": f"sha256:{digest}",
        },
        {
            "event_type": "CHECKPOINT_PROMOTED",
            "stage": "IMPLEMENT",
        },
    ]
    if index == 2:
        events.extend(
            [
                {
                    "event_type": "RECOVERY_CAPSULE_CREATED",
                    "stage": "IMPLEMENT",
                },
                {"event_type": "FAULT_VERIFIED", "stage": "IMPLEMENT"},
            ]
        )
    (harness / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def test_collector_joins_external_and_internal_results(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path, 1, core_passed=2, core_total=2)
    _write_checkpoint(
        tmp_path,
        2,
        core_passed=2,
        core_total=2,
        regression_passed=1,
        regression_total=1,
    )

    combined = collect_fault_typed_results(tmp_path)

    assert combined["session"] == {
        "checkpoints_passed": 2,
        "checkpoint_count": 2,
        "furthest_checkpoint": 2,
        "all_checkpoints_passed": True,
        "final_suite_passed": True,
        "first_regression_checkpoint": None,
        "cumulative_regression_failures": 0,
    }
    second = combined["checkpoints"][1]
    assert second["internal"]["recovery_attempt_count"] == 1
    assert second["internal"]["successful_recovery_count"] == 1
    assert second["stages"]["IMPLEMENT"]["tokens"] == 110
    assert second["faults"]["fault_recurrence_rate"] == 0.5
    assert not combined["integrity_hard_failure"]
    assert (tmp_path / "combined-results.json").is_file()
    assert (tmp_path / "combined-results.jsonl").is_file()
    assert "checkpoint_2" in (tmp_path / "campaign-summary.md").read_text(
        encoding="utf-8"
    )


def test_collector_marks_missing_prompt_fingerprint_as_hard_failure(
    tmp_path: Path,
) -> None:
    _write_checkpoint(
        tmp_path,
        1,
        core_passed=0,
        core_total=1,
        prompt_fingerprint=None,
    )

    combined = collect_fault_typed_results(tmp_path)

    integrity = combined["checkpoints"][0]["integrity"]
    assert integrity["missing_prompt_fingerprint"] == 1
    assert integrity["hard_failure"]
