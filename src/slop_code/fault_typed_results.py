"""Join SCBench correctness with fault-typed harness audit evidence."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

CATEGORIES = ("Core", "Regression", "Error", "Functionality")


def collect_fault_typed_results(run_dir: Path) -> dict[str, Any]:
    """Collect one SCBench session and write its combined scorecard."""
    root = run_dir.resolve()
    checkpoint_names = _checkpoint_names(root)
    if not checkpoint_names:
        raise ValueError(f"No SCBench checkpoint directories found in {root}")
    checkpoints = [
        _collect_checkpoint(root / name) for name in checkpoint_names
    ]
    session = _session_metrics(checkpoints)
    combined = {
        "schema_version": 1,
        "run_dir": str(root),
        "session": session,
        "checkpoints": checkpoints,
        "integrity_hard_failure": any(
            checkpoint["integrity"]["hard_failure"]
            for checkpoint in checkpoints
        ),
    }
    _write_json(root / "combined-results.json", combined)
    with (root / "combined-results.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for checkpoint in checkpoints:
            handle.write(json.dumps(checkpoint, sort_keys=True) + "\n")
    (root / "campaign-summary.md").write_text(
        _render_summary(combined), encoding="utf-8", newline="\n"
    )
    return combined


def _collect_checkpoint(path: Path) -> dict[str, Any]:
    evaluation = _read_json(path / "evaluation.json", required=False)
    adapter_path = path / "agent" / "adapter-result.json"
    adapter = _read_json(adapter_path, required=False)
    harness_dir = path / "agent" / str(adapter.get("run_dir", "harness-run"))
    if not adapter:
        harness_dir = _external_harness_dir(path)
    report = _read_json(harness_dir / "run-report.json", required=False)
    state = _read_json(harness_dir / "state.json", required=False)
    domain = report or state
    events = _read_jsonl(harness_dir / "events.jsonl", required=False)
    if not adapter:
        adapter = _partial_adapter(path, domain, events, harness_dir)
    external = _external_metrics(evaluation)
    if adapter.get("adapter_infrastructure_failure"):
        external["infrastructure_failure"] = True
        external["checkpoint_pass"] = False
    internal = _internal_metrics(adapter, domain, events, harness_dir)
    integrity = _integrity_metrics(adapter, report, events)
    return {
        "checkpoint": path.name,
        "external": external,
        "internal": internal,
        "stages": _stage_metrics(events, harness_dir),
        "faults": _fault_metrics(domain, events, harness_dir),
        "integrity": integrity,
        "links": {
            "adapter_result": "agent/adapter-result.json"
            if adapter_path.is_file()
            else None,
            "harness_run": _relative_link(harness_dir, path),
            "evaluation": "evaluation.json" if evaluation else None,
        },
    }


def _checkpoint_names(root: Path) -> list[str]:
    problem = root / "problem.yaml"
    if problem.is_file():
        raw = yaml.safe_load(problem.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("checkpoints"), dict):
            return sorted(raw["checkpoints"], key=_checkpoint_name_order)
    return [
        path.name
        for path in sorted(root.glob("checkpoint_*"), key=_checkpoint_order)
        if path.is_dir()
    ]


def _external_harness_dir(checkpoint_path: Path) -> Path:
    try:
        index = int(checkpoint_path.name.rsplit("_", maxsplit=1)[-1])
        session_root = checkpoint_path.parents[2]
    except (ValueError, IndexError):
        return checkpoint_path / "agent" / "harness-run"
    run_root = session_root / f"checkpoint-{index}" / "harness-runs"
    candidates = [path for path in run_root.glob("run_*") if path.is_dir()]
    if not candidates:
        return checkpoint_path / "agent" / "harness-run"
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _partial_adapter(
    checkpoint_path: Path,
    domain: dict[str, Any],
    events: list[dict[str, Any]],
    harness_dir: Path,
) -> dict[str, Any]:
    reentry = _mapping(domain.get("reentry"))
    fault_records = _mapping(reentry.get("fault_records"))
    open_faults = [
        fault
        for fault in fault_records.values()
        if isinstance(fault, dict)
        and fault.get("status") in {"OPEN", "ADDRESSED"}
    ]
    telemetry = _telemetry(events, harness_dir)
    prompt_path = checkpoint_path / "prompt.txt"
    run_id = domain.get("run_id")
    if run_id is None and events:
        run_id = events[0].get("run_id")
    terminal_fault = None
    legacy_faults = domain.get("faults")
    if isinstance(legacy_faults, list) and legacy_faults:
        terminal_fault = legacy_faults[-1]
    elif open_faults:
        terminal_fault = open_faults[-1]
    return {
        "task_sha256": _file_sha256(prompt_path)
        if prompt_path.is_file()
        else None,
        "harness_run_id": run_id,
        "harness_status": "INFRASTRUCTURE_FAILURE"
        if harness_dir.is_dir()
        else "NOT_RUN",
        "harness_state": domain.get("current_state"),
        "accepted_checkpoint": reentry.get("accepted_checkpoint_id"),
        "review_verdict": _mapping(domain.get("final_result")).get("verdict"),
        "open_fault_count": len(open_faults),
        "terminal_fault": terminal_fault,
        "model_calls": telemetry["model_calls"],
        "prompt_tokens": telemetry["prompt_tokens"],
        "generated_tokens": telemetry["generated_tokens"],
        "reasoning_tokens": telemetry["reasoning_tokens"],
        "cache_tokens": telemetry["cache_tokens"],
        "model_seconds": telemetry["model_seconds"],
        "wall_seconds": 0.0,
        "workspace_manifest_after": [],
        "adapter_infrastructure_failure": True,
    }


def _telemetry(
    events: list[dict[str, Any]], harness_dir: Path
) -> dict[str, int | float]:
    totals: dict[str, int | float] = {
        "model_calls": 0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "reasoning_tokens": 0,
        "cache_tokens": 0,
        "model_seconds": 0.0,
    }
    for event in events:
        if event.get("event_type") != "MODEL_RESPONSE":
            continue
        telemetry = _mapping(
            _event_payload(event, harness_dir).get("telemetry")
        )
        totals["model_calls"] += 1
        for name in (
            "prompt_tokens",
            "generated_tokens",
            "reasoning_tokens",
            "cache_tokens",
        ):
            totals[name] += _integer(telemetry.get(name))
        totals["model_seconds"] += _number(
            telemetry.get("client_duration_seconds")
        )
    totals["model_seconds"] = round(float(totals["model_seconds"]), 6)
    return totals


def _relative_link(target: Path, checkpoint_path: Path) -> str | None:
    if not target.is_dir():
        return None
    return Path(os.path.relpath(target, checkpoint_path)).as_posix()


def _external_metrics(evaluation: dict[str, Any]) -> dict[str, Any]:
    passed = _mapping(evaluation.get("pass_counts"))
    totals = _mapping(evaluation.get("total_counts"))
    values: dict[str, Any] = {
        "infrastructure_failure": bool(
            evaluation.get("infrastructure_failure", not evaluation)
        )
    }
    all_passed = 0
    all_total = 0
    for category in CATEGORIES:
        prefix = category.lower()
        category_passed = _integer(passed.get(category))
        category_total = _integer(totals.get(category))
        values[f"{prefix}_passed"] = category_passed
        values[f"{prefix}_total"] = category_total
        all_passed += category_passed
        all_total += category_total
    values["checkpoint_pass"] = (
        not values["infrastructure_failure"]
        and all_total > 0
        and all_passed == all_total
    )
    values["passed"] = all_passed
    values["total"] = all_total
    return values


def _internal_metrics(
    adapter: dict[str, Any],
    report: dict[str, Any],
    events: list[dict[str, Any]],
    harness_dir: Path,
) -> dict[str, Any]:
    reentry = _mapping(report.get("reentry"))
    fault_records = _mapping(reentry.get("fault_records"))
    faults = [
        value for value in fault_records.values() if isinstance(value, dict)
    ]
    event_types = Counter(str(event.get("event_type")) for event in events)
    occurrences = sum(
        _integer(fault.get("occurrence_count", fault.get("occurrences", 1)))
        for fault in faults
    )
    return {
        "harness_terminal_status": adapter.get("harness_status"),
        "harness_state": adapter.get("harness_state"),
        "review_verdict": adapter.get("review_verdict"),
        "open_fault_count": _integer(adapter.get("open_fault_count")),
        "unique_fault_count": len(faults),
        "fault_occurrence_count": occurrences,
        "recovery_attempt_count": event_types["RECOVERY_CAPSULE_CREATED"],
        "successful_recovery_count": event_types["FAULT_VERIFIED"],
        "candidate_attempt_count": event_types["ATTEMPT_STARTED"],
        "candidate_rejection_count": event_types["ATTEMPT_REJECTED"],
        "accepted_promotions": event_types["CHECKPOINT_PROMOTED"],
        "model_calls": _integer(adapter.get("model_calls")),
        "prompt_tokens": _integer(adapter.get("prompt_tokens")),
        "generated_tokens": _integer(adapter.get("generated_tokens")),
        "reasoning_tokens": _integer(adapter.get("reasoning_tokens")),
        "cache_tokens": _integer(adapter.get("cache_tokens")),
        "model_seconds": _number(adapter.get("model_seconds")),
        "wall_seconds": _number(adapter.get("wall_seconds")),
        "accepted_checkpoint": adapter.get("accepted_checkpoint"),
        "terminal_fault": adapter.get("terminal_fault"),
        "harness_run_id": adapter.get("harness_run_id"),
        "adapter_infrastructure_failure": bool(
            adapter.get("adapter_infrastructure_failure")
        ),
        "prompt_package_count": len(list(harness_dir.rglob("prompt*.json"))),
    }


def _integrity_metrics(
    adapter: dict[str, Any],
    report: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    event_types = Counter(str(event.get("event_type")) for event in events)
    after = adapter.get("workspace_manifest_after", [])
    hidden_leakage = sum(
        1
        for item in after
        if isinstance(item, dict) and item.get("classification") == "forbidden"
    )
    prompt_events = [
        event for event in events if event.get("event_type") == "MODEL_PROMPT"
    ]
    missing_prompt_fingerprints = sum(
        1 for event in prompt_events if not event.get("prompt_fingerprint")
    )
    not_run = adapter.get("harness_status") == "NOT_RUN"
    report_run_id = report.get("run_id")
    adapter_run_id = adapter.get("harness_run_id")
    values = {
        "accepted_checkpoint_contamination": event_types[
            "ACCEPTED_CHECKPOINT_CONTAMINATION"
        ],
        "invalid_completion": event_types["INVALID_COMPLETION"],
        "open_fault_with_approve": int(
            adapter.get("review_verdict") == "APPROVE"
            and _integer(adapter.get("open_fault_count")) > 0
        ),
        "hidden_test_leakage": hidden_leakage,
        "invalidated_evidence_reuse": event_types[
            "INVALIDATED_EVIDENCE_REUSED"
        ],
        "missing_task_fingerprint": int(
            not not_run and not adapter.get("task_sha256")
        ),
        "missing_prompt_fingerprint": missing_prompt_fingerprints,
        "missing_accepted_checkpoint_identity": int(
            not not_run and not adapter.get("accepted_checkpoint")
        ),
        "report_run_identity_mismatch": int(
            bool(report_run_id)
            and bool(adapter_run_id)
            and report_run_id != adapter_run_id
        ),
    }
    values["hard_failure"] = any(values.values())
    return values


def _stage_metrics(
    events: list[dict[str, Any]], harness_dir: Path
) -> dict[str, dict[str, Any]]:
    stages: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "calls": 0,
            "tokens": 0,
            "duration": 0.0,
            "attempts": 0,
            "faults": 0,
            "successful_exit": False,
            "terminal_fault": None,
        }
    )
    for event in events:
        stage = str(event.get("stage") or "UNSCOPED")
        values = stages[stage]
        event_type = str(event.get("event_type"))
        if event_type == "MODEL_RESPONSE":
            values["calls"] += 1
            payload = _event_payload(event, harness_dir)
            telemetry = _mapping(payload.get("telemetry"))
            values["tokens"] += _integer(telemetry.get("prompt_tokens"))
            values["tokens"] += _integer(telemetry.get("generated_tokens"))
            values["duration"] += _number(
                telemetry.get("client_duration_seconds")
            )
        elif event_type == "ATTEMPT_STARTED":
            values["attempts"] += 1
        elif event_type == "FAULT_RECORDED":
            values["faults"] += 1
            values["terminal_fault"] = event.get("human_summary")
        elif event_type in {
            "STATE_TRANSITION",
            "CHECKPOINT_PROMOTED",
            "ATTEMPT_ACCEPTED_READ_ONLY",
        }:
            values["successful_exit"] = True
    return {
        stage: {**values, "duration": round(values["duration"], 6)}
        for stage, values in sorted(stages.items())
    }


def _fault_metrics(
    report: dict[str, Any],
    events: list[dict[str, Any]],
    harness_dir: Path,
) -> dict[str, Any]:
    reentry = _mapping(report.get("reentry"))
    records = [
        value
        for value in _mapping(reentry.get("fault_records")).values()
        if isinstance(value, dict)
    ]
    group_fields = {
        "fault_type": ("fault_type",),
        "stage": ("origin_stage", "stage"),
        "violated_contract": ("violated_contract",),
        "tool": ("tool",),
        "command": ("command",),
        "file": ("file",),
        "equivalence_signature": (
            "fault_signature",
            "equivalence_signature",
            "signature",
        ),
        "recovery_result": ("status",),
    }
    groups: dict[str, Counter[str]] = {name: Counter() for name in group_fields}
    total_occurrences = 0
    closed = 0
    first_repair = 0
    for record in records:
        occurrence_count = _integer(
            record.get("occurrence_count", record.get("occurrences", 1))
        )
        for group_name, fields in group_fields.items():
            raw = next(
                (record.get(field) for field in fields if record.get(field)),
                None,
            )
            groups[group_name][str(raw or "UNKNOWN")] += occurrence_count
        total_occurrences += occurrence_count
        if record.get("status") in {"VERIFIED", "CLOSED"}:
            closed += 1
            first_repair += int(occurrence_count == 1)
    promotions = sum(
        event.get("event_type") == "CHECKPOINT_PROMOTED" for event in events
    )
    prompt_tokens = sum(
        _integer(
            _mapping(_event_payload(event, harness_dir).get("telemetry")).get(
                "prompt_tokens"
            )
        )
        for event in events
        if event.get("event_type") == "MODEL_RESPONSE"
    )
    return {
        f"by_{name}": dict(sorted(counter.items()))
        for name, counter in groups.items()
    } | {
        "fault_recurrence_rate": (
            round((total_occurrences - len(records)) / total_occurrences, 6)
            if total_occurrences
            else 0.0
        ),
        "first_repair_success_rate": (
            round(first_repair / len(records), 6) if records else 0.0
        ),
        "recovery_yield": (round(closed / len(records), 6) if records else 0.0),
        "faults_per_accepted_checkpoint": (
            round(total_occurrences / promotions, 6) if promotions else None
        ),
        "tokens_per_closed_fault": (
            round(prompt_tokens / closed, 6) if closed else None
        ),
    }


def _session_metrics(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [item["external"]["checkpoint_pass"] for item in checkpoints]
    first_regression = next(
        (
            item["checkpoint"]
            for item in checkpoints
            if item["external"]["regression_passed"]
            < item["external"]["regression_total"]
        ),
        None,
    )
    furthest = 0
    for index, checkpoint_passed in enumerate(passed, start=1):
        if not checkpoint_passed:
            break
        furthest = index
    return {
        "checkpoints_passed": sum(passed),
        "checkpoint_count": len(checkpoints),
        "furthest_checkpoint": furthest,
        "all_checkpoints_passed": all(passed),
        "final_suite_passed": passed[-1],
        "first_regression_checkpoint": first_regression,
        "cumulative_regression_failures": sum(
            item["external"]["regression_total"]
            - item["external"]["regression_passed"]
            for item in checkpoints
        ),
    }


def _event_payload(event: dict[str, Any], harness_dir: Path) -> dict[str, Any]:
    reference = event.get("payload_ref")
    if not isinstance(reference, str) or not reference.startswith("sha256:"):
        return {}
    digest = reference.removeprefix("sha256:")
    return _read_json(
        harness_dir / "artifacts" / "sha256" / digest[:2] / digest[2:],
        required=False,
    )


def _render_summary(combined: dict[str, Any]) -> str:
    session = combined["session"]
    rows = [
        "# Fault-Typed Harness Campaign Summary",
        "",
        f"- Checkpoints passed: {session['checkpoints_passed']}/{session['checkpoint_count']}",
        f"- Furthest consecutive checkpoint: {session['furthest_checkpoint']}",
        f"- Final suite passed: {session['final_suite_passed']}",
        f"- Integrity hard failure: {combined['integrity_hard_failure']}",
        "",
        "| Checkpoint | External | Harness | Review | Calls | Tokens | Faults | Integrity |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for checkpoint in combined["checkpoints"]:
        external = checkpoint["external"]
        internal = checkpoint["internal"]
        rows.append(
            "| {name} | {passed}/{total} | {status} | {review} | {calls} | "
            "{tokens} | {faults} | {integrity} |".format(
                name=checkpoint["checkpoint"],
                passed=external["passed"],
                total=external["total"],
                status=internal["harness_terminal_status"],
                review=internal["review_verdict"],
                calls=internal["model_calls"],
                tokens=internal["prompt_tokens"] + internal["generated_tokens"],
                faults=internal["fault_occurrence_count"],
                integrity="FAIL"
                if checkpoint["integrity"]["hard_failure"]
                else "PASS",
            )
        )
    return "\n".join(rows) + "\n"


def _checkpoint_order(path: Path) -> tuple[int, str]:
    try:
        return int(path.name.rsplit("_", maxsplit=1)[-1]), path.name
    except ValueError:
        return sys.maxsize, path.name


def _checkpoint_name_order(name: str) -> tuple[int, str]:
    return _checkpoint_order(Path(name))


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise ValueError(f"Required result artifact is missing: {path}")
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path, *, required: bool = True) -> list[dict[str, Any]]:
    if not path.is_file():
        if required:
            raise ValueError(f"Required event stream is missing: {path}")
        return []
    return [
        value
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and isinstance((value := json.loads(line)), dict)
    ]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _integer(value: Any) -> int:
    return int(value) if isinstance(value, int | float) else 0


def _number(value: Any) -> float:
    return float(value) if isinstance(value, int | float) else 0.0
