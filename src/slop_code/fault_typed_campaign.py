"""Unattended, immutable SCBench campaigns for the fault-typed harness."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_harness import Harness
from git import Repo

from slop_code.fault_typed_results import collect_fault_typed_results

ROOT = Path(__file__).resolve().parents[2]
BASELINE_CONFIG = (
    ROOT / "configs" / "runs" / "file-backup-fault-typed-baseline-v0.yaml"
)
TREATMENT_FILES = (
    BASELINE_CONFIG,
    ROOT / "configs" / "agents" / "fault-typed-harness.yaml",
    ROOT / "configs" / "harness" / "fault-typed-scbench.toml",
    ROOT / "configs" / "models" / "qwen-local.yaml",
    ROOT / "configs" / "prompts" / "just-solve.jinja",
)


def run_campaign(
    *,
    campaign: str,
    problem: str,
    repetitions: int,
    agent: str,
    model: str,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Run a fixed treatment repeatedly and retain every outcome."""
    if repetitions < 1:
        raise ValueError("Campaign repetitions must be positive")
    if not campaign or any(character in "/\\" for character in campaign):
        raise ValueError(f"Invalid campaign identity: {campaign!r}")
    root = (output_root or ROOT / "benchmark-output" / campaign).resolve()
    treatment = _treatment_manifest(agent=agent, model=model)
    if not treatment["working_trees_clean"]:
        raise RuntimeError(
            "Counted campaigns require clean SCBench and harness working trees"
        )
    root.mkdir(parents=True, exist_ok=False)
    seeds = [7] * repetitions
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": campaign,
        "problem": problem,
        "repetitions": repetitions,
        "agent": agent,
        "model": model,
        "seed_schedule": seeds,
        "started_at": _now(),
        "completed_at": None,
        "treatment": treatment,
        "sessions": [],
    }
    _write_json(root / "campaign.json", manifest)
    for index, seed in enumerate(seeds, start=1):
        current = _treatment_manifest(agent=agent, model=model)
        if current["fingerprint"] != treatment["fingerprint"]:
            raise RuntimeError(
                "Campaign treatment changed after execution began; "
                "the campaign is invalid"
            )
        session_id = f"{campaign}-r{index:02d}-seed{seed}"
        session_root = root / session_id
        scbench_root = session_root / "scbench"
        command = _scbench_command(
            session_root=session_root,
            problem=problem,
            seed=seed,
            agent=agent,
            model=model,
        )
        session_record: dict[str, Any] = {
            "session_id": session_id,
            "repetition": index,
            "seed": seed,
            "started_at": _now(),
            "completed_at": None,
            "command": command,
            "exit_code": None,
            "result": None,
            "error": None,
        }
        manifest["sessions"].append(session_record)
        _write_json(root / "campaign.json", manifest)
        environment = os.environ.copy()
        environment["LOCAL_LLAMA_CPP_READY"] = "1"
        environment["FAULT_TYPED_HARNESS_RUN_ROOT"] = str(root)
        environment["SCBENCH_BENCHMARK_SESSION_ID"] = session_id
        environment.setdefault(
            "SCBENCH_PROBLEM_SET_COMMIT",
            "4d38d300059667d57e43c31969bc455f5c338b52",
        )
        session_root.mkdir(parents=True, exist_ok=False)
        with (
            (session_root / "runner.stdout.log").open(
                "w", encoding="utf-8", newline="\n"
            ) as stdout,
            (session_root / "runner.stderr.log").open(
                "w", encoding="utf-8", newline="\n"
            ) as stderr,
        ):
            completed = subprocess.run(  # noqa: S603
                command,
                cwd=ROOT,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        session_record["exit_code"] = completed.returncode
        problem_dir = scbench_root / problem
        try:
            combined = collect_fault_typed_results(problem_dir)
            session_record["result"] = str(
                (problem_dir / "combined-results.json").relative_to(root)
            ).replace("\\", "/")
            session_record["all_checkpoints_passed"] = combined["session"][
                "all_checkpoints_passed"
            ]
            session_record["integrity_hard_failure"] = combined[
                "integrity_hard_failure"
            ]
        except Exception as exc:  # noqa: BLE001
            session_record["error"] = f"{type(exc).__name__}: {exc}"
        session_record["completed_at"] = _now()
        _write_json(root / "campaign.json", manifest)
    manifest["completed_at"] = _now()
    summary = _campaign_summary(root, manifest)
    manifest["summary"] = summary
    _write_json(root / "campaign.json", manifest)
    _write_json(root / "campaign-results.json", summary)
    (root / "campaign-summary.md").write_text(
        _render_campaign_summary(manifest), encoding="utf-8", newline="\n"
    )
    return manifest


def _scbench_command(
    *,
    session_root: Path,
    problem: str,
    seed: int,
    agent: str,
    model: str,
) -> list[str]:
    agent_config = (
        "fault-typed-harness" if agent == "fault_typed_harness" else agent
    )
    executable = Path(sys.executable).with_name("slop-code.exe")
    if not executable.is_file():
        executable = Path(sys.executable).with_name("slop-code")
    return [
        str(executable),
        "--seed",
        str(seed),
        "run",
        "--config",
        str(BASELINE_CONFIG),
        "--problem",
        problem,
        "--agent",
        agent_config,
        "--model",
        f"local_llama_cpp/{model}",
        "--no-live-progress",
        f"save_dir={session_root}",
        "save_template=scbench",
    ]


def _treatment_manifest(*, agent: str, model: str) -> dict[str, Any]:
    files = {
        str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
        for path in TREATMENT_FILES
    }
    commits = {
        "adapter": Repo(ROOT).head.commit.hexsha,
        "harness": _repository_commit_for_module(Harness),
        "problem_set": os.environ.get(
            "SCBENCH_PROBLEM_SET_COMMIT",
            "4d38d300059667d57e43c31969bc455f5c338b52",
        ),
    }
    raw = json.dumps(
        {"files": files, "commits": commits, "agent": agent, "model": model},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "fingerprint": hashlib.sha256(raw).hexdigest(),
        "files": files,
        "commits": commits,
        "agent": agent,
        "model": model,
        "model_file_hash": _optional_hash(os.environ.get("LLAMA_MODEL_FILE")),
        "llama_cpp_commit": os.environ.get("LLAMA_CPP_COMMIT"),
        "host": platform.platform(),
        "python": sys.version,
        "working_trees_clean": _working_trees_clean(),
    }


def _campaign_summary(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    combined: list[dict[str, Any]] = []
    terminal_clusters: Counter[str] = Counter()
    fault_distribution: Counter[str] = Counter()
    for session in manifest["sessions"]:
        result_path = session.get("result")
        if not result_path:
            terminal_clusters[session.get("error") or "missing result"] += 1
            continue
        result = json.loads((root / result_path).read_text(encoding="utf-8"))
        combined.append(result)
        for checkpoint in result["checkpoints"]:
            internal = checkpoint["internal"]
            for fault_type, count in checkpoint["faults"][
                "by_fault_type"
            ].items():
                fault_distribution[fault_type] += count
            fault = internal.get("terminal_fault")
            if fault:
                terminal_clusters[str(fault.get("fault_type", "UNKNOWN"))] += 1
    checkpoint_count = sum(
        result["session"]["checkpoint_count"] for result in combined
    )
    checkpoints_passed = sum(
        result["session"]["checkpoints_passed"] for result in combined
    )
    all_checkpoints = [
        checkpoint
        for result in combined
        for checkpoint in result["checkpoints"]
    ]
    regression_total = sum(
        checkpoint["external"]["regression_total"]
        for checkpoint in all_checkpoints
    )
    regression_failures = sum(
        checkpoint["external"]["regression_total"]
        - checkpoint["external"]["regression_passed"]
        for checkpoint in all_checkpoints
    )
    clean_completions = sum(
        checkpoint["internal"]["harness_terminal_status"] == "COMPLETE"
        and checkpoint["internal"]["review_verdict"] == "APPROVE"
        and checkpoint["internal"]["open_fault_count"] == 0
        for checkpoint in all_checkpoints
    )
    recovery_attempts = sum(
        checkpoint["internal"]["recovery_attempt_count"]
        for checkpoint in all_checkpoints
    )
    successful_recoveries = sum(
        checkpoint["internal"]["successful_recovery_count"]
        for checkpoint in all_checkpoints
    )
    total_tokens = sum(
        checkpoint["internal"]["prompt_tokens"]
        + checkpoint["internal"]["generated_tokens"]
        for checkpoint in all_checkpoints
    )
    total_model_seconds = sum(
        checkpoint["internal"]["model_seconds"]
        for checkpoint in all_checkpoints
    )
    successful = [
        result
        for result in combined
        if result["session"]["all_checkpoints_passed"]
        and not result["integrity_hard_failure"]
    ]
    successful_checkpoints = [
        checkpoint
        for result in successful
        for checkpoint in result["checkpoints"]
    ]
    survival_curve = {
        str(index): sum(
            result["session"]["furthest_checkpoint"] >= index
            for result in combined
        )
        for index in range(
            1,
            max((len(result["checkpoints"]) for result in combined), default=0)
            + 1,
        )
    }
    return {
        "sessions_requested": manifest["repetitions"],
        "sessions_with_results": len(combined),
        "successful_full_sessions": sum(
            result["session"]["all_checkpoints_passed"]
            and not result["integrity_hard_failure"]
            for result in combined
        ),
        "checkpoint_pass_rate": (
            round(checkpoints_passed / checkpoint_count, 6)
            if checkpoint_count
            else 0.0
        ),
        "checkpoint_survival_curve": survival_curve,
        "regression_rate": (
            round(regression_failures / regression_total, 6)
            if regression_total
            else 0.0
        ),
        "clean_harness_completion_rate": (
            round(clean_completions / len(all_checkpoints), 6)
            if all_checkpoints
            else 0.0
        ),
        "fault_distribution": dict(fault_distribution.most_common()),
        "recovery_yield": (
            round(successful_recoveries / recovery_attempts, 6)
            if recovery_attempts
            else 0.0
        ),
        "tokens_per_checkpoint": (
            round(total_tokens / len(all_checkpoints), 6)
            if all_checkpoints
            else None
        ),
        "model_seconds_per_checkpoint": (
            round(total_model_seconds / len(all_checkpoints), 6)
            if all_checkpoints
            else None
        ),
        "tokens_per_successful_session": (
            round(
                sum(
                    checkpoint["internal"]["prompt_tokens"]
                    + checkpoint["internal"]["generated_tokens"]
                    for checkpoint in successful_checkpoints
                )
                / len(successful),
                6,
            )
            if successful
            else None
        ),
        "model_seconds_per_successful_session": (
            round(
                sum(
                    checkpoint["internal"]["model_seconds"]
                    for checkpoint in successful_checkpoints
                )
                / len(successful),
                6,
            )
            if successful
            else None
        ),
        "integrity_failure_sessions": sum(
            result["integrity_hard_failure"] for result in combined
        ),
        "terminal_failure_clusters": dict(terminal_clusters.most_common()),
    }


def _render_campaign_summary(manifest: dict[str, Any]) -> str:
    summary = manifest["summary"]
    lines = [
        f"# Campaign {manifest['campaign_id']}",
        "",
        f"- Full clean sessions: {summary['successful_full_sessions']}/{summary['sessions_requested']}",
        f"- Checkpoint pass rate: {summary['checkpoint_pass_rate']:.1%}",
        f"- Integrity-failure sessions: {summary['integrity_failure_sessions']}",
        f"- Treatment fingerprint: `{manifest['treatment']['fingerprint']}`",
        "",
        "| Session | Exit | External complete | Integrity | Error |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for session in manifest["sessions"]:
        lines.append(
            "| {session_id} | {exit_code} | {passed} | {integrity} | {error} |".format(
                session_id=session["session_id"],
                exit_code=session["exit_code"],
                passed=session.get("all_checkpoints_passed"),
                integrity=session.get("integrity_hard_failure"),
                error=session.get("error") or "",
            )
        )
    return "\n".join(lines) + "\n"


def _repository_commit_for_module(value: type[Any]) -> str | None:
    module = sys.modules[value.__module__]
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return None
    try:
        return Repo(
            module_file, search_parent_directories=True
        ).head.commit.hexsha
    except Exception:  # noqa: BLE001
        return None


def _working_trees_clean() -> bool:
    repositories = [Repo(ROOT)]
    module = sys.modules[Harness.__module__]
    module_file = getattr(module, "__file__", None)
    if module_file is not None:
        repositories.append(Repo(module_file, search_parent_directories=True))
    return all(
        not repository.is_dirty(untracked_files=True)
        for repository in repositories
    )


def _optional_hash(raw: str | None) -> str | None:
    if raw is None:
        return None
    path = Path(raw)
    return _sha256(path) if path.is_file() else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _now() -> str:
    return datetime.now(UTC).isoformat()
