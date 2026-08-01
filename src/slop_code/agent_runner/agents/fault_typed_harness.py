"""Native SCBench adapter for the fault-typed local agent harness."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import sys
import uuid
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from agent_harness import Harness
from agent_harness import HarnessConfig
from agent_harness import RunResult
from git import Repo
from git.exc import InvalidGitRepositoryError
from git.exc import NoSuchPathError

from slop_code.agent_runner.agent import Agent
from slop_code.agent_runner.agent import AgentConfigBase
from slop_code.agent_runner.credentials import ProviderCredential
from slop_code.agent_runner.models import AgentCostLimits
from slop_code.agent_runner.models import AgentError
from slop_code.agent_runner.registry import register_agent
from slop_code.common import APIPricing
from slop_code.common import ModelDefinition
from slop_code.common import ThinkingPreset
from slop_code.common import TokenUsage
from slop_code.execution import LocalEnvironmentSpec
from slop_code.execution import Session

HARNESS_BASELINE = "3213770b4f64c9597b47b4230d15a6aac56c7112"
SCBENCH_BASELINE = "8e3a8b693f3c5e48143aeb7cb5b1beda1f19c44b"
PROBLEM_SET_BASELINE = "4d38d300059667d57e43c31969bc455f5c338b52"
FORBIDDEN_ROOTS = {".agent-harness", ".evaluation_tests"}
FORBIDDEN_NAMES = {
    "adapter-result.json",
    "benchmark-link.json",
    "events.jsonl",
    "projection.json",
    "run-report.json",
    "session.json",
    "state.json",
}


class FaultTypedHarnessConfig(
    AgentConfigBase,
    agent_type="fault_typed_harness",
):
    """Configuration for the native fault-typed harness adapter."""

    type: Literal["fault_typed_harness"] = "fault_typed_harness"
    harness_config: Path
    run_root: Path
    copy_full_run: bool = True
    require_model_match: bool = True


class FaultTypedHarnessAgent(Agent):
    """Run one fresh harness invocation per persistent SCBench checkpoint."""

    def __init__(
        self,
        *,
        problem_name: str,
        verbose: bool,  # noqa: FBT001
        cost_limits: AgentCostLimits,
        harness_config_path: Path,
        harness_config: HarnessConfig,
        run_root: Path,
        selected_model: str,
        copy_full_run: bool,
        harness: Harness | None = None,
    ) -> None:
        super().__init__(
            agent_name="fault_typed_harness",
            problem_name=problem_name,
            cost_limits=cost_limits,
            pricing=APIPricing(),
            verbose=verbose,
        )
        self.harness_config_path = harness_config_path.resolve()
        self.harness_config = harness_config
        configured_run_root = os.environ.get("FAULT_TYPED_HARNESS_RUN_ROOT")
        self.run_root = Path(configured_run_root or run_root).resolve()
        self.selected_model = selected_model
        self.copy_full_run = copy_full_run
        self._harness = harness or Harness(harness_config)
        self._session: Session | None = None
        requested_session_id = os.environ.get("SCBENCH_BENCHMARK_SESSION_ID")
        self.benchmark_session_id = _session_id(requested_session_id)
        self.checkpoint_sequence = 0
        self.last_result: RunResult | None = None
        self.last_run_directory: Path | None = None
        self.last_adapter_result: dict[str, Any] | None = None
        self.current_checkpoint_usage: dict[str, int | float] = {}
        self.export_manifest: list[dict[str, Any]] = []
        self._checkpoint_records: list[dict[str, Any]] = []
        self._session_dir: Path | None = None
        self._session_manifest: dict[str, Any] | None = None

    @classmethod
    def _from_config(
        cls,
        config: AgentConfigBase,
        model: ModelDefinition,
        credential: ProviderCredential,
        problem_name: str,
        verbose: bool,  # noqa: FBT001
        image: str | None,
        thinking_preset: ThinkingPreset | None = None,
        thinking_max_tokens: int | None = None,
    ) -> FaultTypedHarnessAgent:  # noqa: FBT001
        del credential, image, thinking_preset, thinking_max_tokens
        if not isinstance(config, FaultTypedHarnessConfig):
            raise TypeError(
                "FaultTypedHarnessAgent requires FaultTypedHarnessConfig"
            )
        config_path = config.harness_config.expanduser().resolve()
        if not config_path.is_file():
            raise AgentError(
                f"Harness configuration does not exist: {config_path}"
            )
        harness_config = HarnessConfig.from_toml(config_path)
        selected_model = model.internal_name
        configured_model = harness_config.model.model
        if (
            config.require_model_match
            and configured_model
            and configured_model != selected_model
        ):
            raise AgentError(
                "SCBench model identity does not match harness configuration: "
                f"{selected_model!r} != {configured_model!r}"
            )
        harness_config = harness_config.with_model(selected_model)
        return cls(
            problem_name=problem_name,
            verbose=verbose,
            cost_limits=config.cost_limits,
            harness_config_path=config_path,
            harness_config=harness_config,
            run_root=config.run_root.expanduser().resolve(),
            selected_model=selected_model,
            copy_full_run=config.copy_full_run,
        )

    @property
    def session(self) -> Session:
        if self._session is None:
            raise AgentError("FaultTypedHarnessAgent has not been set up")
        return self._session

    def setup(self, session: Session) -> None:
        if not isinstance(session.spec, LocalEnvironmentSpec):
            raise AgentError(
                "FaultTypedHarnessAgent supports only local SCBench sessions"
            )
        workspace = session.working_dir.resolve()
        if not workspace.is_dir():
            raise AgentError(f"SCBench workspace does not exist: {workspace}")
        if _is_relative_to(self.run_root, workspace):
            raise AgentError(
                f"Harness run root must be outside SCBench workspace: {self.run_root}"
            )
        self.run_root.mkdir(parents=True, exist_ok=True)
        models = self._harness.list_models()
        if self.selected_model not in models:
            raise AgentError(
                f"Selected local model {self.selected_model!r} is unavailable; "
                f"server reported {models!r}"
            )
        self._session = session
        self._session_dir = self.run_root / self.benchmark_session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)
        if (self._session_dir / "session.json").exists():
            raise AgentError(
                "Benchmark session evidence already exists: "
                f"{self._session_dir}"
            )
        self._session_manifest = {
            "benchmark_session_id": self.benchmark_session_id,
            "problem": self.problem_name,
            "started_at": _now(),
            "completed_at": None,
            "harness_commit": _git_commit_for_module(Harness),
            "expected_harness_commit": HARNESS_BASELINE,
            "scbench_commit": _git_commit_for_module(FaultTypedHarnessAgent),
            "expected_scbench_commit": SCBENCH_BASELINE,
            "problem_set_commit": os.environ.get(
                "SCBENCH_PROBLEM_SET_COMMIT", PROBLEM_SET_BASELINE
            ),
            "model_id": self.selected_model,
            "model_file_hash": _optional_file_hash(
                os.environ.get("LLAMA_MODEL_FILE")
            ),
            "llama_cpp_commit": os.environ.get("LLAMA_CPP_COMMIT"),
            "harness_config": str(self.harness_config_path),
            "harness_config_sha256": _file_sha256(self.harness_config_path),
            "host": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python": sys.version,
                "python_executable": sys.executable,
            },
            "checkpoints": self._checkpoint_records,
        }
        self._flush_session_manifest()

    def run(self, task: str) -> None:
        self.checkpoint_sequence += 1
        task_bytes = task.encode("utf-8")
        task_sha256 = hashlib.sha256(task_bytes).hexdigest()
        checkpoint_dir = self._checkpoint_dir()
        run_base = checkpoint_dir / "harness-runs"
        run_base.mkdir(parents=True, exist_ok=False)
        before = _source_tree_manifest(self.session.working_dir)
        _assert_clean_submission_manifest(before)
        before_identity = _manifest_identity(before)
        started_at = _now()
        started = datetime.now(UTC)
        try:
            result = self._harness.run(
                workspace=self.session.working_dir,
                task=task,
                run_base=run_base,
                progress=self._progress,
            )
        except AgentError:
            raise
        except Exception as exc:
            self._record_partial_failure(
                run_base=run_base,
                task_sha256=task_sha256,
                task_bytes=len(task_bytes),
                before=before,
                before_identity=before_identity,
                started_at=started_at,
                started=started,
                error=exc,
            )
            raise AgentError(f"Harness integration failure: {exc}") from exc
        completed = datetime.now(UTC)
        after = _source_tree_manifest(self.session.working_dir)
        _assert_clean_submission_manifest(after)
        report = _read_json(result.report_path)
        if report.get("run_id") != result.run_id:
            raise AgentError(
                "Harness report run identity does not match RunResult"
            )
        telemetry = _summarize_telemetry(result.run_dir)
        self.usage.steps = int(telemetry["model_calls"])
        tokens = TokenUsage(
            input=int(telemetry["prompt_tokens"]),
            output=int(telemetry["generated_tokens"]),
            cache_read=int(telemetry["cache_tokens"]),
            reasoning=int(telemetry["reasoning_tokens"]),
        )
        self.usage.net_tokens = tokens
        self.usage.current_tokens = tokens
        reentry = _mapping(report.get("reentry"))
        fault_records = _mapping(reentry.get("fault_records"))
        open_faults = [
            fault
            for fault in fault_records.values()
            if isinstance(fault, dict)
            and fault.get("status") in {"OPEN", "ADDRESSED"}
        ]
        final_result = _mapping(report.get("final_result"))
        self.current_checkpoint_usage = telemetry
        self.last_result = result
        self.last_run_directory = result.run_dir
        self.last_adapter_result = {
            "benchmark_session_id": self.benchmark_session_id,
            "problem": self.problem_name,
            "checkpoint_index": self.checkpoint_sequence,
            "task_sha256": task_sha256,
            "task_bytes": len(task_bytes),
            "workspace_before_sha256": before_identity,
            "workspace_after_sha256": _manifest_identity(after),
            "workspace_manifest_before": before,
            "workspace_manifest_after": after,
            "harness_run_id": result.run_id,
            "harness_commit": _git_commit_for_module(Harness),
            "harness_status": result.status,
            "harness_state": result.state,
            "accepted_checkpoint": reentry.get("accepted_checkpoint_id"),
            "review_verdict": final_result.get("verdict"),
            "open_fault_count": len(open_faults),
            "terminal_fault": _terminal_fault(report),
            "model_calls": telemetry["model_calls"],
            "prompt_tokens": telemetry["prompt_tokens"],
            "generated_tokens": telemetry["generated_tokens"],
            "reasoning_tokens": telemetry["reasoning_tokens"],
            "cache_tokens": telemetry["cache_tokens"],
            "model_seconds": telemetry["model_seconds"],
            "wall_seconds": round((completed - started).total_seconds(), 6),
            "started_at": started_at,
            "completed_at": _now(),
            "run_dir": "harness-run",
        }
        self._checkpoint_records.append(dict(self.last_adapter_result))
        self._flush_session_manifest()

    def reset(self) -> None:
        self.last_result = None
        self.last_run_directory = None
        self.last_adapter_result = None
        self.current_checkpoint_usage = {}

    def save_artifacts(self, path: Path) -> None:
        if self.last_adapter_result is None or self.last_run_directory is None:
            raise AgentError(
                "No completed harness checkpoint is available to export"
            )
        try:
            path.mkdir(parents=True, exist_ok=True)
            _write_json(path / "adapter-result.json", self.last_adapter_result)
            exported = path / "harness-run"
            if self.copy_full_run:
                shutil.copytree(
                    self.last_run_directory,
                    exported,
                    ignore=_ignore_transient,
                )
            else:
                _copy_minimal_run(self.last_run_directory, exported)
            link = {
                "benchmark_session_id": self.benchmark_session_id,
                "problem": self.problem_name,
                "checkpoint_index": self.checkpoint_sequence,
                "harness_run_id": self.last_result.run_id
                if self.last_result
                else None,
                "task_sha256": self.last_adapter_result["task_sha256"],
                "harness_run": "harness-run",
            }
            _write_json(path / "benchmark-link.json", link)
            exported_files = _export_file_manifest(path)
            export_record = {
                **link,
                "exported_at": _now(),
                "files": exported_files,
            }
            self.export_manifest.append(export_record)
            _write_json(path / "export-manifest.json", export_record)
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError(
                f"Unable to export harness artifacts: {exc}"
            ) from exc

    def cleanup(self) -> None:
        if self._session_manifest is not None:
            self._session_manifest["completed_at"] = _now()
            self._session_manifest["exports"] = self.export_manifest
            self._flush_session_manifest()
        self._session = None

    def _checkpoint_dir(self) -> Path:
        if self._session_dir is None:
            raise AgentError(
                "Adapter setup has not created a benchmark session"
            )
        checkpoint_dir = (
            self._session_dir / f"checkpoint-{self.checkpoint_sequence}"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        return checkpoint_dir

    def _record_partial_failure(
        self,
        *,
        run_base: Path,
        task_sha256: str,
        task_bytes: int,
        before: list[dict[str, Any]],
        before_identity: str,
        started_at: str,
        started: datetime,
        error: Exception,
    ) -> None:
        candidates = sorted(
            (path for path in run_base.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if not candidates:
            return
        run_dir = candidates[-1]
        state = _read_json_optional(run_dir / "state.json")
        reentry = _mapping(state.get("reentry"))
        fault_records = _mapping(reentry.get("fault_records"))
        open_faults = [
            fault
            for fault in fault_records.values()
            if isinstance(fault, dict)
            and fault.get("status") in {"OPEN", "ADDRESSED"}
        ]
        telemetry = _summarize_telemetry_optional(run_dir)
        after = _source_tree_manifest(self.session.working_dir)
        _assert_clean_submission_manifest(after)
        tokens = TokenUsage(
            input=int(telemetry["prompt_tokens"]),
            output=int(telemetry["generated_tokens"]),
            cache_read=int(telemetry["cache_tokens"]),
            reasoning=int(telemetry["reasoning_tokens"]),
        )
        self.usage.steps = int(telemetry["model_calls"])
        self.usage.net_tokens = tokens
        self.usage.current_tokens = tokens
        self.current_checkpoint_usage = telemetry
        self.last_run_directory = run_dir
        self.last_adapter_result = {
            "benchmark_session_id": self.benchmark_session_id,
            "problem": self.problem_name,
            "checkpoint_index": self.checkpoint_sequence,
            "task_sha256": task_sha256,
            "task_bytes": task_bytes,
            "workspace_before_sha256": before_identity,
            "workspace_after_sha256": _manifest_identity(after),
            "workspace_manifest_before": before,
            "workspace_manifest_after": after,
            "harness_run_id": state.get("run_id") or run_dir.name,
            "harness_commit": _git_commit_for_module(Harness),
            "harness_status": "INFRASTRUCTURE_FAILURE",
            "harness_state": state.get("current_state"),
            "accepted_checkpoint": reentry.get("accepted_checkpoint_id"),
            "review_verdict": None,
            "open_fault_count": len(open_faults),
            "terminal_fault": _terminal_fault(state),
            "adapter_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "model_calls": telemetry["model_calls"],
            "prompt_tokens": telemetry["prompt_tokens"],
            "generated_tokens": telemetry["generated_tokens"],
            "reasoning_tokens": telemetry["reasoning_tokens"],
            "cache_tokens": telemetry["cache_tokens"],
            "model_seconds": telemetry["model_seconds"],
            "wall_seconds": round(
                (datetime.now(UTC) - started).total_seconds(), 6
            ),
            "started_at": started_at,
            "completed_at": _now(),
            "run_dir": "harness-run",
        }
        self._checkpoint_records.append(dict(self.last_adapter_result))
        self._flush_session_manifest()

    def _flush_session_manifest(self) -> None:
        if self._session_dir is None or self._session_manifest is None:
            return
        _write_json(self._session_dir / "session.json", self._session_manifest)

    def _progress(self, message: str) -> None:
        self.log.info("agent.fault_typed_harness.progress", message=message)


def _summarize_telemetry(run_dir: Path) -> dict[str, int | float]:
    events_path = run_dir / "events.jsonl"
    if not events_path.is_file():
        raise AgentError(f"Harness event stream is missing: {events_path}")
    totals: dict[str, int | float] = {
        "model_calls": 0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "reasoning_tokens": 0,
        "cache_tokens": 0,
        "model_seconds": 0.0,
    }
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        event = json.loads(line)
        if event.get("event_type") != "MODEL_RESPONSE":
            continue
        payload_ref = event.get("payload_ref")
        if not isinstance(payload_ref, str):
            raise AgentError(
                "MODEL_RESPONSE event is missing its payload artifact"
            )
        payload = _read_json(_artifact_path(run_dir, payload_ref))
        telemetry = _mapping(payload.get("telemetry"))
        totals["model_calls"] += 1
        totals["prompt_tokens"] += _integer(telemetry.get("prompt_tokens"))
        totals["generated_tokens"] += _integer(
            telemetry.get("generated_tokens")
        )
        totals["reasoning_tokens"] += _integer(
            telemetry.get("reasoning_tokens")
        )
        totals["cache_tokens"] += _integer(telemetry.get("cache_tokens"))
        totals["model_seconds"] += _number(
            telemetry.get("client_duration_seconds")
        )
    totals["model_seconds"] = round(float(totals["model_seconds"]), 6)
    return totals


def _summarize_telemetry_optional(
    run_dir: Path,
) -> dict[str, int | float]:
    try:
        return _summarize_telemetry(run_dir)
    except (AgentError, OSError, ValueError, json.JSONDecodeError):
        return {
            "model_calls": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "reasoning_tokens": 0,
            "cache_tokens": 0,
            "model_seconds": 0.0,
        }


def _source_tree_manifest(root: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        first = relative.split("/", maxsplit=1)[0]
        classification = "source"
        if first in FORBIDDEN_ROOTS or path.name in FORBIDDEN_NAMES:
            classification = "forbidden"
        elif first in {".git", ".venv", "__pycache__"}:
            classification = "tooling"
        manifest.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _file_sha256(path),
                "classification": classification,
            }
        )
    return manifest


def _assert_clean_submission_manifest(manifest: list[dict[str, Any]]) -> None:
    forbidden = [
        item["path"]
        for item in manifest
        if item["classification"] == "forbidden"
    ]
    if forbidden:
        raise AgentError(
            "Forbidden benchmark or harness artifacts entered the submission "
            f"workspace: {forbidden}"
        )


def _manifest_identity(manifest: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _terminal_fault(report: dict[str, Any]) -> dict[str, Any] | None:
    faults = report.get("faults")
    if not isinstance(faults, list) or not faults:
        return None
    fault = faults[-1]
    return fault if isinstance(fault, dict) else None


def _artifact_path(run_dir: Path, artifact_id: str) -> Path:
    if not artifact_id.startswith("sha256:"):
        raise AgentError(
            f"Unsupported harness artifact identity: {artifact_id}"
        )
    digest = artifact_id.removeprefix("sha256:")
    return run_dir / "artifacts" / "sha256" / digest[:2] / digest[2:]


def _copy_minimal_run(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=False)
    for name in (
        "events.jsonl",
        "interactions.jsonl",
        "projection.json",
        "run-report.json",
        "state.json",
    ):
        source_file = source / name
        if source_file.is_file():
            shutil.copy2(source_file, target / name)
    for name in ("artifacts", "attempts"):
        source_dir = source / name
        if source_dir.is_dir():
            shutil.copytree(source_dir, target / name, ignore=_ignore_transient)


def _ignore_transient(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name.endswith((".lock", ".sock", ".tmp"))
        or name in {"observer.pid", "audit-server.pid"}
    }


def _export_file_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "export-manifest.json"
    ]


def _git_commit_for_module(value: type[Any]) -> str | None:
    module = sys.modules[value.__module__]
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return None
    try:
        return Repo(
            Path(module_file), search_parent_directories=True
        ).head.commit.hexsha
    except (InvalidGitRepositoryError, NoSuchPathError, TypeError, ValueError):
        return None


def _optional_file_hash(raw: str | None) -> str | None:
    if raw is None:
        return None
    path = Path(raw).expanduser()
    return _file_sha256(path) if path.is_file() else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentError(
            f"Unable to read harness JSON artifact {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise AgentError(f"Harness JSON artifact is not an object: {path}")
    return value


def _read_json_optional(path: Path) -> dict[str, Any]:
    try:
        return _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _integer(value: Any) -> int:
    return int(value) if isinstance(value, int | float) else 0


def _number(value: Any) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _session_id(requested: str | None) -> str:
    if requested is None:
        return f"scbench_{uuid.uuid4().hex}"
    if not requested or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in requested
    ):
        raise AgentError(f"Invalid benchmark session identity: {requested!r}")
    return requested


register_agent("fault_typed_harness", FaultTypedHarnessAgent)
