from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from agent_harness import HarnessConfig
from agent_harness import RunResult

from slop_code.agent_runner.agent import Agent
from slop_code.agent_runner.agents import AgentConfigType
from slop_code.agent_runner.agents.fault_typed_harness import HARNESS_BASELINE
from slop_code.agent_runner.agents.fault_typed_harness import (
    FaultTypedHarnessAgent,
)
from slop_code.agent_runner.agents.fault_typed_harness import (
    FaultTypedHarnessConfig,
)
from slop_code.agent_runner.credentials import CredentialType
from slop_code.agent_runner.credentials import ProviderCredential
from slop_code.agent_runner.models import AgentCostLimits
from slop_code.agent_runner.models import AgentError
from slop_code.agent_runner.registry import get_agent_cls
from slop_code.common import APIPricing
from slop_code.common import ModelDefinition
from slop_code.execution import LocalEnvironmentSpec
from slop_code.execution import Session

ROOT = Path(__file__).resolve().parents[3]
HARNESS_TOML = ROOT / "configs" / "harness" / "fault-typed-scbench.toml"
MODEL_ID = "qwen3.6-27b-100k"


class FakeHarness:
    def __init__(
        self,
        *,
        status: str = "COMPLETE",
        state: str = "COMPLETE",
        fail: Exception | None = None,
    ) -> None:
        self.status = status
        self.state = state
        self.fail = fail
        self.tasks: list[str] = []
        self.workspaces: list[Path] = []
        self.runs: list[Path] = []

    def list_models(self) -> list[str]:
        return [MODEL_ID]

    def run(
        self,
        *,
        workspace: Path,
        task: str,
        run_base: Path,
        progress: Any,
    ) -> RunResult:
        del progress
        if self.fail is not None:
            raise self.fail
        self.tasks.append(task)
        self.workspaces.append(workspace)
        (workspace / "implementation.py").write_text(
            f"CHECKPOINT = {len(self.tasks)}\n", encoding="utf-8"
        )
        run_id = f"run_fake_{len(self.tasks)}"
        run_dir = run_base / run_id
        run_dir.mkdir(parents=True)
        self.runs.append(run_dir)
        payload = {
            "kind": "model_response",
            "telemetry": {
                "prompt_tokens": 12,
                "generated_tokens": 3,
                "client_duration_seconds": 0.5,
            },
        }
        encoded = json.dumps(payload, sort_keys=True).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        artifact = run_dir / "artifacts" / "sha256" / digest[:2] / digest[2:]
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(encoded)
        event = {
            "event_type": "MODEL_RESPONSE",
            "payload_ref": f"sha256:{digest}",
            "prompt_fingerprint": "prompt-fake",
        }
        (run_dir / "events.jsonl").write_text(
            json.dumps(event) + "\n", encoding="utf-8"
        )
        (run_dir / "interactions.jsonl").write_text("{}\n", encoding="utf-8")
        (run_dir / "projection.json").write_text("{}\n", encoding="utf-8")
        (run_dir / "state.json").write_text("{}\n", encoding="utf-8")
        (run_dir / "attempts").mkdir()
        open_fault = self.status == "BLOCKED"
        fault = {
            "fault_id": "fault_fake",
            "fault_type": "RETRY_EXHAUSTED",
            "status": "OPEN",
            "summary": "scripted block",
        }
        report = {
            "run_id": run_id,
            "status": self.status,
            "current_state": self.state,
            "task": task,
            "faults": [fault] if open_fault else [],
            "reentry": {
                "accepted_checkpoint_id": "checkpoint_fake",
                "fault_records": {"fault_fake": fault} if open_fault else {},
            },
            "final_result": (
                {"verdict": "APPROVE"} if self.status == "COMPLETE" else None
            ),
        }
        report_path = run_dir / "run-report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return RunResult(
            run_id=run_id,
            status=self.status,
            state=self.state,
            run_dir=run_dir,
            report_path=report_path,
        )


def _cost_limits() -> AgentCostLimits:
    return AgentCostLimits(
        cost_limit=0,
        net_cost_limit=0,
        step_limit=0,
        max_retries=0,
    )


def _session(tmp_path: Path) -> Session:
    starter = tmp_path / "starter"
    starter.mkdir()
    (starter / "README.md").write_text("starter\n", encoding="utf-8")
    session = Session.from_environment_spec(
        LocalEnvironmentSpec(type="local", name="test"),
        base_dir=starter,
        is_agent_infer=True,
    )
    session.prepare()
    return session


def _agent(
    tmp_path: Path,
    fake: FakeHarness,
    *,
    run_root: Path | None = None,
) -> FaultTypedHarnessAgent:
    config = HarnessConfig.from_toml(HARNESS_TOML).with_model(MODEL_ID)
    return FaultTypedHarnessAgent(
        problem_name="file_backup",
        verbose=False,
        cost_limits=_cost_limits(),
        harness_config_path=HARNESS_TOML,
        harness_config=config,
        run_root=run_root or tmp_path / "external-runs",
        selected_model=MODEL_ID,
        copy_full_run=True,
        harness=fake,  # type: ignore[arg-type]
    )


def test_config_constructs_from_yaml_and_is_registered() -> None:
    raw = yaml.safe_load(
        (ROOT / "configs" / "agents" / "fault-typed-harness.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = FaultTypedHarnessConfig.model_validate(raw)
    assert config.type == "fault_typed_harness"
    assert config.cost_limits.max_retries == 0
    assert get_agent_cls(config.type) is FaultTypedHarnessAgent
    assert "fault_typed_harness" in repr(AgentConfigType)


def test_factory_uses_scbench_model_without_hosted_provider(
    tmp_path: Path,
) -> None:
    config = FaultTypedHarnessConfig(
        harness_config=HARNESS_TOML,
        run_root=tmp_path / "runs",
        cost_limits=_cost_limits(),
    )
    model = ModelDefinition(
        internal_name=MODEL_ID,
        provider="local_llama_cpp",
        pricing=APIPricing(),
    )
    credential = ProviderCredential(
        provider="local_llama_cpp",
        credential_type=CredentialType.ENV_VAR,
        value="1",
        source="LOCAL_LLAMA_CPP_READY",
        destination_key="LOCAL_LLAMA_CPP_READY",
    )
    agent = Agent.from_config(
        config=config,
        model=model,
        credential=credential,
        problem_name="file_backup",
        verbose=False,
        image=None,
    )
    assert isinstance(agent, FaultTypedHarnessAgent)
    assert agent.harness_config.model.model == MODEL_ID
    assert agent.pricing == APIPricing()


def test_factory_rejects_model_mismatch(tmp_path: Path) -> None:
    config = FaultTypedHarnessConfig(
        harness_config=HARNESS_TOML,
        run_root=tmp_path / "runs",
        cost_limits=_cost_limits(),
    )
    model = ModelDefinition(
        internal_name="different-model",
        provider="local_llama_cpp",
        pricing=APIPricing(),
    )
    credential = ProviderCredential(
        provider="local_llama_cpp",
        credential_type=CredentialType.ENV_VAR,
        value="1",
        source="LOCAL_LLAMA_CPP_READY",
        destination_key="LOCAL_LLAMA_CPP_READY",
    )
    with pytest.raises(AgentError, match="does not match"):
        Agent.from_config(
            config=config,
            model=model,
            credential=credential,
            problem_name="file_backup",
            verbose=False,
            image=None,
        )


def test_run_forwards_exact_task_and_maps_usage(tmp_path: Path) -> None:
    session = _session(tmp_path)
    fake = FakeHarness()
    agent = _agent(tmp_path, fake)
    try:
        agent.setup(session)
        task = "checkpoint bytes\r\nremain exact 🧪"
        agent.run(task)
        assert fake.tasks == [task]
        assert fake.workspaces == [session.working_dir]
        assert agent.last_result is not None
        assert agent.last_result.status == "COMPLETE"
        assert agent.usage.steps == 1
        assert agent.usage.net_tokens.input == 12
        assert agent.usage.net_tokens.output == 3
        assert agent.last_adapter_result is not None
        assert (
            agent.last_adapter_result["task_sha256"]
            == hashlib.sha256(task.encode()).hexdigest()
        )
        assert agent.last_adapter_result["accepted_checkpoint"] == (
            "checkpoint_fake"
        )
        assert agent.last_adapter_result["open_fault_count"] == 0
    finally:
        agent.cleanup()
        session.cleanup()


def test_blocked_is_normal_outcome_and_exports_self_contained_run(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    fake = FakeHarness(status="BLOCKED", state="BLOCKED")
    agent = _agent(tmp_path, fake)
    artifacts = tmp_path / "checkpoint-output" / "agent"
    try:
        agent.setup(session)
        agent.run("blocked task")
        assert agent.last_result is not None
        assert agent.last_result.status == "BLOCKED"
        assert agent.last_adapter_result is not None
        assert agent.last_adapter_result["open_fault_count"] == 1
        agent.save_artifacts(artifacts)
        result = json.loads(
            (artifacts / "adapter-result.json").read_text(encoding="utf-8")
        )
        assert result["harness_status"] == "BLOCKED"
        assert result["run_dir"] == "harness-run"
        assert (artifacts / "harness-run" / "events.jsonl").is_file()
        assert (artifacts / "harness-run" / "artifacts").is_dir()
        assert (artifacts / "benchmark-link.json").is_file()
        assert (artifacts / "export-manifest.json").is_file()
    finally:
        agent.cleanup()
        session.cleanup()


def test_reset_preserves_workspace_and_prior_runs(tmp_path: Path) -> None:
    session = _session(tmp_path)
    fake = FakeHarness()
    agent = _agent(tmp_path, fake)
    try:
        agent.setup(session)
        agent.run("first")
        prior_run = agent.last_run_directory
        assert prior_run is not None and prior_run.is_dir()
        agent.reset()
        assert (session.working_dir / "implementation.py").is_file()
        assert prior_run.is_dir()
        agent.run("second")
        assert fake.tasks == ["first", "second"]
    finally:
        agent.cleanup()
        session.cleanup()


def test_nested_run_root_is_rejected_before_creation(tmp_path: Path) -> None:
    session = _session(tmp_path)
    nested = session.working_dir / "benchmark-output"
    agent = _agent(tmp_path, FakeHarness(), run_root=nested)
    try:
        with pytest.raises(AgentError, match="outside SCBench workspace"):
            agent.setup(session)
        assert not nested.exists()
    finally:
        session.cleanup()


def test_campaign_identity_and_external_root_are_runtime_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external = tmp_path / "campaign"
    monkeypatch.setenv("FAULT_TYPED_HARNESS_RUN_ROOT", str(external))
    monkeypatch.setenv("SCBENCH_BENCHMARK_SESSION_ID", "session-0001")
    agent = _agent(tmp_path, FakeHarness())
    session = _session(tmp_path)

    try:
        agent.setup(session)
        assert agent.run_root == external.resolve()
        assert agent.benchmark_session_id == "session-0001"
        assert (external / "session-0001" / "session.json").is_file()
    finally:
        agent.cleanup()
        session.cleanup()


def test_forbidden_workspace_file_is_detected_before_model_execution(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    fake = FakeHarness()
    agent = _agent(tmp_path, fake)
    try:
        agent.setup(session)
        (session.working_dir / "state.json").write_text("{}", encoding="utf-8")
        with pytest.raises(AgentError, match="Forbidden"):
            agent.run("must not execute")
        assert fake.tasks == []
    finally:
        agent.cleanup()
        session.cleanup()


def test_integration_failure_raises_agent_error(tmp_path: Path) -> None:
    session = _session(tmp_path)
    agent = _agent(tmp_path, FakeHarness(fail=RuntimeError("broken report")))
    try:
        agent.setup(session)
        with pytest.raises(AgentError, match="integration failure"):
            agent.run("task")
    finally:
        agent.cleanup()
        session.cleanup()


def test_partial_harness_failure_remains_exportable(tmp_path: Path) -> None:
    class PartialFailureHarness(FakeHarness):
        def run(
            self,
            *,
            workspace: Path,
            task: str,
            run_base: Path,
            progress: Any,
        ) -> RunResult:
            del workspace, task, progress
            run_dir = run_base / "run_partial"
            run_dir.mkdir(parents=True)
            (run_dir / "events.jsonl").write_text("", encoding="utf-8")
            (run_dir / "state.json").write_text(
                json.dumps(
                    {
                        "run_id": "run_partial",
                        "current_state": "IMPLEMENT",
                        "faults": [],
                        "reentry": {
                            "accepted_checkpoint_id": "checkpoint_safe",
                            "fault_records": {},
                        },
                    }
                ),
                encoding="utf-8",
            )
            raise FileNotFoundError("missing observed source")

    session = _session(tmp_path)
    agent = _agent(tmp_path, PartialFailureHarness())
    artifacts = tmp_path / "checkpoint-output" / "agent"
    try:
        agent.setup(session)
        with pytest.raises(AgentError, match="integration failure"):
            agent.run("partial task")
        assert agent.last_adapter_result is not None
        assert agent.last_adapter_result["harness_status"] == (
            "INFRASTRUCTURE_FAILURE"
        )
        assert agent.last_adapter_result["accepted_checkpoint"] == (
            "checkpoint_safe"
        )
        agent.save_artifacts(artifacts)
        assert (artifacts / "harness-run" / "state.json").is_file()
        exported = json.loads(
            (artifacts / "adapter-result.json").read_text(encoding="utf-8")
        )
        assert exported["adapter_error"]["type"] == "FileNotFoundError"
    finally:
        agent.cleanup()
        session.cleanup()


def test_harness_checkout_remains_pinned() -> None:
    config = HarnessConfig.from_toml(HARNESS_TOML)
    assert config.model.timeout_seconds == 0
    assert HARNESS_BASELINE.startswith("3213770")
