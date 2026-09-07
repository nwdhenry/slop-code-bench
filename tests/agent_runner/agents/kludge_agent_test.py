from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slop_code.agent_runner.agent import RETRY_PROMPT
from slop_code.agent_runner.agent import Agent
from slop_code.agent_runner.agents import kludge as kludge_module
from slop_code.agent_runner.agents.kludge import AGENT_NAME
from slop_code.agent_runner.agents.kludge import AGENT_VERSION
from slop_code.agent_runner.agents.kludge import ATTEMPT_BUDGET
from slop_code.agent_runner.agents.kludge import OPENROUTER_BACKEND
from slop_code.agent_runner.agents.kludge import RETENTION_PARAMETER
from slop_code.agent_runner.agents.kludge import STEP_BUDGET
from slop_code.agent_runner.agents.kludge import KludgeAgent
from slop_code.agent_runner.agents.kludge import KludgeConfig
from slop_code.agent_runner.credentials import CredentialType
from slop_code.agent_runner.credentials import ProviderCredential
from slop_code.agent_runner.models import AgentCostLimits
from slop_code.agent_runner.models import AgentError
from slop_code.agent_runner.registry import get_agent_cls
from slop_code.common import APIPricing
from slop_code.common import ModelDefinition
from slop_code.execution import LocalEnvironmentSpec
from slop_code.execution import Session

MODEL_ID = "qwen3.6-27b-100k"


def _model() -> ModelDefinition:
    return ModelDefinition(
        internal_name=MODEL_ID,
        provider="local_llama_cpp",
        pricing=APIPricing(),
    )


def _credential() -> ProviderCredential:
    return ProviderCredential(
        provider="local_llama_cpp",
        credential_type=CredentialType.ENV_VAR,
        value="1",
        source="LOCAL_LLAMA_CPP_READY",
        destination_key="LOCAL_LLAMA_CPP_READY",
    )


def _session(tmp_path: Path, entry_file: str | None = None) -> Session:
    starter = tmp_path / "starter"
    starter.mkdir()
    (starter / "README.md").write_text("starter\n", encoding="utf-8")
    session = Session.from_environment_spec(
        LocalEnvironmentSpec(type="local", name="test"),
        base_dir=starter,
        is_agent_infer=True,
        entry_file=entry_file,
    )
    session.prepare()
    return session


def _agent(tmp_path: Path, **overrides: object) -> KludgeAgent:
    config = KludgeConfig(
        run_root=tmp_path / "runs",
        cost_limits=AgentCostLimits(cost_limit=0.0, net_cost_limit=0.0),
        **overrides,
    )
    agent = Agent.from_config(
        config=config,
        model=_model(),
        credential=_credential(),
        problem_name="file_backup",
        verbose=False,
        image=None,
    )
    assert isinstance(agent, KludgeAgent)
    return agent


def test_the_registry_resolves_the_agent_by_name() -> None:
    assert get_agent_cls(AGENT_NAME) is KludgeAgent


def test_the_config_registers_and_defaults_to_the_local_binding(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)

    assert agent.version == AGENT_VERSION
    assert agent.model == MODEL_ID
    assert agent.backend == "kludge.openai_compat@1"


def test_the_identity_states_what_the_request_carries(tmp_path: Path) -> None:
    agent = _agent(tmp_path)

    identity = agent.identity()

    assert identity["agent"] == AGENT_NAME
    assert identity["version"] == AGENT_VERSION
    assert identity["model"] == MODEL_ID
    assert identity["flow"] == "kludge_swe.flow.swe_flow"
    assert RETENTION_PARAMETER not in identity


def test_an_openrouter_binding_states_its_retention(tmp_path: Path) -> None:
    agent = _agent(tmp_path, backend=OPENROUTER_BACKEND, data_collection="deny")

    assert agent.identity()[RETENTION_PARAMETER] == "deny"


def test_an_openrouter_binding_that_states_no_retention_is_refused(
    tmp_path: Path,
) -> None:
    with pytest.raises(AgentError, match=RETENTION_PARAMETER):
        _agent(tmp_path, backend=OPENROUTER_BACKEND)


def test_setup_makes_the_run_root_outside_the_workspace(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    session = _session(tmp_path, entry_file="main.py")

    agent.setup(session)

    assert agent.run_root.is_dir()
    assert session.working_dir not in agent.run_root.parents


def test_running_before_setup_is_refused(tmp_path: Path) -> None:
    agent = _agent(tmp_path)

    with pytest.raises(AgentError, match="setup"):
        agent.run("write the program")


def test_publishing_copies_the_accepted_tree_into_the_workspace(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    root = tmp_path / "run"
    accepted = root / "workspace" / "accepted"
    accepted.mkdir(parents=True)
    (accepted / "entry.py").write_text("print('ok')\n", encoding="utf-8")
    destination = tmp_path / "submission"
    destination.mkdir()

    agent._publish(root, destination)

    assert (destination / "entry.py").read_text(
        encoding="utf-8"
    ) == "print('ok')\n"


def test_publishing_carries_no_run_evidence_into_the_workspace(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    root = tmp_path / "run"
    accepted = root / "workspace" / "accepted"
    accepted.mkdir(parents=True)
    (accepted / "entry.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "events.jsonl").write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "submission"
    destination.mkdir()

    agent._publish(root, destination)

    assert not (destination / "events.jsonl").exists()
    assert sorted(path.name for path in destination.iterdir()) == ["entry.py"]


def test_save_artifacts_without_a_run_writes_nothing(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    artifacts = tmp_path / "artifacts"

    agent.save_artifacts(artifacts)

    assert not artifacts.exists()


def test_save_artifacts_copies_the_run_directory(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    (root / "events.jsonl").write_text("{}\n", encoding="utf-8")
    agent.last_run_directory = root
    artifacts = tmp_path / "artifacts"

    agent.save_artifacts(artifacts)

    assert (artifacts / "kludge-run" / "events.jsonl").is_file()


def test_reset_clears_the_checkpoint_state_and_keeps_the_sequence(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    agent.checkpoint_sequence = 2
    agent.last_run_directory = tmp_path
    agent.last_terminal = "done"
    agent.last_goal = "write the program"
    agent.blocked_runs = [{"kind": "BLOCKED"}]

    agent.reset()

    assert agent.last_run_directory is None
    assert agent.last_terminal is None
    assert agent.checkpoint_sequence == 2
    assert agent.last_goal is None
    assert agent.blocked_runs == []


def test_cleanup_releases_the_session(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent.setup(_session(tmp_path, entry_file="main.py"))

    agent.cleanup()

    assert agent._session is None


def test_publishing_removes_what_the_flow_removed(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    root = tmp_path / "run"
    accepted = root / "workspace" / "accepted"
    accepted.mkdir(parents=True)
    (accepted / "entry.py").write_text("print('ok')\n", encoding="utf-8")
    destination = tmp_path / "submission"
    (destination / "stale").mkdir(parents=True)
    (destination / "stale" / "old.py").write_text("gone\n", encoding="utf-8")
    (destination / "entry.py").write_text("old\n", encoding="utf-8")

    agent._publish(root, destination)

    assert (destination / "entry.py").read_text(
        encoding="utf-8"
    ) == "print('ok')\n"
    assert not (destination / "stale" / "old.py").exists()
    assert not (destination / "stale").exists()


def test_publishing_without_an_accepted_tree_is_refused(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    root = tmp_path / "run"
    root.mkdir()

    with pytest.raises(AgentError, match="no accepted tree"):
        agent._publish(root, tmp_path / "submission")


def test_the_identity_states_the_flow_budgets_and_the_wall_bound(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)

    identity = agent.identity()

    assert identity["budgets"]["max_steps_per_attempt"] == STEP_BUDGET
    assert identity["budgets"]["max_attempts_per_entry"] == ATTEMPT_BUDGET
    assert identity["wall_time_bound_s"] == 1800


def test_a_run_that_passes_its_wall_bound_is_an_agent_error(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path, wall_time_bound_s=1)

    async def _forever() -> None:
        await asyncio.sleep(30)

    with pytest.raises(AgentError, match="wall-time bound"):
        asyncio.run(agent._bounded(_forever()))


def test_usage_accumulates_steps_across_kernel_runs(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent.usage.steps = 4

    agent.usage.steps += 7

    assert agent.usage.steps == 11


def test_setup_refuses_a_problem_that_names_no_entry_file(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    session = _session(tmp_path)

    with pytest.raises(AgentError, match="file_backup"):
        agent.setup(session)


def test_setup_binds_the_formatted_entry_file(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    session = _session(tmp_path, entry_file="main.py")

    agent.setup(session)

    assert agent.entry_file == "main.py"
    assert agent.identity()["entry_file"] == "main.py"


def test_the_run_names_the_bound_entry_file_as_the_task_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flow's Task carries `entry` equal to the session's entry file, and
    no `path`.

    `path` is the decomposed item's own single-file assignment; a whole-task
    checkpoint carries none, or it would refuse every edit outside that one
    file. `entry` is the path the checkpoint's change targets.

    Everything the flow itself would do with that Task is stubbed out: this
    test verifies only what the adapter hands the kernel's `run`, not what the
    kernel does with it.
    """
    agent = _agent(tmp_path)
    session = _session(tmp_path, entry_file="main.py")
    agent.setup(session)

    captured: dict[str, object] = {}

    async def _accepted() -> SimpleNamespace:
        return SimpleNamespace(
            succeeded=True, terminal_state="accepted", cause=None
        )

    def _fake_run(flow: object, **kwargs: object) -> object:
        captured.update(kwargs["run_inputs"])
        return _accepted()

    monkeypatch.setattr(
        kludge_module, "seed_workspace", lambda root, source: None
    )
    monkeypatch.setattr(
        kludge_module.RunDirectory, "open", lambda root, coordinate: object()
    )
    monkeypatch.setattr(kludge_module, "swe_flow", lambda: object())
    monkeypatch.setattr(
        kludge_module.FileTreeDomain, "open", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        kludge_module, "implementations_of", lambda flow: object()
    )
    monkeypatch.setattr(kludge_module, "run", _fake_run)
    monkeypatch.setattr(agent, "_backend", lambda: object())
    monkeypatch.setattr(agent, "_publish", lambda root, destination: None)
    monkeypatch.setattr(agent, "_record_usage", lambda root: None)

    agent.run("write the program")

    assert captured["task"]["entry"] == "main.py"
    assert "path" not in captured["task"]


def _stub_run_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    agent: KludgeAgent,
    outcome_factory: object,
    captured_goals: list[str],
) -> None:
    """Stub every KLUDGE runtime call `run()` makes except `run_inputs`.

    `captured_goals` collects the `goal` field of each `run_inputs["task"]`
    the adapter hands the kernel, one entry per `run()` call.
    """

    def _fake_run(flow: object, **kwargs: object) -> object:
        captured_goals.append(kwargs["run_inputs"]["task"]["goal"])
        return outcome_factory()

    monkeypatch.setattr(
        kludge_module, "seed_workspace", lambda root, source: None
    )
    monkeypatch.setattr(
        kludge_module.RunDirectory, "open", lambda root, coordinate: object()
    )
    monkeypatch.setattr(kludge_module, "swe_flow", lambda: object())
    monkeypatch.setattr(
        kludge_module.FileTreeDomain, "open", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        kludge_module, "implementations_of", lambda flow: object()
    )
    monkeypatch.setattr(kludge_module, "run", _fake_run)
    monkeypatch.setattr(agent, "_backend", lambda: object())
    monkeypatch.setattr(agent, "_publish", lambda root, destination: None)
    monkeypatch.setattr(agent, "_record_usage", lambda root: None)


def test_a_retry_reruns_with_the_stored_checkpoint_goal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retry hands the kernel the checkpoint's original goal, never the
    harness's literal `RETRY_PROMPT`: the kernel keeps no session to resume
    across `run()` calls, so a run started on `RETRY_PROMPT` would run the
    whole SWE flow against that 33-character string as its task (#603).
    """
    agent = _agent(tmp_path)
    agent.setup(_session(tmp_path, entry_file="main.py"))

    async def _accepted() -> SimpleNamespace:
        return SimpleNamespace(
            succeeded=True, terminal_state="accepted", cause=None
        )

    goals: list[str] = []
    _stub_run_dependencies(monkeypatch, agent, _accepted, goals)

    agent.run("implement the checkpoint's spec")
    agent.retry()

    assert goals == [
        "implement the checkpoint's spec",
        "implement the checkpoint's spec",
    ]


def test_a_retry_before_any_run_carries_the_retry_prompt_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no stored goal to fall back to, a bare `run(RETRY_PROMPT)` carries
    that string through unchanged; there is nothing else to run it with."""
    agent = _agent(tmp_path)
    agent.setup(_session(tmp_path, entry_file="main.py"))

    async def _accepted() -> SimpleNamespace:
        return SimpleNamespace(
            succeeded=True, terminal_state="accepted", cause=None
        )

    goals: list[str] = []
    _stub_run_dependencies(monkeypatch, agent, _accepted, goals)

    agent.run(RETRY_PROMPT)

    assert goals == [RETRY_PROMPT]


def test_a_blocked_run_records_its_terminal_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run the kernel did not accept has its terminal cause recorded on the
    agent before `AgentError` is raised, so a later retry that succeeds does
    not erase this run's failure from the checkpoint's result (#603)."""
    agent = _agent(tmp_path)
    agent.setup(_session(tmp_path, entry_file="main.py"))

    async def _blocked() -> SimpleNamespace:
        cause = SimpleNamespace(
            kind=SimpleNamespace(name="BLOCKED"),
            detail="the chat-completions request failed: ReadTimeout",
        )
        return SimpleNamespace(succeeded=False, terminal_state="plan", cause=cause)

    goals: list[str] = []
    _stub_run_dependencies(monkeypatch, agent, _blocked, goals)

    with pytest.raises(AgentError, match="BLOCKED"):
        agent.run("implement the checkpoint's spec")

    assert agent.blocked_runs == [
        {
            "checkpoint": 1,
            "kind": "BLOCKED",
            "node": "plan",
            "detail": "the chat-completions request failed: ReadTimeout",
        }
    ]


def test_save_artifacts_writes_the_blocked_runs_record(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    agent.last_run_directory = root
    agent.blocked_runs = [
        {"checkpoint": 1, "kind": "BLOCKED", "node": "plan", "detail": "x"}
    ]
    artifacts = tmp_path / "artifacts"

    agent.save_artifacts(artifacts)

    written = json.loads((artifacts / "blocked-runs.json").read_text())
    assert written == agent.blocked_runs


def test_save_artifacts_writes_no_blocked_runs_file_when_there_are_none(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    agent.last_run_directory = root
    artifacts = tmp_path / "artifacts"

    agent.save_artifacts(artifacts)

    assert not (artifacts / "blocked-runs.json").exists()
