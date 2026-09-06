from __future__ import annotations

from pathlib import Path

import pytest

from slop_code.agent_runner.agent import Agent
from slop_code.agent_runner.agents.kludge import AGENT_NAME
from slop_code.agent_runner.agents.kludge import AGENT_VERSION
from slop_code.agent_runner.agents.kludge import OPENROUTER_BACKEND
from slop_code.agent_runner.agents.kludge import RETENTION_PARAMETER
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
    session = _session(tmp_path)

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

    agent.reset()

    assert agent.last_run_directory is None
    assert agent.last_terminal is None
    assert agent.checkpoint_sequence == 2


def test_cleanup_releases_the_session(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent.setup(_session(tmp_path))

    agent.cleanup()

    assert agent._session is None
