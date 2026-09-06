"""SCBench adapter for the KLUDGE SWE flow.

One SCBench checkpoint is one KLUDGE run. The adapter seeds a KLUDGE run
directory from the checkpoint's workspace, runs `kludge_swe`'s SWE flow over it
through the kernel, and writes the accepted tree back into the workspace SCBench
snapshots and evaluates. SCBench owns the specification, the workspace, the
hidden tests, and the evaluation; the adapter owns nothing of them.

KLUDGE evidence stays outside the submission workspace. The run directory lives
under the adapter's own run root, and `save_artifacts` copies it into SCBench's
artifacts directory, so nothing the kernel writes reaches the tree the hidden
tests score.

The adapter calls published KLUDGE distributions and no private module of that
repository: `kludge`, `kludge-api`, `kludge-swe`, `kludge-adapter-filetree`, the
two backend adapters, and `kludge-metrics` for usage. The pin those path
dependencies resolve to is in this branch's `pyproject.toml`.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any, Literal

from kludge.metrics import summarize
from kludge.runtime import LogicalClock
from kludge.runtime import RunCoordinate
from kludge.runtime import RunDirectory
from kludge.runtime import implementations_of
from kludge.runtime import run
from kludge_adapter_filetree import ExecutionEnvironment
from kludge_adapter_filetree import FileTreeDomain
from kludge_swe import domain as swe_domain
from kludge_swe.flow import IMPLEMENTER
from kludge_swe.flow import VERIFIER
from kludge_swe.flow import swe_flow
from kludge_swe.session import WORKSPACE_DIR
from kludge_swe.session import seed_workspace

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
from slop_code.execution import Session

AGENT_NAME = "kludge"
AGENT_VERSION = "0.1.0"
"""The adapter's own version. It is a request-affecting identity: a change to
the loop below is a change to what ran, so a campaign records it."""

ACCEPTED = "accepted"
RUN_ID = "scbench"
DEFAULT_ENDPOINT = "http://127.0.0.1:11434"
DEFAULT_BACKEND = "kludge.openai_compat@1"
OPENROUTER_BACKEND = "kludge.openrouter@1"

RETENTION_PARAMETER = "provider.data_collection"
"""The OpenRouter request member that states whether the provider may retain
the prompt for training. `deny` selects endpoints that collect nothing. The
adapter refuses an OpenRouter binding that leaves it unstated: a benchmark
prompt carries the problem set's canary, and a request that does not say where
its text may go says nothing a record can carry (#277)."""


class KludgeConfig(AgentConfigBase, agent_type=AGENT_NAME):
    """Configuration for the KLUDGE SWE-flow adapter."""

    type: Literal["kludge"] = AGENT_NAME
    version: str = AGENT_VERSION
    run_root: Path = Path(".kludge-runs")
    endpoint: str = DEFAULT_ENDPOINT
    backend: str = DEFAULT_BACKEND
    data_collection: str | None = None


class KludgeAgent(Agent):
    """Run one KLUDGE SWE flow per SCBench checkpoint."""

    def __init__(
        self,
        *,
        problem_name: str,
        verbose: bool,  # noqa: FBT001
        cost_limits: AgentCostLimits,
        version: str,
        run_root: Path,
        endpoint: str,
        backend: str,
        model: str,
        data_collection: str | None,
    ) -> None:
        super().__init__(
            agent_name=AGENT_NAME,
            problem_name=problem_name,
            cost_limits=cost_limits,
            pricing=APIPricing(),
            verbose=verbose,
        )
        self.version = version
        self.run_root = Path(run_root).resolve()
        self.endpoint = endpoint
        self.backend = backend
        self.model = model
        self.data_collection = data_collection
        self._session: Session | None = None
        self.checkpoint_sequence = 0
        self.last_run_directory: Path | None = None
        self.last_terminal: str | None = None

    @classmethod
    def _from_config(
        cls,
        config: AgentConfigBase,
        model: ModelDefinition,
        credential: ProviderCredential,  # noqa: ARG003
        problem_name: str,
        verbose: bool,  # noqa: FBT001
        image: str | None,  # noqa: ARG003
        thinking_preset: ThinkingPreset | None = None,  # noqa: ARG003
        thinking_max_tokens: int | None = None,  # noqa: ARG003
    ) -> KludgeAgent:
        if not isinstance(config, KludgeConfig):
            raise TypeError(
                f"Expected KludgeConfig, got {type(config).__name__}"
            )
        if config.backend == OPENROUTER_BACKEND and not config.data_collection:
            raise AgentError(
                f"the OpenRouter binding states no {RETENTION_PARAMETER}; a "
                "campaign cell declares where its prompts may go before it runs"
            )
        return cls(
            problem_name=problem_name,
            verbose=verbose,
            cost_limits=config.cost_limits,
            version=config.version,
            run_root=config.run_root,
            endpoint=config.endpoint,
            backend=config.backend,
            model=model.internal_name or model.name,
            data_collection=config.data_collection,
        )

    def identity(self) -> dict[str, Any]:
        """Return the request-affecting identity of this binding."""
        record: dict[str, Any] = {
            "agent": AGENT_NAME,
            "version": self.version,
            "implementation": self.backend,
            "endpoint": self.endpoint,
            "model": self.model,
            "flow": "kludge_swe.flow.swe_flow",
        }
        if self.backend == OPENROUTER_BACKEND:
            record[RETENTION_PARAMETER] = self.data_collection
        return record

    def setup(self, session: Session) -> None:
        """Take the session and make the run root that holds KLUDGE evidence."""
        self._session = session
        self.run_root.mkdir(parents=True, exist_ok=True)

    def _backend(self) -> Any:
        """Return the semantic backend this binding names.

        The import is inside the call because a binding names one adapter and a
        host installs the one it uses.
        """
        if self.backend == OPENROUTER_BACKEND:
            from kludge_adapter_openrouter import ProviderPolicy
            from kludge_adapter_openrouter import openrouter

            return openrouter(
                self.model,
                provider=ProviderPolicy(data_collection=self.data_collection),
            )
        from kludge_adapter_openai import openai_compat

        return openai_compat(base_url=self.endpoint, model=self.model)

    def run(self, task: str) -> None:
        """Run one KLUDGE flow over the checkpoint's workspace."""
        session = self._session
        if session is None:
            raise AgentError("the agent was run before setup gave it a session")
        self.checkpoint_sequence += 1
        root = self.run_root / f"{self.problem_name}-{self.checkpoint_sequence}"
        if root.exists():
            shutil.rmtree(root)
        workspace_source = session.working_dir
        seed_workspace(root, workspace_source)
        run_dir = RunDirectory.open(root, coordinate=RunCoordinate(RUN_ID))
        flow = swe_flow()
        environment: ExecutionEnvironment | None = None
        workspace = FileTreeDomain.open(
            root / WORKSPACE_DIR,
            domain=swe_domain.WORKSPACE,
            environment=environment,
        )
        backend = self._backend()
        outcome = asyncio.run(
            run(
                flow,
                run_dir=run_dir,
                clock=LogicalClock(),
                implementations=implementations_of(flow),
                run_inputs={"task": swe_domain.item_task(task, None)},
                domains={swe_domain.WORKSPACE: workspace},
                backends={IMPLEMENTER: backend, VERIFIER: backend},
            )
        )
        self.last_run_directory = root
        self.last_terminal = outcome.terminal_state
        self._publish(root, workspace_source)
        self._record_usage(root)

    def _publish(self, root: Path, destination: Path) -> None:
        """Copy the accepted tree into the workspace SCBench evaluates.

        The accepted tree is what the kernel admitted. Every other path of the
        run directory is KLUDGE evidence and stays out of the workspace.
        """
        accepted = root / WORKSPACE_DIR / ACCEPTED
        if not accepted.is_dir():
            return
        for path in sorted(accepted.rglob("*")):
            if "__pycache__" in path.parts:
                continue
            target = destination / path.relative_to(accepted)
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, target)

    def _record_usage(self, root: Path) -> None:
        """Record what the run spent, reading the run's own persisted records."""
        summary = summarize(root)
        totals = next(iter(summary.usage), None)
        tokens = TokenUsage(
            input=int(totals.input_tokens or 0) if totals else 0,
            output=int(totals.output_tokens or 0) if totals else 0,
            reasoning=int(totals.reasoning_output_tokens or 0) if totals else 0,
        )
        self.usage.step(cost=0.0, tokens=tokens)
        self.usage.steps = summary.step_count

    def reset(self) -> None:
        """Clear per-checkpoint state.

        Each checkpoint is its own KLUDGE run with its own run directory, so
        nothing of the last one is carried; the sequence and the accumulated
        cost are not per-checkpoint state and stay.
        """
        self.last_run_directory = None
        self.last_terminal = None

    def save_artifacts(self, path: Path) -> None:
        """Copy this checkpoint's KLUDGE run directory into the artifacts path."""
        source = self.last_run_directory
        if source is None or not source.is_dir():
            return
        path.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            source,
            path / "kludge-run",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__"),
        )

    def cleanup(self) -> None:
        """Release the session reference. The run root is retained evidence."""
        self._session = None


register_agent(AGENT_NAME, KludgeAgent)
