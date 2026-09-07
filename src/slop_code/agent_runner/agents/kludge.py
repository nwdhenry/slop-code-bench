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
two backend adapters, and `kludge-metrics` for usage. The revision those
dependencies are pinned to is in this branch's `pyproject.toml`.

**What follows the reference adapter.** `feat/fault-typed-harness-agent` on this
fork is the working reference at the same upstream pin, and this adapter mirrors
it where the harness contract is the same: it takes a manifest of the submission
tree before and after each checkpoint and records both identities; it raises
`AgentError` rather than returning quietly when a checkpoint produced no
admissible result; it reports usage from the run's own persisted records; and it
keeps its evidence outside the submission workspace, copying it in through
`save_artifacts`.

**What is treatment-specific.** The reference hands the submission workspace to
its own loop and edits it in place. A KLUDGE run owns a run directory and a
workspace domain beneath it, so this adapter seeds that workspace from the
submission tree, runs the kernel, and mirrors the accepted tree back —
additions, changes, and deletions alike. That copy is the treatment's shape and
no deviation from the harness contract; the manifests either side of it are what
make the copy checkable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
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

from slop_code.agent_runner.agent import RETRY_PROMPT
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
STEP_BUDGET = 15
ATTEMPT_BUDGET = 3
STAGE_ENTRY_BUDGET = 1
IMPLEMENT_ENTRY_BUDGET = 2
"""The SWE flow's own declared budgets, recorded here so a campaign freezes the
numbers its runs were bound by. This adapter sets none of its own.
`docs/evidence/swe-live-1` of the KLUDGE repository established the step and
attempt values — its implement attempts reached a proposal at steps 15, 15, and
12, the accepted one on the third attempt — and
`docs/evidence/quixbugs-campaign-1` is the campaign that produced accepted trees
under them."""
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
    wall_time_bound_s: int = 1800


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
        wall_time_bound_s: int,
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
        self.wall_time_bound_s = wall_time_bound_s
        self._session: Session | None = None
        self.entry_file: str | None = None
        self.checkpoint_sequence = 0
        self.last_run_directory: Path | None = None
        self.last_terminal: str | None = None
        self.last_manifests: dict[str, str] = {}
        self.last_goal: str | None = None
        """The task string the last non-retry `run()` received. A retry hands
        the kernel a fresh run with no session to resume (#603): this is what
        lets it repeat the checkpoint's own goal instead of the harness's
        generic retry prompt."""
        self.blocked_runs: list[dict[str, Any]] = []
        """The terminal cause of every kernel run this checkpoint attempted
        and did not accept, oldest first. A retry that then succeeds would
        otherwise erase the first run's failure from the checkpoint's result
        (kludge issue #603); this keeps it. Cleared per checkpoint by
        `reset()`."""

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
            wall_time_bound_s=config.wall_time_bound_s,
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
            "budgets": {
                "max_steps_per_attempt": STEP_BUDGET,
                "max_attempts_per_entry": ATTEMPT_BUDGET,
                "stage_entries_per_run": STAGE_ENTRY_BUDGET,
                "implement_entries_per_run": IMPLEMENT_ENTRY_BUDGET,
                "source": "the declared constants of kludge_swe.flow",
            },
            "wall_time_bound_s": self.wall_time_bound_s,
            "entry_file": self.entry_file,
            "blocked_runs": list(self.blocked_runs),
        }
        if self.backend == OPENROUTER_BACKEND:
            record[RETENTION_PARAMETER] = self.data_collection
        return record

    def setup(self, session: Session) -> None:
        """Take the session and make the run root that holds KLUDGE evidence.

        The session's entry file is the checkpoint's declared implementation
        target, formatted for the environment by the harness (`local-py`:
        `{entry_file}.py`). The flow's inspection establishes an absent target
        only when a Task names that path (#603), so a problem that names none
        is refused here rather than silently inspecting nothing.
        """
        self._session = session
        self.entry_file = session.entry_file
        if not self.entry_file:
            raise AgentError(
                f"the problem {self.problem_name!r} names no entry file; the "
                "KLUDGE flow has no path to assign the item it runs"
            )
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
        """Run one KLUDGE flow over the checkpoint's workspace.

        A retry hands this the harness's fixed `RETRY_PROMPT` and no session to
        resume, so a KLUDGE run started on it directly would run the kernel's
        whole 15-step, 3-attempt flow against a 33-character goal instead of
        the checkpoint's own task (kludge issue #603). The stored goal from
        this checkpoint's first `run()` is used instead, and a fresh kernel run
        starts on it, since the kernel keeps no session across runs to resume.
        """
        session = self._session
        if session is None:
            raise AgentError("the agent was run before setup gave it a session")
        if task == RETRY_PROMPT and self.last_goal is not None:
            goal = self.last_goal
        else:
            goal = task
            self.last_goal = task
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
        before = _manifest(workspace_source)
        outcome = asyncio.run(
            self._bounded(
                run(
                    flow,
                    run_dir=run_dir,
                    clock=LogicalClock(),
                    implementations=implementations_of(flow),
                    run_inputs={
                        "task": swe_domain.item_task(
                            goal, entry=self.entry_file
                        )
                    },
                    domains={swe_domain.WORKSPACE: workspace},
                    backends={IMPLEMENTER: backend, VERIFIER: backend},
                )
            )
        )
        self.last_run_directory = root
        self.last_terminal = outcome.terminal_state
        self._publish(root, workspace_source)
        self._record_usage(root)
        self.last_manifests = {
            "before": _identity(before),
            "after": _identity(_manifest(workspace_source)),
        }
        if not outcome.succeeded:
            self.blocked_runs.append(
                {
                    "checkpoint": self.checkpoint_sequence,
                    "kind": outcome.cause.kind.name,
                    "node": outcome.terminal_state,
                    "detail": outcome.cause.detail,
                }
            )
            raise AgentError(
                f"the KLUDGE run of {self.problem_name} checkpoint "
                f"{self.checkpoint_sequence} reached no accepted terminal: "
                f"{outcome.cause.kind.name} at {outcome.terminal_state!r} — "
                f"{outcome.cause.detail}"
            )

    async def _bounded(self, coroutine: Any) -> Any:
        """Await one kernel run under this checkpoint's wall-time bound.

        The bound is the harness-side one: `asyncio.wait_for` cancels the run's
        task at it and this adapter reports the cancellation to the harness as an
        agent error, so a checkpoint that hung is a recorded failure and never a
        silent pass. It is the adapter's bound and not the kernel's own
        cancellation protocol, which lives in the KLUDGE repository's own runner.
        """
        try:
            return await asyncio.wait_for(coroutine, self.wall_time_bound_s)
        except TimeoutError as expired:
            raise AgentError(
                f"the KLUDGE run of {self.problem_name} checkpoint "
                f"{self.checkpoint_sequence} passed its "
                f"{self.wall_time_bound_s}s wall-time bound and was cancelled"
            ) from expired

    def _publish(self, root: Path, destination: Path) -> None:
        """Mirror the accepted tree into the workspace SCBench evaluates.

        The accepted tree is what the kernel admitted, and the submission tree is
        made to equal it: a file the flow added or changed is written, and a file
        the flow removed is removed here too. A copy-only publish would leave a
        deleted file standing in the tree the hidden tests score, which is a
        result the run did not produce.

        Every other path of the run directory is KLUDGE evidence and stays out of
        the workspace.
        """
        accepted = root / WORKSPACE_DIR / ACCEPTED
        if not accepted.is_dir():
            raise AgentError(
                f"the KLUDGE run of {self.problem_name} left no accepted tree at "
                f"{accepted}; nothing was published to the submission workspace"
            )
        kept = set()
        for path in sorted(accepted.rglob("*")):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(accepted)
            kept.add(relative)
            target = destination / relative
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, target)
        for path in sorted(destination.rglob("*"), reverse=True):
            if "__pycache__" in path.parts:
                continue
            if path.relative_to(destination) in kept:
                continue
            if path.is_file():
                path.unlink()
            elif path.is_dir() and not any(path.iterdir()):
                path.rmdir()

    def _record_usage(self, root: Path) -> None:
        """Record what the run spent, reading the run's own persisted records.

        Cost is left to the harness's own pricing. This adapter reports token
        counts and steps and prices nothing: a zero it wrote would read as a
        measured cost of nothing, and an unpriced run has an unknown cost and
        never a zero one.

        Steps accumulate across the checkpoint rather than replacing the count,
        so a checkpoint that ran more than one kernel run reports what it spent.
        """
        summary = summarize(root)
        totals = next(iter(summary.usage), None)
        tokens = TokenUsage(
            input=int(totals.input_tokens or 0) if totals else 0,
            output=int(totals.output_tokens or 0) if totals else 0,
            reasoning=int(totals.reasoning_output_tokens or 0) if totals else 0,
        )
        priced = self.pricing.get_cost(tokens) if self.pricing else 0.0
        self.usage.step(cost=priced, tokens=tokens)
        self.usage.steps += summary.step_count

    def reset(self) -> None:
        """Clear per-checkpoint state.

        Each checkpoint is its own KLUDGE run with its own run directory, so
        nothing of the last one is carried; the sequence and the accumulated
        cost are not per-checkpoint state and stay.
        """
        self.last_run_directory = None
        self.last_terminal = None
        self.last_manifests = {}
        self.last_goal = None
        self.blocked_runs = []

    def save_artifacts(self, path: Path) -> None:
        """Copy this checkpoint's KLUDGE run directory into the artifacts path.

        A retry that later succeeds still leaves the checkpoint's earlier
        blocked runs recorded here: `run_checkpoint` (`agent.py`) overwrites
        `had_error`/`error_message` with the retry's own outcome, so this file
        is where the earlier failure survives (kludge issue #603).
        """
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
        if self.blocked_runs:
            (path / "blocked-runs.json").write_text(
                json.dumps(self.blocked_runs, indent=2), encoding="utf-8"
            )

    def cleanup(self) -> None:
        """Release the session reference. The run root is retained evidence."""
        self._session = None


def _manifest(root: Path) -> list[tuple[str, str]]:
    """Return one entry per file of a tree: its path and the digest of its bytes."""
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append((path.relative_to(root).as_posix(), digest))
    return entries


def _identity(manifest: list[tuple[str, str]]) -> str:
    """Return one digest over a whole tree manifest."""
    joined = "\n".join(f"{name} {digest}" for name, digest in manifest)
    return "sha256:" + hashlib.sha256(joined.encode("utf-8")).hexdigest()


register_agent(AGENT_NAME, KludgeAgent)
