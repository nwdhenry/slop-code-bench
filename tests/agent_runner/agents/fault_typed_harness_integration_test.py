from __future__ import annotations

import hashlib
import json
import queue
from pathlib import Path
from typing import Any

from agent_harness import HarnessConfig
from agent_harness import RunResult

from slop_code.agent_runner import runner
from slop_code.agent_runner.agents.fault_typed_harness import (
    FaultTypedHarnessAgent,
)
from slop_code.agent_runner.models import AgentCostLimits
from slop_code.agent_runner.models import AgentRunSpec
from slop_code.evaluation import CheckpointConfig
from slop_code.evaluation import PassPolicy
from slop_code.evaluation import ProblemConfig
from slop_code.execution import LocalEnvironmentSpec
from slop_code.execution.models import CommandConfig
from slop_code.execution.models import EnvironmentConfig
from slop_code.execution.models import SetupConfig

ROOT = Path(__file__).resolve().parents[3]
HARNESS_TOML = ROOT / "configs" / "harness" / "fault-typed-scbench.toml"
MODEL_ID = "qwen3.6-27b-100k"


class FourCheckpointHarness:
    def __init__(self) -> None:
        self.tasks: list[str] = []
        self.workspace_paths: list[Path] = []

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
        self.tasks.append(task)
        self.workspace_paths.append(workspace.resolve())
        checkpoint = len(self.tasks)
        prior = ""
        implementation = workspace / "implementation.py"
        if implementation.exists():
            prior = implementation.read_text(encoding="utf-8")
        implementation.write_text(
            prior + f"CHECKPOINT_{checkpoint} = True\n", encoding="utf-8"
        )
        run_id = f"run_fixture_{checkpoint}"
        run_dir = run_base / run_id
        run_dir.mkdir(parents=True)
        payload = {
            "telemetry": {
                "prompt_tokens": 10 * checkpoint,
                "generated_tokens": checkpoint,
                "client_duration_seconds": 0.1 * checkpoint,
            }
        }
        raw = json.dumps(payload).encode()
        digest = hashlib.sha256(raw).hexdigest()
        artifact = run_dir / "artifacts" / "sha256" / digest[:2] / digest[2:]
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(raw)
        (run_dir / "events.jsonl").write_text(
            json.dumps(
                {
                    "event_type": "MODEL_RESPONSE",
                    "payload_ref": f"sha256:{digest}",
                    "prompt_fingerprint": f"prompt-{checkpoint}",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "interactions.jsonl").write_text("{}\n", encoding="utf-8")
        (run_dir / "projection.json").write_text("{}\n", encoding="utf-8")
        (run_dir / "state.json").write_text("{}\n", encoding="utf-8")
        (run_dir / "attempts").mkdir()
        report = {
            "run_id": run_id,
            "status": "COMPLETE",
            "current_state": "COMPLETE",
            "task": task,
            "faults": [],
            "reentry": {
                "accepted_checkpoint_id": f"checkpoint_{checkpoint}",
                "fault_records": {},
            },
            "final_result": {"verdict": "APPROVE"},
        }
        report_path = run_dir / "run-report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return RunResult(
            run_id=run_id,
            status="COMPLETE",
            state="COMPLETE",
            run_dir=run_dir,
            report_path=report_path,
        )


def test_scbench_drives_four_fresh_runs_over_one_persistent_workspace(
    tmp_path: Path,
) -> None:
    checkpoints = {
        f"checkpoint_{index}": CheckpointConfig(
            name=f"checkpoint_{index}",
            version=1,
            order=index,
            state="Core Tests",
            spec_override=f"EXACT SPECIFICATION {index}",
        )
        for index in range(1, 5)
    }
    problem = ProblemConfig(
        name="four_checkpoint_fixture",
        path=tmp_path,
        version=1,
        description="Deterministic adapter qualification fixture",
        tags=["fixture"],
        entry_file="implementation",
        checkpoints=checkpoints,
    )
    environment = LocalEnvironmentSpec(
        type="local",
        name="fixture-local",
        environment=EnvironmentConfig(include_os_env=True),
        setup=SetupConfig(),
        commands=CommandConfig(command="python", entry_file="{entry_file}.py"),
    )
    fake = FourCheckpointHarness()
    harness_config = HarnessConfig.from_toml(HARNESS_TOML).with_model(MODEL_ID)
    agent = FaultTypedHarnessAgent(
        problem_name=problem.name,
        verbose=False,
        cost_limits=AgentCostLimits(
            cost_limit=0,
            net_cost_limit=0,
            step_limit=0,
            max_retries=0,
        ),
        harness_config_path=HARNESS_TOML,
        harness_config=harness_config,
        run_root=tmp_path / "external-runs",
        selected_model=MODEL_ID,
        copy_full_run=True,
        harness=fake,  # type: ignore[arg-type]
    )
    output = tmp_path / "scbench-output"
    output.mkdir()
    result = runner.run_agent(
        run_spec=AgentRunSpec(
            seed=7,
            template="{{ spec }}",
            problem=problem,
            environment=environment,
            image="local",
            pass_policy=PassPolicy.ANY,
            skip_evaluation=True,
            verbose=False,
            agent_type="fault_typed_harness",
            model_name=MODEL_ID,
        ),
        agent=agent,
        output_path=output,
        progress_queue=queue.Queue(),
    )
    assert result["summary"]["state"] == "completed"
    assert fake.tasks == [
        "EXACT SPECIFICATION 1",
        "EXACT SPECIFICATION 2",
        "EXACT SPECIFICATION 3",
        "EXACT SPECIFICATION 4",
    ]
    assert len(set(fake.workspace_paths)) == 1
    final_source = (
        output / "checkpoint_4" / "snapshot" / "implementation.py"
    ).read_text(encoding="utf-8")
    assert final_source.splitlines() == [
        "CHECKPOINT_1 = True",
        "CHECKPOINT_2 = True",
        "CHECKPOINT_3 = True",
        "CHECKPOINT_4 = True",
    ]
    for index in range(1, 5):
        agent_dir = output / f"checkpoint_{index}" / "agent"
        adapter_result = json.loads(
            (agent_dir / "adapter-result.json").read_text(encoding="utf-8")
        )
        assert adapter_result["checkpoint_index"] == index
        assert adapter_result["harness_run_id"] == f"run_fixture_{index}"
        assert (agent_dir / "harness-run" / "events.jsonl").is_file()
