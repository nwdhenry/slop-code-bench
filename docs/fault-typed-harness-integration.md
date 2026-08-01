# Fault-Typed Harness SCBench Integration

This document is the implementation and accountability ledger for the native
SCBench adapter. The benchmark treatment is frozen to fault-typed-harness-demo
commit `3213770b4f64c9597b47b4230d15a6aac56c7112`, tagged
`benchmark-baseline-v0`. Harness source and prompt changes are prohibited until
the initial counted campaign is complete.

## Pinned boundary

- SCBench upstream: `8e3a8b693f3c5e48143aeb7cb5b1beda1f19c44b`
- SCBench branch: `feat/fault-typed-harness-agent`
- Harness: `3213770b4f64c9597b47b4230d15a6aac56c7112`
- Harness tag: `benchmark-baseline-v0`
- Problem set: `4d38d300059667d57e43c31969bc455f5c338b52` (SCBench catalog v1.0)
- Runtime: uv-managed Python 3.12, local SCBench environment only
- Initial problem: four-checkpoint `file_backup`
- Model treatment: local `qwen3.6-27b-100k`, temperature 0, seed schedule
  declared by the campaign, `timeout_seconds = 0`

SCBench owns checkpoint specifications, workspace persistence, hidden tests,
external evaluation, snapshots, and result directories. The harness owns its
inner workflow and accepted candidate authority. The adapter forwards task text
byte-for-byte, keeps harness evidence outside the submission workspace, maps
usage, and exports self-contained evidence. It never starts an observer, exposes
hidden results to the harness, or interprets harness `COMPLETE` as external
correctness.

## Milestones

| Milestone | Status | Commit | Acceptance evidence |
| --- | --- | --- | --- |
| M18 Freeze and integration scaffold | Complete | `c0fdc73` | pinned revisions, Python 3.12 uv environment, local model/config dry-run, frozen harness suite: 132 passed and 1 skipped |
| M19 Native SCBench adapter | Complete | `c0fdc73` | registration, construction, lifecycle, usage, COMPLETE/BLOCKED, self-contained export |
| M20 Isolation and leakage | Complete | `0c66365` | external-root enforcement, byte-exact tasks, source manifests, forbidden-file detection |
| M21 End-to-end qualification | In progress | `7720885`, `0e9ab98`, `bb09fbd` | deterministic four-checkpoint fixture and combined scorecard pass; natural local-Qwen `file_backup` qualification pending |
| M22 Campaign runner | Complete | `64690c2` | unattended repetition, fixed seed schedule, failure continuation, immutable treatment and result fingerprints |
| M23 Five-run baseline | Pending | Pending | all five sessions retained and reported without treatment changes or selection |
| M24 Prompt evaluation loop | Pending | Pending | first single-variable hypothesis and treatment contract after baseline review |
| M25 Harness design evaluation | Pending | Pending | controlled design-condition scaffold; no premature ablation claim |
| M26 Suite expansion | Pending | Pending | deferred until `file_backup` adapter reliability is demonstrated |

## Integrity gates

The collector treats accepted-candidate contamination, invalid completion,
APPROVE with an open fault, hidden-test leakage, invalidated-evidence reuse,
missing task/prompt fingerprints, and missing accepted-checkpoint identity as
hard failures. Internal harness GREEN and external SCBench PASS remain separate.

Every counted session records harness, adapter, SCBench, and problem-set commits;
model identity and file hash when available; llama.cpp identity; prompt/template
configuration; context and generation limits; seed; retry limits; validation
configuration; and host/Python information. Qualification runs are never counted
as baseline evidence.

## Change control

Integration defects are corrected narrowly in this SCBench branch. A required
harness change receives its own harness commit, invalidates prior counted runs,
and creates a new benchmark treatment. Milestones are marked complete only after
their acceptance checks and commit are recorded here.

## Reproduction

Create the integration environment from the pinned SCBench checkout using
Python 3.12 and install the frozen harness as an editable local package:

```text
uv sync --python 3.12
uv pip install -e ../fault-typed-harness-demo
```

Set `LOCAL_LLAMA_CPP_READY=1`, keep the pinned problem catalog checkout at
`4d38d300059667d57e43c31969bc455f5c338b52`, and verify configuration without
executing a model:

```text
uv run slop-code run --agent fault-typed-harness --model local_llama_cpp/qwen-local --environment local-py --prompt just-solve --problem file_backup --dry-run
```

Run the adapter acceptance suites:

```text
uv run pytest tests/agent_runner/agents/fault_typed_harness_agent_test.py tests/agent_runner/agents/fault_typed_harness_integration_test.py -q
uv run ruff check src/slop_code/agent_runner/agents/fault_typed_harness.py tests/agent_runner/agents/fault_typed_harness_agent_test.py tests/agent_runner/agents/fault_typed_harness_integration_test.py
```

The upstream SCBench suite is not fully Windows-portable at the pinned commit.
On this host it reports 1571 passed, 21 skipped, 71 failed, and 14 errors from
pre-existing POSIX path, executable-bit, shell-builtin, and text-encoding
assumptions. The adapter-focused suites above are the local acceptance gate;
the complete upstream suite remains a Linux CI gate.
