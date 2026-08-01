# `file_backup` Baseline v0

This is the complete counted baseline for local `qwen3.6-27b-100k` against
fault-typed-harness-demo commit
`3213770b4f64c9597b47b4230d15a6aac56c7112`. All five sessions are included.
No prompt, harness, adapter, model, or configuration input changed after the
first counted session.

## Frozen treatment

- Campaign: `file-backup-baseline-v0`
- Treatment fingerprint:
  `ed6c142cf2fa3295d474310e6ef663d41a25e15f41094cacfdb0bcaf679664fd`
- SCBench treatment commit: `2382207`
- Harness commit: `3213770b4f64c9597b47b4230d15a6aac56c7112`
- Problem catalog: `4d38d300059667d57e43c31969bc455f5c338b52`
- Temperature: 0
- Seed: 7 for each v0 repetition, as frozen in the recorded treatment
- Model client timeout: 0 (unbounded)
- Campaign root:
  `experiments/output/scbench/file-backup-baseline-v0`

No model generation was canceled or bounded by an arbitrary wall-clock
timeout. The campaign was unattended, retained every run, and continued to the
next fresh session after an infrastructure failure.

## Outcome

| Metric | Result |
| --- | ---: |
| Full sessions passed | 0/5 |
| External checkpoints passed | 0/20 |
| Checkpoints ending `BLOCKED` | 11 |
| Checkpoints ending in adapter infrastructure failure | 5 |
| Checkpoints not reached after session infrastructure failure | 4 |
| Clean harness completion rate | 0% |
| Integrity-failure sessions | 0 |
| Model calls | 236 |
| Prompt tokens | 1,298,464 |
| Generated tokens | 13,332 |
| Model seconds | 614.696 |
| Recovery yield | 0% |
| Reported regression rate across executed external suites | 89.4737% |

The checkpoint survival curve is zero at every checkpoint because no session
passed checkpoint 1. The result is a valid 0/5 baseline, not a selected-best-run
report.

## Stage concentration

| Stage | Calls | Tokens | Model seconds | Attempts | Faults |
| --- | ---: | ---: | ---: | ---: | ---: |
| Repository inspection | 221 | 1,222,897 | 571.282 | 38 | 35 |
| Plan | 15 | 88,899 | 43.414 | 5 | 1 |

Repository inspection consumed 93.6% of model calls and 94.2% of prompt plus
generated tokens. No counted checkpoint progressed beyond PLAN.

The bounded fault registry reports 20 `MODEL_OUTPUT_INVALID`, 11
`RETRY_EXHAUSTED`, and 5 `INFRASTRUCTURE_FAILURE` occurrences. Terminal outcomes
were 11 retry exhaustion blocks, 2 model-output blocks, and 1 typed
infrastructure block among naturally reported runs; five adapter-level failures
ended their sessions before all four checkpoints could run.

## Failure clusters

Four adapter failures were an unhandled `FileNotFoundError` from `read_file`.
The model requested a source path that did not exist in its isolated attempt
workspace. `Workspace.read_text()` allowed the host exception to escape instead
of returning a native, typed tool result. This is a harness correctness defect,
not evidence that SCBench should retry the checkpoint.

One adapter failure was a projection write race. The running harness and the
read-only observer both used the same `projection.json.tmp` name, and Windows
rejected one atomic replacement with `WinError 32`. Read-only observation must
not be able to terminate a run.

The remaining natural blocks were concentrated in repository inspection. The
model repeatedly requested tools without producing a valid terminal inspection
artifact until the bounded retry policy closed the run. This is the first
prompt/workflow capability cluster, but it must not be treated experimentally
until the two infrastructure defects above are repaired and a new treatment is
frozen.

## Post-processing correction

The original collector required a completed `adapter-result.json` and therefore
omitted sessions whose last checkpoint ended in an adapter infrastructure
failure. The corrected collector reconstructs partial metrics from immutable
`state.json`, `events.jsonl`, and external run identity; marks later checkpoints
`NOT_RUN`; and never fabricates a checkpoint result. Recollection changed only
derived reports. It did not rerun or alter counted evidence.

## First prompt-treatment proposal

After infrastructure qualification, the first controlled prompt hypothesis is:

> A repository-inspection contract that explicitly requires a bounded relevant
> path listing, authoritative reads of selected files, and a terminal findings
> submission will reduce repository-inspection retry exhaustion without moving
> failures to PLAN.

Only the repository-inspection prompt component may change. The model, seeds,
task specifications, context budget, tools, retry limits, validation profile,
and recovery policy remain fixed. The treatment succeeds only if it improves
external checkpoint survival, preserves all integrity invariants, reduces
inspection recurrence across five sessions, and does not merely transfer an
equivalent failure to PLAN.

