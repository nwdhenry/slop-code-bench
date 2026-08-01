# `file_backup` Qualification v0

This is a debugging qualification, not counted benchmark evidence. It validates
the native SCBench lifecycle against the frozen harness commit
`3213770b4f64c9597b47b4230d15a6aac56c7112` and local model
`qwen3.6-27b-100k`.

## Integration findings

Three pre-model integration defects were found and corrected in separate
qualification attempts:

1. The campaign's agent type `fault_typed_harness` did not resolve the required
   config filename `fault-typed-harness.yaml` (`30da38e`).
2. SCBench read UTF-8 checkpoint specifications using the Windows cp1252 host
   default (`d58777a`).
3. SCBench wrote rendered prompts using the same host default (`7c83b28`).

Those attempts made no model calls. A fourth collector-only defect incorrectly
required prompt fingerprints on response events instead of authoritative prompt
events; it was corrected after the natural qualification (`9188c39`).

## Natural four-checkpoint result

Qualification root:

```text
experiments/output/scbench/file-backup-qualification-v0-4
```

All four checkpoints were invoked against one persistent SCBench workspace.
Each checkpoint used a fresh harness run, reached `BLOCKED` naturally, exported
self-contained evidence, and was externally evaluated by SCBench. No generation
was canceled and the model client timeout remained zero.

| Checkpoint | Harness terminal | Terminal stage | Calls | Prompt tokens | Generated tokens | Model seconds | Core | Regression | Error | Functionality |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | BLOCKED | REPOSITORY_INSPECTION | 10 | 57,098 | 340 | 29.275 | 0/1 | 0/0 | 4/4 | 0/27 |
| 2 | BLOCKED | REPOSITORY_INSPECTION | 13 | 77,654 | 557 | 30.355 | 0/1 | 4/32 | 0/0 | 0/17 |
| 3 | BLOCKED | REPOSITORY_INSPECTION | 19 | 83,367 | 1,225 | 42.801 | 0/1 | 4/50 | 0/0 | 1/17 |
| 4 | BLOCKED | PLAN | 13 | 236,772 | 1,230 | 113.716 | 0/1 | 5/68 | 0/0 | 0/20 |

The final combined result is 0/4 external checkpoints passed, 137 cumulative
regression failures, no accepted promotions, and no integrity hard failures.
Observed fault types were `MODEL_OUTPUT_INVALID` and `RETRY_EXHAUSTED`.

This is useful negative qualification evidence. The adapter correctly treats
`BLOCKED` as an agent outcome, preserves the last accepted workspace, permits
SCBench to evaluate it, and continues to the next checkpoint. Harness GREEN was
never established and is not conflated with external correctness.

## Observer verification

The existing observer on port 8769 remained the only listener. No adapter or
campaign process started another server. Playwright verified during the run
that:

- The observer discovered each checkpoint run beneath its configured recursive
  root and followed the newest active run.
- The activity view rendered prompt intent, model responses, requested native
  tools, host results, and telemetry as human-readable cards.
- The primary activity view was not raw JSON.
- The checkpoint-4 view visibly advanced from repository inspection to PLAN.
- The browser console reported zero errors.

## Qualification conclusion

M21 is qualified at the integration boundary: SCBench drives all four
checkpoints, external evaluation remains isolated, evidence exports and combined
results are reconstructable, and observer visibility works during live runs.
The local model did not complete a checkpoint. That outcome is retained as the
pre-baseline capability result and is not repaired by changing harness semantics
or prompts before the counted campaign.
