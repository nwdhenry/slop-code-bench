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
- Problem set: `ef6a9dd13911566b6b01075ca121758c9f7b5c5f`
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
| M18 Freeze and integration scaffold | In progress | Pending | revisions, Python 3.12 environment, configs, reproduction instructions, both baseline suites |
| M19 Native SCBench adapter | Pending | Pending | registration, construction, lifecycle, usage, COMPLETE/BLOCKED, self-contained export |
| M20 Isolation and leakage | Pending | Pending | external-root enforcement, byte-exact tasks, source manifests, forbidden-file detection |
| M21 End-to-end qualification | Pending | Pending | deterministic four-checkpoint fixture, then one natural local-Qwen `file_backup` qualification |
| M22 Campaign runner | Pending | Pending | unattended repetition, seed schedule, failure continuation, traceable fingerprints |
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
