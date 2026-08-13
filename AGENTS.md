# Error Atlas Agent Guide

This file applies to the entire repository. Topic-specific state and evidence
belong in `NEXT_SESSION.md`, `TOPICS.md`, and the relevant topic or experiment
README, not in this file.

## Start every task by orienting

Before changing code, experiments, artifacts, or conclusions:

1. Inspect the Git worktree. Preserve existing and unrelated user changes.
2. Classify the task as explanation/review, research core, scaffolding,
   experiment execution, or artifact publication.
3. For research-core, experiment-execution, and artifact-publication tasks,
   read `NEXT_SESSION.md`, `TOPICS.md`,
   `framework/error_analysis_protocol.md`, and
   `framework/implementation_learning_protocol.md`.
4. For scaffolding, environment, CI, and repository-maintenance tasks, read
   only the repository guidance and files relevant to the requested change.
5. Before changing a topic, read its topic README. Before changing experiment
   code or artifacts, also read the relevant experiment README and
   artifact-status README.

Do not promote a hypothesis discussed in chat to repository fact unless it has
been explicitly frozen, implemented, or supported by recorded evidence.

## Research discipline

Use the repository research loop:

```text
parse -> shrink -> compute -> conjecture -> prove or break
      -> boundary audit -> write
```

For each research case, make the following explicit:

- target;
- reference;
- metric;
- assumptions and applicability domain;
- error sources;
- exact identity, when one exists;
- propagation, bound, and tightness;
- control mechanism and cost;
- validation method and evidence level.

Before running an experiment, record at least:

- **Direction**: expected sign or qualitative movement;
- **Scale**: expected order or magnitude;
- **Boundary**: where assumptions may fail;
- **Failure signature**: what would falsify the prediction or contract.

Prefer the smallest exactly inspectable example before broad batches. A
numerical observation is not a proof, absence of observed failure is not a
guarantee, and a valid bound is not automatically tight.

## Implementation ownership

Research-core logic is learner-owned by default. The learner should explain the
goal and invariant, predict behavior, write pseudocode, and implement the first
version. The agent should use the lowest sufficient hint level: question,
invariant, small example, skeleton, then local pseudocode.

The agent may directly provide scaffolding such as signatures, CLI plumbing,
test harnesses, CSV/JSON writers, provenance checks, and unfamiliar-API
examples. Before rewriting learner-owned core code, review the existing version
and explain the correctness, assumption, stability, or evidence issue.

Preserve an original or committed implementation before a closed-book rewrite.

## Separate facts, metrics, and decisions

Do not collapse distinct evidence layers:

```text
case recipe -> materialized input -> raw observation -> policy-free summary
            -> consumer policy -> assessment
```

Keep the following distinctions explicit:

- source value versus stored value;
- mathematical reference versus correctly rounded target;
- predictor output versus candidate observation;
- prediction conformance versus candidate correctness;
- accuracy versus repeatability;
- deterministic execution versus numerical correctness;
- exact oracle versus cheap screening score;
- consumer tolerance versus correct-rounding policy;
- CPU prototype evidence versus GPU implementation evidence.

Probability mass summing to one is not sufficient evidence of probability
accuracy. A predictor may correctly predict an inaccurate candidate result.

## Evidence and artifact rules

Experiment artifacts under `topics/**/experiments/results/` are versioned
evidence, not disposable build output.

- CSV preserves inspectable observations or derived tables.
- JSON preserves schema, identities, configuration, environment, provenance,
  and hashes.
- PNG may preserve a directly inspectable primary finding.
- Artifact READMEs must distinguish accepted, provisional, calibration,
  unexecuted, superseded, and negative-result evidence.

Never rewrite a frozen preregistration after seeing results. Do not delete raw
observations merely because later interpretation changes; update the evidence
status separately. Do not generalize a single preregistered case to a family or
a constructed counterexample to population performance.

Before executing an artifact-producing runner:

1. inspect its default output path and overwrite behavior;
2. prefer a fresh scratch output directory;
3. preserve checked-in snapshots until the replacement has been reviewed;
4. do not routinely rerun one-shot runners that intentionally refuse to
   overwrite evidence;
5. validate schema, row counts, source hashes, and artifact hashes.

CSV writers should reject undeclared fields instead of silently dropping them.
JSON provenance should use stable serialization and reject non-finite values.
References must be exact or have an explicitly checked applicability domain.

Do not put secrets, personal paths, or secret environment-variable values in
source files, logs, metadata, or artifacts. Prefer repository-relative source
names and `pathlib.Path`.

## Repository layout and content

- Keep research source and its tests easy to locate together.
- Place generated evidence in an experiment-specific results subdirectory.
- Put temporary outputs outside checked-in results unless replacement is an
  explicit task.
- Preserve UTF-8 text.
- Use `$...$` for inline Markdown math and `$$...$$` for display math so local
  previewers and GitHub render consistently.
- Update `NEXT_SESSION.md` only with the current verified milestone, evidence
  boundaries, and concrete next decision.
- Update `TOPICS.md` when a topic's actual research status changes.

## Environment and commands

Run commands from the repository root. The supported baseline is Python 3.10+
with dependencies from `requirements.txt`.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run Taylor tests:

```bash
python -m unittest discover \
  -s topics/taylor-expansion/experiments \
  -p "test_*.py" -v
```

Run Softmax tests:

```bash
python -m unittest discover \
  -s topics/softmax/experiments \
  -p "test_*.py" -v
```

Do not encode a fixed expected test count; the suite is expected to grow.
Specific experiment commands and warnings belong in the experiment README.

Keep normal tests offline and platform-neutral. Use temporary directories for
test artifacts. Case, input, configuration, and policy identities should be
stable across platforms; environment identity and generated timestamps are
expected to change. Do not require Windows and Linux metadata to be bytewise
identical.

## Validation after modifications

Apply validation in proportion to the change:

- documentation only: inspect rendered math structure where relevant and run
  `git diff --check`;
- topic code or tests: run that topic's complete unittest suite;
- shared identity, reference, schema, or oracle code: run the complete affected
  suite plus targeted boundary tests;
- artifact-generating code: regenerate only in scratch first, then verify
  source/artifact hashes, schema, row counts, and evidence status;
- experiment conclusion: compare the recorded prediction with observation and
  audit the claim boundary before updating formal notes.

Tests passing does not by itself establish a research conclusion. Report which
commands ran and distinguish new validation from historical evidence.

## Git, external actions, and safety

- Do not commit, push, open a PR, deploy, or write to external services unless
  the user explicitly asks.
- Do not use destructive Git or filesystem operations to clean a worktree.
- Do not overwrite unrelated user changes.
- Keep machine-local Codex state, credentials, unreviewed personal skills,
  virtual environments, caches, and other machine-local files out of version
  control.
- Repository-scoped skills may be versioned under `.agents/skills/` after
  review. They must be platform-neutral and contain no secrets or local
  absolute paths.
- Keep `.codex/` out of version control unless a portable repository
  configuration is explicitly approved.
- If required authority, data, hardware, or a material research choice is
  missing, stop and report the exact blocker instead of silently broadening the
  task.

GPU claims require evidence from the target hardware and execution contract.
Do not infer an unknown library reduction graph, workspace, occupancy, or
kernel behavior from a CPU model or a fused-kernel label alone.

## Completion report

At handoff, state:

- what changed;
- what was verified and by which command;
- what is a decision or recommendation rather than verified fact;
- artifact and evidence status;
- remaining assumptions, blockers, and the next single research rung.
