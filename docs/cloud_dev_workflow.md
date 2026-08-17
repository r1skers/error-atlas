# Cloud Development Workflow

This note documents the lightweight remote-development setup used for routine repository work. It is operational guidance only and does not define research claims, evidence, or experiment policy.

## Roles

- **VPS**: persistent Linux execution environment for editing, tests, and lightweight CPU experiments.
- **GitHub**: source of truth for versioned repository history and collaboration branches.
- **Local devices**: SSH, browser, or editor clients used to access the VPS.

## Routine loop

From the repository root on the VPS:

```bash
git status
git pull
```

Make or review changes, then run the validation appropriate to the files touched. For Softmax experiment code, the repository-wide test command is:

```bash
python -m unittest discover \
  -s topics/softmax/experiments \
  -p "test_*.py" -v
```

Before publishing changes, inspect the diff and repository state:

```bash
git diff
git status
```

Commit and push only after the change has been reviewed locally.

## Boundaries

- Do not treat the VPS working tree as the only copy of important work; push reviewed milestones to GitHub.
- Do not place credentials, tokens, private keys, or machine-local secrets in the repository.
- Do not use this note to override research protocols, preregistrations, artifact status, or topic-specific evidence rules.
