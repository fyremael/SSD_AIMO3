# AUTOMATION.md
## Documentation Automation

This repo includes lightweight automation to keep navigational docs from drifting.

## What is automated

`scripts/update_docs.py` generates:

- `docs/INDEX.md`
- `docs/STATUS.md`
- `docs/docs_summary.json`

Those generated files summarize:

- documentation pages
- script entry points
- config files
- workflow files
- high-level repo inventory counts

## Local refresh

Run this any time the repo surface changes:

```bash
python scripts/update_docs.py
```

## GitHub automation

The workflow at `.github/workflows/docs-sync.yml` refreshes generated docs:

- on pushes to `main`
- on manual `workflow_dispatch`
- on a weekly schedule

If the generated outputs changed, it commits:

```text
docs: refresh generated indexes
```

## Why this exists

The repo already has multiple operator paths:

- fixture ladder
- real-run plumbing
- Colab GPU deployment
- paired comparison

Without generated navigation and status pages, it becomes too easy for the docs to lag behind the codebase.

## Non-goals

This automation does not try to write experiment conclusions, overwrite hand-authored guides, or invent scientific claims. It only keeps the repo's structural documentation current.
