# .claude/specs/

Implementation specifications for this project, tracked in git despite the rest of
`.claude/` being ignored (see `.gitignore`'s `.claude/*` + `!.claude/specs/` rule).
The root-level [`specs/README.md`](../../specs/README.md) points here for anyone
who lands in the placeholder `specs/` directory expected by the spec's file tree.

## Files

### `01_project_skeleton.md`

The full Step 1 specification: a 12-step build of a DPO preference-data generation
pipeline for fine-tuning a Qwen Coder model on Python tasks, of which this document
covers only step 1 — the project foundation (packaging, CLI, logging, config,
tests). It defines the required directory structure, dependency constraints (PyYAML
only, no ML/training libraries yet), CLI/logging/config requirements, testing
requirements, `CLAUDE.md` content requirements, an explicit list of what must **not**
be implemented yet, and the acceptance criteria and verification commands used to
confirm Step 1 is complete. This is the document [`.claude/plans/01_project_skeleton_plan.md`](../plans/01_project_skeleton_plan.md)
was written against, and everything under `src/`, `tests/`, `data/`, `config.yaml`,
and `CLAUDE.md` was built to satisfy it.
