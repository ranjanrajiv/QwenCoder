# .claude/plans/

Approved implementation plans, tracked in git despite the rest of `.claude/` being
ignored (see `.gitignore`'s `.claude/*` + `!.claude/plans/` rule).

## Files

### `01_project_skeleton_plan.md`

The concrete implementation plan for Step 1, derived from
[`.claude/specs/01_project_skeleton.md`](../specs/01_project_skeleton.md) and
confirmed with the user before implementation started. It records the three
decisions made beyond what the spec itself dictates (skeleton at the repo root
rather than a nested `python-dpo/` directory; PyYAML as the sole runtime
dependency; `.claude/specs/` and `.claude/plans/` becoming tracked while the rest of
`.claude/` stays ignored), an exact file-by-file list of what to create and why,
the verification commands to run afterward, and the deviations from the spec's
literal file tree to flag in the final report (`__main__.py` and `config.py`
weren't in the spec's tree but are required for `python -m python_dpo` and the
config abstraction it calls for). This plan has been fully executed — see the root
[`README.md`](../../README.md) for current project status.

### `02_problem_dataset_plan.md`

The implementation plan for Stage 2 — the problem dataset and ground-truth layer —
derived from [`.claude/specs/02_problem_dataset.md`](../specs/02_problem_dataset.md)
and confirmed with the user before implementation started. It covers the new
`src/python_dpo/problems/` subpackage (schema, catalog, reference solutions,
JSONL storage, a swappable `ReferenceExecutor`, and dataset validation), the
`problems build` / `problems validate` CLI commands, and the unit plus integration
test suites. It pins down the three approved design decisions (frozen dataclasses
rather than Pydantic, reference solutions authored as real Python functions with
their JSONL text derived via `inspect.getsource()`, and the validation summary
printed to stdout from the CLI layer), plus the semantics chosen for each of the ten
problems where the spec required an explicit ruling on ties, ordering, and
invalid-input behavior. **Approved but not yet implemented.**
