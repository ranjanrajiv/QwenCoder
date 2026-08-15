# scripts/

Empty placeholder directory, required by the spec's directory structure (spec §4).
Currently contains only `.gitkeep` so git tracks the directory even though it has no
real content yet.

## Files

### `.gitkeep`

Zero-byte marker file. Git doesn't track empty directories, so this file exists purely
to keep `scripts/` present in the repository until real scripts are added.

## Intended future use

Later pipeline steps are expected to add standalone operational scripts here — for
example, dataset download/preparation helpers, one-off maintenance tasks, or
orchestration scripts that sit outside the `python_dpo` package's public CLI. Nothing
in this directory is implemented yet; per `CLAUDE.md`'s Scope Control rule, it stays
empty until a later step explicitly calls for it.
