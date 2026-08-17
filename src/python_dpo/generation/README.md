# src/python_dpo/generation/

Turns problems into candidate programs:

```
Problem → PromptBuilder → ModelClient → RawGeneration → CodeExtractor
        → Candidate → CandidateRepository → candidates.jsonl
```

The governing rule (spec 03 §3, §50): this package answers **"what code did the model
generate?"** It never decides whether that code is correct. Correctness is established
later by executing candidates against the problem's tests.

## Files

### `strategies.py`

The five strategies from spec 03 §13 — `normal`, `straightforward`, `edge_case_focused`,
`alternative`, `optimized` — and their instruction text. Five different instructions exist
to make five candidates for one problem *differ*; identical candidates carry no preference
signal.

They are **generation prompts, not correctness labels**. Nothing claims the `optimized`
candidate is actually faster.

`resolve_strategies(configured, count, override)` returns exactly `count` names: one per
strategy at the default five, cycling beyond that, or all-one-strategy when `--strategy`
is passed.

### `prompt_builder.py`

`build_prompt(problem, strategy)` — pure, so the same problem and strategy always produce
byte-identical text. No timestamps, no randomness.

`PROMPT_VERSION` is stamped on every candidate. **Changing `_TEMPLATE` — whitespace
included — changes what the model sees and therefore what the dataset means, so the
version must be bumped with it** (spec 03 §15). Otherwise two datasets generated months
apart would be silently incomparable.

### `code_extractor.py`

`extract_code(raw_output) -> ExtractionResult`, tried in order:

| `source_format` | Matched by |
|---|---|
| `python_fence` | A ` ```python ` / ` ```py ` fence |
| `generic_fence` | A bare ` ``` ` fence whose body has a `def`/`class`/`import` line |
| `plain` | No fences at all, but the text has such a line |
| `unknown` | Nothing matched → `extracted=False`, `error="No Python code detected"` |

**This module never repairs code** (spec 03 §18, §44). It strips surrounding whitespace
and returns the bytes untouched. An unterminated fence fails extraction rather than being
guessed at: patching it would mean the stored candidate is no longer what the model
produced, quietly corrupting the preference experiment.

An extraction failure produces **no candidate** — the generator records a
`code_extraction` failure instead. There is nothing to store.

### `validation.py`

Static checks only. **Nothing here executes the candidate.** `ast.parse` builds a syntax
tree; it does not import, evaluate, or run anything, which is why this module is permitted
to touch untrusted code while nothing else in the package is (CLAUDE.md, Security).

- `check_syntax(code) -> SyntaxCheck` — wraps `ast.parse`, capturing the message and line
  number. Also catches `ValueError` for source containing null bytes, which is not a
  `SyntaxError`.
- `check_function_name(code, entry_point) -> bool` — walks the AST for a `FunctionDef` or
  `AsyncFunctionDef` of that name, reusing Stage 2's `Problem.entry_point` rather than
  re-parsing the signature string. `async def` counts (p010 is async) and so do nested
  definitions. Unparseable code returns `False` rather than raising.

Both establish *structure*, never correctness: `def factorial(n): return 123` passes both.

### `generator.py`

`CandidateGenerator` — the orchestrator. Depends on `ModelClient`, so the same code drives
the real model and the mock.

Failure policy, which is where the spec's two readings had to be reconciled
(§19.1, §26, §26.1, §26.2):

| Outcome | Recorded as |
|---|---|
| empty response | `GenerationFailure(empty_output)`, no candidate |
| no extractable code | `GenerationFailure(code_extraction)`, no candidate |
| inference exception | `GenerationFailure(inference)`, no candidate, run continues |
| **code that won't parse** | **`Candidate` with `syntax_valid=false`, no failure record** |
| model won't load | `GenerationFailure(model_load)`, then the run aborts |

The fourth row is the important one: unparseable output is the model's actual work, and
it is exactly what a later stage needs on the rejected side of a preference pair.
Discarding it would bias the dataset toward the model's good days.

Duplicate detection (§41) compares extracted code exactly, scoped per problem, across all
runs — so a resumed run still notices it reproduced an earlier candidate. A regenerated
candidate matching *its own* previous run is not flagged: that is the generation
reproducing itself, not two candidates colliding. Duplicates are always kept.

Both indexes (resume keys and seen code) are loaded once per run and maintained in memory;
re-reading the file per candidate would be quadratic.
