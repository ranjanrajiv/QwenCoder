# src/python_dpo/models/

The inference seam. Everything downstream programs against `ModelClient`; nothing else
in the package imports `torch` or `transformers`.

Two rules govern this package:

1. **Importing it is cheap.** `torch` and `transformers` are imported inside the Qwen
   client's load method, never at module scope, so `import python_dpo` costs milliseconds
   rather than several GB (spec 03 §7). `tests/test_no_heavy_imports.py` asserts this in a
   subprocess, so the rule cannot rot silently.
2. **Hardware knowledge stops here.** Device and dtype resolution live in `qwen.py` alone;
   no other module knows whether a GPU exists (spec 03 §10).

## Files

### `base.py`

The abstraction, with no imports from the rest of the package — which is what lets
`config.py` depend on it without a cycle.

- `ModelClient` — a `runtime_checkable` `Protocol` with `generate(prompt,
  generation_config) -> RawGeneration` plus `name`, `revision`, and `provider`
  properties. Both clients match structurally; neither inherits from it.
- `GenerationConfig` — frozen dataclass of decoding parameters (`temperature`, `top_p`,
  `max_new_tokens`, `do_sample`, `repetition_penalty`, `seed`), range-checked in
  `__post_init__`. Rejects `bool` where a number is expected, since `True` is an `int` in
  Python and `max_new_tokens=True` would silently mean "one token". Its `to_dict()` is
  embedded verbatim in every candidate record.
- `ModelConfig` — which model to load and how: `provider`, `name`, `revision`, `device`,
  `dtype`, `trust_remote_code`, `quantization`. `trust_remote_code` defaults to `False`;
  `quantization` is **rejected** rather than ignored, because accepting a value that has
  no effect would mislead (spec 03 §9 reserves the field without asking for the logic).
- `RawGeneration` — text plus token counts and a finish reason. Deliberately not a
  candidate: a client never decides whether its own output is usable.
- `ModelError` / `ModelLoadError` / `InferenceError` — the exception hierarchy the
  generator catches. The split matters: `ModelLoadError` aborts the run (spec 03 §26.2),
  `InferenceError` is recorded and the run continues.

### `qwen.py`

`QwenModelClient` — Transformers-backed, lazily loaded.

- `_import_backend()` turns a missing optional extra into
  `ModelLoadError("... pip install -e '.[model]'")` rather than a bare `ImportError`.
- `resolve_device(requested, cuda_available)` — `auto` prefers CUDA and falls back to CPU
  with a warning; an explicit `cuda` request without CUDA **raises** instead of silently
  running a hundred times slower.
- `resolve_dtype(requested, device)` — `auto` gives bfloat16 on CUDA, float32 on CPU.
- `build_generation_kwargs(config)` — omits `temperature`/`top_p` for greedy decoding,
  which Transformers otherwise warns about.
- `_render(prompt)` applies the tokenizer's chat template when one exists. Qwen
  `-Instruct` checkpoints are trained with one and degrade noticeably without it; base
  checkpoints have none and get the raw prompt.
- `generate()` seeds per call, decodes, and **slices off the prompt tokens** so
  `raw_output` holds the completion alone.
- `_load_model_with_dtype` probes `dtype=` and falls back to `torch_dtype=`, spanning the
  Transformers releases that renamed the argument.

These four helpers are module-level functions precisely so they can be unit-tested
without a GPU, weights, or even torch installed.

### `mock.py`

`MockModelClient` — deterministic, offline, dependency-free. Lives in `src/` rather than
`tests/` so the CLI can drive it too, via `generate --mock-model`.

- **Synthesized mode (default).** Output is derived from a SHA-256 digest of the prompt:
  same prompt in, same bytes out, and a different strategy produces a different prompt and
  therefore different output. `hashlib` rather than `hash()`, because string hashing is
  randomized per process and would break the determinism this class exists to provide.
  The wrapper format cycles through all four shapes the extractor handles, so an offline
  run exercises markdown fences, generic fences, prose preambles, and bare code.
- **Scripted mode.** `MockModelClient(script=[...])` returns each item in order and
  *raises* any item that is an exception. This is how the tests inject an empty response,
  prose with no code, a syntax error, or an inference failure at one chosen index.
- `call_count` exposes how many times the model was actually asked, which is what proves
  resume skipped work rather than merely discarding it.
