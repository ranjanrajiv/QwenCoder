# scripts/

Standalone operational scripts that sit outside the `python_dpo` package's CLI — things
that are run deliberately by a human rather than as part of the pipeline.

## Files

### `smoke_real_model.sh`

The manual real-model smoke test (spec 03 §34, §48; spec 04 §55). Loads the Qwen model
configured in `config.yaml`, generates **one** candidate for **one** problem into its own
new run directory, extracts the Python, validates its syntax, persists it, and prints the
run's manifest, the candidate's code, and a `runs validate` integrity check.

```bash
scripts/smoke_real_model.sh          # defaults to p001
scripts/smoke_real_model.sh p004
```

This is **not** part of the automated test suite and must never run in CI: it downloads
several GB of weights on first use and wants a GPU to finish quickly. The pytest suite is
offline and CPU-only by design, so real-model inference has to be requested explicitly —
that is the whole point of keeping it here as a script rather than a test.

Prerequisites:

```bash
pip install -e '.[model]'          # the optional inference backend
python -m python_dpo problems build # data/problems/problems.jsonl must exist
```

For gated or private models, export `HF_TOKEN` in your shell before running. Never write
its value into `config.yaml`, source code, or any committed file.

### `.gitkeep`

Zero-byte marker kept from Stage 1, when this directory was still empty. Harmless now
that real scripts live here.
