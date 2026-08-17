# The Stage 6 evaluation sandbox image: the Stage 5 python:3.12-slim base plus a pinned
# pytest, and nothing else (spec 06 sections 33-35).
#
# The container has no network access at evaluation time (spec section 36), so pytest
# must already be present at build time — this is why a dedicated image exists rather
# than installing at run time. Everything else about the sandbox — non-root user, no
# host filesystem, dropped capabilities, resource limits — is unchanged from Stage 5 and
# lives in src/python_dpo/sandbox/, not here.
#
# Build:
#   docker build -t python-dpo-evaluator:1.0 docker/evaluator/
#
# The candidate and generated test files are never baked into this image — they arrive
# through the isolated per-job workspace mounted at /workspace (spec section 35).

FROM python:3.12-slim

# Pinned so evaluation results are reproducible against a known pytest version (spec
# section 74) — never the host's pytest, and never whatever the latest release happens
# to be at build time.
RUN pip install --no-cache-dir --root-user-action=ignore pytest==8.3.4
