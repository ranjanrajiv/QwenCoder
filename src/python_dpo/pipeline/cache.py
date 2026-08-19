"""The stage cache: key construction, reuse decisions, and invalidation cascade
(spec 12 sections 17, 18, 19, 22).

Deliberately three small pure functions rather than a stateful cache object. Everything a
cache key needs -- input hashes, configuration hash, code version, model version -- is
already available to the orchestrator before it decides whether to run a stage, and
nothing here touches disk. The result is a cascade that is *derived*, never tabulated:
changing DPO beta changes only `dpo_training`'s config hash, which changes its adapter's
output hash, which changes `model_evaluation`'s input hash, and so on downstream --
`problem_dataset` and `candidate_generation` are structurally unreachable from that change
(spec section 91's test).

The git commit SHA is deliberately never an input to :func:`cache_key`. Including it would
invalidate every stage on every commit, including a documentation-only one, which would
defeat section 19's requirement that a training hyperparameter change leave problem and
candidate generation cached. It is still recorded in the experiment manifest (section 29);
`--force` and `--set` are the explicit invalidation levers instead.
"""

from __future__ import annotations

from .hashing import config_hash
from .manifest import StageManifest
from .stages import dependents_of, topological_order


def cache_key(
    *,
    stage: str,
    input_hashes: dict[str, str],
    configuration_hash: str,
    code_version: str,
    model_version: str,
) -> str:
    """The section 18 cache key: stage + input hashes + config hash + code + model version.

    ``json.dumps(..., sort_keys=True)`` inside :func:`config_hash` already makes key order
    irrelevant, both for ``input_hashes`` and for the payload itself.
    """
    payload = {
        "stage": stage,
        "input_hashes": input_hashes,
        "configuration_hash": configuration_hash,
        "code_version": code_version,
        "model_version": model_version,
    }
    return config_hash(payload)


def is_reusable(stage_manifest: StageManifest | None, current_key: str) -> bool:
    """Section 17: reuse only a *completed* stage whose recorded key matches exactly."""
    if stage_manifest is None:
        return False
    if stage_manifest.status != "COMPLETED":
        return False
    return stage_manifest.cache_key == current_key


def invalidate(stage: str, *, cascade: bool = True) -> tuple[str, ...]:
    """The ordered set of stages `--force <stage>` invalidates (spec section 22).

    With ``cascade=True`` (the default, and the only mode `--force` uses), every
    transitive dependent of ``stage`` is included -- forcing `candidate_evaluation`
    invalidates `preference_generation`, `dpo_training`, `model_evaluation`,
    `error_analysis` and `packaging` too, since none of their cached outputs are still
    valid once an upstream input changes. With ``cascade=False``, only ``stage`` itself is
    returned; the caller is responsible for knowing that leaves dependents stale.
    """
    targets = (stage, *dependents_of(stage)) if cascade else (stage,)
    return topological_order(names=targets)


__all__ = ["cache_key", "invalidate", "is_reusable"]
