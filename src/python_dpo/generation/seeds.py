"""Per-candidate sampling seeds (spec 03 sections 11, 30).

``config.yaml`` carries a single ``generation.seed``. Feeding that one value to every
``generate()`` call reseeds the sampler identically before each candidate, so a problem's
candidates come back byte-identical no matter how many are requested -- sampling with
``do_sample: true`` and ``temperature: 0.8`` decides nothing, because the random draws are
replayed from the same state every time. That defeats the entire point of generating
several candidates per problem: preference-pair construction needs *variation* to have
anything to prefer between.

:func:`compute_candidate_seed` spreads one configured base seed across candidates, so the
run as a whole stays reproducible from that base seed while each candidate samples its own
trajectory.

The schedule is derived from ``(base_seed, problem_id, generation_index)`` by hashing
rather than from the problem's ordinal position -- deliberately different from
:func:`python_dpo.model_evaluation.generation.compute_seed`, which uses
``problem_index * 1000 + sample_index``. Stage 10 pairs two model variants walking the
*same sorted benchmark*, so an ordinal is stable there and pairing is the whole point.
Stage 3 addresses candidates by ``problem_id`` and supports resuming a run over an
arbitrary subset of problems (``--problem-ids``), where an ordinal would shift underneath
a resumed run and silently reseed the candidates it had yet to generate. Hashing the
``problem_id`` keeps a given candidate's seed the same no matter which problems accompany
it in the run.
"""

from __future__ import annotations

from ..candidates.hashing import sha256_text

# Keep seeds inside the non-negative signed 32-bit range: `transformers.set_seed` feeds
# numpy, whose legacy seeding rejects anything wider.
_SEED_MODULUS = 2**31 - 1


def compute_candidate_seed(base_seed: int, problem_id: str, generation_index: int) -> int:
    """The sampling seed for one ``(problem_id, generation_index)`` candidate.

    Deterministic in all three inputs: the same base seed, problem and index always
    produce the same sampling seed, so a run remains reproducible from ``config.yaml``
    alone. Distinct indices produce unrelated seeds, so a problem's candidates explore
    different sampling trajectories instead of repeating one.
    """
    digest = sha256_text(f"{base_seed}:{problem_id}:{generation_index}")
    return int(digest[:16], 16) % _SEED_MODULUS


__all__ = ["compute_candidate_seed"]
