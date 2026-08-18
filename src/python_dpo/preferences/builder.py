"""Preference-pair construction (spec 08 sections 29-45).

For each problem (grouped strictly by ``problem_id``, spec section 37, so a cross-problem
pair is structurally impossible rather than merely checked for): every unordered candidate
pair ``C(n, 2)`` (spec section 29) is evaluated once. The direction is decided by
**Stage 7's own comparator** (spec section 44) — this module never re-implements scoring or
comparison logic, only what to do with the result. A tie or an indeterminate comparison
never produces a pair (spec sections 17, 38); identical code never produces a pair (spec
sections 33, 34); only one direction is ever emitted (spec sections 30, 71). Every excluded
combination is recorded as a :class:`~python_dpo.preferences.models.PreferenceRejection`
with a specific reason (spec section 77) — nothing is silently dropped.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from ..candidates.models import Candidate
from ..generation.prompt_builder import PROMPT_VERSION
from ..problems.models import Problem
from ..ranking.comparator import CandidateComparator
from ..ranking.models import CandidateAssessment, ComparisonResult
from .dedup import code_identical, dedupe_training_records
from .models import PreferencePair, PreferenceRejection
from .policies import DEFAULT_MINIMUM_SCORE_MARGIN, PreferencePolicy
from .prompt import PromptLineageError, verify_prompt_lineage

BUILDER_VERSION = "v1"

_UNCOMPARED = "UNCOMPARED"


@dataclass(frozen=True)
class ProblemBuildResult:
    pairs: list[PreferencePair]
    rejections: list[PreferenceRejection]


@dataclass(frozen=True)
class BuildResult:
    pairs: list[PreferencePair]
    rejections: list[PreferenceRejection]
    candidates_considered: int


class PreferencePairBuilder:
    """Builds preference pairs for one policy/margin/max-pairs configuration.

    Stateless across calls beyond the configured policy and limits — the same builder can
    be reused across problems and across an entire run.
    """

    def __init__(
        self,
        policy: PreferencePolicy,
        *,
        minimum_score_margin: float = DEFAULT_MINIMUM_SCORE_MARGIN,
        max_pairs_per_problem: int | None = None,
        comparator: CandidateComparator | None = None,
    ) -> None:
        self._policy = policy
        self._minimum_score_margin = minimum_score_margin
        self._max_pairs_per_problem = max_pairs_per_problem
        self._comparator = comparator or CandidateComparator()

    # ------------------------------------------------------------------------- per problem

    def build_problem(
        self,
        *,
        ranking_run_id: str,
        evaluation_run_id: str,
        candidate_run_id: str,
        problem: Problem,
        assessments: list[CandidateAssessment],
        candidates_by_id: dict[str, Candidate],
    ) -> ProblemBuildResult:
        """Build every pair for one problem's candidates (spec section 37).

        ``assessments`` covers all candidates evaluated for ``problem`` under one ranking
        run, including indeterminate ones (spec section 38 — dropped here, never assumed
        to score 0). ``candidates_by_id`` need not contain every assessed candidate; a
        join miss is an integrity failure (spec section 42), not a crash.
        """
        mismatched = [a for a in assessments if a.problem_id != problem.id]
        if mismatched:
            raise ValueError(
                f"build_problem({problem.id!r}) received assessment(s) for a different "
                f"problem: {sorted({a.problem_id for a in mismatched})}"
            )

        ordered = sorted(assessments, key=lambda a: a.candidate_id)
        pairs: list[PreferencePair] = []
        rejections: list[PreferenceRejection] = []

        # Decision 1: verify lineage once, over every candidate this problem can actually
        # join to a real Candidate record. A join miss never joins the lineage check and
        # never joins a pair; every pair touching it is an integrity_failure below.
        joined = [
            candidates_by_id[a.candidate_id] for a in ordered if a.candidate_id in candidates_by_id
        ]
        try:
            canonical_prompt, canonical_prompt_sha256 = verify_prompt_lineage(problem, joined)
        except PromptLineageError as exc:
            # A lineage failure invalidates every pair this problem could have produced —
            # never a silent fallback to an unverified prompt (decision 1).
            for a, b in itertools.combinations(ordered, 2):
                rejections.append(
                    self._bare_rejection(ranking_run_id, problem.id, a, b, "integrity_failure", str(exc))
                )
            return ProblemBuildResult(pairs=pairs, rejections=rejections)

        for a, b in itertools.combinations(ordered, 2):
            outcome = self._evaluate_pair(
                ranking_run_id=ranking_run_id,
                evaluation_run_id=evaluation_run_id,
                candidate_run_id=candidate_run_id,
                problem=problem,
                a=a,
                b=b,
                candidates_by_id=candidates_by_id,
                canonical_prompt=canonical_prompt,
                canonical_prompt_sha256=canonical_prompt_sha256,
            )
            if isinstance(outcome, PreferencePair):
                pairs.append(outcome)
            else:
                rejections.append(outcome)

        kept, truncated = self._truncate(pairs)
        rejections.extend(truncated)
        # Decision 3's training-record dedup is applied per problem: prompts differ
        # across problems (each is a distinct canonical rendering), so a duplicate
        # (prompt, chosen, rejected) triple can only arise from candidates that share
        # code within the same problem. Scoping it here — rather than across the whole
        # run — is what keeps a problem's slice of metadata.jsonl appendable and
        # resumable independently of every other problem (spec section 83).
        deduped = dedupe_training_records(kept)
        return ProblemBuildResult(pairs=deduped, rejections=rejections)

    # ----------------------------------------------------------------------------- per run

    def build_run(
        self,
        *,
        ranking_run_id: str,
        evaluation_run_id: str,
        candidate_run_id: str,
        problems_by_id: dict[str, Problem],
        assessments_by_problem: dict[str, list[CandidateAssessment]],
        candidates_by_id: dict[str, Candidate],
    ) -> BuildResult:
        """Build every pair across every problem (spec section 72's training-record
        deduplication, decision 3, is applied per problem inside :meth:`build_problem`).
        """
        all_pairs: list[PreferencePair] = []
        all_rejections: list[PreferenceRejection] = []
        candidates_considered = 0

        for problem_id in sorted(assessments_by_problem):
            assessments = assessments_by_problem[problem_id]
            candidates_considered += len(assessments)
            problem = problems_by_id.get(problem_id)
            if problem is None:
                ordered = sorted(assessments, key=lambda a: a.candidate_id)
                for a, b in itertools.combinations(ordered, 2):
                    all_rejections.append(
                        self._bare_rejection(
                            ranking_run_id,
                            problem_id,
                            a,
                            b,
                            "integrity_failure",
                            f"unknown problem_id {problem_id!r}",
                        )
                    )
                continue

            result = self.build_problem(
                ranking_run_id=ranking_run_id,
                evaluation_run_id=evaluation_run_id,
                candidate_run_id=candidate_run_id,
                problem=problem,
                assessments=assessments,
                candidates_by_id=candidates_by_id,
            )
            all_pairs.extend(result.pairs)
            all_rejections.extend(result.rejections)

        return BuildResult(
            pairs=all_pairs, rejections=all_rejections, candidates_considered=candidates_considered
        )

    # --------------------------------------------------------------------------- internals

    def _evaluate_pair(
        self,
        *,
        ranking_run_id: str,
        evaluation_run_id: str,
        candidate_run_id: str,
        problem: Problem,
        a: CandidateAssessment,
        b: CandidateAssessment,
        candidates_by_id: dict[str, Candidate],
        canonical_prompt: str,
        canonical_prompt_sha256: str,
    ) -> PreferencePair | PreferenceRejection:
        comparison = self._comparator.compare(ranking_run_id, a, b)

        if comparison.relation in ("TIE", "INDETERMINATE"):
            reason = "tie" if comparison.relation == "TIE" else "indeterminate"
            detail = (
                "equal score; no objective preference (spec section 17)"
                if reason == "tie"
                else "one or both candidates are indeterminate (spec section 38)"
            )
            return self._comparison_rejection(problem.id, a, b, comparison, reason, detail)

        chosen_assessment, rejected_assessment = (
            (a, b) if comparison.relation == "A_BETTER" else (b, a)
        )

        chosen_candidate = candidates_by_id.get(chosen_assessment.candidate_id)
        rejected_candidate = candidates_by_id.get(rejected_assessment.candidate_id)
        if chosen_candidate is None or rejected_candidate is None:
            missing = (
                chosen_assessment.candidate_id
                if chosen_candidate is None
                else rejected_assessment.candidate_id
            )
            return self._comparison_rejection(
                problem.id,
                a,
                b,
                comparison,
                "integrity_failure",
                f"candidate {missing} has no matching record in the candidate run "
                "(spec section 42)",
            )

        if (
            chosen_candidate.code_sha256 is None
            or rejected_candidate.code_sha256 is None
            or chosen_candidate.prompt_sha256 is None
            or rejected_candidate.prompt_sha256 is None
        ):
            return self._comparison_rejection(
                problem.id,
                a,
                b,
                comparison,
                "integrity_failure",
                "a candidate lacks schema 2.0 provenance hashes (legacy schema_version 1.0)",
            )

        if code_identical(chosen_candidate, rejected_candidate):
            return self._comparison_rejection(
                problem.id,
                a,
                b,
                comparison,
                "identical_code",
                "chosen and rejected share identical code (spec sections 33, 34)",
            )

        # Decision 1's defensive re-statement of spec section 41. Structurally guaranteed
        # by grouping-by-problem upstream (spec section 37); this never fires on real
        # data and is retained as a guard against a future grouping bug, exactly like the
        # code-identity check above sometimes never fires on a given dataset.
        if chosen_candidate.problem_id != problem.id or rejected_candidate.problem_id != problem.id:
            return self._comparison_rejection(
                problem.id,
                a,
                b,
                comparison,
                "invalid_prompt_match",
                "chosen and rejected do not both belong to this problem",
            )

        admitted, reason = self._policy.admits(
            chosen_assessment,
            rejected_assessment,
            minimum_score_margin=self._minimum_score_margin,
        )
        if not admitted:
            assert reason is not None  # a policy always explains a rejection
            return self._comparison_rejection(
                problem.id, a, b, comparison, reason, f"rejected by policy {self._policy.name!r}"
            )

        return self._build_pair(
            ranking_run_id=ranking_run_id,
            evaluation_run_id=evaluation_run_id,
            candidate_run_id=candidate_run_id,
            problem=problem,
            chosen_assessment=chosen_assessment,
            rejected_assessment=rejected_assessment,
            chosen_candidate=chosen_candidate,
            rejected_candidate=rejected_candidate,
            canonical_prompt=canonical_prompt,
            canonical_prompt_sha256=canonical_prompt_sha256,
        )

    def _build_pair(
        self,
        *,
        ranking_run_id: str,
        evaluation_run_id: str,
        candidate_run_id: str,
        problem: Problem,
        chosen_assessment: CandidateAssessment,
        rejected_assessment: CandidateAssessment,
        chosen_candidate: Candidate,
        rejected_candidate: Candidate,
        canonical_prompt: str,
        canonical_prompt_sha256: str,
    ) -> PreferencePair:
        # Deterministic, never a UUID (spec section 31). Every candidate_id is already
        # problem-scoped (p001_c001), so repeating the problem id would be redundant.
        preference_id = f"pref_{chosen_assessment.candidate_id}__{rejected_assessment.candidate_id}"
        chosen_correctness = chosen_assessment.correctness
        rejected_correctness = rejected_assessment.correctness
        preference_strength = (
            "strong"
            if (chosen_correctness, rejected_correctness) == ("correct", "incorrect")
            else "medium"
        )
        return PreferencePair(
            preference_id=preference_id,
            problem_id=problem.id,
            candidate_run_id=candidate_run_id,
            ranking_run_id=ranking_run_id,
            evaluation_run_id=evaluation_run_id,
            chosen_candidate_id=chosen_assessment.candidate_id,
            rejected_candidate_id=rejected_assessment.candidate_id,
            prompt=canonical_prompt,
            chosen=chosen_candidate.code,
            rejected=rejected_candidate.code,
            chosen_score=chosen_assessment.score,
            rejected_score=rejected_assessment.score,
            score_margin=chosen_assessment.score - rejected_assessment.score,
            chosen_pass_rate=chosen_assessment.pass_rate,
            rejected_pass_rate=rejected_assessment.pass_rate,
            chosen_tests_passed=chosen_assessment.tests_passed,
            rejected_tests_passed=rejected_assessment.tests_passed,
            chosen_tests_total=chosen_assessment.tests_total,
            rejected_tests_total=rejected_assessment.tests_total,
            chosen_correctness=chosen_correctness,
            rejected_correctness=rejected_correctness,
            preference_strength=preference_strength,
            selection_policy=self._policy.name,
            selection_policy_version=self._policy.version,
            canonical_prompt_sha256=canonical_prompt_sha256,
            prompt_version=PROMPT_VERSION,
            chosen_generation_prompt_sha256=chosen_candidate.prompt_sha256,
            rejected_generation_prompt_sha256=rejected_candidate.prompt_sha256,
            chosen_strategy=chosen_candidate.strategy,
            rejected_strategy=rejected_candidate.strategy,
            chosen_code_sha256=chosen_candidate.code_sha256,
            rejected_code_sha256=rejected_candidate.code_sha256,
        )

    def _truncate(
        self, pairs: list[PreferencePair]
    ) -> tuple[list[PreferencePair], list[PreferenceRejection]]:
        """Deterministically keep at most ``max_pairs_per_problem`` pairs (spec sections
        57, 58) — sorted by descending score margin, then by candidate ids for a total
        order. No RNG anywhere in this module.
        """
        if self._max_pairs_per_problem is None or len(pairs) <= self._max_pairs_per_problem:
            return pairs, []
        ordered = sorted(
            pairs,
            key=lambda p: (-p.score_margin, p.chosen_candidate_id, p.rejected_candidate_id),
        )
        kept = ordered[: self._max_pairs_per_problem]
        dropped = ordered[self._max_pairs_per_problem :]
        detail = (
            f"exceeds max_pairs_per_problem={self._max_pairs_per_problem}; kept the "
            f"{self._max_pairs_per_problem} pair(s) with the largest score margin"
        )
        return kept, [self._pair_rejection(p, "max_pairs_per_problem", detail) for p in dropped]

    @staticmethod
    def _comparison_rejection(
        problem_id: str,
        a: CandidateAssessment,
        b: CandidateAssessment,
        comparison: ComparisonResult,
        reason: str,
        detail: str,
    ) -> PreferenceRejection:
        return PreferenceRejection(
            ranking_run_id=comparison.ranking_run_id,
            problem_id=problem_id,
            candidate_a=a.candidate_id,
            candidate_b=b.candidate_id,
            reason=reason,
            detail=detail,
            relation=comparison.relation,
            score_a=comparison.score_a,
            score_b=comparison.score_b,
            score_margin=comparison.score_margin,
        )

    @staticmethod
    def _bare_rejection(
        ranking_run_id: str,
        problem_id: str,
        a: CandidateAssessment,
        b: CandidateAssessment,
        reason: str,
        detail: str,
    ) -> PreferenceRejection:
        """A rejection recorded without ever calling the comparator (spec section 42's
        candidate/prompt integrity checks happen before comparison is meaningful).
        """
        return PreferenceRejection(
            ranking_run_id=ranking_run_id,
            problem_id=problem_id,
            candidate_a=a.candidate_id,
            candidate_b=b.candidate_id,
            reason=reason,
            detail=detail,
            relation=_UNCOMPARED,
            score_a=a.score,
            score_b=b.score,
            score_margin=abs(a.score - b.score),
        )

    @staticmethod
    def _pair_rejection(pair: PreferencePair, reason: str, detail: str) -> PreferenceRejection:
        """A rejection derived from an already-built pair (spec section 57 truncation) —
        ``candidate_a``/``candidate_b`` restored to candidate-id sort order to match every
        other rejection's convention.
        """
        if pair.chosen_candidate_id < pair.rejected_candidate_id:
            candidate_a, candidate_b = pair.chosen_candidate_id, pair.rejected_candidate_id
            relation, score_a, score_b = "A_BETTER", pair.chosen_score, pair.rejected_score
        else:
            candidate_a, candidate_b = pair.rejected_candidate_id, pair.chosen_candidate_id
            relation, score_a, score_b = "B_BETTER", pair.rejected_score, pair.chosen_score
        return PreferenceRejection(
            ranking_run_id=pair.ranking_run_id,
            problem_id=pair.problem_id,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            reason=reason,
            detail=detail,
            relation=relation,
            score_a=score_a,
            score_b=score_b,
            score_margin=pair.score_margin,
        )


__all__ = ["BUILDER_VERSION", "BuildResult", "PreferencePairBuilder", "ProblemBuildResult"]
