"""Deterministic what-if simulator for AI Search Visibility (V6).

Answers: "Which missing evidence, if it existed, would move this place up the
ranking?" It edits a copy of the constraint matrix and re-runs the existing
deterministic ranker. No Gemini or external API calls are made, and the
original journey is never mutated.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional

from pydantic import BaseModel, Field

from ai_search_journey.models import (
    ConstraintEvaluationResult,
    ConstraintStatus,
    ConstraintSupport,
    RankedCandidate,
    ReferenceLocation,
)
from ai_search_journey.ranking import rank_candidates

SIMULATED_SOURCE_TYPE = "simulated"
DEFAULT_TOP_K = 3
MAX_EXHAUSTIVE_GAPS = 8


class SimulatorError(ValueError):
    """Raised when a simulation cannot be run for the given inputs."""


class EvidenceGap(BaseModel):
    """A constraint that is not SUPPORTED for the selected place."""

    constraint: str
    is_preference: bool
    current_status: ConstraintStatus
    fix_hint: str


class FlipResult(BaseModel):
    """Outcome of treating one or more gaps as SUPPORTED."""

    flipped_constraints: list[str] = Field(default_factory=list)
    score_before: float
    score_after: float
    rank_before: int
    rank_after: int
    enters_top_k: bool


class SimulationReport(BaseModel):
    """Full what-if report for one place in one journey."""

    place_id: str
    candidate_name: str
    top_k: int
    rank_before: int
    score_before: float
    total_candidates: int
    already_in_top_k: bool
    gaps: list[EvidenceGap] = Field(default_factory=list)
    single_flips: list[FlipResult] = Field(default_factory=list)
    minimal_plan: Optional[FlipResult] = None


def _is_category_constraint(name: str) -> bool:
    return name.startswith("Category:")


def _is_preference(name: str) -> bool:
    # Mirrors the preference detection used in ranking._evaluate_single_candidate.
    return "(preference)" in name.lower()


def _clean_name(name: str) -> str:
    return name.replace(" (Preference)", "").strip()


def build_fix_hint(constraint: str, status: ConstraintStatus) -> str:
    """Return a deterministic, human-readable suggestion for closing a gap."""
    lowered = constraint.lower()
    clean = _clean_name(constraint)
    if "open" in lowered or "hour" in lowered:
        action = (
            "Keep Google Business Profile opening hours complete and current, "
            "including late hours, so Places can confirm them."
        )
    elif "group" in lowered or "capacity" in lowered or "seat" in lowered:
        action = (
            "State group seating or capacity on your own website and Business Profile, "
            "and encourage reviews that mention it."
        )
    else:
        action = (
            f"Publish clear, crawlable content on your own domain about '{clean}', "
            "and encourage reviews that mention it."
        )
    if status == ConstraintStatus.NOT_SATISFIED:
        return f"Current evidence says this is not met. Only fix it if it becomes true: {action}"
    return action


def rank_all(
    evaluation: ConstraintEvaluationResult,
    reference_location: Optional[ReferenceLocation] = None,
) -> list[RankedCandidate]:
    """Rank every candidate (no top_n cut) with the production ranker."""
    return rank_candidates(evaluation, reference_location=reference_location)


def _find_ranked(ranked: list[RankedCandidate], place_id: str) -> RankedCandidate:
    for r in ranked:
        if r.candidate.place_id == place_id:
            return r
    raise SimulatorError(f"Place '{place_id}' is not in the ranked candidates.")


def find_evidence_gaps(
    evaluation: ConstraintEvaluationResult,
    place_id: str,
) -> list[EvidenceGap]:
    """List UNKNOWN and NOT_SATISFIED constraints for a place, skipping category eligibility."""
    for cand_eval in evaluation.evaluations:
        if cand_eval.candidate.place_id != place_id:
            continue
        gaps: list[EvidenceGap] = []
        for r in cand_eval.results:
            if _is_category_constraint(r.constraint):
                continue
            if r.status == ConstraintStatus.SUPPORTED:
                continue
            gaps.append(
                EvidenceGap(
                    constraint=r.constraint,
                    is_preference=_is_preference(r.constraint),
                    current_status=r.status,
                    fix_hint=build_fix_hint(r.constraint, r.status),
                )
            )
        return gaps
    raise SimulatorError(f"Place '{place_id}' is not in the constraint matrix.")


def apply_flips(
    evaluation: ConstraintEvaluationResult,
    place_id: str,
    constraints: list[str],
) -> ConstraintEvaluationResult:
    """Return a deep copy where the named constraints are SUPPORTED for one place."""
    edited = evaluation.model_copy(deep=True)
    targets = set(constraints)
    for cand_eval in edited.evaluations:
        if cand_eval.candidate.place_id != place_id:
            continue
        for r in cand_eval.results:
            if r.constraint in targets and not _is_category_constraint(r.constraint):
                r.status = ConstraintStatus.SUPPORTED
                r.supporting_evidence = [
                    ConstraintSupport(
                        source_type=SIMULATED_SOURCE_TYPE,
                        evidence_text="Simulated: assumes this evidence were published.",
                    )
                ]
                r.explanation = "Simulated what-if support (not observed)."
    return edited


def simulate_flips(
    evaluation: ConstraintEvaluationResult,
    place_id: str,
    constraints: list[str],
    *,
    reference_location: Optional[ReferenceLocation] = None,
    top_k: int = DEFAULT_TOP_K,
) -> FlipResult:
    """Re-rank after flipping the given constraints and compare against the baseline."""
    before = _find_ranked(rank_all(evaluation, reference_location), place_id)
    edited = apply_flips(evaluation, place_id, constraints)
    after = _find_ranked(rank_all(edited, reference_location), place_id)
    return FlipResult(
        flipped_constraints=list(constraints),
        score_before=before.score,
        score_after=after.score,
        rank_before=before.rank,
        rank_after=after.rank,
        enters_top_k=before.rank > top_k and after.rank <= top_k,
    )


def find_minimal_plan(
    evaluation: ConstraintEvaluationResult,
    place_id: str,
    gaps: list[EvidenceGap],
    *,
    reference_location: Optional[ReferenceLocation] = None,
    top_k: int = DEFAULT_TOP_K,
) -> Optional[FlipResult]:
    """Find the fewest flips that reach the top K, or None if no evidence fix can.

    Checks every subset by size (gap counts are small). Among plans of the same
    size, the one with the best resulting rank, then the highest score, wins.
    """
    names = [g.constraint for g in gaps][:MAX_EXHAUSTIVE_GAPS]
    for size in range(1, len(names) + 1):
        best: Optional[FlipResult] = None
        for combo in combinations(names, size):
            result = simulate_flips(
                evaluation,
                place_id,
                list(combo),
                reference_location=reference_location,
                top_k=top_k,
            )
            if result.rank_after > top_k:
                continue
            if best is None or (result.rank_after, -result.score_after) < (
                best.rank_after,
                -best.score_after,
            ):
                best = result
        if best is not None:
            return best
    return None


def build_simulation_report(
    evaluation: ConstraintEvaluationResult,
    place_id: str,
    *,
    reference_location: Optional[ReferenceLocation] = None,
    top_k: int = DEFAULT_TOP_K,
) -> SimulationReport:
    """Run the full what-if analysis for one place."""
    if top_k < 1:
        raise SimulatorError("top_k must be at least 1.")
    ranked = rank_all(evaluation, reference_location)
    baseline = _find_ranked(ranked, place_id)
    gaps = find_evidence_gaps(evaluation, place_id)

    single_flips = [
        simulate_flips(
            evaluation,
            place_id,
            [g.constraint],
            reference_location=reference_location,
            top_k=top_k,
        )
        for g in gaps
    ]
    single_flips.sort(key=lambda f: (f.rank_after, -f.score_after))

    already_in_top_k = baseline.rank <= top_k
    minimal_plan = None
    if not already_in_top_k:
        minimal_plan = find_minimal_plan(
            evaluation,
            place_id,
            gaps,
            reference_location=reference_location,
            top_k=top_k,
        )

    return SimulationReport(
        place_id=place_id,
        candidate_name=baseline.candidate.name,
        top_k=top_k,
        rank_before=baseline.rank,
        score_before=baseline.score,
        total_candidates=len(ranked),
        already_in_top_k=already_in_top_k,
        gaps=gaps,
        single_flips=single_flips,
        minimal_plan=minimal_plan,
    )
