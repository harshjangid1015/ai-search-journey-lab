"""Tests for the V6 deterministic what-if simulator."""

import pytest

from ai_search_journey.models import (
    Candidate,
    CandidateConstraintEvaluation,
    ConstraintEvaluationResult,
    ConstraintResult,
    ConstraintStatus,
    ConstraintSupport,
)
from ai_search_journey.ranking import HARD_SUPPORTED_WEIGHT, PREFERENCE_SUPPORTED_WEIGHT
from ai_search_journey.visibility.simulator import (
    SimulatorError,
    apply_flips,
    build_simulation_report,
    find_evidence_gaps,
    simulate_flips,
)

OPEN = "Open after 20:00"
GROUP = "Group size 6"
QUIET = "quiet (Preference)"


def _result(name: str, status: ConstraintStatus) -> ConstraintResult:
    support = (
        [ConstraintSupport(source_type="google_places")]
        if status == ConstraintStatus.SUPPORTED
        else []
    )
    return ConstraintResult(constraint=name, status=status, supporting_evidence=support)


def _eval(place_id: str, statuses: dict[str, ConstraintStatus]) -> CandidateConstraintEvaluation:
    return CandidateConstraintEvaluation(
        candidate=Candidate(place_id=place_id, name=f"Place {place_id}"),
        results=[_result(n, s) for n, s in statuses.items()],
    )


S = ConstraintStatus.SUPPORTED
U = ConstraintStatus.UNKNOWN
N = ConstraintStatus.NOT_SATISFIED


@pytest.fixture
def matrix() -> ConstraintEvaluationResult:
    """Four places; 'target' starts last with two gaps."""
    return ConstraintEvaluationResult(
        evaluations=[
            _eval("a", {OPEN: S, GROUP: S, QUIET: U}),
            _eval("b", {OPEN: S, GROUP: U, QUIET: S}),
            _eval("c", {OPEN: S, GROUP: U, QUIET: U}),
            _eval("target", {OPEN: S, GROUP: U, QUIET: U, "Category: coffee shop": U}),
        ]
    )


def test_gaps_skip_supported_and_category(matrix: ConstraintEvaluationResult) -> None:
    gaps = find_evidence_gaps(matrix, "target")
    assert [g.constraint for g in gaps] == [GROUP, QUIET]
    assert gaps[0].is_preference is False
    assert gaps[1].is_preference is True


def test_hard_flip_adds_hard_weight(matrix: ConstraintEvaluationResult) -> None:
    result = simulate_flips(matrix, "target", [GROUP])
    assert result.score_after - result.score_before == pytest.approx(HARD_SUPPORTED_WEIGHT)
    assert result.rank_after < result.rank_before


def test_preference_flip_adds_preference_weight(matrix: ConstraintEvaluationResult) -> None:
    result = simulate_flips(matrix, "target", [QUIET])
    assert result.score_after - result.score_before == pytest.approx(
        PREFERENCE_SUPPORTED_WEIGHT
    )


def test_original_matrix_is_not_mutated(matrix: ConstraintEvaluationResult) -> None:
    before = matrix.model_dump()
    apply_flips(matrix, "target", [GROUP, QUIET])
    simulate_flips(matrix, "target", [GROUP, QUIET])
    assert matrix.model_dump() == before


def test_category_constraint_is_never_flipped(matrix: ConstraintEvaluationResult) -> None:
    edited = apply_flips(matrix, "target", ["Category: coffee shop"])
    target = next(e for e in edited.evaluations if e.candidate.place_id == "target")
    category = next(r for r in target.results if r.constraint.startswith("Category:"))
    assert category.status == ConstraintStatus.UNKNOWN


def test_report_finds_minimal_plan(matrix: ConstraintEvaluationResult) -> None:
    report = build_simulation_report(matrix, "target", top_k=3)
    assert report.rank_before == 4
    assert report.already_in_top_k is False
    assert report.minimal_plan is not None
    assert report.minimal_plan.flipped_constraints == [GROUP]
    assert report.minimal_plan.rank_after <= 3


def test_report_when_no_fix_can_reach_top_k() -> None:
    matrix = ConstraintEvaluationResult(
        evaluations=[
            _eval("a", {OPEN: S, GROUP: S}),
            _eval("b", {OPEN: S, GROUP: S}),
            _eval("target", {OPEN: N, GROUP: S}),
        ]
    )
    report = build_simulation_report(matrix, "target", top_k=1)
    assert report.minimal_plan is None


def test_unknown_place_raises(matrix: ConstraintEvaluationResult) -> None:
    with pytest.raises(SimulatorError):
        build_simulation_report(matrix, "missing")
