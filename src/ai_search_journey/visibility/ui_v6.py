"""Streamlit tab for the V6 Visibility What-If Simulator."""

from __future__ import annotations

from typing import Optional

import streamlit as st

from ai_search_journey.models import JourneyResult
from ai_search_journey.visibility.simulator import (
    DEFAULT_TOP_K,
    SimulatorError,
    build_simulation_report,
    rank_all,
    simulate_flips,
)


def _format_move(rank_before: int, rank_after: int) -> str:
    delta = rank_before - rank_after
    if delta > 0:
        return f"▲ +{delta}"
    if delta < 0:
        return f"▼ {delta}"
    return "● Unchanged"


def render_what_if_tab(journey: Optional[JourneyResult]) -> None:
    """Render the What-If Simulator for a completed journey."""
    st.subheader("🧪 Visibility What-If Simulator [V6]")
    st.caption(
        "Pick a place and see which missing evidence, if it existed, would move it up. "
        "Results are simulated with the same deterministic ranker. They are not observed."
    )

    if journey is None or journey.constraint_result is None:
        st.info("Run a search journey first. The simulator uses its constraint matrix.")
        return

    evaluation = journey.constraint_result
    ref_loc = journey.reference_location
    ranked = rank_all(evaluation, ref_loc)
    if not ranked:
        st.info("This journey has no ranked candidates to simulate.")
        return

    labels = {f"#{r.rank} {r.candidate.name}": r.candidate.place_id for r in ranked}
    default_index = min(DEFAULT_TOP_K, len(labels) - 1)
    choice = st.selectbox(
        "Place to analyze", list(labels.keys()), index=default_index, key="v6_place"
    )
    top_k = int(
        st.number_input(
            "Target: top K", min_value=1, max_value=10, value=DEFAULT_TOP_K, key="v6_top_k"
        )
    )
    place_id = labels[choice]

    try:
        report = build_simulation_report(
            evaluation, place_id, reference_location=ref_loc, top_k=top_k
        )
    except SimulatorError as e:
        st.error(str(e))
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Current rank", f"#{report.rank_before} of {report.total_candidates}")
    col2.metric("Current score", f"{report.score_before:.2f}")
    col3.metric("Evidence gaps", len(report.gaps))

    if not report.gaps:
        st.success("Every constraint is already SUPPORTED for this place.")
        return

    st.markdown("#### Evidence gaps, best single fix first")
    gap_by_name = {g.constraint: g for g in report.gaps}
    st.dataframe(
        [
            {
                "Constraint": f.flipped_constraints[0],
                "Status now": gap_by_name[f.flipped_constraints[0]].current_status.value,
                "Score change": round(f.score_after - f.score_before, 2),
                "Rank after": f"#{f.rank_after}",
                "Move": _format_move(f.rank_before, f.rank_after),
                "How to fix": gap_by_name[f.flipped_constraints[0]].fix_hint,
            }
            for f in report.single_flips
        ],
        hide_index=True,
        width="stretch",
    )

    st.markdown(f"#### What it takes to reach the top {report.top_k}")
    if report.already_in_top_k:
        st.success(f"Already in the top {report.top_k}.")
    elif report.minimal_plan is not None:
        plan = report.minimal_plan
        st.success(
            f"Fix {len(plan.flipped_constraints)} gap(s): "
            f"{', '.join(plan.flipped_constraints)}. "
            f"Rank #{plan.rank_before} → #{plan.rank_after}, "
            f"score {plan.score_before:.2f} → {plan.score_after:.2f}."
        )
    else:
        st.warning(
            "No combination of evidence fixes reaches the top "
            f"{report.top_k}. Distance, quality, or other candidates' scores dominate."
        )

    st.markdown("#### Try your own combination")
    picked = st.multiselect(
        "Treat these as SUPPORTED", [g.constraint for g in report.gaps], key="v6_flips"
    )
    if picked:
        custom = simulate_flips(
            evaluation, place_id, picked, reference_location=ref_loc, top_k=top_k
        )
        st.info(
            f"Rank #{custom.rank_before} → #{custom.rank_after} "
            f"({_format_move(custom.rank_before, custom.rank_after)}), "
            f"score {custom.score_before:.2f} → {custom.score_after:.2f}."
        )
