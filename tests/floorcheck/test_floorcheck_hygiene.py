"""Slice 2a: ring hygiene and quantisation, specification section 2.2."""

from __future__ import annotations

import math

import pytest

from floorcheck.constants import ANG_TOL_RAD, DOOR_WIDTH_M, JOIN_TOL_M, RPLAN_CELL_M
from floorcheck.hygiene import (
    dedupe,
    drop_collinear_across_rings,
    hygiene,
    orient_ccw,
    quantise_point,
    quantise_rings,
    signed_area,
)


def test_the_step_is_half_an_rplan_cell():
    """Section 2.2: the step "is the `joinTol` constant, whose public derivation
    ... is half of one RPLAN raster cell". Section 2.3 fixes the cell at
    0.0703125 m. The derivation is taken over the specification's rounded 0.035
    (log finding 5)."""
    assert RPLAN_CELL_M == pytest.approx(0.0703125)
    assert JOIN_TOL_M == pytest.approx(0.03515625)
    assert JOIN_TOL_M == RPLAN_CELL_M / 2.0


def test_the_angle_tolerance_is_one_cell_over_the_door_width():
    """Section 2.3: "`angtol` is 0.0862706 rad", one cell of deviation over the
    door width. The derivation is taken over Table 1's rounded 0.086, as for
    `joinTol`."""
    assert ANG_TOL_RAD == pytest.approx(0.0862706, abs=1e-7)
    assert ANG_TOL_RAD == math.atan(RPLAN_CELL_M / DOOR_WIDTH_M)


def test_quantisation_closes_a_gap_narrower_than_the_step():
    """Section 2.2's purpose: two faces of a wall that should coincide land on the
    same lattice point."""
    on_lattice = quantise_point((3.0, 0.0))[0]
    left = quantise_point((on_lattice, 0.0))
    right = quantise_point((on_lattice + JOIN_TOL_M * 0.4, 0.0))
    assert left == right


def test_quantisation_moves_a_coordinate_that_is_not_on_the_lattice():
    """The other face of the same rule, and log finding 7: the step is derived
    from RPLAN's raster, so an RPLAN coordinate lands on it exactly while a source
    whose own resolution is not a multiple of it is moved by up to half a step.
    MSD's 0.38 m grid step is 10.81 quantisation steps, so no MSD width survives
    section 2.2 unchanged."""
    rplan_cell = RPLAN_CELL_M
    assert quantise_point((7 * rplan_cell, 0.0))[0] == pytest.approx(7 * rplan_cell)
    msd_step = 0.38
    assert quantise_point((msd_step, 0.0))[0] != pytest.approx(msd_step)
    assert abs(quantise_point((msd_step, 0.0))[0] - msd_step) <= JOIN_TOL_M / 2.0


def test_quantisation_rounds_rather_than_truncates():
    """Truncation would bias every coordinate toward the origin by up to a full
    step and reopen on one side of a wall the gap the step exists to close."""
    assert quantise_point((JOIN_TOL_M * 0.9, 0.0))[0] == pytest.approx(JOIN_TOL_M)


def test_rings_are_closed_and_deduplicated():
    """Section 2.2: "every ring is closed, deduplicated"."""
    ring = ((0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0))
    assert dedupe(ring) == ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


def test_rings_are_oriented_counter_clockwise():
    """Section 2.2: "oriented counter-clockwise"."""
    clockwise = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))
    assert signed_area(clockwise) < 0
    assert signed_area(orient_ccw(clockwise)) > 0
    counter = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    assert orient_ccw(counter) == counter


def test_a_collinear_vertex_no_other_ring_sees_is_dropped():
    """Section 2.2: "Collinear vertices are dropped exactly"."""
    ring = ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))
    out = drop_collinear_across_rings(quantise_rings([ring]))
    assert len(out[0]) == 4
    assert all(point != (1.0, 0.0) for point in out[0])


def test_a_vertex_a_neighbouring_ring_sees_as_a_corner_survives_in_both():
    """Section 2.2: "only where the vertex is collinear in every ring that carries
    it. A vertex where three rooms meet is a real corner for at least one of them,
    and dropping it from the one ring that sees it as flat reopens a gap that the
    quantisation has just closed"."""
    # The long room's top edge runs from x=0 to x=2 through (1, 1), which is flat
    # for it. Two rooms sit above, meeting at exactly (1, 1), which is a corner
    # for each of them.
    below = ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (0.0, 1.0))
    upper_left = ((0.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0))
    upper_right = ((1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0))

    corner = quantise_point((1.0, 1.0))
    out = hygiene([below, upper_left, upper_right])
    assert corner in out[0], "the shared corner must survive in the ring that sees it as flat"
    assert len(out[0]) == 5


def test_the_drop_is_exact_and_never_within_a_tolerance():
    """Section 2.2: "dropped exactly, never within a tolerance ... a tolerant drop
    is worse still, because it displaces the edge by the tolerance and the
    neighbour's segment stops lying on it"."""
    # One lattice step of bow in the middle of an otherwise straight edge. A
    # tolerant rule would flatten it; an exact rule keeps it.
    ring = (
        (0.0, 0.0),
        (4 * JOIN_TOL_M, JOIN_TOL_M),
        (8 * JOIN_TOL_M, 0.0),
        (8 * JOIN_TOL_M, 8 * JOIN_TOL_M),
        (0.0, 8 * JOIN_TOL_M),
    )
    out = hygiene([ring])
    assert len(out[0]) == 5, "a vertex one step off the line is a corner, not a flat"


def test_hygiene_runs_quantisation_before_the_collinear_drop():
    """Log finding 3: the section's own reason for the cross-ring rule ("reopens a
    gap that the quantisation has just closed") only holds if the drop runs after
    the quantisation."""
    # Off-lattice by less than half a step, and collinear only once snapped.
    ring = (
        (0.0, 0.0),
        (1.0, JOIN_TOL_M * 0.3),
        (2.0, 0.0),
        (2.0, 2.0),
        (0.0, 2.0),
    )
    out = hygiene([ring])
    assert len(out[0]) == 4, "the middle vertex is flat once quantised, so it goes"


def test_a_ring_whose_every_vertex_is_flat_is_left_alone_rather_than_emptied():
    """A degenerate ring is section 6's problem, not hygiene's; hygiene must not
    turn it into an empty tuple that later code cannot report on."""
    line = ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))
    out = hygiene([line])
    assert len(out[0]) >= 3
