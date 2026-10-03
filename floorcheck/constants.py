"""Every constant the checker uses, each citing the specification section it came from.

Every constant in the package cites the specification section it came from.
Nothing in this file is a production value, and the checker reads no private
constant: no value here was taken from one.

Section 1.3 says that for D3 "the ladder starts at the calibration and every
constant that neither a calibration nor a ruling supplies must come from Table 1
of the provenance table or be spelled as its disabling sentinel". Log finding 2
records why the ruling layer is, for a clean-room implementation, exactly the set
of rulings the specification restates in prose.
"""

from __future__ import annotations

import math

# --- section 2.3, the RPLAN frame ------------------------------------------

# "one RPLAN pixel is 0.0703125 m, and the 256 pixel frame spans exactly 18 m"
# (section 2.3). Written as the quotient because section 2.3 calls 18 / 256 a
# constant rather than a measurement: "The primary source is the RPLAN paper's
# own section 4, which describes one shared square region of 18 m by 18 m rather
# than a per-plan normalisation, so 18 / 256 is a constant."
RPLAN_FRAME_SPAN_M = 18.0
RPLAN_FRAME_PIXELS = 256
RPLAN_CELL_M = RPLAN_FRAME_SPAN_M / RPLAN_FRAME_PIXELS

# --- section 2.2 and section 2.3, the six raster-derived tolerances ---------

# Section 2.2: the quantisation step "is the `joinTol` constant, whose public
# derivation in Table 1 of the provenance table is half of one RPLAN raster cell,
# 0.035 m". Section 2.3 repeats it as "`joinTol` at half a cell".
#
# The specification's decimal and its derivation disagree: half of 0.0703125 is
# 0.03515625. Log finding 5 records the disagreement and the choice. The
# derivation is taken as authoritative over the rounded decimal, because section
# 2.3's whole argument is that the six tolerances are "stated as multiples of that
# cell rather than taken from any code".
JOIN_TOL_M = RPLAN_CELL_M / 2.0  # 0.03515625 m, "half a cell" (sections 2.2, 2.3)

# Section 2.3: "`rectifyThreshold` and `localtol` at one cell". Table 1 tables
# both as "0.070 (one cell)", the rounded form of 0.0703125.
#
# Section 8.8 then overrides the first with a ruling of 2026-09-14:
# "`rectifyThreshold` carries 0. The audit's request sets the rectify threshold
# to zero, so no edge collapse is ever attempted." That ruling is restated in the
# specification's own prose, so it reaches D3 under log finding 2, and it beats
# the tabled value under the layering of section 1.3, in which "a ruling beats a
# calibration".
RECTIFY_THRESHOLD_M = 0.0  # ruled 2026-09-14, section 8.8
RECTIFY_THRESHOLD_TABLED_M = RPLAN_CELL_M  # Table 1, "0.070 (one cell)", superseded
LOCAL_TOL_M = RPLAN_CELL_M  # Table 1, "0.070 (one cell)"

# Section 2.3: "`abstol` at one millimetre". Table 1: "One millimetre, three
# orders below the RPLAN cell and below any published dimensional tolerance".
ABS_TOL_M = 0.001

# Section 2.3 lists `exactCutMissTolerance` among the six tolerances "stated as
# multiples of that cell", describing it as "the half-cell band a room's
# perimeter carries", which reads as a length. Table 1 contradicts that: the
# constant's unit is "dimensionless, relative", it is the "Relative area miss at
# which `ExactCutMissed` rejects a cut", and its value is 0.05, derived as "about
# 100 of 2450 cells" for a 3 m by 4 m room. Table 1 is taken, because it is the
# document section 1.3 makes authoritative for a constant and because the
# derivation it gives is an area ratio rather than a length. Log finding 6.
EXACT_CUT_MISS_TOLERANCE = 0.05  # dimensionless, relative (Table 1)

# --- Table 1, the constants with a public derivation -----------------------

# Table 1: "IRC 2021 R311.6 Hallways, 'The width of a hallway shall be not less
# than 3 feet (914 mm)'".
HALLWAY_WIDTH_M = 0.914

# Table 1: "IRC 2021 R311.2 egress door, clear width not less than 32 in
# (813 mm); corroborated by ADA 2010 404.2.3 and ICC A117.1-2017 404.2.2".
DOOR_WIDTH_M = 0.813

# Section 2.3: "`angtol` as one cell of deviation over the door width", and
# "`angtol` is 0.0862706 rad". Table 1 gives the arithmetic, "atan(0.0703125 /
# 0.813) = 4.94 degrees", and tables the rounded 0.086. As with `joinTol`, the
# derivation is authoritative over the rounded decimal, so the value is written as
# the arithmetic. It sits below `DOOR_WIDTH_M` because it reads that constant.
ANG_TOL_RAD = math.atan(RPLAN_CELL_M / DOOR_WIDTH_M)  # 0.0862706 rad (section 2.3)

# Table 1 and section 8.6: "An opening counts only if it can host a door, so the
# floor is the door clear width, IRC 2021 R311.2." This is rung 6's one constant.
# Rung 6 is ruled not transferable on the public path (section 8.6),
# so no rung reads it; it stays as Table 1's row, as `nearDuplicateTolerance`
# does for the disabled rung 5.
MIN_NEAR_NEIGHBOR_TOUCH_M = 0.813

# Table 1: "the solver documents it as half the hallway width".
DEGRADED_CIRCULATION_CLEARANCE_M = 0.457

# Table 1: `nearDuplicateTolerance`, status "not transferable, rung disabled".
# "This is a search-economy constant for deduplicating a candidate set; the
# check-only `graph::build_json` path grades one plan at a time and no gate
# verdict reads it." Tabled at 0 with banding off.
NEAR_DUPLICATE_TOLERANCE_M = 0.0

# --- section 6, the tiling rule --------------------------------------------

# Section 6.4: "A plan is refused when the twice-claimed area, divided by the
# area of the derived outline, exceeds one part per million (`OVERLAP_SHARE`)."
OVERLAP_SHARE = 1e-6

# Section 6.3, the backstop: "the cap at four per cent of the floor is the
# backstop under it". Re-derived on 2026-09-16 from two measured bounds: "the
# largest closed share carried by a plan with no uncovered floor and no
# twice-claimed floor is 3.44 per cent ... The largest closed share on any plan
# whatsoever is 4.70 per cent". The value was two per cent until 2026-09-16.
CLOSED_SEAM_SHARE = 0.04

# Section 6.3, the source pixel table. "Every plan carries the smallest length
# its source can tell apart, in metres, as a field of its own." The synthetic
# fixture states 0.0 and is "stated, and held to one quantisation step".
SOURCE_PIXEL_M: dict[str, float] = {
    "housegan++": 18.0 / 64.0,
    "housegan": 18.0 / 32.0,
    "gsdiff": 18.0 / 256.0,
    "rplan": 18.0 / 256.0,
    "msd": 0.38,  # "its own partition grid step", "0.38 m by default"
    "synthetic": 0.0,  # "exact geometry", "held to one quantisation step"
}

# --- sections 4.1 and 4.2, the structural exposure codes -------------------

# Section 4.1: "A code of zero or more indexes the request's token list; a
# negative code is one of four structural tokens: -1 exterior, -2 corridor, -3
# demising, -4 hallway."
EXPOSURE_EXTERIOR = -1
EXPOSURE_CORRIDOR = -2
EXPOSURE_DEMISING = -3
EXPOSURE_HALLWAY = -4

# Section 4.2: "The shared-wall detection tolerance is 0.3 m".
SHARED_WALL_TOL_M = 0.3

# --- section 5.2, the MSD grid partition -----------------------------------

# Section 5.2: "The grid step is 0.38 m." The step belongs to the grid path,
# which the ruling under subject `partition.DEFAULT_METHOD` stopped leading on
# 2026-09-18 in favour of section 5.6; the step itself is unchanged and section
# 5.2 still states it in the specification's own prose, so it reaches D3 under log
# finding 2's reading of the ruling layer.
MSD_GRID_STEP_M = 0.38

# Section 5.2: "The gap between two room interiors that still counts as one wall
# rather than as the edge of the floor is 0.20 m".
MSD_WALL_GAP_M = 0.20

# Section 5.2: "A hole in the union of grown rooms smaller than 0.5 square metres
# is a wall junction and is handed to a room under the rule of section 6.3".
MSD_JUNCTION_HOLE_MAX_M2 = 0.5
