# The floorcheck plan document

floorcheck grades one plan per JSON file. This page states that file's format for anyone
writing one by hand or with their own converter. It is derived from the code that reads
the document, `floorcheck/model.py` (`plan_from_dict`, `plan_from_json`), and from the
code that consumes the parsed plan: `floorcheck/ingest.py` (`ingest`),
`floorcheck/ladder.py` (`grade`), `floorcheck/scope.py`
(`unit_count`), `floorcheck/tags.py` (`tag_for`) and `floorcheck/report.py` (`check`).
The reference producer is `tools/export_plans.py` (`as_document`), which writes one such
document per plan it converts; every field it writes is listed below.

Three worked examples ship in `floorcheck/examples/*.json`: `rplan-apartment.json`,
`gsdiff-seamed.json` and `msd-wrapped-core.json`. They are hand-written documents, not
converter output, and carry no `meta`.

## Conventions

- The document is one JSON object, read as UTF-8.
- Every length is in metres and every area in square metres. There is no unit field and no
  scale field: a converter divides its source coordinates by its own units-per-metre
  before writing them.
- A point is a JSON array of exactly two finite numbers, `[x, y]`, in metres. The origin
  and the direction of the axes are the producer's choice.
- A ring is a JSON array of at least three points describing one simple closed polygon.
  Closing the ring (repeating the first point at the end) is optional. Winding order is
  free: ingest orients every ring counter-clockwise, removes repeated vertices and
  quantises every coordinate onto a 0.03515625 m grid (half of one 0.0703125 m RPLAN
  raster cell) before anything is measured (`floorcheck/hygiene.py`, `hygiene`).
- Field names are case-sensitive. A key the reader does not know is ignored, at the top
  level, in a room and in `meta`.

## Top-level fields

| field | type | unit | required | default | read by the checker |
|---|---|---|---|---|---|
| `rooms` | array of room objects, at least one | | yes | | yes: the geometry every step grades |
| `source` | non-empty string | | yes | | yes: lower-cased, then picks the source pixel and the corpus (see below) |
| `name` | any JSON value, converted to a string | | no | the path given on the command line | only as the report header's `plan` |
| `entry` | point | metres | no | absent | validated, then carried; no rung reads it |
| `floorExtent` | ring | metres | no | absent | yes: bounds the seam repair at ingest |
| `adjacency` | array of `[i, j]` room-index pairs | | no | `[]` | validated, then carried; no step grades it |
| `meta` | object | | no | `{}` | only `meta.sourcePixelM` |
| `entrySynthetic` | boolean | | no | absent | no: written by the reference producer, ignored |
| `entryOffsetM` | number | metres | no | absent | no: written by the reference producer, ignored |

What each field means:

- `rooms`: the plan's rooms, in an order that `adjacency` and every room index in the
  report refer to. See [Rooms](#rooms).
- `source`: the dataset or generator the plan came from. The value is lower-cased on
  read. It decides the source pixel when `meta.sourcePixelM` is absent (see
  [Source pixel](#source-pixel)) and the corpus the requirements are read from (see
  [Corpus](#corpus)). The report header carries it as `source`.
- `name`: a label for the plan. `plan_from_dict` takes `str(payload["name"])` when the key
  is present, and the file path otherwise.
- `entry`: the point where circulation enters the plan. No rung reads it: rung 6
  (`finalAdjacency`), the only rung that did, is not transferable on the public path
  (specification section 8.6) and is reported as skipped. It is parsed so
  that a malformed point is refused rather than carried silently. Put the point on the
  plan's outer boundary, on a wall of the room the plan is entered through.
- `floorExtent`: the floor the rooms are meant to tile, where the source states one. It
  is not a boundary the rooms are graded against. Its only use is at ingest, where every
  narrow uncovered strip the seam repair would hand to a room is first clipped to this
  polygon. Absent means no clipping at all. It is read with the ring rules above.
- `adjacency`: which rooms the source says are connected. Each entry is a pair of room
  indices into `rooms`, both in range and distinct; pairs are normalised to
  ascending order and deduplicated. No rung reads it. It is parsed so that a malformed list is refused rather than
  carried silently.
- `meta`: provenance. See [meta](#meta).
- `entrySynthetic`, `entryOffsetM`: written by `tools/export_plans.py` (`as_document`)
  whenever it could build the plan's outline. It places `entry` on the outline of the
  room union: at the outline point nearest the entry the source stated, or, where the
  source stated none, at the midpoint of the outline's longest edge. `entrySynthetic` is
  `true` in the second case, and `entryOffsetM` is the distance in metres the entry moved
  to land on the outline. floorcheck reads neither.

## Rooms

Each element of `rooms` is an object:

| field | type | unit | required | default | read by the checker |
|---|---|---|---|---|---|
| `category` | non-empty string | | yes | | yes: mapped to a room tag, which selects the room's requirements |
| `ring` | ring | metres | yes | | yes |
| `unit` | string | | no | absent (`null`) | yes: counts apartments, which selects the floor kind |

- `ring`: exactly one ring. A room is never a polygon with holes, and the reader refuses
  every spelling of one: a `ring` that is an object, a `ring` that is an array of rings,
  or a room carrying a `holes` or `interiors` key (reason `room-with-hole`). A
  self-intersecting ring is not refused at parse time; ingest repairs it and keeps the
  largest lobe as the room. A ring that the quantisation collapses below three vertices
  bounds no floor and is rejected at rung 1 (`roomOutline`).
- `unit`: the apartment this room belongs to, on a plan that holds more than one.
  `floorcheck/scope.py` (`unit_count`) counts the distinct `unit` values; with none the
  plan is one apartment. When the count is two or more, the report's `refusals` list no
  longer names `corridorExposure` and `demisingExposure`; with one unit it names both.
  The count also selects the floor kind rungs 2 and 3 are graded against (see
  [Corpus](#corpus)): an MSD plan with two or more units is a whole floor. The reference
  producer writes it on every MSD room that belongs to an apartment: on a whole floor, the
  apartment `planaudit.units.assign` derives for it, as `u<index>`; on an apartment, its
  own id. A room outside every apartment, and every RPLAN or generated room, carries none.

### Categories and room tags

`category` is the source's own room label. `floorcheck/tags.py` (`tag_for`) strips it,
lower-cases it and looks it up in the table below; any label the table does not name,
including a misspelling, takes the tag `Unset`. A room tagged `Unset` carries no
requirement at all: rung 3 (`roomProportions`) skips it and the report's `observations`
carry no minimum-width entry for it. It is still part of the geometry: it tiles the floor, it
is graded by rungs 1, 2 and 4.

| `category` | tag |
|---|---|
| `livingroom`, `living_room` | `LivingRoom` |
| `masterroom` | `PrimaryBedroom` |
| `bedroom`, `childroom`, `secondroom` | `Bedroom` |
| `guestroom` | `GuestRoom` |
| `kitchen` | `Kitchen` |
| `bathroom` | `Bathroom` |
| `dining`, `diningroom`, `dining room` | `DiningRoom` |
| `balcony` | `Balcony` |
| `entrance` | `Foyer` |
| `storage`, `storeroom`, `walkin_closet` | `Closet` |
| `corridor` | `Hallway` |
| `studyroom`, `stairs` | `Unset` |
| anything else | `Unset` |

The tag decides which row of the requirement table a room is held to: under the default
requirement source, `dataset-p05`, the row of `floorcheck/category-statistics.md` for
that tag within the chosen corpus for `areaReq` and `minWidth`, and the tag's `propReq`
in `floorcheck/plan-level-floors.json` for the plan's floor kind; under
`--requirements standards`, the standards row
for that tag. A room tagged `Hallway` is also left out of rung 2 (`roomCompactness`) and
of rung 4's orthogonal-wall precondition.

## meta

`meta` is an object of provenance. floorcheck reads exactly one key of it,
`sourcePixelM`, and carries every other key without looking at it. The keys below are
the ones the reference producer writes; a hand-written document may omit all of them,
and may add its own.

| field | type | unit | written by the producer | read by the checker |
|---|---|---|---|---|
| `sourcePixelM` | number | metres | always | yes: the source pixel, overriding the built-in table |
| `unitsPerMeter` | number | source coordinate units per metre | always | no |
| `corpus` | string, `rplan` or `msd` | | always | no (see [Corpus](#corpus)) |
| `group` | string | | always | no |
| `sourceId` | string | | always | no |
| `planauditSource` | string | | always | no |
| `accessKinds` | array of strings, one per `adjacency` pair | | when the plan has adjacency | no |
| `offAxisFraction` | number, 0 to 1 | | when the source loader measured it | no |
| `partitionAreaError` | number | | when the source loader reports a nonzero value | no |

- `sourcePixelM`: the smallest length the plan's source can tell apart. It must be a JSON
  number (a numeric string also parses); `null` or any other value that does not read as
  a number refuses the document as `malformed-plan`. See
  [Source pixel](#source-pixel).
- `unitsPerMeter`: the factor the producer divided the source coordinates by. The
  coordinates in the document are already metres; this records the conversion.
- `corpus`: the corpus whose statistics the producer intends the plan to be graded
  against: `msd` for MSD plans, `rplan` for RPLAN plans and for every generated plan,
  because a generated sample is an attempt at an RPLAN plan.
- `group`: the name of the population the plan was exported with, for example `msd`,
  `msd units`, `rplan` or a generator's name. It is also the directory the producer
  writes the document into.
- `sourceId`: the plan's identifier inside its source dataset. The producer's `name` is
  `<group>/<sourceId>`.
- `planauditSource`: the source name the converter's loader gave the plan, before it was
  mapped to the `source` token floorcheck reads.
- `accessKinds`: the kind of opening each `adjacency` pair stood for in the source
  (`adjacency` itself has no room for it), in the same order.
- `offAxisFraction`: the fraction of wall length the source loader found more than two
  degrees off the plan's axes.
- `partitionAreaError`: the room-area error the MSD grid partition reported for this plan.

## Source pixel

Ingest closes every uncovered strip no wider than the plan's seam tolerance. The tolerance is the source pixel plus
0.03515625 m (`floorcheck/model.py`, `Plan.seam_tolerance_m`). The source pixel is taken
from `meta.sourcePixelM` when present, and otherwise from this table, keyed by the
lower-cased `source` (`floorcheck/constants.py`, `SOURCE_PIXEL_M`):

| `source` | source pixel (m) |
|---|---|
| `rplan` | 0.0703125 |
| `gsdiff` | 0.0703125 |
| `housegan++` | 0.28125 |
| `housegan` | 0.5625 |
| `msd` | 0.38 |
| `synthetic` | 0.0 |

A plan whose `source` is not in the table and which carries no `meta.sourcePixelM` is
refused at ingest with `unknown-source-pixel`. A document from any other source is
therefore graded only if it states `meta.sourcePixelM`. `0.0` is a valid value and holds
the plan to exact tiling at the quantisation step.

## Corpus

The corpus decides which section of `floorcheck/category-statistics.md` (`## rplan` or
`## msd`) the `dataset-p05` area and width requirements are read from. Only `rplan` and
`msd` exist. Under `--requirements standards` the requirement values do not depend on it.

The rung 2 compactness floor and the rung 3 `propReq` floors are plan-level: they are read
from `floorcheck/plan-level-floors.json` for the plan's floor kind, which
`floorcheck/requirements.py` (`floor_kind`) takes from the corpus and the apartment count
(`unit_count`): `msd-floor` for an `msd` plan with two or more distinct room `unit`
values, `msd-unit` for any other `msd` plan, and `rplan` for every other corpus. The
floor kind used is reported in the header's `floorKind` field.

The corpus is chosen in `floorcheck/report.py` (`check`), which takes an optional
`corpus` argument: when it is not given, the corpus is `msd` if the plan's `source` is
`msd` and `rplan` for every other source. The `floorcheck` command never passes one, so
on the command line the corpus is always inferred from `source`, and `meta.corpus` is not
read at all: a document whose `meta.corpus` disagrees with its `source` is graded against
the corpus its `source` implies. A document without `meta.corpus` is graded exactly as
one with it. The reference producer writes `meta.corpus` so that a document states the
corpus it was exported for, and a Python caller that wants that corpus passes it
explicitly, `check(plan_from_dict(document), corpus=document["meta"]["corpus"])`. For every
document the producer writes, that value equals the inferred one. The corpus used is
reported in the header's `corpus` field.

## What the report echoes

The report header (`floorcheck/report.py`, `Report.to_dict`) carries `checker`, `version`,
`specification`, `plan` (the document's `name`, or the file path), `source` (the
lower-cased `source`), `requirementSource` (`dataset-p05` or `standards`) and `corpus`.
Nothing from `meta` is echoed.

## Refusals

A document is refused in one of two places. Both are results, not errors: the command
prints the report and exits 0, the ingest `outcome` names the reason, and `ladder` is
`null`.

### The document cannot be read as a plan

Raised as `PlanRejected` by `floorcheck/model.py` (`plan_from_json`, `plan_from_dict`,
`_as_ring`, `_as_point`), or by `floorcheck/report.py` (`check_file`) for a file it cannot
open. The report's `detail` gives the text in the last column, where `N` is a room index,
`W` names the offending value and `(a, b)` the offending pair.

| reason | when | detail |
|---|---|---|
| `unreadable-file` | the file cannot be opened | the operating system's message |
| `malformed-json` | the file is not valid JSON | the JSON parser's message |
| `malformed-plan` | the top level is not an object | `the top level is not an object` |
| `malformed-plan` | `meta` is present and not an object | `meta is not an object` |
| `malformed-plan` | `meta.sourcePixelM` is present and does not read as a number | `meta.sourcePixelM is not a number` |
| `no-rooms` | `rooms` is missing, not an array, or empty | `a plan carries at least one room` |
| `malformed-room` | a room is not an object | `room N is not an object` |
| `missing-category` | a room's `category` is missing, not a string, or empty | `room N carries no category` |
| `room-with-hole` | a room has a `holes` or `interiors` key | `room N declares interior rings` |
| `room-with-hole` | a `ring` (or `floorExtent`) is an object | `W is a polygon object, not a ring` |
| `room-with-hole` | a `ring` (or `floorExtent`) is an array of rings | `W carries more than one ring` |
| `malformed-ring` | a `ring` (or `floorExtent`) is missing or not an array | `W is not a list of points` |
| `malformed-ring` | fewer than three vertices once a closing repeat is dropped | `W has fewer than three distinct vertices` |
| `malformed-point` | a point, or `entry`, is not a pair of numbers | `W is not a pair of numbers` |
| `non-finite-coordinate` | a coordinate is NaN or infinite | `W is not a finite number` |
| `malformed-room` | a room's `unit` is present and not a string | `room N unit is not a string` |
| `missing-source` | `source` is missing, not a string, or empty | `a plan names the source it came from` |
| `malformed-adjacency` | an entry is not a two-element array | `an adjacency entry is not a pair` |
| `malformed-adjacency` | an entry's elements are not integers | `an adjacency entry is not a pair of indices` |
| `malformed-adjacency` | an index is outside `rooms` | `adjacency (a, b) indexes no room` |
| `malformed-adjacency` | a pair joins a room to itself | `adjacency (a, b) joins a room to itself` |

The checks run in document order: rooms first, then `source`, `floorExtent`, `entry`,
`adjacency` and `meta`, and the first failure is the one reported.

### The rooms do not tile a floor

Returned by `floorcheck/ingest.py` (`ingest`) on a document that parsed. These carry the
measurements that decided them under the ingest block's `measurements`.

| reason | when |
|---|---|
| `unknown-source-pixel` | `source` is not in the source pixel table and `meta.sourcePixelM` is absent |
| `empty-plan` | no room has any area |
| `untiled-rooms` | floor no room covers survives the seam repair: a hole in the room union wider than the seam tolerance, or a part of the union, other than its largest, that no channel within the seam tolerance joins to it |
| `closed-seam-share` | the seam repair handed more than 4 per cent of the floor to rooms |
| `rooms-claim-the-same-floor` | rooms overlap by more than one part per million of the floor; the rooms lying almost wholly inside another are listed under `swallowed` |

A plan that clears ingest reaches the rung ladder, whose rejections are verdicts about
the plan rather than about the document.

## A minimal document

Two rooms and an entry on the outer wall of the living room. `name`, `floorExtent`,
`adjacency` and `meta` are all omitted, and the source pixel comes from the table.

```json
{
  "source": "rplan",
  "rooms": [
    {"category": "livingroom", "ring": [[0, 0], [4.2, 0], [4.2, 3.5], [0, 3.5]]},
    {"category": "bedroom", "ring": [[4.2, 0], [7.7, 0], [7.7, 3.5], [4.2, 3.5]]}
  ],
  "entry": [0, 1.75]
}
```

`floorcheck plan.json --format text` accepts it at ingest, passes rungs 1, 2 and 3,
reports rungs 4, 5 and 6 as not transferable, and ends with
`verdict         passed every transferable rung`. The same plan from a source the table
does not name needs one more line, for example `"meta": {"sourcePixelM": 0.05}`.
