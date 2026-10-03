"""GSDiff loader: generated samples out of data/samples/gsdiff.

NO SAMPLES EXIST YET, and the reason is a missing input rather than a missing
loader. GSDiff's released topology-constrained weights downloaded and load, and
its stack runs on this machine's torch 2.11.0+cu128, but every input it consumes
is built by `datasets/rplan-extract.py` and `datasets/rplan-process4.py` from the
RAW four channel RPLAN rasters, reading `cv2.imread(path, IMREAD_UNCHANGED)` and
then `image[:, :, 1]`, the channel carrying RPLAN's room label codes 0 to 13.
This project's RPLAN mirror does not hold those rasters. Its `Network/data.mat`
carries the vector plans, and its 70,333 PNGs come back from that exact call as
three channel previews whose channel 1 holds the three values 0, 202 and 255,
which is a colour, not a label. GSDiff contains no `.mat` reader anywhere, and
its conditioning tensors `bb_semantics` and `bb_adjacency_matrix`, along with the
corner tensors the sampling loop needs beside them, all trace back to that one
`image[:, :, 1]`.

The missing input is not missing information, which this
docstring previously had wrong. It said a raw RPLAN release was the only thing
that would unblock this. It is not. The chain reads one channel of one image and
splits it at a single threshold, `>= 14` wall and `<= 13` not, voting for room
labels only over `<= 12`, so RPLAN's four wall and door codes are one code to
every consumer and openings never have to be placed. The mirror's room rings tile
the footprint exactly and adjacent rooms share their ring edge, so the wall
network is the arrangement of those shared edges and can be drawn rather than
recovered. `tools/baselines/rplan_rasterise.py` draws it, the published scripts
then run unmodified, and on 240 stride 20 plans the round trip returns every room
count, every shared wall adjacency and every room polygon to zero pixels of
centroid error.

So what is missing now is the sampler and a ruling on which graphs to condition
it on, not the data. Until a sampler writes records here, this module must not
invent them, so `default_root()` simply finds nothing and every caller sees an
empty stream.

Frame, as it will be. GSDiff carries wall junction coordinates normalised to
[-1, 1) over the RPLAN frame, and its own scripts denormalise with
`corner * (r // 2) + (r // 2)` for whatever raster resolution they draw at. The
sampler is to denormalise at r = 256, which is RPLAN's own pixel frame, so a
GSDiff sample and the RPLAN plan its graph came from are read on one frame and
`units_per_meter` is 256/18 = 14.2222, the same constant RPLAN's loader declares.

Doors. GSDiff emits none, and this is not a gap in the sampler. Its output
vocabulary is wall junctions and wall segments, rooms are recovered as the faces
of that planar graph, and the word door does not occur anywhere in the
repository: no door class, no opening entity, no door edge attribute. So every
GSDiff Plan carries an empty `access` list, and no access rung can be graded on
this baseline. `Plan.entry` is undefined for the same reason: nothing in the
output says where circulation enters, and a guess would be this project's
invention rather than the model's output.

Semantics. GSDiff predicts six room classes plus an exterior class, which is
coarser than RPLAN's thirteen: its own eval script names them lvr, bed, sto, kit,
bat, bal. The sampler is to spell them out in full and the adapter maps them; the
collapse of RPLAN's primary bedroom and its three other bedroom words into one
`bedroom` is the model's, not this loader's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from planaudit.datasets import generated
from planaudit.plan import Plan

SOURCE = "gsdiff"
FRAME_PIXELS = 256.0
FRAME_METERS = 18.0


def default_root() -> Path:
    return generated.samples_root(SOURCE)


def manifest(root: Optional[Path] = None) -> dict:
    return generated.manifest(Path(root) if root is not None else default_root())


def list_ids(limit: Optional[int] = None, root: Optional[Path] = None) -> list[str]:
    return generated.list_ids(Path(root) if root is not None else default_root(), limit)


def load_one(source_id: str, root: Optional[Path] = None) -> Plan:
    return generated.load_one(SOURCE, Path(root) if root is not None else default_root(), source_id)


def iter_plans(limit: Optional[int] = None, root: Optional[Path] = None) -> Iterator[Plan]:
    yield from generated.iter_plans(SOURCE, Path(root) if root is not None else default_root(), limit)
