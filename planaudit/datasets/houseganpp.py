"""House-GAN++ loader: generated samples out of data/samples/houseganpp.

The samples are produced by tools/baselines/houseganpp_sample.py from the
released `checkpoints/pretrained.pth`, conditioned on bubble graphs built from
RPLAN's Network/data.mat. That script owns everything about how a sample comes
to exist; this module only reads what it wrote, through
planaudit.datasets.generated.

Frame. The generator emits one 64 by 64 mask per graph node, and in training
those masks covered the whole 256 pixel RPLAN frame recentred, so 64 mask pixels
span the same 18 m and `units_per_meter` is 64/18 = 3.5556. That is the same 18 m
frame RPLAN itself is read on.

Rooms are a partition because the sampler makes them one, using the baseline's
own rule and not a new one: `build_graph` in the House-GAN++ dataset paints every
room into a single label image and rebuilds each mask from it, so the training
data never has two rooms sharing a pixel, while generated masks come off
independent per node decoders and do overlap. The same painting is applied before
a ring is traced. What it does not repair is a void enclosed by the rooms, which
the model does produce and which the solver reads as an edge with one adjacent
room; that is a real property of the output and shows up in the D1 build failures
rather than being smoothed away here.

Doors. House-GAN++ emits them. Interior doors and the front door are first class
graph nodes with their own generated masks, so `Plan.access` carries one edge per
generated interior door, paired to rooms by the IoU rule the baseline's own
`estimate_graph` uses, and `Plan.entry` is the centre of the generated front door
mask. An occasional sample drops a door mask entirely, in which case that edge is
simply absent rather than guessed.

What is synthesised rather than recovered, and it matters for reading any number
built on these samples: the CONDITIONING doors. RPLAN's .mat records room
adjacency, not door placement, and this mirror's PNGs carry no opening channel,
so the interior door nodes in the input graph are a maximum spanning tree of the
adjacency graph weighted by shared wall length. The door count therefore matches
the real House-GAN++ inputs, one fewer than the rooms, but the individual door
placements are an assumption of this project's, recorded in the sample manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from planaudit.datasets import generated
from planaudit.plan import Plan

SOURCE = "houseganpp"
FRAME_MASK_PIXELS = 64.0
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
