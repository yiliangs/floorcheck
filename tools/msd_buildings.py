"""A building key for the MSD floors, inferred from their structure.

The MSD v1 release this repository reads (the ICCV 2023 challenge train split,
4,167 floors named `0` to `4166`) carries no building, site or original Swiss
Dwellings floor id: the graphs have no graph attributes, the archive holds no
metadata file, and the challenge repository publishes no mapping. The building
has to be inferred.

Each floor's `struct_in` array carries, per pixel, a structure mask and the
pixel's x and y location in metres. Floors of one building share load-bearing
walls and columns and, as measured below, a common coordinate frame, so two
floors of one building mark largely the same 0.5 m cells as structure. Two
floors are linked when the Jaccard index of their structure cell sets reaches
`THRESHOLD`, and a building is a connected component of those links, labelled
by its smallest floor id.

Over all 8.7 million floor pairs the index is bimodal: 99.6 per cent of pairs
fall below 0.2, and the pairs above 0.3 are a separate population of a few
thousand. The count per 0.1 bin is lowest from 0.4 to 0.6, so 0.5 splits the
two. At 0.3 the links chain 556 floors into one component, which is no longer
a building; 0.4 is kept as the coarser sensitivity setting. A building this
key misses (a ground floor whose structure differs from the storeys above)
splits into smaller clusters, so the key errs toward treating dependent floors
as independent, and two identical buildings on one site merge, which is a
cluster the bootstrap should respect anyway.

Run `python -m tools.msd_buildings` to build the cache
`data/derived/msd-buildings.json` and print the component sizes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
STRUCT = REPO / "data" / "msd" / "train_extracted" / "struct_in"
CACHE = REPO / "data" / "derived" / "msd-buildings.json"
CELL = 0.5
THRESHOLD = 0.5
SENSITIVITY = 0.4


def structure_cells(path: Path, cell: float = CELL) -> set[tuple[int, int]]:
    """The `cell`-metre cells a floor marks as structure, in its own frame."""
    a = np.load(path).astype(np.float32)
    mask = a[..., 0] == 0
    x = np.round(a[..., 1][mask] / cell).astype(int)
    y = np.round(a[..., 2][mask] / cell).astype(int)
    return set(zip(x.tolist(), y.tolist()))


def jaccard(cell_sets: list[set]) -> np.ndarray:
    """Pairwise Jaccard index of the floors' cell sets, zero on the diagonal."""
    import scipy.sparse as sp

    keys: dict = {}
    rows: list[int] = []
    cols: list[int] = []
    for i, cells in enumerate(cell_sets):
        for c in cells:
            rows.append(i)
            cols.append(keys.setdefault(c, len(keys)))
    m = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(cell_sets), max(len(keys), 1)),
    )
    inter = (m @ m.T).toarray()
    size = np.diag(inter).copy()
    union = size[:, None] + size[None, :] - inter
    j = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    np.fill_diagonal(j, 0.0)
    return j


def components(similarity: np.ndarray, ids: list[str], threshold: float) -> dict[str, str]:
    """Floor id to building label: the smallest floor id of its linked component."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    _, labels = connected_components(sp.csr_matrix(similarity >= threshold), directed=False)
    smallest: dict[int, str] = {}
    for floor, label in zip(ids, labels):
        if label not in smallest or int(floor) < int(smallest[label]):
            smallest[label] = floor
    return {floor: smallest[label] for floor, label in zip(ids, labels)}


def build(root: Path = STRUCT) -> dict:
    """The building key at `THRESHOLD` and `SENSITIVITY`, from every floor under `root`."""
    ids = sorted((p.stem for p in root.glob("*.npy")), key=int)
    similarity = jaccard([structure_cells(root / f"{i}.npy") for i in ids])
    return {
        "cell": CELL,
        "floors": len(ids),
        "thresholds": {
            f"{t:.2f}": components(similarity, ids, t) for t in (THRESHOLD, SENSITIVITY)
        },
    }


def load(threshold: float = THRESHOLD, cache: Path = CACHE) -> dict[str, str]:
    """Floor id to building label at `threshold`, from the cache."""
    if not cache.exists():
        raise SystemExit(f"missing {cache}; build it with: python -m tools.msd_buildings")
    return json.loads(cache.read_text(encoding="utf-8"))["thresholds"][f"{threshold:.2f}"]


def building_fold(label: str) -> str:
    """The fold a building lands in: the parity of its smallest floor id."""
    return "A" if int(label) % 2 == 0 else "B"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="floorcheck-buildings", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=STRUCT)
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args(argv)
    key = build(args.root)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.cache.write_text(json.dumps(key, sort_keys=True), encoding="utf-8")
    for threshold, mapping in key["thresholds"].items():
        sizes = np.bincount(np.unique(list(mapping.values()), return_inverse=True)[1])
        first = {f: b for f, b in mapping.items() if int(f) < 4000}
        folds = [building_fold(b) for b in first.values()]
        print(
            f"threshold {threshold}: {len(sizes)} buildings over {key['floors']} floors,"
            f" {int((sizes == 1).sum())} single-floor, largest {int(sizes.max())};"
            f" first 4,000 floors in {len(set(first.values()))} buildings,"
            f" fold A {folds.count('A')} floors, fold B {folds.count('B')}"
        )
    print(f"wrote {args.cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
