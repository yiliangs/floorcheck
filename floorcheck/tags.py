"""Dataset category to `RoomTag`.

Section 8.7 tables the category to `RoomTag` map in full, because naming the
module that also carries it, `CATEGORY_TO_TAG` in the private audit's adapter, is
not stating the map, and the clean-room rule forbids D3 from reading that
module. Log finding 17 is what caused the map to be tabled: the specification
had previously named only fragments of it. What is below is the table of
section 8.7, hard-coded, not inferred and not imported.

Section 8.7 also states the caution the table carries forward here: "Both
dataset label orders are recorded in the adapter as inferred rather than
authoritative, so the per-category assignment is only as sound as that
inference, and the table above is the thing that caution is about."
"""

from __future__ import annotations

# The eleven tags of Table 3 of the provenance table, plus the unset tag section
# 8.7 names. Section 8.7: "`RoomTag` spellings are the variant identifiers, which
# serialise verbatim, and the set of them is `ROOM_TAGS` in the same module."
UNSET = "Unset"
ROOM_TAGS = (
    "LivingRoom",
    "PrimaryBedroom",
    "Bedroom",
    "GuestRoom",
    "Kitchen",
    "Bathroom",
    "DiningRoom",
    "Balcony",
    "Foyer",
    "Closet",
    "Hallway",
)

# Section 8.7's table, "The category to `RoomTag` map", verbatim: every
# dataset category row it gives, plus the two named `Unset` instances the same
# section states: "RPLAN `studyroom` and MSD `stairs` are the two named
# instances, and House-GAN++'s `study room` and `unknown` are two more."
STATED: dict[str, str] = {
    "livingroom": "LivingRoom",
    "masterroom": "PrimaryBedroom",
    "kitchen": "Kitchen",
    "bathroom": "Bathroom",
    "diningroom": "DiningRoom",
    "childroom": "Bedroom",
    "secondroom": "Bedroom",
    "guestroom": "GuestRoom",
    "balcony": "Balcony",
    "entrance": "Foyer",
    "storage": "Closet",
    "walkin_closet": "Closet",
    "bedroom": "Bedroom",
    "dining": "DiningRoom",
    "corridor": "Hallway",
    "storeroom": "Closet",
    "living_room": "LivingRoom",
    "dining room": "DiningRoom",
    "studyroom": UNSET,
    "stairs": UNSET,
}


def tag_for(category: str) -> str:
    """The `RoomTag` a dataset category maps to, or `Unset`.

    Section 8.7: "Any category not in that table takes the unset tag and no
    requirement at all, and the adapter counts it rather than failing on it."
    The same behaviour applies here: a category the table does not name,
    including House-GAN++'s `study room` and `unknown`, falls to `Unset`.
    """
    key = category.strip().lower()
    return STATED.get(key, UNSET)


def is_hallway(category: str) -> bool:
    """Section 8.2's exclusion: the compactness rung excludes hallways "as the
    solver's own lowest-score reduction does", and section 8.3 records that the
    proportions rung "also scores rooms tagged Hallway, which the compactness
    reduction skips"."""
    return tag_for(category) == "Hallway"
