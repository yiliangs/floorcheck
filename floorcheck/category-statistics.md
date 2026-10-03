# Per-category corpus statistics

D2 task 2. Every number below is a statistic of a whole dataset corpus, never of
the plan being graded, which is what the transfer specification
requires. The private audit reads the cached form of these tables as its
"dataset-p05" source, and floorcheck reads this document;
the public derivations of the transfer specification are the other source and
are public rather than measured.

Rooms are grouped by the `RoomTag` they map to under `CATEGORY_TO_TAG` in the
private audit's adapter, which `floorcheck/tags.py` restates, so RPLAN
`childroom` and `secondroom` are pooled into `Bedroom`, and `storage` with
`walkin_closet` into `Closet`. A category that maps to `Unset` takes no
requirement and is not tabled.

Lengths are metres and areas square metres. RPLAN rooms are converted from raster
pixels at 0.0703125 m per pixel, derived in the provenance table's section "The
RPLAN metric scale". MSD polygons are already metric. MSD is counted through
`planaudit.units`, so a room appears once per apartment and floor-level shared
circulation outside every apartment is not counted.

These tables were rendered from the cached fits of an earlier scan of each
corpus, not fitted by a fresh scan of the corpora.

`aspect` is the room's bounding box long side over short side, so 1.0 is square.
`min width` is the shorter bounding-box dimension.

Two caveats before any number here is used as a requirement.

MSD lengths are no longer quantised, and the numbers below are not the ones this
document carried through the eighth gate run. The ruling of 2026-09-18 makes
the rectified partition of `planaudit/partition.py` the default, and it carries
no grid step, so an MSD width below is a multiple of nothing. Every MSD number
here between the 2026-09-14 ruling and that one was fitted on the grid
partition at a 0.38 m cell, which is why the MSD width column used to take only
the values 0.76, 1.14, 1.52, 1.90 and so on. It no longer does, and a reader
meeting an MSD width requirement quoted to a multiple of 0.38 m
elsewhere is reading a figure from a run at or before the eighth.

That change is not only a change of precision, because the partition the corpus
is fitted on is also the partition the graded plans are built on. The grid put
every wall on an axis by construction, which raised a room's measured compactness
and lowered its measured elongation, not by accident but by method, so a floor
fitted from the grid was fitted from rooms the partition had squared.

A floor the loader will not read is skipped, counted and named rather than
dropped, and the count is in the cache beside these tables. It is not a sample:
a run refuses those same floors at ingest and counts them there, so this fit and
the run that reads it cover the same corpus.

The two corpora do not describe the same thing by the same name. RPLAN's
`livingroom` is the combined living, dining and circulation space of an Asian
apartment, which is why its median is 34.9 m2 against MSD's 29.0 m2 and why RPLAN
records a `DiningRoom` only 1,304 times in 80,788 plans. Pooling the two corpora
into one requirement would average two different rooms, so each dataset's
requirements are taken from its own corpus.


## rplan

533,206 rooms over 10 mapped categories.

| RoomTag | count | area p5 | p25 | p50 | p75 | p95 | width p5 | p25 | p50 | aspect p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Bedroom | 103,884 | 7.5196 | 9.6899 | 11.3115 | 13.1111 | 16.3542 | 2.3906 | 2.8125 | 3.0234 | 1.0208 | 1.1000 | 1.2143 | 1.3750 | 1.6977 |
| Bathroom | 97,002 | 2.4373 | 3.3618 | 4.1380 | 5.1218 | 7.1340 | 1.2656 | 1.5469 | 1.7578 | 1.0323 | 1.1333 | 1.3077 | 1.5833 | 2.1667 |
| Balcony | 86,528 | 2.0171 | 3.5299 | 4.7758 | 6.0117 | 9.1362 | 0.8438 | 1.1953 | 1.4062 | 1.1935 | 1.8846 | 2.3913 | 3.0556 | 4.7273 |
| LivingRoom | 80,788 | 25.5251 | 30.9188 | 34.8887 | 39.6299 | 48.0542 | 4.0078 | 5.2031 | 5.9062 | 1.0323 | 1.1717 | 1.3939 | 1.6885 | 2.2548 |
| PrimaryBedroom | 80,444 | 11.0050 | 13.1803 | 14.8711 | 16.8585 | 20.6851 | 2.8125 | 3.1641 | 3.3750 | 1.0333 | 1.1481 | 1.2857 | 1.4737 | 1.8333 |
| Kitchen | 77,719 | 3.7178 | 4.8845 | 5.8535 | 7.0697 | 9.7690 | 1.4062 | 1.6875 | 1.9688 | 1.0556 | 1.2759 | 1.5556 | 1.9091 | 2.6154 |
| Closet | 4,389 | 0.6674 | 1.6908 | 2.6301 | 3.8414 | 7.0499 | 0.4922 | 0.9844 | 1.3359 | 1.0417 | 1.2000 | 1.4783 | 1.9412 | 3.7086 |
| DiningRoom | 1,304 | 4.5027 | 6.8225 | 8.6517 | 11.0544 | 15.8005 | 1.6172 | 2.1797 | 2.5312 | 1.0263 | 1.1316 | 1.3000 | 1.5886 | 2.3945 |
| GuestRoom | 860 | 6.4861 | 8.5133 | 10.2041 | 11.8504 | 15.2172 | 2.1797 | 2.6016 | 2.8125 | 1.0233 | 1.1111 | 1.2093 | 1.3795 | 1.8300 |
| Foyer | 288 | 1.3220 | 2.3384 | 4.1528 | 6.8077 | 12.7700 | 0.7734 | 1.1953 | 1.5469 | 1.0311 | 1.2052 | 1.5811 | 2.0875 | 3.4475 |

Pooled room compactness, 4 pi area over perimeter squared, excluding Hallway as the solver's own lowest-score reduction does. A room-level statistic, reported for reference: the `minPassingPPScore` in force is plan-level, from `floorcheck/plan-level-floors.json`.

| p01 | p05 | p25 | p50 | p75 |
|---|---|---|---|---|
| 0.3764 | 0.4618 | 0.6427 | 0.7483 | 0.7777 |

Every plan in this corpus loaded. No floor was skipped.


## msd

117,225 rooms over 8 mapped categories.

| RoomTag | count | area p5 | p25 | p50 | p75 | p95 | width p5 | p25 | p50 | aspect p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Bedroom | 33,271 | 10.7169 | 13.0949 | 15.4166 | 17.4974 | 23.3078 | 2.5699 | 3.0290 | 3.4497 | 1.0332 | 1.1670 | 1.3465 | 1.5586 | 1.8945 |
| Bathroom | 22,281 | 1.8588 | 3.4804 | 4.5805 | 5.6195 | 7.7728 | 1.0578 | 1.5612 | 1.8376 | 1.0388 | 1.1873 | 1.3661 | 1.6328 | 2.1715 |
| Hallway | 17,143 | 2.4761 | 5.2640 | 8.7551 | 13.0718 | 19.9772 | 1.2185 | 1.7229 | 2.2724 | 1.0503 | 1.2767 | 1.6514 | 2.3020 | 3.6631 |
| Balcony | 15,262 | 2.6789 | 5.6628 | 8.7577 | 13.1255 | 27.6884 | 1.0662 | 1.5092 | 2.0045 | 1.1221 | 1.6132 | 2.2088 | 3.0313 | 5.2317 |
| Kitchen | 14,006 | 4.8957 | 7.2067 | 8.6995 | 10.8709 | 15.1372 | 1.6984 | 2.2254 | 2.5239 | 1.0380 | 1.1816 | 1.4029 | 1.6850 | 2.3979 |
| LivingRoom | 10,535 | 17.4866 | 22.9432 | 29.4543 | 35.1836 | 46.0534 | 3.5528 | 4.0785 | 4.6897 | 1.0282 | 1.1552 | 1.3642 | 1.6253 | 2.3460 |
| Closet | 4,312 | 0.7964 | 1.9047 | 2.7913 | 4.2021 | 8.4403 | 0.6747 | 1.1178 | 1.3798 | 1.0380 | 1.1939 | 1.4252 | 1.8013 | 2.6073 |
| DiningRoom | 415 | 5.3962 | 8.9139 | 11.6463 | 16.0964 | 28.7626 | 1.9902 | 2.6586 | 2.9405 | 1.0472 | 1.1489 | 1.2728 | 1.5249 | 1.8523 |

Pooled room compactness, 4 pi area over perimeter squared, excluding Hallway as the solver's own lowest-score reduction does. A room-level statistic, reported for reference: the `minPassingPPScore` in force is plan-level, from `floorcheck/plan-level-floors.json`.

| p01 | p05 | p25 | p50 | p75 |
|---|---|---|---|---|
| 0.3443 | 0.5141 | 0.6903 | 0.7481 | 0.7747 |

1 plans were skipped because the loader refused them, and the percentiles above are taken over the corpus without them. A run refuses the same plans at ingest and counts them there, so this fit and the run that reads it cover the same corpus.

Skipped: 2118.

| cause | plans |
|---|---|
| GEOSException: TopologyException: Iterated noding failed to converge after 6 iterations (near 25.96 -7.5) | 1 |

