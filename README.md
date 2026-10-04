# floorcheck 0.1.0

floorcheck grades a floor plan, given as room polygons, against the transferable subset of a
gate ladder: an ordered list of validity checks (ingest, geometry, room sizes and proportions,
adjacency), each of which either passes the plan or rejects it with a
reason. It reports a verdict per rung, in JSON or as text. This archive also holds the one
converter that writes floorcheck's plan schema from raw datasets, the Swiss documents that
converter wrote for the thirteenth gate run, the geometry of the generated plans that run graded,
and the checker's own outcome table for that run, so a reader can reproduce the table's rows without
any other software.

## What the archive holds

| path | what it is |
|---|---|
| `floorcheck/` | the checker, its example plans, `SCHEMA.md`, the plan document format it reads, `category-statistics.md`, the per-category room statistics its default requirement source (`dataset-p05`) grades against, and `plan-level-floors.json`, the plan-level floors of rungs 2 and 3 for each kind of plan |
| `tests/floorcheck/` | the checker's tests (`pip install ".[dev]"`, then `pytest`) |
| `planaudit/`, `tools/export_plans.py` | the converter: reads raw RPLAN, MSD or generator files and writes one plan document per plan |
| `tools/checker_outcomes.py`, `tools/intervals.py` | grade every document of an exported directory and print the checker's pass rates with Wilson intervals and the ground-truth over generator margin; with `--profile`, where each plan stopped, check by check, and the share of plans with no verdict; with `--buildings`, the margin with an interval that resamples whole MSD buildings; with `--suppress CHECK`, the same grading with one check's rejection suppressed (the check is evaluated, and a plan it would reject continues to the checks below) |
| `tools/cluster_bootstrap.py` | the cluster bootstrap behind `--buildings`: resamples whole clusters of plans and recomputes the margin in every replicate |
| `tools/msd_buildings.py` | infers which MSD floors belong to one building from the overlap of their structure grids, and writes the building key `--buildings` reads |
| `tools/plan_level_floor.py` | fits the plan-level floors of rungs 2 and 3 at each false-rejection rate on half of the real plans and reports the margin on the other half, MSD split by building |
| `data/msd/` | 18,351 plan documents from the Modified Swiss Dwellings dataset: every floor (`msd/`, 4,028 documents) and every apartment (`msd_units/`, 14,323 documents) of the first 4,167 floors of its training split, under CC BY 4.0 (see `ATTRIBUTION.md`) |
| `data/generated/` | 14,902 generated plans, room types and polygons only: House-GAN++ (`houseganpp/`, 4,000), House-GAN (`housegan/`, 4,000), HouseDiffusion (`housediffusion/`, 3,991) and GSDiff (`gsdiff/`, 2,911), the documents the thirteenth run graded with every other field removed and each renamed `<generator>-<NNNN>` (see `ATTRIBUTION.md`) |
| `tables/checker-outcomes-run13.md` | the checker's own pass rates and margin on the thirteenth run, copied unchanged from the run's committed record |

## What it leaves out, and why

- **RPLAN documents.** RPLAN's licence forbids redistribution, so no document derived from it
  ships. Export them from your own copy of RPLAN (below).
- **The generators' conditioning.** Every generated plan was drawn from a room program taken
  from RPLAN, and each sample record names that RPLAN plan, its record index and bucket, and
  carries the requested adjacency, doors and entry read from RPLAN. All of that is withheld:
  `data/generated/` keeps each room's type and polygon, the source and its pixel size, which is
  everything the checker grades. Names are sequence numbers in the order of a hash of each
  document's own content, so they carry no RPLAN identifier. The converter reads full sample
  records; run it on samples you generate.
- **The specification** the checker implements, of which checks carry over to public data,
  accompanies the paper as supplementary material and is not in this archive.

## Install

Python 3.11 or later.

    python -m venv .venv
    . .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install floorcheck-0.1.0.zip

This installs five commands: `floorcheck`, `floorcheck-export`, `floorcheck-outcomes`,
`floorcheck-buildings` and `floorcheck-floors`.

## Grade one plan

    floorcheck floorcheck-0.1.0/data/msd/msd_units/0#u0.json --format text

The default output is JSON: a header, the ingest outcome, the rungs this implementation does not
grade and why, one entry per rung with its verdict and, on a rejection, the worst offender, and
an overall verdict. `--requirements standards` grades against the public standards instead of
the dataset statistics. The exit status is 0 on any verdict and 2 only when the checker itself
failed. To write your own plan document, by hand or from your own converter, see `floorcheck/SCHEMA.md`.

## Grade the Swiss documents and compare with the table

    floorcheck-outcomes floorcheck-0.1.0/data/msd --out msd-outcomes.md

`msd-outcomes.md` holds the row "MSD ground truth, units path" under each of three
denominators (graded plans, plans the checker admitted, plans that reached a verdict). Each row
must equal the same row of `tables/checker-outcomes-run13.md`. The floor documents are graded
too; the table carries no row for them. The table's other rows and its margin need the RPLAN and
generator documents, which you export yourself.

## Grade the generated plans and compare with the table

    floorcheck-outcomes floorcheck-0.1.0/data/generated --out generated-outcomes.md

The rows House-GAN++, House-GAN and HouseDiffusion under each of the three denominators must equal
the same rows of `tables/checker-outcomes-run13.md`. The GSDiff documents are graded too (117 of
2,911 pass); the table carries no row for them. The margin needs the MSD documents beside them.

## Export from your own data

From your own copy of RPLAN (`data.mat` from the RPLAN distribution), the thirteenth run's sample
of 4,000 plans:

    floorcheck-export --rplan 4000 --rplan-stride 20 --rplan-path /path/to/data.mat --out my-plans
    floorcheck-outcomes my-plans

From the MSD training split, extracted from `modified-swiss-dwellings-v1-train.zip`, the
documents under `data/msd/` again. MSD's graph files are pickled torch tensors, so this needs the
`msd` extra (`pip install "floorcheck-0.1.0.zip[msd]"`):

    floorcheck-export --msd 4167 --msd-units 4000 --msd-root /path/to/train_extracted --out msd-plans

From generator samples, one directory per generator under `--samples-root`:

    floorcheck-export --baseline houseganpp=4000 --samples-root /path/to/samples --out my-plans

`--msd N` and `--msd-units N` take the first N floors in sorted identifier order, so a smaller N
writes a prefix of the same documents.

## Reproduce

These are the commands the archive was tested with before deposit, run in a fresh virtual
environment from a directory outside the archive. `A` is the unpacked archive; `MSD` is the
extracted MSD training split; `RUN` is a directory holding all six groups of the thirteenth run's
export (MSD floors and units, RPLAN, House-GAN, House-GAN++, HouseDiffusion), which you rebuild
with `floorcheck-export` from your own RPLAN copy and generator samples; the deposited generated
plans grade as the run's own documents did, plan by plan.

    python -m venv venv
    . venv/bin/activate         # Windows: venv\Scripts\activate
    pip install "floorcheck-0.1.0.zip[msd]"

    # 1. one deposited document: exit status 0 and a verdict
    floorcheck "A/data/msd/msd_units/0#u0.json" --format text

    # 2. every deposited document: the "MSD ground truth, units path" row and the
    #    House-GAN++, House-GAN and HouseDiffusion rows of each denominator equal
    #    the same rows of A/tables/checker-outcomes-run13.md
    floorcheck-outcomes A/data/msd --out msd-outcomes.md
    floorcheck-outcomes A/data/generated --out generated-outcomes.md

    # 3. the converter reproduces the deposit: every document it writes is
    #    byte-identical to the file of the same name under A/data/msd/
    floorcheck-export --msd 40 --msd-units 40 --msd-root MSD --out exp

    # 4. the whole run: the output is byte-identical to the shipped table,
    #    margin included
    floorcheck-outcomes RUN --out run13-outcomes.md

    # 5. the checker's own tests
    pip install pytest
    cd A && python -m pytest -q

    # 6. where each deposited document stopped, check by check, and the share
    #    with no verdict
    floorcheck-outcomes A/data/msd --profile

    # 7. the whole run with MSD buildings resampled whole: build the building
    #    key from the raw MSD structure grids, then the margin's interval and
    #    the plan-level threshold fit at each false-rejection rate (--workers
    #    sets its processes; without it the fit starts one per core but one)
    floorcheck-buildings --root MSD/struct_in --cache buildings.json
    floorcheck-outcomes RUN --profile --buildings buildings.json
    floorcheck-floors RUN --workers 4 --buildings buildings.json --features features.jsonl --out floors.md

    # 8. the paper's central counterfactual: the whole run with the proportion
    #    check evaluated but its rejection suppressed
    floorcheck-outcomes RUN --suppress roomProportions

At deposit, step 1 reached no verdict, because `roomAppendices` did not run on a precondition
fault; step 2 gave 10,079 passes of 14,323 graded apartments, and the three generator rows equal the
table's; step 3 wrote 39 floors and 149 apartments (floor 103 is refused, as in the deposit),
all identical; step 4 matched the table byte for byte; step 5 passed 131 tests. Grading the
deposit takes about two minutes and the whole run about five. Steps 6 and 7 append their sections
after the table of steps 2 and 4 and leave that table unchanged. Step 8 reverses the margin on
every denominator: MSD apartments over House-GAN++, -13.1 over graded plans, -23.2 over admitted
plans and -3.4 over plans that reached a verdict, with House-GAN++ passing 86.5 per cent of its
graded plans against the MSD apartments' 73.4.

## Licence and citation

The software is under the MIT licence (`LICENSE`). The documents under `data/msd/` are under
CC BY 4.0 with the attribution in `ATTRIBUTION.md`; the documents under `data/generated/` are under
CC BY 4.0, with the generators credited in `ATTRIBUTION.md`. `CITATION.cff` gives the citation.
