# Attribution: the Modified Swiss Dwellings documents

The plan documents under `data/msd/` are derived from the Modified Swiss Dwellings (MSD)
dataset, which is licensed under the Creative Commons Attribution 4.0 International licence
(CC BY 4.0, https://creativecommons.org/licenses/by/4.0/). They are redistributed here under
the same licence.

Source: Casper van Engelenburg, Fatemeh Mostafavi, Emanuel Kuhn, Yuntae Jeon, Michael Franzen,
Matthias Standfest, Jan van Gemert and Seyran Khademi. "MSD: A Benchmark Dataset for Floor Plan
Generation of Building Complexes." European Conference on Computer Vision (ECCV), 2024.
doi:10.1007/978-3-031-73636-0_4. Dataset: https://data.4tu.nl/datasets/e1d89cb5-6872-48fc-be63-aadd687ee6f9

Changes made: each document restates one floor (`data/msd/msd/`) or one apartment of a floor
(`data/msd/msd_units/`) of the first 4,167 floors of the training split, in sorted identifier
order, in floorcheck's plan schema. `floorcheck-export` in this archive wrote them from the
extracted training split and reproduces them byte for byte; no document is edited by hand.

The CC BY 4.0 licence applies to these documents only. The software in this archive is
licensed under the terms in `LICENSE`.
