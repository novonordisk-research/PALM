# Datasets

## amyloidbeta42_753.csv
753 single-point mutant variants of Amyloid-beta 42 (Abeta42). Each row is a full-length sequence with its variant ID (position-mutation format), aggregation nucleation scores (`nscore_c`, `nscore1_c`, `nscore2_c`, `nscore3_c`), familial Alzheimer's disease label (`fAD`), and binary amyloidogenicity label (`value_bool`).

## amypro22.csv
22 full-length amyloidogenic proteins from the [AmyPro database](http://amypro.net/). Each row includes the full sequence, a residue-level binary label string (`res_value_bool`) indicating amyloidogenic segments, and a sequence-level label (`value_bool`).

## amypro22_residues.csv
Residue-level expansion of `amypro22.csv` (4,049 rows). Each row corresponds to a single residue with its parent protein name, amino acid identity (`res_aa`), position index (`res_idx`), and per-residue amyloidogenicity label (`res_value_bool`).

## nnk4.csv
7,039 short peptides (1–20 aa) from an NNK codon library screen, labeled for amyloid nucleation activity (`value_bool`).

## serrano157.csv
158 peptide segments derived from aggregation-prone regions of 30 well-characterized proteins, labeled for aggregation propensity (`value_bool`). A standard benchmark dataset for aggregation prediction.
