# Independent re-derivation of the headline

Snapshot `2026-08-13`, radii `{"intersection_radius_ft": 100.0, "max_join_distance_ft": 150.0, "vzv_buffer_ft": 50.0}`, 10,000 bootstrap iterations. Frame written by `src.pipeline.run`.

Every number below was rebuilt from `data/processed/scored-units.parquet` by `scripts/rederive_headline.py`, which imports nothing from `src/`. It checks the scoring layer only — not the spatial join, not the feature build, not the negative-binomial fit. That limit is the important part; the script's docstring says why.

| Quantity | Published | Re-derived | Δ | |
|---|---:|---:|---:|---|
| universe units | 220,033 | 220,033 | +0.00e+00 | ok |
| holdout casualties | 18,059.0000 | 18,059.0000 | +0.00e+00 | ok |
| priority units (N) | 38,909 | 38,909 | +0.00e+00 | ok |
| citywide N | 38,909 | 38,909 | +0.00e+00 | ok |
| EB blend, max abs(rebuilt - published) | 0.0000 | 0.0000 | +0.00e+00 | ok |
| R1 DOT published, citywide | 48.7402 | 48.7402 | +0.00e+00 | ok |
| R2 raw count, citywide | 64.1121 | 64.1121 | +0.00e+00 | ok |
| R3 empirical bayes, citywide | 82.5350 | 82.5350 | +0.00e+00 | ok |
| lift R3 - R2 | 18.4229 | 18.4229 | +0.00e+00 | ok |
| bootstrap CI low | 17.5361 | 17.5361 | +0.00e+00 | ok |
| bootstrap CI high | 19.2953 | 19.2953 | +0.00e+00 | ok |
| R2 raw count, borough-stratified | 9.9120 | 9.9120 | +0.00e+00 | ok |
| R3 empirical bayes, borough-stratified | 10.9641 | 10.9641 | +0.00e+00 | ok |
| R1 at units treated before the holdout | 60.7147 | 60.7147 | +7.11e-15 | ok |
| R1 at units treated during the holdout | 56.2814 | 56.2814 | +0.00e+00 | ok |
| R1 at untreated units | 34.9270 | 34.9270 | +0.00e+00 | ok |
| units treated before the holdout | 57,772 | 57,772 | +0.00e+00 | ok |
| units treated during the holdout | 4,092 | 4,092 | +0.00e+00 | ok |

**18 of 18 agree.** The headline re-derives.

## How much of each ranking was decided by the tie-break

Nothing above depends on this and the summary makes no claim about it. It is reported because it is the sharpest thing the re-derivation can see: a ranking whose budget runs out inside a block of equal scores has stopped ranking, and the rest of its picks came out of a hash.

| Ranking | Units sharing the cut-off score | Picks the tie-break decided |
|---|---:|---:|
| R2 | 206,321 | 25,197 (64.8% of 38,909) |
| R3 | 30 | 13 (0.033% of 38,909) |

## Notes

- R2: 206,321 units share the cut-off score, so 25,197 of its 38,909 selections (64.8%) were decided by the tie-break and not by the ranking
- R3: 30 units share the cut-off score, so 13 of its 38,909 selections (0.033%) were decided by the tie-break and not by the ranking
- pre-registered bar: lift +18.42pp against a 5.0pp threshold, CI [+17.54, +19.30] excludes zero
