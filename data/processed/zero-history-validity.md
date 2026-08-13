# Zero-history shortlist: is the ordering worth anything?

Stratum: **206,321 units** with no crash history, carrying **7,408 holdout casualties**.

EB ceiling `1/k` per unit type: `corridor` 0.1753, `intersection` 0.3853

| B | ordering | captured | of stratum | random mean | random p95 | corridors | intersections | tie block at cut |
|---|---|---|---|---|---|---|---|---|
| 100 | `eb_estimate` | 47 | 0.63% | 3.6 (0.05%) | 8 | 0 | 100 | 14 |
| 100 | `p_pct_within_type` | 21 | 0.28% | 3.6 (0.05%) | 8 | 65 | 35 | 1 |
| 500 | `eb_estimate` | 248 | 3.35% | 17.8 (0.24%) | 27 | 0 | 500 | 11 |
| 500 | `p_pct_within_type` | 129 | 1.74% | 17.8 (0.24%) | 27 | 337 | 163 | 1 |
| 2000 | `eb_estimate` | 721 | 9.73% | 71.9 (0.97%) | 90 | 0 | 2000 | 1 |
| 2000 | `p_pct_within_type` | 514 | 6.94% | 71.9 (0.97%) | 90 | 1286 | 714 | 188 |
