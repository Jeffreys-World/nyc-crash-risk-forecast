# QA Report — nyc-crash-risk-forecast (second pass)

**Date:** 2026-08-12
**Branch:** main
**Tier:** Standard (+ one low, fixed because it is a provenance field)
**Surface:** Python data pipeline, now running against real NYC data
**Baseline commit:** `2ff4e0b`

The first QA pass ran before the pipeline had ever touched real data. This one runs
against a completed backtest, so it targets a different class of defect: not "does the
code crash" but "does the published number mean what it says."

---

## Summary

| | |
|---|---|
| Issues found | 4 |
| Fixed and verified | 4 |
| Deferred | 0 |
| Tests | 203 → 210 (all green) |
| Lint | clean → clean |
| Headline result | **unchanged** (48.7 / 64.1 / 82.5, +18.4pp) |
| Published number corrected | treatment split |

No fix moved the headline. One moved a number already written into the README.

---

## ISSUE-006 (High) — treatment was flagged but never placed in time
**Fixed** · `src/backtest.py`, `src/pipeline.py`

`join_sip_treatment` computed `treatment_date` with care, including an earliest-wins rule
for units rebuilt twice. Nothing consumed it. `split_by_treatment` used only the boolean
`treated`, so a street rebuilt in 2015 and one rebuilt in 2026 were the same category.

The eng review's decision was "tag which priority locations were treated **and when**."
The "and when" was built and then dropped on the floor.

**Measured:** SIP completion dates run to **2026-05-29**. **4,092 units** were treated
during or after the 2024–2025 holdout window they are scored on, carrying **597 casualties
(3.3%)**.

**Fix:** `split_by_treatment` takes `holdout_start` and returns three buckets rather than
two. A location rebuilt mid-window is neither treated nor untreated for that window, so it
is named and reported instead of being folded into whichever side is convenient.

**Effect on a published number:**

| | Before | After |
|---|---|---|
| Treated | 60.4% (blended) | 60.7% treated-before-holdout |
| Treated during holdout | *(hidden inside "treated")* | 56.3%, 1,580 units |
| Untreated | 34.9% | 34.9% |

Partition verified: 9,179 + 597 + 8,283 = 18,059.

---

## ISSUE-007 (Medium) — the reported training window was not the one used
**Fixed** · `src/pipeline.py`

`train = assigned[crash_date < TRAIN_END]` was bounded on one side only, so it included
every crash back to the 2016 pull start. Meanwhile `run-summary.json` and the README both
reported `train_window: 2019–2023`.

Numerically it changed little, because the trailing features only look back 36 months from
2024-01-01. But a provenance field that does not describe the run it documents is exactly
the failure this project exists to expose, and it was sitting inside the project's own
provenance. Now bounded at both ends.

---

## ISSUE-008 (Medium) — the cache could serve a stale intermediate
**Fixed** · `src/pipeline.py`

The units cache was keyed on the snapshot date alone: `units-2026-08-12.parquet`. Edit a
feature or a spatial join, re-run without `--no-cache`, and the run silently reuses units
built by the previous version of the code.

This is not hypothetical. It happened during this project's own build: a stale units file
survived a change to the label join and produced a `run-summary.json` that disagreed with
the log printed beside it.

For a repo whose central claim is "clone this and reproduce the number," a cache that can
serve a stale intermediate is a correctness bug, not a performance detail. The key now
includes a blake2b fingerprint of `config.py`, `spatial.py`, and `features.py`, and a
rebuild logs why it happened.

---

## ISSUE-009 (Low) — VZV provenance was overwritten
**Fixed** · `src/spatial.py`

`vzv_source` was assigned, not accumulated. The corridor layer runs first and labels both
segments and the nodes along them; the intersection layer then overwrote `vzv_source` for
the 442 nodes it matched. A node that is both a VZV priority intersection *and* a point on
a VZV priority corridor recorded only "intersection."

No effect on any number — but this is a provenance field in a project whose argument rests
on provenance, so it was worth the four lines. Sources now combine as `corridor+intersection`.

---

## Health score

Adapted rubric (no browser surface; see the first report for why).

| Category | Weight | Before | After |
|---|---:|---:|---:|
| Correctness of published numbers | 40% | 70 | 100 |
| Reproducibility | 25% | 60 | 100 |
| Provenance accuracy | 20% | 70 | 100 |
| Lint / style | 15% | 100 | 100 |
| **Weighted** | | **72** | **100** |

Correctness before: one High defect affecting a published number (−15) and one Medium
(−8), against a headline that was itself sound. Reproducibility before: the cache could
silently serve stale intermediates, which is the single thing most likely to make a
reader's re-run disagree with the README.

---

## What this pass did not cover

- **The headline result was re-verified, not re-derived.** R1/R2/R3 and the +18.4pp lift
  are unchanged and were confirmed by a full re-run, but no independent implementation
  checks them.
- **No sensitivity analysis.** The 150 ft join radius, the 100 ft intersection radius, and
  the 50 ft VZV buffer are all unexamined choices that could move the result.
- **No error analysis by road class**, which is the next item in the README's status table
  and the one that would connect this back to the parent dashboard's highway finding.
