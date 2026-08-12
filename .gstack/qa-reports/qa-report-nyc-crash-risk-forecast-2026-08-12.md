# QA Report — nyc-crash-risk-forecast

**Date:** 2026-08-12
**Branch:** main
**Tier:** Standard (critical + high + medium)
**Surface tested:** Python data pipeline (no browser surface exists — see note)
**Baseline commit:** `5e04f03`
**Final commit:** `e1abd5e`

---

## Note on method

This repo has no web application: no server, no HTML, no routes, no `package.json`. Its
only output is a number and a chart in the README. The eng review's test plan said the
same — "Approach A (audit slice). No UI in this scope, so there are no pages or routes."
Browser testing would have produced screenshots of nothing.

So the QA contract was applied to the real surface. Instead of navigating pages, 11
adversarial probes fed the pipeline the inputs real NYC data will actually produce:
missing coordinate columns, all-rows-excluded frames, duplicated indices, null boroughs,
multi-part geometry, Socrata's string-typed numerics, and duplicate join keys. Each bug
found was fixed at source, committed atomically, covered by a regression test, and
re-verified by re-running the probe.

---

## Summary

| | |
|---|---|
| Issues found | 5 |
| Fixed and verified | 5 |
| Deferred | 0 |
| Reverted | 0 |
| Tests | 184 → 194 (all green) |
| Lint | clean → clean |
| **Health score** | **69 → 99** |

**Two of the five would have corrupted the headline number.** Neither was reachable by
the existing 184 tests, because both need input shapes the fixture city doesn't produce.

---

## Issues

### ISSUE-001 (High) — duplicate index labels collapsed crashes
**Status:** fixed, verified · `5ad7d15` · `src/spatial.py`

`assign_crashes_to_units` identifies crashes by index label across its two-stage
intersection-then-corridor handoff. With a non-unique index, `drop_duplicates(subset="_left_idx")`
merged distinct crashes that happened to share a label.

- **Probe:** 4 crashes in → 2 assigned.
- **Trigger in real use:** `pd.concat` of two snapshot parquet files without `ignore_index=True`.
- **Why it wasn't silent:** `AssignmentReport.validate()` caught the imbalance and raised.
  The pipeline would have died mid-run rather than published a wrong number — the guard
  worked. But it died on a precondition nobody had written down.
- **Fix:** reset the index on entry so the precondition is enforced, not assumed.
- **After:** 4 in → 4 assigned.

### ISSUE-002 (High) — null borough became a phantom sixth borough
**Status:** fixed, verified · `dbb2036` · `src/backtest.py`

`select_borough_stratified` grouped with `dropna=False`, so units with no borough
collected under a null key and were treated as a borough in their own right: own casualty
total, own 50% stopping rule, own picks.

- **Probe:** the highest-scoring unit had no borough and was selected on that basis alone,
  under a group keyed `'nan/corridor'`.
- **Why this one matters most:** DOT ranks within five boroughs. A model handed a sixth
  gets picks DOT never had, and its capture rate rises for a reason that has nothing to do
  with the method being better. This lands directly on the pre-registered comparison.
- **Trigger in real use:** certain. Real centerline data carries units with no borough —
  bridges, boundary and shoreline segments. It would have fired on the first real run.
- **Fix:** excluded from stratified selection and counted in the notes, not dropped from
  the universe — the capture-rate denominator must stay the true casualty total, and the
  citywide regime still ranks them. Raises if *every* unit lacks a borough, since that
  means the universe build is broken rather than the data being thin.
- **After:** only the real-borough unit selected; no `nan` group; exclusion reported.

### ISSUE-003 (Medium) — MultiLineString became one unit spanning two places
**Status:** fixed, verified · `46144c7` · `src/spatial.py`

`build_segment_universe` took geometry as given, so a MultiLineString row became a single
unit whose `length_ft` summed disconnected pieces.

- **Probe:** a 2-part MultiLineString produced 1 unit, 0 intersections.
- **Two silent consequences:** the exposure term is wrong and feeds the SPF offset
  directly; and the unit occupies two locations at once, so crashes from both pool into
  one risk score and one ranking row.
- **Trigger in real use:** DCP LION ships MultiLineStrings and is one of the three live
  centerline candidates. Picking it would have produced a ranking that looked fine.
- **Fix:** explode to one row per LineString before measuring length.
- **After:** 2 units, each with its own length.

### ISSUE-004 (Medium) — duplicate unit_id double-counted the capture rate
**Status:** fixed, verified · `091d6df` · `src/backtest.py`

`capture_rate` accepted a repeated `unit_id`, counting its casualties into the denominator
twice and into the numerator too when selected.

- **Probe:** denominator 20 where the truth was 15, numerator 10 where it was 5 — a
  clean-looking 50% that was wrong on both sides.
- **Fix:** raise. `build_universe` already enforces unique ids, so reaching this state
  means a join upstream fanned out. Deduplicating here would paper over the real defect
  while still reporting a number.

### ISSUE-005 (Low) — factor_mix docstring overstated its denominator
**Status:** fixed, verified · `e1abd5e` · `src/features.py`

The docstring claimed "share of a unit's crashes"; the denominator is non-null factor
*mentions*. A crash naming two factors yields 0.5 each, not 1.0.

- **Behaviour unchanged** — it is correct for a mix feature. Only the description was
  wrong, and reading it as a share of crashes overstates how dominant any single factor
  looks, including "Unspecified", which is the most common value in this dataset and
  already carries an interpretation argument.
- **Fix:** corrected the docstring, added a test pinning the semantics so they cannot
  drift silently.

---

## Health score

Browser categories (console, links, visual, accessibility) do not apply. Adapted rubric:

| Category | Weight | Before | After |
|---|---:|---:|---:|
| Correctness | 40% | 54 | 100 |
| Data integrity guards | 25% | 60 | 100 |
| Test coverage | 20% | 85 | 95 |
| Lint / style | 15% | 100 | 100 |
| **Weighted** | | **69** | **99** |

Correctness before: 2 high (−15 each) + 2 medium (−8 each) = 54. Test coverage stays at 95
rather than 100 because the probes found five defects the 184-test suite missed, which is
evidence the suite has blind spots, not that it now has none.

---

## What this did not cover

- **No real data.** The centerline source is still unpinned, so the pipeline has never
  run against NYC Open Data. Every finding here comes from synthetic input.
- **No network path.** `fetch_socrata` is covered by fakes only. Real pagination against
  a live 800k-row endpoint is untested.
- **No numerical validation of the SPF.** Tests confirm the fit converges with positive
  dispersion; nothing confirms the coefficients are right for real crash data.

The next run against real data should be treated as its own QA pass.
