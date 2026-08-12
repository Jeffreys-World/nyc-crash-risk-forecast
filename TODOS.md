# TODOS

Deferred work, with enough context to pick it up cold.

**State as of 2026-08-12:** Approach A is done. The pipeline runs end to end on real
data (snapshot `2026-08-12`), 210 tests green, and the backtest produced a published
result: R1 48.7% / R2 64.1% / R3 82.5%, lift +18.4pp with a 95% CI of [+17.5, +19.3].
That clears the pre-registered bar, so **the Approach B and C gates are now open.**

Ordered by what most threatens or most advances the published result.

---

## P1 — Sensitivity of the result to the three join radii

**What:** Re-run the backtest varying `MAX_JOIN_DISTANCE_FT` (150), `INTERSECTION_RADIUS_FT`
(100), and the VZV buffer (50) in `src/config.py` / `join_vzv_labels`. Report how the
headline moves at, say, 100/150/250 ft and 50/100/150 ft.

**Why:** These are the last unexamined judgement calls sitting underneath the headline.
The intersection radius in particular decides whether a crash lands on a node or a
segment, and 86% of pedestrian casualties are at intersections, so it directly shapes
both the label distribution and R1's footprint. Right now the README asserts a number
with no evidence that it survives a different, equally defensible choice.

**Pros:** It is the single check most likely to find a real problem with the published
result, and the one a reviewer with a transport background will ask for first. If the
number holds across radii, the finding gets much stronger for one afternoon of compute.

**Cons:** Each setting needs a full rebuild (`--no-cache`), roughly 1-2 minutes per run
now that the cache is fingerprinted and `split_by_treatment` is no longer O(n·m). Cheap
in wall-clock, but the result may weaken, which is the point of running it.

**Context:** Raised by the /qa pass on 2026-08-12 as the highest-value uncovered gap.
The cache key now includes a hash of `config.py`, so changing a radius automatically
invalidates the cached units rather than silently reusing them.

**Depends on:** nothing. Ready to run.

---

## P1 — CI, so the 210 tests actually protect something

**What:** `.github/workflows/test.yml` running `pytest` on push and pull request.

**Why:** The suite covers all three silent-failure guards (CRS mismatch, `log(0)`
exposure, zero-denominator capture rate) plus the regression tests for nine bugs found
across two QA passes. None of it runs unless someone remembers to run it. The eng review
deferred CI on the grounds that there was nothing to protect yet; there is now.

**Pros:** Cheap, and it makes the repo read as engineered rather than as notebooks. The
tests need no network and no Socrata token, so CI needs no secrets.

**Cons:** None worth the word. The only care needed is pinning Python 3.12 to match the
local venv.

**Depends on:** nothing.

---

## P2 — Align the SPF window with the holdout so predictions are calibrated

**What:** Fit the SPF on a 24-month trailing count to match the 24-month holdout, or
scale predictions by the window ratio. Then add a calibration plot to the model card.

**Why:** The SPF currently trains on 36-month counts and is scored against a 24-month
holdout, so predicted counts sit on a longer window than observed ones. Measured
observed/predicted is 0.56 on surface streets and 0.78 on highways, against 24/36 = 0.67
— the gap is the window ratio, not a broken model. Ranking is unaffected because the
scale factor is monotone, and ranking is the only claim made. But the predicted counts
are not expected casualties for the holdout window, and a model card showing a
calibration plot would currently be misleading.

**Pros:** Turns "no calibration claim" into a real one, which is what makes an SPF useful
to a DOT engineer for anything beyond ranking.

**Cons:** A 24-month trailing window has fewer events per unit than 36, so the fit may be
noisier. Worth checking whether dispersion and the headline hold before adopting it.

**Context:** Found by the /qa error analysis on 2026-08-12 and disclosed in the README's
"What is NOT claimed" list.

**Depends on:** nothing, but do the P1 sensitivity work first so both are measured
against the same baseline.

---

## P2 — Approach B: the Streamlit page (gate now open)

**What:** Ranked table plus map, an interactive budget slider over N, and a per-location
"why this ranking" explainer.

**Why:** The gate was "does Approach A produce a result worth wrapping." It did. The
budget slider is now unusually well motivated, because the N-sweep is the finding: the
lift is +2.1pp at N=13,712 and +18.4pp at N=38,909, so dragging N is not a gimmick, it
is the argument made interactive.

**Pros:** Turns a README into something a hiring manager can play with, and the
underlying numbers already exist in `data/processed/`.

**Cons:** Real scope. It also invites the reader to quote whichever N flatters the model,
so the page must show the N-dependence rather than hide it behind a default.

**Depends on:** nothing technical. Consider doing P1 first so the page is not built on a
result that then moves.

---

## P2 — Independent re-derivation of the headline

**What:** Recompute the three capture rates from the committed
`data/processed/` artifacts with a short standalone script that shares no code with
`src/backtest.py`.

**Why:** The headline has been re-verified (same code, same inputs, same answer) but
never re-derived. A shared bug in `capture_rate` or `select_citywide_top_n` would
reproduce perfectly and stay invisible.

**Pros:** Cheap, and it closes the last "how do you know" question about the number.

**Cons:** Only checks the scoring, not the upstream spatial join, which is where the
harder bugs have actually been.

**Depends on:** nothing.

---

## P2 — Traffic volume as a real exposure term

**What:** Join NYC traffic count data so the safety performance function uses vehicle
volume alongside segment length, rather than length alone.

**Why:** Every Highway Safety Manual safety performance function is fundamentally a
function of traffic volume and segment length. Length alone still partly conflates
"long" with "dangerous" — a mile of quiet residential street and a mile of arterial
get the same exposure. Closing this is the difference between a model that resembles
the field's standard and one that matches it.

**Pros:** Removes the last structural gap between this model and standard practice.
It is also the second question a transportation interviewer asks after exposure comes
up at all.

**Cons:** NYC traffic counts are sampled at specific locations in specific years, not
a citywide continuous surface. Coverage will be partial, so the join needs an explicit
imputation or missing-data strategy, and that strategy then needs its own disclosure
in the model card. The road-attribute imputation added on 2026-08-12
(`impute_road_attributes`, median plus a `road_attrs_imputed` predictor) is the pattern
to follow.

**Context:** Plan-eng-review Issue 2 (2026-08-12) settled the centerline universe as
the unit of analysis, which makes segment length available as a free exposure term.
That is the decided baseline. Volume is the upgrade on top of it.

**Depends on / blocked by:** ~~The centerline universe (task T1)~~ — **done**. Pinned to
`inkn-q76z`, 122,244 segments. Ready to start.

---

## P3 — Approach C: the named-streets counterfactual (gate now open)

**What:** What today's reactive, count-ranked approach would have funded historically
versus what the model would have funded, and the casualty gap between the two, shown
with real street names.

**Why:** It is the most legible version of the finding for a non-technical reader, and
the ranked output now carries real names ("E Fordham Rd & Webster Ave") rather than unit
ids, which was the blocker in spirit if not on paper.

**Cons:** The honest version has to carry the treated/untreated split, or it implies DOT
would have prevented casualties it may in fact have prevented by rebuilding the street.

**Depends on:** best done after P1, for the same reason as Approach B.

---

## P3 — Polish and small cleanups

- **Non-technical one-page executive summary**, alongside the technical model card.
  Section 10.4 of the original plan; still unwritten.
- **README walkthrough GIF** once Approach B exists, since most reviewers will not run
  the app.
- **`rw_type` is empty for intersections** in `data/processed/top-50-ranked.csv`. Nodes
  carry the derived `is_highway` flag but not the raw code, so the column reads blank for
  every intersection row. Either populate it from the approach segments or drop it from
  the export.
- **`factor_mix` and `temporal_concentration` are still computed** in `build_features`
  but are no longer SPF predictors (they were removed when the crash-derived features
  turned out to be the degenerate-fit cause). They cost time on every rebuild and are
  carried in the units parquet. Keep them if the model card wants them as descriptive
  columns, otherwise stop computing them.
- **`data/cache/` accumulates one parquet per code fingerprint.** By design, so a rebuild
  never silently reuses stale units, but nothing prunes old ones. Add a `--prune` flag or
  a note in the README.
- **`run-summary.json` is written before `top-50-ranked.csv`.** If the CSV step fails the
  summary is already on disk and looks complete. Write both, then move them into place.

---

## Not planned

- **Cyclist and motorist modes.** The label is pedestrian casualties, to match DOT's
  list. Changing it changes the project.
- **Borough recovery via point-in-polygon.** Obsolete: every unit gets its borough from
  centerline geometry, and all 220,033 units resolved with zero nulls.
