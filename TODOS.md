# TODOS

Deferred work, with enough context to pick it up cold.

**State as of 2026-08-13:** Approach A is done and both P1 items are closed. The pipeline
runs end to end on real data (snapshot `2026-08-13`), 249 tests green in CI on Python
3.11 and 3.12, and the backtest produced a published result: R1 48.7% / R2 64.1% /
R3 82.5%, lift +18.4pp with a 95% CI of [+17.5, +19.3].

Two things changed on 2026-08-13 that affect how the rest of this list should be read:

- **The result survived the radius sweep.** Lift runs +16.1pp to +19.9pp across the seven
  settings tested, all clearing the pre-registered bar. The headline is not an artifact
  of three unexamined distances. But it is not radius-free either, and the intersection
  radius is worth ±2pp per 50 ft — anything downstream that quotes a single number should
  carry that.
- **The headline reproduced exactly** on a fresh pull, a day later, on a rebuilt
  environment with newer pandas/numpy/scipy. Same 50 units in the same order. That closes
  the "does it run anywhere else" question and leaves only the genuinely independent
  re-derivation open.

Ordered by what most threatens or most advances the published result.

---

## P1 — Independent re-derivation of the headline

**What:** Recompute the three capture rates from the committed `data/processed/`
artifacts with a short standalone script that shares no code with `src/backtest.py`.

**Why:** Promoted from P2, because it is now the *only* unclosed check on the number. The
headline has been re-verified twice — same code, same inputs, same answer — and once
re-derived on a fresh snapshot and a rebuilt environment. None of that would catch a
shared bug in `capture_rate` or `select_citywide_top_n`, which would reproduce perfectly
every time and stay invisible.

**Pros:** Cheap, and it closes the last "how do you know" question about the number.

**Cons:** Only checks the scoring, not the upstream spatial join, which is where the
harder bugs have actually been. Worth being explicit about that limit wherever the
re-derivation is reported, or it will read as more assurance than it is.

**Context:** `data/processed/run-summary.json` now carries a `join_radii` block, so a
standalone re-derivation can assert it is scoring the same configuration rather than
assuming it.

**Depends on:** nothing.

---

## P2 — Align the SPF window with the holdout so predictions are calibrated

**What:** Fit the SPF on a 24-month trailing count to match the 24-month holdout, or
scale predictions by the window ratio. Then add a calibration plot to the model card.

**Why:** The SPF trains on 36-month counts and is scored against a 24-month holdout, so
predicted counts sit on a longer window than observed ones. Measured observed/predicted
is 0.56 on surface streets and 0.78 on highways, against 24/36 = 0.67 — the gap is the
window ratio, not a broken model. Ranking is unaffected because the scale factor is
monotone, and ranking is the only claim made. But the predicted counts are not expected
casualties for the holdout window, and a model card showing a calibration plot would
currently be misleading.

**Pros:** Turns "no calibration claim" into a real one, which is what makes an SPF useful
to a DOT engineer for anything beyond ranking.

**Cons:** A 24-month trailing window has fewer events per unit than 36, so the fit may be
noisier. Worth checking whether dispersion and the headline hold before adopting it.

**Context:** Found by the /qa error analysis on 2026-08-12 and disclosed in the README's
"What is NOT claimed" list. The radius sweep is now the baseline any change to this is
measured against: re-run `scripts/radius_sensitivity.py` afterwards and the lift range
should still sit near +16 to +20pp. If it moves, the window change did something to the
ranking, not just to the scale.

**Depends on:** ~~do the P1 sensitivity work first~~ — **done**. Ready to start.

---

## P2 — Approach B: the Streamlit page (gate now open)

**What:** Ranked table plus map, an interactive budget slider over N, and a per-location
"why this ranking" explainer.

**Why:** The gate was "does Approach A produce a result worth wrapping." It did, and it
then survived the radius sweep, so the page will not be built on a number that is about
to move. The budget slider is unusually well motivated, because the N-sweep is the
finding: the lift is +2.1pp at N=13,712 and +18.4pp at N=38,909, so dragging N is not a
gimmick, it is the argument made interactive.

**Pros:** Turns a README into something a hiring manager can play with, and the
underlying numbers already exist in `data/processed/`.

**Cons:** Real scope. It also invites the reader to quote whichever N flatters the model,
so the page must show the N-dependence rather than hide it behind a default. The same now
applies to the radius: if the page exposes a single lift number, it is quoting one point
in a +16.1 to +19.9pp range.

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

**Depends on:** nothing.

---

## P3 — Follow-ups the radius work opened

- **The intersection radius deserves a defence, not just a sweep.** It is now the one
  knob shown to matter (±2pp per 50 ft), and 100 ft is currently justified by "this
  mirrors how crash records are conventionally classified" — a claim with no citation
  next to it. Either cite the convention (HSM, FHWA, or NYC DOT's own practice) or say
  plainly that it is a choice, sized by the sweep.
- **A two-knob grid for join distance × intersection radius**, if anyone wants the
  interaction. The current sweep is one-at-a-time, which was the right first pass and
  cannot see interactions. Low expected value: the corridor join distance moves the lift
  by 0.06pp on its own, so an interaction large enough to matter would be surprising.
- **CI could assert the README's numbers against `run-summary.json`.** The workflow
  currently runs lint and tests, both of which pass while the README quotes a stale
  figure — which is exactly what happened to the test count, wrong at "203" until
  2026-08-13. A dozen-line check that the headline percentages in the README match the
  committed summary would close the whole class.

---

## P2 — Design debt from `/plan-design-review`, 2026-08-13

- **No screenshot of the application may ship until the zero-history emitter exists.**
  The prototype at `~/.gstack/projects/Jeffreys-World-nyc-crash-risk-forecast/designs/comparison-view-20260813/prototype.html`
  generates its 500 ranked rows in the browser from a hardcoded street list. The scores are
  invented. They exist so column density and line length can be judged, and the file carries
  a non-dismissible banner saying so — but a cropped screenshot loses that banner, and a
  walkthrough GIF in the README (already a P3 item above) would put fabricated results on a
  portfolio page. This is the exact failure the repo's content-integrity rule exists to stop.
  *Blocked by:* the re-pull and the T3 emitter. *Do:* add a CI check that no image lands under
  `docs/` or in the README before the emitter ships, and delete the synthetic rows the day
  real output exists.

- **Contrast-audit the forked dark palette before trusting it.** The design review chose to
  ship light and dark. The dark severity values (`#e0913a`, `#c9a83c`, `#6fa980`) and
  `--muted #8C8477` were authored by eye for the portfolio's marketing page, where they
  decorate. Here they would carry meaning in a data table, at 11px, on a projector. The light
  ramp turned out to have a real failure — `DESIGN.md`'s `--sev-high #B4600A` measures
  **4.26:1** against `#FAF7F2` and fails AA, while the shipped `style.css` value `#a45709`
  measures 4.98:1 and passes. Assuming the dark set is clean is the same mistake a second
  time. *Do:* measure all four ramp levels plus `--muted` against `#14120F` at both normal and
  presentation type sizes; correct the fork; and **file the `#B4600A` bug against the
  portfolio repo**, where it is live today.

---

## P3 — Polish and small cleanups

- **`/unit/{id}` permalink has no design and is deferred past the presentation.** Decision 11
  pushed it behind the demo spine, but it is the only route where live-feed staleness is
  inspectable, and a per-location URL is the thing an engineer defending a pick would actually
  send someone. Three questions are open and none are answered anywhere: how a visitor gets
  back to the worksheet from a cold permalink load, where keyboard focus enters, and what
  "this incident data is N hours old" looks like. Recorded here so the deferral is a decision
  rather than a disappearance — which is how the mobile and accessibility specs vanished
  between revision 1 and revision 2. *Depends on:* the geometry export and the API tier
  (eng-review decision 12).

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
- **`data/cache/` accumulates one parquet per code fingerprint** — and now one per set of
  join radii as well, so a full sweep leaves seven. By design, so a rebuild never silently
  reuses stale units, but nothing prunes old ones. Add a `--prune` flag or a note in the
  README.
- **`run-summary.json` is written before `top-50-ranked.csv`.** If the CSV step fails the
  summary is already on disk and looks complete. Write both, then move them into place.

---

## Not planned

- **Cyclist and motorist modes.** The label is pedestrian casualties, to match DOT's
  list. Changing it changes the project.
- **Borough recovery via point-in-polygon.** Obsolete: every unit gets its borough from
  centerline geometry, and all 220,033 units resolved with zero nulls.

---

## Done

- **Sensitivity of the result to the three join radii** (2026-08-13). All three swept
  one-at-a-time by `scripts/radius_sensitivity.py`; results in
  `data/processed/radius-sensitivity.md`. Lift +16.1pp to +19.9pp, every setting clearing
  the bar. Closing it needed the radii turned into real parameters that reach the units
  cache key first — patched in memory, they left the cache fingerprint unchanged, and the
  sweep would have served the first setting's units for every later one and reported a
  flat, fictional insensitivity.
- **CI** (2026-08-13). `.github/workflows/test.yml`, on push and pull request, Python
  3.11 and 3.12, ruff plus pytest. No secrets and no network, because every stage is
  covered against the synthetic fixture city.
