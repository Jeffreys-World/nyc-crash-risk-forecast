# Next session — start here

Written 2026-08-14 end of session. Everything below is committed and reproducible.

## Where things stand

The last unclosed check on the headline is closed. `scripts/rederive_headline.py` rebuilds
all 18 published quantities from a committed per-unit frame, in code that imports nothing
from `src/`, and every one agrees — bootstrap bounds to the last digit. It runs in CI on
every push, on both interpreters, against artifacts in the repo, so it needs no snapshot
and no network.

```
.venv/Scripts/python.exe -m pytest                      # 283 tests, ~40 s
.venv/Scripts/python.exe scripts/rederive_headline.py   # ~50 s, no data pull needed
.venv/Scripts/python.exe -m src.pipeline                # ~35 s from the units cache
```

Full path from cold — `scripts/pull_snapshots.py` first, ~5 min — still works and still
reproduces the headline exactly.

**What the re-derivation does not check, and this is the part to carry forward:** the
spatial join, the trailing counts, and the negative-binomial fit. It closes the *scoring*
layer — selection, ranking, tie-breaking, capture rates, the interval, the EB blend. The
harder bugs in this project have all been in geometry, and geometry is still standing on
one implementation.

## The finding that changes the app design

**R2 draws 64.8% of its list out of a hat, and now there is a number for it.**

At N=38,909 the raw-count ranking exhausts the 13,712 units it can order, then fills the
remaining **25,197 places** from the 206,321 tied at zero — decided by a hash of the unit
id, not by any count. The Empirical Bayes ranking draws 13 of 38,909.

That is the README's central claim arriving as a measurement instead of an argument, and
it is what the **"B of N tied" disclosure** in the worksheet should be built on. The
earlier B=100 figure (14 units share the cut-off score in the zero-history stratum) was
the same phenomenon seen through a much smaller window. The re-derivation prints both
numbers for any N, so the interface has a source rather than a hand-computed constant.

## Still true from the previous session

**Two lists, ranked within type.** Segmented control: `Intersections | Corridors`. Each
ranked on `spf_prediction` within its own type, each stating its own base rate. Corridors
have **14.4x lift** at B=100 against intersections' 6.2x, and percentile-normalising across
types would throw that away — measured, and worse: it costs 55% of capture.

**The EB `1/k` ceiling was never the problem.** The engineering review's top finding was
wrong in its causal claim. Removing the ceiling entirely changes capture by zero. The base
rate does the work: 0.076 casualties per intersection against 0.014 per corridor.

## Do this next, in order

1. **Build the app.** Lane A is clear and nothing gates it now. Design is approved and
   prototyped:
   `~/.gstack/projects/Jeffreys-World-nyc-crash-risk-forecast/designs/comparison-view-20260813/`
   — `prototype.html` is interactive (click rows, drag budget, `P` for presentation mode,
   `T` for theme). 12 tasks in `tasks-design-review-20260813-173803.jsonl`, 7 of them P1.
   Start with **D1 (fork DESIGN.md)** — no dependencies, and it unblocks all visual work.
   The worksheet can now read `data/processed/scored-units.parquet` directly instead of
   needing a new emitter: it carries `unit_id`, `unit_type`, `borough`, `spf_prediction`,
   `eb_estimate`, the trailing count, the holdout count, and both flags for all 220,033
   units. Street names are *not* in it — they were left out to keep the file committable,
   so a name join against the centerline is still needed for display.
2. **Amend the design doc** at
   `~/.gstack/projects/Jeffreys-World-nyc-crash-risk-forecast/flextop-main-design-20260813-132249.md`.
   Issue 10 is resolved and its causal claim was wrong. Three "critical gaps" drop to one —
   tie blocks — and that one now has measured numbers to specify against.
3. **Fix the 6 prototype bugs** if you intend to demo the prototype rather than the real
   app. Full report with repro steps: `.gstack/qa-reports/qa-report-prototype-2026-08-13.md`.
   Worst two: the table clips its own columns at 1280×720 (709px table, 640px pane), and
   only 6 rows of 100 are visible because names wrap to three lines.
4. **Align the SPF window with the holdout** (now P1 in TODOS). The model over-predicts by
   roughly the 24/36 window ratio, so it cannot carry a calibration claim. Note that this
   will move the ranking, and the re-derivation in CI will fail until
   `data/processed/` is regenerated in the same commit. That is the guard working, not a
   problem to route around.

## Open, unresolved

- **The why-panel sign inversion.** `is_highway` carries a negative coefficient
  (`README:444`). Emit fitted coefficients and render signed contributions, or the panel
  tells engineers the flag pushed a location up when it pushed it down.
- **Premise 5** (is budget denominated in locations?) and **Premise 6** (no practitioner
  consulted) — outreach messages are drafted and ready to send in
  `Practitioner_Outreach_Messages.md`. Sending one is still the highest-value hour
  available, and nothing built this session changes that.
- **The intersection radius still has no citation.** It is the one knob the headline turns
  on (±2pp per 50 ft) and 100 ft is justified by "this mirrors how crash records are
  conventionally classified", with nothing next to it. Cite HSM, FHWA, or NYC DOT practice,
  or say plainly that it is a choice sized by the sweep.
- **Portfolio repo has a live contrast bug.** `DESIGN.md` specifies `--sev-high #B4600A`,
  which measures 4.26:1 and fails WCAG AA. Shipped `style.css` uses `#a45709` at 4.98:1 and
  is correct. Fix the doc, not the code. Different repo.
- **`gstack-slug` resolves wrong from this directory** (returns
  `Jeffreys-World-personal-website`), so gstack tools misfile learnings. Pin `SLUG`
  explicitly in every gstack bash call.

## Do not

- Do not adopt the percentile-within-type fix. It is measured and it is worse.
- Do not screenshot the prototype into the README. Its rows are synthetic. See TODOS P2.
- Do not edit a number in the README by hand. `tests/test_published_numbers.py` anchors
  roughly forty published figures to the artifacts that produced them, and it will fail —
  which is the point. Regenerate the artifacts, then update the prose.
