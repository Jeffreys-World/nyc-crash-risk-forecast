# Next session — start here

Written 2026-08-13 end of session. Everything below is committed and reproducible.

## Where things stand

The pipeline works again from a cold start. `.env` was never needed — the Socrata token is a
throttle, not a gate (`.env.example:10`). Full path, ~6 minutes total:

```
.venv/bin/python scripts/pull_snapshots.py     # ~5 min, 6 sources, 36MB -> data/raw/
.venv/bin/python -m src.pipeline               # ~35 s, reproduces the headline
.venv/bin/python scripts/zero_history_validity.py   # ~10 s, the T1 check
```

**Headline reproduced exactly:** R3 − R2 = +18.4pp, CI +17.5 to +19.3. 249 tests pass, ruff clean.
That closed BLOCKER-2 from the morning QA (a clone could not previously reproduce its own headline).

## The finding that changes the plan

**T1 validated the wedge.** The zero-history shortlist captures ~13x a random draw from the same
stratum: 47 casualties at B=100 vs 3.6 random, 248 vs 17.8 at B=500. Lane B is unblocked.

**But the engineering review's top finding was wrong in its causal claim.** It said the `1/k` EB
ceiling structurally excludes 136,537 corridors, and called it a critical gap blocking all
front-end work. Measured:

| ordering | B=100 captured | corridors in top 100 |
|---|---|---|
| `eb_estimate` (shipped) | 47 | 0 |
| `spf_prediction` (no ceiling at all) | 47 | 3 |
| `p_pct_within_type` (the prescribed fix) | **21** | 65 |

Removing the ceiling entirely changes capture by zero and admits 3 corridors. **The ceiling is not
why corridors are absent.** Base rate is: 0.076 casualties per intersection vs 0.014 per corridor,
a 5.3x gap. Pedestrians are struck at intersections. The model is reporting a true fact.

The prescribed fix costs 55% of capture because percentile-normalizing **discards the base rate**,
which is signal, not a scale artifact. Both SPFs already output expected casualties for the site,
so they were always comparable.

**And corridors are the model's best subject, not its worst.** Ranked within their own type:
**14.4x lift** at B=100, against intersections' 6.2x. High dispersion (k=5.70) made the corridor
fit look weak; it sits on top of genuinely better ordering.

## Decision taken (approved this session)

**Two lists, ranked within type.** Segmented control: `Intersections | Corridors`. Each ranked on
`spf_prediction` within its own type, each stating its own base rate. Keeps corridors' 14.4x
instead of discarding it, and matches how money is actually allocated — an intersection and a
corridor do not come out of the same line item. `spf_prediction` is already a column on the
scored frame; this needs a `groupby`, not new machinery.

## Do this next, in order

1. **Amend the design doc** at
   `~/.gstack/projects/Jeffreys-World-nyc-crash-risk-forecast/flextop-main-design-20260813-132249.md`.
   Issue 10 is resolved and its causal claim was wrong. Three "critical gaps" drop to one
   (tie blocks). Record the two-list decision.
2. **Finish `/plan-ceo-review`** — it stopped right after the approach decision (D1-A), before
   mode selection and the 11-section review. Or skip it; the decision it existed to force is made.
3. **Build the app.** Lane A is clear. Design is approved and prototyped:
   `~/.gstack/projects/Jeffreys-World-nyc-crash-risk-forecast/designs/comparison-view-20260813/`
   — `prototype.html` is interactive (click rows, drag budget, `P` for presentation mode, `T` for
   theme). 12 tasks in `tasks-design-review-20260813-173803.jsonl`, 7 of them P1.
   Start with **D1 (fork DESIGN.md)** — no dependencies, and it unblocks all visual work.
4. **Fix the 6 prototype bugs** if you intend to demo the prototype rather than the real app.
   Full report with repro steps: `.gstack/qa-reports/qa-report-prototype-2026-08-13.md`.
   Worst two: the table clips its own columns at 1280×720 (709px table, 640px pane), and only
   6 rows of 100 are visible because names wrap to three lines.

## Open, unresolved

- **Tie blocks are real** — 14 units share the cut-off score at B=100. The "B of N tied"
  disclosure is still needed. This is the one surviving critical gap.
- **The why-panel sign inversion** — `is_highway` carries a negative coefficient (`README:444`).
  Emit fitted coefficients and render signed contributions, or the panel tells engineers the
  flag pushed a location up when it pushed it down.
- **Premise 5** (is budget denominated in locations?) and **Premise 6** (no practitioner
  consulted) — outreach messages are drafted and ready to send in
  `Practitioner_Outreach_Messages.md`. Sending one is the highest-value hour available.
- **Portfolio repo has a live contrast bug.** `DESIGN.md` specifies `--sev-high #B4600A`, which
  measures 4.26:1 and fails WCAG AA. Shipped `style.css` uses `#a45709` at 4.98:1 and is correct.
  Fix the doc, not the code. Different repo.
- **`gstack-slug` resolves wrong from this directory** (returns `Jeffreys-World-personal-website`),
  so gstack tools misfile learnings. Pin `SLUG` explicitly in every gstack bash call.

## Do not

- Do not adopt the percentile-within-type fix. It is measured and it is worse.
- Do not screenshot the prototype into the README. Its rows are synthetic. See TODOS P2.
