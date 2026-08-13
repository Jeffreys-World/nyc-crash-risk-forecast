# NYC Crash Risk Forecast

**Does New York's official pedestrian-safety priority list miss locations that a standard
safety model catches?**

NYC DOT publishes a Vision Zero priority list of corridors and intersections, ranked by
historical casualty counts. Ranking by raw historical counts is a method the traffic-safety
literature specifically warns about: it is biased by regression to the mean. A location that
had a terrible year gets picked; a location that is genuinely dangerous but had a quiet year
does not.

This repo rebuilds the ranking using the method the field's own standard prescribes — a
negative-binomial Safety Performance Function blended with observed counts via Empirical
Bayes — and backtests both against what actually happened on held-out years.

---

## Headline result

**Count-based ranking is blind below its own noise floor, and that is where most of New
York is.**

Only **13,712 of 220,033 units (6.2%)** had a single pedestrian casualty in the trailing
36 months. The other 206,321 are tied at zero, so a ranking built on historical counts
cannot order them at all — it is picking at random. Those tied-at-zero locations went on
to produce **7,408 casualties, 41% of everything that happened in 2024–2025.**

A Safety Performance Function can order them, because it reads the road rather than the
history.

Scored on 2024–2025, each ranking selecting 38,909 locations (the footprint of DOT's
published list):

| Ranking | Casualties captured | Share of 18,059 |
|---|---:|---:|
| **R1** DOT's published Vision Zero list | 8,802 | **48.7%** |
| **R2** raw trailing casualty count | 11,578 | **64.1%** |
| **R3** Empirical Bayes (SPF + observed) | 14,905 | **82.5%** |

**R3 − R2 = +18.4pp, 95% CI [+17.5, +19.3].** That clears the
[pre-registered bar](#the-bar-set-in-advance) of ≥5pp with a CI excluding zero.

### The number that keeps that honest

**The advantage is almost entirely about locations with no crash history, and it scales
with how many you have to rank.**

| N selected | R2 raw count | R3 Empirical Bayes | Lift | |
|---:|---:|---:|---:|---|
| 5,000 | 37.1% | 38.9% | +1.8pp | |
| 13,712 | 59.0% | 61.0% | **+2.1pp** | every unit with any history |
| 20,000 | 60.2% | 68.1% | +7.9pp | |
| 38,909 | 64.1% | 82.5% | **+18.4pp** | DOT's list size — the pre-registered N |
| 60,000 | 68.7% | 89.5% | +20.8pp | |

Where both methods have real information — ranking only the 13,712 locations that have
any crash history — **Empirical Bayes adds 2.1pp and would not have cleared the bar.**

The +18.4pp headline is real at the pre-registered N, and it is not a general claim that
EB beats counting by eighteen points. It is a specific claim: *once you must rank more
locations than you have crash history for, counting stops being a method and a model
starts earning its keep.* That is the regression-to-the-mean argument the Highway Safety
Manual makes, measured on real data instead of asserted.

Anyone quoting the 18.4 without the 2.1 is quoting it wrong.

### What it actually ranks

Top of the list, scored before the holdout ([full 50](data/processed/top-50-ranked.csv)):

| # | Location | Borough | Trailing 36mo | On DOT's list |
|---|---|---|---:|---|
| 1 | E Fordham Rd & Webster Ave | Bronx | 21 | yes |
| 2 | Broadway & W 204 St | Manhattan | 23 | **no** |
| 3 | Devoe Park Path & University Ave & W Fordham Rd | Bronx | 18 | yes |
| 4 | Lenox Ave & W 125 St | Manhattan | 20 | yes |
| 5 | Frederick Douglass Blvd & W 145 St | Manhattan | 16 | yes |

**At the top, the model and DOT mostly agree** — only 4 of the top 50 are absent from the
published list, and those 4 carry 9 of the top 50's 254 holdout casualties. That agreement
is a validity check, not a disappointment: a screening method that disagreed with DOT about
Fordham Road would be suspect.

The divergence lives in the tail, which is the same story the N-sweep tells. Note also that
Broadway & W 204 St, the highest-ranked location DOT omits, had 23 trailing casualties and
1 in the holdout — regression to the mean landing on the model's own pick.

---

## Why this is not a greenfield ML problem

Crash hotspot ranking is a solved-form problem with a codified standard. The AASHTO
*Highway Safety Manual* specifies **Empirical Bayes network screening**: fit a negative-binomial
Safety Performance Function (SPF) to predict expected crashes from exposure and site
characteristics, then blend that prediction with the site's observed count, weighted by the
model's dispersion parameter.

```
w  = 1 / (1 + k · P)          k = NB dispersion, P = SPF-predicted count
EB = w · P + (1 − w) · observed
```

The blend is the entire point. A site with few observed crashes gets pulled toward the model
prediction; a site with many gets to keep its own evidence. That is the correction for
regression to the mean that raw-count ranking lacks.

Two consequences drive the design:

1. **An exposure term is mandatory.** Without one, the model ranks *busy*, not *dangerous*.
   This slice uses **segment length**. Traffic volume is the better exposure term and is
   filed as the P2 item in [TODOS.md](TODOS.md).
2. **Naming the method matters.** SPF and EB are the vocabulary a transportation audience
   already uses. Inventing a bespoke ML pipeline here would be a worse answer to a question
   the field has already standardized.

---

## What gets compared

Three rankings, scored against the same held-out years:

| | Ranking | What it represents |
|---|---|---|
| **R1** | DOT's published Vision Zero priority list | The real, shipped artifact |
| **R2** | Raw trailing casualty count | The naive baseline; also DOT's method, reimplemented |
| **R3** | Empirical Bayes (SPF + observed blend) | The HSM-standard method |

R2 exists so the comparison is honest in both directions. If R3 does not beat R2, the
Empirical Bayes machinery bought nothing, and this README will say that.

### Two selection regimes, both reported

DOT does not rank citywide. It ranks **within borough** and stops at a cumulative share of
that borough's casualties (corridors 50%, intersections 15%). A citywide top-N model scored
against a borough-stratified list would win partly by construction — the model concentrates
its picks where density is highest while DOT is obliged to spend picks in every borough.

- **Regime A — borough-stratified.** Model follows DOT's own selection rule. Fair against
  the real published artifact.
- **Regime B — citywide top-N.** All three rankings select citywide, N fixed to the size of
  the published list. Fair method-versus-method.

Both get reported. A result that only appears in one regime is a result about the selection
rule, not about the model.

**Measured, 2024–2025 holdout:**

| | R2 raw count | R3 Empirical Bayes | Lift |
|---|---:|---:|---:|
| Regime A, borough-stratified (1,347 / 1,421 units) | 9.9% | 11.0% | +1.1pp |
| Regime B, citywide top-N (38,909 units) | 64.1% | 82.5% | +18.4pp |

The two regimes select wildly different numbers of locations, because DOT's stopping rule
applied to *this* casualty distribution stops at ~1,400 units, not 38,909. That makes the
regimes non-comparable to each other, and it is another face of the same finding: at
~1,400 selections both methods are working inside the region where history exists, and the
gap nearly vanishes.

### What about DOT's actual list?

R1 captures **48.7%**, below the 64.1% a raw-count ranking of the same size achieves.
That comparison is **confounded by design and must not be read as "DOT is worse."**

Vision Zero priority locations were selected *in order to receive* Street Improvement
Projects. 61,864 units carry an SIP, and treatment has to be placed in time rather than
just flagged: SIP completion dates in this snapshot run to **2026-05-29**, so 4,092 units
were rebuilt *during or after* the window they are being scored on.

| Group | Units | Holdout casualties | R1 capture |
|---|---:|---:|---:|
| Treated before the holdout | 21,576 | 9,179 | **60.7%** |
| Treated during the holdout | 1,580 | 597 | 56.3% |
| Untreated | 15,753 | 8,283 | **34.9%** |

DOT's list overlaps far more strongly with the locations that got rebuilt, which is
exactly what you would expect from a list whose purpose was to direct construction. The
mid-window group is reported separately rather than folded into either side: a street
rebuilt in 2025 is genuinely neither treated nor untreated for a 2024–2025 outcome.

So the honest reading is: a list selected to be *fixed* is being scored on what happened
*after it was fixed*. This project cannot separate "the ranking was wrong" from "the
ranking was right and the intervention worked," and it does not claim to.

---

## The bar, set in advance

Pre-registered before running the backtest, so the threshold cannot be moved to fit the
number that comes out.

**R3 beats R2 only if both hold on the 2024–2025 holdout:**

1. The Empirical Bayes ranking's casualty capture rate exceeds the raw-count ranking's by
   **at least 5 percentage points**, and
2. the **bootstrap 95% confidence interval** on that difference **excludes zero**.

**If that bar is not cleared, the finding is the negative result**, published with the same
prominence: on this dataset, at this unit of analysis, with length as the only exposure term,
the HSM-standard correction did not measurably improve on raw-count ranking. That outcome is
reported, not buried, and the most likely explanation — missing traffic-volume exposure — is
already written down as the next step.

**The comparison against R1 is reported split by treated versus untreated** (see below), never
as a single headline number.

---

## Does the headline survive its own judgement calls?

Three distances sit underneath every number above, and all three were chosen by
judgement rather than measured:

| Constant | Value | What it decides |
|---|---:|---|
| `MAX_JOIN_DISTANCE_FT` | 150 ft | How far a crash may be from a street and still attach to it |
| `INTERSECTION_RADIUS_FT` | 100 ft | How close to a junction a crash must be to count as intersection-related rather than mid-block |
| `VZV_BUFFER_FT` | 50 ft | How far a DOT priority feature reaches when deciding which units are on the published list |

Nothing distinguishes 150 ft from 100 or 250 except that someone had to pick one. So
`scripts/radius_sensitivity.py` re-runs the entire backtest across all three knobs, one
at a time, holding the other two at their published values
([full table](data/processed/radius-sensitivity.md)):

| Knob | Swept (published value in bold) | Lift at each |
|---|---|---|
| Corridor join distance | 100 / **150** / 250 ft | +18.6 / **+18.4** / +18.4pp |
| VZV label buffer | 25 / **50** / 100 ft | +17.1 / **+18.4** / +19.6pp |
| Intersection radius | 50 / **100** / 150 ft | +16.1 / **+18.4** / +19.9pp |

**The lift spans +16.1pp to +19.9pp. Every setting clears the pre-registered 5pp bar,
and every confidence interval excludes zero.** The finding is not an artifact of three
unexamined numbers.

(Seven distinct configurations, shown as nine rows: the published setting sits on all
three axes at once, so it appears in each block and is run once.)

The three knobs are not equally important, and the reasons are worth separating:

- **The corridor join distance barely registers.** It only decides whether a crash far
  from any street is assigned at all — 1,947 crashes out of 1.4 million. Widening it to
  250 ft moves the lift by 0.06pp. This one was never load-bearing.

- **The VZV buffer moves the lift, but through N, not through labeling.** A wider buffer
  puts more units on DOT's list — 35,461 at 25 ft, 38,909 at 50 ft, 43,111 at 100 ft —
  and every ranking is sized to match. The lift it produces at each N tracks
  [the N-sweep](#the-number-that-keeps-that-honest) already published above. This is the
  same finding arriving through a different door, not a new sensitivity.

- **The intersection radius is the one that genuinely matters.** It changes nothing else:
  N stays at 38,909, the same 1,405,552 crashes are assigned, and the capture-rate
  denominator stays at 18,059. Only *which unit* each crash lands on moves. Widening it
  from 50 ft to 150 ft takes the lift from +16.1pp to +19.9pp — roughly 2pp per 50 ft.
  A wider radius pulls casualties off the 136,537 segments and onto the 83,496 nodes,
  concentrating them into a smaller universe. Both rankings capture more (R2 goes
  61.9% → 66.5%), and the Empirical Bayes ranking gains faster than the raw count does.

**The published setting is not the flattering one.** On two of the three axes, a
different and equally defensible choice would have produced a *larger* headline: 150 ft
on the intersection radius gives +19.9pp, and a 100 ft VZV buffer gives +19.6pp. 150 /
100 / 50 sits in the middle of every range, which is what it should look like when the
values were picked before the result existed.

What this does *not* license is quoting +18.4 as a constant. The qualitative claim —
Empirical Bayes substantially out-captures raw counting at DOT's list size — holds
across every setting tested. The specific decimal carries about ±2pp of dependence on
one judgement call about what "at an intersection" means.

---

## Three honesty constraints built into the method

### 1. The label is pedestrian casualties, not KSI

DOT's Vision Zero list is ranked by **KSI** (killed *and severely injured*). The public crash
dataset (`h9gi-nx95`) carries `number_of_pedestrians_killed` and
`number_of_pedestrians_injured` and **no injury-severity field at all**. KSI is therefore not
reproducible from public data.

This project predicts **pedestrian casualties** = killed + injured, and names it that
everywhere. A calibrated "estimated KSI" was considered and rejected: it puts a fitted knob
inside the target variable, which is the one place a knob must never go.

The consequence is real and stated up front: R1 was built to a target this project cannot
exactly reproduce. Any R1 comparison is method-versus-method under a related but different
label.

### 2. DOT's priority locations were treated

Vision Zero priority locations were selected **in order to receive** Street Improvement
Projects. Their later casualty counts reflect that intervention. A raw head-to-head is
therefore uninterpretable in both directions — if DOT's locations improved, the ranking looks
wrong precisely because it worked.

SIP Corridors (`wqhs-q6wd`) and SIP Intersections (`79sh-heg3`) are joined in, each priority
location is tagged treated-or-not and when, and the comparison is reported split accordingly.
The confounder becomes a second finding instead of a footnote.

### 3. The unit of analysis is a segment, not a street name

A VZV corridor is a *segment* — Broadway from W 135th to W 153rd, not all of Broadway. Joining
on street name would assign every Broadway crash to that corridor and inflate priority-corridor
counts in the model's favor.

The candidate universe is the **citywide street-centerline network**. VZV joins onto it
spatially as a labeled subset. This also supplies segment length as a free exposure term, and
it is the only structure that can answer the actual question: *what did DOT's method miss?*

---

## Data and provenance

Every input is a live, mutating API, and NYPD amends past crash records retroactively — so even
a date-filtered query drifts between runs. One pull script writes **dated snapshots**; the
pipeline reads only snapshots, never the API.

| Source | Socrata ID | Role |
|---|---|---|
| Motor Vehicle Collisions — Crashes | `h9gi-nx95` | Crashes and the casualty label |
| VZV Priority Corridors | `kdda-2wcy` | R1, corridor half |
| VZV Priority Intersections | `2nj7-jxah` | R1, intersection half |
| Street Improvement Projects — Corridors | `wqhs-q6wd` | Treatment flag and date |
| Street Improvement Projects — Intersections | `79sh-heg3` | Treatment flag and date |
| Street centerline | `inkn-q76z` | The candidate universe, segment length, borough, road class |

**Snapshot vintage:** `2026-08-13`

The committed artifacts were regenerated from a fresh pull on 2026-08-13, a day after
the run the result was first published from, on a rebuilt environment with newer pandas,
numpy, and scipy. Every number below, and every number in the headline, came back
identical; the top-50 ranking is the same 50 units in the same order, with Empirical
Bayes estimates differing by at most 4e-13. Two independent pulls agreeing to the row is
weak evidence about the *method* and strong evidence about the *pipeline*: nothing in it
depends on the machine it ran on. It is not the independent re-derivation still listed in
[TODOS.md](TODOS.md) — that needs code that shares nothing with `src/backtest.py`.

| | |
|---|---:|
| Crashes pulled (2016-01-01 onward) | 1,541,146 |
| Crashes assigned to a unit | 1,405,552 (91.2%) |
| Dropped — no coordinates | 125,910 |
| Dropped — at (0, 0) | 7,588 |
| Dropped — outside NYC | 149 |
| Dropped — >150 ft from any street | 1,947 |
| Centerline segments | 122,244 |
| Units in the universe | 220,033 (136,537 corridors, 83,496 intersections) |
| VZV features matched | corridors 199/199, intersections 303/304 |
| SIP records | 1,420 (1 undated, excluded) |
| Holdout casualties, 2024–2025 | 18,059 |

**A note on the four dataset IDs.** The VZV and SIP resources were originally pinned to
`kdda-2wcy`, `2nj7-jxah`, `wqhs-q6wd`, and `79sh-heg3`. Those IDs exist and report row
counts, which is why they passed a existence check — but they are
`visualization_canvas_map` views with **zero API-accessible columns**, and every row comes
back as `{}`. The real datasets are the four in the table above. Verifying that a dataset
exists is not the same as verifying its data can be read.

**8.2% of crashes carry no coordinates.** That is a second data-quality gap beneath the
borough gap this project descends from, and for a geometry-based model it is the binding
one: a crash with no latitude cannot attach to any street, so it is invisible here. It is
counted, not hidden.

Raw snapshots are gitignored. The small aggregated intermediate is committed so the headline
is checkable without a full re-pull.

---

## Reproducing this

```bash
git clone https://github.com/Jeffreys-World/nyc-crash-risk-forecast.git
cd nyc-crash-risk-forecast

# environment — uv fetches its own Python, so no system Python is required
uv venv --python 3.12
uv pip install -e ".[dev]"

# credentials — needed only for the data pull, not for the tests
cp .env.example .env      # then paste your own Socrata app token into it

.venv/bin/python -m pytest                      # 249 tests, no network or token needed
.venv/bin/python scripts/pull_snapshots.py      # data/raw/<date>/*.parquet + manifest.json
.venv/bin/python -m src.pipeline                # the headline, from the snapshot
.venv/bin/python scripts/radius_sensitivity.py  # does the headline survive other radii?
```

On Windows the interpreter is at `.venv/Scripts/python.exe`; everything else is the same.

The same two environment commands run in CI on every push and pull request
([`.github/workflows/test.yml`](.github/workflows/test.yml)), on Python 3.11 and 3.12.
If the reproduction path above stops working, CI is what finds out rather than the next
reader.

Get a Socrata app token from
[NYC Open Data developer settings](https://data.cityofnewyork.us/profile/edit/developer_settings).
The pull runs without one, but anonymous requests are throttled hard and this project
walks roughly 1.5M crash rows. `.env` is gitignored; `.env.example` is the committed
template and holds no real value.

The test suite runs green on a clean clone with no data pulled, because every stage is
covered against a synthetic fixture city. That is deliberate: the guards are verifiable
before anyone spends an API call.

Cloning this repo, running the pull, and running the pipeline should then reproduce the
headline number exactly. That reproduction path is the only real proof the result is not
a story.

---

## What is NOT claimed

- **No causal claim.** This ranks risk. It does not establish that a redesign at a
  high-ranked location would prevent a casualty.
- **No KSI claim.** The label is killed + injured pedestrians. See constraint 1.
- **No claim that DOT is wrong.** DOT optimizes for constraints this model does not carry —
  equity across boroughs, construction feasibility, community process, budget cycles. "The
  model ranked a location DOT did not" is a finding about ranking methods, not about judgment.
- **No traffic-volume exposure.** Segment length only. A long quiet residential street and a
  long arterial currently get the same exposure. This is the single largest known gap and is
  written up in [TODOS.md](TODOS.md).
- **No completeness claim on the crash data.** Unreported and under-reported crashes are
  invisible to this and to DOT alike. 8.2% of records carry no coordinates and cannot be
  placed on any street.
- **Nothing about cyclists or motorists.** Pedestrian mode only, to match DOT's list.
- **No general claim that Empirical Bayes beats counting by 18 points.** The lift is
  2.1pp where both methods have crash history to work with. See
  [the N-sweep](#the-number-that-keeps-that-honest).
- **No claim that +18.4 is radius-free.** Across the seven settings swept, the lift runs
  from +16.1pp to +19.9pp, and about 2pp per 50 ft of that depends on where the boundary
  between "at an intersection" and "mid-block" is drawn. Every setting clears the bar, so
  the finding holds; the decimal is a measurement at one defensible choice, not a
  constant. See [the sensitivity sweep](#does-the-headline-survive-its-own-judgement-calls).
- **No calibration claim.** The SPF is fit on 36-month trailing counts and scored against
  a 24-month holdout, so predicted counts sit on a longer window than observed ones. The
  model over-predicts by roughly the window ratio: observed/predicted is 0.56 on surface
  streets and 0.78 on highways, against 24/36 = 0.67. **Ranking is unaffected** — the
  scale factor is monotone, and a ranking is all this project claims — but the predicted
  counts are not expected casualties for the holdout window and must not be read as such.
- **Error analysis by road class, measured.** The model does *not* repeat the parent
  dashboard's blind spot in reverse. Highways are 26.9% of units but carry only 6.4% of
  pedestrian casualties, and the Empirical Bayes ranking selects 4.9% highway units
  against the raw count's 20.2% — the model down-weights highways, correctly, because
  pedestrians are rarely struck on them. The borough bar chart hid highway deaths by
  accident; this model sets them aside on purpose, for a label that genuinely excludes them.
- **No claim that highways are safe.** `is_highway` carries a negative coefficient here
  because pedestrians are rarely struck on limited-access roads, not because those roads
  are safe. This project's label is pedestrian casualties, so the highway finding that
  motivated the parent dashboard — that borough-less rows are 1.67x deadlier overall —
  does not transfer to it. The method carries over; that specific finding does not.

---

## Known silent-failure guards

Three failure modes in this pipeline produce plausible-looking wrong numbers rather than
errors. Each has an explicit guard and a test:

| Failure | Why it is silent | Guard |
|---|---|---|
| CRS mismatch | Buffering in WGS84 degrees while intending feet is wrong by ~364,000x and throws nothing | Assert a projected EPSG before any distance or buffer call |
| `log(0)` exposure offset | A zero-length segment makes the offset `-inf`; the NB fit diverges or fails late | Guard and raise before fitting |
| Zero-denominator capture rate | A borough with no holdout casualties yields a NaN that propagates into the headline | Explicit zero-denominator branch |

---

## The discarded run

The first run against real data, on 2026-08-12, reported **R3 beating the baseline by
+16.7pp with a CI excluding zero** — clearing the pre-registered bar. It was thrown away.
It is recorded here because a discarded run is part of the record, and because the bar
existing in advance is what made throwing it away possible rather than tempting.

Three defects, each of which alone invalidates it:

1. **The SPF was degenerate.** Fitted intercept **−29.5** with a `night_share` coefficient
   of **+21.9**, and **163,556 of 220,033 units** predicted at or below 1e-9. The
   crash-derived predictors were `0.0` both for a unit with no crashes and for a unit
   whose crashes were all in daylight, so the model learned "night_share > 0 means this
   unit had a crash." It was a crash-presence detector wearing an SPF's clothes.
   Underneath sat a methodological error: an HSM Safety Performance Function predicts from
   *road characteristics*; crash history belongs in the Empirical Bayes blend, and feeding
   it to both counts it twice.

2. **The tie-break was rigging the baseline.** Unit IDs are `C…` for corridors and `I…`
   for intersections, and ties sorted on `unit_id` ascending. Since most units are tied at
   zero, the naive baseline spent its entire quota on alphabetically-early corridors —
   which hold 14% of casualties against the intersections' 86%. The measured lift was
   partly an artifact of the alphabet.

3. **DOT's list could not capture its own casualties.** VZV corridor labels stopped at
   segments, but crashes within 100 ft of a junction are assigned to the *node*, and 86%
   of pedestrian casualties happen at intersections. R1's implausible 11.9% measured the
   labeling, not DOT.

A fourth issue surfaced while fixing these: with site characteristics as predictors, the
negative-binomial fit stopped converging, and the convergence guard refused to return
parameters. Starting the search from a Poisson fit (NB2's limit as dispersion → 0) fixed
it. Dispersion went from 2.53 to 5.70 — the degenerate model had been understating
overdispersion, which is the one parameter the entire Empirical Bayes weight depends on.

## Status

Approach A — the audit slice. Deliberately narrow: produce one defensible finding, then decide
whether it is worth wrapping in a tool.

| | |
|---|---|
| ✅ Scope and method settled | Office-hours design review, eng review (7 findings, all folded in) |
| ✅ Pipeline built and tested | Snapshot pull, universe, features, SPF, EB, backtest. 249 tests green, run in CI on 3.11 and 3.12 |
| ✅ Run against real data | Snapshot 2026-08-13, result above, one earlier run discarded |
| ✅ Sensitivity to the join radii | All three swept; see below. The headline does not turn on them |
| ⏭ Next | Independent re-derivation; SPF window aligned to the holdout |
| ⏸ Gated | Streamlit page, budget slider, SHAP explainer (Approach B) |
| ⏸ Gated | The named-streets counterfactual (Approach C) |
| 📋 Deferred, written up | Traffic-volume exposure ([TODOS.md](TODOS.md)) |

Approach B and C are gated deliberately. There is no point wrapping a finding in a tool before
knowing whether the finding exists.

---

## Background

Built on [Motor-Vehicle-Collisions---Crashes-Dashboard](https://github.com/Jeffreys-World/Motor-Vehicle-Collisions---Crashes-Dashboard),
which established the cleaned dataset and the project's throughline: the obvious chart hides
the real risk. That dashboard found the standard borough bar chart drops 30.5% of rows and
39.8% of deaths, and that the dropped rows skew deadlier.

This project applies the same discipline to a ranking instead of a chart — including the part
where the answer is allowed to be "the standard method was already fine."
