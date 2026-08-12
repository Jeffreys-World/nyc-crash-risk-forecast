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

> **PENDING — the pipeline is still being built. No backtest has been run.**
>
> This section will carry one number and one chart. The bar that number must clear is
> defined below, and was written before the number existed. See
> [The bar, set in advance](#the-bar-set-in-advance).

Nothing in this README claims a result yet. When the backtest runs, the number goes here
whether or not it is flattering.

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
| Street centerline | *pending selection* | The candidate universe and segment length |

**Snapshot vintage:** `PENDING — no snapshot pulled yet`
**Row counts:** `PENDING`
**Centerline source:** `PENDING — candidates 3mf9-qshr, inkn-q76z, and DCP LION are not yet
schema-inspected. Whichever is chosen gets pinned here with its vintage.`

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

.venv/bin/python -m pytest                      # 184 tests, no network needed
.venv/bin/python scripts/pull_snapshots.py      # data/raw/<date>/*.parquet + manifest.json
```

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
  invisible to this and to DOT alike.
- **Nothing about cyclists or motorists.** Pedestrian mode only, to match DOT's list.

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

## Status

Approach A — the audit slice. Deliberately narrow: produce one defensible finding, then decide
whether it is worth wrapping in a tool.

| | |
|---|---|
| ✅ Scope and method settled | Office-hours design review, eng review (7 findings, all folded in) |
| ✅ Pipeline built and tested | T1–T10: snapshot pull, universe, features, SPF, EB, backtest. 184 tests green |
| ⛔ Blocked | Centerline source not pinned, so the unit universe cannot be built from real data |
| ⏭ Next | Inspect the centerline candidates, pin one, pull snapshots, run the backtest |
| ⏸ Gated on a result | Streamlit page, budget slider, SHAP explainer, CI (Approach B) |
| ⏸ Gated on a result | The named-streets counterfactual (Approach C) |
| 📋 Deferred, written up | Traffic-volume exposure ([TODOS.md](TODOS.md)) |

The pipeline is code-complete and its guards are verified against a synthetic fixture
city. It has never been run against real NYC data, because the centerline source is still
unpinned — the three candidates (`3mf9-qshr`, `inkn-q76z`, DCP LION) have not been
schema-inspected. `src/config.py` sets `CENTERLINE_SOURCE = None` and the pull script skips
it with a warning rather than substituting a guess.

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
