# Project Plan: NYC Crash Priority-Location Application

**Builds on:** [Motor-Vehicle-Collisions---Crashes-Dashboard](https://github.com/Jeffreys-World/Motor-Vehicle-Collisions---Crashes-Dashboard)
**Author:** Jeff
**Context:** Fellowship project; will be published to a personal website as a portfolio piece.
**Status:** Working plan — confirm changes before implementation.

**Revision history:**
- v1 — predictive hotspot tool
- v2 — reframed after external review to lead with a defensible finding (SPF + Empirical Bayes, exposure variable, audit framing)
- v3 (current) — restructured around the actual deliverable: a polished application, with the audit as its credibility layer rather than a separate output

---

## 1. What this is

An application a NYC DOT safety engineer uses to see where severe and fatal crashes concentrate across the city, and to produce a prioritized list of locations for a safety plan.

The distinguishing feature is *why the rankings can be trusted*: the locations are ranked using the road-safety field's standard method (Safety Performance Function + Empirical Bayes adjustment) rather than by raw crash counts — which corrects a known bias that raw-count rankings suffer from. The application shows that correction rather than hiding it.

**Two things this project is, in order of weight:**

1. **An application** (primary artifact) — explorable, polished, publishable to a website.
2. **A methodological argument** (credibility layer) — the ranking inside it is defensibly better than the obvious one, and the app demonstrates that in-line.

The scenario driving it is a designed one, documented in `docs/persona-validation.md` — grounded in published FHWA and NYC Vision Zero sources, with unverified assumptions stated plainly rather than assumed away.

## 2. Why this extends the existing dashboard well

The current dashboard established the hard part: a defensible, quality-audited dataset (812,318 rows for the 2019–2025 slice; 2,269,187 for the full 2012–2026 pull), with known issues handled — missing boroughs (30.5% of rows, 39.8% of deaths, 1.5x deadlier), padded/unpadded street-name duplicates, coordinate scrubbing, ragged API pagination.

Direct reuse:

- **Section 8 (hotspot density map)** → the application's primary map view
- **Section 9 (severity outliers)** → the KSI (killed or severely injured) outcome being ranked
- **Section 5 (contributing factors)** → model features and per-location explanation content

The throughline also carries: the dashboard found that the standard borough chart hides 40% of deaths. This application applies the same discipline one level up — the standard *ranking* method has a known bias too, and this one corrects it.

## 3. Value narrative (for the fellowship and the hiring manager)

- **It serves a decision, not a curiosity.** The output is a prioritized list an engineer could take into a planning cycle — not a gallery of charts.
- **It uses the field's real method.** SPF + Empirical Bayes is the Highway Safety Manual standard, not a generic ML pipeline pointed at a public dataset. That signals domain grounding immediately to anyone who knows the space, and is explainable to anyone who doesn't.
- **It shows its own correction.** Raw-count ranking and EB-adjusted ranking sit side by side in the interface, so the methodological point is visible in 10 seconds rather than buried in a write-up.
- **It states its limits.** Exposure-data gaps, low-sample locations, the hotspot/systemic scope boundary, and unverified persona assumptions are all documented rather than papered over.

## 4. Scope: systemic ranking with a hotspot correction (state this explicitly)

FHWA distinguishes two prioritization approaches:

- **Hotspot / site-specific** — rank locations by their own crash history.
- **Systemic** — target high-risk *roadway features* network-wide, reaching locations with no crash record.

**This project does both, and the split is structural rather than rhetorical.**

The **SPF is the systemic component**. Its predictors are `posted_speed`, `number_travel_lanes`, `streetwidth`, `is_highway`, and an imputation flag, plus a log-exposure offset (`src/config.py:181`). No crash-derived terms — they were removed deliberately, because feeding crash history into the SPF double-counts the history Empirical Bayes exists to weigh. Roadway characteristics applied network-wide regardless of crash record is FHWA's systemic approach by definition.

The **EB blend is the hotspot correction**, applied on top for the 13,712 units (6.2%) that have history worth correcting.

This is the hybrid the Highway Safety Manual prescribes, and the distinction carries the result. The headline +18.4pp at N=38,909 comes from ordering the 206,321 zero-history units, which only the systemic half can do. Restricted to units with history, lift is +2.1pp and does not clear the pre-registered 5pp bar — published in the README rather than omitted.

**On fundability:** because the zero-history reach is systemic, FHWA's documented acceptance of systemic analysis as HSIP justification is the applicable standard. See `persona-validation.md`. What remains unverified is whether NYC DOT specifically uses that pathway and what evidence it demands.

Naming this accurately is the credibility signal. Claiming hotspot-only would have been the more cautious-sounding framing and would have disclaimed the project's own finding.

## 5. The application

### 5.1 Core views

| View | Purpose |
|---|---|
| **Citywide hotspot map** | Where severe crashes concentrate; the entry point |
| **Ranked location list** | Intersections/corridors ordered by EB-adjusted risk |
| **Raw vs. adjusted toggle** | The methodological argument, made visible and interactive |
| **Location detail panel** | Why this location ranks where it does — crash mix, timing, contributing factors, exposure |
| **Budget scenario control** | "I can fund N projects" → the top N, with what they collectively represent |

### 5.2 Interaction principle

The engineer explores; the app supplies a trustworthy ordering and explains itself on demand. Nothing in the interface should require reading a methodology document first — the raw/adjusted toggle should teach the concept by being used.

### 5.3 Polish targets (these matter — this is a portfolio artifact)

- Fast initial load; no multi-second blank states
- Coherent visual language with the existing dashboard
- Legible on a laptop screen without zooming; degrades acceptably on tablet
- Every number labeled with its date range and source
- A short walkthrough GIF in the README for reviewers who won't run it

## 6. Method

### 6.1 Safety Performance Function

Poisson or negative binomial regression predicting expected KSI count per location from location characteristics and exposure. Negative binomial is the expected choice given overdispersion typical of crash counts.

### 6.2 Empirical Bayes adjustment

Ranking sites by observed crash count suffers **regression-to-the-mean**: some locations look dangerous from noise alone, and naive ranking systematically overstates their future risk. EB shrinks the observed count toward the SPF-predicted mean, weighted by reliability, producing a de-noised long-run risk estimate.

This is the standard correction in road-safety practice, and it is the single most important technical decision in the project. The application surfaces it directly via the raw/adjusted toggle.

### 6.3 Exposure (required)

Without traffic volume or segment length, any ranking confounds "busy" with "dangerous." NYC Open Data publishes traffic volume counts; coverage and join reliability at intersection/corridor level must be verified early (Section 9). If coverage is insufficient, use a documented fallback (segment length, road classification) and state the limitation in the interface, not only in the docs.

### 6.4 Validation

Time-based split only — fit on earlier years, evaluate against a later held-out period. A random split would leak future information and invalidate the forward-looking claim.

### 6.5 Evaluation

Does the EB-adjusted ranking's top-N concentrate more of the *subsequent* severe/fatal crashes than raw-count ranking over the same period? Report this honestly, including if the margin is small or mixed.

## 7. Data & features

**Unit:** named intersections and corridors (chosen over grid cells for interpretability — a ranked list of streets is directly actionable).

**Features per location-period:**

- Trailing crash count and severity mix (12/24 month windows)
- Contributing factor mix, including "Unspecified" treated as informative, not noise
- Temporal concentration (day-of-week, hour-of-day)
- Road type — limited-access highway vs. surface street, using the street list from the borough-gap analysis
- Borough/precinct where available
- **Exposure** — volume or segment length (Section 6.3)

**Outcome:** KSI count over the 12-month forward window (matches an annual planning cadence).

**Carried-forward caveats:** highway-heavy rows are 1.5–1.67x deadlier and must not be dropped; street names require the existing whitespace normalization or the same corridor double-counts.

## 8. Build sequence

The principle: get one thin path working end-to-end before widening anything. But unlike a pure research deliverable, interface quality is part of what's being evaluated here — so polish is interleaved, not deferred to a final phase.

**Phase 1 — Thin vertical slice**
1. Aggregate cleaned data to intersections/corridors
2. Verify exposure join feasibility; choose real variable or documented fallback
3. Fit SPF; apply EB adjustment
4. Backtest EB-adjusted vs. raw-count ranking on held-out period
5. **Checkpoint:** does the adjustment change the ranking, and does the change hold up? Record the answer either way — a small or mixed effect is a real result and still supports the application.

**Phase 2 — Application core**
6. Map view + ranked list wired to real model output
7. Raw vs. adjusted toggle
8. Location detail panel

**Phase 3 — Depth and polish**
9. Budget scenario control
10. Per-location explanation (plain language; SHAP or equivalent)
11. Visual and performance pass
12. Error analysis — does the model underpredict on highway corridors, echoing the borough-gap finding?

**Phase 4 — Publication**
13. Model card and limitations page (reachable from inside the app, not just the repo)
14. Tests for the feature and scoring pipeline; CI on push
15. README rewrite with the throughline; walkthrough GIF
16. Website write-up: the finding, the method, the application, in that order

## 9. Verify early (before Phase 1 step 3)

- **Exposure data**: does NYC's traffic volume dataset actually join to these locations at usable coverage? This is the highest-risk dependency in the plan.
- **Sample sufficiency**: how many intersections/corridors have enough KSI history for a stable EB estimate? This sets the floor for what the app can rank, and belongs in the model card as a stated reliability threshold.
- **NYC published priority list**: if DOT's own priority-location list is publicly available with a documented methodology, comparing against it strengthens the argument considerably. If it isn't cleanly available, the raw-count comparison stands on its own — the project does not depend on this.

## 10. Deliverables checklist

- [ ] Deployed application (Streamlit Community Cloud or equivalent)
- [ ] `docs/persona-validation.md` — designed scenario, documented sources, stated unknowns
- [ ] Model card — method, validation, limitations, reliability threshold
- [ ] Backtest result, reported honestly
- [ ] Tests + CI
- [ ] README with project throughline and walkthrough GIF
- [ ] Website write-up for the portfolio audience

---

*Working draft. Confirm changes before implementation.*
