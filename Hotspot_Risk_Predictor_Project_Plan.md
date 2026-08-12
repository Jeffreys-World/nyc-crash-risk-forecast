# Project Plan: NYC Crash Hotspot Risk Predictor

**Builds on:** [Motor-Vehicle-Collisions---Crashes-Dashboard](https://github.com/Jeffreys-World/Motor-Vehicle-Collisions---Crashes-Dashboard)
**Author:** Jeff
**Status:** Draft for discussion — nothing below is final

---

## 1. The problem statement

A NYC DOT engineer, at the start of quarterly budget planning, wants to know which intersections or corridors are most likely to produce a severe or fatal crash in the next 12 months, so they can prioritize a fixed list of street-redesign projects instead of reacting after the next death.

Today, Vision Zero-style prioritization is largely reactive: a location gets attention after a fatality, a public petition, or a news story. A ranked, forward-looking risk score turns a fixed and limited safety budget into a measurably better-targeted one.

## 2. Why this extends the existing dashboard well

The current app already established the hard part — a defensible, documented, quality-audited dataset (812,318 rows for the 2019–2025 slice; 2,269,187 for the full 2012–2026 pull) with known issues handled: missing boroughs (30.5% of rows, 39.8% of deaths — deadlier on average), padded/unpadded street name duplicates, null coordinate scrubbing, and ragged API pagination.

This project reuses that cleaned data and three existing visualizations directly:

- **Section 8 (hotspot density map)** → the spatial unit of analysis (~100m cells)
- **Section 9 (severity outliers)** → the fatal/severe crash label to predict
- **Section 5 (contributing factors)** → candidate model features

It also converts the project's core theme — "the obvious chart hides the real risk" — from a descriptive finding into a predictive one, which is the difference between a data-analyst portfolio piece and a data-scientist one.

## 3. Business value (for the hiring-manager narrative)

- **Decision, not description.** The existing charts show what happened. This tool tells a budget-holder what to do next, ranked and quantified — the kind of output that maps directly to a real DOT resource-allocation workflow.
- **Backtestable claim.** Because the label (fatal/severe crash) is historical, we can retroactively ask: "If this model had ranked locations in 2022, how many of 2023's severe crashes would have occurred at its top-N flagged locations?" That's a concrete, defensible ROI number to put in front of a hiring manager — far stronger than "the model has 0.81 AUC."
- **Data-quality lineage carries through.** Because the missing-borough / geocoding-bias finding is already proven, the risk model can honestly state its blind spots (e.g., limited-access highways may be under-represented in training data) rather than presenting false confidence — a maturity signal for a DS role.

## 4. Data & feature plan

**Unit of analysis:** ~100m spatial grid cell (reuse existing hotspot binning), by month.

**Candidate features per cell-month:**
- Trailing crash count and severity mix (e.g., last 12/24 months)
- Contributing factor mix (share of "Unspecified," "Driver Inattention," etc.)
- Day-of-week / hour-of-day crash concentration (from the existing heatmap)
- Road type proxy (limited-access highway vs. surface street, using the street-name list already identified in the borough-gap analysis)
- Borough / precinct (where available)

**Target:** binary or count outcome — probability (or expected count) of a severe/fatal crash in the next 12 months for that cell.

**Known data caveats to disclose up front (already documented in the repo):**
- Borough-missing rows skew toward highways and are 1.5–1.67x deadlier — so any model must include highway segments explicitly, not drop them the way the "obvious" bar chart does.
- "Unspecified" is the top contributing factor — treat as a real, informative sparse category, not noise to drop.

## 5. Modeling approach

1. **Baseline (interpretable):** Poisson or negative binomial regression on crash counts per cell-month — appropriate because crash counts are classic overdispersed count data. This is the model to lead with when explaining results to a non-technical stakeholder.
2. **Comparison model:** Gradient-boosted trees (XGBoost or LightGBM) on the same features, to test whether nonlinear interactions meaningfully improve ranking quality.
3. **Validation:** Time-based split only — train on earlier years, test on the most recent held-out period. A random split would leak future information into training and invalidate the forecast claim; this is worth calling out explicitly in the writeup as a modeling-rigor signal.
4. **Metric:** Precision/recall at top-N (since the deliverable is a ranked shortlist, not a calibrated probability), plus the backtest ROI number described in Section 3.

## 6. Deliverable shape

- A ranked table/map of the top-N highest-predicted-risk locations, each with its predicted risk score and the top contributing features (via feature importance or SHAP).
- A simple "budget scenario" control: "You can fund 20 projects this year" → returns the top 20 locations ranked by predicted risk, so the tool answers a real constrained-resource question rather than just producing a leaderboard.
- A short model card / caveats section (reusing the "What is NOT claimed yet" tone already established in the repo's README) stating what the model does not know — e.g., unrecovered-borough locations, reporting gaps, small-sample cells.

## 7. Suggested milestones

| Stage | Notes |
|---|---|
| Define grid cell + monthly aggregation pipeline | Extends `scripts/clean_crash_data.py` |
| Feature engineering table (cell-month level) | New script, e.g. `scripts/build_risk_features.py` |
| Baseline Poisson/NB model + time-based validation | Establish honest baseline first |
| Gradient-boosted comparison model | Only after baseline is documented |
| Backtest ROI simulation | The single strongest hiring-manager slide |
| Streamlit page: ranked list + budget-scenario control | New page/tab in existing app |
| Model card / caveats writeup | Match README's existing rigor and tone |

## 8. Decisions

- **Spatial unit:** Aggregate to named intersections/corridors rather than raw ~100m grid cells. Less granular to model, but far more interpretable and directly actionable for DOT — a ranked list of streets/intersections is something a budget-holder can act on without translation.
- **Borough recovery sequencing:** No hard dependency — implement the point-in-polygon borough recovery before or after the first model pass, whichever is more convenient given how the build unfolds. Worth revisiting once we're closer to feature engineering, since intersection-level aggregation may reduce how much the borough field matters directly (street/corridor identity may already substitute for it).
- **Forecast window:** 12 months. Matches the DOT budget-planning cadence the persona is built around (quarterly planning, annual project slate).

## 9. Updated feature/unit note

Since the spatial unit is now named intersections/corridors rather than grid cells, Section 4's feature list carries over unchanged, but aggregation should key on street/intersection name (with the padded/unpadded name-cleaning already handled in `scripts/clean_crash_data.py`) rather than lat/long bins. The hotspot map (Section 8 of the dashboard) can still inform which corridors matter most visually, even though the model's unit of analysis is now named locations.

## 10. Enhancements to increase impact

All of the following are in scope. They're grouped so build order is clear — rigor and product polish sit on top of the core model from Sections 4–6; storytelling wraps the whole project once the rest exists.

### 10.1 Rigor signals (data science credibility)

- **Model card**, matching the "What is NOT claimed yet" honesty already established in the repo README: calibration plot, confidence interval on the backtest ROI number, and an explicit list of blind spots (low-sample corridors, recency bias, any location type still under-covered).
- **Naive-baseline comparison**: rank locations by raw trailing crash count (no model) and show this side by side with the model's backtest performance. If the model doesn't clearly beat the naive baseline, that needs to be known and disclosed before anyone else finds it; if it does, it becomes the headline result.
- **Error analysis**: characterize where the model is wrong and why — specifically checking whether it systematically underpredicts on limited-access highway corridors, which would directly echo the original borough-gap finding and tie the whole project into one coherent investigation rather than two separate pieces of work.

### 10.2 Product polish (make it feel real, not academic)

- **Interactive budget slider** in the Streamlit app: drag project count N (e.g., 5–50) and watch the ranked list and map update live.
- **Per-location "why this ranking" explainer** (SHAP-based, plain language) — e.g., "ranked #3 mainly due to high injury rate at this intersection during evening rush hour."
- **Before/after counterfactual framing**: what today's reactive approach would have funded historically vs. what the model would have funded, and the resulting fatality-prevention gap between the two, shown visually.

### 10.3 Engineering maturity

- **Tests** for the new feature-engineering and scoring pipeline, extending the existing `tests/` pattern already in the repo.
- **Documented validation reasoning**: a short written explanation of the time-based split choice and why a random split would have leaked future information — reviewers with ML backgrounds specifically look for this.
- **CI via GitHub Actions** running tests on push (if not already configured), so the repo reads as more than notebooks.

### 10.4 Storytelling / presentation

- **Non-technical one-page executive summary**, alongside the technical model card, aimed at a non-technical stakeholder reader.
- **Short recorded walkthrough or GIF** in the README demonstrating the budget-slider interaction, since many reviewers won't run the app themselves.
- **Explicit project throughline in the README**, tying the model back to the founding insight: this project started by finding that the standard borough bar chart hides 40% of deaths; the model, its caveats, and the blind-spot analysis all follow that same discipline of checking what the obvious view leaves out.

## 11. Updated milestone table

| Stage | Notes |
|---|---|
| Define intersection/corridor aggregation pipeline | Extends `scripts/clean_crash_data.py`; reuses existing street-name cleaning |
| Feature engineering table (intersection-month level) | New script, e.g. `scripts/build_risk_features.py` |
| Baseline Poisson/NB model + time-based validation | Establish honest baseline first; document split reasoning (10.3) |
| Naive-baseline comparison | Raw historical count ranking vs. model (10.1) |
| Gradient-boosted comparison model | Only after baseline is documented |
| Backtest ROI simulation | Strongest hiring-manager result |
| Error analysis | Check for highway/corridor blind spots (10.1) |
| Model card + non-technical executive summary | (10.1, 10.4) |
| Streamlit page: ranked list, budget-scenario slider, SHAP explainer | (10.2) |
| Tests + CI for new pipeline | (10.3) |
| README throughline rewrite + walkthrough GIF | (10.4) |
| Borough recovery (point-in-polygon join) | Sequence flexibly, per Section 8 |

---

*This document is a discussion draft. Confirm scope and any changes before implementation begins.*
