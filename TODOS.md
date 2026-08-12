# TODOS

Deferred work, with enough context to pick it up cold.

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
in the model card.

**Context:** Plan-eng-review Issue 2 (2026-08-12) settled the centerline universe as
the unit of analysis, which makes segment length available as a free exposure term.
That is the decided baseline. Volume is the upgrade on top of it. This was explicitly
deferred to keep the Approach A slice at roughly two weeks.

**Depends on / blocked by:** The centerline universe (task T1) must exist first, since
volume counts join to centerline segments.

---

## Already recorded elsewhere, not duplicated here

Approach B (Streamlit page, budget slider, SHAP explainer, model card, CI) and
Approach C (the named-streets counterfactual) are captured in the office-hours design
doc at `~/.gstack/projects/Jeffreys-World-nyc-crash-risk-forecast/`. They are gated on
the Approach A backtest producing a result worth wrapping.
