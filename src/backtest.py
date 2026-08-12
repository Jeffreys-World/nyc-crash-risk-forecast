"""Score three rankings against held-out years, under two selection regimes.

The question this answers: if each method had picked its priority locations before the
holdout window, how many of the pedestrian casualties that actually happened would have
occurred at the locations it picked?

Three rankings:

    R1  DOT's published Vision Zero priority list   the real, shipped artifact
    R2  raw trailing casualty count                  the naive baseline
    R3  Empirical Bayes (SPF blended with observed)  the HSM-standard method

R2 exists so the comparison is honest in both directions. If R3 does not beat R2, the
Empirical Bayes machinery bought nothing and the README says so.

Two regimes, because DOT does not rank citywide. It ranks within borough and stops at a
cumulative share of that borough's casualties. Scoring a citywide model against a
borough-stratified list would hand the model a win by construction: it concentrates
picks where density is highest while DOT must spend picks in every borough.

The silent failure guarded here is the zero-denominator capture rate. A borough with no
holdout casualties yields 0/0, and a NaN that propagates straight into the headline
number without raising anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import (
    BOOTSTRAP_CI,
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    BOROUGH_CUMULATIVE_SHARE,
    MIN_CAPTURE_RATE_LIFT_PP,
)

log = logging.getLogger(__name__)

# Share of bootstrap resamples allowed to have a zero denominator before the interval
# is refused outright. See the note in `bootstrap_capture_difference` for why this is
# 10% and not something higher.
MAX_DEGENERATE_RESAMPLE_FRACTION = 0.10


class BacktestError(RuntimeError):
    """A backtest result would be uninterpretable or silently wrong."""


# --------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------


@dataclass
class Selection:
    """Which units a ranking would have picked, and how it decided."""

    name: str
    regime: str
    unit_ids: list[str]
    per_borough: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.unit_ids)


def _ordered(units: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Rank by score descending, breaking ties on unit_id ascending.

    Ties at the selection boundary are common: many units share a trailing count of 1.
    Without an explicit tie-break, pandas' sort order decides which ones make the cut,
    and the headline number changes between machines and between runs.
    """
    if score_col not in units.columns:
        raise BacktestError(f"score column {score_col!r} not present")
    return units.sort_values(
        [score_col, "unit_id"], ascending=[False, True], kind="mergesort"
    )


def select_citywide_top_n(
    units: pd.DataFrame, score_col: str, n: int, name: str
) -> Selection:
    """Top N units citywide. Fair method-versus-method."""
    if n <= 0:
        raise BacktestError(f"{name}: citywide N must be positive, got {n}")

    ordered = _ordered(units, score_col)
    notes: list[str] = []
    if n > len(ordered):
        notes.append(f"requested N={n} exceeds universe of {len(ordered)}; selected all")
        n = len(ordered)

    picked = ordered.head(n)
    return Selection(
        name=name,
        regime="citywide",
        unit_ids=picked["unit_id"].tolist(),
        per_borough=picked.get("borough", pd.Series(dtype=str)).value_counts().to_dict(),
        notes=notes,
    )


def select_borough_stratified(
    units: pd.DataFrame,
    score_col: str,
    casualty_col: str,
    name: str,
    shares: dict[str, float] | None = None,
) -> Selection:
    """Reproduce DOT's rule: rank within borough, stop at a cumulative casualty share.

    Corridors stop at 50% of the borough's casualties, intersections at 15%. Applying
    the model's ranking under DOT's own stopping rule is what makes the comparison fair
    against the real published artifact rather than against a strawman.

    Two stop conditions have to be explicit or the loop is undefined:

    * A borough whose casualty total is zero has no share to reach. Nothing is selected
      there, and the borough is noted rather than contributing a silent NaN.
    * A borough that never reaches its share even after selecting every unit stops at
      the full borough. This happens when casualties are spread thin across many units.
    """
    shares = shares or BOROUGH_CUMULATIVE_SHARE
    if "borough" not in units.columns:
        raise BacktestError(
            f"{name}: borough-stratified selection needs a 'borough' column. Borough "
            f"comes from unit geometry, so a missing column means the universe was "
            f"built without it."
        )

    picked: list[str] = []
    per_borough: dict[str, int] = {}
    notes: list[str] = []

    for (borough, unit_type), group in units.groupby(["borough", "unit_type"], dropna=False):
        target_share = shares.get(str(unit_type))
        if target_share is None:
            notes.append(f"no stopping share defined for unit_type {unit_type!r}; skipped")
            continue

        total = float(pd.to_numeric(group[casualty_col], errors="coerce").fillna(0).sum())
        if total <= 0:
            notes.append(f"{borough}/{unit_type}: zero casualties, nothing selected")
            continue

        ordered = _ordered(group, score_col)
        casualties = pd.to_numeric(ordered[casualty_col], errors="coerce").fillna(0)
        cumulative = casualties.cumsum() / total

        reached = cumulative >= target_share
        if reached.any():
            cutoff = int(np.argmax(reached.to_numpy())) + 1
        else:
            cutoff = len(ordered)
            notes.append(
                f"{borough}/{unit_type}: cumulative share never reached "
                f"{target_share:.0%} (max {cumulative.iloc[-1]:.1%}); selected all "
                f"{cutoff} units"
            )

        chosen = ordered.head(cutoff)["unit_id"].tolist()
        picked.extend(chosen)
        per_borough[f"{borough}/{unit_type}"] = len(chosen)

    if not picked:
        raise BacktestError(
            f"{name}: borough-stratified selection picked nothing. Every borough had "
            f"zero casualties, which means the casualty column is wrong."
        )

    return Selection(
        name=name,
        regime="borough_stratified",
        unit_ids=picked,
        per_borough=per_borough,
        notes=notes,
    )


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


@dataclass
class CaptureRate:
    """Share of holdout casualties occurring at selected units."""

    name: str
    regime: str
    captured: float
    total: float
    n_selected: int
    defined: bool
    note: str = ""

    @property
    def rate(self) -> float | None:
        """None, never NaN, when undefined. A NaN would reach the README unnoticed."""
        return (self.captured / self.total) if self.defined else None

    @property
    def rate_pp(self) -> float | None:
        r = self.rate
        return None if r is None else 100.0 * r

    def summary(self) -> str:
        if not self.defined:
            return f"{self.name} [{self.regime}]: UNDEFINED - {self.note}"
        return (
            f"{self.name} [{self.regime}]: {self.rate_pp:.1f}% "
            f"({self.captured:.0f}/{self.total:.0f} casualties, "
            f"{self.n_selected} units)"
        )


def capture_rate(
    selection: Selection,
    holdout: pd.DataFrame,
    casualty_col: str = "holdout_casualties",
) -> CaptureRate:
    """Casualties at selected units, over all holdout casualties.

    The zero-denominator branch is explicit. If the holdout window contains no
    casualties at all, the rate is not zero and it is not NaN: it is undefined, and
    saying so is the only honest option. Returning NaN here is how a blank cell reaches
    a README as if it were a measurement.
    """
    if casualty_col not in holdout.columns:
        raise BacktestError(f"holdout frame has no {casualty_col!r} column")

    casualties = pd.to_numeric(holdout[casualty_col], errors="coerce").fillna(0.0)
    total = float(casualties.sum())

    if total <= 0:
        return CaptureRate(
            name=selection.name,
            regime=selection.regime,
            captured=0.0,
            total=0.0,
            n_selected=selection.n,
            defined=False,
            note=(
                "holdout window contains zero casualties, so the capture rate has no "
                "denominator. This is undefined, not 0%."
            ),
        )

    selected = holdout["unit_id"].isin(set(selection.unit_ids))
    captured = float(casualties[selected].sum())

    return CaptureRate(
        name=selection.name,
        regime=selection.regime,
        captured=captured,
        total=total,
        n_selected=selection.n,
        defined=True,
    )


# --------------------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------------------


@dataclass
class BootstrapCI:
    """Confidence interval on the difference between two capture rates."""

    point_estimate_pp: float
    lower_pp: float
    upper_pp: float
    level: float
    iterations: int
    excludes_zero: bool
    note: str = ""

    def summary(self) -> str:
        return (
            f"difference {self.point_estimate_pp:+.1f}pp, "
            f"{self.level:.0%} CI [{self.lower_pp:+.1f}, {self.upper_pp:+.1f}], "
            f"excludes zero: {self.excludes_zero}"
        )


def bootstrap_capture_difference(
    holdout: pd.DataFrame,
    selection_a: Selection,
    selection_b: Selection,
    casualty_col: str = "holdout_casualties",
    iterations: int = BOOTSTRAP_ITERATIONS,
    level: float = BOOTSTRAP_CI,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapCI:
    """Bootstrap the capture-rate difference (A - B) by resampling units.

    Resampling units rather than casualties is the right unit of uncertainty: the
    question is whether this *ranking* would hold up on a different draw of streets,
    not whether individual casualties were counted correctly.

    Small-sample behaviour is reported, not hidden. With few units carrying casualties,
    the interval is wide and the pre-registered bar will simply not be cleared, which is
    the correct outcome rather than a spuriously tight result.
    """
    casualties = pd.to_numeric(holdout[casualty_col], errors="coerce").fillna(0.0)
    total_all = float(casualties.sum())
    if total_all <= 0:
        raise BacktestError(
            "cannot bootstrap: holdout contains zero casualties, so both capture rates "
            "are undefined"
        )

    in_a = holdout["unit_id"].isin(set(selection_a.unit_ids)).to_numpy()
    in_b = holdout["unit_id"].isin(set(selection_b.unit_ids)).to_numpy()
    values = casualties.to_numpy()

    point = 100.0 * (values[in_a].sum() / total_all - values[in_b].sum() / total_all)

    rng = np.random.default_rng(seed)
    n = len(values)
    diffs = np.empty(iterations, dtype=float)
    degenerate = 0

    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        total = values[idx].sum()
        if total <= 0:
            # A resample that drew only zero-casualty units has no denominator.
            # Counted and excluded rather than contributing a NaN to the percentiles.
            diffs[i] = np.nan
            degenerate += 1
            continue
        diffs[i] = 100.0 * (
            values[idx][in_a[idx]].sum() / total - values[idx][in_b[idx]].sum() / total
        )

    usable = diffs[np.isfinite(diffs)]
    degenerate_fraction = degenerate / iterations
    if degenerate_fraction > MAX_DEGENERATE_RESAMPLE_FRACTION:
        # The threshold is set where it can actually fire. If casualties sit in k
        # units, the share of resamples that draw none of them tends to exp(-k), so a
        # single casualty-bearing unit degenerates about 37% of the time and even two
        # only reach 14%. A 50% bar would be unreachable and therefore decorative;
        # 10% corresponds to roughly three or more casualty-bearing units, which is
        # the point below which an interval is not worth quoting.
        raise BacktestError(
            f"bootstrap unusable: {degenerate}/{iterations} resamples "
            f"({degenerate_fraction:.1%}) had a zero denominator, above the "
            f"{MAX_DEGENERATE_RESAMPLE_FRACTION:.0%} limit. Holdout casualties are "
            f"concentrated in too few units to support an interval."
        )

    alpha = (1.0 - level) / 2.0
    lower = float(np.percentile(usable, 100 * alpha))
    upper = float(np.percentile(usable, 100 * (1 - alpha)))

    note = ""
    if degenerate:
        note = f"{degenerate}/{iterations} resamples discarded for a zero denominator"

    return BootstrapCI(
        point_estimate_pp=float(point),
        lower_pp=lower,
        upper_pp=upper,
        level=level,
        iterations=iterations,
        excludes_zero=(lower > 0.0) or (upper < 0.0),
        note=note,
    )


# --------------------------------------------------------------------------------------
# The pre-registered verdict
# --------------------------------------------------------------------------------------


@dataclass
class Verdict:
    """Applies the bar written into the README before the number existed."""

    lift_pp: float
    ci: BootstrapCI
    threshold_pp: float
    clears_bar: bool
    reason: str

    def summary(self) -> str:
        outcome = "CLEARS" if self.clears_bar else "DOES NOT CLEAR"
        return (
            f"{outcome} the pre-registered bar "
            f"(>= {self.threshold_pp:.1f}pp and CI excluding zero): {self.reason}"
        )


def apply_preregistered_bar(
    eb: CaptureRate,
    naive: CaptureRate,
    ci: BootstrapCI,
    threshold_pp: float = MIN_CAPTURE_RATE_LIFT_PP,
) -> Verdict:
    """EB beats raw count only if the lift clears the threshold AND the CI excludes zero.

    Both conditions were fixed in the README before any backtest ran. Relaxing either
    one after seeing the number would turn a pre-registration into a rationalisation.
    """
    if eb.rate_pp is None or naive.rate_pp is None:
        return Verdict(
            lift_pp=float("nan"),
            ci=ci,
            threshold_pp=threshold_pp,
            clears_bar=False,
            reason="a capture rate is undefined, so no comparison is possible",
        )

    lift = eb.rate_pp - naive.rate_pp
    meets_threshold = lift >= threshold_pp

    if meets_threshold and ci.excludes_zero:
        reason = f"lift {lift:+.1f}pp with CI {ci.lower_pp:+.1f} to {ci.upper_pp:+.1f}"
        clears = True
    elif not meets_threshold and ci.excludes_zero:
        reason = f"lift {lift:+.1f}pp is real but below the {threshold_pp:.1f}pp bar"
        clears = False
    elif meets_threshold:
        reason = f"lift {lift:+.1f}pp clears the bar but the CI includes zero"
        clears = False
    else:
        reason = f"lift {lift:+.1f}pp is below the bar and the CI includes zero"
        clears = False

    return Verdict(
        lift_pp=lift,
        ci=ci,
        threshold_pp=threshold_pp,
        clears_bar=clears,
        reason=reason,
    )


# --------------------------------------------------------------------------------------
# Treated / untreated split
# --------------------------------------------------------------------------------------


def split_by_treatment(
    holdout: pd.DataFrame, selection: Selection
) -> dict[str, CaptureRate]:
    """Capture rate among treated and untreated units separately.

    This is the endogeneity control. DOT's priority locations were chosen *in order to*
    receive Street Improvement Projects, so their holdout casualty counts reflect the
    intervention. A single blended number cannot distinguish "the ranking was wrong"
    from "the ranking was right and the fix worked", and those are opposite conclusions.
    """
    if "treated" not in holdout.columns:
        raise BacktestError(
            "holdout frame has no 'treated' column; the SIP join must run before the "
            "backtest or the comparison is confounded"
        )

    out: dict[str, CaptureRate] = {}
    for label, mask in (
        ("treated", holdout["treated"].fillna(False).astype(bool)),
        ("untreated", ~holdout["treated"].fillna(False).astype(bool)),
    ):
        subset = holdout[mask]
        if subset.empty:
            out[label] = CaptureRate(
                name=f"{selection.name} ({label})",
                regime=selection.regime,
                captured=0.0,
                total=0.0,
                n_selected=0,
                defined=False,
                note=f"no {label} units in the holdout",
            )
            continue

        sub_selection = Selection(
            name=f"{selection.name} ({label})",
            regime=selection.regime,
            unit_ids=[u for u in selection.unit_ids if u in set(subset["unit_id"])],
        )
        out[label] = capture_rate(sub_selection, subset)

    return out
