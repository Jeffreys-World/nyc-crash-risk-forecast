"""Negative-binomial Safety Performance Function and the Empirical Bayes blend.

This is the codified method, not an invention. The AASHTO *Highway Safety Manual*
prescribes Empirical Bayes network screening: fit a negative-binomial SPF predicting
expected crashes from exposure and site characteristics, then blend that prediction
with the site's observed count, weighted by the model's overdispersion parameter.

    w  = 1 / (1 + k * P)
    EB = w * P + (1 - w) * observed

The blend is the whole point. A site with few observed crashes gets pulled toward the
model's prediction; a site with many keeps its own evidence. That is the correction for
regression to the mean that raw-count ranking lacks, and it is the reason DOT's
count-ranked priority list is worth auditing at all.

Negative binomial rather than Poisson because crash counts are overdispersed: variance
exceeds the mean, badly. Fitting Poisson here would understate uncertainty at exactly
the low-count sites where the EB correction matters most.

The fit is `statsmodels`, deliberately. A hand-rolled NB likelihood is a place to
introduce a subtle, invisible error in the one number the entire ranking depends on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import MIN_DISPERSION

log = logging.getLogger(__name__)

# Below this, exp(linear predictor) has underflowed rather than genuinely predicted
# zero. A true zero would give the EB blend weight 1.0 on the model and discard the
# site's observed crashes entirely.
MIN_PREDICTION = 1e-9


class SPFError(RuntimeError):
    """The fit cannot be trusted. Never downgraded to a warning."""


@dataclass
class SPFResult:
    """A fitted SPF plus everything needed to defend or reproduce it."""

    params: pd.Series
    dispersion: float
    predictors: list[str]
    unit_type: str
    n_observations: int
    converged: bool
    llf: float

    def summary(self) -> str:
        return (
            f"SPF[{self.unit_type}]: n={self.n_observations}, "
            f"dispersion k={self.dispersion:.6f}, llf={self.llf:.1f}, "
            f"predictors={self.predictors}"
        )


def fit_nb(
    features: pd.DataFrame,
    target: str,
    predictors: list[str],
    unit_type: str = "all",
    offset_col: str = "log_exposure",
    maxiter: int = 200,
) -> tuple[SPFResult, pd.Series]:
    """Fit the negative-binomial SPF with a log-exposure offset.

    The offset, not a coefficient. Exposure enters with its coefficient fixed at 1, so
    the model predicts a *rate* per unit of exposure. Letting the fit choose that
    coefficient would let it partially undo the exposure correction and quietly drift
    back toward ranking busy rather than dangerous.

    Returns the fit and its in-sample predictions.

    Raises:
        SPFError: on non-convergence, on a non-positive dispersion, or if any row
            reaching the fit has a non-finite offset.
    """
    usable = features[features.get("exposure_valid", True)].copy()
    if usable.empty:
        raise SPFError(f"{unit_type}: no units with valid exposure; nothing to fit")

    missing = [c for c in [target, offset_col, *predictors] if c not in usable.columns]
    if missing:
        raise SPFError(f"{unit_type}: missing column(s) {missing}")

    y = pd.to_numeric(usable[target], errors="coerce")
    offset = pd.to_numeric(usable[offset_col], errors="coerce")
    X = usable[predictors].apply(pd.to_numeric, errors="coerce")

    finite = np.isfinite(offset) & y.notna() & X.notna().all(axis=1)
    if not finite.all():
        dropped = int((~finite).sum())
        # An infinite offset is the log(0) failure arriving at the fit. The exposure
        # guard should have caught it upstream; if it reaches here, say so loudly.
        if np.isinf(offset[~finite]).any():
            raise SPFError(
                f"{unit_type}: {dropped} row(s) reached the fit with a non-finite "
                f"log-exposure offset. This is the log(0) failure; fix the exposure "
                f"guard rather than dropping the rows."
            )
        log.warning("%s: dropping %d row(s) with missing values", unit_type, dropped)

    y, X, offset = y[finite], X[finite], offset[finite]
    if len(y) < len(predictors) + 2:
        raise SPFError(
            f"{unit_type}: {len(y)} observations for {len(predictors)} predictors; "
            f"the fit would be unidentified"
        )

    X = sm.add_constant(X, has_constant="add")

    model = sm.NegativeBinomial(y, X, loglike_method="nb2", offset=offset.to_numpy())
    try:
        fit = model.fit(maxiter=maxiter, disp=False)
    except Exception as exc:  # statsmodels raises a variety of linalg errors here
        raise SPFError(f"{unit_type}: negative-binomial fit failed: {exc}") from exc

    converged = bool(getattr(fit, "mle_retvals", {}).get("converged", False))
    if not converged:
        # A non-converged fit still returns parameters. They look like numbers and
        # rank like nonsense, so this raises rather than warns.
        raise SPFError(
            f"{unit_type}: negative-binomial fit did not converge in {maxiter} "
            f"iterations. Refusing to return parameters that would silently produce a "
            f"meaningless ranking."
        )

    dispersion = float(fit.params.get("alpha", np.nan))
    if not np.isfinite(dispersion) or dispersion <= MIN_DISPERSION:
        raise SPFError(
            f"{unit_type}: dispersion k={dispersion!r} is not positive. The Empirical "
            f"Bayes weight w = 1/(1+k*P) is undefined here, and a k at zero means the "
            f"NB collapsed to Poisson - the overdispersion the method corrects for was "
            f"not detected. Check the target and exposure before proceeding."
        )

    predictions = pd.Series(fit.predict(X, offset=offset.to_numpy()), index=y.index)

    result = SPFResult(
        params=fit.params,
        dispersion=dispersion,
        predictors=predictors,
        unit_type=unit_type,
        n_observations=int(len(y)),
        converged=converged,
        llf=float(fit.llf),
    )
    log.info(result.summary())
    return result, predictions


def empirical_bayes(
    predicted: pd.Series,
    observed: pd.Series,
    dispersion: float,
) -> pd.Series:
    """Blend SPF prediction with observed count, HSM style.

        w  = 1 / (1 + k * P)
        EB = w * P + (1 - w) * observed

    Guards two boundaries:

    * `k <= 0` - the weight is undefined and the blend is meaningless. Raises.
    * `P == 0` - the weight becomes exactly 1, so EB collapses to the prediction and
      throws away every observed crash at that site. A genuine zero prediction is
      impossible from `exp()`, so a zero here is numerical underflow. The prediction is
      floored and the event is logged rather than silently discarding real casualties.
    """
    if not np.isfinite(dispersion) or dispersion <= MIN_DISPERSION:
        raise SPFError(
            f"dispersion k={dispersion!r} is not positive; the Empirical Bayes weight "
            f"w = 1/(1+k*P) is undefined"
        )

    p = pd.to_numeric(predicted, errors="coerce").astype(float)
    o = pd.to_numeric(observed, errors="coerce").astype(float).reindex(p.index)

    if p.isna().any() or o.isna().any():
        raise SPFError(
            "empirical_bayes received NaN in predicted or observed. A NaN here "
            "propagates into the ranking and then into the headline number."
        )

    underflowed = int((p <= MIN_PREDICTION).sum())
    if underflowed:
        log.warning(
            "%d prediction(s) at or below %g - floored. At P=0 the EB weight is 1 and "
            "the blend would discard the site's observed crashes entirely.",
            underflowed,
            MIN_PREDICTION,
        )
    p = p.clip(lower=MIN_PREDICTION)

    weight = 1.0 / (1.0 + dispersion * p)
    eb = weight * p + (1.0 - weight) * o

    if not np.isfinite(eb).all():
        raise SPFError("Empirical Bayes produced a non-finite estimate")

    return eb


def fit_and_blend(
    features: pd.DataFrame,
    target: str,
    predictors: list[str],
    observed_col: str,
    by_unit_type: bool = True,
) -> tuple[pd.DataFrame, dict[str, SPFResult]]:
    """Fit an SPF per unit type and attach EB estimates to the feature table.

    Separate fits because a corridor and an intersection are different site types with
    different exposure semantics: length in feet versus approach legs. Pooling them
    would force one dispersion parameter to describe both, and dispersion is precisely
    what sets the blend weight.

    Units excluded for degenerate exposure keep their observed count as the estimate.
    They stay in the universe so the capture-rate denominator is the true casualty
    total, not a total quietly reduced to whatever the model could fit.
    """
    out = features.copy()
    out["spf_prediction"] = np.nan
    out["eb_estimate"] = np.nan

    groups = (
        {t: out[out["unit_type"] == t] for t in out["unit_type"].unique()}
        if by_unit_type
        else {"all": out}
    )

    results: dict[str, SPFResult] = {}
    for unit_type, group in groups.items():
        if group.empty:
            continue
        try:
            result, predictions = fit_nb(group, target, predictors, unit_type=unit_type)
        except SPFError as exc:
            log.error("%s: %s", unit_type, exc)
            raise

        results[unit_type] = result
        out.loc[predictions.index, "spf_prediction"] = predictions
        out.loc[predictions.index, "eb_estimate"] = empirical_bayes(
            predictions, out.loc[predictions.index, observed_col], result.dispersion
        )

    unfitted = out["eb_estimate"].isna()
    if unfitted.any():
        log.warning(
            "%d unit(s) were not fitted (degenerate exposure); falling back to their "
            "observed count so they remain rankable and countable.",
            int(unfitted.sum()),
        )
        out.loc[unfitted, "eb_estimate"] = pd.to_numeric(
            out.loc[unfitted, observed_col], errors="coerce"
        ).fillna(0.0)

    return out, results
