"""The negative-binomial Safety Performance Function.

Two failure modes that must raise rather than warn:

* Non-convergence. statsmodels still returns parameters from a failed fit. They are
  numbers, they rank, and the ranking is noise.
* Non-positive dispersion. The EB weight `w = 1/(1+k*P)` is undefined, and a k at zero
  means the NB collapsed to Poisson, i.e. the overdispersion the entire method exists to
  correct for was not detected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import exposure_term
from src.spf import SPFError, fit_and_blend, fit_nb


@pytest.fixture
def fittable(modelable_units) -> pd.DataFrame:
    out, _ = exposure_term(modelable_units)
    return out


class TestFit:
    def test_converges_on_overdispersed_counts(self, fittable):
        result, _ = fit_nb(fittable, "casualties_36mo", ["night_share"])
        assert result.converged

    def test_recovers_a_positive_dispersion(self, fittable):
        """The fixture counts come from a real NB draw, so k must be detectable."""
        result, _ = fit_nb(fittable, "casualties_36mo", ["night_share"])
        assert result.dispersion > 0

    def test_predictions_are_positive_and_finite(self, fittable):
        _, predictions = fit_nb(fittable, "casualties_36mo", ["night_share"])
        assert (predictions > 0).all()
        assert np.isfinite(predictions).all()

    def test_longer_segments_predict_more_crashes(self, fittable):
        """The exposure offset doing its job: risk scales with length."""
        _, predictions = fit_nb(fittable, "casualties_36mo", ["night_share"])
        length = fittable.loc[predictions.index, "length_ft"]
        assert np.corrcoef(length, predictions)[0, 1] > 0.5

    def test_result_records_what_was_fitted(self, fittable):
        result, _ = fit_nb(fittable, "casualties_36mo", ["night_share"])
        assert result.predictors == ["night_share"]
        assert result.n_observations == len(fittable)


class TestFitGuards:
    def test_non_convergence_raises(self, fittable):
        """A single iteration cannot converge. It must raise, not return numbers."""
        with pytest.raises(SPFError, match="did not converge"):
            fit_nb(fittable, "casualties_36mo", ["night_share"], maxiter=1)

    def test_infinite_offset_raises_and_names_the_bug(self, fittable):
        """CRITICAL gap 2 arriving at the fit despite the upstream guard."""
        broken = fittable.copy()
        broken.loc[broken.index[0], "log_exposure"] = -np.inf
        with pytest.raises(SPFError, match="log\\(0\\)"):
            fit_nb(broken, "casualties_36mo", ["night_share"])

    def test_missing_column_raises(self, fittable):
        with pytest.raises(SPFError, match="missing column"):
            fit_nb(fittable, "casualties_36mo", ["does_not_exist"])

    def test_too_few_observations_raises(self, fittable):
        with pytest.raises(SPFError, match="unidentified"):
            fit_nb(fittable.head(2), "casualties_36mo", ["night_share"])

    def test_no_valid_exposure_raises(self, fittable):
        empty = fittable.copy()
        empty["exposure_valid"] = False
        with pytest.raises(SPFError, match="nothing to fit"):
            fit_nb(empty, "casualties_36mo", ["night_share"])


class TestFitAndBlend:
    def test_attaches_predictions_and_eb_estimates(self, fittable):
        out, results = fit_and_blend(
            fittable, "casualties_36mo", ["night_share"], "casualties_36mo"
        )
        assert out["spf_prediction"].notna().any()
        assert out["eb_estimate"].notna().all()
        assert "corridor" in results

    def test_every_unit_ends_up_rankable(self, fittable):
        """Degenerate units fall back to their observed count rather than dropping out."""
        with_degenerate = fittable.copy()
        with_degenerate.loc[with_degenerate.index[0], "exposure_valid"] = False
        with_degenerate.loc[with_degenerate.index[0], "log_exposure"] = np.nan

        out, _ = fit_and_blend(
            with_degenerate, "casualties_36mo", ["night_share"], "casualties_36mo"
        )
        assert out["eb_estimate"].notna().all()
        assert len(out) == len(fittable)

    def test_fits_each_unit_type_separately(self, fittable):
        """A corridor and an intersection have different exposure semantics."""
        mixed = fittable.copy()
        half = len(mixed) // 2
        mixed.iloc[half:, mixed.columns.get_loc("unit_type")] = "intersection"

        _, results = fit_and_blend(
            mixed, "casualties_36mo", ["night_share"], "casualties_36mo"
        )
        assert set(results) == {"corridor", "intersection"}

    def test_dispersion_differs_by_unit_type(self, fittable):
        """Which is why pooling would be wrong: dispersion sets the blend weight."""
        mixed = fittable.copy()
        half = len(mixed) // 2
        mixed.iloc[half:, mixed.columns.get_loc("unit_type")] = "intersection"

        _, results = fit_and_blend(
            mixed, "casualties_36mo", ["night_share"], "casualties_36mo"
        )
        assert results["corridor"].dispersion != results["intersection"].dispersion
