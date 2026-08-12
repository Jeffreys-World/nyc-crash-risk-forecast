"""The Empirical Bayes blend, and its weight boundary.

    w  = 1 / (1 + k * P)
    EB = w * P + (1 - w) * observed

The boundary that matters: as P approaches zero the weight approaches 1, so EB collapses
onto the prediction and the site's observed casualties stop counting. A true zero is
impossible out of `exp()`, so a zero there is numerical underflow, and underflow silently
erasing real casualties is exactly the class of bug this project keeps guarding against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.spf import MIN_PREDICTION, SPFError, empirical_bayes


class TestBlendFormula:
    def test_matches_the_hsm_formula_exactly(self):
        predicted = pd.Series([2.0, 5.0])
        observed = pd.Series([10.0, 0.0])
        k = 0.5

        expected = []
        for p, o in zip(predicted, observed, strict=True):
            w = 1.0 / (1.0 + k * p)
            expected.append(w * p + (1.0 - w) * o)

        result = empirical_bayes(predicted, observed, k)
        assert result.tolist() == pytest.approx(expected)

    def test_estimate_lies_between_prediction_and_observation(self):
        """A blend, by definition. Outside that range is not a blend."""
        predicted = pd.Series([1.0, 3.0, 7.0])
        observed = pd.Series([12.0, 0.0, 4.0])
        result = empirical_bayes(predicted, observed, 0.4)

        lower = np.minimum(predicted, observed)
        upper = np.maximum(predicted, observed)
        assert ((result >= lower - 1e-9) & (result <= upper + 1e-9)).all()

    def test_higher_counts_shift_weight_toward_observation(self):
        """The regression-to-the-mean correction, stated as a property.

        A busy site keeps its own evidence. A quiet site is pulled toward the model,
        which is what stops one bad year from promoting it up the ranking.
        """
        k = 0.5
        quiet = empirical_bayes(pd.Series([0.5]), pd.Series([10.0]), k).iloc[0]
        busy = empirical_bayes(pd.Series([20.0]), pd.Series([10.0]), k).iloc[0]

        w_quiet = 1.0 / (1.0 + k * 0.5)
        w_busy = 1.0 / (1.0 + k * 20.0)
        assert w_quiet > w_busy
        assert abs(quiet - 0.5) < abs(busy - 20.0)

    def test_zero_dispersion_limit_returns_the_prediction(self):
        """As k approaches 0 the weight approaches 1, so EB approaches P."""
        result = empirical_bayes(pd.Series([3.0]), pd.Series([9.0]), 1e-6)
        assert result.iloc[0] == pytest.approx(3.0, abs=1e-3)


class TestGuards:
    def test_zero_dispersion_raises(self):
        with pytest.raises(SPFError, match="not positive"):
            empirical_bayes(pd.Series([1.0]), pd.Series([2.0]), 0.0)

    def test_negative_dispersion_raises(self):
        with pytest.raises(SPFError, match="not positive"):
            empirical_bayes(pd.Series([1.0]), pd.Series([2.0]), -0.3)

    def test_nan_dispersion_raises(self):
        with pytest.raises(SPFError, match="not positive"):
            empirical_bayes(pd.Series([1.0]), pd.Series([2.0]), float("nan"))

    def test_nan_input_raises_rather_than_propagating(self):
        """A NaN here reaches the ranking, then the headline number."""
        with pytest.raises(SPFError, match="NaN"):
            empirical_bayes(pd.Series([1.0, np.nan]), pd.Series([2.0, 3.0]), 0.5)

    def test_nan_observation_raises(self):
        with pytest.raises(SPFError, match="NaN"):
            empirical_bayes(pd.Series([1.0, 2.0]), pd.Series([2.0, np.nan]), 0.5)


class TestPredictionZeroBoundary:
    def test_zero_prediction_is_floored_not_passed_through(self):
        result = empirical_bayes(pd.Series([0.0]), pd.Series([5.0]), 0.5)
        assert np.isfinite(result.iloc[0])

    def test_floored_prediction_produces_a_weight_of_almost_one(self):
        """Documents the real consequence rather than pretending it away.

        At P=0 the HSM weight is exactly 1, so the blend returns the prediction and the
        observed casualties do not count. Flooring keeps the arithmetic finite; it does
        not rescue the estimate. The value of the guard is that the underflow is
        detected and logged instead of silently producing a zero-risk ranking.
        """
        result = empirical_bayes(pd.Series([0.0]), pd.Series([5.0]), 0.5)
        assert result.iloc[0] < 1e-6

    def test_underflow_is_logged(self, caplog):
        with caplog.at_level("WARNING"):
            empirical_bayes(pd.Series([0.0]), pd.Series([5.0]), 0.5)
        assert "floored" in caplog.text.lower()

    def test_a_prediction_at_the_floor_is_treated_as_underflow(self, caplog):
        with caplog.at_level("WARNING"):
            empirical_bayes(pd.Series([MIN_PREDICTION]), pd.Series([5.0]), 0.5)
        assert "floored" in caplog.text.lower()

    def test_normal_predictions_are_not_flagged(self, caplog):
        with caplog.at_level("WARNING"):
            empirical_bayes(pd.Series([1.5]), pd.Series([5.0]), 0.5)
        assert "floored" not in caplog.text.lower()

    def test_result_is_always_finite(self):
        predicted = pd.Series([0.0, 1e-12, 1.0, 1e6])
        observed = pd.Series([5.0, 0.0, 3.0, 2.0])
        assert np.isfinite(empirical_bayes(predicted, observed, 0.5)).all()
