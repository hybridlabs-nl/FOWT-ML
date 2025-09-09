# ruff: noqa: N803
import time
import warnings
from collections.abc import Iterable
from typing import Any
import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import check_scoring
from sklearn.metrics import get_scorer


class EnsembleModel:
    """Class to handle random forest models and metrics for comparison."""

    ENSEMBLE_REGRESSORS = {
        "ExtraTrees": ExtraTreesRegressor,
        "RandomForest": RandomForestRegressor,
    }

    def __init__(
        self, estimator: str | BaseEstimator, **kwargs: dict[str, Any]
    ) -> None:
        if isinstance(estimator, str):
            if estimator not in self.ENSEMBLE_REGRESSORS:
                raise ValueError(
                    f"Available estimators: {self.ENSEMBLE_REGRESSORS.keys()}"
                )
            self.estimator = self.ENSEMBLE_REGRESSORS[estimator](**kwargs)
        else:
            self.estimator = estimator.set_params(**kwargs)

    def oob_score(
        self, X: ArrayLike, y: ArrayLike, scoring: str | None = None
    ) -> float:
        """Fit and estimate generalization score from out-of-bag samples."""
        if scoring is None:
            oob_score = True
        else:
            scorer = get_scorer(scoring)

            def score_func(y, y_pred, **kwargs):
                return scorer._sign * scorer._score_func(y, y_pred, **kwargs)

            oob_score = score_func
        if not (self.estimator.bootstrap and self.estimator.oob_score):
            warnings.warn(f"Setting `bootstrap=True` and `oob_score={oob_score}`")
            self.estimator.set_params(bootstrap=True, oob_score=oob_score)
        self.estimator.fit(X, y)
        return self.estimator.oob_score_

    def calculate_score(
        self,
        x_train: ArrayLike,
        x_test: ArrayLike,
        y_train: ArrayLike,
        y_test: ArrayLike,
        scoring: str | Iterable | None = None,
    ) -> float | ArrayLike:
        """Fit and calculate a score.

        Scoring paramers overview: https://scikit-learn.org/stable/modules/model_evaluation.html#string-name-scorers
        """  # noqa: E501
        model_fit_start = time.time()
        self.estimator.fit(x_train, y_train)
        model_fit_end = time.time()
        model_fit_time = np.round(model_fit_end - model_fit_start, 2)

        # if "model_fit_time" in scoring, remove it
        if "model_fit_time" in scoring:
            if isinstance(scoring, str):
                scoring = [scoring]
            _scoring = [s for s in scoring if s != "model_fit_time"]

        scorer = check_scoring(self.estimator, scoring=_scoring)
        scorer_dict = scorer(self.estimator, x_test, y_test)

        # if "model_fit_time" in original scoring, add it back
        if "model_fit_time" in scoring:
            scorer_dict["model_fit_time"] = model_fit_time
        return scorer_dict
