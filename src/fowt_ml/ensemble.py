# ruff: noqa: N803
import warnings
from collections.abc import Iterable
from typing import Any
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
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_test: ArrayLike,
        y_test: ArrayLike,
        scoring: str | Iterable | None = None,
    ) -> float | ArrayLike:
        """Fit and calculate a score.

        Scoring paramers overview: https://scikit-learn.org/stable/modules/model_evaluation.html#string-name-scorers
        """  # noqa: E501
        self.estimator.fit(X_train, y_train)
        scorer = check_scoring(self.estimator, scoring=scoring)
        return scorer(self.estimator, X_test, y_test)
