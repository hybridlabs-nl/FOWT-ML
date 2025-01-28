from typing import Any, Protocol, Self

from numpy.typing import ArrayLike
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score


class Estimator(Protocol):
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self: ...
    def predict(self, X: ArrayLike) -> ArrayLike: ...


ENSEMBLE_REGRESSORS = {
    "ExtraTrees": ExtraTreesRegressor,
    "RandomForest": RandomForestRegressor,
}


class EnsembleModel:
    """Class to handle random forest models and metrics for comparison."""
    def __init__(self, estimator: str | Estimator, **kwargs: dict[str, Any]) -> None:
        if isinstance(estimator, str):
            if estimator not in ENSEMBLE_REGRESSORS:
                raise ValueError(f"Available estimators: {ENSEMBLE_REGRESSORS.keys()}")
            self.estimator = ENSEMBLE_REGRESSORS[estimator](**kwargs)
        else:
            self.estimator = estimator.set_params(**kwargs)

        self._is_fitted = False

    def cross_val_score(self, X: ArrayLike, y: ArrayLike, cv: int | None = None, scoring: str | None = None):
        """ Get Cross Validation score.

        Scoring paramers overview: https://scikit-learn.org/stable/modules/model_evaluation.html#string-name-scorers
        """
        return cross_val_score(self.estimator, X, y, cv=cv, scoring=scoring)

    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike = None) -> None:
        """ Fit an estimator. """
        self.estimator.fit(X, y, sample_weight=sample_weight)
        self._is_fitted = True

    def predict(self, X: ArrayLike) -> ArrayLike:
        """ Predict with a fitted estimator. """
        if not self._is_fitted:
            raise RuntimeError("Run `model.fit(...)` first!")
        return self.estimator.predict(X)

    def score(self, X: ArrayLike, y: ArrayLike, scoring: str | None = None) -> float | ArrayLike:
        """ Calculate score for a fitted estimator.

        If "scoring" is not provided, directly use the `.score()` method of the estimator.
        """
        if scoring is None:
            score = self.estimator.score(X, y)
        else:
            scorer = get_scorer(scoring)
            score = scorer(self.estimator, X, y)
        return score
