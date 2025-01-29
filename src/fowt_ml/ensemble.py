# ruff: noqa: N803
from collections.abc import Iterable
from typing import Any
from typing import Protocol
from typing import Self
from numpy.typing import ArrayLike
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import check_scoring
from sklearn.model_selection import cross_val_score


class Estimator(Protocol):
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self: ...  # noqa: D102
    def predict(self, X: ArrayLike) -> ArrayLike: ...  # noqa: D102


class EnsembleModel:
    """Class to handle random forest models and metrics for comparison."""

    ENSEMBLE_REGRESSORS = {
        "ExtraTrees": ExtraTreesRegressor,
        "RandomForest": RandomForestRegressor,
    }

    def __init__(self, estimator: str | Estimator, **kwargs: dict[str, Any]) -> None:
        if isinstance(estimator, str):
            if estimator not in self.ENSEMBLE_REGRESSORS:
                raise ValueError(
                    f"Available estimators: {self.ENSEMBLE_REGRESSORS.keys()}"
                )
            self.estimator = self.ENSEMBLE_REGRESSORS[estimator](**kwargs)
        else:
            self.estimator = estimator.set_params(**kwargs)

    def cross_val_score(
        self,
        X: ArrayLike,
        y: ArrayLike,
        cv: int | None = None,
        scoring: str | Iterable | None = None,
    ) -> float | ArrayLike:
        """Get Cross Validation score.

        Scoring paramers overview: https://scikit-learn.org/stable/modules/model_evaluation.html#string-name-scorers
        """  # noqa: E501
        return cross_val_score(self.estimator, X, y, cv=cv, scoring=scoring)

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
