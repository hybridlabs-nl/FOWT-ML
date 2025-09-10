"""Class to handle random forest models and metrics for comparison."""

import warnings
from typing import Any
import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import get_scorer

from fowt_ml.base import BaseModel


class EnsembleModel(BaseModel):
    """Class to handle random forest models and metrics for comparison."""

    ESTIMATOR_NAMES = {
        "ExtraTrees": ExtraTreesRegressor,
        "RandomForest": RandomForestRegressor,
    }

    def __init__(
        self, estimator: str | BaseEstimator, **kwargs: dict[str, Any]
    ) -> None:
        if isinstance(estimator, str):
            if estimator not in self.ESTIMATOR_NAMES:
                raise ValueError(f"Available estimators: {self.ESTIMATOR_NAMES.keys()}")
            self.estimator = self.ESTIMATOR_NAMES[estimator](**kwargs)
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
