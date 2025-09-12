"""Class to handle random forest models and metrics for comparison."""

import warnings
from numpy.typing import ArrayLike
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

    def oob_score(self, x: ArrayLike, y: ArrayLike, scoring: str) -> float:
        """Fit and estimate generalization score from out-of-bag samples."""
        scorer = get_scorer(scoring)

        def score_func(y, y_pred, **kwargs):
            return scorer._sign * scorer._score_func(y, y_pred, **kwargs)

        oob_score = score_func
        if not (self.estimator.bootstrap and self.estimator.oob_score):
            warnings.warn(f"Setting `bootstrap=True` and `oob_score={oob_score}`")
            self.estimator.set_params(bootstrap=True, oob_score=oob_score)
        self.estimator.fit(x, y)
        return self.estimator.oob_score_
