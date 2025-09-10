"""Module to handle linear models."""

from typing import Any
import numpy as np
import sklearn.linear_model as lm
from sklearn.base import BaseEstimator

from fowt_ml.base import BaseModel


class LinearModels(BaseModel):
    """Class to handle linear models and metrics for comparison."""

    ESTIMATOR_NAMES = {
        "LinearRegression": lm.LinearRegression,
        "RidgeRegression": lm.Ridge,
        "LassoRegression": lm.Lasso,
        "ElasticNetRegression": lm.ElasticNet,
        "LeastAngleRegression": lm.Lars,
    }

    def __init__(
        self, estimator: str | BaseEstimator, **kwargs: dict[str, Any]
    ) -> None:
        """Initialize the class with the estimator."""
        if isinstance(estimator, str):
            if not self.ESTIMATOR_NAMES.get(estimator):
                msg = (
                    f"estimator {estimator} not supported. "
                    f"Choose one of {list(self.ESTIMATOR_NAMES.keys())}"
                    f"or pass a Estimator instance."
                )
                raise ValueError(msg)
            self.estimator = self.ESTIMATOR_NAMES.get(estimator)(**kwargs)
        elif isinstance(estimator, BaseEstimator):
            self.estimator = estimator.set_params(**kwargs)
        else:
            raise ValueError("model must be a string or a Estimator instance.")
