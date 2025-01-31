"""Module to handle linear models."""

from collections.abc import Iterable
from typing import Any
import sklearn.linear_model as lm
import sklearn.metrics as sm
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator


class LinearModels:
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

    def calculate_score(
        self,
        x_train: ArrayLike,
        x_test: ArrayLike,
        y_train: ArrayLike,
        y_test: ArrayLike,
        scoring: str | Iterable = None,
    ) -> float:
        """Calculate the score for the model using test data.

        First it fits the model with the training data, then predicts the test
        data

        Args:
            x_train (ArrayLike): training data for features
            x_test (ArrayLike): test data for features
            y_train (ArrayLike): training data for targets
            y_test (ArrayLike): test data for targets
            scoring (Union[str, scoring]): the scoring to calculate

        Returns:
            float: the scoring value
        """
        self.model = self.estimator.fit(x_train, y_train)
        scorer = sm.check_scoring(self.estimator, scoring=scoring)
        return scorer(self.estimator, x_test, y_test)
