"""Module to handle linear models."""

from dataclasses import dataclass
import sklearn.linear_model as lm
import sklearn.metrics as sm
from numpy.typing import ArrayLike


@dataclass
class Estimator:
    name: str
    func: callable
    reference: str = "sklearn"


@dataclass
class Metric:
    name: str
    func: callable
    reference: str = "sklearn"


class LinearModels:
    """Class to handle linear models and metrics for comparison."""

    ESTIMATOR_NAMES = {
        "LinearRegression": lm.LinearRegression(),
        "RidgeRegression": lm.Ridge(),
        "LassoRegression": lm.Lasso(),
        "ElasticNetRegression": lm.ElasticNet(),
        "LeastAngleRegression": lm.Lars(),
    }
    METRICS_NAMES = {
        "root_mean_squared_error": sm.root_mean_squared_error,
        "mean_squared_error": sm.mean_squared_error,
        "r2_score": sm.r2_score,
        "mean_absolute_error": sm.mean_absolute_error,
    }

    def __init__(self, estimator: str | Estimator, kwargs: dict = None) -> None:
        """Initialize the class with the estimator."""
        if isinstance(estimator, str):
            if not self.ESTIMATOR_NAMES.get(estimator):
                msg = (
                    f"estimator {estimator} not supported. "
                    f"Choose one of {list(self.ESTIMATOR_NAMES.keys())}"
                    f"or pass a Estimator instance."
                )
                raise ValueError(msg)
            self.estimator = Estimator(estimator, self.ESTIMATOR_NAMES.get(estimator))
        elif isinstance(estimator, Estimator):
            self.estimator = estimator
            # TODO: validate if model function is a callable and valid
        else:
            raise ValueError("model must be a string or a Estimator instance.")

        # fill the estimator with the kwargs
        if kwargs:
            for key, value in kwargs.items():
                setattr(self.estimator.func, key, value)

    def calculate_metric(
        self,
        x_train: ArrayLike,
        x_test: ArrayLike,
        y_train: ArrayLike,
        y_test: ArrayLike,
        metric: str | Metric,
    ) -> float:
        """Calculate the metric for the model using test data.

        First it fits the model with the training data, then predicts the test
        data

        Args:
            x_train (ArrayLike): training data for features
            x_test (ArrayLike): test data for features
            y_train (ArrayLike): training data for targets
            y_test (ArrayLike): test data for targets
            metric (Union[str, Metric]): the metric to calculate

        Returns:
            float: the metric value
        """
        if isinstance(metric, str):
            if not self.METRICS_NAMES.get(metric):
                msg = (
                    f"metric {metric} not supported. "
                    f"Choose one of {list(self.METRICS_NAMES.keys())}"
                    f"or pass a Metric instance."
                )
                raise ValueError(msg)
            self.metric = Metric(metric, self.METRICS_NAMES.get(metric))
        elif not isinstance(metric, Metric):
            raise ValueError("metric must be a string or a Metric instance.")

        # TODO: check other arguments of fit, predict and metric functions
        self.model = self.estimator.func.fit(x_train, y_train)
        self.y_pred = self.model.predict(x_test)
        return self.metric.func(y_test, self.y_pred)
