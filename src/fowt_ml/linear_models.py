from typing import Union
import mlflow
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, Lars
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score, mean_absolute_error
from dataclasses import dataclass

@dataclass
class Algorithm:
    name: str
    func: callable

@dataclass
class Metric:
    name: str
    func: callable


class LinearModels:
    ALGORITHM_NAMES = {
        'LinearRegression': LinearRegression(),
        'RidgeRegression': Ridge(),
        'LassoRegression': Lasso(),
        'ElasticNetRegression': ElasticNet(),
        'LeastAngleRegression': Lars()
    }
    METRICS_NAMES = {
        'root_mean_squared_error': root_mean_squared_error,
        'mean_squared_error': mean_squared_error,
        'r2_score': r2_score,
        'mean_absolute_error': mean_absolute_error
    }

    def __init__(self, model: Union[str, Algorithm]):
        """Initialize the class"""
        if isinstance(model, str):
            if not self.ALGORITHM_NAMES.get(model):
                msg = (
                    f"model {model} not supported. "
                    f"Choose one of {list(self.ALGORITHM_NAMES.keys())}"
                    f"or pass a Algorithm instance."
                )
                raise ValueError(msg)
            self.model = Algorithm(model, self.ALGORITHM_NAMES.get(model))
        elif isinstance(model, Algorithm):
            self.model = model
            # TODO: validate if model function is a callable and valid
        else:
            raise ValueError("model must be a string or a Algorithm instance.")

    def calculate_metric(self, X_train, X_test, y_train, y_test, metric: Union[str, Metric]):
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

        self.trained_model = self.model.func.fit(X_train, y_train)
        self.y_pred = self.trained_model.predict(X_test)
        return self.metric.func(y_test, self.y_pred)
