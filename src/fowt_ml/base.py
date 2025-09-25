"""This is the base class for all models in the fowt_ml package."""

import time
from collections.abc import Iterable
from typing import Any
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import check_scoring
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline


class BaseModel:
    """Base class for all models."""

    ESTIMATOR_NAMES: dict[str, type[BaseEstimator]] = {}

    def __init__(
        self, estimator: str | BaseEstimator, **kwargs: dict[str, Any]
    ) -> None:
        """Initialize the class with the estimator."""
        if isinstance(estimator, str):
            if estimator not in self.ESTIMATOR_NAMES:
                raise ValueError(f"Available estimators: {self.ESTIMATOR_NAMES.keys()}")
            self.estimator = self.ESTIMATOR_NAMES[estimator](**kwargs)
        else:
            self.estimator = estimator.set_params(**kwargs)

    def calculate_score(
        self,
        x_train: ArrayLike,
        x_test: ArrayLike,
        y_train: ArrayLike,
        y_test: ArrayLike,
        scoring: str | Iterable,
    ) -> float | dict[str, float]:
        """Calculate the score for the model using test data.

        First, the model is fitted to the training data, and the time taken to
        fit the model is recorded. Then, the model is scored using the provided
        scoring method(s) on the `test` data.

        In multi-output regression, by default, 'uniform_average' is used,
        which specifies a uniformly weighted mean over outputs. see
        https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics

        For scoring paramers overview:
        https://scikit-learn.org/stable/modules/model_evaluation.html#string-name-scorers

        Args:
            x_train (ArrayLike): training data for features
            x_test (ArrayLike): test data for features
            y_train (ArrayLike): training data for targets
            y_test (ArrayLike): test data for targets
            scoring (str | Iterable, optional): scoring method(s) to use.

        Returns:
            float | dict[str, float]: the calculated score(s)
        """  # noqa: E501
        model_fit_time = _measure_fit_time(self.estimator, x_train, y_train)

        # prepare scoring list and check if "model_fit_time" is included
        scoring_list = [scoring] if isinstance(scoring, str) else list(scoring)
        include_fit_time = "model_fit_time" in scoring_list
        include_predict_time = "model_predict_time" in scoring_list

        # Remove custom timing keys before passing to sklearn scorer
        scoring_list = [
            s for s in scoring_list if s not in {"model_fit_time", "model_predict_time"}
        ]

        if scoring_list:
            scorer = check_scoring(self.estimator, scoring=scoring_list)
            scores = scorer(self.estimator, x_test, y_test)
        else:
            scores = {}

        if include_fit_time:
            scores["model_fit_time"] = model_fit_time

        if include_predict_time:
            scores["model_predict_time"] = _measure_predict_latency(
                self.estimator, x_test
            )
        return scores

    def cross_validate(
        self,
        x_train: ArrayLike,
        y_train: ArrayLike,
        scoring: str | Iterable,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Perform cross-validation on the model.

        Args:
            x_train (ArrayLike): features data
            y_train (ArrayLike): target data
            scoring (str | Iterable, optional): scoring method(s) to use.
            **kwargs: additional keyword arguments to pass to `cross_validate`

        Returns:
            dict[str, Any]: dictionary containing cross-validation results
        """
        scoring_list = [scoring] if isinstance(scoring, str) else list(scoring)
        include_fit_time = "model_fit_time" in scoring_list
        include_predict_time = "model_predict_time" in scoring_list

        scoring_list = [
            s for s in scoring_list if s not in {"model_fit_time", "model_predict_time"}
        ]

        scorers = {}
        if scoring_list:
            scorers.update({s: s for s in scoring_list})
        if include_predict_time:
            scorers["model_predict_time"] = _measure_predict_latency

        cv_results = cross_validate(
            self.estimator,
            x_train,
            y_train,
            scoring=scorers or None,
            return_train_score=False,
            **kwargs,
        )

        results = {}
        for k, v in cv_results.items():
            if k.startswith("test_"):
                results[k.replace("test_", "")] = v
            elif include_fit_time and k == "fit_time":
                results["model_fit_time"] = np.round(v, 3)

        return results

    def use_scaled_data(self):
        """Wrap the estimator to use scaled data for both X and y."""
        if isinstance(self.estimator, TransformedTargetRegressor):
            return self  # already wrapped

        # Pipeline for input scaling + model
        regressor = Pipeline([
            ("scaler", StandardScaler()),
            ("model", self.estimator)
        ])

        # Wrap with TransformedTargetRegressor for y scaling
        self.estimator = TransformedTargetRegressor(
            regressor=regressor,
            transformer=StandardScaler()
        )
        return self


def _measure_fit_time(estimator, x_train, y_train) -> float:
    """Fit the estimator and return elapsed time in seconds."""
    start = time.perf_counter()
    estimator.fit(x_train, y_train)
    end = time.perf_counter()
    return np.round(end - start, 3)  # decimal 3 for millisecond


def _measure_predict_latency(estimator, x_test, y=None) -> float:
    """Measure average single-sample prediction latency in seconds."""
    # It doesn't loop over samples to avoid overhead of the loop itself!
    # Batch latency
    start = time.perf_counter()
    _ = estimator.predict(x_test)
    end = time.perf_counter()
    batch_latency_per_sample = (end - start) / len(x_test)

    # Single-sample latency
    if isinstance(x_test, pd.DataFrame):
        sample = x_test.iloc[[0]]  # keep as DataFrame
    elif isinstance(x_test, np.ndarray):
        sample = x_test[:1]  # keep as 2D array
    else:
        raise TypeError("X must be a pandas DataFrame or a NumPy array")

    start = time.perf_counter()
    _ = estimator.predict(sample)
    end = time.perf_counter()
    single_sample_latency = end - start
    return np.round(np.mean([batch_latency_per_sample, single_sample_latency]), 3)
