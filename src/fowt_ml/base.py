"""This is the base class for all models in the fowt_ml package."""

import logging
import time
from collections.abc import Iterable
from typing import Any
import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import check_scoring
from sklearn.model_selection import cross_validate as sklearn_cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted as check_is_fitted

logger = logging.getLogger(__name__)


class TimeSeriesStandardScaler(StandardScaler):
    """Standardize 3D arrays shaped (n_samples, seq_len, n_features).

    It computes mean/std per feature across all timesteps, then reshape back.
    """

    def fit(self, x, y=None):
        """Fit the scaler on 2D array."""
        x2d = self._to_2d(x)
        return super().fit(x2d, y)

    def transform(self, x):
        """Transform the 3D array."""
        orig_shape = x.shape
        x2d = self._to_2d(x)
        x2d_scaled = super().transform(x2d)
        return x2d_scaled.reshape(orig_shape)

    def inverse_transform(self, x):
        """Inverse transform the 3D array."""
        orig_shape = x.shape
        x2d = self._to_2d(x)
        x2d_inv = super().inverse_transform(x2d)
        return x2d_inv.reshape(orig_shape)

    @staticmethod
    def _to_2d(x):
        x = np.asarray(x)
        if x.ndim == 3:
            n, t, f = x.shape
            return x.reshape(n * t, f)
        if x.ndim == 2:
            return x
        raise ValueError(f"Expected 2D or 3D array, got shape {x.shape}")


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
        # allowed_3d=True for RNN-like models
        x_test = _check_arry(x_test, allowed_3d=True)

        # allowed_3d=False because scoring functions expect 2D arrays
        y_test = _check_arry(y_test, allowed_3d=False)

        # prepare scoring list and check if "model_fit_time" is included
        scoring_list = [scoring] if isinstance(scoring, str) else list(scoring)
        include_fit_time = "model_fit_time" in scoring_list
        include_predict_time = "model_predict_time" in scoring_list

        # calculate model fit time
        if is_fitted(self.estimator):
            if include_fit_time:
                logger.warning(
                    "The estimator is already fitted. "
                    "The reported 'model_fit_time' will be 0.0."
                )
                model_fit_time = 0.0
        else:
            model_fit_time = _measure_fit_time(self.estimator, x_train, y_train)

        # Remove custom timing keys before passing to sklearn scorer
        scoring_list = [
            s for s in scoring_list if s not in {"model_fit_time", "model_predict_time"}
        ]

        # estimator is already fitted in _measure_fit_time
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
        cv: int = 5,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Perform cross-validation on the model.

        Args:
            x_train (ArrayLike): features data
            y_train (ArrayLike): target data
            scoring (str | Iterable, optional): scoring method(s) to use.
            cv (int, optional): number of cross-validation folds. Default is 5.
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

        # cross_validate does the fitting internally,
        # so it accepts 3D arrays
        x_train = _check_arry(x_train, allowed_3d=True)
        y_train = _check_arry(y_train, allowed_3d=True)

        if is_fitted(self.estimator):
            logger.warning("The estimator is already fitted.")

        cv_results = sklearn_cross_validate(
            self.estimator,
            x_train,
            y_train,
            cv=cv,
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

    def use_scaled_data(self, data_3d=False):
        """Wrap the estimator to use scaled data for both X and y."""
        if isinstance(self.estimator, TransformedTargetRegressor):
            return self  # already wrapped

        x_scaler = TimeSeriesStandardScaler() if data_3d else StandardScaler()
        y_scaler = StandardScaler()

        # Pipeline for input scaling + model
        regressor = Pipeline([("scaler", x_scaler), ("model", self.estimator)])

        # Wrap with TransformedTargetRegressor for y scaling
        self.estimator = TransformedTargetRegressor(
            regressor=regressor, transformer=y_scaler
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
    elif isinstance(x_test, np.ndarray | torch.Tensor):
        sample = x_test[:1]  # keep as 2D array
    else:
        raise TypeError("X must be a pandas DataFrame, a NumPy array or torch Tensor.")

    start = time.perf_counter()
    _ = estimator.predict(sample)
    end = time.perf_counter()
    single_sample_latency = end - start
    return np.round(np.mean([batch_latency_per_sample, single_sample_latency]), 3)


def _check_arry(arr: ArrayLike, allowed_3d: bool) -> ArrayLike:
    """Check if the input is a valid array-like structure."""
    if not isinstance(arr, (pd.DataFrame | pd.Series | np.ndarray | torch.Tensor)):
        raise TypeError(
            "Input must be a pandas DataFrame/Series, a NumPy array or torch Tensor."
        )
    arr = np.asarray(arr)

    if not allowed_3d and arr.ndim == 3:
        raise ValueError("3D arrays are not allowed for this operation.")

    return arr


def is_fitted(estimator) -> bool:
    """Check if a estimator is fitted.

    Works for sklearn and skorch estimators.
    Returns True if fitted, False otherwise.
    """
    # First, handle skorch estimators
    if hasattr(estimator, "initialized_"):
        return estimator.initialized_

    # Fall back to sklearn's check_is_fitted
    try:
        check_is_fitted(estimator)
        return True
    except NotFittedError:
        return False
