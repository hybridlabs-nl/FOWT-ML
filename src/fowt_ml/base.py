"""This is the base class for all models in the fowt_ml package."""

import time
from collections.abc import Iterable
from typing import Any
import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.metrics import check_scoring
from sklearn.model_selection import cross_validate


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
        model_fit_start = time.time()
        self.estimator.fit(x_train, y_train)
        model_fit_end = time.time()
        model_fit_time = np.round(model_fit_end - model_fit_start, 2)

        # prepare scoring list and check if "model_fit_time" is included
        include_fit_time = False
        scoring_list = [scoring] if isinstance(scoring, str) else list(scoring)
        include_fit_time = "model_fit_time" in scoring_list

        if include_fit_time:
            scoring_list = [s for s in scoring_list if s != "model_fit_time"]

        if scoring_list:
            scorer = check_scoring(self.estimator, scoring=scoring_list)
            scores = scorer(self.estimator, x_test, y_test)
        else:
            scores = {}

        # if "model_fit_time" in original scoring, add it back
        if include_fit_time:
            scores["model_fit_time"] = model_fit_time

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
        include_fit_time = False
        scoring_list = [scoring] if isinstance(scoring, str) else list(scoring)
        include_fit_time = "model_fit_time" in scoring_list

        if include_fit_time:
            scoring_list = [s for s in scoring_list if s != "model_fit_time"]

        if len(scoring_list) == 0:
            scoring_list = None  # to get fit_time

        cv_results = cross_validate(
            self.estimator,
            x_train,
            y_train,
            scoring=scoring_list,  # cannot be empty list
            **kwargs,
        )

        # if "model_fit_time" in original scoring, add it back
        if include_fit_time:
            cv_results["model_fit_time"] = np.round(cv_results.pop("fit_time"), 2)

        # select only scoring keys related to "test" set in each CV split
        cv_results = {
            k.replace("test_", ""): v
            for k, v in cv_results.items()
            if k.startswith("test_") or k == "model_fit_time"
        }

        return cv_results
