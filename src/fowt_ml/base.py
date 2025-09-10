"""This is the base class for all models in the fowt_ml package."""

from numpy.typing import ArrayLike
import numpy as np
import time
from collections.abc import Iterable
from sklearn.metrics import check_scoring

class BaseModel:
    """Base class for all models."""
    def calculate_score(
            self,
            x_train: ArrayLike,
            x_test: ArrayLike,
            y_train: ArrayLike,
            y_test: ArrayLike,
            scoring: str | Iterable | None = None,
        ) -> float | dict[str, float]:
        """Calculate the score for the model using test data.

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
        scoring_list = None
        include_fit_time = False
        if scoring is not None:
            scoring_list = [scoring] if isinstance(scoring, str) else list(scoring)
            include_fit_time = "model_fit_time" in scoring_list

        if include_fit_time:
            scoring_list = [s for s in scoring_list if s != "model_fit_time"]

        scorer = check_scoring(self.estimator, scoring=scoring_list)
        scores = scorer(self.estimator, x_test, y_test)

        # if "model_fit_time" in original scoring, add it back
        if include_fit_time:
            scores["model_fit_time"] = model_fit_time

        return scores
