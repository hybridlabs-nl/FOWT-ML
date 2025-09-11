import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import check_scoring
from fowt_ml.base import BaseModel


class TestCalculateScore:
    def test_calculate_score_default(self):
        model = BaseModel()
        model.estimator = LinearRegression()
        x_train = np.array([[1], [2], [3], [4], [5]])
        y_train = np.array([2, 3, 4, 5, 6])
        x_test = np.array([[6], [7], [8]])
        y_test = np.array([7, 8, 9])

        model.estimator.fit(x_train, y_train)
        scorer = check_scoring(model.estimator)
        expected_scores = scorer(model.estimator, x_test, y_test)

        actual_scores = model.calculate_score(x_train, x_test, y_train, y_test)
        assert actual_scores == expected_scores

    def test_calculate_score_metrics(self):
        model = BaseModel()
        model.estimator = LinearRegression()
        x_train = np.array([[1], [2], [3], [4], [5]])
        y_train = np.array([2, 3, 4, 5, 6])
        x_test = np.array([[6], [7], [8]])
        y_test = np.array([7, 8, 9])

        model_fit_start = time.time()
        model.estimator.fit(x_train, y_train)
        model_fit_end = time.time()
        model_fit_time = np.round(model_fit_end - model_fit_start, 2)

        scorer = check_scoring(model.estimator, scoring=["r2"])
        expected_scores = scorer(model.estimator, x_test, y_test)

        metrics = ["r2", "model_fit_time"]
        actual_scores = model.calculate_score(x_train, x_test, y_train, y_test, metrics)
        assert actual_scores["r2"] == expected_scores["r2"]
        assert np.isclose(actual_scores["model_fit_time"], model_fit_time, rtol=1e-1)
