import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import check_scoring
from fowt_ml.base import BaseModel


class DummyModel(BaseModel):
    ESTIMATOR_NAMES = {
        "LinearRegression": LinearRegression,
    }


class TestInit:
    def test_init_with_string(self):
        model = DummyModel("LinearRegression")
        assert isinstance(model.estimator, LinearRegression)

    def test_init_with_estimator(self):
        estimator = LinearRegression()
        model = DummyModel(estimator)
        assert model.estimator is estimator

    def test_init_with_kwrags(self):
        model = DummyModel("LinearRegression", fit_intercept=False)
        assert model.estimator.fit_intercept is False


class TestCalculateScore:
    def test_calculate_score_default(self):
        model = DummyModel("LinearRegression")
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
        model = DummyModel("LinearRegression")
        x_train = np.array([[1], [2], [3], [4], [5]])
        y_train = np.array([2, 3, 4, 5, 6])
        x_test = np.array([[6], [7], [8]])
        y_test = np.array([7, 8, 9])

        model.estimator.fit(x_train, y_train)
        scorer = check_scoring(model.estimator, scoring=["r2"])
        expected_scores = scorer(model.estimator, x_test, y_test)

        metrics = ["r2", "model_fit_time"]
        actual_scores = model.calculate_score(x_train, x_test, y_train, y_test, metrics)
        assert actual_scores["r2"] == expected_scores["r2"]
        assert actual_scores["model_fit_time"] >= 0
