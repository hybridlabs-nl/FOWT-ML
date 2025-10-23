import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import check_scoring
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
        metrics = ["r2"]
        scorer = check_scoring(model.estimator, scoring=metrics)
        expected_scores = scorer(model.estimator, x_test, y_test)

        actual_scores = model.calculate_score(x_train, x_test, y_train, y_test, metrics)
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

        metrics = ["r2", "model_fit_time", "model_predict_time"]
        actual_scores = model.calculate_score(x_train, x_test, y_train, y_test, metrics)
        assert actual_scores["r2"] == expected_scores["r2"]
        assert actual_scores["model_fit_time"] >= 0
        assert actual_scores["model_predict_time"] >= 0


class TestCrossValidate:
    def test_cross_validate_default(self):
        model = DummyModel("LinearRegression")
        x_train = np.array([[1], [2], [3], [4], [5]])
        y_train = np.array([2, 3, 4, 5, 6])

        metrics = ["model_fit_time"]
        cv_results = model.cross_validate(x_train, y_train, scoring=metrics)
        assert "fit_time" not in cv_results
        assert "model_fit_time" in cv_results
        assert len(cv_results["model_fit_time"]) == 5

    def test_cross_validate_kwargs(self):
        model = DummyModel("LinearRegression")
        x_train = np.array([[1], [2], [3], [4], [5]])
        y_train = np.array([2, 3, 4, 5, 6])

        metrics = ["r2", "model_fit_time", "model_predict_time"]
        cv_results = model.cross_validate(x_train, y_train, scoring=metrics, cv=3)
        assert "test_r2" not in cv_results
        assert "r2" in cv_results
        assert "model_fit_time" in cv_results
        assert "model_predict_time" in cv_results
        assert len(cv_results["r2"]) == 3


class TestUseScaledData:
    def test_use_scaled_data_default(self):
        model = DummyModel("LinearRegression")

        x_train, y_train = make_regression(
            n_samples=200, n_features=3, noise=10, random_state=42
        )
        y_train = y_train * 1000  # pretend target is in dollars
        model.use_scaled_data()
        model.estimator.fit(x_train, y_train)
        y_predict = model.estimator.predict(x_train)

        # Check scaling wrappers
        assert isinstance(model.estimator, TransformedTargetRegressor), (
            "Estimator should be wrapped in TransformedTargetRegressor"
        )

        pipeline = model.estimator.regressor
        assert isinstance(pipeline, Pipeline), "Regressor should be a Pipeline"
        assert isinstance(pipeline.named_steps["scaler"], StandardScaler), (
            "Pipeline should contain a StandardScaler for X"
        )

        assert isinstance(model.estimator.transformer_, StandardScaler), (
            "Target transformer should be StandardScaler"
        )

        # Check prediction accuracy
        from sklearn.metrics import r2_score

        r2 = r2_score(y_train, y_predict)
        np.testing.assert_almost_equal(
            r2,
            0.99,
            decimal=2,
            err_msg=f"R2 score should be approximately {r2} after scaling",
        )

    def test_use_scaled_data_calculate_score(self):
        model = DummyModel("LinearRegression")

        x_train, y_train = make_regression(
            n_samples=200, n_features=3, noise=10, random_state=42
        )
        y_train = y_train * 1000  # pretend target is in dollars
        x_test, y_test = make_regression(
            n_samples=50, n_features=3, noise=10, random_state=24
        )
        y_test = y_test * 1000

        model.estimator.fit(x_train, y_train)
        metrics = ["r2", "neg_mean_squared_error"]
        expected_scores = model.calculate_score(
            x_train, x_test, y_train, y_test, metrics
        )

        # now scale data and check scores again
        model.use_scaled_data()
        actual_scores = model.calculate_score(x_train, x_test, y_train, y_test, metrics)

        np.testing.assert_almost_equal(
            actual_scores["r2"],
            expected_scores["r2"],
            decimal=2,
            err_msg="R2 scores should match after scaling",
        )
        np.testing.assert_almost_equal(
            actual_scores["neg_mean_squared_error"],
            expected_scores["neg_mean_squared_error"],
            decimal=2,
            err_msg="Neg MSE scores should match after scaling",
        )

    def test_use_scaled_data_cv(self):
        model = DummyModel("LinearRegression")

        x_train, y_train = make_regression(
            n_samples=200, n_features=3, noise=10, random_state=42
        )
        y_train = y_train * 1000  # pretend target is in dollars

        model.estimator.fit(x_train, y_train)
        metrics = ["r2", "neg_mean_squared_error"]
        expected_scores = model.cross_validate(x_train, y_train, metrics)

        # now scale data and check scores again
        model.use_scaled_data()
        actual_scores = model.cross_validate(x_train, y_train, metrics)

        np.testing.assert_almost_equal(
            actual_scores["r2"],
            expected_scores["r2"],
            decimal=2,
            err_msg="R2 scores should match after scaling",
        )
        np.testing.assert_almost_equal(
            actual_scores["neg_mean_squared_error"],
            expected_scores["neg_mean_squared_error"],
            decimal=2,
            err_msg="Neg MSE scores should match after scaling",
        )
