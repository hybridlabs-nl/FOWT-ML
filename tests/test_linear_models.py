import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from fowt_ml.linear_models import LinearModels


@pytest.fixture
def simple_dataset():
    """Create a simple dataset for testing."""
    x_train = [[1, 2], [3, 4], [5, 6]]
    x_test = [[7, 8], [9, 10], [11, 12]]
    y_train = [1, 2, 3]
    y_test = [4, 5, 6]
    return x_train, x_test, y_train, y_test


class TestLinearModel:
    def test_init_estimator_lr(self):
        model = LinearModels("LinearRegression")
        assert isinstance(model.estimator, LinearRegression)

    def test_init_estimator_ridge(self):
        model = LinearModels("RidgeRegression")
        assert isinstance(model.estimator, Ridge)

    def test_init_estimator_lasso(self):
        model = LinearModels("LassoRegression")
        assert isinstance(model.estimator, Lasso)

    def test_init_estimator_elasticnet(self):
        model = LinearModels("ElasticNetRegression")
        assert isinstance(model.estimator, ElasticNet)

    def test_init_estimator_lars(self):
        model = LinearModels("LeastAngleRegression")
        assert isinstance(model.estimator, Lars)

    def test_init_with_estimator_instance(self):
        model = LinearModels(Lars())
        assert isinstance(model.estimator, BaseEstimator)

    def test_init_with_invalid_name(self):
        with pytest.raises(ValueError) as e:
            LinearModels("LinearRegressionInvalid")
        assert "Available estimators" in str(e.value)

    def test_init_estimator_kwargs(self):
        model = LinearModels("LinearRegression", fit_intercept=False)
        assert model.estimator.fit_intercept is False

    def test_calculate_score_rmse(self, simple_dataset):
        x_train, x_test, y_train, y_test = simple_dataset
        model = LinearModels("LinearRegression")
        results = model.calculate_score(
            x_train, x_test, y_train, y_test, "neg_mean_squared_log_error"
        )
        np.testing.assert_almost_equal(
            results["neg_mean_squared_log_error"], 0.0, decimal=3
        )

    def test_calculate_score_more(self, simple_dataset):
        x_train, x_test, y_train, y_test = simple_dataset
        model = LinearModels("LinearRegression")
        results = model.calculate_score(
            x_train,
            x_test,
            y_train,
            y_test,
            ["neg_mean_squared_error", "neg_mean_squared_log_error"],
        )
        np.testing.assert_almost_equal(
            results["neg_mean_squared_error"], -0.0, decimal=3
        )
        np.testing.assert_almost_equal(
            results["neg_mean_squared_log_error"], -0.0, decimal=3
        )
