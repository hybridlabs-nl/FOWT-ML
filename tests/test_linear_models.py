import numpy as np
import pytest
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
from fowt_ml.linear_models import Estimator
from fowt_ml.linear_models import LinearModels
from fowt_ml.linear_models import Metric


class TestEstimator:
    def test_init_estimator(self):
        estimator = Estimator("LinearRegression", LinearRegression())
        assert estimator.name == "LinearRegression"
        assert isinstance(estimator.func, LinearRegression)

    def test_init_estimator_with_reference(self):
        estimator = Estimator("LinearRegression", LinearRegression(), "sklearn")
        assert estimator.name == "LinearRegression"
        assert isinstance(estimator.func, LinearRegression)
        assert estimator.reference == "sklearn"


class TestMetric:
    def test_init_metric(self):
        metric = Metric("mean_squared_error", mean_squared_error)
        assert metric.name == "mean_squared_error"
        assert metric.func == mean_squared_error

    def test_init_metric_with_reference(self):
        metric = Metric("mean_squared_error", mean_squared_error, "sklearn")
        assert metric.name == "mean_squared_error"
        assert metric.func == mean_squared_error
        assert metric.reference == "sklearn"


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
        assert model.estimator.name == "LinearRegression"
        assert isinstance(model.estimator.func, LinearRegression)

    def test_init_estimator_ridge(self):
        model = LinearModels("RidgeRegression")
        assert model.estimator.name == "RidgeRegression"
        assert isinstance(model.estimator.func, Ridge)

    def test_init_estimator_lasso(self):
        model = LinearModels("LassoRegression")
        assert model.estimator.name == "LassoRegression"
        assert isinstance(model.estimator.func, Lasso)

    def test_init_estimator_elasticnet(self):
        model = LinearModels("ElasticNetRegression")
        assert model.estimator.name == "ElasticNetRegression"
        assert isinstance(model.estimator.func, ElasticNet)

    def test_init_estimator_lars(self):
        model = LinearModels("LeastAngleRegression")
        assert model.estimator.name == "LeastAngleRegression"
        assert isinstance(model.estimator.func, Lars)

    def test_init_with_estimator_instance(self):
        estimator = Estimator("LinearRegression", LinearRegression())
        model = LinearModels(estimator)
        assert model.estimator.name == "LinearRegression"
        assert isinstance(model.estimator.func, LinearRegression)

    def test_init_with_invalid_name(self):
        with pytest.raises(ValueError) as e:
            LinearModels("LinearRegressionInvalid")
        assert "estimator LinearRegressionInvalid not supported" in str(e.value)

    def test_init_with_invalid_instance(self):
        with pytest.raises(ValueError) as e:
            LinearModels(LinearRegression())
        assert "model must be a string or a Estimator instance" in str(e.value)

    def test_calculate_metric_rmse(self, simple_dataset):
        x_train, x_test, y_train, y_test = simple_dataset
        model = LinearModels("LinearRegression")
        results = model.calculate_metric(
            x_train, x_test, y_train, y_test, "root_mean_squared_error"
        )
        assert model.metric.name == "root_mean_squared_error"
        assert model.metric.func == root_mean_squared_error
        np.testing.assert_almost_equal(results, 0.0, decimal=3)

    def test_calculate_metric_mse(self, simple_dataset):
        x_train, x_test, y_train, y_test = simple_dataset
        model = LinearModels("LinearRegression")
        results = model.calculate_metric(
            x_train, x_test, y_train, y_test, "mean_squared_error"
        )
        assert model.metric.name == "mean_squared_error"
        assert model.metric.func == mean_squared_error
        np.testing.assert_almost_equal(results, 0.0, decimal=3)

    def test_calculate_metric_r2(self, simple_dataset):
        x_train, x_test, y_train, y_test = simple_dataset
        model = LinearModels("LinearRegression")
        results = model.calculate_metric(x_train, x_test, y_train, y_test, "r2_score")
        assert model.metric.name == "r2_score"
        assert model.metric.func == r2_score
        np.testing.assert_almost_equal(results, 1.0, decimal=3)

    def test_calculate_metric_mae(self, simple_dataset):
        x_train, x_test, y_train, y_test = simple_dataset
        model = LinearModels("LinearRegression")
        results = model.calculate_metric(
            x_train, x_test, y_train, y_test, "mean_absolute_error"
        )
        assert model.metric.name == "mean_absolute_error"
        assert model.metric.func == mean_absolute_error
        np.testing.assert_almost_equal(results, 0.0, decimal=3)

    def test_calculate_metric_with_invalid_metric(self, simple_dataset):
        x_train, x_test, y_train, y_test = simple_dataset
        model = LinearModels("LinearRegression")
        with pytest.raises(ValueError) as e:
            model.calculate_metric(x_train, x_test, y_train, y_test, "r2_score_invalid")
        assert "metric r2_score_invalid not supported" in str(e.value)
