import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from fowt_ml.linear_models import Estimator
from fowt_ml.linear_models import LinearModels


class TestLinearModel:
    def test_init_estimator_name(self):
        model = LinearModels("LinearRegression")
        assert model.estimator.name == "LinearRegression"
        assert isinstance(model.estimator.func, LinearRegression)

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

    def test_calculate_metric_name(self):
        # create small dataset
        x_train = [[1, 2], [3, 4], [5, 6]]
        x_test = [[7, 8], [9, 10], [11, 12]]
        y_train = [1, 2, 3]
        y_test = [4, 5, 6]
        model = LinearModels("LinearRegression")
        results = model.calculate_metric(
            x_train, x_test, y_train, y_test, "root_mean_squared_error"
            )
        assert model.metric.name == "root_mean_squared_error"
        assert model.metric.func == root_mean_squared_error
        np.testing.assert_almost_equal(results, 0.0, decimal=3)

    def test_calculate_metric_with_invalid_metric(self):
        # create small dataset
        x_train = [[1, 2], [3, 4], [5, 6]]
        x_test = [[7, 8], [9, 10], [11, 12]]
        y_train = [1, 2, 3]
        y_test = [4, 5, 6]
        model = LinearModels("LinearRegression")
        with pytest.raises(ValueError) as e:
            model.calculate_metric(x_train, x_test, y_train, y_test, "r2_score_invalid")
        assert "metric r2_score_invalid not supported" in str(e.value)
