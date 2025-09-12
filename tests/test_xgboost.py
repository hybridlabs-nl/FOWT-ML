import pytest
import xgboost as xgb
from sklearn.base import BaseEstimator
from fowt_ml.xgboost import XGBoost


@pytest.fixture
def simple_dataset():
    """Create a simple dataset for testing."""
    x_train = [[1, 2], [3, 4], [5, 6]]
    x_test = [[7, 8], [9, 10], [11, 12]]
    y_train = [1, 2, 3]
    y_test = [4, 5, 6]
    return x_train, x_test, y_train, y_test


class TestXGBoost:
    def test_init_estimator_lr(self):
        model = XGBoost("XGBoostRegression")
        assert isinstance(model.estimator, xgb.XGBRegressor)

    def test_init_with_estimator_instance(self):
        model = XGBoost(xgb.XGBRegressor())
        assert isinstance(model.estimator, BaseEstimator)

    def test_init_estimator_kwargs(self):
        model = XGBoost("XGBoostRegression", tree_method="hist")
        assert model.estimator.tree_method == "hist"

    def test_calculate_score_rmse(self, simple_dataset):
        x_train, x_test, y_train, y_test = simple_dataset
        model = XGBoost("XGBoostRegression")
        results = model.calculate_score(x_train, x_test, y_train, y_test, "r2")
        assert "r2" in results

    def test_calculate_score_more(self, simple_dataset):
        x_train, x_test, y_train, y_test = simple_dataset
        model = XGBoost("XGBoostRegression")
        results = model.calculate_score(
            x_train,
            x_test,
            y_train,
            y_test,
            ["neg_mean_squared_error", "model_fit_time"],
        )
        assert "neg_mean_squared_error" in results
        assert "model_fit_time" in results
