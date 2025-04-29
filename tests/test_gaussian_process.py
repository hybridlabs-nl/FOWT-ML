import math
import numpy as np
import pytest
import torch
from fowt_ml.gaussian_process import SklearnGPRegressor
from fowt_ml.gaussian_process import SparseGaussianModel

# Set the random seed for reproducibility
torch.manual_seed(42)


@pytest.fixture
def simple_dataset():
    """Create a simple dataset for testing."""
    x_train = torch.rand(100, 2)

    a = x_train[:, 0] * (2 * math.pi)
    b = x_train[:, 1] * (2 * math.pi)
    y_train = torch.stack(
        [
            torch.sin(a) + torch.randn(100) * 0.2,
            torch.cos(a) + torch.randn(100) * 0.2,
            torch.sin(b) + 2 * torch.cos(b) + torch.randn(100) * 0.2,
            -torch.cos(b) + torch.randn(100) * 0.2,
        ],
        -1,
    )

    x_test = torch.rand(50, 2)
    a = x_test[:, 0] * (2 * math.pi)
    b = x_test[:, 1] * (2 * math.pi)
    y_test = torch.stack(
        [
            torch.sin(a) + torch.randn(50) * 0.2,
            torch.cos(a) + torch.randn(50) * 0.2,
            torch.sin(b) + 2 * torch.cos(b) + torch.randn(50) * 0.2,
            -torch.cos(b) + torch.randn(50) * 0.2,
        ],
        -1,
    )

    return x_train, x_test, y_train, y_test


class TestSparseGaussianModel:
    def test_init_estimator_sklearn(self):
        model = SparseGaussianModel("SklearnGPRegressor")
        assert isinstance(model.estimator, SklearnGPRegressor)

    def test_init_estimator_invalid(self):
        with pytest.raises(ValueError) as e:
            SparseGaussianModel("InvalidEstimator")
        assert "Available estimators" in str(e.value)

    def test_init_with_estimator_instance(self):
        model = SparseGaussianModel(SklearnGPRegressor())
        assert isinstance(model.estimator, SklearnGPRegressor)

    def test_init_with_kwargs(self):
        model = SparseGaussianModel("SklearnGPRegressor", num_latents=3)
        assert model.estimator.num_latents == 3

    def test_calculate_score_rmse(self, simple_dataset):
        x_train, x_test, y_train, y_test = simple_dataset

        inducing_points = x_train[torch.randperm(x_train.size(0))[:25]]
        inducing_points = inducing_points.unsqueeze(0).repeat(1, 1, 1).clone()

        params = {
            "inducing_points": inducing_points,
            "num_latents": 1,
            "num_tasks": 4,
            "num_epochs": 1,
            "batch_size": 10,
        }
        model = SparseGaussianModel("SklearnGPRegressor", **params)
        results = model.calculate_score(
            x_train, x_test, y_train, y_test, "neg_mean_squared_error"
        )
        np.testing.assert_almost_equal(results, -1.1028, decimal=3)
