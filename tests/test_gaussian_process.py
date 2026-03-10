import math
import pytest
import torch
from skorch.probabilistic import GPRegressor
from fowt_ml.gaussian_process import MultitaskGPModelApproximate
from fowt_ml.gaussian_process import SklearnGPRegressor
from fowt_ml.gaussian_process import SparseGaussianModel

# Set the random seed for reproducibility
torch.manual_seed(42)


@pytest.fixture
def simple_dataset():
    """Create a simple dataset for testing."""
    x_train = torch.rand(100, 4)

    a = x_train[:, 0] * (2 * math.pi)
    y_train = torch.stack(
        [
            torch.sin(a) + torch.randn(100) * 0.2,
            torch.cos(a) + torch.randn(100) * 0.2,
        ],
        -1,
    )

    x_test = torch.rand(50, 4)
    a = x_test[:, 0] * (2 * math.pi)
    y_test = torch.stack(
        [
            torch.sin(a) + torch.randn(50) * 0.2,
            torch.cos(a) + torch.randn(50) * 0.2,
        ],
        -1,
    )
    return x_train, x_test, y_train, y_test


class TestSparseGaussianModel:
    def test_init_estimator_sklearn(self, simple_dataset):
        x_train, x_test, y_train, y_test = simple_dataset

        params = {
            "inducing_points": x_train[:25, :],
            "num_latents": 1,
            "num_tasks": y_train.shape[1],
            "num_training_samples": x_train.shape[0],
        }
        model = SparseGaussianModel("SklearnGPRegressor", **params)
        assert isinstance(model.estimator, GPRegressor)

    def test_init_estimator_invalid(self):
        with pytest.raises(ValueError) as e:
            SparseGaussianModel("InvalidEstimator")
        assert "Available estimators" in str(e.value)

    def test_init_with_estimator_instance(self, simple_dataset):
        x_train, x_test, y_train, y_test = simple_dataset
        estimator = SklearnGPRegressor(
            x_train[:25, :], 1, y_train.shape[1], x_train.shape[0]
        )
        model = SparseGaussianModel(estimator)
        assert isinstance(model.estimator, GPRegressor)

    def test_init_with_kwargs(self, simple_dataset):
        x_train, x_test, y_train, y_test = simple_dataset
        params = {
            "inducing_points": x_train[:25, :],
            "num_latents": 1,
            "num_tasks": y_train.shape[1],
            "num_training_samples": x_train.shape[0],
            "batch_size": 10,
        }
        model = SparseGaussianModel("SklearnGPRegressor", **params)
        assert model.estimator.batch_size == 10

    def test_calculate_score_rmse(self, simple_dataset):
        x_train, x_test, y_train, y_test = simple_dataset
        params = {
            "inducing_points": x_train[:25, :],
            "num_latents": 1,
            "num_tasks": y_train.shape[1],
            "num_training_samples": x_train.shape[0],
        }
        model = SparseGaussianModel("SklearnGPRegressor", **params)
        results = model.calculate_score(
            x_train, x_test, y_train, y_test, "neg_mean_squared_error"
        )
        assert "neg_mean_squared_error" in results


class TestMultitaskGPModelApproximate:
    def test_init(self):
        inducing_points = torch.rand(25, 2)
        model = MultitaskGPModelApproximate(
            inducing_points=inducing_points,
            num_latents=1,
            num_tasks=4,
        )
        assert hasattr(model, "variational_strategy")
        assert hasattr(model, "covar")
        assert hasattr(model, "mean")
        assert model.variational_strategy.num_latents == 1
        assert model.variational_strategy.num_tasks == 4
        assert model.mean(inducing_points).shape == torch.Size([1, 25])
        assert model.covar(inducing_points).shape == torch.Size([1, 25, 25])

    def test_forward(self):
        inducing_points = torch.rand(15, 2)
        model = MultitaskGPModelApproximate(
            inducing_points=inducing_points,
            num_latents=1,
            num_tasks=4,
        )
        assert model.forward(inducing_points).loc.shape == torch.Size([1, 15])
