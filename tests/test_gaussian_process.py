import math
from functools import partial
import numpy as np
import pytest
import torch
from sklearn.utils.estimator_checks import parametrize_with_checks
from fowt_ml.gaussian_process import MultitaskGPModelApproximate
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
        params = {
            "num_inducing": 25,
            "num_latents": 1,
        }
        model = SparseGaussianModel("SklearnGPRegressor", **params)
        assert isinstance(model.estimator, SklearnGPRegressor)

    def test_init_estimator_invalid(self):
        with pytest.raises(ValueError) as e:
            SparseGaussianModel("InvalidEstimator")
        assert "Available estimators" in str(e.value)

    def test_init_with_estimator_instance(self):
        estimator = SklearnGPRegressor(25, 1)
        model = SparseGaussianModel(estimator)
        assert isinstance(model.estimator, SklearnGPRegressor)

    def test_init_with_kwargs(self):
        params = {
            "num_inducing": 25,
            "num_latents": 1,
            "batch_size": 10,
        }
        model = SparseGaussianModel("SklearnGPRegressor", **params)
        assert model.estimator.batch_size == 10

    def test_calculate_score_rmse(self, simple_dataset):
        x_train, x_test, y_train, y_test = simple_dataset

        params = {
            "num_inducing": 25,
            "num_latents": 1,
            "num_epochs": 1,
            "batch_size": 10,
        }
        model = SparseGaussianModel("SklearnGPRegressor", **params)
        results = model.calculate_score(
            x_train, x_test, y_train, y_test, "neg_mean_squared_error"
        )
        np.testing.assert_almost_equal(results, -1.102, decimal=3)


def dummy_estimator():
    return SklearnGPRegressor(100, 1, num_epochs=1, batch_size=10)


@parametrize_with_checks([dummy_estimator()])
def test_sklearn_compatibility(estimator, check):
    # some of these checks assmues predict returns 1D y, not multi-output and
    # some of these checks may break due to GPyTorch-specific behavior
    exclude_checks = [
        "check_regressors_train",
        "check_fit_idempotent",
        "check_fit_score_takes_y",
        "check_supervised_y_2d",
        "check_regressor_data_not_an_array",
        "check_regressors_int",
        "check_dtypes",
        "check_n_features_in_after_fitting",
        "check_dtype_object",
    ]
    check_name = check.func.__name__ if isinstance(check, partial) else check.__name__
    if check_name not in exclude_checks:
        check(estimator)


class TestSklearnGPRegressor:
    def test_init(self):
        model = SklearnGPRegressor(num_inducing=25, num_latents=1)
        assert model.num_inducing == 25
        assert model.num_latents == 1

    def test_fit(self, simple_dataset):
        x_train, _, y_train, _ = simple_dataset
        model = SklearnGPRegressor(num_inducing=25, num_latents=1)
        model.fit(x_train, y_train)
        assert hasattr(model, "model_")
        assert hasattr(model, "likelihood_")
        assert hasattr(model, "n_features_in_")
        assert model.is_fitted_

    def test_predict(self, simple_dataset):
        x_train, x_test, y_train, _ = simple_dataset
        model = SklearnGPRegressor(num_inducing=25, num_latents=1)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        assert predictions.shape == (50, 4)


class TestMultitaskGPModelApproximate:
    def test_init(self):
        inducing_points = torch.rand(25, 2)
        model = MultitaskGPModelApproximate(
            inducing_points=inducing_points,
            num_latents=1,
            num_tasks=4,
        )
        assert hasattr(model, "variational_strategy")
        assert hasattr(model, "covar_module")
        assert hasattr(model, "mean_module")
        assert hasattr(model, "likelihood")
        assert model.variational_strategy.num_latents == 1
        assert model.variational_strategy.num_tasks == 4
        assert model.mean_module(inducing_points).shape == torch.Size([1, 25])
        assert model.covar_module(inducing_points).shape == torch.Size([1, 25, 25])

    def test_forward(self):
        inducing_points = torch.rand(15, 2)
        model = MultitaskGPModelApproximate(
            inducing_points=inducing_points,
            num_latents=1,
            num_tasks=4,
        )
        assert model.forward(inducing_points).loc.shape == torch.Size([1, 15])
