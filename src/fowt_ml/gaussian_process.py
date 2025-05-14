"""Module for sparse Gaussian process for multi-output regeression problem."""

from collections.abc import Iterable
from logging import Logger
from typing import Any
import gpytorch
import pandas as pd
import torch
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.metrics import check_scoring
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

logger = Logger(__name__)


class MultitaskGPModelApproximate(gpytorch.models.ApproximateGP):
    """Multitask GP model with approximate inference.

    This module models similarities/correlation in the outputs simultaneously. Each
    output dimension (task) is the linear combination of some latent function. Base
    on example
    https://docs.gpytorch.ai/en/stable/examples/04_Variational_and_Approximate_GPs/SVGP_Multitask_GP_Regression.html#Types-of-Variational-Multitask-Models
    """

    def __init__(self, inducing_points, num_latents, num_tasks):
        # convert inducing points to tensor
        inducing_points = _get_tensorlike(inducing_points)

        # Variational distribution + strategy: posterior for latent GPs
        # CholeskyVariationalDistribution: modeling a full covariance (not
        # diagonal), so it can capture dependencies between inducing points
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0),
            batch_shape=torch.Size([num_latents]),
        )

        # Check inducing points shape before passing to VariationalStrategy
        inducing_points = _check_inducing_points(inducing_points, num_latents)

        # model correlations across tasks (or outputs)
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1,
        )
        super().__init__(variational_strategy)

        # covariance module: kernel: Prior information about latents
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents]),
        )
        # Mean module
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_latents]),
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=num_tasks
        ).to(device)

    def forward(self, x):
        """Forward pass of the model."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SklearnGPRegressor(RegressorMixin, BaseEstimator):
    """Sklearn Wrapper for MultitaskGPModelApproximate."""

    def __init__(
        self,
        num_inducing,
        num_latents,
        num_epochs=10,
        batch_size=1024,
        learning_rate=0.01,
    ):
        self.num_inducing = num_inducing
        self.num_latents = num_latents
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def fit(self, x_train: ArrayLike, y_train: ArrayLike) -> "SklearnGPRegressor":
        """Fit the model to the training data."""
        # Check that X and y have correct shape
        x_train, y_train = check_X_y(x_train, y_train, multi_output=True)

        x_train = _get_tensorlike(x_train)
        y_train = _get_tensorlike(y_train)

        # add some sklearn variables
        self.X_ = x_train
        self.y_ = y_train
        self.n_features_in_ = x_train.shape[1]

        # initialize model
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        inducing_points = x_train[torch.randperm(x_train.size(0))[: self.num_inducing]]

        self.model_ = MultitaskGPModelApproximate(
            inducing_points=inducing_points,
            num_latents=self.num_latents,
            num_tasks=y_train.size(1),
        ).to(device)

        self.likelihood_ = self.model_.likelihood

        # Train the model
        self.model_.train()
        self.likelihood_.train()

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        # marginal log likelihood (mll)
        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood_, self.model_, num_data=x_train.size(0)
        )

        # TODO optimize the loops
        for epoch in range(self.num_epochs):
            total_loss = 0
            if self.batch_size:  # Use batching if batch_size is set
                batches = DataLoader(
                    TensorDataset(x_train, y_train),
                    batch_size=self.batch_size,
                )
            else:  # Treat entire dataset as one batch
                batches = [(x_train, y_train)]
            for x_batch, y_batch in batches:
                optimizer.zero_grad()
                output = self.model_(x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # normalize the loss per data point and output dimension because it
            # gives a better idea of the loss in log
            ave_loss = total_loss / (x_train.size(0) * y_train.size(1))
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs} - Loss: {ave_loss:.3f}")

        self.is_fitted_ = True
        return self

    def predict(self, x_array: ArrayLike) -> ArrayLike:
        """Make predictions using the trained model."""
        # Check if the model has been fitted
        check_is_fitted(self, ["is_fitted_", "model_", "likelihood_"])

        # Check that X has correct shape
        x_array = check_array(x_array)

        # Check number of features
        if x_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, "
                f"but got {x_array.shape[1]} features."
            )
        x_array = _get_tensorlike(x_array)

        self.model_.eval()
        self.likelihood_.eval()

        all_preds = []
        with torch.no_grad():
            if self.batch_size:  # Use batching if batch_size is set
                batches = DataLoader(
                    TensorDataset(x_array),
                    batch_size=self.batch_size,
                )
            else:  # Treat entire dataset as one batch
                batches = [(x_array,)]
            for (x_batch,) in batches:
                predictions = self.likelihood_(self.model_(x_batch))
                all_preds.append(predictions.mean.cpu())

        return torch.cat(all_preds, dim=0).numpy()

    def score(self, x, y):
        """Return the R^2 score of the prediction."""
        y_pred = self.predict(x)
        return r2_score(y, y_pred)


class SparseGaussianModel:
    """Class to handle sparse Gaussian process regression."""

    ESTIMATOR_NAMES = {
        "SklearnGPRegressor": SklearnGPRegressor,
    }

    def __init__(
        self, estimator: str | BaseEstimator, **kwargs: dict[str, Any]
    ) -> None:
        if isinstance(estimator, str):
            if estimator not in self.ESTIMATOR_NAMES:
                raise ValueError(f"Available estimators: {self.ESTIMATOR_NAMES.keys()}")
            self.estimator = self.ESTIMATOR_NAMES[estimator](**kwargs)
        else:
            self.estimator = estimator.set_params(**kwargs)

    def calculate_score(
        self,
        x_train: ArrayLike,
        x_test: ArrayLike,
        y_train: ArrayLike,
        y_test: ArrayLike,
        scoring: str | Iterable | None = None,
    ) -> float | ArrayLike:
        """Fit and calculate a score.

        In multi-output regression, by default, 'uniform_average' is used,
        which specifies a uniformly weighted mean over outputs. see
        https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics

        For scoring paramers overview:
        https://scikit-learn.org/stable/modules/model_evaluation.html#string-name-scorers
        """  # noqa: E501
        self.estimator.fit(x_train, y_train)
        scorer = check_scoring(self.estimator, scoring=scoring)
        return scorer(self.estimator, x_test, y_test)


def _get_tensorlike(array: ArrayLike) -> torch.Tensor:
    """Convert numpy array to tensor."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(array, torch.Tensor):
        return array
    elif isinstance(array, Iterable):
        if isinstance(array, pd.DataFrame):
            return torch.tensor(array.values, dtype=torch.float32).to(device)
        else:
            return torch.tensor(array, dtype=torch.float32).to(device)
    else:
        raise ValueError("Input must be a numpy array or a list.")


def _check_inducing_points(x: torch.Tensor, num_latents: int) -> torch.Tensor:
    """Check inducing points has correct shape."""
    if x.ndim == 2:
        x = x.unsqueeze(0).repeat(num_latents, 1, 1).clone()
    elif x.ndim == 3:
        if x.shape[0] != num_latents:
            raise ValueError(
                f"Inducing points should have {num_latents} latent dimensions, "
                f"but got {x.shape[0]}."
            )
    else:
        raise ValueError(
            "Inducing points should be 2D [N, D] or 3D [B, N, D] tensor. "
            f"Got {x.ndim} dimensions."
        )
    return x
