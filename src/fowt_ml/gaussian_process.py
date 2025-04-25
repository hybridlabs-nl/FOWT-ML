"""Module for sparse Gaussian process for multi-output regeression problem."""

from collections.abc import Iterable
from typing import Any
import gpytorch
import pandas as pd
import torch
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.metrics import check_scoring
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultitaskGPModelApproximate(gpytorch.models.ApproximateGP):
    """Multitask GP model with approximate inference.

    This module models similarities/correlation in the outputs simultaneously. Each
    output dimension (task) is the linear combination of some latent function. Base
    on example
    https://docs.gpytorch.ai/en/stable/examples/04_Variational_and_Approximate_GPs/SVGP_Multitask_GP_Regression.html#Types-of-Variational-Multitask-Models
    """
    def __init__(self, inducing_points, num_latents, num_tasks):
        # Variational distribution + strategy: posterior for latent GPs
        # CholeskyVariationalDistribution: modeling a full covariance (not
        # diagonal), so it can capture dependencies between inducing points
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(1), # first dim is latents
            batch_shape=torch.Size([num_latents])
        )

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
            latent_dim=-1
        )
        super().__init__(variational_strategy)

        # covariance module: kernel: Prior information about latents
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )
        # Mean module
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_latents])
        )

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=num_tasks).to(device)

    def forward(self, x):
        """Forward pass of the model."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SklearnGPRegressor(BaseEstimator, RegressorMixin):
    """Sklearn Wrapper for MultitaskGPModelApproximate."""
    def __init__(self, inducing_points, num_latents, num_tasks,
                 num_epochs=10, batch_size=1024, learning_rate=0.01):
        self.inducing_points = inducing_points.to(device)
        self.num_latents = num_latents
        self.num_tasks = num_tasks
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device

        self.model = None
        self.likelihood = None

    def _initialize_model(self):
        self.model = MultitaskGPModelApproximate(
            inducing_points=self.inducing_points,
            num_latents=self.num_latents,
            num_tasks=self.num_tasks,
        ).to(self.device)

        self.likelihood = self.model.likelihood  # already initialized inside model

    def fit(self, x_train: ArrayLike, y_train: ArrayLike
        ) -> "SklearnGPRegressor":
        """Fit the model to the training data."""
        x_train, y_train = _get_tensorlike(x_train), _get_tensorlike(y_train)

        self._initialize_model()
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
            )
        # marginal log likelihood (mll)
        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model, num_data=x_train.size(0)
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
                output = self.model(x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # normalize the loss per data point and output dimension because it
            # gives a better idea of the loss in print statement
            ave_loss = total_loss / (x_train.size(0) * y_train.size(1))

            # TODO pass this print to logger in mlflow
            print(f"Epoch {epoch + 1}/{self.num_epochs} - Loss: {ave_loss:.3f}")

        return self

    def predict(self, x_array: ArrayLike) -> ArrayLike:
        """Make predictions using the trained model."""
        self.model.eval()
        self.likelihood.eval()

        x_array = _get_tensorlike(x_array)

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
                predictions = self.likelihood(self.model(x_batch))
                all_preds.append(predictions.mean.cpu())

        return torch.cat(all_preds, dim=0).numpy()

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
                raise ValueError(
                    f"Available estimators: {self.ESTIMATOR_NAMES.keys()}"
                )
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
    if isinstance(array, torch.Tensor):
        return array
    elif isinstance(array, Iterable):
        if isinstance(array, pd.DataFrame):
            return torch.tensor(array.values, dtype=torch.float32)
        else:
            return torch.tensor(array, dtype=torch.float32)
    else:
        raise ValueError("Input must be a numpy array or a list.")
