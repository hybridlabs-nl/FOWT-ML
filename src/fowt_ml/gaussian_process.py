"""Module for sparse Gaussian process for multi-output regeression problem."""

from collections.abc import Iterable
from logging import Logger
import gpytorch
import pandas as pd
import torch
from numpy.typing import ArrayLike
from skorch.probabilistic import GPRegressor
from fowt_ml.base import BaseModel

logger = Logger(__name__)

# Set the random seed for reproducibility
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultitaskGPModelApproximate(gpytorch.models.ApproximateGP):
    """Multitask GP model with approximate inference.

    This module models similarities/correlation in the outputs simultaneously. Each
    output dimension (task) is the linear combination of some latent function. Base
    on example
    https://docs.gpytorch.ai/en/stable/examples/04_Variational_and_Approximate_GPs/SVGP_Multitask_GP_Regression.html#Types-of-Variational-Multitask-Models
    """

    def __init__(self, inducing_points, num_latents, num_tasks):
        # convert inducing points to tensor
        inducing_points = _to_tensor(inducing_points, dtype="float32", device=DEVICE)

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
        self.covar = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents]),
        )
        # Mean module
        self.mean = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_latents]),
        )

    def forward(self, x):
        """Forward pass of the model."""
        mean_x = self.mean(x)
        covar_x = self.covar(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def SklearnGPRegressor(  # noqa: N802
    inducing_points: ArrayLike,
    num_latents: int,
    num_tasks: int,
    num_training_samples: int,
    **kwargs,
) -> GPRegressor:
    """Create a skorch GPRegressor with skorch.

    see
    https://skorch.readthedocs.io/en/latest/user/probabilistic.html#approximate-gaussian-processes
    """
    params = dict(
        module=MultitaskGPModelApproximate,
        module__inducing_points=inducing_points,
        module__num_latents=num_latents,
        module__num_tasks=num_tasks,
        criterion__num_data=num_training_samples,
        likelihood=gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=num_tasks
        ),
        verbose=0,
    )
    params.update(kwargs)
    return GPRegressor(**params)


class SparseGaussianModel(BaseModel):
    """Class to handle sparse Gaussian process regression."""

    ESTIMATOR_NAMES = {
        "SklearnGPRegressor": SklearnGPRegressor,
    }

    GP_LIKE_NAMES = {"SklearnGPRegressor"}

    @classmethod
    def is_gp_like(cls, model_name):
        """Check if the model is GP-like."""
        return model_name in cls.GP_LIKE_NAMES


def _to_tensor(
    array: ArrayLike | pd.DataFrame, dtype: str = "float32", device: str = "cpu"
) -> torch.Tensor:
    """Convert numpy array to tensor."""
    if dtype == "float32":
        dtype = torch.float32
    elif dtype == "float64":
        dtype = torch.float64
    else:
        raise ValueError("dtype must be 'float32' or 'float64'.")

    if isinstance(array, torch.Tensor):
        return array.to(dtype=dtype, device=device)
    elif isinstance(array, pd.DataFrame):
        return torch.tensor(array.values, dtype=dtype).to(device)
    elif isinstance(array, Iterable):
        return torch.tensor(array, dtype=dtype).to(device)
    else:
        raise ValueError("Input must be ArrayLike or pd.DataFrame.")


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
