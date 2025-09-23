"""Documentation about fowt_ml."""

import logging
from fowt_ml.config import Config
from fowt_ml.ensemble import EnsembleModel
from fowt_ml.gaussian_process import SparseGaussianModel
from fowt_ml.linear_models import LinearModels
from fowt_ml.neural_network import NeuralNetwork
from fowt_ml.xgboost import XGBoost

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Sarah Alidoost"
__email__ = "f.alidoost@esciencecenter.nl"
__version__ = "0.1.0"

__all__ = [
    "LinearModels",
    "XGBoost",
    "EnsembleModel",
    "SparseGaussianModel",
    "NeuralNetwork",
    "Config",
]
