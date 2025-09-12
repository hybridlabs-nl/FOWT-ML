"""Module to handle Neural Network models."""

import sklearn.neural_network as nn
from fowt_ml.base import BaseModel


class NeuralNetwork(BaseModel):
    """Class to handle Neural Network models and metrics for comparison."""

    ESTIMATOR_NAMES = {
        "MultilayerPerceptron": nn.MLPRegressor,
    }
