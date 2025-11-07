"""Module to handle Neural Network models."""

import sklearn
import skorch
import torch
from fowt_ml.base import BaseModel


class GenericRNNModule(torch.nn.Module):
    def __init__(self, rnn_model, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.rnn = rnn_model(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass of the RNN module."""
        if x.dim() == 2:
            x = x.unsqueeze(0)  # add batch dim
        out, _ = self.rnn(x)

        out_fc = self.fc(out)  # regression on all time steps
        if out_fc.shape[0] == 1:
            out_fc = out_fc.squeeze(0)
        return out_fc


def create_skorch_regressor(
    rnn_model,
    input_size,
    hidden_size,
    output_size,
    num_layers=1,
    **kwargs,
):
    """Create a skorch NeuralNetRegressor with a specified RNN model."""
    params = dict(
        module=GenericRNNModule,
        module__rnn_model=rnn_model,
        module__input_size=input_size,
        module__hidden_size=hidden_size,
        module__output_size=output_size,
        module__num_layers=num_layers,
        verbose=0,
    )
    params.update(kwargs)
    return skorch.regressor.NeuralNetRegressor(**params)


def RNNRegressor(**args):  # noqa: N802
    """Create a skorch NeuralNetRegressor with a standard RNN model."""
    return create_skorch_regressor(torch.nn.RNN, **args)


def LSTMRegressor(**args):  # noqa: N802
    """Create a skorch NeuralNetRegressor with an LSTM model."""
    return create_skorch_regressor(torch.nn.LSTM, **args)


def GRURegressor(**args):  # noqa: N802
    """Create a skorch NeuralNetRegressor with a GRU model."""
    return create_skorch_regressor(torch.nn.GRU, **args)


class NeuralNetwork(BaseModel):
    """Class to handle Neural Network models and metrics for comparison."""

    ESTIMATOR_NAMES = {
        "MultilayerPerceptron": sklearn.neural_network.MLPRegressor,
        "RNNRegressor": RNNRegressor,
        "LSTMRegressor": LSTMRegressor,
        "GRURegressor": GRURegressor,
    }
