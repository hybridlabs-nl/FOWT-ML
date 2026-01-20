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
        # x can be (batch, input_size) or (batch, seq_len, input_size)

        if x.dim() == 2:
            x = x.unsqueeze(1)  # add seq_len=1

        out, _ = self.rnn(x)  # (batch, seq_len, hidden)
        out = out[:, -1, :]  # take last time step (batch, hidden)
        out_fc = self.fc(out)  # (batch, output)
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
    return skorch.regressor.NeuralNetRegressor(**params, train_split=None)


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

    RNN_LIKE_NAMES = {"RNNRegressor", "LSTMRegressor", "GRURegressor"}

    @classmethod
    def is_rnn_like(cls, model_name):
        """Check if the model is RNN-like."""
        return model_name in cls.RNN_LIKE_NAMES
