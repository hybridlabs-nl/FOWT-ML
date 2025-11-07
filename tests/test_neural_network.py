import pytest
import sklearn.neural_network as nn
import skorch
import torch
from sklearn.base import BaseEstimator
from fowt_ml.neural_network import NeuralNetwork


@pytest.fixture
def simple_dataset():
    """Create a simple dataset for testing."""
    x_train = [[1, 2], [3, 4], [5, 6]]
    x_test = [[7, 8], [9, 10], [11, 12]]
    y_train = [1, 2, 3]
    y_test = [4, 5, 6]
    return x_train, x_test, y_train, y_test


@pytest.fixture
def tensor_dataset():
    """Create a simple torch tensor dataset for testing RNNs."""
    x_train = torch.randn(50, 3)
    x_test = torch.randn(10, 3)
    y_train = torch.randn(50, 1)
    y_test = torch.randn(10, 1)
    return x_train, x_test, y_train, y_test


class TestMLP:
    def test_init_estimator_mlp(self):
        model = NeuralNetwork("MultilayerPerceptron")
        assert isinstance(model.estimator, nn.MLPRegressor)

    def test_init_with_estimator_instance(self):
        model = NeuralNetwork(nn.MLPRegressor())
        assert isinstance(model.estimator, BaseEstimator)

    def test_init_estimator_kwargs(self):
        model = NeuralNetwork("MultilayerPerceptron", hidden_layer_sizes=10)
        assert model.estimator.hidden_layer_sizes == 10

    def test_calculate_score_rmse(self, simple_dataset):
        x_train, x_test, y_train, y_test = simple_dataset
        model = NeuralNetwork("MultilayerPerceptron")
        results = model.calculate_score(x_train, x_test, y_train, y_test, "r2")
        assert "r2" in results

    def test_calculate_score_more(self, simple_dataset):
        x_train, x_test, y_train, y_test = simple_dataset
        model = NeuralNetwork("MultilayerPerceptron")
        results = model.calculate_score(
            x_train,
            x_test,
            y_train,
            y_test,
            ["neg_mean_squared_error", "model_fit_time"],
        )
        assert "neg_mean_squared_error" in results
        assert "model_fit_time" in results


class TestRNN:
    def test_init_estimator_rnn(self):
        params = {
            "input_size": 3,
            "hidden_size": 5,
            "output_size": 1,
            "num_layers": 1,
            "max_epochs": 5,
        }
        model = NeuralNetwork("RNNRegressor", **params)
        assert isinstance(model.estimator, skorch.regressor.NeuralNetRegressor)

    def test_init_estimator_kwargs(self):
        model = NeuralNetwork(
            "RNNRegressor", input_size=10, hidden_size=10, output_size=1, num_layers=1
        )
        assert model.estimator.module__hidden_size == 10

    def test_calculate_score_rmse(self, tensor_dataset):
        x_train, x_test, y_train, y_test = tensor_dataset
        params = {
            "input_size": 3,
            "hidden_size": 5,
            "output_size": 1,
            "num_layers": 1,
            "max_epochs": 5,
        }
        model = NeuralNetwork("RNNRegressor", **params)
        results = model.calculate_score(x_train, x_test, y_train, y_test, "r2")
        assert "r2" in results

    def test_cv(self, tensor_dataset):
        x_train, _, y_train, _ = tensor_dataset
        params = {
            "input_size": 3,
            "hidden_size": 5,
            "output_size": 1,
            "num_layers": 1,
            "max_epochs": 5,
        }
        model = NeuralNetwork("RNNRegressor", **params)
        results = model.cross_validate(x_train, y_train, "r2", cv=3)
        assert "r2" in results


class TestLSTM:
    def test_init_estimator_rnn(self):
        params = {
            "input_size": 3,
            "hidden_size": 5,
            "output_size": 1,
            "num_layers": 1,
            "max_epochs": 5,
        }
        model = NeuralNetwork("LSTMRegressor", **params)
        assert isinstance(model.estimator, skorch.regressor.NeuralNetRegressor)

    def test_init_estimator_kwargs(self):
        model = NeuralNetwork(
            "LSTMRegressor", input_size=10, hidden_size=10, output_size=1, num_layers=1
        )
        assert model.estimator.module__output_size == 1

    def test_calculate_score_rmse(self, tensor_dataset):
        x_train, x_test, y_train, y_test = tensor_dataset
        params = {
            "input_size": 3,
            "hidden_size": 5,
            "output_size": 1,
            "num_layers": 1,
            "max_epochs": 5,
        }
        model = NeuralNetwork("LSTMRegressor", **params)
        results = model.calculate_score(x_train, x_test, y_train, y_test, "r2")
        assert "r2" in results


class TestGRU:
    def test_init_estimator_rnn(self):
        params = {
            "input_size": 3,
            "hidden_size": 5,
            "output_size": 1,
            "num_layers": 1,
            "max_epochs": 5,
        }
        model = NeuralNetwork("GRURegressor", **params)
        assert isinstance(model.estimator, skorch.regressor.NeuralNetRegressor)

    def test_init_estimator_kwargs(self):
        model = NeuralNetwork(
            "GRURegressor", input_size=10, hidden_size=10, output_size=1, num_layers=1
        )
        assert model.estimator.module__input_size == 10

    def test_calculate_score_rmse(self, tensor_dataset):
        x_train, x_test, y_train, y_test = tensor_dataset
        params = {
            "input_size": 3,
            "hidden_size": 5,
            "output_size": 1,
            "num_layers": 1,
            "max_epochs": 5,
        }
        model = NeuralNetwork("GRURegressor", **params)
        results = model.calculate_score(x_train, x_test, y_train, y_test, "r2")
        assert "r2" in results
