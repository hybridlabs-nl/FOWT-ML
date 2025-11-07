import logging
from pathlib import Path
from typing import Any
import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.model_selection import train_test_split
from fowt_ml.config import Config
from fowt_ml.datasets import check_data
from fowt_ml.datasets import get_data
from fowt_ml.ensemble import EnsembleModel
from fowt_ml.gaussian_process import SparseGaussianModel
from fowt_ml.linear_models import LinearModels
from fowt_ml.neural_network import NeuralNetwork
from fowt_ml.xgboost import XGBoost

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: str | Config) -> None:
        """Initializes the machine learning pipeline.

        Args:
            config (str | Config): Path to the configuration file or a Config object.
            kwargs: Additional keyword arguments to override the configuration file.

        Returns:
            None
        """
        config = config if isinstance(config, Config) else Config.from_yaml(config)

        self.predictors_labels = config["ml_setup"]["predictors"]
        self.target_labels = config["ml_setup"]["targets"]
        self.model_names = config["ml_setup"]["model_names"]
        self.metric_names = config["ml_setup"]["metric_names"]
        self.train_test_split_kwargs = config["ml_setup"]["train_test_split_kwargs"]
        self.cross_validation_kwargs = config["ml_setup"]["cross_validation_kwargs"]
        self.scale_data = config["ml_setup"]["scale_data"]

        self.work_dir = Path(config["session_setup"]["work_dir"])

        self.data_config = config["data"]
        self.save_grid_scores = config["ml_setup"]["save_grid_scores"]
        self.save_best_model = config["ml_setup"]["save_best_model"]

        self.log_experiment = config["ml_setup"]["log_experiment"]

    def _setup_mlflow(self):
        mlruns_dir = self.work_dir / "mlruns"
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(mlruns_dir.resolve().as_uri())
        mlflow_experiment = mlflow.get_experiment_by_name("comparison")
        if mlflow_experiment:
            self.experiment_id = mlflow_experiment.experiment_id
        else:
            self.experiment_id = mlflow.create_experiment("comparison")

    def get_data(self, data_id: str) -> pd.DataFrame:
        """Returns the dataset for the given data_id.

        Args:
            data_id (str): ID of the data in the configuration file.

        Returns:
            pd.DataFrame: DataFrame for the given data_id, set in the
            configuration file.
        """
        return get_data(data_id, self.data_config)

    def train_test_split(self, **kwargs):
        """Splits the data into training and testing sets.

        The data should be set in self.data before calling this method.
        kwargs are passed to sklearn.model_selection.train_test_split.
        """
        if not hasattr(self, "X_data") or not hasattr(self, "Y_data"):
            raise ValueError("Data not found. Run setup before splitting.")

        return train_test_split(self.X_data, self.Y_data, **kwargs)

    def get_models(self):
        """Returns the models for the given model names.

        Returns:
            dict: Dictionary of models.
        """
        models = {}
        model_classes = [
            LinearModels,
            EnsembleModel,
            SparseGaussianModel,
            NeuralNetwork,
            XGBoost,
        ]
        for model_name, kwrags in self.model_names.items():
            for model_class in model_classes:
                if model_name in model_class.ESTIMATOR_NAMES:
                    models[model_name] = model_class(model_name, **kwrags)
                    break
            else:
                raise ValueError(f"Model {model_name} not supported.")
        return models

    def setup(self, data: pd.DataFrame | str) -> Any:
        """Set up the machine learning experiment.

        - find the data
        - train test split
        - setup the models for comparison

        Args:
            data (pd.DataFrame): DataFrame containing the data.

        Returns:
            Experiment object or similar.

        """
        if isinstance(data, str):
            data = self.get_data(data)

        # check if the data has the required columns, and valid values
        data = check_data(data, self.predictors_labels + self.target_labels)

        self.X_data = data.loc[:, self.predictors_labels]
        self.Y_data = data.loc[:, self.target_labels]

        # convert to numpy arrays for consistency between libraries
        self.X_data = np.asarray(self.X_data, dtype=np.float32)
        self.Y_data = np.asarray(self.Y_data, dtype=np.float32)

        self.X_train, self.X_test, self.Y_train, self.Y_test = self.train_test_split(
            **self.train_test_split_kwargs
        )

        # get the models
        self.model_instances = self.get_models()

        # create work directory
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # setup mlflow if logging is enabled
        if self.log_experiment:
            self._setup_mlflow()

    def _run_model(self, model_name, cross_validation: bool) -> tuple[Any, dict]:
        """Runs the models on the training data.

        Returns:
            dict: Dictionary of trained models.

        """
        model = self.model_instances[model_name]

        if self.scale_data:
            model.use_scaled_data()

        if cross_validation:
            all_scores = model.cross_validate(
                self.X_train,
                self.Y_train,
                scoring=self.metric_names,
                **self.cross_validation_kwargs,
            )
            # calculate mean of the scores
            scores = {k: v.mean() for k, v in all_scores.items()}
            model.estimator.fit(self.X_train, self.Y_train)
        else:
            scores = model.calculate_score(
                self.X_train, self.X_test, self.Y_train, self.Y_test, self.metric_names
            )
        return model.estimator, scores

    def _log_model(self):
        if self.log_experiment:
            logger.info(f"Logging experiment to MLflow with id {self.experiment_id}")
            for model_name in self.model_names:
                with mlflow.start_run(experiment_id=self.experiment_id):
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_metrics(self.scores[model_name])
                    input_example = self.X_train[:1]  # small slice of training data
                    model = self.fitted_models[model_name]
                    signature = mlflow.models.infer_signature(
                        input_example, model.predict(input_example)
                    )
                    mlflow.sklearn.log_model(
                        model,
                        model_name,
                        signature=signature,
                        input_example=input_example,
                    )

    def _save_grid_scores(self):
        if self.save_grid_scores:
            file_name = self.work_dir / "grid_scores.csv"
            logger.info(f"Saving grid scores to {file_name}")
            self.grid_scores_sorted.to_csv(file_name, index=False)

    def _save_best_model(self):
        if self.save_best_model:
            best_model_name = self.grid_scores_sorted.index[0]
            best_model = self.fitted_models[best_model_name]

            # the TransformedTargetRegressor is not supported in ONNX
            # see https://onnx.ai/sklearn-onnx/supported.html#supported-scikit-learn-models
            if self.scale_data | (best_model_name == "SklearnGPRegressor"):
                file_name = self.work_dir / "best_model.joblib"
                joblib.dump(best_model, file_name)
                logger.info(f"Saving best model to joblib format in {file_name}")
            else:
                file_name = self.work_dir / "best_model.onnx"
                _export_to_onnx(
                    best_model_name,
                    best_model,
                    no_features=self.X_train.shape[1],
                    file_name=file_name,
                )
                logger.info(f"Saving best model to ONNX format in {file_name}")

    def compare_models(self, sort: str = "r2", cross_validation: bool = False) -> Any:
        """Compares the models and returns the best model.

        "model_fit_time" is in seconds.

        Args:
            sort (str, optional): Metric to sort the models by. Defaults to "r2".
            cross_validation (bool, optional): Whether to use cross-validation
            for comparison. Defaults to False.

        Returns:
            tuple: (dict of fitted models, pd.DataFrame of grid scores sorted by `sort`)
        """
        self.fitted_models = {}
        self.scores = {}
        for model_name in self.model_names:
            fitted_model, scores = self._run_model(model_name, cross_validation)
            self.fitted_models[model_name] = fitted_model
            self.scores[model_name] = scores

        grid_scores = pd.DataFrame(self.scores).T

        if sort not in grid_scores.columns:
            raise ValueError(
                f"Sort '{sort}' not in metrics {grid_scores.columns.tolist()}"
                " provided. Choose one of the metrics to sort the models."
            )

        ascending = sort in {"model_fit_time", "model_predict_time"}
        self.grid_scores_sorted = grid_scores.sort_values(by=sort, ascending=ascending)

        self._log_model()
        self._save_grid_scores()
        self._save_best_model()

        return self.fitted_models, self.grid_scores_sorted


def _export_to_onnx(model_name, best_model, no_features, file_name):
    if model_name in {"RNNRegressor", "LSTMRegressor", "GRURegressor"}:
        torch_model = best_model.module_
        torch_model.eval()

        example_input = torch.randn(1, no_features)
        torch.onnx.export(
            torch_model,
            example_input,
            file_name,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=11,
        )
    elif model_name == "SklearnGPRegressor":
        raise NotImplementedError(
            "ONNX export for Gaussian Process models is not implemented."
        )
    else:
        initial_type = [("input", FloatTensorType([None, no_features]))]
        onnx_model = convert_sklearn(best_model, initial_types=initial_type)
        with open(file_name, "wb") as f:
            f.write(onnx_model.SerializeToString())
