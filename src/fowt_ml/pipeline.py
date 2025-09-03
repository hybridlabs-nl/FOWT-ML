import logging
from pathlib import Path
from typing import Any
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from fowt_ml.config import read_yaml
from fowt_ml.datasets import get_data
from fowt_ml.linear_models import LinearModels
from fowt_ml.ensemble import EnsembleModel
from fowt_ml.gaussian_process import SparseGaussianModel

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: str | dict, **kwargs: dict[str, Any]) -> None:
        """Initializes the machine learning pipeline.

        Args:
            config (str | dict): Path to the configuration file or a dictionary
        """
        config = config if isinstance(config, dict) else read_yaml(config)
        config.update(kwargs)
        #TODO: validate the config

        self.predictors_labels = config["ml_setup"]["predictors"]
        self.target_labels = config["ml_setup"]["target"]
        self.model_names = config["ml_setup"]["model_names"]
        self.metric_names = config["ml_setup"]["metric_names"]
        self.train_test_split_kwargs = config["ml_setup"]["train_test_split_kwargs"]

        self.work_dir = Path(config["session_setup"]["work_dir"])
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.data_config = config["data"]
        self.save_grid_scores = config["ml_setup"]["save_grid_scores"]
        self.save_best_model = config["ml_setup"]["save_best_model"]

        self.log_experiment = config["ml_setup"]["log_experiment"]
        if self.log_experiment:
            self._setup_mlflow()

    def _setup_mlflow(self):
        mlruns_dir = self.work_dir / "mlruns"
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(str(mlruns_dir))
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
        kwargs are passed to sklearn.model_selection.train_test_split
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
        for model_name, kwrags in self.model_names.items():
            if model_name in LinearModels.ESTIMATOR_NAMES:
                models[model_name] = LinearModels(model_name, **kwrags)
            elif model_name in EnsembleModel.ENSEMBLE_REGRESSORS:
                models[model_name] = EnsembleModel(model_name, **kwrags)
            elif model_name in SparseGaussianModel.ESTIMATOR_NAMES:
                models[model_name] = SparseGaussianModel(model_name, **kwrags)
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

        self.X_data = data[self.predictors_labels]
        self.Y_data = data[self.target_labels]
        self.X_train, self.X_test, self.Y_train, self.Y_test = self.train_test_split(
            **self.train_test_split_kwargs
        )

        self.model_instances = self.get_models()

    def _run_model(self, model_name):
        """Runs the models on the training data.

        Returns:
            dict: Dictionary of trained models.

        """
        model = self.model_instances[model_name]
        scores = model.calculate_score(self.X_train, self.X_test, self.Y_train, self.Y_test, self.metric_names)
        return model.estimator, scores

    def _log_model(self, model_name):
        with mlflow.start_run(experiment_id=self.experiment_id):
            mlflow.log_param("model_name", model_name)
            # mlflow.log_metrics(self.scores[model_name])  #FIXME
            mlflow.sklearn.log_model(self.fitted_models[model_name], model_name)

    def compare_models(self) -> Any:
        """Compares the models and returns the best model.

        Returns:
            Any: Best model from the experiment according to metrics_sort, set
            in the configuration file.

        """
        self.fitted_models = {}
        self.scores = []
        for model_name in self.model_names:
            fitted_model, scores = self._run_model(model_name)
            self.fitted_models[model_name] = fitted_model
            self.scores.append({"model": model_name, **scores})

            if self.log_experiment:
                self._log_model(model_name)

        # if self.save_grid_scores:
        #     raise NotImplementedError("Saving grid score is not implemented yet.")
        #     # TODO: fix it
        #     # session_setup = self.config["session_setup"]
        #     # file_name = Path(session_setup["work_dir"]) / "grid_scores.csv"
        #     # self.grid_scores = experiment.pull()
        #     # self.grid_scores.to_csv(file_name, index=False)
        #     # logger.info(f"Grid scores saved to {file_name}.")

        # if self.save_best_model:
        #     raise NotImplementedError("Saving best model is not implemented yet.")
        #     # TODO: fix it
        #     # session_setup = self.config["session_setup"]
        #     # file_name = Path(session_setup["work_dir"]) / "best_model.onnx"
        #     # save_sklearn_to_onnx(file_name)
        #     # logger.info(f"Best model saved to {file_name}.")
        # raise NotImplementedError("Compare models is not implemented yet.")

        return self.fitted_models, pd.DataFrame(self.scores)
