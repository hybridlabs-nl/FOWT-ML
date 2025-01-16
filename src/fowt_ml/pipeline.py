import logging
from pathlib import Path
from typing import Any
import pandas as pd
from pycaret.regression import RegressionExperiment
from fowt_ml.config import read_yaml
from fowt_ml.datasets import get_data

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config_file: str):
        """Initializes the machine learning pipeline.

        Args:
            config_file (str): Path to the configuration yaml file.
        """
        self.config = read_yaml(config_file)

    def get_data(self, data_id: str) -> pd.DataFrame:
        """Returns the dataset for the given data_id.

        Args:
            data_id (str): ID of the data in the configuration file.

        Returns:
            pd.DataFrame: DataFrame for the given data_id, set in the
            configuration file.
        """
        data_config = self.config["data"]
        return get_data(data_id, data_config)

    def setup(self, data: pd.DataFrame) -> RegressionExperiment:
        """Sets up the machine learning experiment.

        Args:
            data (pd.DataFrame): DataFrame containing the data.

        Returns:
            pycaret.regression: Pycaret RegressionExperiment

        """
        ml_setup = self.config["ml_setup"]
        pycaret_setup = self.config["pycaret_setup"]

        # use regression experiment
        # TODO: this will create a logs.log function. We need to disable it.
        reg_exp = RegressionExperiment()

        exp = reg_exp.setup(
            data=data[ml_setup["predictors"]],
            target=data[ml_setup["target"]],
            n_jobs=pycaret_setup.get("n_jobs", 1),
            use_gpu=pycaret_setup.get("use_gpu", False),
            system_log=pycaret_setup.get("system_log", False),
            preprocess=False,
            session_id=123,
            html=False,
            verbose=False,
        )
        return exp

    def compare_models(self, experiment) -> Any:
        """Compares the models and returns the best model.

        Args:
            experiment (pycaret.regression): Pycaret RegressionExperiment

        Returns:
            Any: Best model from the experiment according to metrics_sort, set
            in the configuration file.

        """
        pycaret_setup = self.config["pycaret_setup"]

        best_model = experiment.compare_models(
            include=pycaret_setup["models"],
            sort=pycaret_setup.get("metrics_sort", "R2"),
            turbo=False,
            verbose=False,
        )

        ml_setup = self.config["ml_setup"]
        if ml_setup.get("save_grid_scores", False):
            self.grid_scores = experiment.pull()
            session_setup = self.config["session_setup"]
            file_name = Path(session_setup["work_dir"]) / "grid_scores.csv"
            self.grid_scores.to_csv(file_name, index=False)
            logger.info(f"Grid scores saved to {file_name}.")

        if ml_setup.get("save_best_model", False):
            raise NotImplementedError("Saving best model is not implemented yet.")
            # session_setup = self.config["session_setup"]
            # file_name = Path(session_setup["work_dir"]) / "best_model.onnx"
            # TODO: fix it
            # save_sklearn_to_onnx(file_name)
            # logger.info(f"Best model saved to {file_name}.")
        return best_model
