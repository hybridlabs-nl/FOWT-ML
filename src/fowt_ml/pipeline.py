import logging
from pathlib import Path
from typing import Any
import pandas as pd
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

    def setup(self, data: pd.DataFrame):
        """Sets up the machine learning experiment.

        Args:
            data (pd.DataFrame): DataFrame containing the data.

        Returns:
            Experiment object or similar.

        """
        ml_setup = self.config["ml_setup"]

        # TODO:
        # train_test_split
        # setup or init the experiment
        # self.exp = ...
        raise NotImplementedError("Setup is not implemented yet.")

    def compare_models(self) -> Any:
        """Compares the models and returns the best model.

        Returns:
            Any: Best model from the experiment according to metrics_sort, set
            in the configuration file.

        """
        # TODO:
        # compare models using self.exp
        # best_model = ...

        ml_setup = self.config["ml_setup"]
        if ml_setup.get("save_grid_scores", False):
            session_setup = self.config["session_setup"]
            file_name = Path(session_setup["work_dir"]) / "grid_scores.csv"
            raise NotImplementedError("Saving grid score is not implemented yet.")
            # TODO: fix it
            # self.grid_scores = experiment.pull()
            # self.grid_scores.to_csv(file_name, index=False)
            # logger.info(f"Grid scores saved to {file_name}.")

        if ml_setup.get("save_best_model", False):
            session_setup = self.config["session_setup"]
            file_name = Path(session_setup["work_dir"]) / "best_model.onnx"
            raise NotImplementedError("Saving best model is not implemented yet.")
            # TODO: fix it
            # save_sklearn_to_onnx(file_name)
            # logger.info(f"Best model saved to {file_name}.")
        raise NotImplementedError("Compare models is not implemented yet.")
        return best_model
