import logging
from pathlib import Path
import pandas as pd
from pycaret.regression import RegressionExperiment
from fowt_ml.config import read_yaml
from fowt_ml.datasets import convert_mat_to_df
from fowt_ml.datasets import read_mat_file

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config_file: str):
        self.config = read_yaml(config_file)

    def get_data(self, data_id: str) -> pd.DataFrame:
        """Returns the dataset for the given data_id."""
        data_info = self.config["data"][data_id]

        hdf = read_mat_file(data_info["mat_file"], data_id)
        df = convert_mat_to_df(hdf)

        # check if wind speed is present
        if "wind_speed" not in df and data_info["wind_speed"] is not None:
            df["wind_speed"] = data_info["wind_speed"]
            msg = (
                f"Wind speed not found in the data file. "
                f"Setting it to {data_info['wind_speed']}."
            )
            logger.info(msg)
        return df

    def setup(self, data_id: str):
        """Sets up the machine learning experiment."""
        data = self.get_data(data_id)

        ml_setup = self.config["ml_setup"]
        pycaret_setup = self.config["pycaret_setup"]

        # use regression experiment
        s = RegressionExperiment()
        exp = s.setup(
            data=data[ml_setup["predictors"]],
            target=data[ml_setup["target"]],
            n_jobs=pycaret_setup["n_jobs"],
            use_gpu=pycaret_setup["use_gpu"],
            system_log=pycaret_setup["system_log"],
            preprocess=False,
            session_id=123,
            html=False,
            verbose=False,
        )
        return exp

    def compare_models(self, experiment):
        """Compares the models and returns the best model."""
        pycaret_setup = self.config["pycaret_setup"]

        best_model = experiment.compare_models(
            include=pycaret_setup["linear_models"],
            sort=pycaret_setup["metrics_sort"],
            turbo=False,
            verbose=False,
        )

        ml_setup = self.config["ml_setup"]
        if ml_setup["save_grid_scores"]:
            grid_scores = experiment.pull()
            session_setup = self.config["session_setup"]
            file_name = Path(session_setup["work_dir"]) / "grid_scores.csv"
            grid_scores.to_csv(file_name, index=False)
            logger.info(f"Grid scores saved to {file_name}.")

        if ml_setup["save_best_model"]:
            session_setup = self.config["session_setup"]
            file_name = Path(session_setup["work_dir"]) / "best_model.onnx"
            # TODO: fix it
            # save_sklearn_to_onnx(file_name)
            # logger.info(f"Best model saved to {file_name}.")
        return best_model
