"""This module contains functions to load and preprocess datasets."""

import logging
import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def convert_mat_to_df(mat_file: str, data_id: str) -> pd.DataFrame:
    """Reads a matlab file and returns a pandas DataFrame.

    Args:
        mat_file (str): Path to a matlab file.
        data_id (str): ID of the data in the matlab file.

    Returns:
        pd.DataFrame: DataFrame containing the data.
    """
    hdf = h5py.File(mat_file, mode="r")

    # validate the file
    if data_id not in hdf:
        raise ValueError(f"Experiment {data_id} not found in the file.")

    if "X" not in hdf[data_id] or "Y" not in hdf[data_id]:
        raise ValueError(f"Experiment {data_id} does not have X or Y data.")

    if "Data" not in hdf[data_id]["X"]:
        raise ValueError(f"Experiment {data_id} does not have X Data.")

    if "Name" not in hdf[data_id]["Y"] or "Data" not in hdf[data_id]["Y"]:
        raise ValueError(f"Experiment {data_id} does not have Y Name or Data.")

    data = {"time": np.array(hdf[data_id]["X"]["Data"][:]).flatten()}
    name_references = np.array(hdf[data_id]["Y"]["Name"][:]).flatten()
    data_references = np.array(hdf[data_id]["Y"]["Data"][:]).flatten()

    for index, (name_ref, data_ref) in enumerate(
        zip(name_references, data_references, strict=False)
    ):
        name = "".join([chr(item[0]) for item in hdf[name_ref]])
        if name in data:
            msg = (
                f"Duplicate name {name} found in the data."
                f" Renaming it to {name}_{index}."
            )
            logger.warning(msg)
            name = f"{name}_{index}"
        data[name] = np.array(hdf[data_ref]).flatten()
    return pd.DataFrame(data)


def get_data(data_id: str, config: dict) -> pd.DataFrame:
    """Returns a dataframe for the given data_id.

    Args:
        data_id (str): ID of the data in the configuration file.
        config (dict): Configuration dictionary.
            Example: {"data_id": {"path_file": "data.mat"}}.

    Returns:
        pd.DataFrame: DataFrame for the given data_id.

    """
    data_info = config[data_id]
    df = convert_mat_to_df(data_info["path_file"], data_id)

    # check if wind speed is present
    if "wind_speed" not in df and "wind_speed" in data_info:
        df["wind_speed"] = data_info["wind_speed"]
        msg = (
            f"Wind speed not found in the data file. "
            f"But found in config file. "
            f"Setting it to {data_info['wind_speed']}."
        )
        logger.info(msg)
    return df


def check_data(df: pd.DataFrame, col_names) -> pd.DataFrame:
    """Checks if the dataframe has the required columns and their are valid.

    Args:
        df (pd.DataFrame): DataFrame to check.
        col_names (list): List of required columns.

    Returns:
        pd.DataFrame: DataFrame with valid required columns.

    """
    missing_columns = [col for col in col_names if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # check if the columns have valid values
    for col in df.columns:
        if df[col].isnull().any():
            raise ValueError(f"Column {col} has NaN values.")
        if not np.issubdtype(df[col].dtype, np.number):
            raise ValueError(f"Column {col} is not numeric.")

    return df


def fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Fixes the column names to remove special characters.

    Args:
        df (pd.DataFrame): DataFrame to fix.

    Returns:
        pd.DataFrame: DataFrame with fixed column names.

    """
    df.rename(
        columns=lambda col: (
            col.replace("[", "_").replace("]", "").replace("<", "_").replace(">", "")
        ),
        inplace=True,
    )
    return df
