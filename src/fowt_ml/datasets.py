"""This module contains functions to load and preprocess datasets."""

import logging
from pathlib import Path
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

    # check if auxiliary data is present in the config file
    if "aux_data" in data_info:
        for key, val in data_info["aux_data"].items():
            if key not in df and val is not None:
                df[key] = val
                msg = (
                    f"{key} not found in the data file. "
                    f"But found in config file. "
                    f"Setting it to {val}."
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


def create_segments(arr: np.array, seq_len: int) -> np.array:
    """Creates segments of the given sequence length from the array."""
    # arr: [samples, features]
    num_samples, num_features = arr.shape
    num_segments = num_samples - seq_len + 1
    # Use stride tricks for efficiency
    shape = (num_segments, seq_len, num_features)
    strides = (arr.strides[0], arr.strides[0], arr.strides[1])
    # Return a view of the array with the new shape and strides, if you need a
    # copy, use .copy() on the result
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def build_config_data(data_dir, aux_df=None, aux_df_column_name=None):
    """Builds a configuration dictionary for the data.

    Args:
        data_dir (str): Path to the data directory containing .mat files.
        aux_df (pd.DataFrame, optional): DataFrame containing auxiliary data.
        aux_df_column_name (str, optional): Column name in aux_df that matches
           the recording number in the data_id.

    Returns:
        dict: Configuration dictionary for the data.

    """
    config = {}

    # list all the files in the data directory with mat extension and get
    # filename without extension as data_id
    for file in Path(data_dir).glob("*.mat"):
        data_id = file.stem
        config[data_id] = {"path_file": str(file)}

    # if aux_df is provided, use it to get the auxiliary data
    if aux_df is not None:
        # check if the aux_df has the required columns
        if aux_df_column_name not in aux_df.columns:
            raise ValueError(f"Column {aux_df_column_name} not found in aux_df.")

        for data_id in config:
            # split exp from the data_id and get the recording number this works
            # even if the data id is without exp, as it will return the whole
            # string
            recording_number = int(data_id.split("exp")[-1])
            df_aux = aux_df[aux_df[aux_df_column_name] == recording_number]
            if not df_aux.empty:
                config[data_id]["aux_data"] = df_aux.to_dict(orient="records")[0]
            else:
                logger.warning(
                    f"No auxiliary data found for {data_id} in the CSV file."
                )
    return config


def get_data_mfiles(data_dir, aux_csv_file=None, aux_df_column_name=None):
    """Merge the data from the data_dir into a dataframe.

    Args:
        data_dir (str): Path to the data directory containing .mat files.
        aux_csv_file (str, optional): Path to the CSV file containing auxiliary data.
        aux_df_column_name (str, optional): Column name in the CSV file that matches
           the recording number in the data_id.

    Returns:
        pd.DataFrame: DataFrame containing the merged data from all the .mat
            files and the auxiliary data from the CSV file.

    """
    if aux_csv_file is not None:
        aux_df = pd.read_csv(aux_csv_file)
        if aux_df_column_name is None:
            raise ValueError(
                "aux_df_column_name must be provided if aux_csv_file is provided."
            )
    else:
        aux_df = None
    config = build_config_data(
        data_dir, aux_df=aux_df, aux_df_column_name=aux_df_column_name
    )

    # loop over data_ids in config and get the data for each data_id and merge
    # them into a single dataframe
    dfs = [get_data(data_id, config) for data_id in config]
    data = pd.concat(dfs, ignore_index=True)

    return data
