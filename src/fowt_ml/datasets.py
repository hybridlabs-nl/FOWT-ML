"""This module contains functions to load and preprocess datasets."""

import logging
import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def read_mat_file(mat_file: str, data_id: str) -> h5py.File:
    """Reads a matlab file and returns the h5py file object."""
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
    return hdf


def convert_mat_to_df(hdf: h5py.File) -> pd.DataFrame:
    """Converts a h5py file object to a pandas DataFrame."""
    data = {"time": np.array(hdf["exp699"]["X"]["Data"][:]).flatten()}
    name_references = np.array(hdf["exp699"]["Y"]["Name"][:]).flatten()
    data_references = np.array(hdf["exp699"]["Y"]["Data"][:]).flatten()

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
