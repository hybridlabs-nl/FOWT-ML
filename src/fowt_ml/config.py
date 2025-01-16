"""This module contains functions to read configuration files."""

import yaml


def read_yaml(file_path: str) -> dict:
    """Reads a yaml file and returns a dictionary.

    Args:
        file_path (str): Path to the yaml file.

    Returns:
        dict: Dictionary containing the yaml file.
    """
    with open(file_path) as file:
        return yaml.safe_load(file)


# TODO: validate the config file
