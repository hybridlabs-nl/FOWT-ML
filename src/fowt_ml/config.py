"""This module contains functions to read configuration files."""

import inspect
import os
from logging import Logger
from pathlib import Path
from typing import Any
import pydantic
import skorch
import yaml
from pydantic_core import PydanticUndefined
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from fowt_ml.ensemble import EnsembleModel
from fowt_ml.gaussian_process import SparseGaussianModel
from fowt_ml.linear_models import LinearModels
from fowt_ml.neural_network import NeuralNetwork
from fowt_ml.neural_network import create_skorch_regressor
from fowt_ml.xgboost import XGBoost

logger = Logger(__name__)


def get_allowed_kwargs(func_or_class):
    """Return valid keyword args for a function or class constructor."""
    if inspect.isclass(func_or_class):
        # Handle sklearn-style estimators (incl. XGBoost)
        if hasattr(func_or_class, "get_params"):
            try:
                return set(func_or_class().get_params().keys())
            except Exception:
                pass
        sig = inspect.signature(func_or_class.__init__)
    else:
        sig = inspect.signature(func_or_class)
    return set(sig.parameters.keys()) - {"self", "kwargs"}  # drop 'self' and 'kwargs'


class BaseConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        extra="forbid",
        validate_default=True,
        validate_assignment=True,
        populate_by_name=True,
    )

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, item):
        return item in self.__class__.model_fields

    def as_dict(self, *, by_alias: bool = False) -> dict:
        """Return the config as a (nested) dict."""
        return self.model_dump(by_alias=by_alias)


class ExperimentConfig(BaseConfig):
    path_file: str
    aux_data: dict[str, float] = {}


class MLConfig(BaseConfig):
    targets: list[str]
    predictors: list[str]
    save_grid_scores: bool = False
    save_best_model: bool = True
    log_experiment: bool = False
    n_jobs: int = 1
    use_gpu: bool = False
    scale_data: bool = True
    train_test_split_kwargs: dict[str, Any] = {
        "test_size": 0.25,
        "random_state": 42,
        "shuffle": True,
    }
    cross_validation_kwargs: dict[str, Any] = {}
    model_names: dict[str, dict[str, Any]]
    metric_names: list[str]

    @pydantic.field_validator("train_test_split_kwargs")
    @classmethod
    def validate_tts_kwargs(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate train_test_split kwargs."""
        allowed_tts_kwargs = get_allowed_kwargs(train_test_split)

        if invalid := set(v.keys()) - allowed_tts_kwargs:
            raise ValueError(
                f"Invalid train_test_split kwargs: {invalid}. "
                f"Allowed: {sorted(allowed_tts_kwargs)}"
            )
        return v

    @pydantic.field_validator("cross_validation_kwargs")
    @classmethod
    def validate_cv_kwargs(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate cross_validate kwargs."""
        allowed_cv_kwargs = get_allowed_kwargs(cross_validate)
        if invalid := set(v.keys()) - allowed_cv_kwargs:
            raise ValueError(
                f"Invalid cross_validate kwargs: {invalid}. "
                f"Allowed: {sorted(allowed_cv_kwargs)}"
            )
        return v

    @pydantic.field_validator("model_names")
    @classmethod
    def validate_models(cls, v: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Validate model names and their kwargs."""
        estimator_map = {
            name: est_cls
            for model_class in [
                LinearModels,
                EnsembleModel,
                SparseGaussianModel,
                NeuralNetwork,
                XGBoost,
            ]
            for name, est_cls in model_class.ESTIMATOR_NAMES.items()
        }

        for model_name, kwargs in v.items():
            if model_name not in estimator_map:
                raise ValueError(
                    f"Model '{model_name}' not supported. "
                    f"Available: {list(estimator_map.keys())}"
                )

            # Get the constructor signature for that model class
            model_class = estimator_map[model_name]
            allowed_kwargs = get_allowed_kwargs(model_class)
            if model_name in {"RNNRegressor", "LSTMRegressor", "GRURegressor"}:
                model_class = create_skorch_regressor
                allowed_kwargs = get_allowed_kwargs(model_class)
                model_class = skorch.net.NeuralNet
                allowed_kwargs = allowed_kwargs | get_allowed_kwargs(model_class)

            if invalid := set(kwargs.keys()) - allowed_kwargs:
                raise ValueError(
                    f"Invalid kwargs for model '{model_name}': {invalid}. "
                    f"Allowed: {allowed_kwargs}"
                )
        return v


class Config(BaseConfig):
    """Base class for configuration files."""

    name: str = "default_experiment"
    description: str = "Experiment description"
    data: dict[str, ExperimentConfig]
    ml_setup: MLConfig
    session_setup: dict = {
        "work_dir": ".",
    }

    @classmethod
    def from_yaml(cls, config_file):
        """Read configs from a config.yaml file.

        If key is not found in config.yaml, the default value is used.
        """
        if not Path(config_file).exists():
            raise FileNotFoundError(f"Config file {config_file} not found.")

        with open(config_file) as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                raise SyntaxError(f"Error parsing config file {config_file}.") from exc
        return cls(**cfg)

    @classmethod
    def to_yaml(cls, config_file):
        """Write configs to a yaml config_file."""
        if Path(config_file).exists():
            logger.warning(f"Overwriting config file {config_file}.")

        cfg = _schema(cls)
        with open(config_file, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)


def get_config_file():
    """Get the config file path."""
    config_path = Path.home() / ".config" / "fowt_ml"
    if os.environ.get("CONFIG_PATH"):
        return os.environ.get("CONFIG_PATH")
    elif os.path.exists(config_path):
        yml_files = list(Path.glob(config_path, "*.yml"))
        if len(yml_files) > 1:
            raise ValueError(
                f"Multiple config files found in {config_path}. Please specify one."
            )
        return config_path / yml_files[0]
    else:
        raise FileNotFoundError(
            f"Config file not found. Please specify one in {config_path}"
            " or as an environment variable `CONFIG_PATH`."
        )


def _schema(model: type[pydantic.BaseModel]) -> dict:
    """Return the schema of a pydantic model."""
    schema_dict = {}
    for name, field in model.model_fields.items():
        field_type = field.annotation
        if isinstance(field_type, type) and issubclass(field_type, pydantic.BaseModel):
            schema_dict[name] = _schema(field_type)
        else:
            schema_dict[name] = field.default
            if schema_dict[name] == PydanticUndefined:
                schema_dict[name] = f"undefined with type ({field_type.__name__})"
    return schema_dict
