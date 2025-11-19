import pytest
import yaml
from pydantic_core import ValidationError
from fowt_ml.config import Config
from fowt_ml.config import ExperimentConfig
from fowt_ml.config import MLConfig
from fowt_ml.config import get_allowed_kwargs
from fowt_ml.config import get_config_file
from fowt_ml.gaussian_process import SparseGaussianModel
from fowt_ml.xgboost import XGBoost
from . import creat_dummy_config


class TestBaseConfig:
    def test_getitem_setitem(self):
        cfg = ExperimentConfig(
            path_file="test_config",
            aux_data={"wind_speed": 10.0},
        )
        assert cfg["path_file"] == "test_config"
        cfg["aux_data"]["wind_speed"] = 4.0
        assert cfg.aux_data["wind_speed"] == 4.0

    def test_contains(self):
        cfg = ExperimentConfig(
            path_file="test_config",
        )
        assert cfg["aux_data"] == {}
        assert "wind_speed" not in cfg


class TestConfig:
    def test_from_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        creat_dummy_config(config_file, "data.mat")

        cfg = Config.from_yaml(config_file)
        assert cfg.name == "dummy_experiment"
        assert cfg.ml_setup["targets"] == ["pos_act6[0]"]
        assert cfg.ml_setup["predictors"] == [
            "force_tt_meas6[0]",
            "force_tt_meas6[1]",
            "force_tt_meas6[2]",
        ]
        assert cfg.ml_setup["model_names"] == {
            "LinearRegression": {},
            "SklearnGPRegressor": {
                "num_inducing": 50,
                "num_latents": 3,
                "num_epochs": 1,
            },
            "RNNRegressor": {
                "input_size": 3,
                "hidden_size": 5,
                "num_layers": 2,
                "output_size": 1,
                "max_epochs": 5,
            },
        }

    def test_to_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        Config.to_yaml(config_file)

        with open(config_file) as file:
            expected_cfg = yaml.safe_load(file)
        assert expected_cfg["name"] == "default_experiment"

    def test_as_dict(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        creat_dummy_config(config_file, "data.mat")

        cfg = Config.from_yaml(config_file)
        cfg_dict = cfg.as_dict()
        assert cfg_dict["name"] == "dummy_experiment"
        assert isinstance(cfg_dict["ml_setup"], dict)

    def test_invalid_model_name(self):
        with pytest.raises(
            ValidationError, match="Model 'InvalidModel' not supported."
        ):
            MLConfig(
                targets=["target1"],
                predictors=["pred1"],
                model_names={"InvalidModel": {}},
                metric_names=["r2"],
            )

    def test_invalid_model_args(self):
        with pytest.raises(
            ValidationError,
            match="Invalid kwargs for model 'LinearRegression': {'invalid_arg'}.",
        ):
            MLConfig(
                targets=["target1"],
                predictors=["pred1"],
                model_names={"LinearRegression": {"invalid_arg": 123}},
                metric_names=["r2"],
            )

    def test_valid_model_gpregressor(self):
        MLConfig(
            targets=["target1"],
            predictors=["pred1"],
            model_names={"SklearnGPRegressor": {"num_inducing": 123}},
            metric_names=["r2"],
        )

    def test_valid_model_xgboost(self):
        MLConfig(
            targets=["target1"],
            predictors=["pred1"],
            model_names={"XGBoostRegression": {"tree_method": "hist"}},
            metric_names=["r2"],
        )

    def test_invalid_train_test_split_kwargs(self):
        with pytest.raises(
            ValidationError, match="Invalid train_test_split kwargs: {'invalid_arg'}."
        ):
            MLConfig(
                targets=["target1"],
                predictors=["pred1"],
                model_names={"LinearRegression": {}},
                metric_names=["r2"],
                train_test_split_kwargs={"invalid_arg": 123},
            )

    def test_invalid_cv_kwargs(self):
        with pytest.raises(
            ValidationError, match="Invalid cross_validate kwargs: {'invalid_arg'}."
        ):
            MLConfig(
                targets=["target1"],
                predictors=["pred1"],
                model_names={"LinearRegression": {}},
                metric_names=["r2"],
                cross_validation_kwargs={"invalid_arg": 123},
            )


def test_get_config_file_env_var():
    """Test the get_config_file function."""
    # add a dummy path to env variable
    import os

    os.environ["CONFIG_PATH"] = "/dummy/path/to/config"

    config_path = get_config_file()
    assert config_path == "/dummy/path/to/config"


def test_get_config_file_no_default():
    """Test the get_config_file function."""
    # remove the env variable
    import os

    if "CONFIG_PATH" in os.environ:
        del os.environ["CONFIG_PATH"]

    with pytest.raises(FileNotFoundError):
        get_config_file()


def test_get_allowed_kwargs_sparsegaussian():
    model_class = SparseGaussianModel.ESTIMATOR_NAMES["SklearnGPRegressor"]
    allowed_kwargs = get_allowed_kwargs(model_class)
    assert "num_inducing" in allowed_kwargs
    assert "num_epochs" in allowed_kwargs
    assert "learning_rate" in allowed_kwargs
    assert "batch_size" in allowed_kwargs
    assert "num_latents" in allowed_kwargs


def test_get_allowed_kwargs_xgboost():
    model_class = XGBoost.ESTIMATOR_NAMES["XGBoostRegression"]
    allowed_kwargs = get_allowed_kwargs(model_class)
    assert "n_estimators" in allowed_kwargs
    assert "max_depth" in allowed_kwargs
    assert "tree_method" in allowed_kwargs
