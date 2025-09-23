import pytest
import yaml
from pydantic_core import ValidationError
from fowt_ml.config import Config
from fowt_ml.config import MLConfig
from fowt_ml.config import get_config_file
from . import creat_dummy_config


class TestConfig:
    def test_from_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        creat_dummy_config(config_file, "data.mat")

        cfg = Config.from_yaml(config_file)
        assert cfg.name == "dummy_experiment"
        assert cfg.ml_setup["targets"] == ["acc_tb_meas3[0]"]
        assert cfg.ml_setup["predictors"] == ["acc_tb_meas3[1]", "acc_tb_meas3[2]"]
        assert cfg.ml_setup["model_names"] == {
            "LeastAngleRegression": {},
            "LinearRegression": {},
        }

    def test_to_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        Config.to_yaml(config_file)

        with open(config_file) as file:
            expected_cfg = yaml.safe_load(file)
        assert expected_cfg["name"] == "basic_config"

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
