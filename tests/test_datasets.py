import logging
from fowt_ml.datasets import convert_mat_to_df
from fowt_ml.datasets import get_data
from . import create_dummy_mat_file

logger = logging.getLogger(__name__)


def test_convert_mat_to_df(tmp_path):
    mat_file = tmp_path / "test_data.mat"
    create_dummy_mat_file(mat_file)

    data_id = "exp1"
    df = convert_mat_to_df(mat_file, data_id)

    assert df.shape == (10, 4)
    assert "time" in df.columns
    assert "acc_tb_meas3[0]" in df.columns
    assert "acc_tb_meas3[1]" in df.columns
    assert "acc_tb_meas3[2]" in df.columns


def test_get_data_with_wind_speed(tmp_path, caplog):
    mat_file = tmp_path / "test_data.mat"
    create_dummy_mat_file(mat_file)

    data_id = "exp1"
    config = {
        data_id: {
            "mat_file": str(mat_file),
            "wind_speed": 10.0,
            "description": "Test data",
        }
    }
    with caplog.at_level(logging.INFO):
        df = get_data(data_id, config)

    assert df.shape == (10, 5)  # wind_speed column added
    assert "wind_speed" in df.columns
    assert "description" not in df.columns
    assert any("Wind speed not found" in record.message for record in caplog.records)


def test_get_data_without_wind_speed(tmp_path):
    mat_file = tmp_path / "test_data.mat"
    create_dummy_mat_file(mat_file)

    data_id = "exp1"
    config = {
        data_id: {
            "mat_file": str(mat_file),
            "description": "Test data",
        }
    }

    df = get_data(data_id, config)

    assert df.shape == (10, 4)
    assert "wind_speed" not in df.columns
