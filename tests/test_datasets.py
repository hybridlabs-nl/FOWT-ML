import logging
import numpy as np
import pandas as pd
import pytest
from fowt_ml.datasets import build_config_data
from fowt_ml.datasets import check_data
from fowt_ml.datasets import convert_mat_to_df
from fowt_ml.datasets import create_segments
from fowt_ml.datasets import fix_column_names
from fowt_ml.datasets import get_data
from fowt_ml.datasets import get_data_mfiles
from . import create_dummy_mat_file

logger = logging.getLogger(__name__)


def test_convert_mat_to_df(tmp_path):
    mat_file = tmp_path / "test_data.mat"
    create_dummy_mat_file(mat_file)

    data_id = "exp1"
    df = convert_mat_to_df(mat_file, data_id)
    assert df.shape == (50, 5)
    assert "time" in df.columns
    assert "force_tt_meas6[0]" in df.columns
    assert "force_tt_meas6[1]" in df.columns
    assert "force_tt_meas6[2]" in df.columns
    assert "pos_act6[0]" in df.columns


def test_get_data_with_wind_speed(tmp_path, caplog):
    mat_file = tmp_path / "test_data.mat"
    create_dummy_mat_file(mat_file)

    data_id = "exp1"
    config = {
        data_id: {
            "path_file": str(mat_file),
            "aux_data": {"wind_speed": 10.0},
            "description": "Test data",
        }
    }
    with caplog.at_level(logging.INFO):
        df = get_data(data_id, config)

    assert df.shape == (50, 6)  # wind_speed column added
    assert "wind_speed" in df.columns
    assert "description" not in df.columns
    assert any(
        "wind_speed not found in the data file." in record.message
        for record in caplog.records
    )


def test_get_data_without_wind_speed(tmp_path):
    mat_file = tmp_path / "test_data.mat"
    create_dummy_mat_file(mat_file)

    data_id = "exp1"
    config = {
        data_id: {
            "path_file": str(mat_file),
            "description": "Test data",
        }
    }

    df = get_data(data_id, config)

    assert df.shape == (50, 5)
    assert "wind_speed" not in df.columns


def test_check_data(tmp_path):
    mat_file = tmp_path / "test_data.mat"
    create_dummy_mat_file(mat_file)

    data_id = "exp1"
    config = {
        data_id: {
            "path_file": str(mat_file),
            "description": "Test data",
        }
    }

    df = get_data(data_id, config)
    col_names = [
        "pos_act6[0]",
        "force_tt_meas6[0]",
        "force_tt_meas6[1]",
        "force_tt_meas6[2]",
    ]
    df = check_data(df, col_names)

    assert isinstance(df, type(df))
    assert not df[col_names].isnull().any().any()
    for col in col_names:
        assert np.issubdtype(df[col].dtype, np.number)


def test_check_missing(tmp_path):
    mat_file = tmp_path / "test_data.mat"
    create_dummy_mat_file(mat_file)

    data_id = "exp1"
    config = {
        data_id: {
            "path_file": str(mat_file),
            "description": "Test data",
        }
    }

    df = get_data(data_id, config)
    col_names = ["pos_act6[1]"]
    with pytest.raises(ValueError, match="Missing columns:"):
        df = check_data(df, col_names)


def test_check_null(tmp_path):
    mat_file = tmp_path / "test_data.mat"
    create_dummy_mat_file(mat_file)

    data_id = "exp1"
    config = {
        data_id: {
            "path_file": str(mat_file),
            "description": "Test data",
        }
    }

    df = get_data(data_id, config)
    col_names = [
        "pos_act6[0]",
        "force_tt_meas6[0]",
        "force_tt_meas6[1]",
        "force_tt_meas6[2]",
    ]
    df.loc[0, "pos_act6[0]"] = np.nan
    with pytest.raises(ValueError, match="has NaN values"):
        df = check_data(df, col_names)


def test_check_numeric(tmp_path):
    mat_file = tmp_path / "test_data.mat"
    create_dummy_mat_file(mat_file)

    data_id = "exp1"
    config = {
        data_id: {
            "path_file": str(mat_file),
            "description": "Test data",
        }
    }

    df = get_data(data_id, config)
    col_names = [
        "pos_act6[0]",
        "force_tt_meas6[0]",
        "force_tt_meas6[1]",
        "force_tt_meas6[2]",
    ]
    df["pos_act6[0]"] = df["pos_act6[0]"].astype(object)
    with pytest.raises(ValueError, match="is not numeric"):
        df = check_data(df, col_names)


def test_fix_column_names(tmp_path):
    mat_file = tmp_path / "test_data.mat"
    create_dummy_mat_file(mat_file)

    data_id = "exp1"
    config = {
        data_id: {
            "path_file": str(mat_file),
            "description": "Test data",
        }
    }

    df = get_data(data_id, config)
    df = fix_column_names(df)
    assert "pos_act6[0]" not in df.columns
    assert "pos_act6_0" in df.columns
    assert isinstance(df, type(df))


def test_create_segments(tmp_path):
    mat_file = tmp_path / "test_data.mat"
    create_dummy_mat_file(mat_file)

    data_id = "exp1"
    config = {
        data_id: {
            "path_file": str(mat_file),
            "description": "Test data",
        }
    }

    df = get_data(data_id, config)
    df = fix_column_names(df)
    train_data = np.asarray(df)
    segment_length = 10
    segments = create_segments(train_data, segment_length)
    assert segments.shape == (41, 10, df.shape[1])


def test_build_config_data(tmp_path):
    # create mat files in tmp_path
    create_dummy_mat_file(tmp_path / "exp1.mat", data_id="exp1")
    create_dummy_mat_file(tmp_path / "exp2.mat", data_id="exp2")

    # create aux df
    aux_df = pd.DataFrame(
        {
            "recording_number": [1, 2],
            "wind_speed": [10.0, 15.0],
            "description": ["Test data 1", "Test data 2"],
        }
    )

    config = build_config_data(
        tmp_path, aux_df=aux_df, aux_df_column_name="recording_number"
    )

    assert config["exp1"]["path_file"] == str(tmp_path / "exp1.mat")
    assert config["exp2"]["path_file"] == str(tmp_path / "exp2.mat")
    assert config["exp1"]["aux_data"]["wind_speed"] == 10.0
    assert config["exp1"]["aux_data"]["description"] == "Test data 1"
    assert config["exp2"]["aux_data"]["wind_speed"] == 15.0


def test_build_config_data_wrong_aux_df_column_name(tmp_path):
    # create mat files in tmp_path
    create_dummy_mat_file(tmp_path / "exp1.mat", data_id="exp1")
    create_dummy_mat_file(tmp_path / "exp2.mat", data_id="exp2")

    # create aux df
    aux_df = pd.DataFrame(
        {
            "recording_number": [1, 2],
            "wind_speed": [10.0, 15.0],
            "description": ["Test data 1", "Test data 2"],
        }
    )

    with pytest.raises(ValueError, match="Column wrong_column not found in aux_df."):
        build_config_data(tmp_path, aux_df=aux_df, aux_df_column_name="wrong_column")


def test_build_config_data_without_aux_data(tmp_path):
    # create mat files in tmp_path
    create_dummy_mat_file(tmp_path / "exp1.mat", data_id="exp1")
    create_dummy_mat_file(tmp_path / "exp2.mat", data_id="exp2")

    # create aux df
    aux_df = pd.DataFrame(
        {
            "recording_number": [1],
            "wind_speed": [10.0],
            "description": ["Test data 1"],
        }
    )
    # this should work with warning
    config = build_config_data(
        tmp_path, aux_df=aux_df, aux_df_column_name="recording_number"
    )
    assert "aux_data" in config["exp1"]
    assert "aux_data" not in config["exp2"]


def test_get_data_mfiles(tmp_path):
    # create mat files in tmp_path
    create_dummy_mat_file(tmp_path / "exp1.mat", data_id="exp1")
    create_dummy_mat_file(tmp_path / "exp2.mat", data_id="exp2")

    # create csv file for aux data
    aux_df = pd.DataFrame(
        {
            "recording_number": [1, 2],
            "wind_speed": [10.0, 15.0],
            "description": ["Test data 1", "Test data 2"],
        }
    )
    aux_csv_file = tmp_path / "aux_data.csv"
    aux_df.to_csv(aux_csv_file, index=False)

    df = get_data_mfiles(
        tmp_path, aux_csv_file=aux_csv_file, aux_df_column_name="recording_number"
    )
    assert len(df) == 100
    assert len(df.columns) == 8  # original 5 + 3 in aux
