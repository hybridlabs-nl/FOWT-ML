from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from fowt_ml.config import read_yaml
from fowt_ml.pipeline import Pipeline
from . import creat_dummy_config
from . import create_dummy_mat_file


class TestPipelineInit:
    def test_init_config_file(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)

        my_pipeline = Pipeline(config_file)

        assert hasattr(my_pipeline, "predictors_labels")
        assert hasattr(my_pipeline, "target_labels")
        assert "LinearRegression" in my_pipeline.model_names
        # check mlflow directory created
        assert Path("mlruns").exists()

    def test_init_config_dict(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        config_dict = read_yaml(config_file)

        my_pipeline = Pipeline(config_dict)

        assert hasattr(my_pipeline, "predictors_labels")
        assert hasattr(my_pipeline, "target_labels")
        assert "LinearRegression" in my_pipeline.model_names


class TestPipelineGetData:
    def test_get_data(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test get_data
        my_pipeline = Pipeline(config_file)
        df = my_pipeline.get_data("exp1")
        assert df.shape == (50, 4)


class TestPipelineSplit:
    def test_train_test_split_default(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test train_test_split
        my_pipeline = Pipeline(config_file)

        df = my_pipeline.get_data("exp1")
        my_pipeline.X_data = df[my_pipeline.predictors_labels]
        my_pipeline.Y_data = df[my_pipeline.target_labels]
        results = my_pipeline.train_test_split()
        assert len(results) == 4

        # X_train and Y_train have same number of samples
        assert results[0].shape[0] == results[2].shape[0]

        # X_test and Y_test have same number of samples
        assert results[1].shape[0] == results[3].shape[0]

        # X_train and X_test have same number of features
        assert results[0].shape[1] == results[1].shape[1]

    def test_train_test_split_kwargs(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test train_test_split
        my_pipeline = Pipeline(config_file)

        df = my_pipeline.get_data("exp1")
        my_pipeline.X_data = df[my_pipeline.predictors_labels]
        my_pipeline.Y_data = df[my_pipeline.target_labels]
        results = my_pipeline.train_test_split(**{"train_size": 0.2})
        assert len(results) == 4

        # train size is 0.2
        assert results[0].shape[0] == 0.2 * my_pipeline.X_data.shape[0]

        # test size is 0.8
        assert results[1].shape[0] == 0.8 * my_pipeline.X_data.shape[0]


class TestPipelineGetModels:
    def test_get_models(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test get_models
        my_pipeline = Pipeline(config_file)
        models = my_pipeline.get_models()
        assert "LeastAngleRegression" in models
        assert "LinearRegression" in models
        assert len(models) == 2


class TestPipelineSetup:
    def test_setup_str(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test setup
        my_pipeline = Pipeline(config_file)
        my_pipeline.setup(data="exp1")
        assert hasattr(my_pipeline, "X_data")
        assert hasattr(my_pipeline, "Y_data")
        assert hasattr(my_pipeline, "X_train")
        assert hasattr(my_pipeline, "X_test")
        assert hasattr(my_pipeline, "Y_train")
        assert hasattr(my_pipeline, "Y_test")
        assert hasattr(my_pipeline, "model_instances")

    def test_setup_df(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test setup
        my_pipeline = Pipeline(config_file)
        df = my_pipeline.get_data("exp1")
        my_pipeline.setup(df)
        assert hasattr(my_pipeline, "X_data")
        assert hasattr(my_pipeline, "Y_data")
        assert hasattr(my_pipeline, "X_train")
        assert hasattr(my_pipeline, "X_test")
        assert hasattr(my_pipeline, "Y_train")
        assert hasattr(my_pipeline, "Y_test")
        assert hasattr(my_pipeline, "model_instances")


class TestPipelineCompare:
    def test_compare_models_default(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test setup
        my_pipeline = Pipeline(config_file)
        my_pipeline.setup(data="exp1")
        models, scores = my_pipeline.compare_models()
        assert isinstance(scores, pd.DataFrame)
        assert "r2" in scores
        assert "model_fit_time" in scores
        assert isinstance(models, dict)
        assert "LeastAngleRegression" in models
        assert "LinearRegression" in models
        # check model is fitted
        assert check_is_fitted(models["LinearRegression"]) is None
        assert Path("grid_scores.csv").exists()
        assert Path("best_model.onnx").exists()
        # check sorting of scores
        assert scores["r2"].iloc[0] > scores["r2"].iloc[1]

    def test_compare_models_sort(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test setup
        my_pipeline = Pipeline(config_file)
        my_pipeline.setup(data="exp1")
        _, scores = my_pipeline.compare_models(sort="model_fit_time")
        # check sorting of scores
        assert scores["model_fit_time"].iloc[0] <= scores["model_fit_time"].iloc[1]

    def test_compare_models_onnx(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test setup
        my_pipeline = Pipeline(config_file)
        my_pipeline.setup(data="exp1")
        model, scores = my_pipeline.compare_models()
        best_model_name = scores.index[0]
        fitted_model = model[best_model_name]
        expected_pred = fitted_model.predict(my_pipeline.X_test)

        # read the onnx model
        import onnxruntime as rt
        sess = rt.InferenceSession("best_model.onnx")
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        x = my_pipeline.X_test.to_numpy(dtype=np.float32)
        actual_pred = sess.run([output_name], {input_name: x})[0]

        np.testing.assert_allclose(expected_pred, actual_pred.flatten(), rtol=1e-5)
