from fowt_ml.pipeline import Pipeline
from . import creat_dummy_config
from . import create_dummy_mat_file


class TestPipeline:
    def test_init(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)

        my_pipeline = Pipeline(config_file)

        assert "data" in my_pipeline.config
        assert "ml_setup" in my_pipeline.config
        assert "pycaret_setup" in my_pipeline.config
        assert my_pipeline.config["data"]["exp1"]["mat_file"] == str(mat_file)

    def test_get_data(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test get_data
        my_pipeline = Pipeline(config_file)
        df = my_pipeline.get_data("exp1")
        assert df.shape == (10, 4)

    def test_setup(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test setup
        my_pipeline = Pipeline(config_file)
        df = my_pipeline.get_data("exp1")
        exp = my_pipeline.setup(df)
        assert exp is not None

    def test_compare_models(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test setup
        my_pipeline = Pipeline(config_file)
        df = my_pipeline.get_data("exp1")
        exp = my_pipeline.setup(df)
        result = my_pipeline.compare_models(exp)
        assert result is not None
