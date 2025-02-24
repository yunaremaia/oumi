import os
import tempfile

from lema.core.types import TrainingConfig
from omegaconf import OmegaConf


def test_config_serialization():
    with tempfile.TemporaryDirectory() as folder:
        original_config = TrainingConfig()
        original_config.model.model_name = "my_test_model"

        filename = os.path.join(folder, "test_config.yaml")
        original_config.to_yaml(filename)

        assert os.path.exists(filename)

        loaded_config = TrainingConfig.from_yaml(filename)
        assert loaded_config.model.model_name == "my_test_model"
        assert original_config == loaded_config


def test_config_equality():
    config_a = TrainingConfig()
    config_b = TrainingConfig()
    assert config_a == config_b

    config_a.model.model_name = "test_model"
    assert config_a != config_b


def test_config_override():
    low_priority_config = TrainingConfig()
    low_priority_config.model.model_name = "model_low_priority"

    high_priority_config = TrainingConfig()
    high_priority_config.model.model_name = "model_high_priority"

    # Override with CLI arguments if provided
    merged_config = OmegaConf.merge(low_priority_config, high_priority_config)
    assert merged_config.model.model_name == "model_high_priority"
    assert merged_config == high_priority_config
    assert merged_config != low_priority_config