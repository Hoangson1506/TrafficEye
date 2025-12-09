import pytest
import os
import yaml
from utils.config import load_config

def test_load_config_success(tmp_path):
    # Create a dummy config file
    config_data = {
        'section': {
            'key': 'value'
        }
    }
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    # Load it
    config = load_config(str(config_file))
    assert config['section']['key'] == 'value'

def test_load_config_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config.yaml")

def test_default_config_exists():
    # Verify that the actual config.yaml exists and can be loaded
    assert os.path.exists("config.yaml")
    config = load_config("config.yaml")
    assert 'detections' in config
    assert 'tracking' in config
    assert 'violation' in config
