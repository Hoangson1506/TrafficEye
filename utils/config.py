import yaml
import os

def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def save_config(config, config_path="config.yaml"):
    """Save configuration to a YAML file.

    Args:
        config (dict): Configuration dictionary to be saved.
        config_path (str, optional): Path to save the configuration file. Defaults to config.yaml
    """

    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)

    print(f"Configuration saved to {config_path}")