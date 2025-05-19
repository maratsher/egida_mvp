import yaml

def load_config(path: str) -> dict:
    """
    Load YAML configuration from the given path.
    """
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg