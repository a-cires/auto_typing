import yaml
from pathlib import Path

# Define the root of the project
ROOT_DIR = Path(__file__).resolve().parent.parent

def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path
    with config_path.open('r') as f:
        return yaml.safe_load(f)
