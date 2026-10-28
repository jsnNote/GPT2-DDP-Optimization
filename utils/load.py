import yaml
import os


def load_yaml(file_path):
    parent_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(parent_path), "config")
    yaml_path = os.path.join(path, f"{file_path}")

    with open(yaml_path, mode="r") as f:
        return yaml.safe_load(f)
