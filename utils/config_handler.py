import json
import os

def save_config(config, folder):
    """Saves the configuration dictionary as a JSON file in the specified folder."""
    config_path = os.path.join(folder, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved at: {config_path}")

