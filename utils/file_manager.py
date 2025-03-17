import os
from datetime import datetime

def create_run_folder(base_dir="runs"):
    """Creates a unique run folder using a timestamp."""
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    return run_folder

def create_data_folders(run_folder):
    """Creates subfolders for results, models, and replay buffers inside the run folder."""
    paths = {
        "results": os.path.join(run_folder, "results"),
        "models": os.path.join(run_folder, "models"),
        "replay_buffers": os.path.join(run_folder, "replay_buffers"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths

