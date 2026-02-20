"""
Core pipeline utilities for image editing operations.
"""
import os
import subprocess
import signal
import json
import yaml
import platform
from typing import Dict, List, Optional


def load_pipeline_config(config_path: str = "configs/pipeline_config.yaml") -> dict:
    """Load pipeline configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def load_dataset_config(config_path: str = "configs/dataset_config.yaml") -> dict:
    """Load dataset configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def load_combined_config(
    pipeline_config_path: str = "configs/pipeline_config.yaml",
    dataset_config_path: str = "configs/dataset_config.yaml"
) -> dict:
    """Load and combine pipeline and dataset configurations."""
    pipeline_config = load_pipeline_config(pipeline_config_path)
    dataset_config = load_dataset_config(dataset_config_path)
    
    # Merge configs - pipeline config takes precedence for overlapping keys
    combined_config = {**dataset_config, **pipeline_config}
    
    # Explicitly add output_directories and image_sources from dataset config
    combined_config['output_directories'] = dataset_config.get('output_directories', {})
    combined_config['image_sources'] = dataset_config.get('image_sources', {})
    
    return combined_config


def get_gimp_command(config: dict) -> List[str]:
    """Get appropriate GIMP command based on platform."""
    system = platform.system().lower()
    
    if system == "darwin":
        gimp_path = config["gimp"]["paths"]["macos"]
    elif system == "linux":
        gimp_path = config["gimp"]["paths"]["linux"] 
    elif system == "windows":
        gimp_path = config["gimp"]["paths"]["windows"]
    else:
        raise RuntimeError(f"Unsupported platform: {system}")
    
    current_dir = os.getcwd()
    image_ops_path = os.path.join(current_dir, 'image_ops')
    
    if system == "linux" and "flatpak" in gimp_path:
        return gimp_path.split() + [
            "--quit",
            "-idf",
            "--batch-interpreter", config["gimp"]["batch_interpreter"],
            "-b", f"import sys;sys.path=['{current_dir}','{image_ops_path}']+sys.path;exec(open('{image_ops_path}/gimp_pipeline.py').read())",
        ]
    else:
        return [
            gimp_path,
            "--quit",
            "-idf",
            "--batch-interpreter", config["gimp"]["batch_interpreter"],
            "-b", f"import sys;sys.path=['{current_dir}','{image_ops_path}']+sys.path;exec(open('{image_ops_path}/gimp_pipeline.py').read())",
        ]


def update_pipeline_file_paths(
    file_path: str, 
    config_path: str, 
    src_image_path: str, 
    output_image_path: str
) -> None:
    """Update the paths in the GIMP pipeline file."""
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()

        with open(file_path, "w") as file:
            for line in lines:
                if line.strip().startswith("config_path"):
                    file.write(f"config_path = '{config_path}'\n")
                elif line.strip().startswith("src_image_path"):
                    file.write(f"src_image_path = '{src_image_path}'\n")
                elif line.strip().startswith("output_image_path"):
                    file.write(f"output_image_path = '{output_image_path}'\n")
                else:
                    file.write(line)
        print("Pipeline file paths updated successfully.")
    except Exception as e:
        print(f"Error updating pipeline file paths: {e}")
        raise


def execute_gimp_pipeline(config: Optional[dict] = None) -> bool:
    """
    Execute the GIMP pipeline with current configuration.

    Returns:
        True on successful execution, False on timeout/failure.
    """
    if config is None:
        config = load_pipeline_config()
    
    # Ensure temp directories exist before GIMP execution
    if "image_processing" in config and "temp_paths" in config["image_processing"]:
        for temp_path in config["image_processing"]["temp_paths"]:
            ensure_directory(os.path.dirname(temp_path))
    
    command = get_gimp_command(config)
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = config["gimp"]["python_warnings"]
    timeout_seconds = int(config.get("processing", {}).get("timeout_seconds", 120))

    try:
        # start_new_session allows us to terminate the full process tree on timeout.
        process = subprocess.Popen(
            command,
            env=env,
            start_new_session=(os.name != "nt"),
        )
    except Exception as exc:
        print(f"Failed to launch GIMP pipeline: {exc}")
        return False

    try:
        process.wait(timeout=timeout_seconds if timeout_seconds > 0 else None)
    except subprocess.TimeoutExpired:
        print(
            f"GIMP pipeline timed out after {timeout_seconds}s. "
            "Terminating process tree and continuing with non-GIMP output."
        )
        _terminate_process_tree(process)
        return False

    if process.returncode != 0:
        print(f"GIMP pipeline exited with code {process.returncode}.")
        return False

    return True


def _terminate_process_tree(process: subprocess.Popen) -> None:
    """Best-effort termination for a timed-out subprocess and its children."""
    if process.poll() is not None:
        return

    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
        return
    except Exception:
        pass

    try:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    except Exception:
        pass


def flip_json_signs(file_path: str) -> None:
    """Flip the signs of all numeric values in a JSON file."""
    def flip_signs(data):
        if isinstance(data, dict):
            return {key: flip_signs(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [flip_signs(item) for item in data]
        elif isinstance(data, (int, float)):
            return -data
        return data

    with open(file_path, 'r') as file:
        data = json.load(file)

    updated_data = flip_signs(data)

    with open(file_path, 'w') as file:
        json.dump(updated_data, file, indent=4)


def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if not."""
    os.makedirs(path, exist_ok=True)


def get_image_source_path(
    image_index: str, 
    config: dict, 
    source_type: str = "auto"
) -> str:
    """Get the appropriate source image path based on image index prefix."""
    sources = config["image_sources"]
    
    if source_type == "auto":
        if image_index.startswith("a"):
            return f"{sources['adobe5k_720']}/{image_index}.tif"
        elif image_index.startswith("00"):
            return f"{sources['flickr2k']}/{image_index}.png"
        else:
            return f"{sources['ppr10k_target_a']}/{image_index}.tif"
    else:
        return f"{sources[source_type]}/{image_index}.tif"
