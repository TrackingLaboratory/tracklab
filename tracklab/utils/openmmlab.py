import os
from pathlib import Path

import requests
try:
    from mim.utils import get_installed_path
except ImportError:
    get_installed_path = lambda x: "."
from tqdm import tqdm

import logging

log = logging.getLogger(__name__)

def get_config_path(package, config_name):
    installed_path = Path(get_installed_path(package))
    # configs will be put in package/.mim in PR #68
    possible_config_paths = [
        installed_path / '.mim' / config_name,
        installed_path / config_name,
    ]
    for config_path in possible_config_paths:
        if config_path.exists():
            return config_path

def get_checkpoint(path_to_checkpoint, download_url):
    os.makedirs(os.path.dirname(path_to_checkpoint), exist_ok=True)
    if not os.path.exists(path_to_checkpoint):
        log.info("Checkpoint not found at {}".format(path_to_checkpoint))
        log.info("Downloading checkpoint from {}".format(download_url))
        response = requests.get(download_url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit="B", unit_scale=True)
        with open(path_to_checkpoint, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            log.warning(
                f"Something went wrong while downloading or writing {download_url} to {path_to_checkpoint}"
            )
        else:
            log.info("Checkpoint downloaded successfully")
