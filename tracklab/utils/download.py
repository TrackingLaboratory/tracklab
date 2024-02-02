from pathlib import Path

import requests
import hashlib
from tqdm import tqdm

def download_file(url, local_filename, md5=None):
    # NOTE the stream=True parameter below
    if Path(local_filename).exists():
        if md5 is not None:
            if check_md5(local_filename, md5):
                return local_filename
            else:
                raise ValueError(f'MD5 checksum mismatch for file {local_filename}, '
                                 f'please re-download it from {url}')

    Path(local_filename).parent.mkdir(exist_ok=True, parents=True)
    file_hash = hashlib.md5()
    with (requests.get(url, stream=True) as r):
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        chunk_size = 8192
        with (open(local_filename, 'wb') as f,
             tqdm(desc=f"Downloading {Path(local_filename).name}", total=total_size, unit="B", unit_scale=True) as progress_bar):
            for chunk in r.iter_content(chunk_size=chunk_size):
                file_hash.update(chunk)
                f.write(chunk)
                progress_bar.update(len(chunk))
    if md5 is not None:
        if md5 != file_hash.hexdigest():
            raise ValueError(f'MD5 checksum mismatch when downloading file from {url}. '
                             f'Please download it manually from {url} to {local_filename}.')
    return local_filename


def check_md5(local_filename, md5):
    with open(local_filename, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest() == md5
