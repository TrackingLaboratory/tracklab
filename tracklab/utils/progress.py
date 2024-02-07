from tqdm import tqdm
from rich.progress import track

use_rich = False


def progress(sequence, desc="", total=None):
    if use_rich:
        return track(sequence, description=desc, total=total)
    else:
        return tqdm(sequence, desc=desc, total=total)
