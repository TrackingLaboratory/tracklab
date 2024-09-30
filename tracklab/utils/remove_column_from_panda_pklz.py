import pandas as pd
import pickle
import zipfile
from functools import lru_cache
import os

class ZipPickleHandler:
    def __init__(self, input_zip_path, output_zip_path):
        self.input_zip_path = input_zip_path
        self.output_zip_path = output_zip_path
        self._zf = None

    @property
    def zf(self):
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.input_zip_path, mode="r")
        return self._zf

    @lru_cache(maxsize=1000)
    def _load_pickle(self, sample_video_id):
        with self.zf.open(sample_video_id, "r") as fp:
            df = pickle.load(fp)
        return df

    def modify_and_save(self):
        # Open the new zip file for writing the reduced data
        with zipfile.ZipFile(self.output_zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as output_zf:
            for sample_video_id in self.zf.namelist():
                # Check if the file is a .pkl
                if sample_video_id.endswith('.pkl'):
                    try:
                        # Load the pickle file
                        df = self._load_pickle(sample_video_id)

                        # Remove 'body_masks' column if it exists
                        if 'body_masks' in df.columns:
                            df = df.drop(columns=['body_masks'])

                        # Write the modified DataFrame back into the new zip file
                        with output_zf.open(sample_video_id, 'w') as fp:
                            pickle.dump(df, fp)

                    except Exception as e:
                        print(f"Error processing {sample_video_id}: {e}")

                else:
                    # For non-pickle files like .json, copy them without changes
                    with self.zf.open(sample_video_id, "r") as source_file:
                        # Read the file contents
                        file_data = source_file.read()
                        # Write the file unchanged to the new zip
                        with output_zf.open(sample_video_id, 'w') as target_file:
                            target_file.write(file_data)

        print(f"Modified DataFrames and untouched files have been saved to {self.output_zip_path}.")

# Usage
# set_type = "train"
# set_type = "val"
# input_zip_path = f"/home/ucl/elen/vsomers/datasets/shared_datasets/DanceTrack/states/simformer_training/ddsort-kpr-dancetrack_{set_type}_old_with_masks.pklz"
# output_zip_path = f"/home/ucl/elen/vsomers/datasets/shared_datasets/DanceTrack/states/simformer_training/ddsort-kpr-dancetrack_{set_type}.pklz"

# set_type = "train"
set_type = "val"
input_zip_path = f"/home/ucl/elen/vsomers/datasets/shared_datasets/SportsMOT/states/simformer_training/ddsort-kpr-sportsmot_{set_type}_old_with_masks.pklz"
output_zip_path = f"/home/ucl/elen/vsomers/datasets/shared_datasets/SportsMOT/states/simformer_training/ddsort-kpr-sportsmot_{set_type}.pklz"

handler = ZipPickleHandler(input_zip_path, output_zip_path)
handler.modify_and_save()