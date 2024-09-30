import zipfile
import os


def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def create_zip(output_path, directory_to_zip):
    with zipfile.ZipFile(output_path, 'w') as zipf:
        for root, dirs, files in os.walk(directory_to_zip):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, directory_to_zip))


def combine_pklz_files(file1, file2, output_file):
    # Create temporary directories for extraction
    temp_dir1 = '/auto/globalscratch/users/v/s/vsomers/logs/simformer/tmp/temp_extract1'
    temp_dir2 = '/auto/globalscratch/users/v/s/vsomers/logs/simformer/tmp/temp_extract2'

    # Extract both zip files
    os.makedirs(temp_dir1, exist_ok=True)
    os.makedirs(temp_dir2, exist_ok=True)

    print(f"Extracting {file1} to {temp_dir1}")
    extract_zip(file1, temp_dir1)

    print(f"Extracting {file2} to {temp_dir2}")
    extract_zip(file2, temp_dir2)

    # Combine directories
    combined_dir = '/auto/globalscratch/users/v/s/vsomers/logs/simformer/tmp/temp_combined'
    os.makedirs(combined_dir, exist_ok=True)

    # Move files from both directories to combined directory
    for temp_dir in [temp_dir1, temp_dir2]:
        for item in os.listdir(temp_dir):
            print(f"Moving {item} to {combined_dir}")
            s = os.path.join(temp_dir, item)
            d = os.path.join(combined_dir, item)
            if os.path.isdir(s):
                os.makedirs(d, exist_ok=True)
                for subitem in os.listdir(s):
                    os.rename(os.path.join(s, subitem), os.path.join(d, subitem))
            else:
                os.rename(s, d)

    # Create a new zip file from the combined directory
    print(f"Creating combined zip file {output_file}")
    create_zip(output_file, combined_dir)

    # Clean up temporary directories
    for temp_dir in [temp_dir1, temp_dir2, combined_dir]:
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(temp_dir)


# File paths
file1 = "/auto/globalscratch/users/v/s/vsomers/logs/simformer/2024-09-17/12-09-18/states/simformer.pklz"
file2 = "/auto/globalscratch/users/v/s/vsomers/logs/simformer/2024-09-17/12-09-43/states/simformer.pklz"
# file2 = "/auto/globalscratch/users/v/s/vsomers/logs/simformer/2024-09-13/19-21-28/states/simformer.pklz"
# file1 = "/auto/globalscratch/users/v/s/vsomers/logs/simformer/2024-09-13/19-18-32/states/simformer.pklz"
# file2 = '/auto/globalscratch/users/v/s/vsomers/logs/simformer/2024-09-12/20-56-32/states/simformer.pklz'  # split 0
# file1 = '/auto/globalscratch/users/v/s/vsomers/logs/simformer/2024-09-12/20-56-18/states/simformer.pklz'  # split 1
# file1 = '/auto/globalscratch/users/v/s/vsomers/logs/simformer/2024-09-12/18-05-50/states/simformer.pklz'
# file2 = '/auto/globalscratch/users/v/s/vsomers/logs/simformer/2024-09-12/18-07-08/states/simformer.pklz'
output_file = '/auto/globalscratch/users/v/s/vsomers/logs/simformer/states/combined_simformer.pklz'

# Combine the files
combine_pklz_files(file1, file2, output_file)
