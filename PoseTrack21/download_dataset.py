import argparse
import os 
import requests
import shutil
import tarfile
import zipfile
from tqdm.auto import tqdm
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

def download_posetrack_videos(download_path, video_source_url):
    archive_path = f"{video_source_url}/posetrack18_images.tar.a"

    os.makedirs(download_path, exist_ok=True)

    files = [97 + i for i in range(18)]
    for i, f in enumerate(files):
        file_letter = chr(f)
        file_name = f"posetrack18_images.tar.a{file_letter}" 
        save_path = os.path.join(download_path, file_name) 

        if not os.path.exists(save_path) or True:
            # download the file 
            remote_url = f"{archive_path}{file_letter}" 
            print(f"[{i} / {len(files)}]Downloading {remote_url}")
            with requests.get(remote_url, stream=True, verify=False) as r: 
                print(r.headers.get("Content-Length"))
                total_length = int(r.headers.get("Content-Length"))

                with tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as raw:
                    with open(save_path, "wb") as fp:
                        shutil.copyfileobj(raw, fp)
        else:
            print(f"[Skip] {file_name} already exists")

    print("Done")
    print("Merging splits")
    total_file = os.path.join(download_path, 'total.tar')
    if not os.path.exists(total_file):
        with open(total_file, 'wb') as fp:
            for f in tqdm(files):
                file_letter = chr(f) 
                file_name = f"posetrack18_images.tar.a{file_letter}" 
                save_path = os.path.join(download_path, file_name) 

                with open(save_path, 'rb') as read:
                    fp.write(read.read())

    print("Done")
    print("Deleting splits")
    files = [97 + i for i in range(18)]
    for i, f in enumerate(files):
        file_letter = chr(f)
        file_name = f"posetrack18_images.tar.a{file_letter}" 
        save_path = os.path.join(download_path, file_name) 
        os.remove(save_path)

    return total_file

def download_annotations(download_save_path, download_path):
    anno_path = os.path.join(download_save_path, 'annotations.zip')

    if not os.path.exists(anno_path):
        print("Downloading annotations")
        with requests.get(download_path, stream=True) as r:
            total_length = int(r.headers.get("Content-Length"))

            with tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as raw:
                with open(anno_path, "wb") as out:
                    shutil.copyfileobj(raw, out)
    else:
        print("Annotations already downloaded")

    return anno_path 

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--save_path', type=str, default='data/PoseTrack21')
    parser.add_argument('--download_url', type=str, default='https://github.com/anDoer/PoseTrack21/releases/download/v0.1/posetrack21_annotations.zip')
    parser.add_argument('--video_source_url', type=str, default='https://posetrack.net/posetrack18-data/')
    parser.add_argument('--token',  type=str, required=True)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    download_path = 'downloads'
    archive_path = download_posetrack_videos(download_path, video_source_url=args.video_source_url)
    annotation_path = download_annotations(download_path, args.download_url)
    
    print("Unpacking Dataset")

    with tarfile.open(archive_path) as archive_fp:
        archive_fp.extractall(save_path)

    with zipfile.ZipFile(annotation_path, 'r') as zip_fp:
        zip_fp.extractall(save_path, pwd=bytes(args.token,  'utf-8'))
    
