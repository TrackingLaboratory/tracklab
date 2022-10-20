## I get an error when running `download_dataset.py`, what can I do? 

### `posetrack.net` is offline! 
For unknown reasons, `posetrack.net` is **offline** and the dataset can not be downloaded :frowning_face: . We already contacted the [posetrack.net team](mailto:admin@posetrack.net). If you have not downloaded the video data yet, please contact us and we will provide to you an alternative download link. For legal reasons, we can not share the alternative download link publicly.

Please run the following command with the `alternative video source` to obtain the dataset

```
python3 download_dataset.py --save_path /target/root/path/of/the/dataset --token="[your token]" --video_source_url="[NEW URL]"
```

### Alternative download options (outdated as `posetrack.net` is offline)

~~Our script downloads the video data from `posetrack.net`, which is **not** maintained by us. Unfortunately, the SSL-certificate is expiered and for some users this results in an error.~~

~~Alternatively, you can try to download the video data with the following script, which ignores an outdated SSL-certificate~~
```
for part in a b c d e f g h i j k l m n o p q r
do
    wget https://posetrack.net/posetrack18-data/posetrack18_images.tar.a${part} --no-check-certificate
done;
```
~~Once you downloaded the files, move them into your download folder, uncomment lines `14-31` in `download_dataset.py` and re-run the script.~~

~~If you are still not able to download the data, don't hesitate to contact us at `posetrack21[at]googlegroups[dot]com` and we will help you get the data.~~



