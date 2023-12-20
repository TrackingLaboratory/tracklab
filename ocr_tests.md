---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import easyocr
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
```

```python
# img_path = '/globalscratch/users/a/b/abolfazl/PbTrack_files/dataset/TinyPoseTrack21/images/test/000001_mpiinew_test/000081.jpg'
img_dir = '/globalscratch/users/a/b/abolfazl/PbTrack_files/'
images = os.listdir(img_dir)

```

```python
reader = easyocr.Reader(['en'])
for img_ in images:
    if img_.endswith('.png'):
        img_path = os.path.join(img_dir, img_)
        print(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = reader.readtext(img, low_text=0.2, mag_ratio=1.5,text_threshold=0.1, link_threshold=0.1)
        print(result)
        for i in result:
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = cv2.putText(img, text=i[1], org=i[0][0], fontFace=font, fontScale=1, color=(0,255,0), thickness=5)
            img = cv2.rectangle(img,i[0][0], i[0][2], color=(255,0,0), thickness=3)
        plt.figure()
        plt.imshow(img)
        # plt.scatter(*i[0][0])
        # plt.scatter(*i[0][1])
        # plt.scatter(*i[0][2])
        # plt.scatter(*i[0][3])
        # raise
        
```

```python
result
```

```python
for i in result:
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img, text=i[1], org=i[0][0], fontFace=font, fontScale=1, color=(0,255,0), thickness=5)
```

```python
plt.imshow(img)
```

# using pytesseract

```python
from tesserocr import get_languages
```

```python
img_path = '/globalscratch/users/a/b/abolfazl/PbTrack_files/Screenshot 2023-12-13 at 16.58.49.png'
img = cv2.imread(img_path)

# Adding custom options
custom_config = r'--oem 3 --psm 6'
pytesseract.image_to_string(img, config=custom_config)

```

# using paddleocr

```python
from paddleocr import PaddleOCR,draw_ocr
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
img_path = '/globalscratch/users/a/b/abolfazl/PbTrack_files/Screenshot 2023-12-13 at 16.58.49.png'
result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)


# draw result
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

```python
txts
```

# work on data frame from pbtrack

```python
import pickle
import os
file_path = '/globalscratch/users/a/b/abolfazl/PbTrack_files/outputs/pbtrack/2023-12-13/16-26-38'
with open(file_path + '/detections.pkl', 'rb') as f:
    detections = pickle.load(f)
# with open(file_path + '/metadata.pkl', 'rb') as f:
#     metadata = pickle.load(f)
with open(file_path + '/batch.pkl', 'rb') as f:
    batch = pickle.load(f)
# with open(file_path + '/image.pkl', 'rb') as f:
#     image = pickle.load(f)
```

```python
detections
```

```python
batch['bbox'].device
```

```python
import matplotlib.pyplot as plt
for img, bbox in zip(batch['img'], batch['bbox']):
    plt.figure()
    bbox = bbox.to(int)
    img = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    plt.imshow(img)
```
