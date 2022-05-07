# [FaceLib](https://github.com/sajjjadayobi/FaceLib): Face Analysis
Used for face detection, facial expression, AgeGender estimation and recognition with PyTorch.
- Instalation: `pip install git+https://github.com/sajjjadayobi/FaceLib.git`

## How to use:
Check this [example_notebook](https://github.com/sajjjadayobi/FaceLib/blob/master/example_notebook.ipynb) or take a look at the following sections
 
## 1. Face Detection: RetinaFace
You can use these backbone networks: Resnet50, mobilenet. Default model is `mobilenet` and it will be automatically downloaded.
- The following example illustrates the ease of use of this package on your webcam:
```python
     from facelib import WebcamFaceDetector
   detector = WebcamFaceDetector()
   detector.run()
```
- Low-level access to bounding boxes and face landmarks
```python
   from facelib import FaceDetector
   detector = FaceDetector()
   boxes, scores, landmarks = detector.detect_faces(image)
```

## 2. Face Alignment: Using face landmarkd
For face aligment always use the `detect_align` function it gives you better performance.
- Face detection and aligment using the `detect_align` function.
```python
 from facelib import FaceDetector
 detector = FaceDetector()
 faces, boxes, scores, landmarks = detector.detect_align(image)
```

Original | Aligned & Resized | Original | Aligned & Resized |
|---|---|---|---|
|![image](https://github.com/sajjjadayobi/FaceLib/blob/master/facelib/imgs/input1.jpg)|![image](https://github.com/sajjjadayobi/FaceLib/blob/master/facelib/imgs/res1.jpg)|![image](https://github.com/sajjjadayobi/FaceLib/blob/master/facelib/imgs/input2.jpg)|![image](https://github.com/sajjjadayobi/FaceLib/blob/master/facelib/imgs/res2.jpg)|


## 3. Age & Gender Estimation:
`ShufflenetFull` is the default model, and it will be automatically downloaded.
- Age and gender estimation live on your webcam (or any camera)
 ```python
from facelib import WebcamAgeGenderEstimator
estimator = WebcamAgeGenderEstimator()
estimator.run()
```
  
- Low-lvel access to ages and genders 
```python
from facelib import FaceDetector, AgeGenderEstimator
face_detector = FaceDetector()
age_gender_detector = AgeGenderEstimator()

faces, boxes, scores, landmarks = face_detector.detect_align(image)
genders, ages = age_gender_detector.detect(faces)
print(genders, ages)
```

## 4. Facial Expression Recognition:
The default model is `densnet121` and it will be automatically downloaded. Note that face size must be (224, 224).
- Emotion detector live on your webcam
```python
from facelib import WebcamEmotionDetector
detector = WebcamEmotionDetector()
detector.run()
```

- Emotions as an array with their probabilities
```python
from facelib import FaceDetector, EmotionDetector
face_detector = FaceDetector(face_size=(224, 224))
emotion_detector = EmotionDetector()

faces, boxes, scores, landmarks = face_detector.detect_align(image)
emotions, probab = emotion_detector.detect_emotion(faces)
```
- on my Webcam 🙂
![Alt Text](https://github.com/sajjjadayobi/FaceLib/blob/master/facelib/imgs/emotion.gif)

## 5. Face Recognition: InsightFace
- This module is a pytorch reimplementation of Arcface(paper), or Insightface(Github)

#### Pretrained Models & Performance
- IR-SE50


| LFW(%) | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) | calfw(%) | cplfw(%) | vgg2_fp(%) |
| ------ | --------- | --------- | ----------- | -------- | -------- | ---------- |
| 0.9952 | 0.9962    | 0.9504    | 0.9622      | 0.9557   | 0.9107   | 0.9386     |
- Mobilefacenet

| LFW(%) | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) | calfw(%) | cplfw(%) | vgg2_fp(%) |
| ------ | --------- | --------- | ----------- | -------- | -------- | ---------- |
| 0.9918 | 0.9891    | 0.8986    | 0.9347      | 0.9402   | 0.866    | 0.9100     |

#### Prepare the Facebank
Save the images of the **faces** you want to detect in this folder
```
Insightface/models/data/facebank/
  ---> person_1/
      ---> img_1.jpg
      ---> img_2.jpg
  ---> person_2/
      ---> img_1.jpg
      ---> img_2.jpg
```
You can save a new preson in facebank with 2 ways:
- Use `add_from_webcam`: it takes 4 images and saves them on facebank.
```python
 from facelib import add_from_webcam
 add_from_webcam(person_name='sajjad')
```
- use `add_from_folder`: it takes a path with some images from just a person.
```python
 from facelib import add_from_folder
 add_from_folder(folder_path='./', person_name='sajjad')
```

#### Recognizer
The default model is `mobilenet` and it will be automatically downloaded 
- Face Recognition live on your webcam
```python
from facelib import WebcamVerify
verifier = WebcamVerify(update=True)
verifier.run()
```
- Low-level access to your images
```python
import cv2
from facelib import FaceRecognizer, FaceDetector
from facelib import update_facebank, load_facebank, special_draw, get_config

conf = get_config()
# conf.use_mobilenet=False # if you want to use the bigger model
detector = FaceDetector(device=conf.device)
face_rec = FaceRecognizer(conf)

# set True when you add someone new to the facebank
update_facebank_for_add_new_person = False
if update_facebank_for_add_new_person:
    targets, names = update_facebank(conf, face_rec.model, detector)
else:
    targets, names = load_facebank(conf)

image = cv2.imread(your_path)
faces, boxes, scores, landmarks = detector.detect_align(image)
results, score = face_rec.infer(faces, targets)
print(names[results.cpu()])
for idx, bbox in enumerate(boxes):
    special_draw(image, bbox, landmarks[idx], names[results[idx]+1], score[idx])
```
![image](https://github.com/sajjjadayobi/FaceLib/blob/master/facelib/imgs/face_rec.jpg)

Reference: [InsightFace](https://github.com/TreB1eN/InsightFace_Pytorch)
