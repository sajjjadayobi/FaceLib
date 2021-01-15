# [FaceLib](https://github.com/sajjjadayobi/FaceLib):
- use for Detection, Facial Expression, Age & Gender Estimation and  Recognition with PyTorch
- this repository works with CPU and GPU(Cuda)

## New Version (is here)
  - improve performance
  - add some new features
    - add default argument for all functions and classes
    - automatic download weight files into codes
    - an example jupyter notebooks
    - fix some bugs
  - Webcam Classes
    - WebcamVerifier
    - WebcamFaceDetector
    - WebcamAgeGenderEstimator
    - WebcamEmotionDetector

## Installation
- Clone and install with this command:
    - ```pip install git+https://github.com/sajjjadayobi/FaceLib.git```
    - or `git clone https://github.com/sajjjadayobi/FaceLib.git`

## How to use:
  - the simplest way is at `example_notebook.ipynb`
  - for low-level usage check out the following sections
  - if you have an NVIDIA GPU don't change the device param if not use `cpu`
 
## 1. Face Detection: RetinaFace

 - you can use these backbone networks: Resnet50, mobilenet
    - default weights and model is `mobilenet` and it will be automatically download
 - for more details, you can see the documentation
 - The following example illustrates the ease of use of this package:

  ```python
   from facelib import FaceDetector
   detector = FaceDetector()
   boxes, scores, landmarks = detector.detect_faces(image)
  ```
 - you can check or change it [Face Detector]()
 - based on this repository [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface)

#### WiderFace Validation Performance on a single scale When using Mobilenet for backbone
| Style | easy | medium | hard |
|:-|:-:|:-:|:-:|
| Pytorch (same parameter with Mxnet) | 88.67% | 87.09% | 80.99% |
| Pytorch (original image scale) | 90.70% | 88.16% | 73.82% |
| Mxnet(original image scale) | 89.58% | 87.11% | 69.12% |


## 2. Face Alignment: Similar Transformation
- always use detect_align it gives you better performance
- you can use this module like this:
  - `detect_align` instead of `detect_faces`

  ```python
   from facelib import FaceDetector
   detector = FaceDetector()
   faces, boxes, scores, landmarks = detector.detect_align(image)
  ```
  
- for more details read detect_image function documentation
- let's see a few examples

Original | Aligned & Resized | Original | Aligned & Resized |
|---|---|---|---|
|![image](https://github.com/sajjjadayobi/FaceLib/blob/master/facelib/imgs/input1.jpg)|![image](https://github.com/sajjjadayobi/FaceLib/blob/master/facelib/imgs/res1.jpg)|![image](https://github.com/sajjjadayobi/FaceLib/blob/master/facelib/imgs/input2.jpg)|![image](https://github.com/sajjjadayobi/FaceLib/blob/master/facelib/imgs/res2.jpg)|


## 3. Age & Gender Estimation:
- I used UTKFace DataSet for Age & Gender Estimation
  - default weights and model is `ShufflenetFull` and it will be automatically download
- you can use this module like this:

 ```python
    from facelib import FaceDetector, AgeGenderEstimator

    face_detector = FaceDetector()
    age_gender_detector = AgeGenderEstimator()

    faces, boxes, scores, landmarks = face_detector.detect_align(image)
    genders, ages = age_gender_detector.detect(faces)
    print(genders, ages)
  ```


## 4. Facial Expression Recognition:
- Facial Expression Recognition using Residual Masking Network
  - default weights and model is `densnet121` and it will be automatically download
- face size must be (224, 224), you can fix it in FaceDetector init function with face_size=(224, 224)

 ```python
   from facelib import FaceDetector, EmotionDetector
  
   face_detector = FaceDetector(face_size=(224, 224))
   emotion_detector = EmotionDetector()

   faces, boxes, scores, landmarks = face_detector.detect_align(image)
   list_of_emotions, probab = emotion_detector.detect_emotion(faces)
   print(list_of_emotions)
  ```

- like this image:

![image](https://github.com/sajjjadayobi/FaceLib/blob/master/facelib/imgs/expression.jpg)


## 5. Face Recognition: InsightFace
- This module is a reimplementation of Arcface(paper), or Insightface(Github)

#### Pretrained Models & Performance

- IR-SE50

| LFW(%) | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) | calfw(%) | cplfw(%) | vgg2_fp(%) |
| ------ | --------- | --------- | ----------- | -------- | -------- | ---------- |
| 0.9952 | 0.9962    | 0.9504    | 0.9622      | 0.9557   | 0.9107   | 0.9386     |

- Mobilefacenet

| LFW(%) | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) | calfw(%) | cplfw(%) | vgg2_fp(%) |
| ------ | --------- | --------- | ----------- | -------- | -------- | ---------- |
| 0.9918 | 0.9891    | 0.8986    | 0.9347      | 0.9402   | 0.866    | 0.9100     |

#### Prepare the Facebank (For testing over camera, video or image)
- the faces images you want to detect it save them in this folder:
    ```
    Insightface/models/data/facebank/
              ---> person_1/
                  ---> img_1.jpg
                  ---> img_2.jpg
              ---> person_2/
                  ---> img_1.jpg
                  ---> img_2.jpg
    ```
- you can save a new preson in facebank with 3 ways:

  - use `add_from_webcam`: it takes 4 images and saves them on facebank
   ```python
      from facelib import add_from_webcam
      add_from_webcam(person_name='sajjad')
   ```
  
  - use `add_from_folder`: it takes a path with some images from just a person 
    ```python
        from facelib import add_from_folder
        add_from_webcam(folder_path='./', person_name='sajjad')
    ```
  
  - or add faces manually (just face of a person not image of a person)
    - I don't suggest this

#### Using
- default weights and model is `mobilenet` and it will be automatically download

```python
    import cv2
    from facelib import FaceRecognizer, FaceDetector
    from facelib import update_facebank, load_facebank, special_draw, get_config
 
    conf = get_config()
    detector = FaceDetector()
    face_rec = FaceRecognizer(conf)
    face_rec.model.eval()
    
    # set True when you add someone new 
    update_facebank_for_add_new_person = False
    if update_facebank_for_add_new_person:
        targets, names = update_facebank(conf, face_rec.model, detector)
    else:
        targets, names = load_facebank(conf)

    image = cv2.imread(your_path)
    faces, boxes, scores, landmarks = detector.detect_align(image)
    results, score = face_rec.infer(conf, faces, targets)
    for idx, bbox in enumerate(boxes):
        special_draw(image, bbox, landmarks[idx], names[results[idx]+1], score[idx])
```

- example of run this code:

![image](https://github.com/sajjjadayobi/FaceLib/blob/master/facelib/imgs/face_rec.jpg)

## Reference:
- [InsightFace](https://github.com/TreB1eN/InsightFace_Pytorch)
- [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface)
- [Facial Expression](https://github.com/phamquiluan/ResidualMaskingNetwork)
## Citation:

 ```
    - Author : Sajjad Ayoubi
    - Title : FaceLib
    - Year = 2021
 ```
