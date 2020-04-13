# [FaceLib](https://github.com/sajjjadayobi/FaceLib):
- use for Detection, Facial Expression, Age & Gender Estimation and  Recognition with PyTorch
- this repository works with CPU and GPU(Cuda)


## Installation
- Clone and install with this command: 
  
    ```https://github.com/sajjjadayobi/FaceLib.git```
- Prerequisites
  - Python 3.6+
    - PyTorch 1.4+
  - Torchvision 0.4.0+
    - OpenCV 2.0+
  - requirements.txt
  
  
## 1. Face Detection: RetinaFace

 - you can use these backbone networks: Resnet50, mobilenet, slim, RFB
 - for more details, you can use the documentation
 - The following example illustrates the ease of use of this package:

  ```python
   from Retinaface.Retinaface import FaceDetector
   detector = FaceDetector(name='mobilenet', weight_path='mobilenet.pth', device='cpu')
   boxes, scores, landmarks = detector.detect_faces(image)
  ```
 - downlaod weight of network from google drive [RetinaFace](https://drive.google.com/open?id=1JtO_ZdWUDLHUswJKDBEWImmfMA-rCxlx)
 - you can cheche code and change it [Face Detector]()
 - based on this repository [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface)
  
#### WiderFace Validation Performance on a single scale When using Mobilenet for backbone
| Style | easy | medium | hard |
|:-|:-:|:-:|:-:|
| Pytorch (same parameter with Mxnet) | 88.67% | 87.09% | 80.99% |
| Pytorch (original image scale) | 90.70% | 88.16% | 73.82% |
| Mxnet(original image scale) | 89.58% | 87.11% | 69.12% |


## 2. Face Alignment: Similar Transformation
- you can use this module like this:

  ```python
   from Retinaface.Retinaface import FaceDetector
   detector = FaceDetector(name='mobilenet', weight_path='mobilenet.pth', device='cuda')
   faces, boxes, scores, landmarks = detector.detect_align(image)
  ```
- or run on webcam and shows the result on the image
  
    ```python Retinaface/from_camera.py``` 
    
- detect_image() instead detect_faces()
- for more details read detect_image function documentation
- let's see a few examples

Original | Aligned & Resized | Original | Aligned & Resized |
|---|---|---|---|
|![image](https://github.com/sajjjadayobi/FaceRec/blob/master/imgs/input1.jpg)|![image](https://github.com/sajjjadayobi/FaceRec/blob/master/imgs/res1.jpg)|![image](https://github.com/sajjjadayobi/FaceRec/blob/master/imgs/input2.jpg)|![image](https://github.com/sajjjadayobi/FaceRec/blob/master/imgs/res2.jpg)|


## 3. Age & Gender Estimation:
- I used UTKFace DataSet for Age & Gender Estimation
- you can use these backbone networks: full, tiny
- you can use this module like this:

 ```python
    from Retinaface.Retinaface import FaceDetector
    from AgeGender.Detector import AgeGender
     
    face_detector = FaceDetector(name='mobilenet', weight_path='mobilenet.pth', device='cuda')
    age_gender_detector = AgeGender(name='full', weight_path='ShufflenetFull.pth', device='cuda')

    faces, boxes, scores, landmarks = face_detector.detect_align(image)
    genders, ages = age_gender_detector.detect(faces)
    print(genders, ages)
  ```
 - or run on webcam and shows the result on the image
  
    ```python AgeGender/from_camera.py``` 
 
 - downlaod weight of network from google drive [ShufleNet](https://drive.google.com/open?id=1ija2VNl2xTZM73e5-dnnpE_4-v3qmLN6)
 


## 4. Facial Expression Recognition:
- Facial Expression Recognition using Residual Masking Network
- face size must be (224, 224), you can fix it in FaceDetector init function with face_size=(224, 224)
 
 ```python
   from Retinaface.Retinaface import FaceDetector
   from FacialExpression.FaceExpression import EmotionDetector
   
   face_detector = FaceDetector(name='mobilenet', weight_path='mobilenet.pth', face_size=(224, 224))
   emotion_detector = EmotionDetector(name='resnet34', weight_path='resnet34.pth', device='cuda')
   
   faces, boxes, scores, landmarks = face_detector.detect_align(image)
   list_of_emotions, probab = emotion_detector.detect_emotion(faces)
   print(list_of_emotions)
  ```
- or run on webcam and shows the result on the image
  
    ```python FacialExpression/from_camera.py``` 
    
- downlaod weight of network from google drive [Expression](https://drive.google.com/open?id=1Ocy7TB11med-z6QfaHUQGCSki7Dk_PVV)
- like this image:

![image](https://github.com/sajjjadayobi/FaceRec/blob/master/imgs/expression.jpg)


## 5. Face Recognition: InsightFace
- This module is a reimplementation of Arcface(paper), or Insightface(Github)
- For models, including the PyTorch implementation of the backbone modules of IR-SE50 and MobileFacenet

#### Pretrained Models & Performance

- IR-SE50 

| LFW(%) | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) | calfw(%) | cplfw(%) | vgg2_fp(%) |
| ------ | --------- | --------- | ----------- | -------- | -------- | ---------- |
| 0.9952 | 0.9962    | 0.9504    | 0.9622      | 0.9557   | 0.9107   | 0.9386     |

- Mobilefacenet

| LFW(%) | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) | calfw(%) | cplfw(%) | vgg2_fp(%) |
| ------ | --------- | --------- | ----------- | -------- | -------- | ---------- |
| 0.9918 | 0.9891    | 0.8986    | 0.9347      | 0.9402   | 0.866    | 0.9100     |

##### Prepare the Facebank (For testing over camera or video) 
- Provide the face images your want to detect in the data/face_bank folder, and guarantee it have a structure like following:
    ```
    data/facebank/
            ---> person_1/
                ---> img_1.jpg
            ---> person_2/
                ---> img_1.jpg
                ---> img_2.jpg
    ```
- you can save a preson with 3 ways:

  - use ```python add_face_from_camera.py -n NAME```
    - use ```python add_face_from_dir.py -n NAME```
    - or add faces manually (just face of person not image of a person)
    
- you can use this module like this for camera verification:
  
  ```
    python camera_verify.py -u update -m True
    ```

  - u argument: update FaceBank if add a new person
  - m argument: use Mobilenet for backbone



- and use into your code:

```python
    from InsightFace.data.config import get_config
    from InsightFace.models.Learner import face_learner
    from InsightFace.utils import update_facebank, load_facebank, special_draw
    from Retinaface.Retinaface import FaceDetector
    
    
    conf = get_config(training=False)
    detector = FaceDetector(name='mobilenet', weight_path='mobilenet.pth', device=conf.device)
    conf.use_mobilfacenet = True or False
    face_rec = face_learner(conf, inference=True)
    face_rec.model.eval()
    
    if update_facebank_for_add_new_person:
        targets, names = update_facebank(conf, face_rec.model, detector)
    else:
        targets, names = load_facebank(conf)
    
    faces, boxes, scores, landmarks = detector.detect_align(image)
    results, score = face_rec.infer(conf, faces, targets)
    for idx, bbox in enumerate(boxes):
        special_draw(image, bbox, landmarks[idx], names[results[idx]+1], score[idx])
```

-  downlaod weight of network from google drive [InsightFace](https://drive.google.com/open?id=1vHRseSFfqKZrrcSTfPf3wX0a0Y_ipKPR)
- example of run this code:

![image](https://github.com/sajjjadayobi/FaceRec/blob/master/imgs/face_rec.jpg)

## Reference:
- [InsightFace](https://github.com/TreB1eN/InsightFace_Pytorch)
- [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface)
- [Facial Expression](https://github.com/phamquiluan/ResidualMaskingNetwork)
## Citation:

 ```
    - Author : Sajjad Ayoubi
    - Title : FaceLib
    - Year = 2020
 ```
