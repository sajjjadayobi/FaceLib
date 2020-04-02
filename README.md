# FaceLib : Detection and Facial Expression and Age & Gender and  Recognition with pytorch

## Face Detection: RetinaFace

 - you can use this backbone networks: Resnet50, mobilenet, slim, RFB
 - The following example illustrates the ease of use of this package:

  ```python
   from Retinaface.Retinaface import FaceDetector
   detector = FaceDetector(name='mobilenet', weight_path='mobilenet.pth', device='cpu')
   boxes, scores, landmarks = detector.detect_faces(image)
  ```
  
## WiderFace Validation Performance in single scale When using Mobilenet
| Style | easy | medium | hard |
|:-|:-:|:-:|:-:|
| Pytorch (same parameter with Mxnet) | 88.67% | 87.09% | 80.99% |
| Pytorch (original image scale) | 90.70% | 88.16% | 73.82% |
| Mxnet | 88.72% | 86.97% | 79.19% |
| Mxnet(original image scale) | 89.58% | 87.11% | 69.12% |


## Image Alignment: similar transformation

  ```python
   from Retinaface.Retinaface import FaceDetector
   detector = FaceDetector(name='mobilenet', weight_path='mobilenet.pth', device='cuda')
   faces, boxes, scores, landmarks = detector.detect_align(image)
  ```
- let us see a few examples

Original | Aligned & Resized | Original | Aligned & Resized |
|---|---|---|---|
|![image](https://github.com/sajjjadayobi/FaceRec/blob/master/imgs/input1.jpg)|![image](https://github.com/sajjjadayobi/FaceRec/blob/master/imgs/res1.jpg)|![image](https://github.com/sajjjadayobi/FaceRec/blob/master/imgs/input2.jpg)|![image](https://github.com/sajjjadayobi/FaceRec/blob/master/imgs/res2.jpg)|

## Age & Gender Classification:
- I use UTKFace DataSet for Age & Gender
   
 ```python
    from Retinaface.Retinaface import FaceDetector
    from AgeGender.Detector import AgeGender
     
    face_detector = FaceDetector(name='mobilenet', weight_path='mobilenet.pth', device='cuda')
    age_gender_detector = AgeGender(name='full', weight_path='ShufflenetFull.pth', device='cuda')

    faces, boxes, scores, landmarks = face_detector.detect_align(image)
    genders, ages = age_gender_detector.detect(faces)
  ```
## Facial Expression Recognition:
- face must be (224, 224)
 
 ```python
   from Retinaface.Retinaface import FaceDetector
   from FacialExpression.FaceExpression import EmotionDetector
   
   face_detector = FaceDetector(name='mobilenet', weight_path='mobilenet.pth', device='cuda')
   emotion_detector = EmotionDetector(name='resnet34', weight_path='resnet34.pth', device='cuda')
   
   faces, boxes, scores, landmarks = face_detector.detect_align(image)
   list_of_emotions, probab = emotion_detector.detect_emotion(faces)
  ```
- like this image:
![image](https://github.com/sajjjadayobi/FaceRec/blob/master/imgs/expression.jpg)

## Face Recognition: InsightFace
- This repo is a reimplementation of Arcface(paper), or Insightface(github)
- For models, including the pytorch implementation of the backbone modules of ir_se50 and MobileFacenet

##### Prepare Facebank (For testing over camera or video) 
- Provide the face images your want to detect in the data/face_bank folder
    ```
    data/facebank/
            ---> person_1/
                ---> img_1.jpg
            ---> person_2/
                ---> img_1.jpg
                ---> img_2.jpg
    ```
- you can use this repo like this for camera verification:

```
    python camera_verify.py -u update -m True
```

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

## Reference:
- [InsightFace](https://github.com/TreB1eN/InsightFace_Pytorch)
- [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface)
- [Facial Expression](https://github.com/phamquiluan/ResidualMaskingNetwork)
## Citation:
    - Author : Sajjad Ayoubi
    - Title : FaceLib
    - Year = 2020
