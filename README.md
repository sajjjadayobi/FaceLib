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
## Face Recognition
- very soon will be completed


## Citation
- Author : Sajjad Ayoubi
- Title : FaceLib
- Year = 2020
