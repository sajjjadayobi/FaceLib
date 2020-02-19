# FaceLib : Face Detection and Face Recognition with pytorch

## Detection: RetinaFace

A PyTorch implementation of RetinaFace: Single-stage Dense Face Localisation in the Wild. Model size only 1.7M, 
when Retinaface The official code in Mxnet can be found here.

 you can use this backbone networks:
    Resnet50, mobilenet, slim, RFB
 
 The following example illustrates the ease of use of this package:

  ```python
   from Retinaface import FaceDetector
   detector = FaceDetector(name='mobilenet', weight_path='mobilenet.pth', device='cpu')
   boxes, scores, landmarks = detector.detect_faces(your_image)
  ```
  
## WiderFace Validation Performance in single scale When using Mobilenet as backbone net.
| Style | easy | medium | hard |
|:-|:-:|:-:|:-:|
| Pytorch (same parameter with Mxnet) | 88.67% | 87.09% | 80.99% |
| Pytorch (original image scale) | 90.70% | 88.16% | 73.82% |
| Mxnet | 88.72% | 86.97% | 79.19% |
| Mxnet(original image scale) | 89.58% | 87.11% | 69.12% |
<p align="center"><img src="curve/Widerface.jpg" width="640"\></p>


## Alignment: similarity transform


## recognition
