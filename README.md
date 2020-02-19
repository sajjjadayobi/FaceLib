# FaceLib : Face Detection and Face Recognition with pytorch

## Detection: RetinaFace
 you can use this backbone networks:
    Resnet50, mobilenet, slim, RFB
 
.. code:: python

    >>> from Retinaface import FaceDetector
    >>> detector = FaceDetector(name='mobilenet', weight_path='mobilenet.pth', device='cpu')
    >>> boxes, scores, landmarks = detector.detect_faces(your_image)


## Alignment: similarity transform


## recognition
