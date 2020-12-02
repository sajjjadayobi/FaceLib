from facelib import FaceDetector
from facelib import AgeGenderEstimator
from facelib import special_draw
import cv2
from time import time

face_detector = FaceDetector(name='mobilenet', weight_path='../Retinaface/weights/mobilenet.pth', device='cuda')
age_gender_detector = AgeGenderEstimator(name='full', weight_path='weigths/ShufflenetFull.pth', device='cuda')

vid = cv2.VideoCapture(1)
vid.set(3, 1280)
vid.set(4, 720)
while True:
    ret, frame = vid.read()
    faces, boxes, scores, landmarks = face_detector.detect_align(frame)
    if len(faces.shape) > 1:
        tic = time()
        genders, ages = age_gender_detector.detect(faces)
        print(time()-tic)
        for i, b in enumerate(boxes):
            special_draw(frame, b, landmarks[i], name=genders[i]+' '+str(ages[i]))

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()