import cv2
from time import time
from facelib import special_draw
from facelib import FaceDetector

detector = FaceDetector(name='mobilenet', weight_path='weights/mobilenet.pth', device='cuda', face_size=(224, 224))

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)
while True:
    _, frame = cap.read()
    tic = time()
    # boxes, scores, landmarks = detector.detect_faces(frame)
    faces, boxes, scores, landmarks = detector.detect_align(frame)
    print('forward time: ', time() - tic)
    if len(faces.shape) > 1:
        for idx, bbox in enumerate(boxes):
            special_draw(frame, bbox, landmarks[idx], name='face', score=scores[idx])

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
