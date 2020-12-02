from facelib import FaceDetector
from facelib import EmotionDetector
from facelib import special_draw
import cv2

face_detector = FaceDetector(name='mobilenet', weight_path='../Retinaface/weights/mobilenet.pth', device='cuda', face_size=(224, 224))
emotion_detector = EmotionDetector(name='densnet121', weight_path='weights/densnet121.pth', device='cuda')

vid = cv2.VideoCapture(1)
vid.set(3, 1280)
vid.set(4, 720)
while True:
    ret, frame = vid.read()
    faces, boxes, scores, landmarks = face_detector.detect_align(frame)
    if len(faces.shape) > 1:
        emotions, emo_probs = emotion_detector.detect_emotion(faces)
        for i, b in enumerate(boxes):
            special_draw(frame, b, landmarks[i], name=emotions[i], score=emo_probs[i])

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()