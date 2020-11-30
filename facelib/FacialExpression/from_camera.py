from Retinaface.Retinaface import FaceDetector
from FacialExpression.FaceExpression import EmotionDetector
import cv2

face_detector = FaceDetector(name='mobilenet', weight_path='../Retinaface/weights/mobilenet.pth', device='cuda', face_size=(224, 224))
emotion_detector = EmotionDetector(name='densnet121', weight_path='weights/densnet121.pth', device='cuda')


vid = cv2.VideoCapture(0)
vid.set(3, 1280)
vid.set(4, 720)
while True:
    ret, frame = vid.read()
    faces, boxes, scores, landmarks = face_detector.detect_align(frame)
    if len(faces.shape) > 1:
        emotions, emo_probs = emotion_detector.detect_emotion(faces)

        for b in boxes:
            cv2.putText(frame, emotions[0],  (int(b[0]), int(b[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 1.1, [0, 200, 0], 3)
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 3)

        for p in landmarks:
            for i in range(5):
                cv2.circle(frame, (p[i][0], p[i][1]), 3, (0, 255, 0), -1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()