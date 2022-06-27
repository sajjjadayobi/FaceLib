from facelib import FaceDetector, EmotionDetector
import matplotlib.pyplot as plt

face_detector = FaceDetector(face_size=(224, 224))
emotion_detector = EmotionDetector()
img = plt.imread("Yannos_Papantoniou.jpg")

faces, boxes, scores, landmarks = face_detector.detect_align(img)
emotions, probab = emotion_detector.detect_emotion(faces)
print(emotions)

