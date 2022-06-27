import matplotlib.pyplot as plt
from facelib import FaceDetector, AgeGenderEstimator

img = plt.imread("Yannos_Papantoniou.jpg")
face_detector = FaceDetector()
age_gender_detector = AgeGenderEstimator()

faces, boxes, scores, landmarks = face_detector.detect_align(img)
genders, ages = age_gender_detector.detect(faces)
print(genders, ages)