import cv2
from Retinaface import FaceDetector

# detector = FaceDetector(name='mobilenet', weight_path='weights/mobilenet.pth', device='cuda', face_padding=0.1, face_size=(224, 224))
img = cv2.imread('imgs/antoni.jpg')
img = cv2.resize(img, (224, 244))
# faces, boxes, scores, landmarks = detector.detect_align(img)
# cv2.imshow('test', faces[0])
cv2.imwrite('antoni.jpg', img)
