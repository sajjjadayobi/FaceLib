import cv2 as cv
from time import time
from Retinaface import FaceDetector

cap = cv.VideoCapture(0)
detector = FaceDetector(name='mobilenet', weight_path='weights/mobilenet.pth', device='cpu', face_size=(224, 224))
while True:
    _, frame = cap.read()

    tic = time()
    # boxes, scores, landmarks = detector.detect_faces(frame)
    faces, boxes, scores, landmarks = detector.detect_align(frame)
    print('time: ', time() - tic)
    for i, f in enumerate(faces.cpu().numpy()):
        cv.imshow(f'align_{i}', f)

    for b in boxes:
        cv.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 255), 1)
    for p in landmarks:
        for i in range(5):
            cv.circle(frame, (p[i][0], p[i][1]), 1, (255, 0, 0), -1)

    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()




