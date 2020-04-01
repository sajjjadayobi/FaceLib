import cv2
from time import time
from Retinaface.Retinaface import FaceDetector

detector = FaceDetector(name='mobilenet', weight_path='Retinaface/weights/mobilenet.pth', device='cuda', face_size=(224, 224))

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
while True:
    _, frame = cap.read()

    tic = time()
    # boxes, scores, landmarks = detector.detect_faces(frame)
    faces, boxes, scores, landmarks = detector.detect_align(frame)
    print('forward time: ', time() - tic)
    if len(faces.shape) > 1:
        for i, f in enumerate(faces.cpu().numpy()):
            cv2.imshow(f'align_{i}', f)

        for b in boxes:
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 3)
        for p in landmarks:
            for i in range(5):
                cv2.circle(frame, (p[i][0], p[i][1]), 3, (0, 255, 0), -1)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
