import cv2
import torch
from facelib import special_draw
from facelib import FaceDetector


class WebcamFaceDetector:
    def __init__(self, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        print('loading ...') 
        self.detector = FaceDetector(face_size=(224, 224), device=device)


    def run(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        cap.set(3, 1280)
        cap.set(4, 720)
        print('type q for exit')
        while cap.isOpened():
            ret , frame = cap.read()
            if ret == False:
                raise Exception('the camera not recognized: change camera_index param to ' + str(0 if camera_index == 1 else 1))
            # boxes, scores, landmarks = detector.detect_faces(frame)
            faces, boxes, scores, landmarks = self.detector.detect_align(frame)
            if len(faces.shape) > 1:
                for idx, bbox in enumerate(boxes):
                    special_draw(frame, bbox, landmarks[idx], name='face', score=scores[idx])

                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
