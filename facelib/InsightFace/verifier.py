import cv2
import torch
import argparse
from facelib import get_config, special_draw
from facelib import update_facebank, load_facebank
from facelib import FaceRecognizer
from facelib import FaceDetector


class WebcamVerify:
    """
    WebcamVerify: face verfication with cv2
    
    if you add new person in to facebank
    pass update True
    """  
    def __init__(self, update=True, tta=True, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):   
        print('loading ...') 
        self.tta = tta
        self.detector = FaceDetector(device=device)
        self.conf = get_config()
        self.conf.device = device
        recognizer = FaceRecognizer(self.conf)
        recognizer.model.eval()
        self.recognizer = recognizer

        if update:
            self.targets, self.names = update_facebank(self.conf, recognizer.model, self.detector, tta=self.tta)
        else:
            self.targets, self.names = load_facebank(self.conf)
            
        

    def run(self, camera_index=0):
        if len(self.targets) < 1:
            raise Exception("you don't have any person in facebank: add new person with 'add_from_webcam' or 'add_from_folder' function")
            
        cap = cv2.VideoCapture(camera_index)
        cap.set(3, 1280)
        cap.set(4, 720)
        # frame rate 6 due to my laptop is quite slow...
        print('type q for exit')
        while cap.isOpened():
            ret , frame = cap.read()
            if ret == False:
                raise Exception('the camera not recognized: change camera_index param to ' + str(0 if camera_index == 1 else 1))
            faces, boxes, scores, landmarks = self.detector.detect_align(frame)
            if len(faces.shape) > 1:
                results, score = self.recognizer.infer(self.conf, faces, self.targets, tta=self.tta)
                for idx, bbox in enumerate(boxes):
                    special_draw(frame, bbox, landmarks[idx], self.names[results[idx] + 1], score[idx])
            cv2.imshow('face Capture', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()