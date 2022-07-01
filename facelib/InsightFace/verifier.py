import cv2
import torch
import argparse
from facelib import get_config, special_draw
from facelib import update_facebank, load_facebank
from facelib import FaceRecognizer
from facelib import FaceDetector
import os
import gc
from multiprocessing import Process, Manager

class WebcamVerify:
    def __init__(self, update=True, tta=True, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),weight=1280,height=720,top=100):   
        print('loading ...') 
        self.tta = tta
        self.detector = FaceDetector(device=device)
        self.conf = get_config()
        self.conf.device = device
        recognizer = FaceRecognizer(self.conf)
        recognizer.model.eval()
        self.recognizer = recognizer

        self.weight = weight
        self.height = height
        self.top = top

        if update:
            self.targets, self.names = update_facebank(self.conf, recognizer.model, self.detector, tta=self.tta)
        else:
            self.targets, self.names = load_facebank(self.conf)
            
    def img_resize(self,img):
        img_new = cv2.resize(img, (self.weight, self.height))
        return img_new  
              
    def write(self,stack, cam) -> None:
        print('Process to write: %s' % os.getpid())
        cap = cv2.VideoCapture(cam)
        while True:
            _, img = cap.read()
            if _:
                stack.append(img)
                if len(stack) >= self.top:
                    del stack[:]
                    gc.collect()

    def read(self,stack) -> None:
        print('Process to read: %s' % os.getpid())
        while True:
            if len(stack) != 0:
                value = stack.pop()
                img_new = self.img_resize(value)
                faces, boxes, scores, landmarks = self.detector.detect_align(img_new)
                if len(faces.shape) > 1:
                    results, score = self.recognizer.infer(faces, self.targets, tta=self.tta)
                    for idx, bbox in enumerate(boxes):
                        special_draw(img_new, bbox, landmarks[idx], self.names[results[idx] + 1], score[idx])
             
                cv2.imshow("img", img_new)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    def run(self, camera_index=0):
        if len(self.targets) < 1:
            raise Exception("you don't have any person in facebank: add new person with 'add_from_webcam' or 'add_from_folder' function")
        q = Manager().list()
        pw = Process(target=self.write, args=(q,camera_index))
        pr = Process(target=self.read, args=(q,))
        pw.start()
        pr.start()
        pr.join()
        pw.terminate()             
  
