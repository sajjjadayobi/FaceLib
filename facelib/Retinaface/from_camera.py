import cv2
import torch
from facelib import special_draw
from facelib import FaceDetector
import os
import gc
from multiprocessing import Process, Manager

class WebcamFaceDetector:
    def __init__(self, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),weight=1280,height=720,top=100):
        print('loading ...') 
        self.weight = weight
        self.height = height
        self.top = top
        self.detector = FaceDetector(face_size=(224, 224), device=device)

    def img_resize(self,img):
        """
        :param img: 视频帧
        :return: 处理后的视频帧
        """
        img_new = cv2.resize(img, (self.weight, self.height))
        return img_new

    def write(self,stack, cam) -> None:
        """
        :param cam: 摄像头参数
        :param stack: Manager.list对象
        :return: None
        """
        print('Process to write: %s' % os.getpid())
        cap = cv2.VideoCapture(cam)
        while True:
            _, img = cap.read()
            if _:
                stack.append(img)
                # 每到一定容量清空一次缓冲栈
                # 利用gc库，手动清理内存垃圾，防止内存溢出
                if len(stack) >= self.top:
                    del stack[:]
                    gc.collect()


    def read(self,stack) -> None:
        print('Process to read: %s' % os.getpid())
        while True:
            if len(stack) != 0:
                value = stack.pop()
                # 对获取的视频帧分辨率重处理
                img_new = self.img_resize(value)
                faces, boxes, scores, landmarks = self.detector.detect_align(img_new)
                if len(faces.shape) > 1:
                    for idx, bbox in enumerate(boxes):
                        special_draw(img_new, bbox, landmarks[idx], name='face', score=scores[idx])
                # 显示处理后的视频帧
                cv2.imshow("img", img_new)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break


    def run(self, camera_index=0):
        q = Manager().list()
        pw = Process(target=self.write, args=(q,camera_index))
        pr = Process(target=self.read, args=(q,))
        pw.start()
        pr.start()
        pr.join()
        pw.terminate()        
