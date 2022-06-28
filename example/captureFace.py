# 进行人脸检测，并且将其保存
# 通过调用平板摄像头，进行人脸检测，每隔10秒钟对画面中的头像进行一次截图保存，存放在captureImg文件夹中

import cv2
import torch
from facelib import special_draw
from facelib import FaceDetector
from datetime import datetime

#摄像头
cameraUrl = "http://192.168.2.147:4747/video"
cap = cv2.VideoCapture(cameraUrl)
cap.set(3, 1280)
cap.set(4, 720)

# 人脸检测
detector = FaceDetector(face_size=(224, 224), device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

lastTime = datetime.now()
faceImgNum = 0

while cap.isOpened():
    # 读取画面
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # 检测人脸
    faces, boxes, scores, landmarks = detector.detect_align(frame)
    face = detector.detect_align(frame)[0].cpu().numpy()
    facePos = 0
    if len(faces.shape) > 1:
     for idx, bbox in enumerate(boxes):
        special_draw(frame, bbox, landmarks[idx], name='face', score=scores[idx])
        # 每一轮头像保存间隔10秒
        if((datetime.now() - lastTime).seconds > 10):
            cv2.imwrite('./captureImg/'+str(faceImgNum)+'.jpg',face[facePos])
            facePos = facePos + 1
            faceImgNum= faceImgNum+1
            print("save face：" + str(faceImgNum))
            if(facePos >= faces.shape[0]):
                lastTime = datetime.now()

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
