import os
import cv2
import gc
from multiprocessing import Process, Manager
import torch
from facelib import special_draw
from facelib import FaceDetector
from datetime import datetime

def img_resize(img):
    """
    :param img: 视频帧
    :return: 处理后的视频帧
    """
    # 对获取的视频帧分辨率重处理
    img_new = cv2.resize(img, (1280, 720))
    return img_new

# 向共享缓冲栈中写入数据:
def write(stack, cam, top: int) -> None:
    """
    :param cam: 摄像头参数
    :param stack: Manager.list对象
    :param top: 缓冲栈容量
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
            if len(stack) >= top:
                del stack[:]
                gc.collect()


# 在缓冲栈中读取数据:
def read(stack) -> None:
    print('Process to read: %s' % os.getpid())
    lastTime = datetime.now()
    faceImgNum = 0
    detector = FaceDetector(face_size=(224, 224), device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    while True:
        if len(stack) != 0:
            value = stack.pop()
            # 对获取的视频帧分辨率重处理
            img_new = img_resize(value)
            faces, boxes, scores, landmarks = detector.detect_align(img_new)
            if len(faces.shape) > 1:
                for idx, bbox in enumerate(boxes):
                    special_draw(img_new, bbox, landmarks[idx], name='face', score=scores[idx])
            # 显示处理后的视频帧
            if((datetime.now() - lastTime).seconds > 10):
                cv2.imwrite('./captureImg/'+str(faceImgNum)+'.jpg',img_new)
                faceImgNum= faceImgNum+1
                print("save face：" + str(faceImgNum))
                lastTime = datetime.now()

            cv2.imshow("img", img_new)

            # save_img(yolo_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

# rtmp://192.168.2.206/live/test
if __name__ == '__main__':
    # 父进程创建缓冲栈，并传给各个子进程：
    q = Manager().list()
    pw = Process(target=write, args=(q, "http://192.168.2.147:4747/video", 500))
    pr = Process(target=read, args=(q,))
    # 启动子进程pw，写入:
    pw.start()
    # 启动子进程pr，读取:
    pr.start()

    # 等待pr结束:
    pr.join()

    # pw进程里是死循环，无法等待其结束，只能强行终止:
    pw.terminate()