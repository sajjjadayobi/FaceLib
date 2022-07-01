# 通过摄像头进行人脸识别，运行前先导入识别对象
from facelib import WebcamVerify

# 通过写入地址，导入
# #from facelib import add_from_folder
# add_from_folder(folder_path='./feng/', person_name='feng')
# add_from_folder(folder_path='./qiu/', person_name='qiu')

# 通过摄像头建立识别对象
# 打开python输入：
# from facelib import add_from_webcam
# add_from_webcam(person_name='feng', camera_index="http://192.168.2.147:4747/video")
# 按t 拍4站图（点到播放器里面去按）

if __name__ == '__main__':
    verifier = WebcamVerify(update=True)
    verifier.run("http://192.168.2.147:4747/video")