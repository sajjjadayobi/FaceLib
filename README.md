# [FaceLib](https://github.com/lissettecarlr/FaceLib)
fork的[sajjjadayobi/FaceLib](https://github.com/sajjjadayobi/FaceLib)，对其进行修改

## 环境
* 推荐使用conda
* python 3.7
* 然后使用如下命令安装
```
pip install git+https://github.com/lissettecarlr/FaceLib
```

## 使用
进入example文件夹中执行
### 年龄和性别

* 对图片进行识别
```
python AgeGenderEstimator.py

```
* 打开摄像头进行识别
```
python CameraAgeGenderEstimator.py
```

### 表情

* 对图片进行识别
```
python EmotionDetector.py

```
* 打开摄像头进行识别
```
python CameraEmotionDetector.py
```

### 人脸识别

* 打开摄像头进行识别
首先添加需要判断的人，对他啪啪啪照4张图。
然后在example中建立文件夹扔进去

修改CameraFaceRecognition.py，文件夹地址和名字
```
add_from_folder(folder_path='./feng/', person_name='feng')
```
直接执行即可
```
python CameraFaceRecognition.py
```
我这儿用的是windows，所以实际被导入到了C:\ProgramData\Anaconda3\envs\face2\Lib\site-packages\facelib\InsightFace\models\data\facebank
注意的是导入只需要一次，第二次运行可以删除add_from_folder，否则会报错。

* 图片识别
新角色入库还是和add_from_folder(folder_path='./feng/', person_name='feng')，以下是库中有这个人的前提下的步骤

首先拍一张这个人的新照片，放到example文件夹中，然后打开FaceRecognition.py，修改图片名称
```
image = cv2.imread("2.jpg")
```
然后执行
```
python CameraFaceRecognition.py
```
在example中会生成识别后的图像