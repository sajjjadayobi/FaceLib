# [FaceLib](https://github.com/lissettecarlr/FaceLib)

## 仓库说明
fork的[sajjjadayobi/FaceLib](https://github.com/sajjjadayobi/FaceLib)，对其进行修改
* master分支是lit分支进行过测试，可以直接使用的代码
* lit分支是编码进行时分支，比master新，但是大概率会出现问题
* mul分支是最初FaceLib的功能，人脸检测、年龄性别、表情、人脸识别。不会更新

## 环境
推荐使用conda，通过[官网](https://www.anaconda.com/products/distribution#macos)下载安装即可。然后命令行创建环境
```
conda create -n face python=3.7
activate face
```


## 安装（不同分支自己改）
直接安装:
```
pip install git+https://github.com/lissettecarlr/FaceLib
```
克隆后安装：
```
git clone git@github.com:lissettecarlr/FaceLib.git
cd FaceLib
python setup.py install
```
安装其他软件包的时候报错可以直接通过requirements安装
```
pip install -r requirements.txt
```

## 使用
执行example中的程序
### 人脸检测

* 通过摄像头进行人脸检测：CameraFaceDetector.py
* 通过摄像头，周期保存进行过人脸检测的帧：captureFace.py

### 人脸识别
* 通过摄像头进行人脸识别：CameraFaceRecognition.py，运行前需要导入检测对象，方式在此文件中有说明。
* 通过传入图片，输出名字：FaceRecognition


## 独立使用
如果不想安装facelib包则打开independent文件夹来使用

* 首先需要建立faceData文件夹，存放准备识别对象的图片，里面一个人一个文件夹
* 下载模型存放到weights里面
mobilenet.pth
```
https://drive.google.com/uc?export=download&id=15zP8BP-5IvWXWZoYTNdvUJUiBqZ1hxu1
```
resnet50.pth
```
https://www.dropbox.com/s/8sxkgc9voel6ost/resnet50.pth?dl=1
```
mobilenet_2.pth
```
https://drive.google.com/uc?export=download&id=15zP8BP-5IvWXWZoYTNdvUJUiBqZ1hxu1
```
* 生成识别库
```
python main.py -m update
```
输出
```
load model from D:\code\face\my\FaceLib\independent\weights\resnet50.pth
load model from D:\code\face\my\FaceLib\independent\weights\mobilenet_2.pth
2022-08-17 15:47:26.788 | INFO     | __main__:data_align:22 - data align start
2022-08-17 15:47:26.788 | DEBUG    | __main__:data_align:25 - input_path使用默认路径D:\code\face\my\FaceLib\independent\faceData
2022-08-17 15:47:26.803 | DEBUG    | __main__:data_align:28 - save_path使用默认路径D:\code\face\my\FaceLib\independent\data\facebank
2022-08-17 15:47:26.803 | INFO     | __main__:data_align:33 - start join people feng
2022-08-17 15:47:30.496 | INFO     | __main__:data_align:45 - feng img 1.jpg join success
2022-08-17 15:47:33.968 | INFO     | __main__:data_align:45 - feng img 2.jpg join success
2022-08-17 15:47:37.491 | INFO     | __main__:data_align:45 - feng img 3.jpg join success
2022-08-17 15:47:37.491 | INFO     | __main__:data_align:33 - start join people qiu
2022-08-17 15:47:40.980 | INFO     | __main__:data_align:45 - qiu img 1.jpg join success
2022-08-17 15:47:44.532 | INFO     | __main__:data_align:45 - qiu img 2.jpg join success
2022-08-17 15:47:48.083 | INFO     | __main__:data_align:45 - qiu img 3.jpg join success
2022-08-17 15:47:51.710 | INFO     | __main__:data_align:45 - qiu img 4.jpg join success
2022-08-17 15:47:51.726 | DEBUG    | __main__:update_facebank:55 - 识别库使用默认路径D:\code\face\my\FaceLib\independent\data\facebank
2022-08-17 15:47:51.726 | INFO     | __main__:update_facebank:65 - start update facebank
2022-08-17 15:47:51.726 | INFO     | __main__:update_facebank:70 - start join people feng
2022-08-17 15:47:51.838 | INFO     | __main__:update_facebank:70 - start join people qiu
2022-08-17 15:47:52.042 | INFO     | __main__:update_facebank:96 - update facebank success
```
* 识别，识别完成的图片会报错到out文件夹里
```
python main.py -m find -i ./faceData/2.jpg
```
输出
```
load model from D:\code\face\my\FaceLib\independent\weights\resnet50.pth
load model from D:\code\face\my\FaceLib\independent\weights\mobilenet_2.pth
2022-08-17 16:24:48.635 | INFO     | __main__:load_facebank:107 - load facebank success
2022-08-17 16:24:52.396 | INFO     | __main__:main:158 - name=feng,score=0.42378219962120056
```