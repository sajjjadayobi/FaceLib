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