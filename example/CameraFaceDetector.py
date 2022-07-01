from facelib import WebcamFaceDetector

if __name__ == '__main__':
    cameraUrl = "http://192.168.2.147:4747/video"
    detector = WebcamFaceDetector(weight=800,height=600,top=300)
    detector.run(cameraUrl)