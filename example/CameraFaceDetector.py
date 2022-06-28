from facelib import WebcamFaceDetector

cameraUrl = "http://192.168.2.147:4747/video"
detector = WebcamFaceDetector()
detector.run(cameraUrl)