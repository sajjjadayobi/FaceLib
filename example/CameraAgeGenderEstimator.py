from facelib import WebcamAgeGenderEstimator

cameraUrl = "http://192.168.2.147:4747/video"
estimator = WebcamAgeGenderEstimator()
estimator.run(cameraUrl)

