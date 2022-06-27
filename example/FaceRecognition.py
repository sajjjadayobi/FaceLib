import cv2
from facelib import FaceRecognizer, FaceDetector
from facelib import update_facebank, load_facebank, special_draw, get_config

conf = get_config()
# conf.use_mobilenet=False # if you want to use the bigger model
detector = FaceDetector(device=conf.device)
face_rec = FaceRecognizer(conf)

# set True when you add someone new to the facebank
update_facebank_for_add_new_person = False
if update_facebank_for_add_new_person:
    targets, names = update_facebank(conf, face_rec.model, detector)
else:
    targets, names = load_facebank(conf)

image = cv2.imread("2.jpg")
# cv2.imshow("showing",image)
# k = cv2.waitKey(10)

faces, boxes, scores, landmarks = detector.detect_align(image)
results, score = face_rec.infer(faces, targets)
for idx, bbox in enumerate(boxes):
    special_draw(image, bbox, landmarks[idx], names[results[idx]+1], score[idx])
    print(names[results[idx]+1])

cv2.imwrite("./new.jpg",image)
#cv2.imshow("showing",image)
#k = cv2.waitKey(5000)