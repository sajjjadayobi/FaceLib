from InsightFace.data.config import get_config
from InsightFace.models.Learner import face_learner
from InsightFace.utils import update_facebank, load_facebank, special_draw
from Retinaface.Retinaface import FaceDetector


conf = get_config(training=False)
detector = FaceDetector(name='mobilenet', weight_path='mobilenet.pth', device=conf.device)
conf.use_mobilfacenet = True or False
face_rec = face_learner(conf, inference=True)
face_rec.model.eval()

if update_facebank_for_add_new_person:
    targets, names = update_facebank(conf, face_rec.model, detector)
else:
    targets, names = load_facebank(conf)

faces, boxes, scores, landmarks = detector.detect_align(image)
results, score = face_rec.infer(conf, faces, targets)
for idx, bbox in enumerate(boxes):
    special_draw(image, bbox, landmarks[idx], names[results[idx] + 1], score[idx])