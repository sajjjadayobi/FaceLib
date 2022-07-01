from .Retinaface.Retinaface import FaceDetector

from .InsightFace.models.Learner import FaceRecognizer
from .InsightFace.models.utils import update_facebank, load_facebank, special_draw
from .InsightFace.models.data.config import get_config
from .InsightFace.add_face import add_from_webcam, add_from_folder
# webcame classes
from .InsightFace.verifier import WebcamVerify
from .Retinaface.from_camera import WebcamFaceDetector

__all__ = ['FaceDetector', 'FaceRecognizer', 'WebcamVerify', 'WebcamFaceDetector',
           'get_config', 'update_facebank', 'load_facebank', 'special_draw', 'add_from_webcam', 'add_from_folder', ]

