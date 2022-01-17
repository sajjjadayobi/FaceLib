import os
import torch
import cv2
import numpy as np
from skimage import transform

from .utils.alignment import get_reference_facial_points, FaceWarpException, alignment
from .utils.box_utils import decode, decode_landmark, prior_box, nms
from .utils.config import cfg_mnet, cfg_re50
from .models.retinaface import RetinaFace
from facelib.utils import download_weight


class FaceDetector:

    def __init__(self, name='mobilenet', weight_path=None, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), confidence_threshold=0.99,
                 top_k=5000, nms_threshold=0.4, keep_top_k=750, face_size=(112, 112), crop_size=(96, 112), verbose=True):
        """
        RetinaFace Detector with 5points landmarks
        Args:
            name: name of backbone (resnet, mobilenet, slim, rfb)
            weight_path: path of network weight
            device: running device (cuda, cpu)
            face_size: final face size
            face_padding: padding for bounding boxes
        """

        model, cfg = None, None
        if name == 'mobilenet':
            cfg = cfg_mnet
            model = RetinaFace(cfg=cfg, phase='test')
            file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mobilenet.pth')
            weight_path = os.path.join(os.path.dirname(file_name), 'weights/mobilenet.pth')
            if os.path.isfile(weight_path) == False:
                os.makedirs(os.path.split(weight_path)[0], exist_ok=True)
                download_weight(link='https://drive.google.com/uc?export=download&id=15zP8BP-5IvWXWZoYTNdvUJUiBqZ1hxu1',
                                file_name=file_name,
                                verbose=verbose)
                os.rename(file_name, weight_path)
        elif name == 'resnet':
            cfg = cfg_re50
            model = RetinaFace(cfg=cfg, phase='test')
            file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resnet50.pth')
            weight_path = os.path.join(os.path.dirname(file_name), 'weights/resnet50.pth')
            if os.path.isfile(weight_path) == False:
                os.makedirs(os.path.split(weight_path)[0], exist_ok=True)
                download_weight(link='https://www.dropbox.com/s/8sxkgc9voel6ost/resnet50.pth?dl=1',
                                file_name=file_name,
                                verbose=verbose)
                os.rename(file_name, weight_path)
        else:
            exit('FaceDetector Exit: model name can be either mobilenet or resnet')

             
        # settings for model
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device).eval()
        self.model = model
        self.device = device
        self.cfg = cfg
        # settings for face detection
        self.thresh = confidence_threshold
        self.top_k = top_k
        self.nms_thresh = nms_threshold
        self.keep_top_k = keep_top_k
        # settings for face alignment
        self.trans = transform.SimilarityTransform()
        self.out_size = face_size
        self.ref_pts = get_reference_facial_points(output_size=face_size, crop_size=crop_size)


    def preprocessor(self, img_raw):
        img = torch.tensor(img_raw, dtype=torch.float32).to(self.device)
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]]).to(self.device)
        img -= torch.tensor([104, 117, 123]).to(self.device)
        img = img.permute(2, 0, 1).unsqueeze(0)
        return img, scale

    def detect_faces(self, img_raw):
        """
        get a image from ndarray, detect faces in image
        Args:
            img_raw: original image from cv2(BGR) or PIL(RGB)
        Notes:
            coordinate is corresponding to original image
            and type of return image is corresponding to input(cv2, PIL)
        Returns:
            boxes:
                faces bounding box for each face
            scores:
                percentage of each face
            landmarks:
                faces landmarks for each face
        """

        img, scale = self.preprocessor(img_raw)
        # tic = time.time()
        with torch.no_grad():
            loc, conf, landmarks = self.model(img)  # forward pass
            # print('net forward time: {:.4f}'.format(time.time() - tic))

        priors = prior_box(self.cfg, image_size=img.shape[2:]).to(self.device)
        boxes = decode(loc.data.squeeze(0), priors, self.cfg['variance'])
        boxes = boxes * scale
        scores = conf.squeeze(0)[:, 1]
        landmarks = decode_landmark(landmarks.squeeze(0), priors, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]]).to(self.device)
        landmarks = landmarks * scale1

        # ignore low scores
        index = torch.where(scores > self.thresh)[0]
        boxes = boxes[index]
        landmarks = landmarks[index]
        scores = scores[index]

        # keep top-K before NMS
        order = scores.argsort(dim=0, descending=True)[:self.top_k]
        boxes = boxes[order]
        landmarks = landmarks[order]
        scores = scores[order]

        # Do NMS
        keep = nms(boxes, scores, self.nms_thresh)
        boxes = torch.abs(boxes[keep, :])
        scores = scores[:, None][keep, :]
        landmarks = landmarks[keep, :].reshape(-1, 5, 2)

        # # keep top-K faster NMS
        landmarks = landmarks[:self.keep_top_k, :]
        scores = scores[:self.keep_top_k, :]
        boxes = boxes[:self.keep_top_k, :]

        return boxes, scores, landmarks
    
    def detect_align(self, img):
        """
        get a image from ndarray, detect faces in image,
        cropped face and align face
        Args:
            img: original image from cv2(BGR) or PIL(RGB)
        Notes:
            coordinate is corresponding to original image
            and type of return image is corresponding to input(cv2, PIL)

        Returns:
            faces:
                a tensor(n, 112, 112, 3) of faces that aligned
        """
        boxes, scores, landmarks = self.detect_faces(img)

        warped = []
        for src_pts in landmarks:
            if max(src_pts.shape) < 3 or min(src_pts.shape) != 2:
                raise FaceWarpException('facial_pts.shape must be (K,2) or (2,K) and K>2')

            if src_pts.shape[0] == 2:
                src_pts = src_pts.T

            if src_pts.shape != self.ref_pts.shape:
                raise FaceWarpException('facial_pts and reference_pts must have the same shape')

            self.trans.estimate(src_pts.cpu().numpy(), self.ref_pts)
            face_img = cv2.warpAffine(img, self.trans.params[0:2, :], self.out_size)
            warped.append(face_img)

        faces = torch.tensor(np.array(warped)).to(self.device)
        return faces, boxes, scores, landmarks
    

    def detect(self, img, align=False):
        boxes, scores, landmarks = self.detect_faces(img)
        box = boxes[0].int()
        face = img[box[1]: box[3], box[0]: box[2]]
        face = cv2.resize(face, self.out_size)
        if align:
            landmark = landmarks[0].int()[:3].numpy()
            return alignment(face, *landmark)
        else:
            return face
