from datetime import datetime
import numpy as np
import io, cv2, os
from .model import l2_norm
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def faces_preprocessing(faces, device):
    faces = faces.permute(0, 3, 1, 2).float()
    faces = faces.div(255).to(device)
    mu = torch.as_tensor([.5, .5, .5], dtype=faces.dtype, device=device)
    faces[:].sub_(mu[:, None, None]).div_(mu[:, None, None])
    return faces


def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn


def update_facebank(conf, model, detector, tta=True):
    if os.path.exists(conf.facebank_path) == False:
        raise Exception("you don't have facebank yet: create with add_from_webcam or add_from_folder function")
    model.eval()
    faces_embs = torch.empty(0).to(conf.device)
    names = np.array(['Unknown'])
    for path in conf.facebank_path.iterdir():
        if path.is_file():
            continue
        faces = []
        for img_path in path.iterdir():
            face = cv2.imread(str(img_path))
            if face.shape[:2] != (112, 112):  # if img be not face
                face = detector.detect_align(face)[0]
                cv2.imwrite(img_path, face)
            else:
                face = torch.tensor(face).unsqueeze(0)
            faces.append(face)

        faces = torch.cat(faces)
        if len(faces.shape) <= 3:
            continue

        with torch.no_grad():
            faces = faces_preprocessing(faces, device=conf.device)
            if tta:
                face_emb = model(faces)
                hflip_emb = model(faces.flip(-1))  # image horizontal flip
                face_embs = l2_norm(face_emb + hflip_emb)
            else:
                face_embs = model(faces)

        faces_embs = torch.cat((faces_embs, face_embs.mean(0, keepdim=True)))
        names = np.append(names, path.name)

    torch.save(faces_embs, conf.facebank_path/'facebank.pth')
    np.save(conf.facebank_path/'names', names)
    print('from FaceRecognizer: facebank updated')
    return faces_embs, names


def load_facebank(conf):
    if os.path.exists(conf.facebank_path) == False:
        raise Exception("you don't have facebank yet: create with add_from_webcam or add_from_folder function")
    
    embs = torch.load(conf.facebank_path/'facebank.pth')
    names = np.load(conf.facebank_path/'names.npy')
    print('from FaceRecognizer: facebank loaded')
    return embs, names


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf


def draw(bbox, name, frame):
    frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)
    frame = cv2.putText(frame, name, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
    return frame


def special_draw(img, box, landmarsk, name, score=100):
    """draw a bounding box on image"""
    color = (148, 133, 0)
    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    c1, c2 = (box[0], box[1]), (box[2], box[3])
    # draw bounding box
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    # draw landmark
    for land in landmarsk:
        cv2.circle(img, tuple(land.int().tolist()), 3, color, -1)
    # draw score
    score = 100-(score*100/1.4)
    score = 0 if score < 0 else score
    bar = (box[3] + 2) - (box[1] - 2)
    score_final = bar - (score*bar/100)
    cv2.rectangle(img, (box[2] + 1, box[1] - 2 + score_final), (box[2] + (tl+5), box[3] + 2), color, -1)
    # draw label
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(name, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(img, c1, c2, color, -1)  # filled
    cv2.putText(img, name, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
