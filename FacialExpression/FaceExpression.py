import torch
import numpy as np
from .models.densenet import densenet121
from .models.resnet import resnet34

labels = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])


class EmotionDetector:

    def __init__(self, name='resnet34', device='cpu', weight_path='weights/resnet34.pth'):
        """
        Residual Masking Emotion Detector from a list of labels
        :param name: name of backbone of networks (resnet34, densenet121)
        :param device: model run in cpu or gpu (cuda, cpu)
        :param weight_path: path of network weight

        Notice: image size must be 224x224

        Method detect_emotion:
                :param faces: 4d tensor of face for example size(1, 3, 224, 224)
                :returns emotions list and probability of emotions
        """

        self.device = device
        self.model = None
        if name == 'resnet34':
            self.model = resnet34()
        elif name == 'densnet121':
            self.model = densenet121()
        else:
            exit('EmotionDetector: Network does not support!! \n just(resnet34, densnet121)')

        self.model.load_state_dict(torch.load(weight_path))
        self.model.to(device).eval()


    def detect_emotion(self, faces):
        if len(faces) > 0:
            faces = faces.permute(0, 3, 1, 2)
            faces = faces.float().div(255).to(self.device)
            emotions = self.model(faces)
            prob = torch.softmax(emotions, dim=1)
            emo_prob, emo_idx = torch.max(prob, dim=1)
            return labels[emo_idx.tolist()], emo_prob.tolist()
        else:
            return 0, 0
