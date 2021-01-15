import torch
import os
import numpy as np
from facelib.utils import download_weight
from .models.densenet import densenet121
from .models.resnet import resnet34

labels = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])

class EmotionDetector:

    def __init__(self, name='densnet121', weight_path=None, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
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
            exit('from EmotionDetector: Network does not support!! \n just(resnet34, densnet121)')

        # download the default weigth
        if weight_path is None:
            file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'densnet121.pth')
            weight_path = os.path.join(os.path.dirname(file_name), 'weights/densnet121.pth')
            if os.path.isfile(weight_path) == False:
                print('from EmotionDetector: download defualt weight started')
                os.makedirs(os.path.split(weight_path)[0], exist_ok=True)
                download_weight(link='https://drive.google.com/uc?export=download&id=1G3VsfgiQb16VyFnOwEVDgm2g8-9qN0-9', file_name=file_name)
                os.rename(file_name, weight_path)

        self.model.load_state_dict(torch.load(weight_path, map_location=device))
        self.model.to(device).eval()
        print('from EmotionDetector: weights loaded')


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
