from .model import Backbone, Arcface, MobileFaceNet, l2_norm
from .evaluatation import evaluate
import torch
from torch import optim
import numpy as np
import os
from tqdm import tqdm
from facelib.utils import download_weight
from .utils import get_time, gen_plot, separate_bn_paras
from .utils import faces_preprocessing
from PIL import Image
from torchvision import transforms as trans
import math
from matplotlib import pyplot as plt
plt.switch_backend('agg')


class FaceRecognizer:

    def __init__(self, conf, verbose=True):
        self.device = conf.device
        self.threshold = conf.threshold
        
        try:
            if conf.use_mobilfacenet:
                self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
                # download the default weigth
                file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mobilenet.pth')
                weight_path = os.path.join(os.path.dirname(file_name), 'weights/mobilenet.pth')
                if os.path.isfile(weight_path) == False:
                    os.makedirs(os.path.split(weight_path)[0], exist_ok=True)
                    download_weight(link='https://drive.google.com/uc?export=download&id=1W9nM7LE6zUKQ4tncL6OnBn-aXNyiRPNH',
                                    file_name=file_name, 
                                    verbose=verbose
                                   )
                    os.rename(file_name, weight_path)

                self.model.load_state_dict(torch.load(weight_path, map_location=conf.device))
            else:
                self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
                # download the default weigth
                file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ir_se50.pth')
                weight_path = os.path.join(os.path.dirname(file_name), 'weights/ir_se50.pth')
                if os.path.isfile(weight_path) == False:
                    os.makedirs(os.path.split(weight_path)[0], exist_ok=True)
                    download_weight(link='https://www.dropbox.com/s/a570ucg6a5z22zf/ir_se50.pth?dl=1', 
                                    file_name=file_name,
                                   verbose=verbose
                                   )
                    os.rename(file_name, weight_path)

                self.model.load_state_dict(torch.load(weight_path, map_location=conf.device))
        except IOError as e:
            exit(f'from FaceRecognizer Exit: the weight does not exist,'
                 f' \n download and putting up in "{conf.work_path}" folder \n {e}')
            
        self.model.eval()
        torch.set_grad_enabled(False)


    def evaluate(self, conf, carray, issame, nrof_folds=5, tta=False):
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(carray), conf.embedding_size])
        with torch.no_grad():
            while idx + conf.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + conf.batch_size])
                if tta:
                    fliped = batch.flip(-1)  # I do not test in this case
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + conf.batch_size] = self.model(batch.to(conf.device)).cpu()
                idx += conf.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                if tta:
                    fliped = batch.flip(-1)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.to(conf.device)).cpu()
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor


    def train(self, conf, epochs):
        self.model.train()
        running_loss = 0.
        for e in range(epochs):
            print('epoch {} started'.format(e))
            if e == self.milestones[0]:
                self.schedule_lr()
            if e == self.milestones[1]:
                self.schedule_lr()
            if e == self.milestones[2]:
                self.schedule_lr()
            for imgs, labels in tqdm(iter(self.loader)):
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)
                self.optimizer.zero_grad()
                embeddings = self.model(imgs)
                thetas = self.head(embeddings, labels)
                loss = conf.ce_loss(thetas, labels)
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    running_loss = 0.

                if self.step % self.evaluate_every == 0 and self.step != 0:
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.agedb_30,
                                                                               self.agedb_30_issame)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.lfw, self.lfw_issame)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.cfp_fp, self.cfp_fp_issame)
                    self.model.train()
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)

                self.step += 1

        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def schedule_lr(self):
        for params in self.optimizer.param_groups:
            params['lr'] /= 10
        print(self.optimizer)

    def infer(self, faces, target_embs, tta=False):
        """
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        """
        faces = faces_preprocessing(faces, self.device)
        if tta:
            faces_emb = self.model(faces)
            hflip_emb = self.model(faces.flip(-1))  # image horizontal flip
            embs = l2_norm((faces_emb + hflip_emb)/2)  # take mean
        else:
            embs = self.model(faces)

        diff = embs.unsqueeze(-1) - target_embs.transpose(1, 0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1  # if no match, set idx to -1
        return min_idx, minimum
    
    def feature_extractor(self, faces, tta=False):
        """
        faces : stack of torch.tensors
        tta : test time augmentation (hfilp, that's all)
        """
        faces = faces_preprocessing(faces, self.device)
        if tta:
            faces_emb = self.model(faces)
            hflip_emb = self.model(faces.flip(-1))  # horizontal flip
            embs = l2_norm((faces_emb + hflip_emb)/2)  # take mean
        else:
            embs = self.model(faces) 
        return embs
