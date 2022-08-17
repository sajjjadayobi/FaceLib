from .model import *
import os ,sys
import torch
import numpy as np
from .evaluatation import evaluate
from .utils import *
from PIL import Image
from torchvision import transforms as trans
from tqdm import tqdm
from matplotlib import pyplot as plt

class FaceRecognizer:

    def __init__(self, conf, verbose=True,name='mobilenet'):
        self.device = conf.device
        self.threshold = conf.threshold
        self.weight_path = conf.work_path
        self.useNet = name

        if self.useNet == 'mobilenet':
            self.model = MobileFaceNet(conf.embedding_size).to(conf.device)

            if(self.weight_path == None):
                file_name = os.path.dirname(os.path.realpath(sys.argv[0])) + "\\weights/mobilenet_2.pth"
                self.weight_path = os.path.dirname(os.path.realpath(sys.argv[0])) + "\\weights"
            else:
                file_name = self.weight_path + "\\mobilenet_2.pth"

            #判断模型是否存在
            isExists = os.path.exists(file_name)
            if not isExists:
                url = "https://drive.google.com/uc?export=download&id=15zP8BP-5IvWXWZoYTNdvUJUiBqZ1hxu1"
                print("not found model {},please download to {},rename mobilenet_2".format(file_name,url))
                return
            else:
                print("load model from {}".format(file_name))

            self.model.load_state_dict(torch.load(file_name, map_location=conf.device))


        elif name == 'ir_se50':
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
            if(self.weight_path == None):
                file_name = os.path.dirname(os.path.realpath(sys.argv[0])) + "\\weights/ir_se50.pth"
                self.weight_path = os.path.dirname(os.path.realpath(sys.argv[0])) + "\\weights"
            else:
                file_name = self.weight_path + "\\ir_se50.pth"

            #判断模型是否存在
            isExists = os.path.exists(file_name)
            if not isExists:
                url = "https://www.dropbox.com/s/a570ucg6a5z22zf/ir_se50.pth?dl=1"
                print("not found model {},please download to {}".format(file_name,url))
                return
            else:
                print("load model from {}".format(file_name))

            print(file_name)
            self.model.load_state_dict(torch.load(file_name, map_location=conf.device))
        else:
            print("not found model {}".format(name))


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
        #预处理
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
        #print("minimum:{} , threshold : {}".format(minimum,self.threshold))
        #print(minimum > self.threshold)
        #print("min_idx:{}".format(min_idx))
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
