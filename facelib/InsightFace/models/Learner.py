from .data.data_pipe import get_train_loader, get_val_data
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

    def __init__(self, conf, inference=True):

        if conf.use_mobilfacenet:
            self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
        else:
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)

        if not inference:
            self.milestones = conf.milestones
            self.loader, self.class_num = get_train_loader(conf)
            self.step = 0
            self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)
            print('two model heads generated')

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)

            if conf.use_mobilfacenet:
                self.optimizer = optim.SGD([
                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                    {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                    {'params': paras_only_bn}
                ], lr=conf.lr, momentum=conf.momentum)
            else:
                self.optimizer = optim.SGD([
                    {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                    {'params': paras_only_bn}
                ], lr=conf.lr, momentum=conf.momentum)
            print(self.optimizer)
            print('optimizers generated')
            self.board_loss_every = len(self.loader) // 100
            self.evaluate_every = len(self.loader) // 10
            self.save_every = len(self.loader) // 5
            self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(self.loader.dataset.root.parent)
        else:
            self.threshold = conf.threshold
            try:
                if conf.use_mobilfacenet:
                    # download the default weigth
                    file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mobilenet.pth')
                    weight_path = os.path.join(os.path.dirname(file_name), 'weights/mobilenet.pth')
                    if os.path.isfile(weight_path) == False:
                        print('from FaceRecognizer: download defualt weight started')
                        os.makedirs(os.path.split(weight_path)[0], exist_ok=True)
                        download_weight(link='https://drive.google.com/uc?export=download&id=1W9nM7LE6zUKQ4tncL6OnBn-aXNyiRPNH', file_name=file_name)
                        os.rename(file_name, weight_path)
                    
                    self.model.load_state_dict(torch.load(weight_path, map_location=conf.device))
                    print('from FaceRecognizer: MobileFaceNet Loaded')
                else:
                    self.model.load_state_dict(torch.load(f'{conf.work_path}/ir_se50.pth'))
                    print('from FaceRecognizer: IR_SE_50 Loaded')
            except IOError as e:
                exit(f'from FaceRecognizer Exit: the weight does not exist,'
                     f' \n download and putting up in "{conf.work_path}" folder \n {e}')


    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        torch.save(
            self.model.state_dict(), save_path('model_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
        if not model_only:
            torch.save(
                self.head.state_dict(), save_path /
                                        ('head_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
            torch.save(
                self.optimizer.state_dict(), save_path /
                                             ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy,self.step, extra)))


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

    def find_lr(self, conf, init_value=1e-8, final_value=10., beta=0.98, bloding_scale=3., num=None):
        if not num:
            num = len(self.loader)
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        self.model.train()
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for i, (imgs, labels) in tqdm(enumerate(self.loader), total=num):

            imgs = imgs.to(conf.device)
            labels = labels.to(conf.device)
            batch_num += 1

            self.optimizer.zero_grad()

            embeddings = self.model(imgs)
            thetas = self.head(embeddings, labels)
            loss = conf.ce_loss(thetas, labels)

            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta ** batch_num)
            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
                print('exited with best_loss at {}'.format(best_loss))
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses
            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            # Do the SGD step
            # Update the lr for the next step

            loss.backward()
            self.optimizer.step()

            lr *= mult
            for params in self.optimizer.param_groups:
                params['lr'] = lr
            if batch_num > num:
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses


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

    def infer(self, conf, faces, target_embs, tta=False):
        """
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        """
        faces = faces_preprocessing(faces, conf.device)
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
