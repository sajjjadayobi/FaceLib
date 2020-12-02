"""
Sajjad Ayoubi: Age Gender Detection
I use UTKFace DataSet
from: https://susanqq.github.io/UTKFace/
download it and put it on FaceSet dir
and I create a annotation file data.npy
which there is in weights folder
"""

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from facelib.AgeGender.models.model import ShuffleneFull, TrainModel


class MultitaskDataset(Dataset):

    def __init__(self, data, tfms, root='FaceSet/'):
        self.root = root
        self.tfms = tfms
        self.ages = data[:, 3]
        self.races = data[:, 2]
        self.genders = data[:, 1]
        self.imgs = data[:, 0]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.tfms(Image.open(self.root + self.imgs[i])), torch.tensor(
            [self.genders[i], self.races[i], self.ages[i]]).float()

    def __repr__(self):
        return f'{type(self).__name__} of len {len(self)}'


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sz = 112
bs = 256

tf = {'train': transforms.Compose([
    transforms.RandomRotation(degrees=0.2),
    transforms.RandomHorizontalFlip(p=.5),
    transforms.RandomGrayscale(p=.2),
    transforms.Resize((sz, sz)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)]),
    'test': transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])}

data = np.load('data.npy', allow_pickle=True)
train_data = data[data[:, -1] == 1]
valid_data = data[data[:, -1] == 0]

valid_ds = MultitaskDataset(data=valid_data, tfms=tf['test'])
train_ds = MultitaskDataset(data=train_data, tfms=tf['train'])

train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=8)
valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=True, num_workers=4)


def multitask_loss(input, target):
    input_gender = input[:, :2]
    input_age = input[:, -1]

    loss_gender = F.cross_entropy(input_gender, target[:, 0].long())
    loss_age = F.l1_loss(input_age, target[:, 2])

    return loss_gender / (.16) + loss_age * 2


model = ShuffleneFull().cuda()
optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
history = TrainModel(model, train_dl, valid_dl, optimizer, multitask_loss, scheduler, 5)
print(history)
