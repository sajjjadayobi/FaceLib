import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import time
import sys

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sz = 112


class ShuffleneTiny(nn.Module):

    def __init__(self):
        super(ShuffleneTiny, self).__init__()
        self.model = models.shufflenet_v2_x0_5()
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 3))

    def forward(self, x):
        return self.model(x)


class ShuffleneFull(nn.Module):

    def __init__(self):
        super(ShuffleneFull, self).__init__()
        self.model = models.shufflenet_v2_x1_0()
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 3))

    def forward(self, x):
        return self.model(x)


class TrainModel:

    def __init__(self, model, train_dl, valid_dl, optimizer, certrion, scheduler, num_epochs):

        self.num_epochs = num_epochs
        self.model = model
        self.scheduler = scheduler
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.optimizer = optimizer
        self.certrion = certrion

        self.loss_history = []
        self.best_acc_valid = 0.0
        self.best_wieght = None

        self.training()

    def training(self):

        valid_acc = 0
        for epoch in range(self.num_epochs):

            print('Epoch %2d/%2d' % (epoch + 1, self.num_epochs))
            print('-' * 15)

            t0 = time.time()
            train_acc = self.train_model()
            valid_acc = self.valid_model()
            if self.scheduler:
                self.scheduler.step()

            time_elapsed = time.time() - t0
            print('  Training complete in: %.0fm %.0fs' % (time_elapsed // 60, time_elapsed % 60))
            print('| val_acc_gender | val_l1_loss | acc_gender | l1_loss |')
            print('| %.3f | %.3f | %.3f | %.3f   \n' % (valid_acc[0], valid_acc[1], train_acc[0], train_acc[1]))

            if valid_acc[0] > self.best_acc_valid:
                self.best_acc_valid = valid_acc[1]
                self.best_wieght = self.model.state_dict().copy()
        return

    def train_model(self):

        self.model.train()
        N = len(self.train_dl.dataset)
        step = N // self.train_dl.batch_size

        avg_loss = 0.0
        acc_gender = 0.0
        loss_age = 0.0

        for i, (x, y) in enumerate(self.train_dl):
            x, y = x.cuda(), y.cuda()
            # forward
            pred_8 = self.model(x)
            # loss
            loss = self.certrion(pred_8, y)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # statistics of model training
            avg_loss = (avg_loss * i + loss) / (i + 1)
            acc_gender += accuracy_gender(pred_8, y)
            loss_age += l1loss_age(pred_8, y)

            self.loss_history.append(avg_loss)

            # report statistics
            sys.stdout.flush()
            sys.stdout.write("\r  Train_Step: %d/%d | runing_loss: %.4f" % (i + 1, step, avg_loss))

        sys.stdout.flush()
        return torch.tensor([acc_gender, loss_age]) / N

    def valid_model(self):
        print()
        self.model.eval()
        N = len(self.valid_dl.dataset)
        step = N // self.valid_dl.batch_size
        acc_gender = 0.0
        loss_age = 0.0

        with torch.no_grad():
            for i, (x, y) in enumerate(self.valid_dl):
                x, y = x.cuda(), y.cuda()

                score = self.model(x)
                acc_gender += accuracy_gender(score, y)
                loss_age += l1loss_age(score, y)

                sys.stdout.flush()
                sys.stdout.write("\r  Vaild_Step: %d/%d" % (i, step))

        sys.stdout.flush()
        return torch.tensor([acc_gender, loss_age]) / N


def accuracy_gender(input, targs):
    pred = torch.argmax(input[:, :2], dim=1)
    y = targs[:, 0]
    return torch.sum(pred == y)


def l1loss_age(input, targs):
    return F.l1_loss(input[:, -1], targs[:, -1]).mean()
