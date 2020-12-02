from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import bcolz


def de_preprocess(tensor):
    return tensor * 0.5 + 0.5


def get_train_dataset(imgs_folder):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = ImageFolder(imgs_folder, train_transform)
    class_num = ds[-1][1] + 1
    return ds, class_num


def get_train_loader(conf):
    if conf.data_mode in ['ms1m', 'concat']:
        ms1m_ds, ms1m_class_num = get_train_dataset(conf.ms1m_folder / 'imgs')
        print('ms1m loader generated')
    if conf.data_mode in ['vgg', 'concat']:
        vgg_ds, vgg_class_num = get_train_dataset(conf.vgg_folder / 'imgs')
        print('vgg loader generated')
    if conf.data_mode == 'vgg':
        ds = vgg_ds
        class_num = vgg_class_num
    elif conf.data_mode == 'ms1m':
        ds = ms1m_ds
        class_num = ms1m_class_num
    elif conf.data_mode == 'concat':
        for i, (url, label) in enumerate(vgg_ds.imgs):
            vgg_ds.imgs[i] = (url, label + ms1m_class_num)
        ds = ConcatDataset([ms1m_ds, vgg_ds])
        class_num = vgg_class_num + ms1m_class_num
    elif conf.data_mode == 'emore':
        ds, class_num = get_train_dataset(conf.emore_folder / 'imgs')
    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=True, pin_memory=conf.pin_memory,
                        num_workers=conf.num_workers)
    return loader, class_num


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=path / name, mode='r')
    issame = np.load(path / '{}_list.npy'.format(name))
    return carray, issame


def get_val_data(data_path):
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame


class train_dataset(Dataset):
    def __init__(self, imgs_bcolz, label_bcolz, h_flip=True):
        self.imgs = bcolz.carray(rootdir=imgs_bcolz)
        self.labels = bcolz.carray(rootdir=label_bcolz)
        self.h_flip = h_flip
        self.length = len(self.imgs) - 1
        if h_flip:
            self.transform = trans.Compose([
                trans.ToPILImage(),
                trans.RandomHorizontalFlip(),
                trans.ToTensor(),
                trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        self.class_num = self.labels[-1] + 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = torch.tensor(self.imgs[index + 1], dtype=torch.float)
        label = torch.tensor(self.labels[index + 1], dtype=torch.long)
        if self.h_flip:
            img = de_preprocess(img)
            img = self.transform(img)
        return img, label
