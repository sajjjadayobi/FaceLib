from easydict import EasyDict as edict
from pathlib import Path
import os
import torch
from torch.nn import CrossEntropyLoss


def get_config(inference=True):
    conf = edict()
    conf.data_path = Path(os.path.dirname(os.path.realpath(__file__)))
    conf.work_path = Path('weights/')
    conf.model_path = conf.work_path / 'models'
    conf.log_path = conf.work_path / 'log'
    conf.save_path = conf.work_path
    conf.input_size = [112, 112]
    conf.embedding_size = 512
    conf.use_mobilfacenet = True
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.net_mode = 'ir_se'  # or 'ir'
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.data_mode = 'emore'
    conf.vgg_folder = conf.data_path / 'faces_vgg_112x112'
    conf.ms1m_folder = conf.data_path / 'faces_ms1m_112x112'
    conf.emore_folder = conf.data_path / 'faces_emore'
    conf.batch_size = 100  # irse net depth 50
    #   conf.batch_size = 200 # mobilefacenet
    if inference:
    # --------------------Inference Config ------------------------
        conf.facebank_path = conf.data_path / 'facebank'
        conf.threshold = 1.5
        conf.face_limit = 10
        # when inference, at maximum detect 10 faces in one image, my laptop is slow
    else:
    # --------------------Training Config ------------------------
        conf.log_path = conf.work_path / 'log'
        conf.save_path = conf.work_path / 'save'
        #  conf.weight_decay = 5e-4
        conf.lr = 1e-3
        conf.momentum = 0.9
        conf.pin_memory = True
        # conf.num_workers = 4 # when batchsize is 200
        conf.num_workers = 3
        conf.ce_loss = CrossEntropyLoss()
        
    return conf
