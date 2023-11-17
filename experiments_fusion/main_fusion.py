import numpy as np
import os
import random
import wandb

import torch
import argparse
import logging
import yaml
from model import load_model

from data import create_dataloader, make_datasets
from train_fusion import fit
from log import setup_default_logging
from utils import ConsistencyCos

import torch.nn as nn

_logger = logging.getLogger('train')

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

def run(cfg):
    # make save directory
    savedir = os.path.join(cfg['RESULT']['savedir'], cfg['EXP_NAME'])
    os.makedirs(savedir, exist_ok=True)

    setup_default_logging(log_path=os.path.join(savedir,'log.txt'))
    torch_seed(cfg['SEED'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    # build Model
    # if cfg['MODIFYING'] == True:
    #     model = load_model(num_classes=2)
    #     model.to(device)

    if cfg['MODEL']['model_name']=='vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        model.to(device)

    else:
        model1 = load_model(num_classes=2, pre_logits=True, transfer_weight_path=cfg['MODEL']['pretrain_path'])
        if cfg['MODEL']['pretrain_path2']:
            model2 = load_model(num_classes=2, pre_logits=True, transfer_weight_path=cfg['MODEL']['pretrain_path2'])
        else:
            model2 = load_model(num_classes=2, pre_logits=True, transfer_weight_path=cfg['MODEL']['pretrain_path'])
        model1.to(device)
        model2.to(device)
        
    _logger.info('# of params: {}'.format(np.sum([p.numel() for p in model2.parameters()])))

    trainset, validset = make_datasets(
        data_path=cfg['DATASET']['datadir'],
        data_name=cfg['DATASET']['dataname'],
        mode = cfg['DATASET']['mode'],
        model_name=cfg['MODEL']['model_name']
        )

    # load dataloader
    trainloader = create_dataloader(dataset=trainset, batch_size=cfg['TRAINING']['batch_size'], shuffle=True)
    validloader = create_dataloader(dataset=validset, batch_size=cfg['TRAINING']['test_batch_size'], shuffle=False)

    # set training
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = ConsistencyCos()
    optimizer = torch.optim.Adam(
        params       = model2.parameters(), 
        lr           = cfg['OPTIMIZER']['lr'], 
    )


    # scheduler
    if cfg['TRAINING']['use_scheduler']:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['TRAINING']['epochs'])
    else:
        scheduler = None

    # initialize wandb
    wandb.init(name=cfg['EXP_NAME'], 
               project='DF_detection', 
               config=cfg)

    # fitting model
    fit(model1        = model1, 
        model2        = model2, 
        trainloader  = trainloader, 
        validloader  = validloader, 
        criterion1    = criterion1, 
        criterion2    = criterion2, 
        consistency_rate = cfg['TRAINING']['consistency_rate'], 
        optimizer    = optimizer, 
        scheduler    = scheduler, 
        epochs       = cfg['TRAINING']['epochs'], 
        savedir      = savedir, 
        log_interval = cfg['TRAINING']['log_interval'], 
        device       = device, 
        modifying    = cfg['MODIFYING'], 
        amp         = cfg['TRAINING']['amp'])

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='DF_detection')
    parser.add_argument('--default_setting', type=str, default=None, help='exp config file')
    parser.add_argument('--model_name', type=str, default='xception', help='model name')
    parser.add_argument('--modifying', type=str, default='False', help='modifying')
    parser.add_argument('--dataname', type=str, default=None, help='data name')
    parser.add_argument('--mode', type=str, default='train', help='mode')
    parser.add_argument('--pre_logits', type=str, default='False', help='pre_logits')
    parser.add_argument('--amp', type=str, default='True', help='amp')
    parser.add_argument('--consistency_rate', type=float, default=1.0, help='consistency_rate')
    parser.add_argument('--exp_name', type=str, default=None, help='exp_name')


    args = parser.parse_args()

    # config -> default setting (batch size, lr, optimizer, etc..)
    cfg = yaml.load(open(args.default_setting,'r'), Loader=yaml.FullLoader)
    
    # cfg가 이중 딕셔너리고, 첫 딕셔너리의 키는 대문자로 쓰는 듯.
    cfg['MODEL']['model_name'] = args.model_name # cfg에 없는 모델 키 추가
    cfg['MODEL']['pre_logits'] = args.pre_logits
    cfg['MODIFYING'] = args.modifying # cfg에 없는 모델 키 추가
    cfg['DATASET']['dataname'] = args.dataname # cfg에 없는 데이터셋 추가
    cfg['DATASET']['mode'] = args.mode
    cfg['TRAINING']['amp'] = args.amp
    cfg['TRAINING']['consistency_rate'] = args.consistency_rate
    if args.exp_name:
        cfg['EXP_NAME'] = args.exp_name
    else:
        cfg['EXP_NAME'] = f"{args.model_name}_OC-{args.dataname}" # 실험이름
    run(cfg)