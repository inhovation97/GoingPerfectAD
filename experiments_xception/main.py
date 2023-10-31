import numpy as np
import os
import random
import wandb

import torch
import argparse
import logging
import yaml
import timm
from timm import create_model
from modified_xception import load_model


from train import fit
from data import create_dataloader, make_datasets
from log import setup_default_logging

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
    savedir = os.path.join(cfg['RESULT']['savedir'], cfg['DATASET']['dataname']) 
    os.makedirs(savedir, exist_ok=True)

    setup_default_logging(log_path=os.path.join(savedir,'log.txt'))
    torch_seed(cfg['SEED'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    # build Model
    if cfg['MODIFYING'] == True:
        model = load_model(num_classes=2)
        model.to(device)

    else:
        model = create_model(cfg['MODEL'], num_classes=2, pretrained=True)
        model.to(device)
    _logger.info('# of params: {}'.format(np.sum([p.numel() for p in model.parameters()])))

    trainset, validset = make_datasets(
        data_path=cfg['DATASET']['datadir'],
        data_name=cfg['DATASET']['dataname']
        )

    # load dataloader
    trainloader = create_dataloader(dataset=trainset, batch_size=cfg['TRAINING']['batch_size'], shuffle=True)
    validloader = create_dataloader(dataset=validset, batch_size=cfg['TRAINING']['test_batch_size'], shuffle=False)

    # set training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params       = model.parameters(), 
        lr           = cfg['OPTIMIZER']['lr'], 
    )


    # scheduler
    if cfg['TRAINING']['use_scheduler']:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) # 그냥 간단하게
    else:
        scheduler = None

    # initialize wandb
    wandb.init(name=cfg['EXP_NAME'], 
               project='DF_detection', 
               config=cfg)

    # fitting model
    fit(model        = model, 
        trainloader  = trainloader, 
        validloader  = validloader, 
        criterion    = criterion, 
        optimizer    = optimizer, 
        scheduler    = scheduler,
        epochs       = cfg['TRAINING']['epochs'], 
        savedir      = savedir,
        data_name    = cfg['DATASET']['dataname'],
        log_interval = cfg['TRAINING']['log_interval'],
        modifying    = cfg['MODIFYING'],
        device       = device)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='DF_detection')
    parser.add_argument('--default_setting', type=str, default=None, help='exp config file')    
    parser.add_argument('--modelname', type=str, default='xception', help='model name')
    parser.add_argument('--modifying', type=str, default='False', help='modifying')
    parser.add_argument('--dataname', type=str, default=None, help='data name')

    args = parser.parse_args()

    # config -> default setting (batch size, lr, optimizer, etc..)
    cfg = yaml.load(open(args.default_setting,'r'), Loader=yaml.FullLoader)
    
    # cfg가 이중 딕셔너리고, 첫 딕셔너리의 키는 대문자로 쓰는 듯.
    cfg['MODEL'] = args.modelname # cfg에 없는 모델 키 추가
    cfg['MODIFYING'] = args.modifying # cfg에 없는 모델 키 추가
    cfg['DATASET']['dataname'] = args.dataname # cfg에 없는 데이터셋 추가
    cfg['EXP_NAME'] = f"Xception-{args.dataname}" # 실험이름
    run(cfg)