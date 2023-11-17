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
from train import Train_OC
from test import Test_OC


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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    # build Model
    # if cfg['MODIFYING'] == True:
        # path = '/home/inho/df_detection/experiments_OC/model/xception-43020ad28.pth'
    if cfg['MODEL']['pretrain_path'] is not None:
        path = cfg['MODEL']['pretrain_path']
        weight_name = path.split('/')[-1].split('.')[0]
        oc_model = load_model(OC=True, transfer_weight_path=path)
        oc_model.to(device)

    else:
        path = '/home/inho/df_detection/experiments_OC/model/xception-43020ad28.pth'
        oc_model = load_model(OC=True, transfer_weight_path=path)
        oc_model.to(device)
    _logger.info('# of params: {}'.format(np.sum([p.numel() for p in oc_model.parameters()])))

################ training ##################
    if cfg['DATASET']['mode'] != 'test':
        # load dataloader
        trainset = make_datasets(
            data_path=cfg['DATASET']['datadir'],
            data_name=cfg['DATASET']['dataname'],
            mode=cfg['DATASET']['mode']
            )
        trainloader = create_dataloader(dataset=trainset, batch_size=cfg['TRAINING']['batch_size'], shuffle=True)
        
        # set training
        # criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            params       = oc_model.parameters(), 
            lr           = cfg['OPTIMIZER']['lr'], 
        )

        # scheduler
        if cfg['TRAINING']['use_scheduler']:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['TRAINING']['epochs'])
        else:
            scheduler = None

    # fitting model
        OC = Train_OC(
            log_interval = cfg['TRAINING']['log_interval'],
            device       = device,
            epochs       = cfg['TRAINING']['epochs'], 
            savedir      = savedir, 
            data_name    = cfg['DATASET']['dataname'], 
            scheduler    = scheduler, 
            dataloader  = trainloader, 
            optimizer    = optimizer, )
    
        OC.train(model = oc_model, pretrainC_path = cfg['MODEL']['pretrain_path'])

################ testing ##################
    else:
        testset = make_datasets(
            data_path=cfg['DATASET']['datadir'],
            data_name=cfg['DATASET']['dataname'],
            mode=cfg['DATASET']['mode']
            )
        testloader = create_dataloader(dataset=testset, batch_size=cfg['TRAINING']['test_batch_size'], shuffle=False)

        OC = Test_OC(
            device       = device,
            savedir      = savedir, 
            data_name    = cfg['DATASET']['dataname'],
            dataloader = testloader)

        score_lst, label_lst = OC.test(model = oc_model, pretrainC_path = cfg['MODEL']['pretrain_path'])
        OC.ploting(score_lst, label_lst, weight_name)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='DF_detection')
    parser.add_argument('--default_setting', type=str, default=None, help='exp config file')    
    parser.add_argument('--modelname', type=str, default='xception', help='model name')
    parser.add_argument('--modifying', type=str, default='False', help='modifying')
    parser.add_argument('--dataname', type=str, default=None, help='data name') # 사실상 결과 저장 디렉토리 이름
    # parser.add_argument('--pretrain_path', type=bool, default=None, help='pretrain_path') # 모델 가중치와 OC center point 저장 경로
    # parser.add_argument('--mode', type=bool, default=None, help='mode') # if mode == df_real: 딥페이크 OC하기


    args = parser.parse_args()

    # config -> default setting (batch size, lr, optimizer, etc..)
    cfg = yaml.load(open(args.default_setting,'r'), Loader=yaml.FullLoader)
    
    # cfg가 이중 딕셔너리고, 첫 딕셔너리의 키는 대문자로 쓰는 듯.
    # cfg['MODEL'] = args.modelname # cfg에 없는 모델 키 추가
    # cfg['MODEL']['pretrain_path'] = args.pretrain_path
    cfg['MODIFYING'] = args.modifying # cfg에 없는 모델 키 추가
    cfg['DATASET']['dataname'] = args.dataname # cfg에 없는 데이터셋 추가
    cfg['EXP_NAME'] = f"Xception-{args.dataname}" # 실험이름
    run(cfg)