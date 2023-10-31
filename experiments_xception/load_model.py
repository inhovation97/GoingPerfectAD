import os
import argparse

import timm
import torch
#import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision

def transfer_weights(modelchoice: str, weight_path: str, num_out_classes=2):
    modelchoice = modelchoice
    if modelchoice == 'xception':
        model = timm.create_model('xception', pretrained=False)
    # finetuning for bianry classification
        in_features = model.fc.in_features # 1차원 2048 벡터
        model.last_linear = nn.Linear(in_features, num_out_classes) # 출력층 fine-tuning
        del model.fc # last_linear가 추가된 상태라서, 원래 있던 fc layer제거

    elif modelchoice == 'resnet50':
        model = torchvision.models.resnet50(pretrained=False)
        # Replace fc
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_out_classes)            
    else:
        raise Exception('Choose valid model, e.g. ADD, QAD, cored, xception...')

    if weight_path is not None:
        state_dict = torch.load(weight_path)
        if 'state_dict' in state_dict.keys():
            weights = state_dict['state_dict']
        else:
            weights = state_dict
        model.load_state_dict(weights)
    
    # fc layer 이름 바뀜을 방지
    if modelchoice == 'xception' and model.last_linear is not None:
        model.fc = model.last_linear
        del model.last_linear
        
    return model

def model_selection(model_name: str, weight_path: str, num_out_classes: int):
    """
    :param modelname:
    :return: model, image size, pretraining<yes/no>, input_list
    """
    if model_name == 'xception' or model_name == 'cored':
        return transfer_weights(modelchoice='xception',
                             weight_path = weight_path,
                             num_out_classes=num_out_classes)

    elif 'qad' in model_name.lower() or 'add' in model_name.lower():
        return transfer_weights(modelchoice='resnet50',
                             weight_path = weight_path,
                             num_out_classes=num_out_classes)

    else:
        raise NotImplementedError(model_name)