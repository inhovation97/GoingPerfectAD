import os
import json
import torch
from collections import OrderedDict
import torch.nn.functional as F

from barbar import Bar
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Test_OC:
    def __init__(self,
                 device: str, 
                 savedir: str, 
                 data_name: str,
                 dataloader):
        self.device = device
        self.savedir = savedir
        self.data_name = data_name
        self.dataloader = dataloader

    def test(self, model, pretrainC_path):
        """Training the Deep SVDD model"""
        model = model.to(self.device)
        if pretrainC_path is not None:
            state_dict = torch.load(pretrainC_path)
            c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            c = torch.randn(2048).to(self.device)

        # ROC AUC score 계산 
        score_lst = []
        label_lst = []

        model.eval()
        for x, y in Bar(self.dataloader):
            x = x.float().to(self.device)
            z = model(x)
            score = torch.mean(torch.sum((z - c) ** 2, dim=1))
            converted_values = ['Real' if value == 0 else 'Fake' for value in [y[0].cpu().tolist()]]
            score_lst.append(float(score.detach().cpu()))
            label_lst.append(converted_values)
            
        return score_lst, label_lst
        
    def ploting(self, score_lst, label_lst, weight_name):
        result_path = os.path.join(self.savedir, self.data_name, 'modifided')
        data_df = pd.concat([pd.DataFrame(score_lst), pd.DataFrame(label_lst)], axis=1)
        data_df.columns = ['data', 'label']
        os.makedirs(result_path, exist_ok=True)
        palette_colors = {"Real": "steelblue", "Fake": "firebrick"}
        ax = sns.kdeplot(data = data_df, x='data', hue='label', fill=True, palette=palette_colors)
        # ax.set(xlim=(0,0.005))
        plt.savefig(os.path.join(result_path, f'{weight_name}_test_kdeplot.png'))