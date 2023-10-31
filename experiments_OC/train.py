import logging
import wandb
import time
import os
import json
import torch
from collections import OrderedDict
import torch.nn.functional as F

from barbar import Bar
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Train_OC:
    def __init__(self,
                 log_interval: int, 
                 device: str, 
                 epochs: int, 
                 savedir: str, 
                 data_name: str,
                 scheduler,
                 dataloader, 
                 optimizer):
        self.log_interval = log_interval
        self.device = device
        self.epochs = epochs
        self.savedir = savedir
        self.data_name = data_name
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.optimizer = optimizer

    def set_c(self,model, eps=0.1):
        """Initializing the center for the hypersphere"""
        model.eval()
        z_ = []
        with torch.no_grad():
            for x, _ in self.dataloader:
                x = x.float().to(self.device)
                z = model.encode(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c


    def train(self, model, pretrainC_path):
        """Training the Deep SVDD model"""
        model = model.to(self.device)
        if pretrainC_path is not None:
            state_dict = torch.load(pretrainC_path)
            c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            c = torch.randn(2048).to(self.device)

        model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for x, _ in Bar(self.dataloader):
                x = x.float().to(self.device)

                self.optimizer.zero_grad()
                z = model(x)
                loss = torch.mean(torch.sum((z - c) ** 2, dim=1))
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            self.scheduler.step()
            print('Training OC... Epoch: {}, Loss: {:.3f}'.format(
                    epoch, total_loss/len(self.dataloader)))

            if epoch % 10 == 0:
                result_path = os.path.join(self.savedir, self.data_name, 'modifided')
                os.makedirs(result_path, exist_ok=True)
                torch.save({'center': c.cpu().data.numpy().tolist(),
                            'model_dict': model.state_dict()}, 
                            os.path.join(result_path, f'{epoch}_state_dict.pth'))
                sns.histplot(c.cpu().data.numpy().tolist())
                plt.savefig(os.path.join(result_path, f'{epoch}_histplot.png'))