import logging
import torch.nn as nn
import torch

logger = logging.getLogger()

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset() # 들어오면 일단 리셋 -> 인자들 생성한 뒤 0으로 맞춰짐.

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EmptyWith:
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
class ConsistencyCos(nn.Module):
    def __init__(self):
        super(ConsistencyCos, self).__init__()
        self.mse_fn = nn.MSELoss()

    def forward(self, feat):
        feat = nn.functional.normalize(feat, dim=1)

        feat_0 = feat[:int(feat.size(0)/2),:]
        feat_1 = feat[int(feat.size(0)/2):,:]
        cos = torch.einsum('nc,nc->n', [feat_0, feat_1]).unsqueeze(-1)
        labels = torch.ones((cos.shape[0],1), dtype=torch.float, requires_grad=False)
        if torch.cuda.is_available():
            labels = labels.cuda()
        loss = self.mse_fn(cos, labels)
        return loss