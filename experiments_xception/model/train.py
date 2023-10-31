import logging
import wandb
import time
import os
import json
import torch
from collections import OrderedDict
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from metrics import confusion_matrix, plot_roc_curve


_logger = logging.getLogger('train') # 이름이 train인 로거 객체 생성

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

def train(model, dataloader, criterion, optimizer, log_interval: int, device: str) -> dict:   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_m = AverageMeter()
    losses_m = AverageMeter()
    
    end = time.time()
    
    model.train()
    optimizer.zero_grad()
    for idx, (inputs, targets) in enumerate(dataloader):
        data_time_m.update(time.time() - end)
        
        inputs, targets = inputs.to(device), targets.to(device)

        # predict
        outputs = model(inputs)
        loss = criterion(outputs, targets)    
        loss.backward()

        # loss update
        optimizer.step()
        optimizer.zero_grad()
        losses_m.update(loss.item())

        # accuracy
        preds = outputs.argmax(dim=1) 
        acc_m.update(targets.eq(preds).sum().item()/targets.size(0), n=targets.size(0)) # eq는 element 별로 비교해서, True of Flase로 출력 (sum으로 맞힌 개수)
        
        batch_time_m.update(time.time() - end)
    
        if (idx+1) % log_interval == 0 and idx != 0: 
            _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) ' # train logger에 쌓아 놓음
                        'Acc: {acc.avg:.3%} '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        idx+1, len(dataloader), 
                        loss       = losses_m, 
                        acc        = acc_m, 
                        lr         = optimizer.param_groups[0]['lr'],
                        batch_time = batch_time_m,
                        rate       = inputs.size(0) / batch_time_m.val,
                        rate_avg   = inputs.size(0) / batch_time_m.avg,
                        data_time  = data_time_m))
    
        end = time.time()
    
    return OrderedDict([('acc',acc_m.avg), ('loss',losses_m.avg)])
        
def test(model, dataloader, criterion, log_interval: int, device: str, mode: str=None) -> dict:
    correct = 0
    total = 0
    total_loss = 0

    whole_probs = []
    whole_targets = []
    
    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # predict
            outputs = model(inputs)
            
            # loss 
            loss = criterion(outputs, targets)
            
            # total loss and acc
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += targets.eq(preds).sum().item()
            total += targets.size(0)

            if idx % log_interval == 0 and idx != 0: 
                _logger.info('TEST [%d/%d]: Loss: %.3f | Acc: %.3f%% [%d/%d]' % 
                            (idx+1, len(dataloader), total_loss/(idx+1), 100.*correct/total, correct, total))

                    
    return OrderedDict([('acc',correct/total), ('loss',total_loss/len(dataloader))])

def hyp_test(model, dataloader, device: str) -> dict:
    correct = 0
    total = 0

    whole_probs = []
    whole_preds = []
    whole_targets = []
   
    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # predict
            outputs = model(inputs)
            
            preds = outputs.argmax(dim=1)
            correct += targets.eq(preds).sum().item()
            total += targets.size(0)


            # for AUROC
        
            whole_probs.extend(F.softmax(outputs, dim=1)[:,1].cpu().numpy())
            whole_preds.extend(preds.cpu().numpy())
            whole_targets.extend(targets.cpu().tolist())

    # AUROC score
    confusion_matrix(testY=whole_targets, test_pred=whole_preds)
    plot_roc_curve(testY=whole_targets, test_pred=whole_preds, test_prob=whole_probs)
    return OrderedDict([('acc',correct/total)])

                
def fit(
    model, trainloader, validloader, criterion, optimizer, scheduler, 
    epochs: int, savedir: str, data_name: str, log_interval: int, device: str, modifying: bool ) -> None:

    best_acc = 0
    step = 0

    epoch_for_record = []
    acc_for_record = []
    for epoch in range(epochs):
        _logger.info(f'\nEpoch: {epoch+1}/{epochs}')
        train_metrics = train(model, trainloader, criterion, optimizer, log_interval, device)
        eval_metrics = test(model, validloader, criterion, log_interval, device)

        # wandb
        metrics = OrderedDict(lr=optimizer.param_groups[0]['lr'])
        metrics.update([('train_' + k, v) for k, v in train_metrics.items()])
        metrics.update([('eval_' + k, v) for k, v in eval_metrics.items()])
        wandb.log(metrics, step=step)

        step += 1

        # step scheduler
        if scheduler:
            scheduler.step()

        # checkpoint
        if best_acc < eval_metrics['acc']:
            # save results
            if modifying == True:
                result_path = os.path.join(savedir, data_name, 'modifided')
            else:
                result_path = os.path.join(savedir, data_name)
            os.makedirs(result_path, exist_ok=True)
            state = {'best_epoch':epoch, 'best_acc':eval_metrics['acc']}
            json.dump(state, open(os.path.join(result_path, f'best_results.json'),'w'), indent=4)

            # save model
            torch.save(model.state_dict(), os.path.join(result_path, f'best_model.pt'))
            
            _logger.info('Best Accuracy {0:.3%} to {1:.3%}'.format(best_acc, eval_metrics['acc']))

            best_acc = eval_metrics['acc']

    _logger.info('Best Metric: {0:.3%} (epoch {1:})'.format(state['best_acc'], state['best_epoch']))