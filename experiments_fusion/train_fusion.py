import logging
import wandb
import time
import os
import json
import torch
from collections import OrderedDict
import torch.nn.functional as F
from utils import AverageMeter, EmptyWith
from barbar import Bar
import torch.nn as nn


_logger = logging.getLogger('train') # 이름이 train인 로거 객체 생성

def train(model1, model2, dataloader, criterion1, criterion2, consistency_rate, optimizer, log_interval: int, device: str, amp: bool) -> dict:   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_m = AverageMeter()
    losses_m = AverageMeter()
    feature_norm_m = AverageMeter()
    train_loss_consistency_m = AverageMeter()
    train_loss_ce_m = AverageMeter()
    
    end = time.time()
    
    # torch_autocast 인자
    if amp:
       scaler = torch.cuda.amp.GradScaler()
       
    model1.eval() # Normal feature extractor
    model2.train() # DF Classifier
    optimizer.zero_grad()
    for idx, (inputs, targets) in enumerate(dataloader):
        data_time_m.update(time.time() - end)
        
        inputs, targets = inputs.to(device), targets.to(device)
        N = targets.size(0)

        # set autocast
        if amp:
            amp_class = torch.cuda.amp.autocast
        else:
            amp_class = EmptyWith

        # forward
        with amp_class(): # auto_cast
            features1, _ = model1(inputs)
            features2, outputs = model2(inputs)
            features = torch.cat([features1, features2],dim=0) # Grouping features in batch_dim
            feature_norm_m.update(torch.mean(torch.sqrt(torch.sum(features*features,dim=1))).item(),N) # feature normalization

            #loss
            loss_ce = criterion1(outputs, targets) # Cross entropy
            loss_consistency = criterion2(features) # cos. sim.
            loss = consistency_rate * loss_consistency + loss_ce

        # backward
        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()
        losses_m.update(loss.item())

        # accuracy
        preds = outputs.argmax(dim=1) 
        train_loss_consistency_m.update(loss_consistency.item(), N)
        train_loss_ce_m.update(loss_ce.item(), N)
        acc_m.update(targets.eq(preds).sum().item()/targets.size(0), n=targets.size(0)) # eq는 element 별로 비교해서, True of Flase로 출력 (sum으로 맞힌 개수)
        batch_time_m.update(time.time() - end)
    
        if (idx+1) % log_interval == 0 and idx != 0: 
            _logger.info('TRAIN [{:>4d}/{}] ' 
                        'Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                        'Consistency Loss: {loss_consistency.val:>6.4f} ({loss_consistency.avg:>6.4f}) '
                        'CE Loss {loss_ce.val:>6.4f} ({loss_ce.avg:>6.4f}) '
                        'Feature_norm {feature_norm.val:>6.4f} ({feature_norm.avg:>6.4f}) '
                        'Acc: {acc.avg:.3%} '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        idx+1, len(dataloader), 
                        loss       = losses_m, 
                        loss_consistency = train_loss_consistency_m,
                        loss_ce = train_loss_ce_m,
                        feature_norm = feature_norm_m,
                        acc        = acc_m, 
                        lr         = optimizer.param_groups[0]['lr'],
                        batch_time = batch_time_m,
                        rate       = inputs.size(0) / batch_time_m.val,
                        rate_avg   = inputs.size(0) / batch_time_m.avg,
                        data_time  = data_time_m))
    
        end = time.time()
    
    return OrderedDict([('acc',acc_m.avg), ('loss',losses_m.avg)])
      
def test(model1, model2, dataloader, criterion1, criterion2, consistency_rate, amp, log_interval: int, device: str) -> dict:
    correct = 0
    total = 0
    total_loss = 0
    
    model1.eval()
    model2.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            N = targets.size(0)

            # set autocast
            if amp:
                amp_class = torch.cuda.amp.autocast
            else:
                amp_class = EmptyWith

            # forward
            with amp_class(): # auto_cast
                features1, _ = model1(inputs)
                features2, outputs = model2(inputs)
                features = torch.cat([features1, features2], dim=0) # Grouping features in batch_dim
                
                #loss
                loss_ce = criterion1(outputs, targets) # Cross entropy
                loss_consistency = criterion2(features) # cos. sim.
                loss = consistency_rate * loss_consistency + loss_ce
            
            # total loss and acc
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += targets.eq(preds).sum().item()
            total += targets.size(0)

            if idx % log_interval == 0 and idx != 0: 
                _logger.info('TEST [%d/%d]: Loss: %.3f | Acc: %.3f%% [%d/%d]' % 
                            (idx+1, len(dataloader), total_loss/(idx+1), 100.*correct/total, correct, total))
    return OrderedDict([('acc',correct/total), ('loss',total_loss/len(dataloader))])

# def hyp_test(model, dataloader, device: str) -> dict:
#     correct = 0
#     total = 0

#     whole_probs = []
#     whole_preds = []
#     whole_targets = []
   
#     model.eval()
#     with torch.no_grad():
#         for idx, (inputs, targets) in enumerate(dataloader):
#             inputs, targets = inputs.to(device), targets.to(device)
            
#             # predict
#             outputs = model(inputs)
            
#             preds = outputs.argmax(dim=1)
#             correct += targets.eq(preds).sum().item()
#             total += targets.size(0)


#             # for AUROC
        
#             whole_probs.extend(F.softmax(outputs, dim=1)[:,1].cpu().numpy())
#             whole_preds.extend(preds.cpu().numpy())
#             whole_targets.extend(targets.cpu().tolist())

#     # AUROC score
#     confusion_matrix(testY=whole_targets, test_pred=whole_preds)
#     plot_roc_curve(testY=whole_targets, test_pred=whole_preds, test_prob=whole_probs)
#     return OrderedDict([('acc',correct/total)])

                
def fit( model1, model2, trainloader, validloader, criterion1, criterion2, consistency_rate, optimizer, scheduler, 
    epochs: int, savedir: str, log_interval: int, device: str, modifying: bool, amp:bool ) -> None:

    best_acc = 0
    step = 0

    for epoch in range(epochs):
        _logger.info(f'\nEpoch: {epoch+1}/{epochs}')
        train_metrics = train(model1, model2, trainloader, criterion1, criterion2, consistency_rate, optimizer, log_interval, device, amp)
        eval_metrics = test(model1, model2, validloader, criterion1, criterion2, consistency_rate, amp, log_interval, device)

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
                result_path = os.path.join(savedir, 'modifided')
            else:
                result_path = os.path.join(savedir)
            os.makedirs(result_path, exist_ok=True)
            state = {'best_epoch':epoch, 'best_acc':eval_metrics['acc']}
            json.dump(state, open(os.path.join(result_path, f'best_results.json'),'w'), indent=4)

            # save model
            torch.save(model2.state_dict(), os.path.join(result_path, f'best_model.pt'))
            
            _logger.info('Best Accuracy {0:.3%} to {1:.3%}'.format(best_acc, eval_metrics['acc']))

            best_acc = eval_metrics['acc']

    _logger.info('Best Metric: {0:.3%} (epoch {1:})'.format(state['best_acc'], state['best_epoch']))