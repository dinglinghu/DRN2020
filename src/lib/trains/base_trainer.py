from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter
import numpy as np
import matplotlib.pyplot as plt
import sys


class ModleWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModleWithLoss, self).__init__()
    self.model = model
    self.loss = loss
  
  def forward(self, batch, fsm=False):
    if fsm:
      # outputs = self.model(batch['input'], angle_gt=batch['dense_angle'], angle_gt_mask=batch['dense_angle_mask'])
      outputs = self.model(batch['input'], angle_gt=batch['dense_angle'], train=True)
    else:
      outputs = self.model(batch['input'])
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats

class BaseTrainer(object):
  def __init__(
    self, opt, model, optimizer=None):
    self.opt = opt
    self.optimizer = optimizer
    self.loss_stats, self.loss = self._get_losses(opt)
    self.model_with_loss = ModleWithLoss(model, self.loss)

  def set_device(self, gpus, chunk_sizes, device):
    if len(gpus) > 1:
      self.model_with_loss = DataParallel(
        self.model_with_loss, device_ids=gpus, 
        chunk_sizes=chunk_sizes).to(device)
    else:
      self.model_with_loss = self.model_with_loss.to(device)

    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)


  def run_epoch(self, phase, epoch, data_loader):
    model_with_loss = self.model_with_loss
    if phase == 'train':
      model_with_loss.train()
    else:
      if len(self.opt.gpus) > 1:
        model_with_loss = self.model_with_loss.module
      model_with_loss.eval()
      torch.cuda.empty_cache()

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)

      for k in batch:
        if k != 'meta':
          # print("k=",k)
          # print("batch[k].size=",batch[k].size())
          batch[k] = batch[k].to(device=opt.device, non_blocking=True)
      output, loss, loss_stats = model_with_loss(batch, fsm=self.opt.fsm)
      loss = loss.mean()
      if phase == 'train':
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
      batch_time.update(time.time() - end)
      end = time.time()

      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td)
      for l in avg_loss_stats:
        if l not in loss_stats:
          continue
        avg_loss_stats[l].update(
          loss_stats[l].mean().item(), batch['input'].size(0))
        Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

      if not opt.hide_data_time:
        Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      if opt.print_iter > 0:
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
      else:
        bar.next()
      
      if opt.debug > 0:
        self.debug(batch, output, iter_id)
      
      if opt.test:
        self.save_result(output, batch, results)
      del output, loss, loss_stats
    
    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results


  def find_lr(self, epoch, data_loader, lr_start=1e-5, lr_end=1e0, beta=0.98):
    model_with_loss = self.model_with_loss
    model_with_loss.train()

    opt = self.opt
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()

    lr_mul = (lr_end / lr_start) ** (1. / (num_iters-1))
    lrs = []
    losses = []
    avg_loss = 0
    best_loss = 1e9
    for param_group in self.optimizer.param_groups:
      param_group['lr'] = lr_start
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)

      for k in batch:
        if k != 'meta':
          # print("k=",k)
          # print("batch[k].size=",batch[k].size())
          batch[k] = batch[k].to(device=opt.device, non_blocking=True)
      e2e = True if self.opt.e2e else False
      output, loss, loss_stats = model_with_loss(batch, e2e=e2e)
      loss = loss.mean()
      avg_loss = beta * avg_loss + (1 - beta) * loss.data
      smoothed_loss = avg_loss / (1 - beta ** (iter_id+1))
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      batch_time.update(time.time() - end)
      end = time.time()
      if smoothed_loss < best_loss:
        best_loss = smoothed_loss
      lrs.append(self.optimizer.param_groups[0]['lr'])
      losses.append(float(smoothed_loss))

      if smoothed_loss > 4 * best_loss and iter_id > 0:
        break

      for param_group in self.optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_mul

      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase='train',
        total=bar.elapsed_td, eta=bar.eta_td)
      for l in avg_loss_stats:
        if l not in loss_stats:
          continue
        avg_loss_stats[l].update(
          loss_stats[l].mean().item(), batch['input'].size(0))
        Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
      Bar.suffix = Bar.suffix + '|lr {:.6f}'.format(lrs[-1])
      if not opt.hide_data_time:
        Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                  '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      if opt.print_iter > 0:
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
      else:
        bar.next()

      if opt.debug > 0:
        self.debug(batch, output, iter_id)
      del output, loss, loss_stats
    plt.figure()
    plt.xticks(np.log([1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0]),(1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0))
    plt.xlabel('learning rate')
    plt.ylabel('loss')
    plt.plot(np.log(lrs), losses)
    # plt.show()
    plt.savefig(opt.exp_id+'_lr_loss.png')
    sys.exit()

  def debug(self, batch, output, iter_id):
    raise NotImplementedError

  def save_result(self, output, batch, results):
    raise NotImplementedError

  def _get_losses(self, opt):
    raise NotImplementedError
  
  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader, find_lr=False):
    if find_lr:
      self.find_lr(epoch, data_loader)
    return self.run_epoch('train', epoch, data_loader)

