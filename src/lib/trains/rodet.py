from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss, BinRotLoss, RegMSELoss, DenseRegL1Loss
from models.decode import rodet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer


class RodetLoss(torch.nn.Module):
  def __init__(self, opt):
    super(RodetLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
              NormRegL1Loss() if opt.norm_wh else \
              RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
    self.crit_angle = BinRotLoss() if opt.rotate_binloss else RegMSELoss()
    self.crit_dense_angle = DenseRegL1Loss() if opt.dense_angle else None
    self.opt = opt

  # def forward(self, outputs, batch):
  #   opt = self.opt
  #   hm_loss, wh_loss, angle_loss, off_loss = 0, 0, 0, 0
  #   for s in range(opt.num_stacks):
  #     output = outputs[s]
  #     if not opt.mse_loss:
  #       output['hm'] = _sigmoid(output['hm'])
  #
  #     if opt.eval_oracle_hm:
  #       output['hm'] = batch['hm']
  #     if opt.eval_oracle_wh:
  #       output['wh'] = torch.from_numpy(gen_oracle_map(
  #         batch['wh'].detach().cpu().numpy(),
  #         batch['ind'].detach().cpu().numpy(),
  #         output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
  #
  #     if opt.eval_oracle_angle:
  #       output['angle'] = torch.from_numpy(gen_oracle_map(
  #         batch['angle'].detach().cpu().numpy(),
  #         batch['ind'].detach().cpu().numpy(),
  #         output['angle'].shape[3], output['angle'].shape[2])).to(opt.device)
  #
  #     if opt.eval_oracle_offset:
  #       output['reg'] = torch.from_numpy(gen_oracle_map(
  #         batch['reg'].detach().cpu().numpy(),
  #         batch['ind'].detach().cpu().numpy(),
  #         output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)
  #
  #     hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
  #     if opt.wh_weight > 0:
  #       if opt.dense_wh:
  #         mask_weight = batch['dense_wh_mask'].sum() + 1e-4
  #         wh_loss += (
  #           self.crit_wh(output['wh'] * batch['dense_wh_mask'],
  #           batch['dense_wh'] * batch['dense_wh_mask']) /
  #           mask_weight) / opt.num_stacks
  #       elif opt.cat_spec_wh:
  #         wh_loss += self.crit_wh(
  #           output['wh'], batch['cat_spec_mask'],
  #           batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
  #       else:
  #         wh_loss += self.crit_reg(
  #           output['wh'], batch['reg_mask'],
  #           batch['ind'], batch['wh']) / opt.num_stacks
  #
  #     if opt.angle_weight > 0:
  #       output['angle'] = _sigmoid(output['angle'])
  #       if opt.dense_angle:
  #         # mask_weight = batch['dense_angle_mask'].sum() + 1e-4
  #         # angle_loss += (
  #         #   self.crit_angle(output['angle'] * batch['dense_angle_mask'],
  #         #   batch['dense_angle'] * batch['dense_angle_mask']) /
  #         #   mask_weight) / opt.num_stacks
  #         angle_loss += self.crit_dense_angle(output['angle'], batch['dense_angle_mask'], batch['dense_angle']) / len(
  #           outputs)
  #       elif opt.cat_spec_angle:
  #         angle_loss += self.crit_angle(
  #           output['angle'], batch['cat_spec_angle_mask'],
  #           batch['ind'], batch['cat_spec_angle']) / opt.num_stacks
  #       else:
  #         angle_loss += self.crit_reg(
  #           output['angle'], batch['reg_mask'],
  #           batch['ind'], batch['angle']) / opt.num_stacks
  #
  #     if opt.reg_offset and opt.off_weight > 0:
  #       off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
  #                            batch['ind'], batch['reg']) / opt.num_stacks
  #
  #   loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
  #          opt.off_weight * off_loss + opt.angle_weight * angle_loss
  #   loss_stats = {'loss': loss, 'hm_loss': hm_loss,
  #                 'wh_loss': wh_loss, 'angle_loss': angle_loss, 'off_loss': off_loss}
  #   return loss, loss_stats

  def forward(self, outputs, batch):
    opt = self.opt
    hm_loss, wh_loss, angle_loss, off_loss= 0, 0, 0, 0
    for s in range(len(outputs)):
      output = outputs[s]
      if not opt.mse_loss:
        output['hm'] = _sigmoid(output['hm'])

      if opt.eval_oracle_hm:
        output['hm'] = batch['hm']

      if opt.eval_oracle_wh:
        output['wh'] = torch.from_numpy(gen_oracle_map(
          batch['wh'].detach().cpu().numpy(),
          batch['ind'].detach().cpu().numpy(),
          output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)

      if opt.eval_oracle_angle:
        output['angle'] = torch.from_numpy(gen_oracle_map(
          batch['angle'].detach().cpu().numpy(),
          batch['ind'].detach().cpu().numpy(),
          output['angle'].shape[3], output['angle'].shape[2])).to(opt.device)

      if opt.eval_oracle_offset:
        output['reg'] = torch.from_numpy(gen_oracle_map(
          batch['reg'].detach().cpu().numpy(),
          batch['ind'].detach().cpu().numpy(),
          output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

      hm_loss += self.crit(output['hm'], batch['hm']) / len(outputs)

      if opt.wh_weight > 0:
        if opt.dense_wh:
          mask_weight = batch['dense_wh_mask'].sum() + 1e-4
          wh_loss += (
                             self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                                          batch['dense_wh'] * batch['dense_wh_mask']) /
                             mask_weight) / len(outputs)
        elif opt.cat_spec_wh:
          wh_loss += self.crit_wh(
            output['wh'], batch['cat_spec_mask'],
            batch['ind'], batch['cat_spec_wh']) / len(outputs)
        else:
          wh_loss += self.crit_reg(
            output['wh'], batch['reg_mask'],
            batch['ind'], batch['wh']) / len(outputs)


      if opt.angle_weight > 0:
        # output['angle'] = _sigmoid(output['angle'])
        if opt.dense_angle:
          # mask_weight = batch['dense_angle_mask'].sum() + 1e-4
          # angle_loss += (self.crit_angle(output['angle'] * batch['dense_angle_mask'],
          #                 batch['dense_angle'] * batch['dense_angle_mask']) /mask_weight) / len(outputs)
          angle_loss += self.crit_dense_angle(output['angle'], batch['dense_angle_mask'], batch['dense_angle']) / len(outputs)
        elif opt.cat_spec_angle:
          angle_loss += self.crit_angle(
            output['angle'], batch['cat_spec_angle_mask'],
            batch['ind'], batch['cat_spec_angle']) / len(outputs)
        else:
          angle_loss += self.crit_reg(
            output['angle'], batch['reg_mask'],
            batch['ind'], batch['angle']) / len(outputs)

      if opt.reg_offset and opt.off_weight > 0:
        off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                  batch['ind'], batch['reg']) / len(outputs)

    loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
           opt.off_weight * off_loss + opt.angle_weight * angle_loss
    loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                  'wh_loss': wh_loss, 'angle_loss': angle_loss, 'off_loss': off_loss}

    return loss, loss_stats

class RodetTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(RodetTrainer, self).__init__(opt, model, optimizer=optimizer)
  
  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_loss', 'wh_loss', 'angle_loss', 'off_loss']
    loss = RodetLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    dets = rodet_decode(
      output['hm'], output['wh'],output['angle'], reg=reg,
      cat_spec_wh=opt.cat_spec_wh, cat_spec_angle=opt.cat_spec_angle, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets[:, :, :4] *= opt.down_ratio

    # dets_gt_dense = rodet_decode(
    #   batch['hm'], batch['dense_wh'], batch['dense_angle'], reg=reg,
    #   cat_spec_wh=opt.cat_spec_wh, cat_spec_angle=opt.cat_spec_angle, K=opt.K)
    # dets_gt_dense = dets_gt_dense.detach().cpu().numpy().reshape(1, -1, dets_gt_dense.shape[2])
    # dets_gt_dense[:, :, :4] *= opt.down_ratio

    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    img_name = batch['meta']['img_name']
    dets_gt[:, :, :4] *= opt.down_ratio
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      # gt_ang_mask = debugger.gen_colormap(batch['dense_angle_mask'][i].detach().cpu().numpy())
      # gt_ang = debugger.gen_colormap(batch['dense_angle'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, '{}_pred_hm'.format(img_name))
      # debugger.add_blend_img(img, gt_ang_mask, 'gt_angle_mask')
      # debugger.add_blend_img(img, gt_ang, 'gt_angle')
      debugger.add_blend_img(img, gt, '{}_gt_hm'.format(img_name))
      debugger.add_img(img, img_id='{}_out_pred'.format(img_name))
      for k in range(len(dets[i])):
        if dets[i, k, 5] > opt.center_thresh:
          # print("pred dets add_rbbox=======================")
          debugger.add_rbbox(dets[i, k, :5], dets[i, k, -1],
                                 dets[i, k, 5], show_txt=False, img_id='{}_out_pred'.format(img_name))

      # debugger.add_img(img, img_id='{}_dets_gt_dense'.format(img_name))
      # for k in range(len(dets_gt_dense[i])):
      #   if dets_gt_dense[i, k, 5] > opt.center_thresh:
      #     # print("pred dets add_rbbox=======================")
      #     debugger.add_rbbox(dets_gt_dense[i, k, :5], dets_gt_dense[i, k, -1],
      #                        dets_gt_dense[i, k, 5], show_txt=False, img_id='{}_dets_gt_dense'.format(img_name))

      debugger.add_img(img, img_id='{}_out_gt'.format(img_name))
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 5] > opt.center_thresh:
          # print("GT add_rbbox=======================")
          # 说明add_rbbox(self, rbbox, cat, conf=1, show_txt=True, img_id='default')
          # gt格式 gt_det.append([ct[0], ct[1], w, h, a, 1, cls_id])
          debugger.add_rbbox(dets_gt[i, k, :5], dets_gt[i, k, -1],
                                 dets_gt[i, k, 5], show_txt=False, img_id='{}_out_gt'.format(img_name))

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    reg = output['reg'] if self.opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets_out = ctdet_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]