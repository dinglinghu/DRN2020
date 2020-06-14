from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from external.nms import soft_nms
# from external.monte_carlo_nms import mtcarlo_nms
from external.angle_nms.angle_soft_nms import angle_soft_nms, angle_soft_nms_new

from models.decode import rodet_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import rodet_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector

class RodetDetector(BaseDetector):
  def __init__(self, opt):
    super(RodetDetector, self).__init__(opt)
  
  def process(self, images, return_time=False):
    # output是各个head的字典，本模块对各个head处理，生成rbbox
    with torch.no_grad():
      output = self.model(images)[-1]
      # import pdb;
      # pdb.set_trace()
      hm = output['hm'].sigmoid_()
      wh = output['wh']
      angle = output['angle']
      reg = output['reg'] if self.opt.reg_offset else None
      if self.opt.flip_test:
        hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
        wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
        # angle = (angle[0:1] + 2.0*np.pi-flip_tensor(angle[1:2])) / 2
        angle = (angle[0:1] + 1.0-flip_tensor(angle[1:2])) / 2
        reg = reg[0:1] if reg is not None else None
      torch.cuda.synchronize()
      forward_time = time.time()
      dets = rodet_decode(hm, wh, angle, reg=reg, K=self.opt.K)

    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1, adjust_score=False):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = rodet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_classes)
    if adjust_score:
      ts = 1-scale if scale>=1.0 else 2.0*(1-scale)
      ts = ts/10.
      for j in range(1, self.num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 6)
        dets[0][j][:, :4] /= scale
        modular = np.power(np.max(dets[0][j][:,2:4], axis=1)/96.,ts)
        dets[0][j][:,5] *= modular
        dets[0][j][:,5] = np.clip(dets[0][j][:,5], 0.0, 1.0)
      return dets[0]
    for j in range(1, self.num_classes + 1):
      scales = np.ones(len(dets[0][j])).reshape(-1,1) * scale
      dets_j = np.array(dets[0][j], dtype=np.float32).reshape(-1, 6)
      dets_j = np.hstack((dets_j, scales))
      dets[0][j] = dets_j
      dets[0][j][:, :4] /= scale
    return dets[0]

  def merge_outputs(self, detections, all_cls_nms=False):
    results = {}
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
      if len(self.scales) > 1 or self.opt.nms:
        # import pdb;
        # pdb.set_trace()
        #
        # soft_nms(results[j], Nt=0.5, method=2)
        # keep_nms = angle_soft_nms(results[j], 0.15)
        result_nms = angle_soft_nms(results[j], Nt=0.5, method=1, threshold=0.001)
        results[j] = result_nms
    if all_cls_nms:
      for j in range(1, self.num_classes + 1):
        cls = np.ones((results[j].shape[0],1)) * j
        results[j] = np.hstack([results[j], cls])
      all_dets = np.vstack(
        [results[j] for j in range(1, self.num_classes + 1)])
      all_dets_nms = angle_soft_nms_new(all_dets, Nt=0.5, method=1, threshold=0.001, all_cls=True, cls_decay=1.8)
      # all_dets_nms = all_dets
      results = {}
      for j in range(1, self.num_classes + 1):
        j_ind = all_dets_nms[:,-1] == j
        results[j] = all_dets_nms[j_ind,:-1]

    scores = np.hstack(
      [results[j][:, 5] for j in range(1, self.num_classes + 1)])
    if len(scores) > self.max_per_image:
      # kth = len(scores) - self.max_per_image
      # thresh = np.partition(scores, kth)[kth]
      thresh = 0.03
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, 5] >= thresh)
        results[j] = results[j][keep_inds]
    return results
  def conv_to_img(self, conv_tensor):
    conv_ndary = np.max(conv_tensor.permute(1,2,0).detach().cpu().numpy(), axis=2, keepdims=True)
    conv_ndary = conv_ndary - np.min(conv_ndary)
    conv_ndary = conv_ndary/np.max(conv_ndary)
    conv_ndary = conv_ndary*255
    return conv_ndary

  def debug(self, debugger, images, dets, output, scale=1, image_name=None):
    # detection格式为[cx,cy,w,h,angle, scores, clses]
    # import pdb;
    # pdb.set_trace()
    blend_id = image_name +'_hm_{:.1f}'if image_name is not None else 'pred_hm_{:.1f}'
    img_id= image_name + 'det_{:.1f}' if image_name is not None else 'out_pred_{:.1f}'
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :4] *= self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, blend_id.format(scale))
      debugger.add_img(img, img_id=img_id.format(scale))
      for k in range(len(dets[i])):
        if detection[i, k, 5] > self.opt.center_thresh:
          # add_rbbox(self, rbbox, cat, conf=1, show_txt=True, img_id='default'):
          debugger.add_rbbox(detection[i, k, :5], detection[i, k, -1],
                                 detection[i, k, 5],
                                 img_id=img_id.format(scale))

  def show_results(self, debugger, image, results, image_name=None):
    img_id = image_name if image_name is not None else 'rodet'
    debugger.add_img(image, img_id=img_id)
    for j in range(1, self.num_classes + 1):
      for bbox in results[j]:
        if bbox[5] > self.opt.vis_thresh:
          debugger.add_rbbox(bbox[:5], j - 1, bbox[5], img_id=img_id)
    debugger.save_all_imgs(self.opt.debug_dir)