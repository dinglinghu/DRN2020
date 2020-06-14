from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds
from .image import get_affine_transform, affine_transform
from .image import get_four_points, float_to_int
from .ddd_utils import ddd2locrot


def get_pred_depth(depth):
  return depth

def get_alpha(rot):
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[:, 0]
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = np.arctan(rot[:, 2] / rot[:, 3]) + (-0.5 * np.pi)
  alpha2 = np.arctan(rot[:, 6] / rot[:, 7]) + ( 0.5 * np.pi)
  return alpha1 * idx + alpha2 * (1 - idx)
  

def ddd_post_process_2d(dets, c, s, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  include_wh = dets.shape[2] > 16
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (opt.output_w, opt.output_h))
    classes = dets[i, :, -1]
    for j in range(opt.num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :3].astype(np.float32),
        get_alpha(dets[i, inds, 3:11])[:, np.newaxis].astype(np.float32),
        get_pred_depth(dets[i, inds, 11:12]).astype(np.float32),
        dets[i, inds, 12:15].astype(np.float32)], axis=1)
      if include_wh:
        top_preds[j + 1] = np.concatenate([
          top_preds[j + 1],
          transform_preds(
            dets[i, inds, 15:17], c[i], s[i], (opt.output_w, opt.output_h))
          .astype(np.float32)], axis=1)
    ret.append(top_preds)
  return ret

def ddd_post_process_3d(dets, calibs):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  for i in range(len(dets)):
    preds = {}
    for cls_ind in dets[i].keys():
      preds[cls_ind] = []
      for j in range(len(dets[i][cls_ind])):
        center = dets[i][cls_ind][j][:2]
        score = dets[i][cls_ind][j][2]
        alpha = dets[i][cls_ind][j][3]
        depth = dets[i][cls_ind][j][4]
        dimensions = dets[i][cls_ind][j][5:8]
        wh = dets[i][cls_ind][j][8:10]
        locations, rotation_y = ddd2locrot(
          center, alpha, dimensions, depth, calibs[0])
        bbox = [center[0] - wh[0] / 2, center[1] - wh[1] / 2,
                center[0] + wh[0] / 2, center[1] + wh[1] / 2]
        pred = [alpha] + bbox + dimensions.tolist() + \
               locations.tolist() + [rotation_y, score]
        preds[cls_ind].append(pred)
      preds[cls_ind] = np.array(preds[cls_ind], dtype=np.float32)
    ret.append(preds)
  return ret

def ddd_post_process(dets, c, s, calibs, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  dets = ddd_post_process_2d(dets, c, s, opt)
  dets = ddd_post_process_3d(dets, calibs)
  return dets


def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(
          dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret

def rodet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  point_dets = np.zeros([dets.shape[0],dets.shape[1], 8], dtype = np.float32)
  for i in range(dets.shape[0]):
    top_preds = {}
    # 获取四个角点
    for dets_k in range(dets.shape[1]):
      rbbox = dets[i, dets_k, :]
      # 右下、左下、左上、右上
      angle = (rbbox[4]-0.5) * np.pi
      pt1, pt2, pt3, pt4 = get_four_points((rbbox[0], rbbox[1]), rbbox[4], rbbox[2], rbbox[3])
      # print(pt1, pt2, pt3, pt4)
      point_dets[i, dets_k] = np.array([pt1[0, 0],pt1[0, 1], pt2[0, 0],pt2[0, 1], pt3[0, 0], pt3[0, 1], pt4[0, 0],pt4[0, 1]] )
    # import pdb
    # pdb.set_trace()
    point_dets[i, :, 0:2] = transform_preds(point_dets[i, :, 0:2], c[i], s[i], (w, h))
    point_dets[i, :, 2:4] = transform_preds(point_dets[i, :, 2:4], c[i], s[i], (w, h))
    point_dets[i, :, 4:6] = transform_preds(point_dets[i, :, 4:6], c[i], s[i], (w, h))
    point_dets[i, :, 6:8] = transform_preds(point_dets[i, :, 6:8], c[i], s[i], (w, h))
    # angle不进行仿射变换
    # 得到中心点坐标，长宽以及角度
    ct = np.array([(point_dets[i, :, 0] + point_dets[i, :, 4]) / 2, (point_dets[i, :, 1] + point_dets[i, :, 5]) / 2], dtype=np.float32)
    w = np.linalg.norm(point_dets[i, :, 0:2] - point_dets[i, :, 2:4], axis=1)
    h = np.linalg.norm(point_dets[i, :, 0:2] - point_dets[i, :, 6:8], axis=1)

    dets[i, :, 0:2] = np.transpose(ct)
    dets[i, :, 2] = w
    dets[i, :, 3] = h
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :5].astype(np.float32),
        dets[i, inds, 5:6].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret

def multi_pose_post_process(dets, c, s, h, w):
  # dets: batch x max_dets x 40
  # return list of 39 in image coord
  ret = []
  for i in range(dets.shape[0]):
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
    pts = transform_preds(dets[i, :, 5:39].reshape(-1, 2), c[i], s[i], (w, h))
    top_preds = np.concatenate(
      [bbox.reshape(-1, 4), dets[i, :, 4:5], 
       pts.reshape(-1, 34)], axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret
