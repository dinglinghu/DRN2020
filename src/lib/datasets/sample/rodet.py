# -*- coding:utf-8 -*-
# 旋转检测采样

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg, draw_dense_reg_uni
import math

DEBUG = False


class RoDetDataset(data.Dataset):
    def _coco_box_to_bbox(self, box):
        # coco box是左上点和长宽
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def _get_four_points(self, centre, theta, width, height):
        theta = -theta
        c, s = np.cos(theta), np.sin(theta)
        R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))
        p1 = [+ width / 2, + height / 2]  # 右下
        p2 = [- width / 2, + height / 2]  # 左下
        p3 = [- width / 2, - height / 2]  # 左上
        p4 = [+ width / 2, - height / 2]  # 右上
        p1_new = np.dot(p1, R) + centre
        p2_new = np.dot(p2, R) + centre
        p3_new = np.dot(p3, R) + centre
        p4_new = np.dot(p4, R) + centre
        return p1_new, p2_new, p3_new, p4_new

    def _float_to_int(self, point):
        return int(point[0, 0]), int(point[0, 1])

    # 每次读入单个图像
    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']  # 读出图像名称
        img_path = os.path.join(self.img_dir, file_name)  # 图像完成文件名称
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)  # 读取图像对应的GT检测框

        num_objs = min(len(anns), self.max_objs)
        # 读入图像，并对图像进行预处理
        # print(img_id, img_path)
        img = cv2.imread(img_path)
        # import pdb
        # pdb.set_trace()
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

        flipped = False
        if self.split == 'train' or self.split == 'debug1':
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.7, 1.3, 0.1))
                w_border = self._get_border(512, img.shape[1])
                h_border = self._get_border(512, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
                # c[0] = np.random.randint(low=0.4*img.shape[1], high=0.6*img.shape[1] )
                # c[1] = np.random.randint(low=0.4*img.shape[0], high=0.6*img.shape[0])
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        # 根据偏移的c和s得到变换矩阵，之后所有的框也可以按照变换矩阵进行移动
        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        # 0-255转为0-1
        if DEBUG:
            raw_img = inp.copy()
        inp = (inp.astype(np.float32) / 255.)
        # 色彩偏移
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        # 减均值除方差
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        # 图像预处理结束

        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)  # heatmap

        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)  # dense的wh
        angle = np.zeros((self.max_objs, 1), dtype=np.float32)
        dense_angle = np.zeros((1, output_h, output_w), dtype=np.float32)  # dense的angle
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)  # offset偏差值
        ind = np.zeros((self.max_objs), dtype=np.int64)  # 物体在图像上编号，编号根据坐标得到
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)  # 对于图像变化后不存在了物体mask设置为0
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)  # 分类的长宽
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)  # 分类的长宽mask
        cat_spec_angle = np.zeros((self.max_objs, num_classes), dtype=np.float32)  # 分类的长宽
        cat_spec_angle_mask = np.zeros((self.max_objs, num_classes), dtype=np.uint8)  # 分类的长宽mask

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        gt_det = []
        # 遍历所有的物体
        for k in range(num_objs):
            ann = anns[k]
            # bbox = self._coco_box_to_bbox(ann['rbbox'])
            bbox = ann['rbbox']
            cls_id = int(self.cat_ids[ann['category_id']])

            # 跟随图像变化，对于检测框进行相同变换
            if flipped:
                # cx做镜像处理
                bbox[0] = width - bbox[0] - 1

            # 获取四个角点
            pt1, pt2, pt3, pt4 = self._get_four_points((bbox[0], bbox[1]), bbox[-1], bbox[2], bbox[3])
            pt1 = affine_transform((pt1[0, 0], pt1[0, 1]), trans_output)
            pt2 = affine_transform((pt2[0, 0], pt2[0, 1]), trans_output)
            pt3 = affine_transform((pt3[0, 0], pt3[0, 1]), trans_output)
            pt4 = affine_transform((pt4[0, 0], pt4[0, 1]), trans_output)

            # 得到中心点坐标，长宽以及角度
            ct = np.array(
                [(pt1[0] + pt3[0]) / 2, (pt1[1] + pt3[1]) / 2], dtype=np.float32)
            w = np.linalg.norm(pt1 - pt2)
            h = np.linalg.norm(pt1 - pt4)
            # 计算新的angle
            # vec_base = np.array([0, 1], dtype=np.float32)
            # vec_angle = np.array([(pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2], dtype=np.float32) - ct
            # norm_base = np.linalg.norm(vec_base)
            # norm_angle = np.linalg.norm(vec_angle)
            # cos_angle = vec_base.dot(vec_angle) / (norm_base * norm_angle + np.finfo(float).eps)
            # a = np.arccos(cos_angle)

            if self.opt.dataset == 'hrsc':
                a = bbox[-1]
                if flipped:
                    a = np.pi - a
            elif self.opt.dataset == 'dota':
                a = bbox[-1]
                # ####### dota的json文件角度是0到2pi ##########
                if flipped:
                    a = 2 * np.pi - a
            elif self.opt.dataset == 'rosku':
                # ####### rosku的json文件角度是-0.5pi到0.5pi ##########
                a = bbox[-1] / math.pi
                if flipped:
                    a = -1 * a
                a = np.clip(a, -0.5, 0.5)
                a = a + 0.5
            else:
                raise Exception('Wrong dataset.')

            if DEBUG:
                color = [255, 0, 0]
                line_width = 2
                # ####### rosku的json文件角度是-0.5pi到0.5pi ##########
                # temp_a = (a - 0.5) * math.pi
                temp_a = a
                npt1, npt2, npt3, npt4 = self._get_four_points((ct[0], ct[1]), temp_a, w, h)
                npt1 = self._float_to_int(npt1)
                npt2 = self._float_to_int(npt2)
                npt3 = self._float_to_int(npt3)
                npt4 = self._float_to_int(npt4)
                cv2.line(raw_img, npt1, npt2, color, line_width)
                cv2.line(raw_img, npt2, npt3, color, line_width)
                cv2.line(raw_img, npt3, npt4, color, line_width)
                cv2.line(raw_img, npt4, npt1, color, line_width)


            if 0 <= ct[0] <= output_w - 1 and 0 <= ct[1] <= output_h - 1:
                # 热力图，GT进行一定扩散
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = self.opt.hm_gauss if self.opt.mse_loss else radius
                ct_int = ct.astype(np.int32)
                # 中心点绘制GT
                draw_gaussian(hm[cls_id], ct_int, radius)

                wh[k] = 1. * w, 1. * h
                angle[k] = 1. * a
                ind[k] = ct_int[1] * output_w + ct_int[0]  # 物体在特征图上索引值
                reg[k] = ct - ct_int  # ct的实际值和整数化后的偏移
                reg_mask[k] = 1
                # wh设置
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                if self.opt.dense_wh:
                    draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)

                # angle设置
                cat_spec_angle[k, cls_id] = angle[k]
                cat_spec_angle_mask[k, cls_id] = 1
                if self.opt.dense_angle or self.opt.fsm:
                    draw_dense_reg(dense_angle, hm.max(axis=0), ct_int, angle[k], radius)
                # ang_radius = max(int(1.0), int(radius/2.))
                # draw_dense_reg_uni(dense_angle[0, :], ct_int, angle[k], ang_radius)
                gt_det.append([ct[0], ct[1], w, h, angle[k], 1, cls_id])

        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'angle': angle}

        # wh
        if self.opt.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']
        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            del ret['wh']

        # angle
        if self.opt.dense_angle or self.opt.fsm:
            dense_angle_mask = hm.max(axis=0, keepdims=True)
            ret.update({'dense_angle': dense_angle, 'dense_angle_mask': dense_angle_mask})
            if self.opt.dense_angle:
                del ret['angle']
        elif self.opt.cat_spec_angle:
            ret.update({'cat_spec_angle': cat_spec_angle, 'cat_spec_angle_mask': cat_spec_angle_mask})
            del ret['angle']

        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 7), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id, 'img_name':file_name}
            ret['meta'] = meta

        if DEBUG:
            ret['raw_img'] = raw_img
            ret['gt_det'] = gt_det
            ret['img_id'] = img_id
            cv2.imwrite(os.path.join('./cache', '%s.jpg' % img_id), raw_img)
        return ret