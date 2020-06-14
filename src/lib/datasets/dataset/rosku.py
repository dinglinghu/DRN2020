# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import pycocotools.coco as coco
# from pycocotools.cocoeval import COCOeval
import pycocotools_ro.coco as coco
from pycocotools_ro.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data


class ROSKU(data.Dataset):
    num_classes = 1  # real class num ,without background
    # input resolution must can dividable by 32 (or 128 if you are using HourglassNet).
    default_resolution = [768, 768]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(ROSKU, self).__init__()
        self.data_dir = os.path.join(opt.data_dir, 'rosku')
        # self.img_dir = os.path.join(self.data_dir, 'images', '{}'.format(split))
        self.img_dir = os.path.join(self.data_dir, 'JPEGImages')
        if split == 'test':
            if opt.test_split != -1:
                self.annot_path = os.path.join(
                    self.data_dir, '../rosku_testsplitjson',
                    'voc_sku110k-r_{}_{}.json').format(split, opt.test_split)
            else:
                self.annot_path = os.path.join(
                    self.data_dir, 'cocoformatJson',
                    'voc_sku110k-r_test.json').format(split)
        else:
            if opt.task == 'exdet':
                self.annot_path = os.path.join(
                    self.data_dir, 'annotations',
                    'instances_extreme_{}2017.json').format(split)
            else:
                self.annot_path = os.path.join(
                    self.data_dir, 'cocoformatJson',
                    'voc_sku110k-r_{}.json').format(split)
        self.max_objs = opt.K
        self.class_name = ['__background__', '0']
        self._valid_ids = np.arange(1, 2, dtype=np.int32)
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split
        self.opt = opt

        print('==> initializing sku-110K ROSKU {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    score = bbox[5]
                    bbox_out = list(map(self._to_float, bbox[0:5]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "rbbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    # if len(bbox) > 5:
                    #     extreme_points = list(map(self._to_float, bbox[5:13]))
                    #     detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "rbbox")
        coco_eval.params.maxDets = [1, 10, 300]
        # coco_eval.params.imgIds = [50]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()