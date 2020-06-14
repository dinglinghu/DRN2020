# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import pycocotools.coco as coco
# from pycocotools.cocoeval import COCOeval
import pycocotools_ro.coco as coco
from pycocotools_ro.cocoeval import COCOeval
from pycocotools_ro import mask as maskUtils
import json
import os
from collections import defaultdict
import cv2
import numpy as np
import torch.utils.data as data
from utils.debugger import Debugger

class DOTA(data.Dataset):
    num_classes = 15  # real class num ,without background
    # input resolution must can dividable by 32 (or 128 if you are using HourglassNet).
    default_resolution = [1024, 1024]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(DOTA, self).__init__()
        self.data_dir = os.path.join(opt.data_dir, 'dota')
        # self.img_dir = os.path.join(self.data_dir, 'images', '{}'.format(split))
        self.img_dir = os.path.join(self.data_dir, 'JPEGImages')
        if split == 'test':
            if opt.test_split != -1:
                self.annot_path = os.path.join(
                    self.data_dir, 'voc_DOTA_testset_test4s_{}.json'.format(opt.test_split))
            else:
                self.annot_path = os.path.join(
                    self.data_dir, 'voc_DOTA_testset_test.json')
            self.img_dir = os.path.join(self.data_dir, 'test_images')
        elif split == 'trainval':
            self.annot_path = os.path.join(
                self.data_dir, 'DOTA_CONDI/trainval/cocoformatJson',
                'DOTA_aug_trainval.json')
            self.img_dir = os.path.join(self.data_dir, 'DOTA_CONDI/trainval/JPEGImages')
            split = 'train'
        else:
            if opt.task == 'exdet':
                self.annot_path = os.path.join(
                    self.data_dir, 'annotations',
                    'instances_extreme_{}2017.json').format(split)
            else:
                self.annot_path = os.path.join(
                    self.data_dir, 'cocoformatJson',
                    'voc_6-5augmentation_{}.json').format(split)
        self.max_objs = opt.K
        self.class_name = ['__background__', 'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                'small-vehicle', 'large-vehicle', 'ship',
                'tennis-court', 'basketball-court',
                'storage-tank', 'soccer-ball-field',
                'roundabout', 'harbor',
                'swimming-pool', 'helicopter']
        # 所有的id list，新数据注意更改.
        self._valid_ids = np.arange(1, 16, dtype=np.int32)
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

        print('==> initializing DOTA {} data.'.format(split))
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
                    if len(bbox) == 7:
                        scale = bbox[6]
                        detection["scale"] = float(scale)
                    # if len(bbox) > 5:
                    #     extreme_points = list(map(self._to_float, bbox[5:13]))
                    #     detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections


    def convert_voc_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = defaultdict(list)
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    score = bbox[5]
                    bbox_out = list(map(self._to_float, bbox[0:5]))
                    det_tuple = list()
                    det_tuple.append(int(image_id))
                    det_tuple.append(float("{:.2f}".format(score)))
                    det_tuple.extend(bbox_out)
                    detections[category_id].append(det_tuple)
        return detections

    def convert_from_result_to_voc(self, results):
        detections = defaultdict(list)
        for det in results:
            score = det['score']
            if score >=0.00:
                cat_id = det['category_id']
                image_id = det['image_id']
                rbbox = det['rbbox']
                det_tuple = list()
                det_tuple.append(image_id)
                det_tuple.append(score)
                det_tuple.extend(rbbox)
                detections[cat_id].append(det_tuple)
        return detections

    def convert_from_result_to_eval(self, results):
        all_dets = dict()
        for img_id in self.images:
                all_dets[img_id] = defaultdict(list)
        for det in results:
            score = det['score']
            if score >=0.03:
                cat_id = det['category_id']
                image_id = det['image_id']
                rbbox = det['rbbox']
                det_tuple = list()
                det_tuple.extend(rbbox)
                det_tuple.append(score)
                all_dets[image_id][cat_id].append(det_tuple)
        return all_dets

    def load_gts(self):
        gts = dict()
        for cls_i in self._valid_ids:
            gts[cls_i] = defaultdict(list)
            anns_cls = self.coco.loadAnns(self.coco.getAnnIds(catIds=cls_i))
            for ann in anns_cls:
                img_id = ann['image_id']
                gt_tuple = dict()
                gt_tuple['rbbox'] = ann['rbbox']
                gt_tuple['ignore'] = ann['ignore']
                gt_tuple['iscrowd'] = ann['iscrowd']
                gts[cls_i][img_id].append(gt_tuple)
        return gts

    def show_results(self,image, gts, dets, save_dir, img_name):
        debugger = Debugger(dataset='dota', ipynb=(self.opt.debug == 3),
                            theme='white')
        debugger.add_img(image, img_name)
        for j in dets:
            for bbox in dets[j]:
                if bbox[5] > 0.01:
                    debugger.add_rbbox(bbox[:5], j - 1, bbox[5], img_id=img_name)
        for ann in gts:
            bbox = ann['rbbox']
            cat_id = ann['category_id']
            debugger.add_rbbox(bbox, cat_id-1, 1, img_id=img_name, gt=True)
        save_dir = os.path.join(save_dir, 'voc_results')
        debugger.save_all_imgs(save_dir)
    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))


    def run_eval(self, results, save_dir):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        if results :
            self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "rbbox")
        coco_eval.params.maxDets = [1, 10, 300]
        # coco_eval.params.imgIds = [50]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    def run_voc_eval(self, results, save_dir, year, show_results=None):
        if results :
            self.save_results(results, save_dir)
        with open('{}/results.json'.format(save_dir), 'r') as f:
            results = json.load(f)
        dets = self.convert_from_result_to_voc(results)

        if show_results:
            eval_dets = self.convert_from_result_to_eval(results)
            for img_id in eval_dets:
                gts = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
                image_ann = self.coco.loadImgs(ids=img_id)[0]
                image = cv2.imread(os.path.join(self.img_dir, image_ann['file_name']))
                self.show_results(image, gts, eval_dets[img_id], save_dir, image_ann['file_name'])
        gts = self.load_gts()
        aps = []

        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        for i, cls in enumerate(self.class_name):
            if cls == '__background__':
                continue
            rec, prec, ap = self.voc_eval(gts[i], dets[i], ovthresh=0.5,
                use_07_metric=use_07_metric, use_diff=False)
            aps += [ap]
            print(('AP for {} = {:.4f}'.format(cls, ap)))
        print(('Mean AP = {:.4f}'.format(np.mean(aps))))
        print('~~~~~~~~')

    def voc_eval(self,
                 gts,
                 dets,
                 ovthresh=0.5,
                 use_07_metric=False,
                 use_diff=False):
        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for image_id in self.images:
            R = [obj for obj in gts[image_id]]
            bbox = np.array([x['rbbox'] for x in R])
            if use_diff:
                difficult = np.array([False for _ in R]).astype(np.bool)
            else:
                difficult = np.array([x['ignore'] for x in R]).astype(np.bool)
            iscrowd = np.array([x['iscrowd'] for x in R]).astype(np.int)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[image_id] = {'rbbox': bbox,
                                    'difficult': difficult,
                                    'det': det,
                                    'iscrowd': iscrowd}
        # dets
        image_ids = [x[0] for x in dets]
        confidence = np.array([float(x[1]) for x in dets])
        BB = np.array([[z for z in x[2:]] for x in dets])

        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        if BB.shape[0] > 0:
            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            for d in range(nd):
                R = class_recs[image_ids[d]]
                bb = BB[d, :].reshape(-1,5)
                ovmax = -np.inf
                BBGT = R['rbbox']
                iscrowd = R['iscrowd']

                if BBGT.size > 0:
                    # # compute overlaps
                    # # intersection
                    # ixmin = np.maximum(BBGT[:, 0], bb[0])
                    # iymin = np.maximum(BBGT[:, 1], bb[1])
                    # ixmax = np.minimum(BBGT[:, 2], bb[2])
                    # iymax = np.minimum(BBGT[:, 3], bb[3])
                    # iw = np.maximum(ixmax - ixmin + 1., 0.)
                    # ih = np.maximum(iymax - iymin + 1., 0.)
                    # inters = iw * ih
                    #
                    # # union
                    # uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                    #        (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                    #        (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
                    #
                    # overlaps = inters / uni
                    overlaps = maskUtils.riou(bb, BBGT, iscrowd)
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)

        return rec, prec, ap

def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
        # print(t, p)
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap


