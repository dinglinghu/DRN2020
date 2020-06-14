import numpy as np
import os
import json
import _init_paths
from collections import defaultdict
from external.angle_nms.angle_soft_nms import angle_soft_nms, angle_soft_nms_new


def _to_float(x):
    return float("{:.2f}".format(x))

_valid_ids = np.arange(1, 16, dtype=np.int32)
def convert_eval_format(all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
        for cls_ind in all_bboxes[image_id]:
            category_id = _valid_ids[cls_ind - 1]
            for bbox in all_bboxes[image_id][cls_ind]:
                score = bbox[5]
                bbox_out = list(map(_to_float, bbox[0:5]))

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

def convert_from_result_to_voc(results, score_thr):
    all_dets = dict()
    for det in results:
        score = det['score']
        if score >=score_thr:
            cat_id = det['category_id']
            image_id = det['image_id']
            rbbox = det['rbbox']
            det_tuple = list()
            det_tuple.extend(rbbox)
            det_tuple.append(score)
            if 'scale' in det:
                det_tuple.append(det['scale'])
            if image_id not in all_dets:
                all_dets[image_id] = defaultdict(list)
            all_dets[image_id][cat_id].append(det_tuple)
    return all_dets

def merge_result_from_splits(res_dir, save_dir, splits, file_temp, total_result_name,load=True):
    """
    Merge results from multiple splits.
    In each split, the results are in a dictionary:
    res[i][j]: i-image id, j-class id
    args:
        res_dir: path to results
        save_dir: path to save
        splits: tuple or list or int
        file_temp: file name of result
    return:
        total_res: the total results merged from multiple splits
    """
    if load:
        total_res = []
        res_split = np.arange(splits) if isinstance(splits) == int else splits
        for spl in res_split:
            file_name = file_temp.format(spl)
            file_path = os.path.join(res_dir, file_name)
            with open(file_path, 'r') as fp:
                res_json= json.load(fp)
                total_res.extend(res_json)
        save_path = os.path.join(save_dir, total_result_name)
        with open(save_path, 'w') as sp:
            json.dump(total_res, sp)
    else:
        res_path = os.path.join(save_dir,'results_135_6s.json')
        with open(res_path, 'r') as rp:
            total_res = json.load(rp)

    return total_res

def save_results(results, save_dir, final_res_name=None):
    res_name = r'results.json' if final_res_name is None else final_res_name
    json.dump(convert_eval_format(results),
              open(os.path.join(save_dir, res_name), 'w'))

def merge_results(dets, nms_cross_cls=False, adjust_score=True, max_per_image=800, num_classes =15):
    """
    merge results from multiple scales
    :param dets: detection results saved in a two-level dictionary
                 dets[i][j] i-> image id; j-> class id
    :param nms_cross_cls: process dets by using nms across multiple classes
    :param adjust_score: switch for whether to adjust scores according to scale
    :param max_per_image: max number of object in each image\
    :param num_classes: number of classes of dataset
    :return: return detection results after merge
    """
    all_dets = {}
    for image_id in dets:
        class_ids = dets[image_id].keys()
        results_im = {}
        for cls_id in class_ids:
            cnt = len(dets[image_id][cls_id])
            dets_ij = np.array(dets[image_id][cls_id]).reshape(cnt,-1)
            if adjust_score:
              scale = dets_ij[:, -1]
              # ts = 1-scale if scale>=1.0 else 2.0*(1-scale)
              ts = np.array([1-sca if sca>=1.0 else 2.0*(1-sca) for sca in scale])
              ts = ts/10.
              modular = np.power(np.max(dets_ij[:,2:4], axis=1)/96.,ts)
              dets_ij[:,5] *= modular
              dets_ij[:,5] = np.clip(dets_ij[:,5], 0.0, 1.0)
            result_nms = angle_soft_nms(dets_ij, Nt=0.5, method=1, threshold=0.01)
            results_im[cls_id] = result_nms

        if nms_cross_cls:
            for j in range(1, num_classes + 1):
                cls = np.ones((results_im[j].shape[0], 1)) * j
                results_im[j] = np.hstack([results_im[j], cls])
            all_dets = np.vstack(
                [results_im[j] for j in range(1, num_classes + 1)])
            all_dets_nms = angle_soft_nms_new(all_dets, Nt=0.1, method=1, threshold=0.01, all_cls=True, cls_decay=1.8)

            results_im = {}
            for j in range(1, num_classes + 1):
                j_ind = all_dets_nms[:, -1] == j
                results_im[j] = all_dets_nms[j_ind, :-1]
        scores = np.hstack(
            [results_im[j][:, 5] for j in class_ids])
        if len(scores) > max_per_image:
            kth = len(scores) - max_per_image
            thresh = np.partition(scores, kth)[kth]
            thresh = 0.03
            for j in class_ids:
                keep_inds = (results_im[j][:, 5] >= thresh)
                results_im[j] = results_im[j][keep_inds]
        all_dets[image_id] = results_im
    return all_dets

def post_process(res_dir, save_dir, splits, file_temp, total_res_name,final_res_name):
    total_res = merge_result_from_splits(res_dir, save_dir, splits, file_temp, total_res_name, load=False)
    eval_res = convert_from_result_to_voc(total_res, score_thr=0.03)
    final_res = merge_results(eval_res, nms_cross_cls=False,adjust_score=True)
    save_results(final_res, save_dir, final_res_name)

if __name__ == '__main__':
    split_dir = r'../exp/rodet'
    file_temp = r'dota_hg_drn_tv{}/result.json'
    save_dir = r'../exp/rodet/dota_hg_drn_yun/'
    total_res_name ='model_135_total.json'
    final_res_name ='results_135_ads_nms_5_01_top03.json'
    post_process(split_dir,save_dir,8,file_temp, total_res_name, final_res_name)