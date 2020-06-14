import numpy as np
import os


results_dir = '../exp/rodet/dota_hg_df_condi/dota_hg_df_condi_flip'
save_dir = '../exp/rodet/dota_hg_df_condi/dota_hg_df_condi_flip_hor'
for cls_txt in os.listdir(results_dir):
    txt_path = os.path.join(results_dir, cls_txt)
    with open(txt_path, 'r') as tp:
        pred_bboxes = tp.readlines()
    hor_bboxes = []
    for bbox in pred_bboxes:
        hor_bbox = []
        pred_bbox = bbox.strip().split()
        hor_bbox.extend(pred_bbox[:2])
        pt_x = pred_bbox[2::2]
        pt_y = pred_bbox[3::2]
        pt_x_ =[float(x) for x in pt_x]
        pt_y_ =[float(y) for y in pt_y]
        hor_bbox.append(str(min(pt_x_)))
        hor_bbox.append(str(min(pt_y_)))
        hor_bbox.append(str(max(pt_x_)))
        hor_bbox.append(str(max(pt_y_)))
        hor_bboxes.append(hor_bbox)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, cls_txt.replace('1','2'))
    with open(save_path, 'w') as sp:
        for hor_bbox in hor_bboxes:
            str_hor_bbox = ' '.join(hor_bbox) + "\n"
            sp.write(str_hor_bbox)
