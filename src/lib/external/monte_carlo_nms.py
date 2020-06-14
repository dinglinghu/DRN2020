# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# written by yichaoxiong
# 2019.02.02

import numpy as np
# from shapely import affinity
# from shapely.geometry import Polygon
# import pdb

import torch
import time

pi = np.pi

def bbox_vector(theta):
    """
    theta size:m
    """
    theta_x = theta + pi / 2 + 3 * pi / 2
    theta_y = theta - pi + 3 * pi / 2
    cos_x = torch.unsqueeze(torch.cos(theta_x), 1)
    cos_y = torch.unsqueeze(torch.cos(theta_y), 1)
    sin_x = torch.unsqueeze(torch.sin(theta_x), 1)
    sin_y = torch.unsqueeze(torch.sin(theta_y), 1)
    vector = torch.cat((cos_x, sin_x, cos_y, sin_y), 1)
    return vector


def convert_bbox(boxes):
    """
    boxes [n,5]
    """
    theta = boxes[:, 4]
    wh = boxes[:, 2:4]
    center = boxes[:, :2]
    size = boxes.size(0)
    vector = bbox_vector(theta)

    # X = tf.cat((center_x,torch.unsqueeze(wh[:,0] * vector[:,0]),torch.unsqueeze(wh[:,1] * vector[:,2])),1)

    px1 = torch.unsqueeze(center[:, 0] - 0.5 * wh[:, 0] * vector[:, 0] - 0.5 * wh[:, 1] * vector[:, 2], 1)
    px2 = torch.unsqueeze(center[:, 0] + 0.5 * wh[:, 0] * vector[:, 0] - 0.5 * wh[:, 1] * vector[:, 2], 1)
    px3 = torch.unsqueeze(center[:, 0] + 0.5 * wh[:, 0] * vector[:, 0] + 0.5 * wh[:, 1] * vector[:, 2], 1)
    px4 = torch.unsqueeze(center[:, 0] - 0.5 * wh[:, 0] * vector[:, 0] + 0.5 * wh[:, 1] * vector[:, 2], 1)
    py1 = torch.unsqueeze(center[:, 1] - 0.5 * wh[:, 0] * vector[:, 1] - 0.5 * wh[:, 1] * vector[:, 3], 1)
    py2 = torch.unsqueeze(center[:, 1] + 0.5 * wh[:, 0] * vector[:, 1] - 0.5 * wh[:, 1] * vector[:, 3], 1)
    py3 = torch.unsqueeze(center[:, 1] + 0.5 * wh[:, 0] * vector[:, 1] + 0.5 * wh[:, 1] * vector[:, 3], 1)
    py4 = torch.unsqueeze(center[:, 1] - 0.5 * wh[:, 0] * vector[:, 1] + 0.5 * wh[:, 1] * vector[:, 3], 1)

    bbox = torch.cat((px1, py1, px2, py2, px3, py3, px4, py4), 1)
    return bbox


def point_in_boxes(pts, boxes):
    """
    pts [m,k,2]
    boxes [n,8]
    return [m,n,k]

    """
    size_p = pts.size()
    size_b = boxes.size()
    pts = torch.unsqueeze(pts, 1)
    pts = pts.repeat(1, size_b[0], 1, 4)
    boxes = torch.unsqueeze(boxes, 0)
    boxes = torch.unsqueeze(boxes, 2)
    boxes = boxes.repeat(size_p[0], 1, size_p[1], 1)
    q = boxes - pts
    q1 = torch.cat((torch.unsqueeze(q[:, :, :, 3], 3), torch.unsqueeze(q[:, :, :, 2], 3),
                    torch.unsqueeze(q[:, :, :, 5], 3), torch.unsqueeze(q[:, :, :, 4], 3),
                    torch.unsqueeze(q[:, :, :, 7], 3), torch.unsqueeze(q[:, :, :, 6], 3),
                    torch.unsqueeze(q[:, :, :, 1], 3), torch.unsqueeze(q[:, :, :, 0], 3)), 3)

    q2 = q * q1
    m1 = q2[:, :, :, 0] - q2[:, :, :, 1]
    m2 = q2[:, :, :, 2] - q2[:, :, :, 3]
    m3 = q2[:, :, :, 4] - q2[:, :, :, 5]
    m4 = q2[:, :, :, 6] - q2[:, :, :, 7]
    m = torch.cat((torch.unsqueeze(m1, 3), torch.unsqueeze(m2, 3), torch.unsqueeze(m3, 3), torch.unsqueeze(m4, 3)), 3)
    t = (m > 0).float()
    u = t[:, :, :, 0] * t[:, :, :, 1] * t[:, :, :, 2] * t[:, :, :, 3]
    return u


def bbox_ious(boxes1, boxes2, num=500):
    b1theta = boxes1[:, 4]
    b1wh = boxes1[:, 2:4]
    b1center = boxes1[:, :2]
    b1_len = boxes1.size(0)
    b1vector = bbox_vector(b1theta)
    b2wh = boxes2[:, 2:4]
    b2_len = boxes2.size(0)

    lamda = (torch.rand(b1_len, 2 * num) - 0.5).cuda()
    b1_center_x = torch.unsqueeze(b1center[:, 0], 1)
    b1_center_y = torch.unsqueeze(b1center[:, 1], 1)
    px = b1_center_x.repeat(1, num) + lamda[:, :num] * torch.unsqueeze(b1wh[:, 0] * b1vector[:, 0], 1).repeat(1, num) + \
         lamda[:, num:2 * num] * torch.unsqueeze(b1wh[:, 1] * b1vector[:, 2], 1).repeat(1, num)
    py = b1_center_y.repeat(1, num) + lamda[:, :num] * torch.unsqueeze(b1wh[:, 0] * b1vector[:, 1], 1).repeat(1, num) + \
         lamda[:, num:2 * num] * torch.unsqueeze(b1wh[:, 1] * b1vector[:, 3], 1).repeat(1, num)
    px = torch.unsqueeze(px, 2)
    py = torch.unsqueeze(py, 2)
    pts = torch.cat((px, py), 2)

    bbox = convert_bbox(boxes2)
    u = point_in_boxes(pts, bbox)
    s = torch.sum(u, 2)

    areas1 = torch.unsqueeze(b1wh[:, 0] * b1wh[:, 1], 1)
    areas2 = torch.unsqueeze(b2wh[:, 0] * b2wh[:, 1], 0)
    areas1 = areas1.repeat(1, b2_len)
    areas2 = areas2.repeat(b1_len, 1)
    p = s / num * areas1
    iou = p / (areas1 + areas2 - p)
    return iou

def mtcarlo_soft_nms(all_dets, thresh):
    """Pure Python NMS baseline."""
    # dets = np.concatenate((all_dets[:, 0:4], all_dets[:, -1:]), axis=1)
    # scores = all_dets[:, 4]
    # cx,cy,w,h,angle,score
    boxes = all_dets
    # scores = all_dets[:, -1]
    N = all_dets.shape[0]

    for i in range(N):
        maxscore = boxes[i, 5]
        maxpos = i
        # 将第i个bbox存在temp
        tcx = boxes[i, 0]
        tcy = boxes[i, 1]
        tw = boxes[i, 2]
        th = boxes[i, 3]
        tangle = boxes[i, 4]
        ts= boxes[i, 5]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 5]:
                maxscore = boxes[pos, 5]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection
        boxes[i, 0] = boxes[maxpos, 0]
        boxes[i, 1] = boxes[maxpos, 1]
        boxes[i, 2] = boxes[maxpos, 2]
        boxes[i, 3] = boxes[maxpos, 3]
        boxes[i, 4] = boxes[maxpos, 4]
        boxes[i, 5] = boxes[maxpos, 5]

        # swap ith box with position of max box
        boxes[maxpos, 0] = tcx
        boxes[maxpos, 1] = tcy
        boxes[maxpos, 2] = tw
        boxes[maxpos, 3] = th
        boxes[maxpos, 4] = tangle
        boxes[maxpos, 5] = ts

        # 此时第i个位最大score的，重新将第i个bbox存在temp
        tcx = boxes[i, 0]
        tcy = boxes[i, 1]
        tw = boxes[i, 2]
        th = boxes[i, 3]
        tangle = boxes[i, 4]
        ts= boxes[i, 5]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:

            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua  # iou between max box and detection box

                    if method == 1:  # linear
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2:  # gaussian
                        weight = np.exp(-(ov * ov) / sigma)
                    else:  # original NMS
                        if ov > Nt:
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight * boxes[pos, 4]

                    # if box score falls below threshold, discard the box by swapping with last box
                    # update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos, 0] = boxes[N - 1, 0]
                        boxes[pos, 1] = boxes[N - 1, 1]
                        boxes[pos, 2] = boxes[N - 1, 2]
                        boxes[pos, 3] = boxes[N - 1, 3]
                        boxes[pos, 4] = boxes[N - 1, 4]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    keep = [i for i in range(N)]
    return keep


    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            return keep
        cur_box = torch.tensor([dets[i, :] ]).cuda()
        other_boxes = torch.tensor(dets[order[1:], :]).cuda()
        ovr = bbox_ious(cur_box, other_boxes )
        inds = (ovr <= thresh).nonzero()
        order = order[inds[:, 1].cpu().numpy() + 1]

    return keep

def mtcarlo_nms(all_dets, thresh):
    """Pure Python NMS baseline."""
    # dets = np.concatenate((all_dets[:, 0:4], all_dets[:, -1:]), axis=1)
    # scores = all_dets[:, 4]
    dets = all_dets[:, 0:5]
    scores = all_dets[:, -1]
    # dets[:, 4] = pi * (dets[:, 4] - 0.5)
    # dets[:, 4] = pi * (dets[:, 4] - 0.5)
    #print(x1, y1, x2, y2)
    #areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            return keep
        cur_box = torch.tensor([dets[i, :] ]).cuda()
        other_boxes = torch.tensor(dets[order[1:], :]).cuda()
        ovr = bbox_ious(cur_box, other_boxes )
        inds = (ovr <= thresh).nonzero()
        order = order[inds[:, 1].cpu().numpy() + 1]

    return keep



if __name__ == '__main__':
    # a = np.array([(0, 0, 30/1000, 10/1000, .9, 0), (0, 0, 30/1000, 10/1000, .98, 0.25)], [(-5, -5, 5, 5, .98, 45), (-5, -5, 6, 6, .99, 30)])
    #print(py_cpu_nms(a, 0.45))
    # print(py_cw_nms(a, 0.45))
    #print(Polygon(a).area)
    boxes1 = torch.tensor([[20, 20, 5, 10, 0 * pi]])
    boxes1 = boxes1.repeat((100, 1)).cuda()
    boxes2 = torch.tensor([[20, 20, 5, 10, 0.5 * pi]])
    boxes2 = boxes2.repeat((100, 1)).cuda()
    start = time.time()
    iou = bbox_ious(boxes1, boxes2)
    end = time.time()
    print(iou)
    print(end - start)
