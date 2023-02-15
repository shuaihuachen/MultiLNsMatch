"""
Created by Wang Han on 2019/10/23 22:19.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2019 Wang Han. SCU. All Rights Reserved.
"""
import numpy as np

from .nms import nms, iou


def accuracy(pbb, lbb, conf_th, nms_th, detect_th):
    pbb = pbb[pbb[:, 0] >= conf_th]
    pbb = nms(pbb, nms_th)

    tp = []
    fp = []
    fn = []
    l_flag = np.zeros((len(lbb),), np.int32)
    for p in pbb:
        flag = 0
        bestscore = 0
        for i, l in enumerate(lbb):
            score = iou(p[1:5], l)
            if score > bestscore:
                bestscore = score
                besti = i
        if bestscore > detect_th:
            flag = 1
            if l_flag[besti] == 0:
                l_flag[besti] = 1
                tp.append(np.concatenate([p, [bestscore]], 0))
            else:
                fp.append(np.concatenate([p, [bestscore]], 0))
        if flag == 0:
            fp.append(np.concatenate([p, [bestscore]], 0))

    for i, l in enumerate(lbb):
        if l_flag[i] == 0:
            score = []
            for p in pbb:
                score.append(iou(p[1:5], l))
            if len(score) != 0:
                bestscore = np.max(score)
            else:
                bestscore = 0

            if bestscore < detect_th:
                fn.append(np.concatenate([l, [bestscore]], 0))

    return tp, fp, fn, len(lbb)
