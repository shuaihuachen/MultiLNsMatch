"""
Created by Wang Han on 2019/11/19 15:47.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2019 Wang Han. SCU. All Rights Reserved.
"""

import os
import warnings
from collections import namedtuple

import cv2
import numpy as np
import torch
from scipy.ndimage.interpolation import rotate, zoom
from torch.utils.data import Dataset

warnings.filterwarnings('ignore', '.*output shape of zoom.*')


class ClassifierDataset(Dataset):
    def __init__(self, records_path, config, phase='train', load_all_ln=False):
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        # note: 不必再通过索引值访问tuple 可以看作一个字典通过名字进行访问 但其中的值是不能改变的
        # Item = namedtuple('Item', ['case_name', 'image_path', 'box', 'partition', 'is_nodule', 'left_or_right'])
        Item = namedtuple('Item', ['case_name', 'image_path', 'box', 'partition', 'scaling_binary_lable'])
        self.crop_size = config['crop_size']
        self.stride = config['stride']
        self.augtype = config['augtype']
        self.blacklist = config['blacklist']
        self.phase = phase
        self.items = []

        # data_root = config['data_root']
        data_root = config['processed_data_root']
        # 加载数据名称
        records = np.load(records_path)
        if phase != 'test':
            records = [f for f in records if (f not in self.blacklist)]
        for idx, record in enumerate(records):
            filename = os.path.join(data_root, '{}_clean.npy'.format(record))
            # get [z, y, x, d, p, is_positive， left_or_right]
            # get [x, y, z, d, p, scaling]
            labels = np.load(os.path.join(data_root, '{}_label.npy'.format(record)), allow_pickle=True)
            # if not load_all_ln:
            #     # 筛选带有阴阳性标注的标签
            #     labels = labels[~np.isnan(labels[:, 5])]
            #     for label in labels:
            #         left_or_right = 0 if np.isnan(label[6]) else label[6]
            #         item = Item(
            #             case_name=record, image_path=filename, box=label[0:4], partition=label[4], is_nodule=label[5],
            #             left_or_right=left_or_right)
            #         self.items.append(item)
            # else:
            #     # 排除分区为空的淋巴结
            #     labels = labels[~np.isnan(labels[:, 4])]
            #     # 排除分区为歧义候选和结直肠系膜的淋巴结
            #     labels = labels[(labels[:, 4] != 5) & (labels[:, 4] != 6)]
            #     for label in labels:
            #         # 如果未标注阴阳性，视为阴淋巴结
            #         is_nodule = 0 if np.isnan(label[5]) else label[5]
            #         left_or_right = 0 if np.isnan(label[6]) else label[6]
            #         item = Item(
            #             case_name=record, image_path=filename, box=label[0:4], partition=label[4], is_nodule=is_nodule,
            #             left_or_right=left_or_right)
            #         self.items.append(item)
            if not load_all_ln:
                for label in labels:
                    item = Item(
                        case_name=record, image_path=filename, box=label[0:4], partition=label[4], scaling_binary_lable=label[6])
                    self.items.append(item)
            else:
                # 排除分区为空的淋巴结
                labels = labels[~np.isnan(labels[:, 4])]
                # 排除分区为歧义候选和结直肠系膜的淋巴结
                labels = labels[(labels[:, 4] != 21) & (labels[:, 4] < 14) & (labels[:, 4] > 19)]
                for label in labels:
                    item = Item(
                        case_name=record, image_path=filename, box=label[0:4], partition=label[4], scaling_binary_lable=label[6])
                    self.items.append(item)

        self.crop = Crop(config, phase)

    def __getitem__(self, idx):
        if self.phase != 'test':
            item = self.items[idx]
            img = np.load(item.image_path)
            target = item.box
            # is_nodule = int(item.is_nodule)
            scaling_binary_lable = int(item.scaling_binary_lable)
            crop_img, mask_img = self.crop(img, target)
            input = np.concatenate([crop_img, mask_img], axis=0)
            input = (input.astype(np.float32) - 128) / 128
            if self.phase == 'train':
                input = augment(input,
                                is_flip=self.augtype['flip'], is_rotate=self.augtype['rotate'],
                                is_swap=self.augtype['swap'])

            # 处理掩码经过增广后的非整数值
            input[1] = np.round(input[1])
            sample = {
                'image': torch.from_numpy(input).float(),
                'target': scaling_binary_lable
                # 'target': is_nodule
            }
            return sample
        else:
            item = self.items[idx]
            img = np.load(item.image_path)
            target = item.box
            # is_nodule = int(item.is_nodule)
            scaling_binary_lable = int(item.scaling_binary_lable)
            partition = int(item.partition)
            # left_or_right = int(item.left_or_right)
            case_name = item.case_name
            crop_img, mask_img = self.crop(img, target)
            input = np.concatenate([crop_img, mask_img], axis=0)
            input = (input.astype(np.float32) - 128) / 128
            # 处理掩码经过增广后的非整数值
            input[1] = np.round(input[1])
            sample = {
                'image': torch.from_numpy(input).float(),
                'target': scaling_binary_lable,
                # 'target': is_nodule,
                'box': target,
                'partition': partition,
                'case_name': case_name
                # 'left_or_right': left_or_right
            }
            return sample

    def __len__(self):
        return len(self.items)


class Crop():
    def __init__(self, config, phase):
        self.crop_size = config['crop_size']
        self.scaleLim = config['scaleLim']
        self.radiusLim = config['radiusLim']
        self.jitter_range = config['jitter_range']
        self.isScale = config['augtype']['scale'] and phase == 'train'
        self.stride = config['stride']
        self.filling_value = config['filling_value']
        self.phase = phase

    def __call__(self, imgs, target):
        if self.isScale:
            radiusLim = self.radiusLim
            scaleLim = self.scaleLim
            scaleRange = [np.min([np.max([(radiusLim[0] / target[3]), scaleLim[0]]), 1]),
                          np.max([np.min([(radiusLim[1] / target[3]), scaleLim[1]]), 1])]
            scale = np.random.rand() * (scaleRange[1] - scaleRange[0]) + scaleRange[0]
            crop_size = (np.array(self.crop_size).astype('float') / scale).astype('int')
        else:
            crop_size = np.array(self.crop_size).astype('int')
        if self.phase == 'train':
            jitter_range = target[3] * self.jitter_range
            jitter = (np.random.rand(3) - 0.5) * jitter_range
        else:
            jitter = 0
        start = (target[:3] - crop_size / 2 + jitter).astype('int')
        pad = [[0, 0]]
        for i in range(3):
            if start[i] < 0:
                leftpad = -start[i]
                start[i] = 0
            else:
                leftpad = 0
            if start[i] + crop_size[i] > imgs.shape[i + 1]:
                rightpad = start[i] + crop_size[i] - imgs.shape[i + 1]
            else:
                rightpad = 0
            pad.append([leftpad, rightpad])
        imgs = np.pad(imgs, pad, 'constant', constant_values=self.filling_value)
        crop = imgs[:,
               start[0]:start[0] + crop_size[0],
               start[1]:start[1] + crop_size[1],
               start[2]:start[2] + crop_size[2]]
        # generate target mask image
        new_target = target.copy()
        new_target[0:3] = target[0:3] - start
        target_img = generate_mask(new_target, crop_size)

        if self.isScale:
            crop = zoom(crop, [1, scale, scale, scale], order=1)
            target_img = zoom(target_img, [1, scale, scale, scale], order=1)
            newpad = self.crop_size[0] - crop.shape[1:][0]
            if newpad < 0:
                crop = crop[:, :-newpad, :-newpad, :-newpad]
                target_img = target_img[:, :-newpad, :-newpad, :-newpad]
            elif newpad > 0:
                pad2 = [[0, 0], [0, newpad], [0, newpad], [0, newpad]]
                crop = np.pad(crop, pad2, 'constant', constant_values=self.filling_value)
                target_img = np.pad(target_img, pad2, 'constant', constant_values=self.filling_value)

        return crop, target_img


def augment(sample, is_flip=True, is_rotate=True, is_swap=True):
    if is_rotate:
        angle = np.random.rand() * 180
        sample = rotate(sample, angle, axes=(2, 3), reshape=False)
    if is_swap:
        if sample.shape[1] == sample.shape[2] and sample.shape[1] == sample.shape[3]:
            # note: np.random.permutation生成随机序列
            axisorder = np.random.permutation(3)
            # note: np.transpose 默认情况下，反转维度，否则根据给定的值对轴进行排列
            sample = np.transpose(sample, np.concatenate([[0], axisorder + 1]))

    if is_flip:
        flipid = np.array([np.random.randint(2), np.random.randint(2), np.random.randint(2)]) * 2 - 1
        # note: 返回和传入的数组类似的内存中连续的数组
        sample = np.ascontiguousarray(sample[:, ::flipid[0], ::flipid[1], ::flipid[2]])
    return sample


def generate_mask(target, crop_size, r_margin=0):
    z, y, x, d = target
    # 淋巴结直径
    r = target[-1] / 2 + r_margin
    # 淋巴结z轴的范围
    lim = list(range(int(np.round(z - r)), int(np.round(z + r))))
    mask = np.zeros(crop_size, dtype='uint8')

    for idx in range(mask.shape[0]):
        if idx in lim:
            center = (int(round(x)), int(round(y)))
            if idx == z:
                radius = int(np.round(r))
            else:
                radius_content = np.round(r) ** 2 - np.abs(z - idx) ** 2
                if radius_content < 0:
                    radius_content = 0
                radius = int(np.sqrt(radius_content))
            mask[idx] = cv2.circle(mask[idx], center, radius, (255, 255, 255), -1)

    # mask[mask == 255] = 1
    return mask[np.newaxis, ...]
