# TODO: 构造输入孪生网络中的数据集
# note: 以t2时间为模版构造淋巴结对
from torch.utils.data.dataset import Dataset
from collections import namedtuple
import numpy as np
import os
import torch
from scipy.ndimage.interpolation import rotate, zoom
import random

class SiameseDataset(Dataset):
    def __init__(self, records_path, config, phase='train', load_all_ln=True, position_eb=False):
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        Item = namedtuple('Item', ['case_name', 'image_path', 'box', 'partition', 'key', 'uuid'])
        self.crop_size = config['crop_size']
        self.stride = config['stride']
        self.augtype = config['augtype']
        self.phase = phase
        # TODO: template_items -> template
        self.template_items = []

        data_root = config['processed_data_root']
        self.data_root = config['processed_data_root']

        # note: new position embedding method
        new_label_root = config['new_processed_label_root']
        self.new_label_root = config['new_processed_label_root']



        # TODO: position embedding
        self.position_eb = position_eb

        # note: 加载数据名称
        # records_path组织形式：PatientID_StudyDate(template):PatientID_StudyDate(searching)
        records = np.load(records_path)

        self.t1_items = []
        self.t2_items = []

        if self.phase != 'test':
            for idx, record in enumerate(records):
                # note: 模版数据 以t2时间为模版
                template_record = record.split(':')[1]
                template_filename = os.path.join(data_root, '{}_clean.npy'.format(template_record))
                if position_eb:
                    template_labels = np.load(os.path.join(new_label_root, '{}_label.npy'.format(template_record)), allow_pickle=True)
                else:
                    template_labels = np.load(os.path.join(data_root, '{}_label.npy'.format(template_record)), allow_pickle=True)

        
                if load_all_ln:
                    #TODO: 组织输入孪生网络的数据 template searching
                    # note: new position embedding method
                    if position_eb:
                        Item = namedtuple('Item', ['case_name', 'image_path', 'box', 'partition', 'key', 'uuid', 'spatial_prior'])
                        for label in template_labels:
                            item1 = Item(
                                case_name=record, image_path=template_filename, box=label[0:4], partition=label[4], key=label[5], uuid=label[6], spatial_prior= label[7:])
                            self.template_items.append(item1)
                    else:
                        for label in template_labels:
                            #TODO: item1 -> template
                            #TODO: item2 -> searching
                            item1 = Item(
                                case_name=record, image_path=template_filename, box=label[0:4], partition=label[4], key=label[5], uuid=label[6])
                            self.template_items.append(item1)
            
                #TODO: 排除某些淋巴结 
                #note: 只加载LLN
                else:
                    for label in template_labels:
                        if label[4] < 14 :
                            item1 = Item(
                                case_name=record, image_path=template_filename, box=label[0:4], partition=label[4], key=label[5], uuid=label[6])
                            self.template_items.append(item1)
        else:
            if position_eb:
                Item = namedtuple('Item', ['case_name', 'image_path', 'box', 'partition', 'key', 'uuid', 'spatial_prior'])
                test_t1_all_label_record = np.load('/root/workspace/data/position_embedding_anchor_point/processed_test_npy/test_t1_all_label_record.npy', allow_pickle=True)
                test_t2_all_label_record = np.load('/root/workspace/data/position_embedding_anchor_point/processed_test_npy/test_t2_all_label_record.npy', allow_pickle=True)
                for record1 in test_t1_all_label_record:
                    image_file_path = os.path.join(data_root, '{}_clean.npy'.format(record1[0]))
                    test_item1 = Item(
                                case_name=record1[0], image_path=image_file_path, box=record1[1:5], partition=record1[5], key=record1[6], uuid=record1[7], spatial_prior= record1[8:])
                    self.t1_items.append(test_item1)
                for record2 in test_t2_all_label_record:
                    image_file_path2 = os.path.join(data_root, '{}_clean.npy'.format(record2[0]))
                    test_item2 = Item(
                                case_name=record2[0], image_path=image_file_path2, box=record2[1:5], partition=record2[5], key=record2[6], uuid=record2[7], spatial_prior= record2[8:])
                    self.t2_items.append(test_item2)
            else:
                test_t1_all_label_record = np.load('/root/workspace/data/related_file_0122/test_t1_all_label_record.npy', allow_pickle=True)
                test_t2_all_label_record = np.load('/root/workspace/data/related_file_0122/test_t2_all_label_record.npy', allow_pickle=True)
                for record1 in test_t1_all_label_record:
                    image_file_path = os.path.join(data_root, '{}_clean.npy'.format(record1[0]))
                    test_item1 = Item(
                                case_name=record1[0], image_path=image_file_path, box=record1[1:5], partition=record1[5], key=record1[6], uuid=record1[7])
                    self.t1_items.append(test_item1)
                for record2 in test_t2_all_label_record:
                    image_file_path2 = os.path.join(data_root, '{}_clean.npy'.format(record2[0]))
                    test_item2 = Item(
                                case_name=record2[0], image_path=image_file_path2, box=record2[1:5], partition=record2[5], key=record2[6], uuid=record2[7])
                    self.t2_items.append(test_item2)

        # print('---init----', len(self.template_items))
        self.crop = Crop(config, phase)
    
    def selectSearchingItem(self, item1, searching_filename, searching_labels):
        selected_label = random.choice(searching_labels)
        if item1.key != '1':
            should_get_same_class = random.randint(0,1) # 0-1随机数
            if should_get_same_class:
                for label in searching_labels:
                    key = label[5]
                    if key == item1.key:
                        selected_label = label
                        break
            else:
                # TODO: 难样本挖掘
                while True:
                    label = random.choice(searching_labels)
                    key = label[5]
                    # print("*",item1.key, key)
                    if key != item1.key:
                        # print("get")
                        selected_label = label
                        break
        # else:
        #     if selected_label[4] >= 14:
        #          while True:
        #             label = random.choice(searching_labels)
        #             key = label[5]
        #             # print("*",item1.key, key)
        #             if key != item1.key and label[4] < 14:
        #                 # print("get")
        #                 selected_label = label
        #                 break

        # note: new position embedding method
        if self.position_eb:
            Item = namedtuple('Item', ['case_name', 'image_path', 'box', 'partition', 'key', 'uuid', 'spatial_prior'])
            item2 = Item(case_name=item1.case_name, image_path=searching_filename, box=selected_label[0:4], partition=selected_label[4], key=selected_label[5], uuid=selected_label[6], spatial_prior=selected_label[7:])
        else:
            Item = namedtuple('Item', ['case_name', 'image_path', 'box', 'partition', 'key', 'uuid'])
            item2 = Item(case_name=item1.case_name, image_path=searching_filename, box=selected_label[0:4], partition=selected_label[4], key=selected_label[5], uuid=selected_label[6])
        return item2


    def __getitem__(self, idx):
        if self.phase != 'test':
            # note: 加载图像数据
            item1 = self.template_items[idx]
            template_img = np.load(item1.image_path)
            # note: searching数据 以t2时间数据为searching
            searching_record = item1.case_name.split(':')[0]
            searching_filename = os.path.join(self.data_root, '{}_clean.npy'.format(searching_record))
            if self.position_eb:
                searching_labels = np.load(os.path.join(self.new_label_root, '{}_label.npy'.format(searching_record)), allow_pickle=True)
            else:
                searching_labels = np.load(os.path.join(self.data_root, '{}_label.npy'.format(searching_record)), allow_pickle=True)
            # TODO: 50%为匹配的淋巴结对 50%为不匹配的淋巴结对
            item2 = self.selectSearchingItem(item1, searching_filename, searching_labels)
            searching_img = np.load(item2.image_path)
        # TODO: 测试阶段的数据集组织 
        else:
            item1 = self.t1_items[idx]
            template_img = np.load(item1.image_path)
           
            item2 = self.t2_items[idx]
            searching_img = np.load(item2.image_path)

        # note: crop淋巴结 
        template_img_crop_target = item1.box
        crop_template_img = self.crop(template_img, template_img_crop_target)
        searching_img_crop_target = item2.box
        crop_searching_img = self.crop(searching_img, searching_img_crop_target)


        # note: 标签 匹配：1 不匹配：0
        # matching_label = torch.from_numpy(np.array([int(item1.key==item2.key)],dtype=np.float32))
        matching_label = int(item1.key==item2.key)
        key1 = item1.key
        key2 = item2.key
        uuid1 = item1.uuid
        uuid2 = item2.uuid
        case_name = item1.case_name.split('_')[0]

        # note: np.concatenate 对0轴的数组对象进行纵向的拼接
        template_input = crop_template_img
        template_input = (template_input.astype(np.float32) - 128) / 128
        searching_input = crop_searching_img
        searching_input = (searching_input.astype(np.float32) - 128) / 128

        if self.phase == 'train':
            template_input = augment(template_input,
                                        is_flip=self.augtype['flip'], is_rotate=self.augtype['rotate'],
                                        is_swap=self.augtype['swap']
            )
            searching_input = augment(searching_input,
                                        is_flip=self.augtype['flip'], is_rotate=self.augtype['rotate'],
                                        is_swap=self.augtype['swap']
            )


        if not self.position_eb:
            sample = {
                'template': torch.from_numpy(template_input).float(),
                'searching': torch.from_numpy(searching_input).float(),
                'uuid1': uuid1,
                'key2': key2,
                'uuid2': uuid2,
                'key1': key1,
                'target': matching_label,
                'case_name': case_name
            }
        else:
            # t_coord = np.delete(template_img_crop_target, -1, axis=0)
            # s_coord = np.delete(searching_img_crop_target, -1, axis=0)
            # note: new postion embedding method
            t_coord = item1.spatial_prior
            s_coord = item2.spatial_prior
            sample = {
                'template': torch.from_numpy(template_input).float(),
                'searching': torch.from_numpy(searching_input).float(),
                'uuid1': uuid1,
                'key2': key2,
                'uuid2': uuid2,
                'key1': key1,
                'target': matching_label,
                't_coord': torch.from_numpy(t_coord.astype(float)).float(),
                's_coord': torch.from_numpy(s_coord.astype(float)).float(),
                'case_name': case_name
            }
        return sample


    def __len__(self):
        if self.phase != 'test':
            return len(self.template_items)
        else:
            assert ( len(self.t1_items) == len(self.t2_items) )
            return len(self.t1_items)

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

        if self.isScale:
            crop = zoom(crop, [1, scale, scale, scale], order=1)
            newpad = self.crop_size[0] - crop.shape[1:][0]
            if newpad < 0:
                crop = crop[:, :-newpad, :-newpad, :-newpad]
            elif newpad > 0:
                pad2 = [[0, 0], [0, newpad], [0, newpad], [0, newpad]]
                crop = np.pad(crop, pad2, 'constant', constant_values=self.filling_value)
                
        return crop


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

