import argparse
import random
from matplotlib.pyplot import axis

import torch
import torch.backends.cudnn as cudnn

from utils.gpu_util import set_gpu
from utils.parse_util import parse_yaml
from datasets.lymph_nodes_feat_datasets_for_graph import FeatDataset
from torch.utils.data.dataloader import DataLoader
from nets.feat_resnet3d import resnet34
from tqdm import tqdm
import os
import torch.nn as nn
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch LymphNode Classifier')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed for training. default=42')
parser.add_argument('--use_cuda', default='true', type=str,
                    help='whether use cuda. default: true')
parser.add_argument('--use_parallel', default='false', type=str,
                    help='whether use parallel. default: false')
parser.add_argument('--gpu', default='all', type=str,
                    help='use gpu device. default: all')
parser.add_argument('--model', default='cnn', type=str,
                    choices=['cnn', 'gnn', 'reid'], help='use model. default: reid')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint. default: none')
parser.add_argument('--last_epoch', default=-1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts).')
parser.add_argument('--config', default='cfgs/classifier.yaml', type=str,
                    help='configuration file. default=cfgs/classifier.yaml')
parser.add_argument('--test', default='false', type=str,
                    help='test or not. default: false')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def load(ckpt_path, config, net, ignore_list=None, display=True):
    if ignore_list is None:
        ignore_list = []
    network_params = config['network']
    # note: 用来加载模型
    ckpt = torch.load(ckpt_path, map_location=torch.device(config['network']['device']))
    state_dict = ckpt['state_dict']
    if network_params['use_parallel']:
        model_dict = net.module.state_dict()
    else:
        model_dict = net.state_dict()
    for _key in list(state_dict.keys()):
        key = _key.replace('module.', '')
        res = True
        for rule in ignore_list:
            if key.startswith(rule):
                res = False
                break
        if res:
            if key in model_dict.keys():
                if display:
                    print("Loading parameter {}".format(key))
                model_dict[key] = state_dict[_key]

    if network_params['use_parallel']:
        net.module.load_state_dict(model_dict)
    else:
        net.load_state_dict(model_dict)
    print(">>> Loading model successfully from {}.".format(ckpt_path))

def main():
    args = parser.parse_args()
    num_gpus = set_gpu(args.gpu)
    # set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    config = parse_yaml(args.config)
    network_cfgs = config['network']
    network_cfgs['seed'] = args.seed if network_cfgs['seed'] != args.seed else network_cfgs['seed']
    network_cfgs['device'] = "cuda" if str2bool(args.use_cuda) else "cpu"
    network_cfgs['use_parallel'] = str2bool(args.use_parallel)
    network_cfgs['num_gpus'] = num_gpus
    if num_gpus > 1:
        network_cfgs['use_parallel'] = True
    config['network'] = network_cfgs

    train_cfgs = config['train']
    train_cfgs['batch_size'] = train_cfgs['batch_size'] * num_gpus
    train_cfgs['num_workers'] = train_cfgs['num_workers'] * num_gpus
    config['train'] = train_cfgs

    data_params = config['data']
    test_params = config['test']
    device = torch.device(network_cfgs['device'])
    # dataset
    all_dataset = FeatDataset(
        records_path=config['data']['patient_records_path'],
        config=data_params,
        phase='test')
    
    print('batch_size:', test_params['batch_size'])
    all_loader = DataLoader(
        all_dataset,
        batch_size=test_params['batch_size'],
        shuffle=False,
        num_workers=test_params['num_workers'],
        pin_memory=False)

    print("[Info] The size of training dataset: {}".format(len(all_dataset)))
    # model
    net = resnet34(color_channels=data_params['color_channels'], num_classes=data_params['num_classes'])
    if network_cfgs['use_parallel']:
        # pytorch并行计算
        net = nn.DataParallel(net)
    net = net.to(device)
    # 加载模型参数
    logging_params = config['logging']
    # ckpt_path = os.path.join(logging_params['ckpt_path'], logging_params['logging_dir'])
    # model_path = ckpt_path + '/best_auc.pth'
    model_path = '/root/workspace/lymph_node_classification/ckpts/0325/resnet_del_inappropriate_data_randomsample_large_crop_weightce_feature_metric_revise_baseline/fpnresnet_del_inappropriate_data_randomsample_large_crop_weightce_feature_metric_revise_baseline/best_auc.pth'
    load(model_path, config, net)
    net.eval()
    # feat = model(input)
    all_feat_record = np.empty([0,512])
    with torch.no_grad():
        with tqdm(total=len(all_loader)) as pbar:
            for batch_id, sample in enumerate(all_loader):
                img = sample['img'].to(device)
                case_name = np.array(sample['case_name'])
                uuid = np.array(sample['uuid'])
                key = np.array(sample['key'])
               
                # note: position embedding
                # t_coord = sample['t_coord'].to(device)
                # s_coord = sample['s_coord'].to(device)

                feat = net(img, is_feat=True)
                # 记录格式 case_name uuid key feat
                # np.hstack((np.dstack((case_name, uuid, key)).squeeze(),feat.cpu().numpy()))
                # one_batch_feat_record = np.hstack((np.dstack((case_name, uuid, key)).squeeze(),feat.cpu().numpy()))
                all_feat_record = np.append(all_feat_record, feat.cpu().numpy(), axis=0)

                pbar.update(1)
                pbar.set_description("[Get Feat] Batch:{}".format(batch_id))
                
                # import pdb;
                # pdb.set_trace()
            print(all_feat_record.shape)
            np.save('/root/workspace/data/feat_gnn/test_feat2.npy', all_feat_record)

if __name__ == '__main__':
    main()