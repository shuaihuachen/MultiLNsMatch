"""
Created by Wang Han on 2019/10/23 22:23.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2019 Wang Han. SCU. All Rights Reserved.
"""

import argparse
import random

import torch
import torch.backends.cudnn as cudnn

from utils.gpu_util import set_gpu
from utils.parse_util import parse_yaml

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

    if args.model == 'cnn':
        from models.classifier_model import Model
    elif args.model == 'gnn':
        from models.graph_classifier_model import Model
    elif args.model =='reid':
        from models.reid_model import Model

    model = Model(config)
    if str2bool(args.test):
        model.test()
    else:
        model.run()


if __name__ == '__main__':
    main()
