import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from nets.cores.focalloss import FocalLoss
from nets.cores.ghmloss import GHMC_Loss
from nets.cores.contrastiveloss import ContrastiveLoss
from torch import device, optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.visualization_util import save_confusion_matrix


from datasets.classifier_dataset import ClassifierDataset
# note: 3d
# note: 以t1时间为模版构造淋巴结对
from datasets.siamese_dataset import SiameseDataset
# note: 以t2时间为模版构造淋巴结对
# from datasets.siamese_dataset_revise import SiameseDataset
# note: 2.5d
# from datasets.siamese_25d_dataset import SiameseDataset
# from datasets.siamese_dataset import SiameseDataset
# from datasets.siamese_dataset_new import SiameseDataset
from nets.cores.meter import AverageMeter, ConfusionMeter
# note: 2.5d
# from nets.siamese_resnet2d import resnet34
from nets.siamese_resnet3d import resnet34, resnet50
from nets.feature_pyramid import FeaturePyramidResNet
from utils.log_util import get_logger                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
from utils.parse_util import format_config
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pylab as plt
import gc

# tag: 测试
def saveFPFNLN(pr, gt, uuid1, uuid2):
    # tag: 假阳匹配
    if os.path.exists('/root/workspace/lymph_node_classification/test/fp_pair_1207.npy'):
        fp_uuid_pair = np.load('/root/workspace/lymph_node_classification/test/fp_pair_1207.npy', allow_pickle=True)
        fp_uuid_pair = fp_uuid_pair.tolist()
    else:
        fp_uuid_pair = []
    
    # tag: 假阴匹配
    if os.path.exists('/root/workspace/lymph_node_classification/test/fn_pair_1207.npy'):
        fn_uuid_pair = np.load('/root/workspace/lymph_node_classification/test/fn_pair_1207.npy', allow_pickle=True)
        fn_uuid_pair = fn_uuid_pair.tolist()
    else:
        fn_uuid_pair = []

    for i, pr_i in enumerate(pr.cpu().numpy().tolist()):
        uuid_pair = uuid1[i] + ':' + uuid2[i]
        if gt[i] == 0 and pr_i == 1:
            fp_uuid_pair.append(uuid_pair)
        if gt[i] == 1 and pr_i == 0:
            fn_uuid_pair.append(uuid_pair)
    
    np.save('/root/workspace/lymph_node_classification/test/fp_pair_1207.npy', fp_uuid_pair)
    np.save('/root/workspace/lymph_node_classification/test/fn_pair_1207.npy', fn_uuid_pair)

# TODO: 排除一对多的情况
def metrics_by_max_similarity(eval_pair):
    pr_list = list(eval_pair.groupby('t1_uuid')['1'].transform('max'))
    # pr_list2 = list(eval_pair.groupby('t2_uuid')['1'].transform('max'))
    t2_uuid_record = []
    for idx, row in eval_pair.iterrows():
        if pr_list[idx] >= 0.5 and row['1'] == pr_list[idx]:
            pr_list[idx] = 1
            if row['t2_uuid'] not in t2_uuid_record:
                t2_uuid_record.append(row['t2_uuid'])
        else:
            pr_list[idx] = 0

    eval_pair['pr1'] = pr_list
    for t2_uuid in t2_uuid_record:
        p_record = eval_pair[(eval_pair.t2_uuid == t2_uuid) & (eval_pair.pr1 == 1)].index.tolist()
        if p_record != None and len(p_record) > 1:
            score = []
            for index in p_record:
                score.append(eval_pair.iloc[index, :]['1'])
            max_score_index = score.index(max(score))
            for i,index in enumerate(p_record):
                if i != max_score_index:
                    pr_list[index] = 0

    gt_list = eval_pair.loc[:,['gt']]['gt'].tolist()
    gt_tensor = torch.Tensor(gt_list)
    pr_tensor = torch.Tensor(pr_list)

    # 混淆矩阵
    conf = torch.zeros(2, 2)
    for gt_i in range(2):
            for pr_i in range(2):
                num = (gt_tensor == gt_i) * (pr_tensor == pr_i)
                conf[gt_i][pr_i] += num.sum()
    
    return conf


class Model:
    def __init__(self, config):
        self.config = config
        # loading parameters
        network_params = config['network']
        data_params = config['data']
        optim_params = config['optim']
        logging_params = config['logging']

        self.device = torch.device(network_params['device'])
        # self.net = ClassifierNet(color_channels=data_params['color_channels'], num_classes=data_params['num_classes'])
        # self.net = resnet18(color_channels=data_params['color_channels'], num_classes=data_params['num_classes'])
        # note: 2.5d
        # self.net = resnet34(num_classes=data_params['num_classes'])
        # note: 3d
        # self.net = resnet34(color_channels=data_params['color_channels'], num_classes=data_params['num_classes'])
        # note: 特征金字塔
        self.net = FeaturePyramidResNet(color_channels=data_params['color_channels'], num_classes=data_params['num_classes'])
        # self.net = resnet50(color_channels=data_params['color_channels'], num_classes=data_params['num_classes'])
        if network_params['use_parallel']:
            # pytorch并行计算
            self.net = nn.DataParallel(self.net)
        # note: .to(device)可以指定CPU或者GPU
        # note: 将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去
        # note: 之后的运算都在GPU上进行
        self.net = self.net.to(self.device)
        self.epochs = optim_params['num_epochs']
        self.last_epoch = optim_params['last_epoch']
        self.contrastive_loss = False

        # note: 2.5d resume
        if optim_params['resume']:
            self.load(optim_params['resume'], optim_params['ignore_list'])

        run_timestamp = datetime.now().strftime("%Y%b%d-%H%M%S")
        if logging_params['logging_dir'] is None:
            self.ckpt_path = os.path.join(logging_params['ckpt_path'], run_timestamp)
        else:
            self.ckpt_path = os.path.join(logging_params['ckpt_path'], logging_params['logging_dir'])
        if logging_params['use_logging']:
            if not os.path.exists(self.ckpt_path):
                os.makedirs(self.ckpt_path)
            else:
                select = input("whether delete the files in ckpt_path: (yes/not): ")
                if select.strip().lower() == 'yes':
                    for f in os.listdir(self.ckpt_path):
                        if not os.path.isdir(f):
                            os.remove(os.path.join(self.ckpt_path, f))
            self.logger = get_logger(os.path.join(self.ckpt_path, 'classifier_reid.log'))
            self.logger.info(">>> The config is:")
            self.logger.info(format_config(config))
            self.logger.info(">>> The net is:")
            self.logger.info(self.net)
        if logging_params['use_tensorboard']:

            if logging_params['logging_dir'] is None:
                self.run_path = os.path.join(logging_params['run_path'], run_timestamp)
            else:
                self.run_path = os.path.join(logging_params['run_path'], logging_params['logging_dir'])
            if not os.path.exists(self.run_path):
                os.makedirs(self.run_path)
            else:
                select = input("whether delete the files in run_path: (yes/not): ")
                if select.strip().lower() == 'yes':
                    for f in os.listdir(self.run_path):
                        if not os.path.isdir(f):
                            os.remove(os.path.join(self.run_path, f))
            self.writer = SummaryWriter(self.run_path)

    # note: model.run()
    def run(self):
        optim_params = self.config['optim']
        if optim_params['optim_method'] == 'sgd':
            sgd_params = optim_params['sgd']
            optimizer = optim.SGD(
                self.net.parameters(),
                lr=sgd_params['base_lr'],
                momentum=sgd_params['momentum'],
                weight_decay=sgd_params['weight_decay'],
                nesterov=sgd_params['nesterov'])
        elif optim_params['optim_method'] == 'adam':
            adam_params = optim_params['adam']
            optimizer = optim.Adam(
                self.net.parameters(),
                lr=adam_params['base_lr'],
                betas=adam_params['betas'],
                weight_decay=adam_params['weight_decay'],
                amsgrad=adam_params['amsgrad'])
        elif optim_params['optim_method'] == 'adamW':
            adamW_params = optim_params['adamW']
            optimizer = optim.AdamW(
                self.net.parameters(),
                lr=adamW_params['base_lr'],
                betas=adamW_params['betas'],
                weight_decay=adamW_params['weight_decay'],
                amsgrad=adamW_params['amsgrad'])
        else:
            raise Exception('Not support optim method: {}.'.format(optim_params['optim_method']))

        # choosing whether to use lr_decay and related parameters
        lr_decay = None
        if optim_params['use_lr_decay']:
            from torch.optim import lr_scheduler
            if optim_params['lr_decay_method'] == 'cosine':
                cosine_params = optim_params['cosine']
                lr_decay = lr_scheduler.CosineAnnealingLR(
                    optimizer, eta_min=cosine_params['eta_min'], T_max=cosine_params['T_max'])
            elif optim_params['lr_decay_method'] == 'exponent':
                exponent_params = optim_params['exponent']
                lr_decay = lr_scheduler.ExponentialLR(
                    optimizer, gamma=exponent_params['gamma'])
            elif optim_params['lr_decay_method'] == 'warmup':
                warmup_params = optim_params['warmup']
                from nets.cores.warmup_scheduler import GradualWarmupScheduler
                if warmup_params['after_scheduler'] == 'cosine':
                    cosine_params = optim_params['cosine']
                    after_scheduler = lr_scheduler.CosineAnnealingLR(
                        optimizer, eta_min=cosine_params['eta_min'], T_max=cosine_params['T_max'])
                elif warmup_params['after_scheduler'] == 'exponent':
                    exponent_params = optim_params['exponent']
                    after_scheduler = lr_scheduler.ExponentialLR(
                        optimizer, gamma=exponent_params['gamma'])
                else:
                    raise Exception('Not support after_scheduler method: {}.'.format(warmup_params['after_scheduler']))

                lr_decay = GradualWarmupScheduler(optimizer, multiplier=warmup_params['multiplier'],
                                                  total_epoch=warmup_params['total_epoch'],
                                                  after_scheduler=after_scheduler)

        data_params = self.config['data']
        # making train dataset and dataloader
        train_params = self.config['train']
        # note: 构造训练数据集
        # train_dataset = ClassifierDataset(
        #     records_path=data_params['train_records_path'],
        #     config=data_params,
        #     phase='train')

        # note: position embedding
        train_dataset = SiameseDataset(
            records_path=data_params['train_records_path'],
            config=data_params,
            phase='train')
            # position_eb=True)
        
        print('batch_size:', train_params['batch_size'])
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_params['batch_size'],
            shuffle=True,
            num_workers=train_params['num_workers'],
            pin_memory=False)
    
        self.logger.info("[Info] The size of training dataset: {}".format(len(train_dataset)))

        test_params = self.config['test']
        # note: 构造测试数据集
        # test_dataset = ClassifierDataset(
        #     records_path=data_params['test_records_path'],
        #     config=data_params,
        #     phase='val')

        # note: position embedding
        test_dataset = SiameseDataset(
            records_path=data_params['test_records_path'],
            config=data_params,
            phase='val')
            # position_eb=True)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=test_params['batch_size'],
            shuffle=False,
            num_workers=test_params['num_workers'],
            pin_memory=False)
        self.logger.info("[Info] The size of testing dataset: {}".format(len(test_dataset)))

        # import pdb; pdb.set_trace()

        # note: 评价指标
        # choosing criterion
        criterion_params = self.config['criterion']
        # note: CE Loss
        if criterion_params['criterion_method'] == 'cross_entropy_loss':
            self.logger.info("cross_entropy_loss")
            loss_params = criterion_params['cross_entropy_loss']
            if loss_params['use_weight']:
                weight = torch.Tensor(loss_params['weight'])
                criterion = nn.CrossEntropyLoss(weight).to(self.device)
            else:
                criterion = nn.CrossEntropyLoss().to(self.device)
        # TODO: Contrastive Loss
        elif criterion_params['criterion_method'] == 'focal_loss':
            self.logger.info("focal_loss")
            loss_params = criterion_params['focal_loss']
            criterion = FocalLoss(gamma=loss_params['gamma'], alpha=loss_params['alpha']).to(self.device)
        elif criterion_params['criterion_method'] == 'contrastive_loss':
            self.logger.info("contrastive_loss")
            criterion = ContrastiveLoss().to(self.device)
            self.contrastive_loss = True
        elif criterion_params['criterion_method'] == 'ghm_loss':
             self.logger.info("ghm_loss")
             loss_params = criterion_params['ghm_loss']
             criterion = GHMC_Loss(bins=loss_params['bins'], alpha=loss_params['alpha'], device=self.device).to(self.device)
        else:
            raise Exception('Not support criterion method: {}.'
                            .format(criterion_params['criterion_method']))
        # recording the best model
        best_f_beta = 0
        best_epoch = 0
        best_auc = 0
        best_auc_epoch = 0
        for epoch in range(self.last_epoch + 1, self.epochs):
            # note: optim模块：https://zhuanlan.zhihu.com/p/41127426
            for param_group in optimizer.param_groups:
                self.logger.info("[Info] epoch:{}, lr {:.6f}".format(epoch, param_group['lr']))
            
            # note: 100个epoch后换ghm loss
            # if epoch >= 99:
            #     if epoch == 99:
            #         self.logger.info("change loss to ghm loss")
            #     loss_params = criterion_params['ghm_loss']
            #     criterion = GHMC_Loss(bins=loss_params['bins'], alpha=loss_params['alpha'], device=self.device).to(self.device)
                   
            # note: 训练
            self.train(epoch, train_loader, criterion, optimizer)
            # note: 验证
            test_f_beta, test_auc = self.validate(epoch, test_loader, criterion)
            if lr_decay is not None:
                # 模型参数更新
                lr_decay.step()
            # saving the best model
            if test_f_beta >= best_f_beta:
                best_f_beta = test_f_beta
                best_epoch = epoch
                self.save(epoch)
            if test_auc >= best_auc:
                best_auc = test_auc
                best_auc_epoch = epoch
                self.save(epoch, note='auc')
            self.logger.info('[Info] The maximal f_beta is {:.4f} at epoch {}, maximal auc is {:.4f} at epoch {}'.format(
                best_f_beta,
                best_epoch,
                best_auc,
                best_auc_epoch))

    # note: 模型训练
    def train(self, epoch_id, data_loader, criterion, optimizer):
        # 记录和更新变量
        loss_meter = AverageMeter()
        matrix_meter = ConfusionMeter(self.config['data']['num_classes'])
        self.net.train()
        with tqdm(total=len(data_loader)) as pbar:
            for batch_id, sample in enumerate(data_loader):
                # note: template img
                template_img = sample['template'].to(self.device)
                key1 = sample['key1']
                # note: searching_img
                searching_img = sample['searching'].to(self.device)
                uuid2 = sample['uuid2']
                # img = sample['image'].to(self.device)
                target = sample['target'].to(self.device)
                # import pdb; pdb.set_trace()
                # note: position embedding
                # t_coord = sample['t_coord'].to(self.device)
                # s_coord = sample['s_coord'].to(self.device)

                # 清空所有被优化过的Variable的梯度
                optimizer.zero_grad()
                # note: 数据输入网络
                if self.contrastive_loss:
                    logit, output1, output2 = self.net(template_img, searching_img, contrastive_loss=True)
                    # import pdb
                    # pdb.set_trace()
                    # output1 = output1.squeeze(-1).squeeze(-1)
                    # output2 = output2.squeeze(-1).squeeze(-1)
                    criterion2 = nn.CrossEntropyLoss().to(self.device)
                    contrastive_loss = criterion(output1, output2, target)
                    ce_loss = criterion2(logit, target)
                    loss = 1e-2 * contrastive_loss + ce_loss
                else:
                    # note: position embedding
                    logit = self.net(template_img, searching_img)
                    # logit = self.net(template_img, searching_img, pos_eb=True, t_pos=t_coord, s_pos=s_coord)
                    # import pdb; pdb.set_trace()
                    loss = criterion(torch.unsqueeze(logit, dim=-1), torch.unsqueeze(target, dim=-1))
                    # note: ghm loss
                    # if epoch_id >= 99:
                    #     one_hot_target = self.mask2onehot(target.cpu().numpy(), 2)
                    #     one_hot_target = torch.from_numpy(one_hot_target.T.astype(float)).float()
                    #     loss = criterion(logit, one_hot_target.to(self.device))
                    # else:
                    #     loss = criterion(logit, target)

                loss.backward()
                optimizer.step()
                loss_meter.update(loss.data.item(), template_img.size(0))
                matrix_meter.add(logit, target)

                pbar.update(1)
                pbar.set_description("[Train] Epoch:{}, Loss:{:.4f}".format(epoch_id, loss_meter.avg))

            # note: metrics
            cls_metrics = self.cls_metrics(matrix_meter.value())

            logging_params = self.config['logging']
            if logging_params['use_logging']:
                self.logger.info("[Train] Epoch:{}, Loss:{:.4f}, ConfusionMeter:\n{}, \nCls Metric \n{}"
                                 .format(epoch_id, loss_meter.avg, matrix_meter.value(), cls_metrics))
            if logging_params['use_tensorboard']:
                self.writer.add_scalar('train/loss', loss_meter.avg, epoch_id)
                self.writer.add_scalar('train/precision', cls_metrics[0], epoch_id)
                self.writer.add_scalar('train/recall', cls_metrics[1], epoch_id)
                self.writer.add_scalar('train/f_beta', cls_metrics[2], epoch_id)
                self.writer.add_scalar('train/accuracy', cls_metrics[3], epoch_id)
                # save_confusion_matrix(matrix_meter.value(),
                #                       os.path.join(self.run_path, "train_cm_{}.png".format(epoch_id)))

    # note: 模型验证
    def validate(self, epoch_id, data_loader, criterion):
        loss_meter = AverageMeter()
        matrix_meter = ConfusionMeter(self.config['data']['num_classes'])
        self.net.eval()
        gt_ndarry = np.empty([0]) 
        pr_ndarry = np.empty([0])
        with torch.no_grad():
            with tqdm(total=len(data_loader)) as pbar:
                for batch_id, sample in enumerate(data_loader):
                    # img = sample['image'].to(self.device)
                    template_img = sample['template'].to(self.device)
                    key1 = sample['key1']
                    searching_img = sample['searching'].to(self.device)
                    uuid2 = sample['uuid2']
                    target = sample['target'].to(self.device)
                    # note: position embedding
                    # t_coord = sample['t_coord'].to(self.device)
                    # s_coord = sample['s_coord'].to(self.device)

                    if self.contrastive_loss:
                        logit, output1, output2 = self.net(template_img, searching_img, contrastive_loss=True)
                        criterion2 = nn.CrossEntropyLoss().to(self.device)
                        contrastive_loss = criterion(output1, output2, target)
                        ce_loss = criterion2(logit, target)
                        loss = 1e-2 * contrastive_loss + ce_loss
                    else:
                        # note: position embedding
                        logit = self.net(template_img, searching_img)
                        # logit = self.net(template_img, searching_img, pos_eb=True, t_pos=t_coord, s_pos=s_coord)
                        loss = criterion(torch.unsqueeze(logit, dim=-1), torch.unsqueeze(target, dim=-1))
                        # note: ghm loss
                        # if epoch_id >= 99:
                        #     one_hot_target = self.mask2onehot(target.cpu().numpy(), 2)
                        #     one_hot_target = torch.from_numpy(one_hot_target.T.astype(float)).float()
                        #     loss = criterion(logit, one_hot_target.to(self.device))
                        # else:
                        #     loss = criterion(logit, target)
                    loss_meter.update(loss.data.item(), template_img.size(0))
                    matrix_meter.add(logit, target)

                    logit_softmax = logit.softmax(dim=1).cpu()
                    match_pro = logit_softmax[:,1:].squeeze(1).numpy().tolist()
                    gt_list = target.cpu().numpy().tolist()
                    gt_ndarry = np.append(gt_ndarry, gt_list)
                    pr_ndarry = np.append(pr_ndarry, match_pro)

                    pbar.update(1)
                    pbar.set_description("[Eval] Epoch:{}, Loss:{:.4f}".format(epoch_id, loss_meter.avg))

            fpr, tpr, thresholds = roc_curve(gt_ndarry, pr_ndarry, pos_label=1)
            the_auc = auc(fpr, tpr)
            # print("-----sklearn:", auc(fpr, tpr))
            cls_metrics = self.cls_metrics(matrix_meter.value())

            logging_params = self.config['logging']
            if logging_params['use_logging']:
                self.logger.info("[Eval] Epoch:{}, Loss:{:.4f}, ConfusionMeter:\n{}, \nCls Metric \n{}, \nAUC:{}"
                                 .format(epoch_id, loss_meter.avg, matrix_meter.value(), cls_metrics, the_auc))
            if logging_params['use_tensorboard']:
                self.writer.add_scalar('eval/loss', loss_meter.avg, epoch_id)
                self.writer.add_scalar('eval/precision', cls_metrics[0], epoch_id)
                self.writer.add_scalar('eval/recall', cls_metrics[1], epoch_id)
                self.writer.add_scalar('eval/f_beta', cls_metrics[2], epoch_id)
                self.writer.add_scalar('eval/accuracy', cls_metrics[3], epoch_id)
                # save_confusion_matrix(matrix_meter.value(),
                #                       os.path.join(self.run_path, "eval_cm_{}.png".format(epoch_id)))

        return cls_metrics[2], the_auc

    # tag: 测试
    def test(self, display=True):
        print('-------test--------')
        # tag: 当前模型
        model_path = self.ckpt_path + '/best_auc.pth'
        # tag: 指定模型
        # model_path = '/root/workspace/lymph_node_classification/ckpts/0523/resnet_del_inappropriate_data_reviserandomsample_large_crop_ce_feature_metric_revise_l1metric_baseline/0523/resnet_del_inappropriate_data_reviserandomsample_large_crop_ce_feature_metric_revise_l1metric_baseline/best_auc.pth'
        # model_path = '/root/workspace/lymph_node_classification/ckpts/0330/fpnresnet_del_inappropriate_data_randomsample_large_crop_weightce_feature_metric_revise_pe/fpnresnet_del_inappropriate_data_randomsample_large_crop_weightce_feature_metric_revise_pe/best_auc.pth'
        # model_path = '/root/workspace/lymph_node_classification/ckpts/1129/fpn_distancerevise/fpn_distancerevise/best.pth'
        # model_path = '/root/workspace/lymph_node_classification/ckpts/1202/fpnresnet_dataset_change_normal_crop/fpnresnet_dataset_change_normal_crop/best.pth'
        self.load(model_path)
        criterion = nn.CrossEntropyLoss().to(self.device)
        # criterion = FocalLoss(gamma=2, alpha=0.25).to(self.device)
        # criterion = nn.CrossEntropyLoss().to(self.device)

        # note: position embedding
        test_dataset = SiameseDataset(
            records_path=self.config['data']['inference_records_path'],
            config=self.config['data'],
            phase='test')
            # position_eb=True)

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['test']['batch_size'],
            shuffle=False,
            num_workers=self.config['test']['num_workers'],
            pin_memory=False)
        loss_meter = AverageMeter()
        matrix_meter = ConfusionMeter(self.config['data']['num_classes'])
        self.net.eval()
        best_f_beta = 0
        best_epoch = 0
        # for epoch in range(self.last_epoch + 1, self.epochs):
        pair = pd.DataFrame()
        gt_ndarry = np.empty([0]) 
        pr_ndarry = np.empty([0])
        with torch.no_grad():
            with tqdm(total=len(test_loader)) as pbar:
                for batch_id, sample in enumerate(test_loader):
                    # img = sample['image'].to(self.device)
                    template_img = sample['template'].to(self.device)
                    case_name = sample['case_name']
                    uuid1 = sample['uuid1']
                    searching_img = sample['searching'].to(self.device)
                    uuid2 = sample['uuid2']
                    target = sample['target'].to(self.device)
                    # note: position embedding
                    # t_coord = sample['t_coord'].to(self.device)
                    # s_coord = sample['s_coord'].to(self.device)

                    if self.contrastive_loss:
                        logit, output1, output2 = self.net(template_img, searching_img, contrastive_loss=True)
                        criterion2 = nn.CrossEntropyLoss().to(self.device)
                        contrastive_loss = criterion(output1, output2, target)
                        ce_loss = criterion2(logit, target)
                        loss = 1e-2 * contrastive_loss + ce_loss
                    else:
                        # note: postition embedding
                        logit = self.net(template_img, searching_img)
                        # logit = self.net(template_img, searching_img, pos_eb=True, t_pos=t_coord, s_pos=s_coord)
                        loss = criterion(logit, target)
                    # import pdb;
                    # pdb.set_trace()
                    loss_meter.update(loss.data.item(), template_img.size(0))
                    # TODO: 打出来看一下logit
                    matrix_meter.add(logit, target)

                    logit_softmax = logit.softmax(dim=1).cpu()
                    unmatch_pro = logit_softmax[:,:1].squeeze(1).numpy().tolist()
                    match_pro = logit_softmax[:,1:].squeeze(1).numpy().tolist()
                    gt_list = target.cpu().numpy().tolist()

                    pr = logit.argmax(1).int()
                    pr_list = pr.cpu().numpy().tolist()

                    cur_pair = pd.DataFrame(
                                    [case_name, uuid1, uuid2, unmatch_pro, match_pro, gt_list, pr_list],
                                    index=['patient', 't1_uuid', 't2_uuid', '0', '1', 'gt', 'pr']).T
                    pair = pd.concat([pair, cur_pair], 0, ignore_index=True)

                    pbar.update(1)
                    pbar.set_description("[Test] Batch:{}, Loss:{:.4f}".format(batch_id, loss_meter.avg))
                    # TODO：可视化假阳淋巴结
                    # pr = logit.argmax(1).int()
                    # gt = target.int()
                    # pr_list = pr.cpu().tolist()

                    gt_ndarry = np.append(gt_ndarry, gt_list)
                    pr_ndarry = np.append(pr_ndarry, match_pro)

                    # saveFPLN(pr, gt, uuid1, uuid2)
                    # saveFPFNLN(pr, gt, uuid1, uuid2)
            
            # tag: 测试 保存pair.csv
            pair.to_csv(os.path.join('/root/workspace/code/LNsMatch/test', "0_fpn_large_crop_l1metric_wce_best_auc_test_pair.csv"), index=False)

            fpr, tpr, thresholds = roc_curve(gt_ndarry, pr_ndarry, pos_label=1)
            print("-----sklearn:", auc(fpr, tpr))
            
            # draw FROC curve
            totalNumberOfImages = 45
            numberOfMatchedLesions = sum(gt_ndarry)
            totalNumberOfCandidates = len(pr_ndarry)
            # FROC
            fps = fpr * (totalNumberOfCandidates - numberOfMatchedLesions) / totalNumberOfImages
            sens = tpr
            plt.plot(fps, sens, color='b', lw=2, label='auc:{:.3f}'.format(auc(fpr, tpr)))
            plt.legend(loc='lower right')
            plt.xlim([0.125, 8])
            plt.ylim([0, 1.1])
            plt.xlabel('Average number of false positives per scan')
            plt.ylabel('True Positive Rate')
            plt.title('FROC performance')
            # tag: 测试 保存FRCO曲线
            plt.savefig(os.path.join('/root/workspace/code/LNsMatch/test', "0_fpn_large_crop_l1metric_wce_best_auc_froc.png"))
            plt.show()

            cls_metrics = self.cls_metrics(matrix_meter.value())
            new_conf = metrics_by_max_similarity(pair)

            del [[pair]]
            gc.collect()

            # logging_params = self.config['logging']
            # if logging_params['use_logging']:
            # self.logger.info("[Test] Batch:{}, Loss:{:.4f}, ConfusionMeter:\n{}, \nCls Metric \n{}"
            #                         .format(batch_id, loss_meter.avg, matrix_meter.value(), cls_metrics))
            print("[Test] Batch:{}, Loss:{:.4f}, ConfusionMeter:\n{}, \nCls Metric \n{}, New Conf:\n{}"
                                    .format(batch_id, loss_meter.avg, matrix_meter.value(), cls_metrics, new_conf))
            # tag: 测试 保存混淆矩阵
            save_confusion_matrix(matrix_meter.value(),
                                        os.path.join('/root/workspace/code/LNsMatch/test', "0_fpn_large_crop_l1metric_wce_best_auc_cm.png"))

                # if logging_params['use_tensorboard']:
                #     save_confusion_matrix(matrix_meter.value(),
                #                           os.path.join(self.run_path, "test_cm_{}.png".format(batch_id)))
            # test_f_beta = cls_metrics[2]
            # if test_f_beta >= best_f_beta:
            #     best_f_beta = test_f_beta
            #     best_epoch = epoch
            # self.logger.info('[Info] The maximal f_beta is {:.4f} at epoch {}'.format(
            #     best_f_beta,
            #     best_epoch))
        # pass

    def mask2onehot(self, mask, num_classes):
        """
        Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
        hot encoding vector

        """
        _mask = [mask == i for i in range(num_classes)]
        return np.array(_mask).astype(np.uint8)

    def load(self, ckpt_path, ignore_list=None, display=True):
        if ignore_list is None:
            ignore_list = []
        network_params = self.config['network']
        # note: 用来加载模型
        ckpt = torch.load(ckpt_path, map_location=torch.device(self.config['network']['device']))
        # note: 2.5d 修改
        state_dict = ckpt['state_dict']
        # state_dict = ckpt
        # if key=='state_dict':
        #     state_dict = ckpt['state_dict']
        # else:
        #     state_dict = ckpt[key]
        if network_params['use_parallel']:
            model_dict = self.net.module.state_dict()
        else:
            model_dict = self.net.state_dict()
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
            self.net.module.load_state_dict(model_dict)
        else:
            self.net.load_state_dict(model_dict)
        print(">>> Loading model successfully from {}.".format(ckpt_path))

    def save(self, epoch, note='f_beta'):
        if self.config['network']['use_parallel']:
            state_dict = self.net.module.state_dict()
        else:
            state_dict = self.net.state_dict()
        # note: 保存模型 
        # 模型(models) 张量(tensors) 和文件夹(dictionaries) 都是可以用这个函数保存的目标类型
        if note== 'f_beta':
            torch.save({'epoch': epoch,
                        'state_dict': state_dict},
                    os.path.join(self.ckpt_path, 'best.pth'))
        else:
            torch.save({'epoch': epoch,
                        'state_dict': state_dict},
                    os.path.join(self.ckpt_path, 'best_'+note+'.pth'))

    @staticmethod
    def cls_metrics(confusion_matrix, label_idx=1, beta=1):
        # recall
        label_total_sum = confusion_matrix.sum(axis=1)[label_idx]
        label_correct_sum = confusion_matrix[label_idx][label_idx]
        recall = 0
        if label_total_sum != 0:
            recall = round(100 * float(label_correct_sum) / float(label_total_sum), 3)
        # precision
        label_total_sum = confusion_matrix.sum(axis=0)[label_idx]
        label_correct_sum = confusion_matrix[label_idx][label_idx]
        precision = 0
        if label_total_sum != 0:
            precision = round(100 * float(label_correct_sum) / float(label_total_sum), 3)
        # f_beta
        f_beta = 0
        if (precision + recall) != 0:
            f_beta = round((beta ** 2 + 1) / (beta ** 2 / recall + 1 / precision), 3)
        # accuracy
        accuracy = 0
        label_total_sum = confusion_matrix.sum()
        label_correct_sum = confusion_matrix[0][0] + confusion_matrix[1][1]
        if label_total_sum != 0:
            accuracy = round(100 * float(label_correct_sum) / float(label_total_sum), 3)

        return np.array([precision, recall, f_beta, accuracy])