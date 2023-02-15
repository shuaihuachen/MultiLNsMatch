from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.non_local_dot_product import NONLocalBlock3D

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4), device=out.device).zero_()
    out = torch.cat([out.data, zero_pads], dim=1)

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 color_channels,
                 shortcut_type='B',
                 num_classes=400):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # 3D卷积，输入的shape是(N, Cin, D, H, W)，输出的shape(N, Cout, Dout, Hout, Wout)
        self.conv1 = nn.Conv3d(
            color_channels,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d([1, 1, 1])
        # 用于设置网络中的全连接层 [batch_size, in_features]的输入张量-> [batch_size, out_features]的输出张量
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # note: cat spatial embedding
        # self.fc = nn.Linear(640 * block.expansion, num_classes)
        self.siamfc = nn.Linear(1 * block.expansion, num_classes)
        # note: PairwiseDistance时对应faltten FC
        self.pdflatten = nn.Flatten()
        self.pdfc = nn.Linear(9 * block.expansion, num_classes)
        # note: cat positon embeding pairwise distance
        # self.pdfc = nn.Linear(10 * block.expansion, num_classes)
        # note: position embeding FC层-FPN
        # self.posfc = nn.Linear(3, 128)
        # note: large crop 96*96
        # self.posfc = nn.Linear(3, 288)
        # self.posfc = nn.Linear(7, 288)
        # self.posfc2 = nn.Linear(288, 288)
        # note: cat postition embeding L1 metric
        # self.posfc = nn.Linear(7, 512)
        # self.posfc2 = nn.Linear(512, 512)
        # self.posfc = nn.Linear(7, 512)
        # self.posfc2 = nn.Linear(512, 512)
        self.posfc = nn.Linear(7, 128)
        self.posfc2 = nn.Linear(128, 128)
        self.catfeatfc = nn.Linear(640, 640)
        # note: Non Local Block
        # self.nonlocalblock1 = NONLocalBlock3D(64,sub_sample=True, bn_layer=True)
        # self.nonlocalblock2 = NONLocalBlock3D(128,sub_sample=True, bn_layer=True)
        # self.nonlocalblock3 = NONLocalBlock3D(256,sub_sample=True, bn_layer=True)
        # self.nonlocalblock4 = NONLocalBlock3D(512,sub_sample=True, bn_layer=True)
        
        # self.nonlocalblock = NONLocalBlock3D(512,sub_sample=True, bn_layer=True)
        # self.nonlocalblock = NONLocalBlock3D(1,sub_sample=True, bn_layer=True)
        self.pdist = nn.PairwiseDistance(p=2)
        self.expansion = block.expansion
        self.drop = nn.Dropout3d(p=0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                # 部分使用某个函数 冻结住某个函数的某些参数 让它们保证为某个值 并生成一个可调用的新函数对象
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # TODO: postion embedding
    def forward_postion(self, pos):
        pos_eb = F.leaky_relu(self.posfc(pos))
        pos_eb = F.leaky_relu(self.posfc2(pos_eb))
        return pos_eb

    # TODO: 处理一个输入 
    def forward_once(self, x, pos_eb=False, eb=None):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2 = self.maxpool(x1)

        x2 = self.layer1(x2)
        # note: nonlocal block each layer
        # x2 = self.nonlocalblock1(x2)
        x3 = self.layer2(x2)
        # x3 = self.nonlocalblock2(x3)
        # note: position embedding
        if pos_eb:
            eb = torch.reshape(eb, (eb.size(0),2,12,12))
            eb = eb.unsqueeze(1)
            x3 = x3 + eb
        x4 = self.layer3(x3)
        # x4 = self.nonlocalblock3(x4)
        x5 = self.layer4(x4)
        # x5 = self.nonlocalblock4(x5)

        # TODO: 修改： 特征图进行L1度量 然后池化 drop
        # x6 = self.avgpool(x5)
        # x7 = self.drop(x6)

        # TODO: Pair-wise Distance层 相减取绝对值 abs 
        # x8 = x7.sum(1)
        
        # TODO: 修改： 特征图进行L1度量 然后池化 drop
        # return x7, x1, x2, x3, x4, x5
        return x1, x2, x3, x4, x5

    
    def forward(self, input1, input2, is_feat=False, contrastive_loss=False, pos_eb=False, t_pos=None, s_pos=None):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        
        # print('model forward')

        # TODO: Pair-wise Distance层 相减取绝对值 abs
        # TODO: 修改： 特征图进行L1度量 然后池化 drop
        # output1, x1_1, x1_2, x1_3, x1_4, x1_5 = self.forward_once(input1)
        # output2, x2_1, x2_2, x2_3, x2_4, x2_5 = self.forward_once(input2)

        if pos_eb:
            t_poseb = self.forward_postion(t_pos)
            s_poseb = self.forward_postion(s_pos)
            # x1_1, x1_2, x1_3, x1_4, x1_5 = self.forward_once(input1, pos_eb=True, eb=t_poseb)
            # x2_1, x2_2, x2_3, x2_4, x2_5 = self.forward_once(input2, pos_eb=False, eb=s_poseb)
            x1_1, x1_2, x1_3, x1_4, x1_5 = self.forward_once(input1)
            x2_1, x2_2, x2_3, x2_4, x2_5 = self.forward_once(input2)
        else:
            x1_1, x1_2, x1_3, x1_4, x1_5 = self.forward_once(input1)
            x2_1, x2_2, x2_3, x2_4, x2_5 = self.forward_once(input2)
        # note: Non Local Block
        # x1_5 = self.nonlocalblock(x1_5)
        # x2_5 = self.nonlocalblock(x2_5)

        if is_feat:
            if pos_eb:
                t_poseb = self.forward_postion(t_pos)
                # t_poseb = torch.reshape(t_poseb, (t_poseb.size(0),2,8,8))
                # large crop 96*96
                t_poseb = torch.reshape(t_poseb, (t_poseb.size(0),2,12,12))
                t_poseb = t_poseb.unsqueeze(1)
                s_poseb = self.forward_postion(s_pos)
                # s_poseb = torch.reshape(s_poseb, (s_poseb.size(0),2,8,8))
                s_poseb = torch.reshape(s_poseb, (s_poseb.size(0),2,12,12))
                s_poseb = s_poseb.unsqueeze(1)
                return x1_1, x1_2, x1_3, x1_4, x1_5, x2_1, x2_2, x2_3, x2_4, x2_5, t_poseb, s_poseb
            # TODO: 连接特征金字塔
            return x1_1, x1_2, x1_3, x1_4, x1_5, x2_1, x2_2, x2_3, x2_4, x2_5
        else:
            # note: 特征图的每个点做L1距离metric-池化-FC
            pdist = (x1_5-x2_5).abs()
            # note: Non Local Block
            # pdist = self.nonlocalblock(pdist)
            pdist = self.avgpool(pdist)
            pdist = self.drop(pdist)
            feat =  pdist.view(pdist.size(0), -1)
            if pos_eb:
                # poseb_pdist = self.pdist(t_poseb, s_poseb)
                poseb_pdist = (t_poseb-s_poseb).abs()
                # feat = feat + poseb_pdist
                # note: cat poseb_pdist
                feat = torch.cat((feat, poseb_pdist),1)
                feat = self.catfeatfc(feat)
            output = self.fc(feat)

            # note: 特征图通道上的特征向量做欧式距离metric-展平-FC
            # pdist = self.pdist(x1_5,x2_5)
            # # note: Non Local Block
            # # pdist =torch.unsqueeze(pdist,1)
            # # pdist = self.nonlocalblock(pdist)

            # feat = self.pdflatten(pdist)
            # if pos_eb:
            #   poseb_pdist = self.pdist(t_poseb, s_poseb)
            #   feat = torch.cat((feat, torch.unsqueeze(poseb_pdist,1)), 1)
            # output = self.pdfc(feat)

            # note: 重构张量的维度，相当于numpy中resize()的功能
            # note: torch.view(参数a, -1), 在参数b未知，参数a已知的情况下自动补齐列向量长度
            # feat = x.view(x.size(0), -1)
            # x = self.fc(feat)

            if contrastive_loss:
                # return output, output1, output2
                return output, x1_5, x2_5
            else:
                return output


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet10(**kwargs):
    """Constructs a ResNet-10 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
