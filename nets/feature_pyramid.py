import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.siamese_resnet3d import *
from nets.layers import conv1x1x1, conv3x3x3
from nets.non_local_dot_product import NONLocalBlock3D

class FeaturePyramid_v3(nn.Module):
    def __init__(self, resnet):
        super(FeaturePyramid_v3, self).__init__()

        self.resnet = resnet

        # applied in a pyramid
        self.pyramid_transformation_1 = conv3x3x3(64, 256, padding=1)
        self.pyramid_transformation_2 = conv3x3x3(64, 256, padding=1)
        self.pyramid_transformation_3 = conv3x3x3(128, 256, padding=1)
        self.pyramid_transformation_4 = conv1x1x1(256, 256)
        self.pyramid_transformation_5 = conv1x1x1(512, 256)

        # applied after upsampling
        self.upsample_transform_1 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_2 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_3 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_4 = conv3x3x3(256, 256, padding=1)

        # note: 减少通道数
        # self.pyramid_transformation_1 = conv3x3x3(64, 64, padding=1)
        # self.pyramid_transformation_2 = conv3x3x3(64, 64, padding=1)
        # self.pyramid_transformation_3 = conv3x3x3(128, 64, padding=1)
        # self.pyramid_transformation_4 = conv1x1x1(256, 64)
        # self.pyramid_transformation_5 = conv1x1x1(512, 64)

        # self.upsample_transform_1 = conv3x3x3(64, 64, padding=1)
        # self.upsample_transform_2 = conv3x3x3(64, 64, padding=1)
        # self.upsample_transform_3 = conv3x3x3(64, 64, padding=1)
        # self.upsample_transform_4 = conv3x3x3(64, 64, padding=1)

        # resnet
        self.avgpool = nn.AdaptiveAvgPool3d([1, 1, 1])
        # todo: 余弦距离
        # self.siamfc = nn.Linear(1, 2)
        # note: L1距离
        self.siamfc = nn.Linear(256, 2)
        self.drop = nn.Dropout3d(p=0.5)
        # note: Non Local Block
        # self.nonlocalblock = NONLocalBlock3D(256, sub_sample=True, bn_layer=True)
        # note: pairwise distance
        # self.pairwisedistance = nn.PairwiseDistance(p=2)
        # self.pdavgpool = nn.AdaptiveAvgPool3d([2, 8, 8])
        # self.pdflatten = nn.Flatten()
        # self.pdsiamfc = nn.Linear(128, 2)

    def _upsample(self, original_feature, scaled_feature, scale_factor=2):
        # is this correct? You do lose information on the upscale...
        depth, height, width = scaled_feature.size()[2:]
        return F.interpolate(original_feature, scale_factor=scale_factor)[:, :, :depth, :height, :width]

    
    def forward_once(self, resnet_feature_1, resnet_feature_2, resnet_feature_3, resnet_feature_4, resnet_feature_5):
        pyramid_feature_5 = self.pyramid_transformation_5(resnet_feature_5)     # transform c5 from 2048d to 256d
        pyramid_feature_4 = self.pyramid_transformation_4(resnet_feature_4)     # transform c4 from 1024d to 256d
        upsampled_feature_5 = self._upsample(pyramid_feature_5, pyramid_feature_4)   # deconv c5 to c4.size

        pyramid_feature_4 = self.upsample_transform_4(
            torch.add(upsampled_feature_5, pyramid_feature_4)               # add up-c5 and c4, and conv
        )

        pyramid_feature_3 = self.pyramid_transformation_3(resnet_feature_3)     # transform c3 from 512d to 256d
        upsampled_feature_4 = self._upsample(pyramid_feature_4, pyramid_feature_3)    # deconv c4 to c3.size

        pyramid_feature_3 = self.upsample_transform_3(
            torch.add(upsampled_feature_4, pyramid_feature_3)               # add up-c4 and c3, and conv
        )
        
        pyramid_feature_2 = self.pyramid_transformation_2(resnet_feature_2)                                # c2 is 256d, so no need to transform
        upsampled_feature_3 = self._upsample(pyramid_feature_3, pyramid_feature_2)    # deconv c3 to c2.size

        pyramid_feature_2 = self.upsample_transform_2(
            torch.add(upsampled_feature_3, pyramid_feature_2)              # add up-c3 and c2, and conv
        )
        
        pyramid_feature_1 = self.pyramid_transformation_1(resnet_feature_1)  # use conv3x3x3 up c1 from 64d to 256d
        upsampled_feature_2 = self._upsample(pyramid_feature_2, pyramid_feature_1)    # deconv c2 to c1.size

        pyramid_feature_1 = self.upsample_transform_1(
            torch.add(upsampled_feature_2, pyramid_feature_1)             # add up-c2 and c1, and conv
        )
        
        return (pyramid_feature_1,             # 8
                pyramid_feature_2,             # 16
                pyramid_feature_3)             # 32

    def processed_feature(self, x):
        # note: pairwise distance度量
        # x9 = self.pdavgpool(x)
        # x10 = self.drop(x9)
        # feat =  x10.view(x10.size(0), -1)
        # output = self.pdsiamfc(feat)

        x9 = self.avgpool(x)
        x10 = self.drop(x9)
        # x11 = x10.sum(1)
        # TODO: 修改
        feat =  x10.view(x10.size(0), -1)
        output = self.siamfc(feat)

        # return x10
        return output
        #add后一起fc return feat

    def pdist(self, feature1, feature2):
        # note: L1距离
        pdist = (feature1-feature2).abs()
        # pdistfun = nn.PairwiseDistance(p=2)
        # pdist = pdistfun(feature1, feature2)
        # TODO: 修改
        # feat =  pdist.view(pdist.size(0), -1)

        # TODO: 余弦距离
        # pdist = F.cosine_similarity(feature1, feature2, dim=1)
        # feat =  pdist.view(pdist.size(0), -1)

        # TODO: 修改
        # output = self.siamfc(feat)
        # return output
        return pdist
    
    # note: pairwisedistance process
    def pairwise_process(self, feature1, feature2):
        # feature1 = self.pdavgpool(feature1)
        # feature2 = self.pdavgpool(feature2)
        pairwise_dist = self.pairwisedistance(feature1, feature2)
        pairwise_dist = self.pdavgpool(pairwise_dist)
        pairwise_dist = self.drop(pairwise_dist)
        # pairwise_feat = pairwise_dist.view(pairwise_dist.size(0), -1)
        pairwise_feat = self.pdflatten(pairwise_dist)
        pairwise_output = self.pdsiamfc(pairwise_feat)

        # return pairwise_feat
        return pairwise_output

    
    def forward(self, input1, input2, pos_eb=False, t_pos=None, s_pos=None):
        if not pos_eb:
            resnet_feature_1_1, resnet_feature_1_2, resnet_feature_1_3, resnet_feature_1_4, resnet_feature_1_5, \
            resnet_feature_2_1, resnet_feature_2_2, resnet_feature_2_3, resnet_feature_2_4, resnet_feature_2_5 = self.resnet(input1, input2, is_feat=True)
        
            pyramid_feature_1_1, pyramid_feature_1_2, pyramid_feature_1_3 = self.forward_once(resnet_feature_1_1, resnet_feature_1_2, resnet_feature_1_3, resnet_feature_1_4, resnet_feature_1_5)
            pyramid_feature_2_1, pyramid_feature_2_2, pyramid_feature_2_3 = self.forward_once(resnet_feature_2_1, resnet_feature_2_2, resnet_feature_2_3, resnet_feature_2_4, resnet_feature_2_5)

            # output1_1 = self.processed_feature(pyramid_feature_1_1)
            # output2_1 = self.processed_feature(pyramid_feature_2_1)
            # output1 = self.pdist(output1_1, output2_1)

            # output1_2 = self.processed_feature(pyramid_feature_1_2)
            # output2_2 = self.processed_feature(pyramid_feature_2_2)
            # output2 = self.pdist(output1_2, output2_2)

            # output1_3 = self.processed_feature(pyramid_feature_1_3)
            # output2_3 = self.processed_feature(pyramid_feature_2_3)
            # output3 = self.pdist(output1_3, output2_3)

            # TODO: 修改特征图度量
            # note: Non Local Block
            # pyramid_feature_1_1 = self.nonlocalblock(pyramid_feature_1_1)
            # pyramid_feature_2_1 = self.nonlocalblock(pyramid_feature_2_1)
            # pyramid_feature_1_2 = self.nonlocalblock(pyramid_feature_1_2)
            # pyramid_feature_2_2 = self.nonlocalblock(pyramid_feature_2_2)
            # pyramid_feature_1_3 = self.nonlocalblock(pyramid_feature_1_3)
            # pyramid_feature_2_3 = self.nonlocalblock(pyramid_feature_2_3)
            
            # note: pairwiseDistance度量
            # pdist1 = self.pairwisedistance(pyramid_feature_1_1, pyramid_feature_2_1)
            # output1 = self.processed_feature(pdist1)

            # pdist2 = self.pairwisedistance(pyramid_feature_1_1, pyramid_feature_2_1)
            # output2 = self.processed_feature(pdist2)

            # pdist3 = self.pairwisedistance(pyramid_feature_1_1, pyramid_feature_2_1)
            # output3 = self.processed_feature(pdist3)

            # feat = torch.add(feat1, feat2)
            # feat = torch.add(feat, feat3)
            # output = self.pdsiamfc(feat)
            
            # note: featuremapabsdistance addfeatrevise
            pdist1 = self.pdist(pyramid_feature_1_1, pyramid_feature_2_1)
            output1 = self.processed_feature(pdist1)

            pdist2 = self.pdist(pyramid_feature_1_2, pyramid_feature_2_2)
            output2 = self.processed_feature(pdist2)

            pdist3 = self.pdist(pyramid_feature_1_3, pyramid_feature_2_3)
            output3 = self.processed_feature(pdist3)

            # output = torch.add(output1, output2)
            # output = torch.add(output, output3)
            # output = self.siamfc(output)

            # note: pairwisedistance addfeat
            # pairwise_output1 = self.pairwise_process(pyramid_feature_1_1, pyramid_feature_2_1)
            # pairwise_output2 = self.pairwise_process(pyramid_feature_1_2, pyramid_feature_2_2)
            # pairwise_output3 =  self.pairwise_process(pyramid_feature_1_3, pyramid_feature_2_3)
            # return pairwise_output1, pairwise_output2, pairwise_output3

            # pairwise_feat = torch.add(pairwise_feat1, pairwise_feat2)
            # pairwise_feat = torch.add(pairwise_feat, pairwise_feat3)
            # output = self.pdsiamfc(pairwise_feat)


            # feat = torch.add(feat1, feat2)
            # feat = torch.add(feat, feat3)
            # output = self.pdsiamfc(feat)
        
            return  output1, output2, output3
            # return output
        else:
            resnet_feature_1_1, resnet_feature_1_2, resnet_feature_1_3, resnet_feature_1_4, resnet_feature_1_5, \
            resnet_feature_2_1, resnet_feature_2_2, resnet_feature_2_3, resnet_feature_2_4, resnet_feature_2_5, t_poseb, s_poseb = self.resnet(input1, input2, is_feat=True, pos_eb=True, t_pos=t_pos, s_pos=s_pos)
        
            pyramid_feature_1_1, pyramid_feature_1_2, pyramid_feature_1_3 = self.forward_once(resnet_feature_1_1, resnet_feature_1_2, resnet_feature_1_3, resnet_feature_1_4, resnet_feature_1_5)
            pyramid_feature_2_1, pyramid_feature_2_2, pyramid_feature_2_3 = self.forward_once(resnet_feature_2_1, resnet_feature_2_2, resnet_feature_2_3, resnet_feature_2_4, resnet_feature_2_5)

            # TODO: 修改特征图度量
            pdist1 = self.pdist(pyramid_feature_1_1, pyramid_feature_2_1)
            output1 = self.processed_feature(pdist1)

            pdist2 = self.pdist(pyramid_feature_1_2, pyramid_feature_2_2)
            output2 = self.processed_feature(pdist2)

            # TODO: 处理postion embedding
            pyramid_feature_1_3 = pyramid_feature_1_3 + t_poseb
            pyramid_feature_2_3 = pyramid_feature_2_3 + s_poseb
            pdist3 = self.pdist(pyramid_feature_1_3, pyramid_feature_2_3)
            output3 = self.processed_feature(pdist3)
        
            return  output1, output2, output3          

class SubNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SubNet, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool3d([1, 1, 1])
        # self.siamfc = nn.Linear(1, num_classes)
        self.siamfc = nn.Linear(512, num_classes)
        self.drop = nn.Dropout3d(p=0.5)
        self.pdist = nn.PairwiseDistance(p=2)
    
    def forward_once(self, x):
        x9 = self.avgpool(x)
        x10 = self.drop(x9)
        # x11 = x10.sum(1)
        return x10

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        # pdist = (output1-output2).abs()
        pdist = self.pdist(output1, output2)
        # TODO: 全连接层
        feat =  pdist.view(pdist.size(0), -1)
        output = self.siamfc(feat)
        return output

class FeaturePyramidResNet(nn.Module):
    def __init__(self, color_channels=1,  num_classes=2):
        super(FeaturePyramidResNet, self).__init__()	
        self.feature_pyramid = FeaturePyramid_v3(resnet34(color_channels=color_channels, num_classes=num_classes))
    
    def forward(self, input1, input2, pos_eb=False, t_pos=None, s_pos=None):
        # TODO: 设置resnet34 返回feature
        # note: 修改0414
        if not pos_eb:
            output1, output2, output3 = self.feature_pyramid(input1, input2)
            # output = self.feature_pyramid(input1, input2)
        else:
            output1, output2, output3 = self.feature_pyramid(input1, input2, pos_eb=True, t_pos=t_pos, s_pos=s_pos)
        output = torch.add(output1, output2)
        output = torch.add(output, output3)
        return output