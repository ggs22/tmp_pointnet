import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, \
    PointNetFeaturePropagation, PointNetRegressionHead


class GetModel(nn.Module):
    def __init__(self, num_classes, normal_channel=False, channels_offset: int = None, num_point: int = None):
        super(GetModel, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            if channels_offset is None:
                additional_channel = 0
            else:
                additional_channel = channels_offset
        self.normal_channel = normal_channel
        self.set_abstraction1 = PointNetSetAbstractionMsg(npoint=512,
                                                          radius_list=[0.1, 0.2, 0.4],
                                                          nsample_list=[32, 64, 128],
                                                          in_channel=3+(additional_channel*(additional_channel > 0)),
                                                          mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.set_abstraction2 = PointNetSetAbstractionMsg(npoint=128,
                                                          radius_list=[0.4, 0.8],
                                                          nsample_list=[64, 128],
                                                          in_channel=128+128+64,
                                                          mlp_list=[[128, 128, 256], [128, 196, 256]])
        self.set_abstraction3 = PointNetSetAbstraction(npoint=None,
                                                       radius=None,
                                                       nsample=None,
                                                       in_channel=512 + 3,
                                                       mlp=[256, 512, 1024],
                                                       group_all=True)
        self.feature_propagation3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.feature_propagation2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.feature_propagation1 = PointNetFeaturePropagation(in_channel=150+additional_channel-14, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        self.regression_head = PointNetRegressionHead(in_channel=128 * num_point, keypoint_num=5)

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        batch_size, input_channels, num_points = xyz.shape  # Batch, Channels, Num points?
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.set_abstraction1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.set_abstraction2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.set_abstraction3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.feature_propagation3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.feature_propagation2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = cls_label.view(batch_size, cls_label.shape[2], 1).repeat(1, 1, num_points)
        l0_points = self.feature_propagation1(l0_xyz,
                                              l1_xyz,
                                              torch.cat([cls_label_one_hot, l0_xyz, l0_points], 1),
                                              l1_points)
        # Keypoints Regression FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))

        flat_feat = torch.flatten(feat)
        res = self.regression_head(flat_feat)

        # # part seg FC layers
        # feat = F.relu(self.bn1(self.conv1(l0_points)))
        # x = self.drop1(feat)
        # x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)

        return res


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.mse_loss(pred, target, reduction='mean')

        # TODO add bce loss to adapt for classification

        return total_loss
