import metrics.metrics as m
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, \
    PointNetFeaturePropagation, PointNetRegressionHead
import provider
import pytorch_lightning as pl


class KeypointNet2(pl.LightningModule):
    def __init__(self, num_classes, batch_size, normal_channel=False, channels_offset: int = None, num_point: int = None, optimizer_name='SGD', learning_rate=1e-3):
        super().__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        if normal_channel:
            additional_channel = 3
        else:
            if channels_offset is None:
                additional_channel = 0
            else:
                additional_channel = channels_offset

        self.normal_channel = normal_channel

        self.set_abstraction1 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.1, 0.2, 0.4],
            nsample_list=[32, 64, 128],
            in_channel=3 + (additional_channel * (additional_channel > 0)),
            mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )

        self.set_abstraction2 = PointNetSetAbstractionMsg(
            npoint=128,
            radius_list=[0.4, 0.8],
            nsample_list=[64, 128],
            in_channel=128 + 128 + 64,
            mlp_list=[[128, 128, 256], [128, 196, 256]]
        )

        self.set_abstraction3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=512 + 3,
            mlp=[256, 512, 1024],
            group_all=True
        )

        self.feature_propagation3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.feature_propagation2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.feature_propagation1 = PointNetFeaturePropagation(in_channel=150 + additional_channel - 14, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, self.num_classes, 1)
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
        l0_points = self.feature_propagation1(
            l0_xyz,
            l1_xyz,
            torch.cat([cls_label_one_hot, l0_xyz, l0_points], 1),
            l1_points
        )
        # Keypoints Regression FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))

        flat_feat = torch.flatten(feat)
        res = self.regression_head(flat_feat)

        return res

    def configure_optimizers(self):
        optim_class = getattr(torch.optim, self.optimizer_name)
        return optim_class(self.parameters(), lr=self.learning_rate)

    def training_step(self, train_batch, batch_idx):
        points, label, target, sample_id = train_batch
        points_device = points.device
        points = points.data.cpu().numpy()
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        points = torch.tensor(points, device=points_device)
        points, label, target = points.float(), label.long(), target.float()
        points = points.transpose(2, 1)

        kpts_pred = self(points, self._to_categorical(label))

        if (torch.isnan(kpts_pred).sum() > 0):
            print('PROBLEM')

        # TODO: Convert distance in a metrics
        dist = m.euclidian_dist(kpts_pred, torch.squeeze(target))

        loss = F.mse_loss(kpts_pred, torch.squeeze(target))

        self.log_dict(
            dictionary={
                'train_loss': loss,
                # 'distance': dist
            },
            batch_size=self.batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        return {
            'loss': loss,
            'points': points,
            'keypoint_predictions': kpts_pred,
            'sample_id': sample_id[0],
            'stage': 'train'
        }


    def validation_step(self, val_batch, batch_idx):
        points, label, target, sample_id = val_batch
        cur_batch_size, num_point, _ = points.size()
        points = points.float()
        label = label.long()
        target = target.float()
        points = points.transpose(2, 1)
        keypoints_predictions = self(points, self._to_categorical(label))

        # TODO: Convert distance in a metrics
        dist = m.euclidian_dist(keypoints_predictions, torch.squeeze(target))
        loss = F.mse_loss(keypoints_predictions, torch.squeeze(target))

        self.log_dict(
            dictionary={
                'val_loss': loss,
                # 'distance': dist
            },
            batch_size=self.batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        return {
            'points': points,
            'keypoint_predictions': keypoints_predictions,
            'sample_id': sample_id[0],
            'stage': 'val'
        }

    def _to_categorical(self, y):
        y_prime = torch.eye(self.num_classes)[y.cpu().data.numpy(),]
        if y.is_cuda:
            return y_prime.cuda()
        return y_prime
