import torch
import torch.nn as nn
import torch.nn.functional as F


class ClsHead(nn.Module):
    """Simplest classification head"""

    def __init__(self,
                 with_avg_pool=True,
                 temporal_feature_size=1,
                 spatial_feature_size=7,
                 dropout_ratio=0.8,
                 in_channels=2048,
                 num_classes=101,
                 fcn_testing=False,
                 init_std=0.01):

        super(ClsHead, self).__init__()

        self.with_avg_pool = with_avg_pool
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.dropout_ratio = dropout_ratio
        self.temporal_feature_size = temporal_feature_size
        self.spatial_feature_size = spatial_feature_size
        self.init_std = init_std
        self.fcn_testing = fcn_testing

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        # self.with_avg_pool = fcn_testing
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool3d((temporal_feature_size, spatial_feature_size, spatial_feature_size), (1, 1, 1),
                                         (0, 0, 0))
        if self.fcn_testing:
            self.new_cls = None
            self.in_channels = in_channels
            self.num_classes = num_classes
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def init_weights(self):
        nn.init.normal_(self.fc_cls.weight, 0, self.init_std)
        nn.init.constant_(self.fc_cls.bias, 0)

    def forward(self, x):
        if not self.fcn_testing:
            if x.ndimension() == 4:
                x = x.unsqueeze(2)
            assert x.shape[1] == self.in_channels
            assert x.shape[2] == self.temporal_feature_size
            assert x.shape[3] == self.spatial_feature_size
            assert x.shape[4] == self.spatial_feature_size
            if self.with_avg_pool:
                x = self.avg_pool(x)
            if self.dropout is not None:
                x = self.dropout(x)
            x = x.view(x.size(0), -1)
            cls_score = self.fc_cls(x)
            return cls_score
        else:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            if self.new_cls is None:
                self.new_cls = nn.Conv3d(self.in_channels, self.num_classes, 1, 1, 0).cuda()
                self.new_cls.weight.copy_(self.fc_cls.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                self.new_cls.bias.copy_(self.fc_cls.bias)
                self.fc_cls = None
            class_map = self.new_cls(x)
            # return class_map.mean([2,3,4])
            return class_map

    def loss(
        self,
        cls_score,
        labels
    ):
        losses = dict()
        losses['loss_cls'] = F.mse_loss(cls_score, labels)

        return losses
