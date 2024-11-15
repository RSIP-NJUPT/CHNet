import torch
import torch.nn as nn
import torch.nn.functional as F
import GCommon
from config import opt


class _Gate(nn.Module):
    def __init__(self,):
        super(_Gate, self).__init__()

        self.flatten = nn.Flatten()
        self.IA_fc1 = nn.Linear(in_features=(32+opt.pca_components)*(int)(opt.patch_size/2)*(int)(opt.patch_size/2), out_features = 1000)#FLIR KAIST llvip
        self.IA_fc2 = nn.Linear(in_features=1000, out_features = 100)
        self.IA_fc3 = nn.Linear(in_features=100, out_features = 2)
        self.pool = nn.AvgPool2d(kernel_size=2) #FLIR KAIST

    def forward(self, img_hsi,img_lidar):

        x1= self.pool(torch.cat([img_hsi, img_lidar], dim=1))
        x2 = self.flatten(x1)
        x3 = self.IA_fc1(x2)
        x4 = self.IA_fc2(x3)
        weights = self.IA_fc3(x4)

        return weights



class ConvFusion(nn.Module):
    """Fusion Module

    注：其实看作一个门控（或者是feature-wise的dropout)

    Args:
        img_hsi (Tensor): The HSI input
        img_lidar (Tensor): The LiDAR input
    """

    def __init__(self):
        super(ConvFusion, self).__init__()
        self.Route = _Gate()
        self.expert_hsi = GCommon.FPNBlock(67)
        self.expert_lidar = GCommon.FPNBlock(67)
        self.bn = nn.BatchNorm2d(67)
    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def forward(self, hsi_feature, lidar_feature, common_feature,img_hsi,img_lidar):
        """Forward function."""

        route = self.Route(img_hsi,img_lidar)
        route = F.softmax(route, dim=1)
        x_hsi_exclusive = self.expert_hsi(hsi_feature)
        x_lidar_exclusive = self.expert_lidar(lidar_feature)



        unique_feature_fusion=[]
        for i in range(len(x_hsi_exclusive)):
            unique_feature_fusion.append(route[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*x_hsi_exclusive[i]+route[:,1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*x_lidar_exclusive[i])
        unique_feature_fusion = tuple(unique_feature_fusion)
        outs = []
        for i in range(len(common_feature)):
            outs.append(0.7*common_feature[i]+0.3*unique_feature_fusion[i])

        return outs

