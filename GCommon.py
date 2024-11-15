import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from config import opt

class Conv_Feature(nn.Module):
    def __init__(self, in_channel , out_channels = [64, 128, 256, 512]):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channels
        self.conv_1 = nn.Conv2d(self.in_channel, out_channels[0], 3, 1, 0)
        self.conv_2 = nn.Conv2d(out_channels[0], out_channels[1], 3, 1, 0)
        self.conv_3 = nn.Conv2d(out_channels[1], out_channels[2], 3, 1, 0)
        self.conv_4 = nn.Conv2d(out_channels[2], out_channels[3], 3, 1, 0)
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.bn3 = nn.BatchNorm2d(out_channels[2])
        self.bn4 = nn.BatchNorm2d(out_channels[3])
        self.relu = nn.ReLU()

    def forward(self, x):
        c1 = self.conv_1(x)    
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c2 = self.conv_2(c1)
        c2 = self.bn2(c2)
        c2 = self.relu(c2)
        c3 = self.conv_3(c2)
        c3 = self.bn3(c3)
        c3 = self.relu(c3)
        c4 = self.conv_4(c3)
        c4 = self.bn4(c4)
        c4 = self.relu(c4)
        outputs=[]
        outputs.append(c1)
        outputs.append(c2)
        outputs.append(c3)
        outputs.append(c4)
        return outputs


class FPNBlock(nn.Module):
    def __init__(self,out_c):
        super(FPNBlock, self).__init__()
        self.top = nn.Conv2d(512, out_c, kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d(256,out_c,1,1,0)
        self.latlayer2 = nn.Conv2d(128,out_c,1,1,0)
        self.latlayer3 = nn.Conv2d(64,out_c,1,1,0)

        self.smooth1 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(1,2)
    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y

    def forward(self, feature):
        outputs = []
        c1 = feature[0]
        c2 = feature[1]
        c3 = feature[2]
        c4 = feature[3]
        p5 =self.top(c4)
        p4 = self._upsample_add(p5,self.latlayer1(c3))
        p3 = self._upsample_add(p4,self.latlayer2(c2))
        p2 = self._upsample_add(p3,self.latlayer3(c1))
        p6 = self.maxpool(p5)
        p4 = self.smooth3(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth1(p2)
        outputs.append(p2)
        outputs.append(p3)
        outputs.append(p4)
        outputs.append(p5)
        outputs.append(p6)

        return outputs

def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


class Extract_Edge(nn.Module):
    def __init__(self,in_c,out_c,num_features):
        super(Extract_Edge, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)
        self.conv_op =  nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        #设置Laplacian算子
        self.sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 9
        self.sobel_kernel = self.sobel_kernel.reshape((1, 1, 3, 3))
        # 输入图的通道
        self.sobel_kernel = np.repeat(self.sobel_kernel, in_c, axis=1)
        # 卷积输出通道
        self.sobel_kernel = np.repeat(self.sobel_kernel, out_c, axis=0)
        self.conv_op.weight.data = torch.from_numpy(self.sobel_kernel)
        freeze(self.conv_op)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,img):
        img = self.bn(img)
        edge_detect = self.conv_op(img)
        edge_detect = self.relu(edge_detect)
        return edge_detect


class CommonFeatureGenerator(nn.Module):
    """ Mask Generator

    Args:
        img_hsi (Tensor): The hsi input
        img_lidar (Tensor): The lidar input
    """

    def __init__(self):
        super(CommonFeatureGenerator,self).__init__()
        self.EE_h = Extract_Edge(opt.pca_components,3,opt.pca_components)
        self.EE_l = Extract_Edge(32,3,32)

        self.backbone_hsi = Conv_Feature(in_channel=opt.pca_components)
        self.backbone_lidar = Conv_Feature(in_channel=32)
        self.neck_hsi  = FPNBlock(64)
        self.neck_lidar = FPNBlock(64)
        input_size = opt.patch_size
        self.strides = []
        target_sizes = [(opt.patch_size-2,opt.patch_size-2), (opt.patch_size-4, opt.patch_size-4), (opt.patch_size-6, opt.patch_size-6), (opt.patch_size-8, opt.patch_size-8), ((opt.patch_size-8)/2 + 1,(opt.patch_size-8)/2 + 1)]
        for target_size in target_sizes:
            stride = input_size / target_size[0]
            self.strides.append(stride)


    def edge_fusion(self,hsi_edge,lidar_edge):
        #simple fusion
        fused_edge = hsi_edge + lidar_edge
        fused_edge = fused_edge/fused_edge.std()
        return fused_edge


    def forward(self, img_hsi,img_lidar):
        """Forward function."""

        common_features = []
        #获取边缘特征
        x_hsi = self.backbone_hsi(img_hsi)
        hsi_edge = self.EE_h(img_hsi)
        lidar_edge = self.EE_l(img_lidar)
        fused_edge = self.edge_fusion(hsi_edge,lidar_edge)

        x_lidar = self.backbone_lidar(img_lidar)
        fused_edge = 0.05*fused_edge



        hsi_feature = self.neck_hsi(x_hsi)
        lidar_feature = self.neck_lidar(x_lidar)
        # 加入边缘信息
        img_fused_edge_scale_list = []
        for stride in self.strides:
            img_fused_edge_down = F.interpolate(fused_edge, scale_factor=1/stride, mode='bicubic')
            img_fused_edge_scale_list.append(img_fused_edge_down)
        x_common_list = []

        for i in range(len(hsi_feature)):
            x_common = 0.5*(hsi_feature[i]+lidar_feature[i])
            x_common = torch.cat([x_common,img_fused_edge_scale_list[i]],dim=1)
            x_common_list.append(x_common)
        common_features= tuple(x_common_list)

        return common_features

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = CommonFeatureGenerator().to(device)
    t1 = torch.rand(1, 1, 30, 11, 11).to(device)
    t2 = torch.rand(1, 1, 11, 11).to(device)
    output = net(t1, t2)