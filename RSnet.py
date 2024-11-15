import warnings

import torch
from torch import nn
import GMask
import GCommon
import GFusion
import FHFL
from einops import rearrange
import torch.nn.functional as F
from config import opt


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num:int, 
                 group_num:int , 
                 eps:float = 1e-10
                 ):
        super(GroupBatchnorm2d,self).__init__()
        assert c_num    >= group_num
        self.group_num  = group_num
        self.weight     = nn.Parameter( torch.randn(c_num, 1, 1)    )
        self.bias       = nn.Parameter( torch.zeros(c_num, 1, 1)    )
        self.eps        = eps
    def forward(self, x):
        N, C, H, W  = x.size()
        x           = x.view(   N, self.group_num, -1   )
        mean        = x.mean(   dim = 2, keepdim = True )
        std         = x.std (   dim = 2, keepdim = True )
        x           = (x - mean) / (std+self.eps)
        x           = x.view(N, C, H, W)
        return x * self.weight + self.bias

class SR(nn.Module):
    def __init__(self,
                 oup_channels:int, 
                 group_num:int = 1,
                 gate_treshold:float = 0.5,
                 torch_gn:bool = True
                 ):
        super().__init__()
        
        self.gn             = nn.GroupNorm( num_channels = oup_channels, num_groups = group_num ) if torch_gn else GroupBatchnorm2d(c_num = oup_channels, group_num = group_num)
        self.gate_treshold  = gate_treshold
        self.sigomid        = nn.Sigmoid()

    def forward(self,x):
        gn_x        = self.gn(x)
        w_gamma     = self.gn.weight/sum(self.gn.weight)
        w_gamma     = w_gamma.view(1,-1,1,1)
        reweigts    = self.sigomid( gn_x * w_gamma )
        # Gate
        w1          = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts) # 大于门限值的设为1，否则保留原值
        w2          = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts) # 大于门限值的设为0，否则保留原值
        x_1         = w1 * x
        x_2         = w2 * x
        y           = self.reconstruct(x_1,x_2)
        return y
    
    def reconstruct(self,x_1,x_2):
        x_11,x_12 = torch.split(x_1, x_1.size(1)//2, dim=1)
        x_21,x_22 = torch.split(x_2, x_2.size(1)//2, dim=1)
        return torch.cat([ x_11+x_22, x_12+x_21 ],dim=1)


class RSnet(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone_hsi = GCommon.Conv_Feature(opt.pca_components)
        self.backbone_lidar = GCommon.Conv_Feature(32)
        self.sr = SR(32)
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size= 1,stride= 1 )
        self.Gmask = GMask.UniqueMaskGenerator()
        self.Gcommon = GCommon.CommonFeatureGenerator()
        self.FeaFusion = GFusion.ConvFusion()
        self.flatten = nn.Flatten()
        self.fhfl_1 = FHFL.FHFL(opt.patch_size - 2)
        self.fhfl_2 = FHFL.FHFL(opt.patch_size - 4)
        self.fhfl_3 = FHFL.FHFL(opt.patch_size - 6)
        self.fhfl_4 = FHFL.FHFL(opt.patch_size - 8)
        self.fhfl_5 = FHFL.FHFL((int)((opt.patch_size-8)/2+1))
        
    def forward(self, img_hsi, img_lidar):
        """Network forward process. """
        img_hsi =  rearrange(img_hsi, 'b c h w y ->b (c h) w y')
        mask_hsi = self.Gmask(img_hsi)
        ##### EXTRACT_unique_feature
        #SRR module
        hsi_fre = torch.fft.fft2(img_hsi)
        fre_m_hsi = torch.abs(hsi_fre)
        fre_m_hsi = torch.fft.fftshift(fre_m_hsi)
        fre_p_hsi = torch.angle(hsi_fre)
        masked_fre_m_hsi = fre_m_hsi * mask_hsi
        masked_fre_m_hsi = torch.fft.ifftshift(masked_fre_m_hsi)
        fre_hsi = masked_fre_m_hsi * torch.e ** (1j * fre_p_hsi)
        img_hsi_unique = torch.real(torch.fft.ifft2(fre_hsi))


        img_lidar = self.conv1(img_lidar)
        img_lidar_unique = self.sr(img_lidar)
        x_common = self.Gcommon(img_hsi_unique, img_lidar_unique)

        x_hsi = self.backbone_hsi(img_hsi_unique)
        x_lidar = self.backbone_lidar(img_lidar_unique)
        x = self.FeaFusion(x_hsi, x_lidar, x_common, img_hsi, img_lidar)
        ####fhfl
        y1 = self.fhfl_1(x[0])
        y2 = self.fhfl_2(x[1])
        y3 = self.fhfl_3(x[2])
        y4 = self.fhfl_4(x[3])
        y5 = self.fhfl_5(x[4])
        x = y1+y2+y3+y4+y5
        return x





