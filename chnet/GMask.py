import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from config import opt


class TinyGMask_h(nn.Module):
    def __init__(self, patch_num ):
        super(TinyGMask_h, self).__init__()
        self.patch_num = patch_num
        self.conv1 = nn.Conv2d(opt.pca_components, 64, 3, 1, 1)
        self.flatten = nn.Flatten()
        self.trans2list = nn.Sequential(            
            nn.Linear(int(64 * patch_num * patch_num), 1000),
            nn.Linear(in_features=1000, out_features=np.power(self.patch_num, 2) * opt.pca_components, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        x=self.conv1(x)
        x_final = self.trans2list(self.flatten(x))
        return x_final

class MaskFunction_h(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mask_list_):
        mask_list_topk ,topk_index = mask_list_.topk(int(0.8*opt.pca_components*opt.patch_size*opt.patch_size))
        mask_list_min = torch.min(mask_list_topk, dim=1).values
        mask_list_min_ = mask_list_min.unsqueeze(-1)
        ge = torch.ge(mask_list_, mask_list_min_)
        zero = torch.zeros_like(mask_list_)
        one = torch.ones_like(mask_list_)
        mask_list = torch.where(ge, one, zero)
        return mask_list

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output



class GMaskBinaryList_h(nn.Module):
    def __init__(self,
                 patch_num
                 ):
        super( GMaskBinaryList_h, self).__init__()

        self.g_mask_binary_list = TinyGMask_h(patch_num)
    def forward(self, x):
        mask_list = self.g_mask_binary_list(x)
        return mask_list




class UniqueMaskGenerator(nn.Module):
    """
    Args:
        patch_num (int): raw or column patch number
        keep_low (bool):
    """
    def __init__(self,
                 patch_num=opt.patch_size,
                 ):
        super(UniqueMaskGenerator, self).__init__()
        self.patch_num = patch_num

        self.Gmaskbinarylist_hsi = GMaskBinaryList_h(patch_num)

        self.MaskFun_hsi = MaskFunction_h()

        self.flatten = nn.Flatten()
    def forward(self, img_hsi):
        """Forward function."""

        hsi_fre = torch.fft.fft2(img_hsi)
        fre_m_hsi = torch.abs(hsi_fre)  # 幅度谱，求模得到
        fre_m_hsi = torch.fft.fftshift(fre_m_hsi)


       
        mask_hsi_list_ = self.Gmaskbinarylist_hsi(fre_m_hsi)
        mask_hsi_list = self.MaskFun_hsi.apply(mask_hsi_list_).reshape((-1, opt.pca_components,self.patch_num , self.patch_num))
        mask_hsi_list[:, :, int(self.patch_num / 2) - 1:int(self.patch_num / 2) + 1,int(self.patch_num / 2) - 1:int(self.patch_num / 2) + 1] = 1
        
        mask_hsi = F.interpolate(mask_hsi_list, scale_factor=[self.patch_num / self.patch_num, self.patch_num / self.patch_num], mode='nearest')
        
        return mask_hsi
