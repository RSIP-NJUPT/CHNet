#coding:utf-8
from torch import nn
import torch
import torch.nn.functional as F

from config import opt
from torch.nn import Parameter as Parameter




class SC(nn.Module):
    def __init__(self,beta):
        super(SC, self).__init__()
        self.device=torch.device("cuda")
        self.beta=beta

    def forward(self, input):
       
        zero = torch.zeros(input.shape).to(self.device)
        output = torch.mul(torch.sign(input),torch.max((torch.abs(input)-self.beta/2),zero))
        

        return output
class FHFL(nn.Module):
    """ FHFL Module
    Args:
        x (Tensor): The Mutiple-scale feature
    """
    def __init__(self,size):  
        super(FHFL, self).__init__()
        
        self.output_dim = self.JOINT_EMB_SIZE = opt.RANK_ATOMS * opt.NUM_CLUSTER 
        self.input_dim = opt.down_chennel

        self.Linear_dataproj_k = nn.Linear(opt.down_chennel, self.JOINT_EMB_SIZE)
        self.Linear_dataproj2_k = nn.Linear(opt.down_chennel, self.JOINT_EMB_SIZE) 

        self.Linear_predict = nn.Linear(opt.NUM_CLUSTER, opt.class_num)

        self.sc = SC(beta=opt.BETA)
        
        self.Avgpool = nn.AvgPool1d(kernel_size=size*size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data,)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
        

    def forward(self, x): 
        bs, c, w, h = x.shape[0:4]

        bswh = bs*w*h
        x = x.permute(0,2,3,1)                  
        x = x.contiguous().view(-1,c)           

        x1 = self.Linear_dataproj_k(x)             
        x2 = self.Linear_dataproj2_k(x)

        bi = x1.mul(x2)  

        bi = bi.view(-1, 1, opt.NUM_CLUSTER, opt.RANK_ATOMS)       
        bi = torch.squeeze(torch.sum(bi, 3))                        

        bi = self.sc(bi)

        bi = bi.view(bs,h*w,-1)                                     
        bi = bi.permute(0,2,1)    
        bi = self.Avgpool(bi)                                  
        bi = torch.squeeze(bi)                   

        b2 = torch.sqrt(F.relu(bi)) - torch.sqrt(F.relu(-bi))      
        b3 = F.normalize(b2, p=2, dim=1)

        y = self.Linear_predict(b3) 
        return y


