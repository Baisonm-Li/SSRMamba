import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from mamba import VMambaBlock,SS2DBlcok,VSA

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ss2d = VMambaBlock(in_channels)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.ss2d(x)
        return self.maxpool_conv(x)

    
class UP(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(dim,dim,1,1,groups=dim),
            nn.Conv2d(dim,dim // 2,1,1),
            nn.ReLU(inplace=True)
        )
        self.dim = dim
        self.ss2d = VMambaBlock(dim // 2)
    def forward(self, x):
        
        x = F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=True)
        x = self.conv(x)
        x = self.ss2d(x)
        return x
    

class SpatailMamaba(nn.Module):
    def __init__(self,dim):
        super(SpatailMamaba, self).__init__()
        self.ss2d1 = SS2DBlcok(dim)
        self.conv1 = nn.Conv2d(dim,dim,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(dim,dim,kernel_size=5,padding=2)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(dim * 2,dim * 2,kernel_size=3,padding=1,groups=dim),
            nn.Conv2d(dim * 2,dim,kernel_size=1,padding=0)
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=3,padding=1,groups=dim),
            nn.Conv2d(dim,dim,kernel_size=1,padding=0)
        )
        
    def forward(self, x):
        res = x
        x = F.relu(self.dwconv1(F.relu(torch.cat([self.conv1(x),self.conv2(x)],dim=1))))
        x = F.layer_norm(x,x.shape[1:])
        x = self.ss2d1(x)
        x = F.relu(self.dwconv2(x))
        return x + res

class SpectralMamaba(nn.Module):
    def __init__(self,dim):
        super(SpectralMamaba, self).__init__()
        self.linear1 = nn.Linear(dim,dim)
        self.linear2 = nn.Linear(dim,dim)
        self.ssd = SS2DBlcok(dim)
        self.dwconv1 = nn.Sequential(
        nn.Conv2d(dim,dim,kernel_size=3,padding=1,groups=dim),
        nn.Conv2d(dim,dim,kernel_size=1,padding=0)
    )
        self.dwconv2 = nn.Sequential(
        nn.Conv2d(dim,dim,kernel_size=3,padding=1,groups=dim),
        nn.Conv2d(dim,dim,kernel_size=1,padding=0)
    )
    def forward(self, x):
        res = x
        score = F.adaptive_max_pool2d(x,(1,1))
        score = rearrange(score,'b c 1 1 -> b c')
        score = F.relu(self.linear1(score))
        score = F.relu(self.linear2(score))
        score = rearrange(score,'b c -> b c 1 1')

        x = F.relu(self.dwconv1(x))
        x = F.layer_norm(x,x.shape[1:])
        x = self.ssd(x)
        x = x * score
        x = F.relu(self.dwconv2(x))
        return x + res
    
    
class SSRMamba(nn.Module):
    def __init__(self,hsi_bands=31,dim=128):
        super(SSRMamba, self).__init__()
        self.hsi_bands = hsi_bands
        self.dim = dim
        self.head = nn.Sequential(
            nn.Conv2d(3,dim,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )
        self.spatail_mambas = nn.ModuleList([SpatailMamaba(dim) for _ in range(1)])
        self.vssblocks = nn.ModuleList([VMambaBlock(dim) for _ in range(6)])
        self.spectral_mambas = nn.ModuleList([SpectralMamaba(dim) for _ in range(1)])
    
        self.tail = nn.Sequential(
            nn.Conv2d(dim,dim,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim,hsi_bands,3,1,1)
            )
        
    def forward(self, x):
        b,c,h,w = x.shape
        shortcut = torch.zeros_like(b,self.hsi_bands,h,w)
        shortcut[:c] = x
        x = self.head(x)
        for spatial_mamba in self.spatail_mambas:
            x = spatial_mamba(x)
        for vssblock in self.vssblocks:
            x = vssblock(x)
        for spectral_mamba in self.spectral_mambas:
            x = spectral_mamba(x)
        x = self.tail(x) # b hsi_bands h w 
        return x + shortcut 
