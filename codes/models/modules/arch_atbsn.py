import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class crop(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.shape
        x = x[0:N, 0:C, 0:H-1, 0:W] # crop last row
        return x

class shift(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift_down = nn.ZeroPad2d((0,0,1,0))
        self.crop = crop()

    def forward(self, x):
        x = self.shift_down(x)
        x = self.crop(x)
        return x

class super_shift(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, x, hole_size=1):
        shift_offset = (hole_size+1)//2 # hole_size must be = 1, 3, 5, 7...

        x = nn.ZeroPad2d((0,0,shift_offset,0))(x) # left right top bottom
        N, C, H, W = x.shape
        x = x[0:N, 0:C, 0:H-shift_offset, 0:W] # crop last rows
        return x

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, blind=True,stride=1, padding=1, kernel_size=3):
        super().__init__()
        self.blind = blind
        if blind:
            self.shift_down = nn.ZeroPad2d((0,0,1,0)) # left right top bottom
            self.crop = crop()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias) 
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        if self.blind:
            x = self.shift_down(x)
        x = self.conv(x)
        if self.blind:
            x = self.crop(x)
        x = self.relu(x)        
        return x

class Pool(nn.Module):
    def __init__(self, blind=True):
        super().__init__()
        self.blind = blind
        if blind:
            self.shift = shift()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        if self.blind:
            x = self.shift(x)
        x = self.pool(x)
        return x

class rotate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x90 = x.transpose(2,3).flip(3)
        x180 = x.flip(2).flip(3)
        x270 = x.transpose(2,3).flip(2)
        x = torch.cat((x,x90,x180,x270), dim=0)
        return x

class unrotate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x0, x90, x180, x270 = torch.chunk(x, 4, dim=0)
        x90 = x90.transpose(2,3).flip(2)
        x180 = x180.flip(2).flip(3)
        x270 = x270.transpose(2,3).flip(3)
        x = torch.cat((x0,x90,x180,x270), dim=1)
        return x

class ENC_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, reduce=True, blind=True):
        super().__init__()
        self.reduce = reduce
        self.conv1 = Conv(in_channels, out_channels, bias=bias, blind=blind)
        if reduce:
            self.pool = Pool(blind=blind)

    def forward(self, x):
        x = self.conv1(x)
        if self.reduce:
            x = self.pool(x)
        return x

class DEC_Conv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bias=True, increase=True, blind=True):
        super().__init__()
        self.increase = increase
        self.conv1 = Conv(in_channels, mid_channels, bias=bias, blind=blind)
        self.conv2 = Conv(mid_channels, out_channels, bias=bias, blind=blind)
        if increase:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, x_in):
        x = torch.cat((x, x_in), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        if self.increase:
            x = self.upsample(x)
        return x

class Blind_UNet(nn.Module):
    def __init__(self, n_channels=3, mid_channels=48, n_output=96, bias=True, blind=True):
        super().__init__()
        self.intro = Conv(n_channels, mid_channels, bias=bias, blind=blind)
        self.enc1 = ENC_Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.enc2 = ENC_Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.enc3 = ENC_Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.enc4 = ENC_Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.enc5 = ENC_Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.enc6 = ENC_Conv(mid_channels, mid_channels, bias=bias, reduce=False, blind=blind)
        self.dec5 = DEC_Conv(mid_channels*2, mid_channels*2, mid_channels*2, bias=bias, blind=blind)
        self.dec4 = DEC_Conv(mid_channels*3, mid_channels*2, mid_channels*2, bias=bias, blind=blind)
        self.dec3 = DEC_Conv(mid_channels*3, mid_channels*2, mid_channels*2, bias=bias, blind=blind)
        self.dec2 = DEC_Conv(mid_channels*3, mid_channels*2, mid_channels*2, bias=bias, blind=blind)
        self.dec1 = DEC_Conv(mid_channels*2+n_channels, mid_channels*2, n_output, bias=bias, increase=False, blind=blind)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, input):
        x = self.intro(input)
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x = self.enc6(x5)
        x = self.upsample(x)
        x = self.dec5(x, x4)
        x = self.dec4(x, x3)
        x = self.dec3(x, x2)
        x = self.dec2(x, x1)
        x = self.dec1(x, input)
        return x

# class Blind_UNet(nn.Module):
#     def __init__(self, n_channels=3, mid_channels=48, n_output=96, bias=True, blind=True):
#         super().__init__()
#         self.intro = Conv(n_channels, mid_channels, bias=bias, blind=blind)
        
#         self.enc_layers = nn.ModuleList([
#             ENC_Conv(mid_channels, mid_channels, bias=bias, blind=blind) for _ in range(6)
#         ])
        
#         dec_channels = [
#             (mid_channels*2, mid_channels*2),  # dec5
#             (mid_channels*3, mid_channels*2),  # dec4
#             (mid_channels*3, mid_channels*2),  # dec3
#             (mid_channels*3, mid_channels*2),  # dec2
#             (mid_channels*2+n_channels, n_output)  # dec1
#         ]
        
#         self.dec_layers = nn.ModuleList([
#             DEC_Conv(in_channels, out_channels, bias=bias, blind=blind)
#             for in_channels, out_channels in dec_channels
#         ])
        
#         self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

#     def forward(self, input):
#         x = self.intro(input)
#         enc_outputs = [x]
        
#         for enc_layer in self.enc_layers[:-1]:
#             x = enc_layer(x)
#             enc_outputs.append(x)
        
#         # 最后一个编码层不存储输出
#         x = self.enc_layers[-1](x)
#         x = self.upsample(x)
        
#         for i, dec_layer in enumerate(self.dec_layers):
#             skip_connection = enc_outputs[-2-i] if i < len(self.enc_layers) else input
#             x = dec_layer(x, skip_connection)
        
#         return x
        
class ATBSN(nn.Module):  # 1.27m
    def __init__(self, n_channels=3, mid_channels=48, n_output=3, bias=True, blind=True):
        super().__init__()
        self.blind = blind
        self.rotate = rotate()
        self.unet = Blind_UNet(n_channels=n_channels, mid_channels=mid_channels, n_output=mid_channels*2, bias=bias, blind=blind)
        self.shift = super_shift()
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(mid_channels*8, mid_channels*8, 1, bias=bias)
        self.nin_B = nn.Conv2d(mid_channels*8, mid_channels*2, 1, bias=bias)
        self.nin_C = nn.Conv2d(mid_channels*2, n_output, 1, bias=bias)

        with torch.no_grad():
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0.1)
                m.bias.data.zero_()
        nn.init.kaiming_normal_(self.nin_C.weight.data, nonlinearity="linear")
    
    def forward(self, x, hole_size=1):              
        x = self.rotate(x)
                
        x = self.unet(x)
        if self.blind:
            x = self.shift(x, hole_size)
        x = self.unrotate(x)

        x0 = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x0 = F.leaky_relu_(self.nin_B(x0), negative_slope=0.1)
        x0 = self.nin_C(x0)
            
        return x0
    
class N_BSN(nn.Module): #student c, 1.00m (1.02m in the paper is a typo)
    def __init__(self, n_channels=3, mid_channels=48, n_output=3, bias=True, blind=False):
        super().__init__()
        self.unet = Blind_UNet(n_channels=n_channels, mid_channels=mid_channels, n_output=n_output, bias=bias, blind=blind)

        with torch.no_grad():
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0.1)
                m.bias.data.zero_()
    
    def forward(self, x):                              
        x0 = self.unet(x)
            
        return x0

# from .arch_atbsn_small import Blind_UNet_small
# from .arch_atbsn_small2 import Blind_UNet_small2

class DoubleNet(nn.Module):
    def __init__(self):
        super(DoubleNet, self).__init__()
        # self.nbsn = Blind_UNet_small() #student a
        # self.nbsn = Blind_UNet_small2() #student b
        self.nbsn = N_BSN() #student c
        self.bsn = ATBSN()
        # freeze bsn
        for param in self.bsn.parameters():
            param.requires_grad = False
        self.bsn.eval()
        
    def forward(self, x, hole_size=[0,1,3,5,7,9,11], mode="test"):
        if mode == "train":
            res_list = []
            with torch.no_grad():
                for hs in hole_size:
                    res_list.append(self.bsn(x, hs))
            x_atbsn = self.nbsn(x)
            return x_atbsn, res_list
        elif mode == "test":
            x_atbsn = self.nbsn(x)
            return x_atbsn

if __name__ == '__main__':
    # model = N_BSN(n_channels=3, n_output=3, bias=True, blind=False)
    model = ATBSN(n_channels=3, n_output=3, bias=True, blind=True)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))