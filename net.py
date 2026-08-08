import numbers
from einops import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(torch.nn.Module):
    def __init__(self, num_filter):
        super(SELayer, self).__init__()
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv_double = torch.nn.Sequential(
            torch.nn.Conv2d(num_filter, num_filter // 16, 1, 1, 0, bias=True),
            torch.nn.ReLU(),
            torch.nn.Conv2d(num_filter // 16, num_filter, 1, 1, 0, bias=True),
            torch.nn.Sigmoid())

    def forward(self, x):
        mask = self.global_pool(x)
        mask = self.conv_double(mask)
        x = x * mask
        return x


class ResBlock(nn.Module):
    def __init__(self, num_filter):
        super(ResBlock, self).__init__()
        body = []
        for i in range(2):
            body.append(nn.ReflectionPad2d(1))
            body.append(nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=0))
            if i == 0:
                body.append(nn.ReLU())
        body.append(SELayer(num_filter))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        x = res + x
        return x


class L_net(nn.Module):
    def __init__(self, num = 64):
        super(L_net, self).__init__()

        self.L_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),                          
            ResBlock(num),
            ResBlock(num),  
            ResBlock(num),
            ResBlock(num),             
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),                                        
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),  
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 1, 3, 1, 0),
        )

    def forward(self, input):
        return torch.sigmoid(self.L_net(input))


class R_net(nn.Module):
    def __init__(self, num = 64):
        super(R_net, self).__init__()

        self.R_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, num, 3, 1, 0),
            nn.ReLU(),     
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),  
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),              
            ResBlock(num),
            ResBlock(num), 
            ResBlock(num),
            ResBlock(num),                         
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),   
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),  
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 3, 3, 1, 0),
        )

    def forward(self, input):
        return torch.sigmoid(self.R_net(input))

class enhance_net(nn.Module):
    def __init__(self):
        super(enhance_net, self).__init__()
        self.L_net = L_net(num=64)
        self.R_net = R_net(num=64)

    def forward(self, input):
        L = self.L_net(input)
        R = self.R_net(input)
        return L, R
