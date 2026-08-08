import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from utils import *


def gradient(img):
    height = img.size(2)
    width = img.size(3)
    gradient_h = (img[:,:,2:,:]-img[:,:,:height-2,:]).abs()
    gradient_w = (img[:, :, :, 2:] - img[:, :, :, :width-2]).abs()
    gradient_h = F.pad(gradient_h, [0, 0, 1, 1], 'replicate')
    gradient_w = F.pad(gradient_w, [1, 1, 0, 0], 'replicate')
    return gradient_h*gradient_h, gradient_w*gradient_w

def normalize(x):
    minv = torch.min(x)
    maxv = torch.max(x)
    return (x-minv)/(maxv - minv + 0.000001)


def rec_loss(image, illumination, reflectance):#重建损失
    return torch.nn.MSELoss()(image, illumination * reflectance)


def ill_loss(image, illumination):
    g_kernel_size = 11
    pad = 5
    sigma = 3
    kx = cv2.getGaussianKernel(g_kernel_size,sigma)
    ky = cv2.getGaussianKernel(g_kernel_size,sigma)
    gaussian_kernel = np.multiply(kx,np.transpose(ky))
    gaussian_kernel = torch.FloatTensor(gaussian_kernel).unsqueeze(0).unsqueeze(0).cuda()

    gray_tensor = 0.299*image[0,0,:,:] + 0.587*image[0,1,:,:] + 0.114*image[0,2,:,:]
    gradient_gray_h, gradient_gray_w = gradient(gray_tensor.unsqueeze(0).unsqueeze(0))
    gradient_illu_h, gradient_illu_w = gradient(illumination)

    weight_h = 1/(F.conv2d(gradient_gray_h, weight=gaussian_kernel, padding=pad)+1e-5)
    weight_w = 1/(F.conv2d(gradient_gray_w, weight=gaussian_kernel, padding=pad)+1e-5)
    weight_h.detach()
    weight_w.detach()
    loss_h = weight_h * gradient_illu_h
    loss_w = weight_w * gradient_illu_w

    max_rgb, _ = torch.max(image, 1)
    max_rgb = max_rgb.unsqueeze(1)

    gray_tensor1 = 0.2126*image[0,0,:,:] + 0.7152*image[0,1,:,:] + 0.0722*image[0,2,:,:]
    weighted_rgb = gray_tensor1.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]

    loss1 = torch.nn.MSELoss()(illumination, weighted_rgb) # L0
    loss2 = loss_h.mean() + loss_w.mean() # TV
    return loss1, loss2

class R_exp(nn.Module):
    def __init__(self, mean_val=0.5):#ideal exposure v = 0.5
        super(R_exp, self).__init__()
        self.mean_val = mean_val
        self.avg_pool = nn.AvgPool2d(16, stride=16)
    def forward(self, x, w):
        x = torch.mean(x, 1, keepdim=True)
        w = normalize(w)
        x = self.avg_pool(x)
        w = self.avg_pool(w)
        tg = torch.ones_like(x) * self.mean_val
        d = (x - tg) ** 2
        d = d * (1 + w)
        d = torch.mean(d)
        return d


def ref_loss(image, illumination, reflectance):
    refrence_reflect = image/illumination
    loss1 = torch.nn.MSELoss()(reflectance, refrence_reflect)
    loss2 = R_exp()(reflectance, illumination.detach())
    return loss1, loss2
