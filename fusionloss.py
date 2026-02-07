import torch.nn
from torch import optim
from torch.utils.data import DataLoader
import os
import torch
from torch import nn
from tqdm import tqdm
from decom import enhance_d
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np

class fusionloss(nn.Module):
    def __init__(self):
        super(fusionloss, self).__init__()
        self.sobelconv = Sobelxy()
    def forward(self, image_vis, image_ir, generate_img):
        y_grad = self.sobelconv(image_vis)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
        return loss_grad

def angle(a, b):
    vector = torch.multiply(a, b)
    up = torch.sum(vector)
    down = torch.sqrt(torch.sum(torch.square(a))) * torch.sqrt(torch.sum(torch.square(b)))
    theta = torch.acos(up / down)
    return theta

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


            