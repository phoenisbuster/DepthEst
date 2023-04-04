import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from pytorch_msssim import ssim
import utils

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AP_Loss(nn.Module):
    def __init__(self, alpha) -> None:
        super().__init__()
        self.alpha = alpha

    def gradient_x(self, img):
        img = F.pad(img, (0, 1, 0, 0), mode='replicate')
        grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
        return grad_x

    def gradient_y(self, img):
        img = F.pad(img, (0, 0, 0, 1), mode='replicate')
        grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
        return grad_y
    
    def forward(self, disp, target_img, wrap_img):
        # dist = torch.histc(abs_loss, bins=50, min=0, max=5)
        # bins = torch.linspace(0, 5, 50)
        # dist_map = 1 - utils.get_dist(loss=abs_loss, dist=dist.int(), bins=bins)/dist.max()

        target_img_gradients_x = torch.abs(self.gradient_x(target_img))
        target_img_gradients_y = torch.abs(self.gradient_y(target_img))
        # target_img_gradients_y = torch.abs(self.gradient_y(target_img))
        wrap_img_gradients_x = torch.abs(self.gradient_x(wrap_img))
        wrap_img_gradients_y = torch.abs(self.gradient_y(wrap_img))
        # wrap_img_gradients_y = torch.abs(self.gradient_y(wrap_img))

        # weight_x = 1 - torch.exp(-3*torch.abs(self.gradient_x(target_img))) * (1 - torch.exp(-5*torch.abs(self.gradient_x(disp))))
        # weight_y = 1 - torch.exp(-3*torch.abs(self.gradient_y(target_img))) * (1 - torch.exp(-5*torch.abs(self.gradient_y(disp))))
        weight = 1

        loss =  (
            self.alpha*(
                (1-ssim(wrap_img,target_img,1))/2
            ) + \
            (1-self.alpha)*(
                # (torch.abs(target_img - wrap_img)*weight).mean() + \
                # (torch.abs(wrap_img_gradients_x - target_img_gradients_x)*weight).mean()# + \
                # (torch.abs(wrap_img_gradients_y - target_img_gradients_y)*weight).mean()
                F.l1_loss(wrap_img,target_img) + \
                F.l1_loss(wrap_img_gradients_x,target_img_gradients_x) + \
                F.l1_loss(wrap_img_gradients_y,target_img_gradients_y)
            )
        )
    
        return loss


class DOOR_Loss(nn.Module):
    def __init__(self, threshold) -> None:
        super(DOOR_Loss, self).__init__()
        self.threshold = threshold

    def forward(self, disp):
        mask = ((-disp < 0)*(-disp > self.threshold))*1.0
        return (torch.abs(disp)*mask).mean()


class DS_Loss(nn.Module):
    def __init__(self) -> None:
        super(DS_Loss, self).__init__()

    def gradient_x(self, img):
        img = F.pad(img, (0, 1, 0, 0), mode='replicate')
        grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
        return grad_x

    def gradient_y(self, img):
        img = F.pad(img, (0, 0, 0, 1), mode='replicate')
        grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
        return grad_y

    def forward(self, disp, img):
        smoothness_x = torch.abs(self.gradient_x(disp)) * torch.exp(-torch.abs(self.gradient_x(img)))
        smoothness_y = torch.abs(self.gradient_y(disp)) * torch.exp(-torch.abs(self.gradient_y(img)))
        smoothness_loss = (smoothness_x + smoothness_y).mean()

        return smoothness_loss


class LR_Loss(nn.Module):
    def __init__(self, alpha=0.84) -> None:
        super().__init__()
        self.alpha = alpha

    def gradient_x(self, img):
        img = F.pad(img, (0, 1, 0, 0), mode='replicate')
        grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
        return grad_x

    def gradient_y(self, img):
        img = F.pad(img, (0, 0, 0, 1), mode='replicate')
        grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
        return grad_y
    
    def forward(self, target_img, wrap_img):
        return F.l1_loss(wrap_img,target_img)