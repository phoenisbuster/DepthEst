import torch
import torch.nn as nn
import torch.nn.functional as F
import functorch
import numpy as np

def readlines(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def scale_disparity(disp, min_depth, max_depth):
    min_disp = 1/max_depth
    max_disp = 1/min_depth
    
    scaled_disp = (disp - min_disp)/(max_disp - min_disp)
    return scaled_disp

def wrap_image(disparity, images, device='cuda:0'):
    img_height, img_width = images.shape[2:4]
    h_flow = disparity
    v_flow = torch.nn.Parameter(torch.zeros((1, img_height, img_width))).to(device=device)
    v_flow = v_flow * h_flow

    # coordinate grid, normalized to [-1, 1] to fit into grid_sample
    coord_x = np.tile(range(img_width), (img_height, 1)) / ((img_width-1)/2) - 1
    coord_y = np.tile(range(img_height), (img_width, 1)).T / ((img_height-1)/2) - 1
    grid = np.stack([coord_x, coord_y])
    grid = torch.Tensor(grid).permute(1,2,0)
    grid = torch.nn.Parameter(grid, requires_grad=False).to(device=device)

    # warping transformation
    trans = torch.cat([h_flow, v_flow], dim=1)
    grid_warp = grid + trans.permute(0,2,3,1)

    # back warping
    img_warp = F.grid_sample(images, grid_warp, padding_mode="border", align_corners=False)

    return img_warp

def post_process_disparity(disp, disp_prime, device='cuda:0'):
    _, _, h, w = disp.shape
    m_disp = 0.5*(disp + disp_prime)

    l, _ = torch.meshgrid([torch.linspace(0, 1, w), torch.linspace(0, 1, h)], indexing='xy')
    l_mask = (1.0 - torch.clip(20 * (l - 0.05), 0, 1)).to(device=device)
    r_mask = torch.fliplr(l_mask).to(device=device)
    return l_mask*disp_prime + r_mask*disp + (1.0 - l_mask - r_mask)*m_disp

def multi_scale(left_images, num_of_scale):
    image_scales = []
    image_scales.append(left_images)
    for i in range(num_of_scale-1):
        m = nn.AvgPool2d(kernel_size=2, stride=2)
        image_scales.insert(0, m(image_scales[0]))
    return image_scales

def multi_wrap(img, l_disp, r_disp, n=0, isLeft=True, device='cuda:0'):
    output = img
    if isLeft:
        for _ in range(n):
            output = wrap_image(
                disparity=l_disp, 
                images=wrap_image(disparity=-r_disp, images=output, device=device), 
                device=device
            )
    else:
        for _ in range(n):
            output = wrap_image(
                disparity=-r_disp, 
                images=wrap_image(disparity=l_disp, images=output, device=device), 
                device=device
            )
    return output

def get_dist(loss, dist, bins):
    indexes = (loss/bins[1]).long()
    return functorch.vmap(lambda x: dist[x])(indexes)