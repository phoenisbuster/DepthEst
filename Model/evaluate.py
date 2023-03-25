import argparse
import cv2
from torch.utils.data import DataLoader
import dataset
import networks
import numpy as np
import os
import utils
import torch
import torch.nn.functional as F
from torchvision import transforms
import math

STEREO_SCALE_FACTOR = 5.4
MIN_DEPTH = 1e-3
MAX_DEPTH = 80

width_to_focal = {}

width_to_focal[1224] = 707.0493
width_to_focal[1226] = 707.0912
width_to_focal[1238] = 718.3351
width_to_focal[1241] = 718.856
width_to_focal[1242] = 721.5377

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', dest='height', type=int, default=320, help="image's height")
    parser.add_argument('--width', dest='width', type=int, default=1024, help="image's width")
    parser.add_argument('--model_path', dest='model_path', type=str)
    parser.add_argument('--data_path', dest='data_path', type=str)
    parser.add_argument('--split', dest='split', type=str)

    return parser.parse_args()

def compute_rmse(predict, ground_truth):
    se = (predict - ground_truth)**2
    mse = se.mean()
    rmse = torch.sqrt(mse)
    return rmse

def compute_log_rmse(predict, ground_truth):
    log_se = (torch.log(predict) - torch.log(ground_truth))**2
    log_mse = log_se.mean()
    log_rmse = torch.sqrt(log_mse)
    return log_rmse

def compute_abs_rel(predict, ground_truth):
    abs_rel = torch.abs(predict - ground_truth) / ground_truth
    return abs_rel.mean()

def compute_sq_rel(predict, ground_truth):
    sq_rel = ((predict - ground_truth)**2) / ground_truth
    return sq_rel.mean()

def compute_accuracies(predict, ground_truth):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.maximum((predict / ground_truth), (ground_truth / predict))

    a1 = ((thresh < 1.25     )*1.0).mean()
    a2 = ((thresh < (1.25**2))*1.0).mean()
    a3 = ((thresh < (1.25**3))*1.0).mean()

    return a1, a2, a3

def disps_to_depths(disps):
    mask = (disps > 0)*1
    _, _, height, width = disps.shape
    depth = torch.zeros((height, width)).to(device=device)
    depth = depth + mask*(width_to_focal[width] * 0.54 / (disps + 1 - mask))
    return depth

# Hyperparameter
args = parse_arguments()
model_path = args.model_path
img_height = args.height
img_width = args.width
data_path = args.data_path
split = args.split

# Check if model folder exists
assert os.path.isdir(model_path), \
    "Cannot find a folder at {}".format(model_path)
print("-> Loading weights from {}".format(model_path))


# Load data
root = data_path

data_transform = transforms.Compose([
    transforms.Resize(size=(img_height, img_width)),
    transforms.ToTensor()
])

eval_dataset = dataset.KITTIEvaluate(
    root=root,
    num_samples=-1,
    split=split,
    train=False,
    transform=data_transform
)

eval_loader = DataLoader(
    dataset=eval_dataset, 
    batch_size=1, 
    shuffle=True,
)


# Setup device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Load models
encoder_path = os.path.join(model_path, "encoder.pt")
decoder_path = os.path.join(model_path, "decoder.pt")

models = {}
models["encoder"] = networks.ResNet(networks.Bottleneck, [3, 4, 6, 3]).to(device=device)
models["decoder"] = networks.Decoder().to(device=device)

models["encoder"].load_state_dict(torch.load(encoder_path))
models["decoder"].load_state_dict(torch.load(decoder_path))
models["encoder"].eval()
models["decoder"].eval()


# Evaluate
print("Begin Evaluating!")
rms_list = []
log_rms_list = []
abs_rel_list = []
sq_rel_list = []
a1_list = []
a2_list = []
a3_list = []
with torch.no_grad():
    for left_images, _, gt in eval_loader:
        # Pass data to device
        left_images = left_images.to(device=device)
        flipped_img = torch.flip(left_images,dims=[3]).to(device=device)
        gt = gt.to(device=device)

        # Calculate disparities
        features = models["encoder"](left_images)
        pred_disps = models["decoder"](features)[-1]

        l_pred_disp = pred_disps[:, 0].unsqueeze(1) #Get left disparity
        r_pred_disp = pred_disps[:, 1].unsqueeze(1) #Get right disparity

        tmp_disp = models["encoder"](flipped_img)
        tmp_disp = models["decoder"](tmp_disp)[-1]

        flipped_disp = tmp_disp[:, 0].unsqueeze(1) #Get left disparity
        disp_prime = torch.flip(flipped_disp,dims=[3])

        pred_disp = utils.post_process_disparity(l_pred_disp, disp_prime)
        # pred_disp = l_pred_disp
        
        _, _, H, W = gt.shape
        pred_disp = torch.abs(F.interpolate(pred_disp, (H, W), mode='bilinear'))
        temp=pred_disp
        pred_disp = (pred_disp*W/2)
        pred_depths = disps_to_depths(pred_disp)
        
        if split == "stereo2015":
            gt_depths = disps_to_depths(gt)
        elif split == "eigen":
            gt_depths = gt
            
        gt_depths[gt_depths < MIN_DEPTH] = MIN_DEPTH
        gt_depths[gt_depths > MAX_DEPTH] = MAX_DEPTH
        pred_depths[pred_depths < MIN_DEPTH] = MIN_DEPTH
        pred_depths[pred_depths > MAX_DEPTH] = MAX_DEPTH

        mask = torch.logical_and(gt_depths > MIN_DEPTH, gt_depths < MAX_DEPTH)
        if split == "eigen":
            crop = np.array([0.40810811 * H,  0.99189189 * H,   
                0.03594771 * W,   0.96405229 * W]).astype(np.int32)
            crop_mask = torch.zeros((H, W)).to(device=device)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1.0

            mask = torch.logical_and(mask, crop_mask)

        gt_depths = gt_depths[mask]
        pred_depths = pred_depths[mask]
        
        rms_list.append(compute_rmse(pred_depths, gt_depths))
        log_rms_list.append(compute_log_rmse(pred_depths, gt_depths))
        abs_rel_list.append(compute_abs_rel(pred_depths, gt_depths))
        sq_rel_list.append(compute_sq_rel(pred_depths, gt_depths))
        a1, a2, a3 = compute_accuracies(pred_depths, gt_depths)
        a1_list.append(a1)
        a2_list.append(a2)
        a3_list.append(a3)

# Calculate average errors
avg_rms = torch.tensor(rms_list).mean()
avg_log_rms = torch.tensor(log_rms_list).mean()
avg_abs_rel = torch.tensor(abs_rel_list).mean()
avg_sq_rel = torch.tensor(sq_rel_list).mean()
avg_a1 = torch.tensor(a1_list).mean()
avg_a2 = torch.tensor(a2_list).mean()
avg_a3 = torch.tensor(a3_list).mean()

# Print results
print("--- Evaluation result ---")
print(("{:^18} | " * 7).format("RMS", "Log RMS", "Absolute Rel", "Square Rel", "Acc delta < 1.25", "Acc delta < 1.25^2", "Acc delta < 1.25^3"))
print(("{:^18.3f} | " * 7).format(avg_rms, avg_log_rms, avg_abs_rel, avg_sq_rel, avg_a1, avg_a2, avg_a3))