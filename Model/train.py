import argparse
import dataset
import loss
import networks
import os
import utils
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', dest='model_path', type=str, default="")
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--epochs', dest='epochs', type=int, default=100, help="total training epochs")
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-4, help="weight decay")
    parser.add_argument('--img_size', dest='img_size', type=int, default=360, help="image's size")
    parser.add_argument('--early_stopping', dest='early_stopping', type=int, default=-1, help="patient epoch")
    return parser.parse_args()


def save_model(folder_path, models, model_optimizer, best_loss):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for model_name, model in models.items():
        save_path = os.path.join(folder_path, "{}.pt".format(model_name))
        to_save = model.state_dict()
        torch.save(to_save, save_path)

    save_path = os.path.join(folder_path, "{}.pt".format(model_optimizer.__class__.__name__))
    torch.save(model_optimizer.state_dict(), save_path)

    save_path = os.path.join(folder_path, "{}.pt".format("best_loss"))
    torch.save(best_loss, save_path)


# Hyperparameter configurations
args = parse_arguments()
model_path = args.model_path
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.lr
weight_decay = args.weight_decay
img_size = args.img_size
early_stopping = args.early_stopping

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load data
data_transform = [
    transforms.CenterCrop(400),
    transforms.Resize(size=img_size),
    transforms.ToTensor()
]

train_dataset = dataset.NYUv2(
    root="../dataset/nyuv2/data",
    train=True,
    rgb_transform=transforms.Compose([
        *data_transform, 
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    seg_transform=transforms.Compose(data_transform),
    sn_transform=transforms.Compose(data_transform),
    depth_transform=transforms.Compose(data_transform)
)

val_dataset = dataset.NYUv2(
    root="../dataset/nyuv2/data",
    train=False,
    rgb_transform=transforms.Compose([
        *data_transform, 
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    seg_transform=transforms.Compose(data_transform),
    sn_transform=transforms.Compose(data_transform),
    depth_transform=transforms.Compose(data_transform)
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
val_loader = DataLoader(
    dataset=val_dataset, 
    batch_size=batch_size,
    shuffle=True
)


# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize model
models = {}
models["backbone"] = networks.Backbone().to(device=device)
models["decoder"] = networks.Decoder().to(device=device)
models["semantic_to_depth"] = networks.SematicToDepth().to(device=device)
models["depth_to_semantic"] = networks.DepthToSemantic(class_num=13).to(device=device)

if model_path != "":
    backbone_path = os.path.join(model_path, "backbone.pt")
    decoder_path = os.path.join(model_path, "decoder.pt")
    semantic_to_depth_path = os.path.join(model_path, "semantic_to_depth.pt")
    depth_to_semantic_path = os.path.join(model_path, "depth_to_semantic.pt")
    models["backbone"].load_state_dict(torch.load(backbone_path))
    models["decoder"].load_state_dict(torch.load(decoder_path))
    models["semantic_to_depth"].load_state_dict(torch.load(semantic_to_depth_path))
    models["depth_to_semantic"].load_state_dict(torch.load(depth_to_semantic_path))

parameters_to_train = []
parameters_to_train += list(models["backbone"].parameters())
parameters_to_train += list(models["decoder"].parameters())
parameters_to_train += list(models["semantic_to_depth"].parameters())
parameters_to_train += list(models["depth_to_semantic"].parameters())


# Config loss function and optimizer
seg_loss_module = nn.CrossEntropyLoss()
depth_loss_module = nn.L1Loss()

model_optimizer = torch.optim.Adam(params=parameters_to_train, lr=learning_rate, betas=(0.9,0.999), eps=1e-8, weight_decay=weight_decay)

if model_path != "":
    optimizer_path = os.path.join(model_path, "{}.pt".format(model_optimizer.__class__.__name__))
    model_optimizer.load_state_dict(torch.load(optimizer_path))


# Traing
best_loss = torch.tensor(float('inf'))
if model_path != "":
    path = os.path.join(model_path, "{}.pt".format("best_loss"))
    best_loss= torch.load(path)

print("Begin Training!")
early_stopping_cnt = 0
for epoch in range(epochs):
    early_stopping_cnt += 1
    print('======= Begin Epoch [%d/%d] =======' % (epoch+1, epochs))
    train_loss_list = []
    models["backbone"].train()
    models["decoder"].train()
    models["semantic_to_depth"].train()
    models["depth_to_semantic"].train()
    p = 0
    for batch_idx, img_datas in enumerate(train_loader):
        # Pass data to cuda
        rgb = img_datas[0].to(device=device)
        gt_seg = img_datas[1].to(device=device)
        gt_depth = img_datas[3].to(device=device)

        # Forward
        backbone_out, refined_fp = models["backbone"](rgb)
        semantic_feature, common_rep, depth_feature = models["decoder"](backbone_out, refined_fp)
        
        # Calculating Loss
        loss_ = None
        if p == 0:
            pred_seg = models["depth_to_semantic"](semantic_feature, common_rep, depth_feature)
            loss_ = seg_loss_module(input=pred_seg, target=gt_seg)

            for param in models["semantic_to_depth"].parameters():
                param.requires_grad = False
            for param in models["depth_to_semantic"].parameters():
                param.requires_grad = True
            p = 1
        else:
            pred_depth = models["semantic_to_depth"](semantic_feature, common_rep, depth_feature)
            loss_ = depth_loss_module(input=pred_depth, target=gt_depth)
            
            for param in models["semantic_to_depth"].parameters():
                param.requires_grad = True
            for param in models["depth_to_semantic"].parameters():
                param.requires_grad = False
            p = 0
        train_loss_list.append(loss_.item())

        # Backward
        model_optimizer.zero_grad()
        loss_.backward()
        model_optimizer.step()

        # Print Iteration Summary
        print('--- Epoch [%d/%d] - Iteration [%d/%d] --- Loss: %.4f -- Best Avg Traning Loss: %.4f \
        ' % (
            epoch+1, 
            epochs,
            batch_idx+1,
            len(train_loader),
            loss_.item(),
            best_loss.item()
        ))
    train_loss = torch.Tensor(train_loss_list).mean()

    print("Begin Validating!")
    val_loss_list = []
    models["backbone"].eval()
    models["decoder"].eval()
    models["semantic_to_depth"].eval()
    models["depth_to_semantic"].eval()
    with torch.no_grad():
        for batch_idx, img_datas in enumerate(val_loader):
            # Pass data to cuda
            rgb = img_datas[0].to(device=device)
            gt_seg = img_datas[1].to(device=device)
            gt_depth = img_datas[3].to(device=device)

            # Forward
            refined_fp, backbone_out = models["backbone"](rgb)
            semantic_feature, common_rep, depth_feature = models["decoder"](refined_fp, backbone_out)
            pred_depth = models["semantic_to_depth"](semantic_feature, common_rep, depth_feature)
            pred_seg = models["depth_to_semantic"](semantic_feature, common_rep, depth_feature)

            # Calculating Loss
            loss_ = depth_loss_module(pred_depth, gt_depth) + seg_loss_module(pred_seg, gt_seg)
            val_loss_list.append(loss_.item())
    val_loss = torch.Tensor(val_loss_list).mean()
    
    print("""
    ========== Summary Epoch [%d/%d] ==========
    Training Loss: %.4f -- Validation Loss: %.4f -- Best Loss: %.4f
    ===========================================
    """ % (
        epoch+1, 
        epochs,
        train_loss,
        val_loss,
        best_loss.item()
    ))
    
    # Save better model
    save_best_folder = "./best/{}x{}".format(img_size, img_size)
    save_last_folder = "./last/{}x{}".format(img_size, img_size)
    save_besttest_folder = "./besttest/{}x{}".format(img_size, img_size)
    save_test_folder = "./lasttest/{}x{}".format(img_size, img_size)
    if val_loss < best_loss:
        early_stopping_cnt = 0
        best_loss = val_loss
        save_model(folder_path=save_best_folder, models=models, optim=model_optimizer, best_loss=best_loss)
    save_model(folder_path=save_last_folder, models=models, optim=model_optimizer, best_loss=best_loss)

    # Check early stopping
    if early_stopping_cnt == early_stopping:
        break