#!/usr/bin/env python

from tkinter import SW
from torch import optim, softmax
from model import UNet, SwinUnet # , Map
from model.map2022061002 import MultiAveragePool as Map
from model.upernet import UPerNet
from model.swin_unet_lateralsV2 import SwinTransformerSys as SwinUnet_Laterals # t_version v2
from model.swin_unet_newembedV2 import SwinTransformerSys as SwinUnet_NewEmbed # t_version v3
from model.swin_fpn import SwinTransformerSys as SwinUnet_FPN # t_version v4
from model.namodel import NonAver # t_version v5
import os
import torch
import torch.cuda
import torch.utils.data
import torch.utils.data.distributed
import torch.nn as nn
import torch.distributed
import random
import numpy as np
from tqdm import tqdm
from tqdm.auto import trange
from dataset import ISIC2017IterDataset # ISICDataset2017,
from utils.loss import DiceLoss, Criterion
from utils.traintools import get_linear_schedule_with_warmup, DebugLog
from torch.utils.tensorboard import SummaryWriter
from utils.model_evaluate import get_parameter_number, time
# setting config
modelname = "upernet"
data_root_dir = "/home/phys/.58e4af7ff7f67242082cf7d4a2aac832cfac6a84/datasetisic/"
pt_root_dir = "/home/phys/.58e4af7ff7f67242082cf7d4a2aac832cfac6a84/multifiles/"
weight_dir = None # "/home/phys/.58e4af7ff7f67242082cf7d4a2aac832cfac6a84/weights/SGD_swinlateral_global_step=9450__last_model_loss=0.053315818309783936.pt/model.bin"# None
train_batch_size = 64
size = (224, 224)
iter_ratio = 1
num_workers = 2 
gradient_accumulation_steps = 1
num_train_epochs = 250
learning_rate = 5e-4 # 1e-5
num_classes = 1
input_channel = 3 
fp16 = False
fp16_opt_level = "02"
loss_scale = 0
warmup_steps = 100
log = DebugLog()
max_grad_norm = 1000.
use_log = True
logging_steps = 1
save_directory = "./weights/Adam_" + modelname + "_"
device_name = "cuda:2"
device_name_valid = "cuda:2"
use_static = True


def save_model(model, save_directory, only_print=False):
    if os.path.isfile(save_directory):
        log.error("Provided path should be a directory, not a file",
                  dict(save_directory=save_directory))
        return
    os.makedirs(save_directory, exist_ok=True)

    # Only save the model itself if we are using distributed training
    model_to_save = model.module if hasattr(model, "module") else model

    state_dict = model_to_save.state_dict()
    output_model_file = os.path.join(save_directory, "model.bin")
    torch.save(state_dict, output_model_file)
    log.info("Saved model weights.", dict(path=output_model_file))


# # tb_log
# if use_log:
#     tb_writer = SummaryWriter(filename_suffix="learningratex50")
#     log = DebugLog()

def set_seed(seed, cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda > 0:
        torch.cuda.manual_seed_all(seed)
# easy running:
def setup_optim(model, learning_rate, use_scheduler=False):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if use_scheduler:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )
        return optimizer, scheduler
    return optimizer
# @time_it
def train(
    model:nn.Module, train_dataloader:torch.utils.data.DataLoader, 
    num_train_epochs:int, valid_dataloader:torch.utils.data.DataLoader, gradient_accumulation_steps:int=gradient_accumulation_steps,
    logging_steps:int=logging_steps, device=None, save_directory=None, warmup_epoch=100, stepbefore=None, valid_device=None):
    global_step = 0 if stepbefore is None else stepbefore
    # tr_loss, logging_loss = 0., 0.
    best_loss = None
    # best_model = None
    optimizer = setup_optim(model, learning_rate)
    # optimizer, scheduler = setup_optim(model, learning_rate)
    model = model.to(device)
    if device is None:
        device = torch.device("cuda:0")
    else:
        device = torch.device(device) if isinstance(device, str) else device
    valid_device = valid_device or device
    for epoch in trange(int(num_train_epochs), desc="Epoch", leave=False):
        with tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=False) as batch_iterator:
            for step, (x, y) in enumerate(batch_iterator):
                model.train()
                # use iter dataloader
                x = x.reshape(x.shape[1:]).to(device)
                y = y.reshape(y.shape[1:]).to(device)
                outputs = model(x)
                loss = criterion(outputs, y, softmax=True) 
                dloss = dice_loss(outputs, y, softmax=True)
                loss += dloss
                if gradient_accumulation_steps > 1:
                        loss = loss / gradient_accumulation_steps
                loss.backward()
                batch_iterator.set_postfix(loss=loss.item())
                # tr_loss += loss.item()
                tb_writer.add_scalar(
                        "loss", loss.item(), global_step)

                # When we reached accumulation steps, do the optimization step
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm)
                    optimizer.step()
                    # scheduler.step()
                    model.zero_grad()
                    global_step += 1
                    tb_writer.add_scalar("epoch", epoch, global_step)


        if (epoch+1) % logging_steps == 0 and valid_dataloader is not None:
            model.to(valid_device)
            
            with tqdm(valid_dataloader, desc=f"Epoch {epoch}", leave=False) as vbatch_iterator:
                valid_loss = 0.
                weight_total = 0
                for step, (vx, vy) in enumerate(vbatch_iterator):
                    vx = vx.reshape(vx.shape[1:]).to(valid_device)
                    vy = vy.reshape(vy.shape[1:]).to(valid_device)
                    model.eval()
                    outputs = model(vx)
                    dloss = dice_loss(outputs, vy, softmax=True)
                    # if best_loss is None or best_loss > dloss:
                    #     best_loss = dloss
                    #     if save_directory and epoch > warmup_epoch:
                    #         best_loss_save_directory = save_directory + f"best_loss={dloss.item()}.pt"
                    #         # save_model(model, save_directory=best_loss_save_directory)
                    valid_loss += dloss.item() * len(vx)
                    weight_total += len(vx)
                
                # tb_writer.add_scalar(
                #     "lr", scheduler.get_last_lr()[0], global_step)
                total_valid_loss = valid_loss / weight_total
                tb_writer.add_scalar("valid_loss", total_valid_loss, global_step)
                # logging_loss = tr_loss
            model.to(device)
    save_model(model, save_directory + f"global_step={global_step}__last_model_loss={total_valid_loss}.pt")




# load model dataloader
def make_model(modelname):
    if modelname == "unet":
        model = UNet(input_channel, num_classes) # 31 037 633 31M
    elif modelname == "swinunet":
        model = SwinUnet(in_chans=input_channel, num_classes=num_classes, mlp_ratio=2) # 27 168 132 27M
    elif modelname == "map":
        model = Map(size,input_channel, num_classes) # 556 289 0.5M
    elif modelname == "upernet":
        model = UPerNet(input_channel, num_classes)
    elif modelname in ("swinlateral", "swinv2"):
        model = SwinUnet_Laterals(in_chans=input_channel, num_classes=num_classes, mlp_ratio=2)
    elif modelname in ("swinv3", "swinnewembed"):
        model = SwinUnet_NewEmbed(in_chans=input_channel, num_classes=num_classes, mlp_ratio=2)
    elif modelname in ("swinv4", "swinfpn"):
        model = SwinUnet_FPN(in_chans=input_channel, num_classes=num_classes, mlp_ratio=2)
    elif modelname in ("swinv5", "NonAver"):
        model = NonAver(img_size=size, in_chans=input_channel, num_classes=num_classes, embed_dim=48)
    return model

## use original data
# train_dataset = ISICDataset2017(data_root_dir=data_root_dir)
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, pin_memory=True)
# valid_dataset = ISICDataset2017(data_root_dir=data_root_dir, mode="valid")
# valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=train_batch_size, pin_memory=True)
# t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
# print(f"origin dataloader t total: {t_total}")

## use pregrocessed data
train_dataset = ISIC2017IterDataset(pt_root_dir + 'train/', ratio=iter_ratio)
train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=num_workers)
length = 0
for _ in train_dataloader:
    length += 1
t_total = length // gradient_accumulation_steps * num_train_epochs
if use_static:
    t_total *= 1e3
valid_num_workers = min(2, num_workers)
valid_dataset = ISIC2017IterDataset(pt_root_dir + 'valid/', ratio=iter_ratio)
valid_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=valid_num_workers)


# load loss fn
device = torch.device(device_name)
dice_loss = DiceLoss(num_classes).to(device)
criterion = Criterion(num_classes).to(device)
# if num_classes



# load optimizer scheduler
# if fp16:
#     try:
#         from apex.optimizers import FusedAdam, FusedSGD
#         from apex import amp
#     except ImportError:
#         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

#     optimizer = FusedSGD(model.parameters(),
#                             lr=learning_rate)
#     model, optimizer = amp.initialize(
#         model,
#         optimizers=optimizer,
#         opt_level=fp16_opt_level,
#         keep_batchnorm_fp32=False,
#         loss_scale="dynamic" if loss_scale == 0 else loss_scale,
#     )
#     log.info("FP16 launched")
# else:
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
# scheduler = get_linear_schedule_with_warmup(
#     optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
# )
# del modelname
# if modelname == "all":
#     to_do = ("unet","swinunet", "map")

# to_do = ("unet","swinunet", "map", "upernet", "swinlateral", "swinnewembed", "swinfpn") if modelname == "all" else (modelname, )
to_do = ( "swinlateral", "swinnewembed", "swinfpn") if modelname == "all" else (modelname, )
for modelname in to_do:
    save_directory = "./weights/SGD_" + modelname + "_"
    # # tb_log
    if use_log:
        startime = time.ctime().replace(" ","_")
        tb_writer = SummaryWriter(
            log_dir="/home/phys/.58e4af7ff7f67242082cf7d4a2aac832cfac6a84/runs/20220621mean_valid_loss",
            filename_suffix=f"{startime}__{modelname}")
        log = DebugLog()

    model = make_model(modelname)
    if weight_dir is not None:
        model.load_state_dict(torch.load(weight_dir))
    print(f"----{modelname}----")
    get_parameter_number(model, True)
    start = time.time()
    train(model, train_dataloader, num_train_epochs, valid_dataloader, save_directory=save_directory, device=device, valid_device=device_name_valid)#, stepbefore=9450)
    end = time.time()
    delta = end - start
    print("Running Total Time: {:.2f} seconds".format(delta))


                

