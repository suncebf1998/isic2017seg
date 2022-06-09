#!/usr/bin/env python

from torch import optim, softmax
from model import UNet, SwinUnet, Map
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
from dataset import ISICDataset2017,ISIC2017IterDataset
from utils import is_main_process
from utils.loss import DiceLoss, Criterion
from utils.traintools import get_linear_schedule_with_warmup, DebugLog
from torch.utils.tensorboard import SummaryWriter

# setting config
modelname = "map"
data_root_dir = "/home/phys/.58e4af7ff7f67242082cf7d4a2aac832cfac6a84/datasetisic/"
pt_root_dir = "/home/phys/.58e4af7ff7f67242082cf7d4a2aac832cfac6a84/multifiles/"
train_batch_size = 64
size = (224, 224)
iter_ratio = 1
num_workers = 2 
gradient_accumulation_steps = 1
num_train_epochs = 250
learning_rate = 1e-5
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
save_directory = "./weights/"
device_name = "cuda:0"


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


# tb_log
if use_log:
    tb_writer = SummaryWriter()
    log = DebugLog()

def set_seed(seed, cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda > 0:
        torch.cuda.manual_seed_all(seed)

def train(
    model:nn.Module, train_dataloader:torch.utils.data.DataLoader, 
    num_train_epochs:int, valid_dataloader:torch.utils.data.DataLoader, gradient_accumulation_steps:int=gradient_accumulation_steps,
    logging_steps:int=logging_steps, device=None, save_directory=None, warmup_epoch=100):
    global_step = 0
    tr_loss, logging_loss = 0., 0.
    best_loss = None
    # best_model = None
    model = model.to(device)
    if device is None:
        device = torch.device("cuda:0")
    else:
        device = torch.device(device) if isinstance(device, str) else device
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
                # print()
                # print(dloss)
                # print(type(dloss))
                # print()
                # 1 / 0
                loss += dloss
                if gradient_accumulation_steps > 1:
                        loss = loss / gradient_accumulation_steps
                # Don't call zero_grad() until we have accumulated enough steps
                loss.backward()
                batch_iterator.set_postfix(loss=loss.item())
                tr_loss += loss.item()

                # When we reached accumulation steps, do the optimization step
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1


        if (epoch+1) % logging_steps == 0 and valid_dataloader is not None:
            with tqdm(valid_dataloader, desc=f"Epoch {epoch}", leave=False) as vbatch_iterator:
                for step, (vx, vy) in enumerate(vbatch_iterator):
                    vx = vx.reshape(vx.shape[1:]).to(device)
                    vy = vy.reshape(vy.shape[1:]).to(device)
                    model.eval()
                    outputs = model(vx)
                    dloss = dice_loss(outputs, vy, softmax=True)
                    if best_loss is None or best_loss > dloss:
                        best_loss = dloss
                        if save_directory and epoch > warmup_epoch:
                            best_loss_save_directory = save_directory + f"best_loss={dloss.item()}.pt"
                            # save_model(model, save_directory=best_loss_save_directory)
                    tb_writer.add_scalar(
                        "lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss", (tr_loss - logging_loss) / logging_steps, global_step)
                    tb_writer.add_scalar("valid_loss", dloss, global_step)
                    logging_loss = tr_loss
    save_model(model, save_directory + f"last_model_loss={dloss.item()}.pt")




# load model dataloader
if modelname == "unet":
    model = UNet(input_channel, num_classes) # 31 037 633 31M
elif modelname == "swinunet":
    model = SwinUnet(in_chans=input_channel, num_classes=num_classes, mlp_ratio=2) # 27 168 132 27M
elif modelname == "map":
    model = Map(size,input_channel, num_classes) # 556 289 0.5M

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
valid_num_workers = min(2, num_workers)
valid_dataset = ISIC2017IterDataset(pt_root_dir + 'valid/', ratio=iter_ratio)
valid_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=valid_num_workers)


# load loss fn
device = torch.device(device_name)
dice_loss = DiceLoss(num_classes).to(device)
criterion = Criterion(num_classes).to(device)
# if num_classes



# load optimizer scheduler
if fp16:
    try:
        from apex.optimizers import FusedAdam, FusedSGD
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    optimizer = FusedSGD(model.parameters(),
                            lr=learning_rate)
    model, optimizer = amp.initialize(
        model,
        optimizers=optimizer,
        opt_level=fp16_opt_level,
        keep_batchnorm_fp32=False,
        loss_scale="dynamic" if loss_scale == 0 else loss_scale,
    )
    log.info("FP16 launched")
else:
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
)


train(model, train_dataloader, num_train_epochs, valid_dataloader, save_directory=save_directory, device=device)


                

