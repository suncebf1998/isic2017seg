#!/usr/bin/env python
from typing import Dict
from torch import optim
from torch.nn.parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils import data
from model import UNet
import sys
import os
import torch
import torch.cuda
import torch.utils.data
import torch.utils.data.distributed
import torch.nn as nn
import torch.distributed
import random
import numpy as np
from torch.multiprocessing import Process
from tqdm import tqdm
from tqdm.auto import trange
from utils import getLoggerWithRank, redirect_warnings_to_logger
from dataset import ISICDataset2017
import argparse
import warnings
from utils import is_main_process
from utils.loss import DiceLoss


def setup(args):
    global log
    if sys.platform == 'win32':
        raise NotImplementedError("Unsupported Platform")
    else:
        args.local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))
        log = getLoggerWithRank(__name__, int(
            os.environ.get("RANK", "-1")), args.local_rank)
        redirect_warnings_to_logger(log)
        # Setup CUDA, GPU & distributed training
        if args.local_rank == -1 or args.no_cuda:
            if torch.cuda.is_available() and not args.no_cuda:
                device = torch.device("cuda")
            else:
                log.critical("!!!! Using CPU for training !!!!")
                device = torch.device("cpu")
            args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
            log.info("Using DataParallel for training.",
                     dict(n_gpu=args.n_gpu))
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            log.warning("Initializing process group.")
            torch.distributed.init_process_group(backend="nccl")
            args.node_rank = torch.distributed.get_rank()
            args.world_size = torch.distributed.get_world_size()
            log.info("Initialized distributed training process group.", dict(
                backend=torch.distributed.get_backend(), world_size=args.world_size))
            args.n_gpu = 1
        args.device = device
        args.train_batch_size = args.per_gpu_train_batch_size * \
            max(1, args.n_gpu)
        set_seed(args)

    log.warning("Finish setup.", dict(device=args.device, n_gpu=args.n_gpu,
                                      distributed_training=bool(args.local_rank != -1)))