from utils.model_evaluate import get_parameter_number
from model import *
import torch
from utils.loss import DiceLoss

device = torch.device("cuda:1")


logs = [torch.cuda.memory_allocated(device)]
ret = {}
x = torch.randn(32, 3, 224, 224).to(device)
y = torch.randint(0, 2, (32, 1, 224, 224)).to(device)
logs.append(torch.cuda.memory_allocated(device))
ret["data"] = logs[-1] - logs[-2]
model = Map((224, 224), 3, 1).to(device)
# model = UNet(3, 1).to(device)
logs.append(torch.cuda.memory_allocated(device))
ret["model_map_size"] = logs[-1] - logs[-2]
model.train()
dice_loss = DiceLoss(1).to(device)
logs.append(torch.cuda.memory_allocated(device))
ret["loss_size"] = logs[-1] - logs[-2]
outputs = model(x)
loss = dice_loss(outputs, y, softmax=True)
logs.append(torch.cuda.memory_allocated(device))
ret["after_running"] = logs[-1] - logs[-2]
loss.backward()
logs.append(torch.cuda.memory_allocated(device))
ret["backward"] = logs[-1] - logs[-2]
for key, value in ret.items():
    print(key, value, sep='\t')

