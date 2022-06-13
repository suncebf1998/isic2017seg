import os
import random

import torch
import torch.utils.data
from torchvision.io import read_image
from torchvision.transforms import Resize, RandomHorizontalFlip, RandomVerticalFlip

subpath = {
    "train": {
        "data": "ISIC-2017_Training_Data",
        "groudtruth": "ISIC-2017_Training_Part1_GroundTruth"
    },
    "valid": {
        "data": "ISIC-2017_Validation_Data",
        "groudtruth": "ISIC-2017_Validation_Part1_GroundTruth"
    },
    "test": {
        "data": "ISIC-2017_Test_v2_Data",
        "groudtruth": "ISIC-2017_Test_v2_Part1_GroundTruth"
    }
}

randomseed = 2022

random.seed(randomseed)

def make_cache(data_root_dir: str, mode: str="train", overlap: bool=False, random=random):
    if mode not in ("train", "valid", "test"):
        raise AttributeError(f"mode({mode}) must be one of \"train\" \"valid\" \"test\".") 
    if os.path.isfile(data_root_dir):
        raise AttributeError(f"root dir of data cache({data_root_dir}) is file.") 
    if not os.path.isdir(data_root_dir):
        os.makedirs(data_root_dir)
    
    cache_file = data_root_dir + mode + '.bin' if data_root_dir.endswith('/') else data_root_dir+ '/' + mode + '.bin'

    datadir = data_root_dir + subpath[mode]["data"] if data_root_dir.endswith('/') else data_root_dir + '/' + subpath[mode]["data"]
    gtdir = data_root_dir + subpath[mode]["groudtruth"] if data_root_dir.endswith('/') else data_root_dir + '/' + subpath[mode]["groudtruth"] 
    if os.path.isfile(cache_file) and not overlap:
        pathdict = torch.load(cache_file)
    
    else:
        datapaths = os.listdir(datadir)
        gtpaths = set(os.listdir(gtdir))
        random.shuffle(datapaths)
        retdatapath = []
        retgtpath = []
        for path in datapaths:
            gt_path = path[:-4] + '_segmentation.png'
            if gt_path in gtpaths:
                retdatapath.append(path)
                retgtpath.append(gt_path)
        pathdict = {
            "data": retdatapath,
            "groudtruth": retgtpath
        }
        torch.save(pathdict, cache_file)
    return pathdict, datadir, gtdir

class Augmenting(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hflipper = RandomHorizontalFlip(p=1.)
        self.vflipper = RandomVerticalFlip(p=1.)

    def forward(self, x, y):
        n = random.randint(0, 2)
        if n == 0:
            return x, y
        elif n == 1:
            return self.hflipper(x), self.hflipper(y)
        else:
            return self.vflipper(x), self.vflipper(y)

class ISICDataset2017(torch.utils.data.Dataset):

    def __init__(self, data_root_dir: str, mode: str="train", size: tuple=(224, 224), overlap: bool=False, random=random):
        pathdict, self.datadir, self.gtdir = make_cache(data_root_dir, mode, overlap, random)
        self.datapaths = pathdict["data"]
        self.gtpaths = pathdict["groudtruth"]
        self.resize = Resize(size)
    
    def __len__(self):
        return len(self.datapaths)

    def __getitem__(self, index):
        datapath = self.datadir + '/' + self.datapaths[index]
        gtpath = self.gtdir + '/' + self.gtpaths[index]
        data = (read_image(datapath) / 255 - 0.5)
        data = self.resize(data)
        gt = (read_image(gtpath) / 255).int()
        gt = self.resize(gt)
        gt = gt.view(gt.shape[1:])
        return data, gt

class ISICDataset2017_Augmenting(torch.utils.data.Dataset):

    def __init__(self, data_root_dir: str, mode: str="train", size: tuple=(224, 224), overlap: bool=False, random=random):
        pathdict, self.datadir, self.gtdir = make_cache(data_root_dir, mode, overlap, random)
        self.datapaths = pathdict["data"]
        self.gtpaths = pathdict["groudtruth"]
        self.resize = Resize(size)
        self.aug = Augmenting() if mode == "train" else torch.nn.Identity()
    
    def __len__(self):
        return len(self.datapaths)

    def __getitem__(self, index):
        datapath = self.datadir + '/' + self.datapaths[index]
        gtpath = self.gtdir + '/' + self.gtpaths[index]
        data = (read_image(datapath) / 255 - 0.5)
        data = self.resize(data)
        gt = (read_image(gtpath) / 255).int()
        gt = self.resize(gt)
        gt = gt.view(gt.shape[1:])
        return self.aug(data, gt)
# for importing easily
ISICDataset, Dataset = ISICDataset2017, ISICDataset2017


if __name__ == "__main__":
    from tqdm import tqdm
    data_root_dir = "/home/phys/.58e4af7ff7f67242082cf7d4a2aac832cfac6a84/datasetisic/"
    for mode in ("train", "valid", "test"):
        dataset = Dataset(data_root_dir, mode)
        count = 0
        for data, gt in tqdm(dataset):
            count += 1
            break
        print(f"mode: {mode}, count: {count}, shape: {data.shape} {data.dtype} {gt.shape} {gt.dtype}")
