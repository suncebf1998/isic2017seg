import os
import random
import torch
import torch.utils.data
from torchvision.io import read_image
from torchvision.transforms import Resize, RandomHorizontalFlip, RandomVerticalFlip
from torch.nn.functional import interpolate
# from functools import partial

subpath = {
    "train": {
        "data": "ISIC2018_Task1-2_Training_Input",
        "groudtruth": "ISIC2018_Task2_Training_GroundTruth_v3"
    },
    "valid": {
        "data": "ISIC2018_Task1-2_Validation_Input",
        "groudtruth": "ISIC2018_Task2_Validation_GroundTruth"
    },
    "test": {
        "data": "ISIC2018_Task1-2_Test_Input",
        "groudtruth": None
    }
}

kinds = (
    'globules.png',
    'milia_like_cyst.png',
    'negative_network.png',
    'pigment_network.png',
    'streaks.png'
)

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
            if not path.endswith('.jpg'):
                continue
            retdatapath.append(path)
            ret_item = []
            for kind in kinds:
                gt_path = path[:12] + "_attribute_" + kind
                if gt_path in gtpaths:
                    ret_item.append(gt_path)
                else:
                    ret_item.append("")
            retgtpath.append(ret_item)
        pathdict = {
            "data": retdatapath,
            "groudtruth": retgtpath
        }
        torch.save(pathdict, cache_file)
    return pathdict, datadir, gtdir


class ISICDataset2018(torch.utils.data.Dataset):

    def __init__(self, data_root_dir: str, mode: str="train", size: tuple=(224, 336), overlap: bool=False, random=random):
        pathdict, self.datadir, self.gtdir = make_cache(data_root_dir, mode, overlap, random)
        self.datapaths = pathdict["data"]
        self.gtpaths = pathdict["groudtruth"]
        self.resize = Resize(size) # partial(interpolate, size=size, mode="bilinear")
        self.size = size
    
    def __len__(self):
        return len(self.datapaths)

    def __getitem__(self, index):
        datapath = self.datadir + '/' + self.datapaths[index]
        gtpath = self.gtpaths[index]
        data = (read_image(datapath) / 255 - 0.5)
        if data.shape[-2] > data.shape[-1]:
            transpose = True
        else:
            transpose = False  
        data = data.transpose(-2, -1) if transpose else data
        data = self.resize(data)
        gt = self.read_labels(gtpath)
        gt = self.resize(gt)
        if transpose:
            print(index)
#         gt = gt.view(gt.shape[1:])
        return data, gt
    
    def read_labels(self, gtpath_items, transpose=False):
        gts = []
        for gtpath_item in gtpath_items:
            if gtpath_item == "":
                gt = torch.zeros(self.size).int()
                gts.append(gt)
                continue
            path = self.gtdir + '/' + gtpath_item
            gt = (read_image(path) / 255).int() # .unsqueeze(0)
            gts.append(gt)
        gts = torch.cat(gts, 0)
        if transpose:
            gts = gts.transpose(-2, -1)
        return gts

if __name__ == "__main__":
    from tqdm import tqdm
    data_root_dir = "./"

    for mode in ( "train", "valid"):
        dataset = ISICDataset2018(data_root_dir, mode)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=16)
        files = []
        root = f"./numpyfile/{mode}/"
        prefix = "batch="
        # enumerate
        i = 0
        with tqdm(dataloader, desc=f"Epoch {mode}", leave=False) as batch_iterator:
            for batch in batch_iterator:
                i += 1
                # x, y = batch
                # print(type(batch))
                # raise AttributeError
                x, y = batch
                file = f'{prefix}{len(x)}_{i}.pt'
                torch.save(batch, root+file)
                batch_iterator.set_postfix(file=root+file)
                files.append(file)
        torch.save(files, root+'reader.pt')