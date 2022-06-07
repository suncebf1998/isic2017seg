import torch
import random
import math

class ISIC2017IterDataset(torch.utils.data.IterableDataset):
    def __init__(self, root_dir:str, shuffle=False, seed=1024, ratio=1):
        super(ISIC2017IterDataset).__init__()
        self.root_dir = root_dir if root_dir.endswith('/') else root_dir + '/'
        self.reader:tuple = torch.load(self.root_dir + "reader.pt")
        if shuffle:
            random.seed(seed)
            random.shuffle(self.reader)
        self.ratio = int(ratio)
        assert self.ratio >= 0, f"ratios must be int and be larger or equal than 1, while ratio is {ratio}"
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.single()
        else:
            return self.multi(worker_info)
    
    def single(self, reader=None):
        reader = reader or self.reader
        xs = []
        ys = []
        for i, readeritem in enumerate(reader, 1):
            path = self.root_dir + readeritem[1:] 
            x, y = torch.load(path)
            xs.append(x)
            ys.append(y)
            if i % self.ratio == 0 or i == len(reader):
                yield torch.cat(xs, dim=0), torch.cat(ys, dim=0)
                xs,ys = [], []
    
    def multi(self, worker_info:torch.utils.data._utils.worker.WorkerInfo):
        assert worker_info is not None, "Multi must specify worker_info."
        per_worker = int(math.ceil(len(self.reader) / float(worker_info.num_workers)))
        worker_id = worker_info.id
        iter_start = worker_id * per_worker
        iter_end = min(iter_start + per_worker, len(self.reader))
        reader = self.reader[iter_start:iter_end]
        return self.single(reader)


if __name__ == "__main__":
    prefix_root_dir = "/home/phys/.58e4af7ff7f67242082cf7d4a2aac832cfac6a84/multifiles/"
    for mode in ("train", "test", "valid"):
        root_dir = prefix_root_dir + mode + "/" if random.random() > 0.5 else prefix_root_dir + mode
        print(f"root directory is {root_dir}")
        datasets = ISIC2017IterDataset(root_dir, ratio=2)
        count_datasets = 0
        for _ in datasets:
            count_datasets += 1
        dataloader = torch.utils.data.DataLoader(datasets, num_workers=2)
        count_dataloader = 0
        for _ in dataloader:
            count_dataloader += 1
        print(f"mode: {mode} count of datasets and dataloader are {count_datasets}, {count_dataloader}")