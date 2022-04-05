import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class VitDataset(Dataset):

    def __init__(self, img_dir, gt_anno, transform=None):
        super(VitDataset, self).__init__()
        assert os.path.exists(gt_anno), print("annotation file not found: {}".format(gt_anno))
        self.img_list = []
        self.label_list = []
        with open(gt_anno, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split(',')
            img_path = os.path.join(img_dir, line[0])
            assert os.path.exists(img_path), print("img_path not found: {}".format(img_path))
            self.img_list.append(img_path)
            self.label_list.append(int(line[1]))
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert("RGB")
        label = self.label_list[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img_list)


def get_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory):

    # if args.distributed:
    #     num_tasks = misc.get_world_size()
    #     global_rank = misc.get_rank()
    #     sampler = torch.utils.data.DistributedSampler(
    #         dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    #     )
    #     print("Sampler_train = %s" % str(sampler))
    # else:
    #     sampler = torch.utils.data.RandomSampler(dataset)

    sampler = torch.utils.data.RandomSampler(dataset)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        sampler=sampler, num_workers=num_workers, pin_memory=pin_memory,
                        drop_last=True)
    return loader