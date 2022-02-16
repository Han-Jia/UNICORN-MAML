import torch
import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from .randaugment import RandAugmentMC
import jpeg4py as jpeg

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
# /amax/data/tiered_imagenet_raw
IMAGE_PATH1 = '/mnt/data51/tiered_imagenet_raw'
SPLIT_PATH = osp.join(ROOT_PATH, 'data/tieredimagenet/split')
CACHE_PATH = osp.join(ROOT_PATH, '.cache/')
split_map = {'train':IMAGE_PATH1, 'val':IMAGE_PATH1, 'test':IMAGE_PATH1}

def identity(x):
    return x

def get_transforms(size, backbone, s = 1):
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    
    if backbone == 'ConvNet':
        normalization = transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                             np.array([0.229, 0.224, 0.225]))       
    elif backbone == 'Res12':
        normalization = transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                             np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
    elif backbone == 'Res18' or backbone == 'Res50':
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
    else:
        raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')
    
    data_transforms_aug = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.ToTensor(),
                                              normalization])
    
    data_transforms = transforms.Compose([transforms.Resize(size + 8),
                                          transforms.CenterCrop(size),
                                          transforms.ToTensor(),
                                          normalization])
    
    return data_transforms_aug, data_transforms


class tieredImageNet(Dataset):
    """ Usage:
    """
    def __init__(self, setname, args):
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        self.data, self.label = self.parse_csv(csv_path, setname)
        self.num_class = len(set(self.label))

        image_size = 84
        self.transform_aug, self.transform = get_transforms(image_size, args.backbone_class)

    def parse_csv(self, csv_path, setname):
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in tqdm(lines, ncols=64):
            name, wnid = l.split(',')
            path = osp.join(split_map[setname], name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append( path )
            label.append(lb)

        return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        try:
            image = self.transform(Image.fromarray(jpeg.JPEG(data).decode()).convert('RGB'))
        except:
            image = self.transform(Image.open(data).convert('RGB'))
        return image, label

