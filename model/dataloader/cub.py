import os.path as osp
import PIL
from PIL import Image

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import jpeg4py as jpeg

THIS_PATH = osp.dirname(__file__)
ROOT_PATH1 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..'))
IMAGE_PATH = osp.join(ROOT_PATH1, 'data/cub/images')
SPLIT_PATH = osp.join(ROOT_PATH2, 'data/cub/split')
CACHE_PATH = osp.join(ROOT_PATH2, '.cache/')

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


class CUB(Dataset):

    def __init__(self, setname, args):
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        self.data, self.label = self.parse_csv(csv_path)
        self.num_class = len(set(self.label))

        image_size = 84
        self.transform_aug, self.transform = get_transforms(image_size, args.backbone_class)
        
    def parse_csv(self, txt_path):
        data = []
        label = []
        lb = -1
        self.wnids = []
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]

        for l in lines:
            context = l.split(',')
            name = context[0] 
            wnid = context[1]
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
                
            data.append(path)
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
