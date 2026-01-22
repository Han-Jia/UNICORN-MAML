import os.path as osp
import os
import PIL
from PIL import Image
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import jpeg4py as jpeg
import h5py
import pdb
from torchvision import transforms
import itertools
import torchfile
from scipy.io import loadmat

THIS_PATH = osp.dirname(__file__)
ROOT_PATH1 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..'))
IMAGE_PATH = '/mnt/scratch/users/zha503/Unicorn Folder/data/flower/jpg'
SPLIT_PATH = '/mnt/scratch/users/zha503/Unicorn Folder/data/flower/split'
CACHE_PATH = osp.join(ROOT_PATH2, '.cache/')
TEXT_PATH = '/mnt/scratch/users/zha503/Unicorn Folder/data/flower/text_c10'#w2v_c10'#text_c10'


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



class Flower(Dataset): #Dataset

    def __init__(self, setname, args):
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        self.data,  self.label = self.parse_csv(csv_path) #self.txts,
        self.num_class = len(set(self.label))

       
               
        self.args = args
       
        #print(len(self.corpus))693
        self.terms= None
        self.tfidf_matrix = None
        image_size = 32
        self.transform_aug, self.transform = get_transforms(image_size, args.backbone_class)


        

    def parse_csv(self, csv_path):
        data = []
        label = []
        stats= []
        lb = -1
        self.wnids = []
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        for l in tqdm(lines, ncols=64):# class
            name, wnid = l.split(',')
            files_name=[]

            foldname,clssname = name.split('/')
            text_path = osp.join(TEXT_PATH, clssname) # no .t7
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
                for filename in os.listdir(text_path):
                    if filename.endswith(".txt"):
                        if len(files_name) < 16 and filename not in files_name:
                            files_name.append(files_name)
                            path = osp.join(IMAGE_PATH, filename[:-3] + "jpg")
                            txtfile_path = osp.join(text_path,filename)
                            Stat_vec = [x for x in open(txtfile_path, 'r').readlines()]
                            #stats.append(Stat_vec)
                            data.append(path)
                            label.append(lb)
                        else:
                            break

        return data,  label #stats,


    def __len__(self):
        return len(self.data)


    

    def __getitem__(self, i):
        data,  label = self.data[i], self.label[i]

                #print(len(text_data) ) get 75 stats per sample correct
        

        try:
            image = self.transform(Image.fromarray(jpeg.JPEG(data).decode()).convert('RGB'))
        except:
            image = self.transform(Image.open(data).convert('RGB'))
        
  
        return image, label
