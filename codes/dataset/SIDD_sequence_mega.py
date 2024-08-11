import os
import sys
sys.path.append('..')

from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as Transforms

class Sequence(Dataset):
    def __init__(
        self, 
        root_path: Path, 
        seq_path: str, 
        phase: str='train',
        transforms=None,
        reduce_scale: int=1, # not 1 for less validation data
        ):

        assert phase in ['train', 'val', 'test'], 'ERROR: \'phase\' should be train, val or test, but got %s' % phase
        self.phase = phase
        self.reduce_scale = reduce_scale

        self.GTFolder = root_path / 'GT' / ('%s' % seq_path)
        self.NoisyFolder = root_path / 'Noisy' / ('%s' % seq_path)
    
        if self.phase == 'val':
            self.noisy_filelist = sorted(list([ff for ff in self.NoisyFolder.iterdir()]))
            self.gt_filelist = sorted(list([ff for ff in self.GTFolder.iterdir()]))
        elif self.phase == 'test':
            self.noisy_filelist = sorted(list([ff for ff in self.NoisyFolder.iterdir()]))
            self.gt_filelist = sorted(list([ff for ff in self.NoisyFolder.iterdir()])) # dummy gt
        elif self.phase == 'train':
            # 用于取没有切块的大尺寸数据集
            from natsort import natsorted
            from glob import glob
            import time
            tik = time.time()
            files = natsorted(glob(os.path.join('to_your_dataset_path/SIDD_Medium_Srgb/Data', '*', '*.PNG')))
            self.noisy_filelist = []
            for file_ in files:
                filename = os.path.split(file_)[-1]
                if 'NOISY' in filename:
                    file_ = Image.open(file_).convert('RGB')
                    file_ = Transforms.ToTensor()(file_)
                    self.noisy_filelist.append(file_)
            self.gt_filelist = None
            tok = time.time()
            print('load mega Medium time: ', tok-tik)
                    
        self.transforms = transforms

    def __getitem__(self, index):
        index *= self.reduce_scale
        
        if self.phase != 'train':
            gt_path = self.gt_filelist[index]
            gt = Image.open(gt_path).convert('RGB')
            gt = Transforms.ToTensor()(gt) 

            noisy_path = self.noisy_filelist[index]
            noisy = Image.open(noisy_path).convert('RGB')
            noisy = Transforms.ToTensor()(noisy)
        
            if self.transforms:
                gt, noisy = self.transforms([gt, noisy])
        else:
            noisy = self.noisy_filelist[index]
            if self.transforms:
                [noisy] = self.transforms([noisy])
            gt = noisy
            gt_path = 1

        return {
            'gt': gt*255, 
            'noisy': noisy*255,
            'img_path': str(gt_path), 
            }

    def __len__(self):
        return len(self.noisy_filelist)//self.reduce_scale