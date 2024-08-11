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
import torchvision.transforms as transforms

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
    
        if self.phase == 'train' or self.phase == 'val':
            self.noisy_filelist = sorted(list([ff for ff in self.NoisyFolder.iterdir()]))
            if self.phase == 'val':
                self.gt_filelist = sorted(list([ff for ff in self.GTFolder.iterdir()]))
            else:
                self.gt_filelist = self.noisy_filelist
        elif self.phase == 'test':
            self.noisy_filelist = sorted(list([ff for ff in self.NoisyFolder.iterdir()]))
            self.gt_filelist = sorted(list([ff for ff in self.NoisyFolder.iterdir()])) # dummy gt

        self.transforms = transforms

    def __getitem__(self, index):
        index *= self.reduce_scale
        
        if not self.transforms: # valid 
            gt_path = self.gt_filelist[index]
            gt = Image.open(gt_path).convert('RGB')
            gt = transforms.ToTensor()(gt)
            # gt_path = self.gt_filelist[index]
            # gt = cv2.imread(str(gt_path))
            # gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            # gt = transforms.ToTensor()(gt)        
            noisy_path = self.noisy_filelist[index]
            noisy = Image.open(noisy_path).convert('RGB')
            noisy = transforms.ToTensor()(noisy)
            # noisy_path = self.noisy_filelist[index]
            # noisy = cv2.imread(str(noisy_path))
            # noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
            # noisy = transforms.ToTensor()(noisy)
            
        else: # train    
            noisy_path = self.noisy_filelist[index]
            noisy = Image.open(noisy_path).convert('RGB')
            noisy = transforms.ToTensor()(noisy)
            # noisy_path = self.noisy_filelist[index]
            # noisy = cv2.imread(str(noisy_path))
            # noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
            # noisy = transforms.ToTensor()(noisy)
            gt = noisy
            gt_path = 1
            gt, noisy = self.transforms([gt, noisy])

        return {
            'gt': gt*255, 
            'noisy': noisy*255,
            'img_path': str(gt_path), 
            }

    def __len__(self):
        return len(self.noisy_filelist)//self.reduce_scale