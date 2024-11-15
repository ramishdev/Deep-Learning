from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        torchvision = tv.transforms
        self.train_transform  = torchvision.Compose([torchvision.ToPILImage(),torchvision.ToTensor(),torchvision.Normalize(train_mean,train_std)])
        self.validation_transform  = torchvision.Compose([torchvision.ToPILImage(),torchvision.ToTensor(),torchvision.Normalize(train_mean,train_std)])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        data = self.data.iloc[index]
        img = imread(data['filename'], as_gray=True)
        img = gray2rgb(img)
        label = np.array([data['crack'], data['inactive']])
        img = self.validation_transform(img) if self.mode == "val" else self.train_transform(img)
        return img, label
    
    @property
    def transform(self):
        return self._transform
    
    @transform.setter
    def transform(self, transform):
        self._transform = transform