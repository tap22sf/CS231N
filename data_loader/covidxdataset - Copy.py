import os
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
from utils import read_filepaths
from PIL import Image
from torchvision import transforms

import h5py

class COVIDxDataset(Dataset):
    """
    Code for reading the COVIDxDataset
    """

    def __init__(self, mode, n_classes=3, dataset_path='./datasets', dim=(256, 256), num_samples=None):
        
        COVIDxDICT = {'pneumonia': 0, 'normal': 1, 'COVID-19': 2}

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transformer = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop((dim[0]), scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        val_transformer = transforms.Compose([
            transforms.Resize(dim[0]),
            transforms.CenterCrop(dim[0]),
            transforms.ToTensor(),
            normalize
        ])

        
        self.root = str(dataset_path) + '/' + mode + '/'

        self.CLASSES = n_classes
        self.dim = dim
        self.COVIDxDICT = {'pneumonia': 0, 'normal': 1, 'COVID-19': 2}
        testfile = str(dataset_path) + '/' + 'test_split_v3.txt'
        trainfile = str(dataset_path) + '/' + 'train_split_v3.txt'
        if (mode == 'train'):
            self.paths, self.labels = read_filepaths(trainfile, num_samples)
            self.transform = train_transformer
        elif (mode == 'test'):
            self.paths, self.labels = read_filepaths(testfile, num_samples)
            self.transform = val_transformer
        print("{} examples =  {}".format(mode, len(self.paths)))
        self.mode = mode
        
        # Create the hdf5 version of the dataset if needed


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image_tensor = self.load_image(self.root + self.paths[index], self.dim, augmentation=self.mode)

        label_tensor = torch.tensor(self.COVIDxDICT[self.labels[index]], dtype=torch.long)

        return image_tensor, label_tensor

    def load_image(self, img_path, dim, augmentation='test'):

        # look for h5 version
        dir = os.path.dirname(img_path) + '/h5'
        if not os.path.exists(dir):
            os.mkdir(dir)

        m5fn = dir + '/' + os.path.basename(img_path)
        if os.path.exists(m5fn):
            with h5py.File(m5fn, "r") as f:
                image = Image.open(m5fn).convert('RGB')

        # Create an h5 version of tyhe image
        else:
            if not os.path.exists(img_path):
                print("IMAGE DOES NOT EXIST {}".format(img_path))
            
            image = Image.open(img_path).convert('RGB')

            with h5py.File(m5fn, "w") as f:
                image = Image.open(m5fn).convert('RGB')

        
        image = image.resize(dim)

        # Data augmentation
        image_tensor = self.transform(image)

        return image_tensor

