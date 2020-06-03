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

    def __init__(self, mode, h5=False, n_classes=3, dataset_path='./datasets', dim=(224, 224), num_samples=None, use_transform=True):
        
        COVIDxDICT = {'pneumonia': 0, 'normal': 1, 'COVID-19': 2}

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        null_transformer = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_transformer = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        val_transformer = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])


        self.CLASSES = n_classes
        self.dim = dim
        self.h5 = h5
        self.COVIDxDICT = {'pneumonia': 0, 'normal': 1, 'COVID-19': 2}

        if (mode == 'train'):
            self.transform = train_transformer
        elif (mode == 'test'):
            self.transform = val_transformer

        if not use_transform:
            self.transform = null_transformer

        self.mode = mode

        # Load an h5 dataset
        lend = 0
        self.h5filename = mode + '.h5'

        if h5:
            with h5py.File(self.h5filename, 'r') as db:
                lens=len(db['labels'])
                self.len = lens 
        else:
            self.root = str(dataset_path) + '/' + mode + '/'
            testfile = str(dataset_path) + '/' + 'test_split_v3.txt'
            trainfile = str(dataset_path) + '/' + 'train_split_v3.txt'

            if (mode == 'train'):
                self.paths, self.labels = read_filepaths(trainfile, num_samples)
            elif (mode == 'test'):
                self.paths, self.labels = read_filepaths(testfile, num_samples)
            self.len = len(self.paths)

        if num_samples and self.len > num_samples:
            self.len = num_samples

        print("{} examples =  {}".format(self.mode, self.len))
        
    def __len__(self):
        if self.h5:
            with h5py.File(self.h5filename, 'r') as db:
                lens=len(db['labels'])
                return lens 
            
        return self.len

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        if self.h5:
            with h5py.File(self.h5filename, 'r') as hf:

                d = hf['images'][index]
                l = hf['labels'][index]

                npimg = np.transpose(d,(1,2,0))
                image_pil = Image.fromarray((npimg * 255).astype(np.uint8))
                label_tensor = torch.tensor(l, dtype=torch.long)

        else:
            image_pil = self.load_image(self.root + self.paths[index], self.dim)
            label_tensor = torch.tensor(self.COVIDxDICT[self.labels[index]], dtype=torch.long)

        image_tensor = self.transform(image_pil)
        return image_tensor, label_tensor

    def load_image(self, img_path, dim, augmentation='test'):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))
        image = Image.open(img_path).convert('RGB')
        image = image.resize(dim)
        return image

