import os

from torch.utils.data import DataLoader

from data_loader.covid_ct_dataset import CovidCTDataset
from data_loader.covidxdataset import COVIDxDataset

print ("Importing train.py")

def initialize_datasets(args, train_size=None, val_size=None, use_transform=True):

    train_params = {'batch_size': args.batch_size,
                    'shuffle': True,
                    'num_workers': 1}

    val_params = {'batch_size': args.batch_size,
                   'shuffle': False,
                   'num_workers': 1}

    if args.dataset_name == 'COVIDx':
        print("Loading COVIDx dataset")

        train_dataset = COVIDxDataset(mode='train', h5=args.h5, n_classes=args.classes, dataset_path=args.root_path, dim=(224, 224), num_samples=train_size, use_transform=use_transform)
        val_dataset = COVIDxDataset(mode='test', h5=args.h5, n_classes=args.classes, dataset_path=args.root_path, dim=(224, 224), num_samples=val_size, use_transform=use_transform)
        test_dataset = None

        training_loader = DataLoader(train_dataset, **train_params)
        val_loader = DataLoader(val_dataset, **val_params)
        test_loader = None

    return train_dataset, val_dataset, test_loader