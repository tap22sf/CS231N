import os

from torch.utils.data import DataLoader

from data_loader.covid_ct_dataset import CovidCTDataset
from data_loader.covidxdataset import COVIDxDataset

print ("Importing train.py")

def initialize_datasets(args):

    train_params = {'batch_size': args.batch_size,
                    'shuffle': True,
                    'num_workers': 1}

    val_params = {'batch_size': args.batch_size,
                   'shuffle': False,
                   'num_workers': 1}

    if args.dataset_name == 'COVIDx':
        print("Loading COVIDx dataset")

        train_dataset = COVIDxDataset(mode='train', n_classes=args.classes, dataset_path=args.root_path, dim=(224, 224))
        val_dataset = COVIDxDataset(mode='test', n_classes=args.classes, dataset_path=args.root_path, dim=(224, 224))
        test_dataset = None

        training_loader = DataLoader(train_dataset, **train_params)
        val_loader = DataLoader(val_dataset, **val_params)
        test_loader = None

    return train_dataset, val_dataset, test_loader