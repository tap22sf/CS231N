import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data_loader.covid_ct_dataset import CovidCTDataset
from data_loader.covidxdataset import COVIDxDataset
from model.metric import accuracy
from utils.util import print_stats, print_summary, select_model, select_optimizer


def initialize(args):

    model = select_model(args)
    optimizer = select_optimizer(args, model)

    train_params = {'batch_size': args.batch_size,
                    'shuffle': True,
                    'num_workers': 0}

    test_params = {'batch_size': args.batch_size,
                   'shuffle': False,
                   'num_workers': 0}

    if args.dataset_name == 'COVIDx':
        print("Loading COVIDx dataset")

        train_loader = COVIDxDataset(mode='train', n_classes=args.classes, dataset_path=args.root_path, dim=(224, 224))
        val_loader = COVIDxDataset(mode='test', n_classes=args.classes, dataset_path=args.root_path, dim=(224, 224))
        test_loader = None

        training_generator = DataLoader(train_loader, **train_params)
        val_generator = DataLoader(val_loader, **test_params)
        test_generator = None

    return model, optimizer, training_generator, val_generator, test_generator


def train(args, model, trainloader, optimizer, epoch, writer, device):
    
    # Set train mode
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='mean')

    running_correct = 0
    running_total = 0

    for batch_idx, input_tensors in enumerate(trainloader):
        optimizer.zero_grad()
        input_data, target = input_tensors[0].to(device), input_tensors[1].to(device)

        output = model(input_data)

        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        correct, total, acc = accuracy(output, target)
        running_correct += correct
        running_total += total

        num_samples = batch_idx * args.batch_size + 1
        
        # Early out
        #if batch_idx >5:
        #    break
    
    acc = running_correct/running_total

    return loss.item(), acc

def validation(args, model, testloader, epoch, writer, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='mean')
    confusion_matrix = torch.zeros(args.classes, args.classes)

    running_correct = 0
    running_total = 0
    with torch.no_grad():
        for batch_idx, input_tensors in enumerate(testloader):

            input_data, target = input_tensors[0].to(device), input_tensors[1].to(device)
            output = model(input_data)
            loss = criterion(output, target)

            correct, total, acc = accuracy(output, target)
            running_correct += correct
            running_total += total
            
            num_samples = batch_idx * args.batch_size + 1
            _, preds = torch.max(output, 1)

            for t, p in zip(target.cpu().view(-1), preds.cpu().view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            # Early out
            #if batch_idx >5:
            #    break

    acc = running_correct/running_total

    #print('Confusion Matrix\n{}'.format(confusion_matrix.cpu().numpy()))
    return loss.item(), acc, confusion_matrix
