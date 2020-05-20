import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data_loader.covid_ct_dataset import CovidCTDataset
from data_loader.covidxdataset import COVIDxDataset
from model.metric import accuracy
from utils.util import print_stats, print_summary, select_model, select_optimizer, MetricTracker

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


def train(args, model, trainloader, optimizer, epoch, writer, device):
    
    print("Training")

    # Set train mode
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='mean')

    metric_ftns = ['loss', 'correct', 'total', 'accuracy']
    train_metrics = MetricTracker(*[m for m in metric_ftns], writer=writer, mode='train')
    train_metrics.reset()

    running_correct = 0
    running_total = 0
    running_loss = 0
    cnt = 0
    
    maxalloc = torch.cuda.max_memory_allocated()

    for batch_idx, input_tensors in enumerate(trainloader):
        
        maxalloc = torch.cuda.max_memory_allocated()

        optimizer.zero_grad()
        input_data, target = input_tensors[0].to(device), input_tensors[1].to(device)

        maxalloc = torch.cuda.max_memory_allocated()

        output = model(input_data)

        loss = criterion(output, target)
        running_loss += loss.item()

        loss.backward()

        optimizer.step()
        correct, total, acc = accuracy(output, target)
        running_correct += correct
        running_total += total

        num_samples = batch_idx * args.batch_size + 1
        cnt += 1


        train_metrics.update_all_metrics({'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc},
                                         writer_step=(epoch - 1) * len(trainloader) + batch_idx)
        print_stats(args, epoch, num_samples, trainloader, train_metrics)

        # Early out
        #if batch_idx >3:
        #    break
    
    acc = running_correct/running_total
    ls =  running_loss/cnt

    return ls, acc

def validation(args, model, testloader, epoch, writer, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='mean')
    confusion_matrix = torch.zeros(args.classes, args.classes)

    running_correct = 0
    running_total = 0
    running_loss = 0
    cnt = 0
    with torch.no_grad():
        for batch_idx, input_tensors in enumerate(testloader):
            torch.cuda.empty_cache()
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
