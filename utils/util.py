import json
import os
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from model.model import CovidNet, CNN


def write_score(writer, iter, mode, metrics):
    writer.add_scalar(mode + '/loss', metrics.data['loss'], iter)
    writer.add_scalar(mode + '/acc', metrics.data['correct'] / metrics.data['total'], iter)


def write_train_val_score(writer, epoch, train_stats, val_stats):
    writer.add_scalars('Loss', {'train': train_stats[0],
                                'val': val_stats[0],
                                }, epoch)
    writer.add_scalars('Coeff', {'train': train_stats[1],
                                 'val': val_stats[1],
                                 }, epoch)

    writer.add_scalars('Air', {'train': train_stats[2],
                               'val': val_stats[2],
                               }, epoch)

    writer.add_scalars('CSF', {'train': train_stats[3],
                               'val': val_stats[3],
                               }, epoch)
    writer.add_scalars('GM', {'train': train_stats[4],
                              'val': val_stats[4],
                              }, epoch)
    writer.add_scalars('WM', {'train': train_stats[5],
                              'val': val_stats[5],
                              }, epoch)
    return


def showgradients(model):
    for param in model.parameters():
        print(type(param.data), param.size())
        print("GRADS= \n", param.grad)


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def save_checkpoint(state, path, filename='last'):
    name = os.path.join(path, filename + '_checkpoint.pt')
    torch.save(state, name)


def save_model(model, id, args, score, epoch, best_score, confusion_matrix):
    
    save_path = args.save + '/' + id 
    make_dirs(save_path)

    with open(save_path + '/training_arguments.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    is_best = False
    if score > best_score:
        is_best = True
        best_score = score
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict()},
                        save_path, "best")
        np.savetxt(save_path + '/best_confusion_matrix.csv', confusion_matrix.cpu().numpy(), delimiter=',')

    else:
        save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict()},  save_path, "last")

    return best_score


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_stats_files(path):
    train_f = open(os.path.join(path, 'train.csv'), 'w')
    val_f = open(os.path.join(path, 'val.csv'), 'w')
    return train_f, val_f


def read_json_file(fname):
    with open(fname, 'r') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json_file(content, fname):
    with open(fname, 'w') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def read_filepaths(file, num_samples):
    paths, labels = [], []
    with open(file, 'r') as f:
        lines = f.read().splitlines()

        for idx, line in enumerate(lines):
            if num_samples and idx >= num_samples: break

            if ('/ c o' in line):
                break
            subjid, path, label,_ = line.split(' ')

            paths.append(path)
            labels.append(label)


    return paths, labels

def select_model(name, classes):
    if name == 'COVIDNet_small':
        return CovidNet('small', n_classes=classes)
    elif name == 'COVIDNet_large':
        return CovidNet('large', n_classes=classes)
    elif name in ['resnet18', 'mobilenet_v2', 'densenet169', 'resnext50_32x4d']:
        return CNN(classes, name)


def select_optimizer(args, model):
    if args.opt == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=0.5, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data


def print_stats(args, epoch, num_samples, trainloader, metrics):
    print("Epoch:{:2d}\tSample:{:5d}/{:5d}\tTrain Loss:  {:.4f}\tTrain PPV:{:.2f}\tTrain Sens:{:.2f}\t\tTrain Accuracy:  {:.2f}".format(epoch,
                                                                                     num_samples,
                                                                                     len(trainloader) * args.batch_size,
                                                                                     metrics.avg('loss'),
                                                                                     metrics.avg('ppv'),
                                                                                     metrics.avg('sens'),
                                                                                     metrics.avg('accuracy')))

def print_summary(args, epoch, metrics):
    print("\nSUMMARY EPOCH:{:2d}\tVal Loss:{:.4f}\tVal PPV:{:.2f}\tVal Sens:{:.2f}\tVal Accuracy:{:.2f}\n".format(epoch,
                                                                                    metrics.avg('loss'),
                                                                                    metrics.avg('ppv'),
                                                                                    metrics.avg('sens'),
                                                                                    metrics.avg('accuracy')))



class MetricTracker:
    def __init__(self, *keys, writer=None, mode='/'):

        self.writer = writer
        self.mode = mode + '/'
        self.keys = keys
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def update_all_metrics(self, values_dict, n=1):
        for key in values_dict:
            self.update(key, values_dict[key], n)
        
    def write_tb(self, index):
        # Add the tensorboard data
        if self.writer:
            d = dict(self._data.average)
            self.writer.add_scalar(self.mode + 'Loss', d['loss'], index)
            self.writer.add_scalar(self.mode + 'Accuracy', d['accuracy'], index)
            self.writer.add_scalar(self.mode + 'Sensitivity', d['sens'], index)
            self.writer.add_scalar(self.mode + 'PPV', d['ppv'], index)

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def print_all_metrics(self):
        s = ''
        d = dict(self._data.average)
        for key in dict(self._data.average):
            s += "{} {:.4f}\t".format(key, d[key])

        return s
