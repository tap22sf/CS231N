import argparse

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import utils.util as util
from trainer.train import initialize, train, validation

import matplotlib.pyplot as plt

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def main():
    args = get_arguments()
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load the models and generators
    model, optimizer, training_generator, val_generator, test_generator = initialize(args)
    model.to(device)

    #print(model)

    best_pred_loss = 1000.0
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-5, verbose=True)
    
    print('Checkpoint folder ', args.save)
    if args.tensorboard:
        writer = SummaryWriter('./runs/' + util.datestr())
        images, labels = next(iter(training_generator))
        writer.add_graph(model, images.to(device))
    else:
        writer = None

    args.nEpochs = 100
    for epoch in range(1, args.nEpochs + 1):
        train_loss, train_acc = train(args, model, training_generator, optimizer, epoch, writer, device)
        val_loss, val_acc, confusion_matrix = validation(args, model, val_generator, epoch, writer, device)
        print("{:3d} train loss/acc : {:5.2f}, % {:3.1f}  val loss/acc : {:5.2f}, % {:3.1f}"
              .format(epoch, train_loss, 100.0*train_acc, val_loss, 100.0*val_acc))
        
        # Add the tensorboard data
        if args.tensorboard:
            writer.add_scalar('Loss/train',         train_loss, epoch)
            writer.add_scalar('Accuracy/train',     train_acc, epoch)
            
            writer.add_scalar('Loss/val',           val_loss, epoch)
            writer.add_scalar('Accuracy/val',       val_acc, epoch)

        best_pred_loss = util.save_model(model, optimizer, args, val_loss, epoch, best_pred_loss, confusion_matrix)
        scheduler.step(val_loss)

    writer.close()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=24, help='batch size for training')
    parser.add_argument('--log_interval', type=int, default=1000, help='steps to print metrics and loss')
    parser.add_argument('--dataset_name', type=str, default='COVIDx', help='dataset name COVIDx or COVID_CT')
    parser.add_argument('--nEpochs', type=int, default=250, help='total number of epochs')
    parser.add_argument('--seed', type=int, default=123, help='select seed number for reproducibility')
    parser.add_argument('--classes', type=int, default=3, help='dataset classes')
    parser.add_argument('--lr', default=2e-5, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', default=1e-7, type=float,
                        help='weight decay (default: 1e-6)')
    parser.add_argument('--tensorboard', action='store_true', default=True,
                        help='use tensorboard for loggging and visualization')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='COVIDNet_large',
                        choices=('COVIDNet_small', 'resnet18', 'mobilenet_v2', 'densenet169', 'COVIDNet_large'))
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--root_path', type=str, default='G:/combinedDataset/data',
                        help='path to dataset ')
    parser.add_argument('--save', type=str, default='./save/COVIDNet' + util.datestr(),
                        help='path to checkpoint save directory ')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()

