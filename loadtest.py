import numpy as np
import torch 
import argparse
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torchvision
import matplotlib.pyplot as plt    

num_workers = 3

def main():

    train = True
    
    args = get_arguments()
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load the models and generators
    dataset, val_dataset, test_generator = initialize_datasets(args)
    
    # Load from a model if not training
    model = select_model(args)
    if not train:
        #path = './save/COVIDNet20200520_0504/COVIDNet_large_best_checkpoint.pt'
        path = './gold_save/COVIDNet_large_best_checkpoint.pt'
        checkpoint  = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        
    model.to(device)
    
    best_pred_loss = 1000.0
    
    print('Checkpoint folder ', args.save)
    if args.tensorboard and train:
        writer = SummaryWriter('./runs/' + util.datestr())
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers = num_workers)
        images, labels = next(iter(data_loader))
        writer.add_graph(model, images.to(device))

        img_grid = torchvision.utils.make_grid(images)
        
        # show images
        newimg = matplotlib_imshow(img_grid, one_channel=True)

        ## write to tensorboard
        writer.add_image('Xrays',img_grid)
        
    else:
        writer = None

    if train:

        best_pred_loss = 1000
        optimizer = select_optimizer(args, model)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-5, verbose=True)

        for epoch in range(1, args.nEpochs + 1):

            # Train 1 epoch
            train_metrics, writer_step = train2(model, args, device, writer, scheduler, optimizer, dataset, epoch)

            # Run Inference on val set
            val_loss = inference (args, model, val_dataset, epoch, writer, device, writer_step)
            best_pred_loss = util.save_model(model, args, val_loss, epoch, best_pred_loss, confusion_matrix)
            scheduler.step(val_loss)

    # Just evaluate a trained model
    else:
        inference (args, model, val_dataset, 1, writer, device, 0)

   
def inference(args, model, val_dataset, epoch, writer, device, writer_step):

    # Run Inference on val set
    val_metrics, confusion_matrix = val2(args, model, val_dataset, epoch, writer, device)
    val_loss = val_metrics.avg('loss')
    val_metrics.write_tb (writer_step)

    # Print a summary message
    print_summary(args, epoch, val_metrics)

    # Print the confusion matrix
    print('Confusion Matrix\n{}'.format(confusion_matrix.cpu().numpy()))

    return val_loss


def train2(model, args, device, writer, scheduler, optimizer, dataset, epoch):
    
    # Set train mode
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='mean')

    metric_ftns = ['loss', 'correct', 'total', 'accuracy']
    metrics = MetricTracker(*[m for m in metric_ftns], writer=writer, mode='train')
    metrics.reset()

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers = num_workers)
    
    for batch_idx, input_tensors in enumerate(data_loader):
        input_data, target = input_tensors[0].to(device), input_tensors[1].to(device)

        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        correct, total, acc = accuracy(output, target)
        metrics.update_all_metrics({'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc})
    
        # Save TB stats
        writer_step = (epoch - 1) * len(data_loader) + batch_idx
        if (batch_idx % args.log_interval == 0):
            metrics.write_tb (writer_step)
            num_samples = batch_idx * args.batch_size
            print_stats(args, epoch, num_samples, data_loader, metrics)
        #if batch_idx >3: break

    return metrics, writer_step

def val2(args, model, val_dataset, epoch, writer, device):
    
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction='mean')
    confusion_matrix = torch.zeros(args.classes, args.classes)
    metric_ftns = ['loss', 'correct', 'total', 'accuracy']
    metrics = MetricTracker(*[m for m in metric_ftns], writer=writer, mode='val')
    metrics.reset()
    
    data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers = num_workers)

    with torch.no_grad():
        for batch_idx, input_tensors in enumerate(data_loader):
            torch.cuda.empty_cache()
            input_data, target = input_tensors[0].to(device), input_tensors[1].to(device)
            output = model(input_data)
            loss = criterion(output, target)
            
            num_samples = batch_idx * args.batch_size + 1
            correct, total, acc = accuracy(output, target)
            metrics.update_all_metrics({'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc})
            
            _, preds = torch.max(output, 1)

            for t, p in zip(target.cpu().view(-1), preds.cpu().view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
         
    return metrics, confusion_matrix


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=23, help='batch size for training')
    parser.add_argument('--log_interval', type=int, default=50, help='steps to print metrics and loss')
    parser.add_argument('--dataset_name', type=str, default='COVIDx', help='dataset name COVIDx or COVID_CT')
    parser.add_argument('--nEpochs', type=int, default=100, help='total number of epochs')
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

    #plt.show()
    return img


# Multi-processor safe:)
if __name__ == '__main__':
    from utils.util import print_stats, print_summary, select_model, select_optimizer, MetricTracker
    import utils.util as util
    from trainer.train import initialize_datasets
    from data_loader.covidxdataset import COVIDxDataset
    from model.metric import accuracy

    main()

    
