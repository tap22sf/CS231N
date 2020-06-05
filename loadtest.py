import numpy as np
import torch 
import argparse
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torchvision
import matplotlib.pyplot as plt    
import h5py

num_workers = 2
num_workers = 0

def main():

    save_h5 = False    
    #save_h5 = True

    args = get_arguments()
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Set the path for the dataset
    args.root_path = 'G:/datasets/covidx_v3_data'
    weight_path = './save/COVIDNet20200520_0709/COVIDNet_large_best_checkpoint.pt'

    if save_h5:
        args.h5 = False
        dataset, val_dataset, test_generator = initialize_datasets(args, use_transform=False)
        #dataset, val_dataset, test_generator = initialize_datasets(args,  train_size=10, val_size=100, use_transform=False)

        # Save of the datasets as h5 is desired
        for ds in (dataset, val_dataset, test_generator):
            if ds: 
                imgs =[]
                labels = []
                for n, imgTen in enumerate (ds):

                    # Create an h5 version of the image
                    image = imgTen[0].numpy()
                    label = imgTen[1].numpy()
                    imgs.append(image)
                    labels.append(label)

                imgs = np.array(imgs)
                labels = np.array(labels)

                # Create the dataset
                hf = h5py.File(ds.mode+'.h5', 'w') 
                hf.create_dataset('images', data=imgs)
                hf.create_dataset('labels', data=labels)

                hf.close()

    # Load the h5 datasets if available
    args.h5 = True
    #args.h5 = False
    #dataset, val_dataset, test_dataset = initialize_datasets(args, train_size=300, val_size=100)
    dataset, val_dataset, test_dataset = initialize_datasets(args)
    
    # Reload weights if desired
    args.batch_size = 256
    args.log_interval = 10
    args.nEpochs = 100
    args.nEpochs = 3

    test_list = [   
        #[True, False, 'COVIDNet_small',  False, None, 100, 36, 50, 5e-5],
        #[True, False, 'COVIDNet_large',  False, None, 100, 28, 50, 5e-5],
        #[True, True,  'resnet18',        False, None, 100, 256, 50, 2e-5],
        [True, True,  'mobilenet_v2',    False, None, 100, 256, 50, 2e-5],
        [True, True,  'densenet169',     False, None, 100, 256, 50, 2e-5],
        [True, True,  'resnext50_32x4d', False, None, 100, 256, 50, 2e-5]
    ]

    # Iterate over tests
    for trainme, transfer, model_name, reload_weights, weight_path, args.nEpochs, args.batch_size, args.log_interval, args.lr in test_list:
        if transfer: 
            code = "_Transfer"
        else:
            code = ""
    
        id = model_name+code + util.datestr()

        model = select_model(model_name, args.classes)
        if reload_weights:
            print("Loading model with weights from: {}".format(weight_path))
            checkpoint  = torch.load(weight_path)
            model.load_state_dict(checkpoint['state_dict'])
        
        model.to(device)
        #print(model)
        
        # Freeze model if transfer learning
        if transfer:
            set_parameter_requires_grad(model)

        if args.tensorboard and train:
            writer = SummaryWriter('./runs/' + id)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory= False, num_workers = num_workers)
            images, labels = next(iter(data_loader))
            writer.add_graph(model, images.to(device))

            img_grid = torchvision.utils.make_grid(images)
        
            # show images
            newimg = matplotlib_imshow(img_grid, one_channel=True)

            ## write to tensorboard
            writer.add_image('Xrays',img_grid)
        
        else:
            writer = None

        if trainme:
            best_score = 0
            optimizer = select_optimizer(args, model)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory= False, num_workers = num_workers)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory= False, num_workers = num_workers)

            for epoch in range(1, args.nEpochs + 1):

                # Train 1 epoch
                train_metrics, writer_step = train(model, args, device, writer,  optimizer, train_loader, epoch)
                
                # Run Inference on val set
                val_score, confusion_matrix = inference (args, model, val_loader, epoch, writer, device, writer_step)
                best_score = util.save_model(model, id, args, val_score, epoch, best_score, confusion_matrix)

        # Just evaluate a trained model
        else:
            inference (args, model, val_loader, 1, writer, device, 0)

   
def inference(args, model, val_loader, epoch, writer, device, writer_step):

    # Run Inference on val set
    val_metrics, cm  = val(args, model, val_loader, epoch, writer, device)
    val_metrics.write_tb (writer_step)

    # Print a summary message
    print_summary(args, epoch, val_metrics)

    # Print the confusion matrix
    print('Confusion Matrix\n{}'.format(cm.cpu().numpy()))

    val_score = val_metrics.avg('sens') *  val_metrics.avg('ppv') 

    return val_score, cm


def train(model, args, device, writer, optimizer, data_loader, epoch):
    
    # Set train mode
    model.train()

    criterion = nn.CrossEntropyLoss(reduction='mean')
    metric_ftns = ['loss', 'correct', 'total', 'accuracy', 'sens', 'ppv']
    metrics = MetricTracker(*[m for m in metric_ftns], writer=writer, mode='train')
    metrics.reset()

    cm = torch.zeros(args.classes, args.classes)

    for batch_idx, input_tensors in enumerate(data_loader):

        input_data, target = input_tensors[0].to(device), input_tensors[1].to(device)
    
        # Forward
        output = model(input_data)
        loss = criterion(output, target)

        correct, total, acc = accuracy(output, target)
        update_confusion_matrix(cm, output, target)
        metrics.update_all_metrics({'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc})

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save TB stats
        writer_step = (epoch - 1) * len(data_loader) + batch_idx 
        if ((batch_idx+1) % args.log_interval == 0):

            # Calculate confusion for this bucket
            ppv, sens = update_confusion_calc(cm)
            metrics.update_all_metrics({'sens': sens, 'ppv': ppv})
            cm = torch.zeros(args.classes, args.classes)

            metrics.write_tb (writer_step)

            num_samples = batch_idx * args.batch_size
            print_stats(args, epoch, num_samples, data_loader, metrics)


    return metrics, writer_step

def val(args, model, data_loader, epoch, writer, device):
    
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction='mean')
    metric_ftns = ['loss', 'correct', 'total', 'accuracy', 'ppv', 'sens']
    metrics = MetricTracker(*[m for m in metric_ftns], writer=writer, mode='val')
    metrics.reset()
    
    cm = torch.zeros(args.classes, args.classes)

    with torch.no_grad():
        for batch_idx, input_tensors in enumerate(data_loader):
            torch.cuda.empty_cache()
            input_data, target = input_tensors[0].to(device), input_tensors[1].to(device)
            
            # Forward
            output = model(input_data)
            loss = criterion(output, target)

            correct, total, acc = accuracy(output, target)
            update_confusion_matrix(cm, output, target)

            # Update the metrics record
            metrics.update_all_metrics({'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc})

        ppv, sens = update_confusion_calc(cm)
        metrics.update_all_metrics({'sens': sens, 'ppv': ppv})

    return metrics, cm


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20, help='batch size for training')
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
    parser.add_argument('--save', type=str, default='./save',
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

def set_parameter_requires_grad(model):
    frozen = 0
    
    for cnt, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            if cnt <= model.freezeLayerCnt:
                print ("{} - Freezing : {}".format(cnt,name))
                param.requires_grad = False
                frozen += 1
            else:
                print ("{} - Training : {}".format(cnt, name))
            
    print ("Total Params: {}\tFrozen Params: {}".format (cnt, frozen))

# Multi-processor safe:)
if __name__ == '__main__':
    from utils.util import print_stats, print_summary, select_model, select_optimizer, MetricTracker
    import utils.util as util
    from train.train import initialize_datasets
    from data_loader.covidxdataset import COVIDxDataset
    from model.metric import accuracy, update_confusion_calc, update_confusion_matrix

    main()

    
