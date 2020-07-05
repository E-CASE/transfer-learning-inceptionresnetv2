import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
from inceptionresnetv2 import InceptionResNetV2
from labelsmoothing_crossentropy import LabelSmoothLoss
import argparse
import logging
import datetime
from pathlib import Path
from torch.utils.data import DataLoader


def log_string(str):
    logger.info(str)
    print(str)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batchSize', type=int, default=8, help='input batch size')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument(
        '--learning_rate', type=float, help='learning rate of optimizer', default=3e-4)
    parser.add_argument(
        '--train_epochs', type=int, help='number of epochs to train model', default=80)
    parser.add_argument(
        '--steplr_stepsize', type=int, help='step size of the step learning rate scheduler', default=9)
    parser.add_argument(
        '--steplr_gamma', type=float, help='gamma of the step learning rate scheduler', default=0.7)
    parser.add_argument(
        '--smoothing', type=float, help='label smoothing epsilon', default=0.2)
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')

    return parser.parse_args()


args = parse_args()

show_images = False
fine_tune = True
label_smoothing = args.smoothing

"Create DIR for logging"
timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
experiment_dir = Path('./log/')
experiment_dir.mkdir(exist_ok=True)
experiment_dir = experiment_dir.joinpath('classification')
experiment_dir.mkdir(exist_ok=True)
if args.log_dir is None:
    experiment_dir = experiment_dir.joinpath(timestr)
else:
    experiment_dir = experiment_dir.joinpath(args.log_dir)
experiment_dir.mkdir(exist_ok=True)
checkpoints_dir = experiment_dir.joinpath('checkpoints/')
checkpoints_dir.mkdir(exist_ok=True)
log_dir = experiment_dir.joinpath('logs/')
log_dir.mkdir(exist_ok=True)

'''LOG'''
args = parse_args()
logger = logging.getLogger("Model")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('%s.txt' % (log_dir))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
log_string('PARAMETER ...')
log_string(args)

mean = np.array([0.9824, 0.9824, 0.9824])
std = np.array([0.0991, 0.0991, 0.0991])

data_transforms = {
    'train': transforms.Compose(
        [transforms.RandomAffine(degrees=[-5, 5], translate=(0, 0.02), fillcolor=(255, 255, 255)),
         transforms.RandomHorizontalFlip(),
         transforms.Resize(299),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)]),

    'test': transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

log_string('''Train Transforms:
                mean = np.array([0.9824, 0.9824, 0.9824])
                std = np.array([0.0991, 0.0991, 0.0991])
                transforms.RandomAffine(degrees=[-5, 5], translate=(0, 0.02), fillcolor=(255, 255, 255)),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(299),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)''')

data_dir = 'SHREC14-imagefolder-no-val'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}

train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=int(args.batchSize),
                                           shuffle=True, num_workers=int(args.workers))
test_loader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=int(args.batchSize),
                                          shuffle=False, num_workers=int(args.workers))
dataloaders = {'train': train_loader, 'test': test_loader}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if show_images:
    def imshow(inp, title):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        plt.title(title)
        plt.show()


    # Get a batch of training data
    inputs, classes = next(iter(train_loader))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])


def train_model(model, criterion, optimizer, scheduler, num_epochs=1):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # list to plot train loss and train accuracy
    epoch_loss_train = []
    epoch_acc_train = []

    # list to plot validation loss and validation accuracy
    epoch_loss_val = []
    epoch_acc_val = []

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['instance_acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    for epoch in tqdm(range(start_epoch, num_epochs)):

        # each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # set model to training mode
            else:
                model.eval()  # set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0.0

            # iterate over data (batches)
            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train mode
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase:
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                # cross entropy loss divided by batch hence multiply by batch to correct
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # append training loss and acc
            if phase == 'train':
                epoch_loss_train.append(epoch_loss)
                epoch_acc_train.append(epoch_acc)

            # append validation loss and acc
            if phase == 'test':
                epoch_loss_val.append(epoch_loss)
                epoch_acc_val.append(epoch_acc)

            log_string(f'epoch:{epoch} {phase} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': best_acc,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

        print()

    # save epoch loss and acc data for training and validation data
    with open(str(experiment_dir) + '/epoch_loss_train.txt', 'w') as f:
        for s in epoch_loss_train:
            f.write(str(s) + "\n")
    f.close()
    with open(str(experiment_dir) + '/epoch_acc_train.txt', 'w') as f:
        for s in epoch_acc_train:
            f.write(str(s) + "\n")
    f.close()
    with open(str(experiment_dir) + '/epoch_loss_val.txt', 'w') as f:
        for s in epoch_loss_val:
            f.write(str(s) + "\n")
    f.close()
    with open(str(experiment_dir) + '/epoch_acc_val.txt', 'w') as f:
        for s in epoch_acc_val:
            f.write(str(s) + "\n")
    f.close()

    time_elapsed = time.time() - since
    log_string('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    log_string('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Finetuning inceptionresnetv2

if fine_tune:
    # load pre-trained inception model
    model = InceptionResNetV2()
    FILE = 'inceptionresnetv2-520b38e4.pth'
    model.load_state_dict(torch.load(FILE))

    # change the number of classes in final layer to number of classes in shrec14, 171 sketch classes
    num_ftrs = model.last_linear.in_features
    model.last_linear = nn.Linear(num_ftrs, len(class_names))

    model = model.to(device)
    criterion = LabelSmoothLoss(label_smoothing, len(class_names))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.steplr_stepsize, gamma=args.steplr_gamma)
    print('Training Model')
    model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=args.train_epochs)

    # test model
    print('Testing Model')
    model.eval()
    with torch.no_grad():
        since = time.time()
        correct = 0
        total = 0

        n_class_correct = [0 for i in range(len(class_names))]
        n_class_samples = [0 for i in range(len(class_names))]

        for images, labels in tqdm(dataloaders['test']):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # find per class accuracy
            for i in range(list(labels.size())[0]):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        total_acc = 100.0 * correct / total
        print(f'Test Accuracy of the network: {total_acc} %')

        per_cls_acc = []
        # print per class accuracy
        for i in range(len(class_names)):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            per_cls_acc.append(acc)
            print(f'Accuracy of {class_names[i]}: {acc:.4f} %')

        with open(str(checkpoints_dir) + 'test_acc_per_cls.txt', 'w') as f:
            for s in range(len(per_cls_acc)):
                f.write(str(class_names[s]) + ' ' + str(per_cls_acc[s]) + "\n")
        f.close()

        time_elapsed = time.time() - since
        print('Testing complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    # save model state_dict
    FILE = str(experiment_dir) + '/model_{}_{:.2f}.pth'.format(label_smoothing, total_acc)
    torch.save(model.state_dict(), FILE)
