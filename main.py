from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import config as cfg
from models import *
from loss_function import *
from data import *

# choose GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0     # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
# data = MNIST('/data/qinyuxiang/Dataset/MNIST')
data = SVHN('/data/qinyuxiang/Paper_test/cifar_pytorch/data')
trainloader = data.getTrainLoader()
testloader = data.getTestLoader()

# Model
print('==> Building model..')
net = VGG(cfg.vgg_set[data.dataName])
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, features = net(inputs)

        cross_entropy_loss = criterion(outputs, targets)

        pairwise_loss = pairwise_gaussian_loss(euclidean_dist_all(features), targets, data.numClass, beta=cfg.BETA)
        # pairwise_loss = pairwise_sigmoid_loss(euclidean_dist_all(features), targets, data.numClass, aerfa=cfg.AERFA)
        # pairwise_loss = pairwise_cauchy_loss(euclidean_dist_all(features), targets, data.numClass, gamma=cfg.GAMMA)
        # pairwise_loss = pairwise_hinge_loss(euclidean_dist_all(features), targets, data.numClass)

        loss = cross_entropy_loss + cfg.PARM * pairwise_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if(batch_idx % 50 == 0):
            print('[Epoch:%d] Learning Rate: %.05f | Loss: %.03f | Acc: %.3f '
              % (epoch, args.lr, train_loss / (batch_idx + 1), 100. * correct / total))

def test(epoch):
    global best_acc
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, features = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    print('current accuracy: %.3f | best accuracy: %.3f' % (acc, best_acc))

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

for epoch in range(start_epoch, start_epoch+50):
    train(epoch)
    test(epoch)
