import os
import torch
import torchvision
import torchvision.transforms as transforms

class MNIST:
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    imageSize = 28
    batchSize = 128
    numClass = 10
    imageChannel = 1
    dataName = 'MNIST'

    def __init__(self, fileDir):
        self.fileDir = fileDir

    def getTrainLoader(self):
        trainSet = torchvision.datasets.MNIST(root=self.fileDir, train=True, download=True, transform=MNIST.transform)
        trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=MNIST.batchSize, shuffle=True, num_workers=2)
        return trainLoader

    def getTestLoader(self):
        testSet = torchvision.datasets.MNIST(root=self.fileDir, train=False, download=True, transform=MNIST.transform)
        testLoader = torch.utils.data.DataLoader(testSet, batch_size=MNIST.batchSize, shuffle=False, num_workers=2)
        return testLoader

class SVHN:
    transformTrain = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transformTest = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    imageSize = 32
    batchSize = 128
    numClass = 10
    imageChannel = 3
    dataName = 'SVHN'

    def __init__(self, fileDir):
        self.fileDir = fileDir

    def getTrainLoader(self):
        trainSet = torchvision.datasets.SVHN(root=self.fileDir, split='train', download=True, transform=SVHN.transformTrain)
        extraSet = torchvision.datasets.SVHN(root=self.fileDir, split='extra', download=True, transform=SVHN.transformTrain)
        trainSet += extraSet
        trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=SVHN.batchSize, shuffle=True, num_workers=2)
        return trainLoader

    def getTestLoader(self):
        testSet = torchvision.datasets.SVHN(root=self.fileDir, split='test', download=True, transform=SVHN.transformTest)
        testLoader = torch.utils.data.DataLoader(testSet, batch_size=SVHN.batchSize, shuffle=False, num_workers=2)
        return testLoader

class CIFAR10:
    transformTrainWithoutDataAug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transformTrainWithDataAug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transformTest = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    imageSize = 32
    batchSize = 128
    numClass = 10
    imageChannel = 3
    dataName = 'CIFAR10'

    def __init__(self, fileDir, dataAug=True):
        self.fileDir = fileDir
        self.dataAug = dataAug

    def getTrainLoader(self):
        if self.dataAug:
            transformTrain = CIFAR10.transformTrainWithDataAug
        else:
            transformTrain = CIFAR10.transformTrainWithoutDataAug
        trainSet = torchvision.datasets.CIFAR10(root=self.fileDir, train=True, download=True, transform=transformTrain)
        trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=CIFAR10.batchSize, shuffle=True, num_workers=2)
        return trainLoader

    def getTestLoader(self):
        testSet = torchvision.datasets.CIFAR10(root=self.fileDir, train=False, download=True, transform=CIFAR10.transformTest)
        testLoader = torch.utils.data.DataLoader(testSet, batch_size=CIFAR10.batchSize, shuffle=False, num_workers=2)
        return testLoader

class CIFAR100:
    transformTrainWithoutDataAug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transformTrainWithDataAug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transformTest = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    imageSize = 32
    batchSize = 128
    numClass = 100
    imageChannel = 3
    dataName = 'CIFAR100'

    def __init__(self, fileDir, dataAug=True):
        self.fileDir = fileDir
        self.dataAug = dataAug

    def getTrainLoader(self):
        if self.dataAug:
            transformTrain = CIFAR100.transformTrainWithDataAug
        else:
            transformTrain = CIFAR100.transformTrainWithoutDataAug
        trainSet = torchvision.datasets.CIFAR100(root=self.fileDir, train=True, download=True, transform=transformTrain)
        trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=CIFAR100.batchSize, shuffle=True, num_workers=2)
        return trainLoader

    def getTestLoader(self):
        testSet = torchvision.datasets.CIFAR100(root=self.fileDir, train=False, download=True,
                                               transform=CIFAR100.transformTest)
        testLoader = torch.utils.data.DataLoader(testSet, batch_size=CIFAR100.batchSize, shuffle=False, num_workers=2)
        return testLoader

class ImageNet:
    imageSize = 224
    batchSize = 128
    numClass = 1000
    imageChannel = 3
    dataName = 'ImageNet'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def __init__(self, fileDir):
        self.fileDir = fileDir

    def getTrainLoader(self):
        trainDir = os.path.join(self.fileDir, 'train')
        trainLoader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(
                trainDir, transforms.Compose([
                transforms.RandomResizedCrop(self.imageSize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])),
            batch_size=self.batchSize, shuffle=True,
            num_workers=8, pin_memory=True)
        return trainLoader

    def getTestLoader(self):
        testDir = os.path.join(self.fileDir, 'val')
        testLoader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(
                testDir, transforms.Compose([
                transforms.Resize(int(self.imageSize / 0.875)),
                transforms.CenterCrop(self.imageSize),
                transforms.ToTensor(),
                self.normalize,
            ])),
            batch_size=self.batchSize, shuffle=False,
            num_workers=8, pin_memory=True)
        return testLoader