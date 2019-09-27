'''VGG in Pytorch.'''
import torch
import torch.nn as nn

cfg = {
    'CIFAR100': [96, 96, 96, 96, 96, 'M', 192, 192, 192, 192, 'M', 384, 384, 384, 384, 'M'],
    'CIFAR10': [64, 64, 64, 64, 64, 'M', 96, 96, 96, 96, 'M', 128, 128, 128, 128, 'M']
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name]) # 全部的卷积和池化层
        self.fc1 = nn.Linear(15488, 256)
        self.classifier = nn.Linear(256, num_classes)  # 全连接层

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        feature = out
        out = self.classifier(out)
        return out, feature

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, stride=1, padding=2),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def test():
    net = VGG('CIFAR10')
    x = torch.randn(1,3,32,32)
    y, feature = net(x)
    print(y.size(), feature.size())

# test()
