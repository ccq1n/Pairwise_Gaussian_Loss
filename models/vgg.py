'''VGG in Pytorch.'''
import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, vgg_net):
        super(VGG, self).__init__()
        self.classifier = nn.Linear(vgg_net['arch'][-2], vgg_net['data_class'].numClass)
        self.imageSize = vgg_net['data_class'].imageSize
        self.imageChannel = vgg_net['data_class'].imageChannel
        self.features = self._make_layers(vgg_net['arch'])

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        feature = out
        out = self.classifier(out)
        return out, feature

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.imageChannel
        in_size = self.imageSize
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                in_size = in_size // 2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, stride=1, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=in_size, stride=1)]
        return nn.Sequential(*layers)
