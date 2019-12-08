from data import MNIST, CIFAR10, CIFAR100, SVHN

vgg_set = {
    'MNIST': {'arch': [64, 64, 64, 64, 'M', 64, 64, 64, 'M', 64, 64, 64, 'M'],
              'data_class': MNIST},
    'CIFAR100': {'arch': [96, 96, 96, 96, 96, 'M', 192, 192, 192, 192, 'M', 384, 384, 384, 384, 'M'],
                 'data_class': CIFAR100},
    'CIFAR10': {'arch': [64, 64, 64, 64, 64, 'M', 96, 96, 96, 96, 'M', 128, 128, 128, 128, 'M'],
                'data_class': CIFAR10},
    'SVHN': {'arch': [64, 64, 64, 64, 64, 'M', 96, 96, 96, 96, 'M', 128, 128, 128, 128, 'M'],
             'data_class': SVHN}
}

# loss function set
PARM = 1.0
# ====> Gaussian
BETA = 0.05
# ====> Sigmoid
AERFA = 1.0
# ====> Cauchy
GAMMA = 5