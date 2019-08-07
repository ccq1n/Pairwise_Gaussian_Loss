# Pairwise Gaussian Loss for Convolutional Neural Networks

## Introduction

We introduce a pairwise gaussian loss (PGL) for convolutional neural networks. PGL can greatly improve the generalization ability of CNNs, so it is very suitable for general classification, feature embedding and biometrics verification. We give the 2D feature visualization on MNIST to illustrate our PGL.

<img src="image/Softmax_vs_Gloss.png" width="50%" height="50%">

## Prepare dataset

[MNIST](http://yann.lecun.com/exdb/mnist/), 
[CIFAR10](http://www.cs.toronto.edu/~kriz/cifar.html), CIFAR10+, 
[CIFAR100](http://www.cs.toronto.edu/~kriz/cifar.html), CIFAR100+ and 
[SVHN](http://ufldl.stanford.edu/housenumbers/) datasets.

Download dataset and create some directories:

```plain
Pairwise_Gaussion_Loss
└── Dataset
       ├── MNIST     <-- 7481 train data
       |   ├── train <-- empty directory
       |   └── test  <-- empty directory
       └── CIFAR10   <-- 
       |   ├── train <-- empty directory
       |   └── test  <-- empty directory
       └── CIFAR100
       |   ├── train <-- 
       |   └── test  <-- empty directory
       └── CIFAR100
       |   ├── train <-- 
       |   └── test  <-- empty directory
       └── SVHN
       |   ├── train <-- 
       |   └── test  <-- empty directory
       |
```

## Files
- Tensorflow

  Under the VGG network structure, we compare the classification effects of our PGL, original Softmax, 
  Softmax+Sigmoid [1], Softmax+Hinge-like [2], Softmax+Cauchy [3], and Center Loss [4].
   
- PyTorch

  Compare the effects of different network structures on the CIFAR10 with data augmentation, 
  such as MobileNetV2, ResNet18, ResNeXt29, DenseNet121, PreActResNet18. 
  
- MNIST visualization
  
  We use jupyter to 

## References
[1] Yandong Wen, Kaipeng Zhang, Zhifeng Li, and Yu Qiao. A discriminative feature learning approach for
deep face recognition. In European conference on computer vision, pages 499–515. Springer, 2016.

[2]

[3]

[4]

[5]
