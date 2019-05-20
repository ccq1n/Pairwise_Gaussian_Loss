# Pairwise Guassian Loss for Convolutional Neural Networks

By Yuxiang Qin, Guanjun Liu, Zhenchuan Li, Chungang Yan, Changjun Jiang

## Introduction

We introduce a pairwise guassian loss (PGL) for convolutional neural networks. PGL can greatly improve the generalization ability of CNNs, so it is very suitable for general classification, feature embedding and biometrics verification. We give the 2D feature visualization on MNIST to illustrate our PGL.

<img src="image/Softmax_vs_Gloss.png" width="50%" height="50%">

## Experiments

We are playing with [Tensorflow](https://tensorflow.google.cn/) on the 
[MNIST](http://yann.lecun.com/exdb/mnist/), 
[CIFAR10](http://www.cs.toronto.edu/~kriz/cifar.html), CIFAR10+, 
[CIFAR100](http://www.cs.toronto.edu/~kriz/cifar.html), CIFAR100+ and 
[SVHN](http://ufldl.stanford.edu/housenumbers/) datasets.

## Files
- Tensorflow
  Base on the VGG
   
- PyTorch

  Compare the effects of different network structures on the CIFAR10 with data augmentation, 
  such as MobileNetV2, ResNet18, ResNeXt29, DenseNet121, PreActResNet18. 
  
- Features of MNIST viewed on 2D

## Third-party re-implementation

- PyTorch: [code](https://github.com/qinyuxiang1995/Pairwise_Guassian_Loss/tree/master/pytorch) by [qinyuxiang1995](https://github.com/qinyuxiang1995).
- TensorFlow: [code](https://github.com/qinyuxiang1995/Pairwise_Guassian_Loss/tree/master/tensorflow) by [qinyuxiang1995](https://github.com/qinyuxiang1995).

