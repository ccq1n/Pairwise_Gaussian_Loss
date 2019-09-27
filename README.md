# Pairwise Gaussian Loss

We introduce a pairwise gaussian loss (PGL) for convolutional neural networks. PGL can greatly improve the generalization ability of CNNs, so it is very suitable for general classification, feature embedding and biometrics verification. We give the 2D feature visualization on MNIST to illustrate our PGL.

<img src="image/Softmax_vs_Gloss.png" width="50%" height="50%">

## Test

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10+ dataset.

## Model
- VGG for CIFAR10
- [ResNet18](https://arxiv.org/abs/1512.03385) 
- [MobileNetV2](https://arxiv.org/abs/1801.04381) 
- [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431) 
- [DenseNet121](https://arxiv.org/abs/1608.06993) 
- [PreActResNet18](https://arxiv.org/abs/1603.05027) 

## Learning rate adjustment
I manually change the `lr` during training:
- `0.1` for epoch `[0,150)`
- `0.01` for epoch `[150,250)`
- `0.001` for epoch `[250,350)`

Resume the training with `python main.py`
