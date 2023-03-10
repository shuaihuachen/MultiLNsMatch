import torch.nn as nn

def init_conv_weights(layer, weights_std=0.01,  bias=0):
    '''
    RetinaNet's layer initialization

    :layer
    :

    '''
    nn.init.xavier_normal(layer.weight)
    nn.init.constant(layer.bias.data, val=bias)
    return layer

def conv1x1x1(in_channels, out_channels, **kwargs):
    '''Return a 1x1 convolutional layer with RetinaNet's weight and bias initialization'''

    layer = nn.Conv3d(in_channels, out_channels, kernel_size=1, **kwargs)
    layer = init_conv_weights(layer)

    return layer


def conv3x3x3(in_channels, out_channels, **kwargs):
    '''Return a 3x3 convolutional layer with RetinaNet's weight and bias initialization'''

    layer = nn.Conv3d(in_channels, out_channels, kernel_size=3, **kwargs)
    layer = init_conv_weights(layer)

    return layer