import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import math

class ConvolutionalBlock(nn.Module):
    '''
    Convolutional block: Convolution, BatchNorm, Activation
    credits: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution/blob/master/models.py
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        '''
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size
        :param stride: stride
        :param batch_norm: include a BN layer?
        :param activation: Type of activation; None if none
        '''
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}

        # container, that will hold the layers in this convolutional block
        layers = list()
        # convolutional layer
        layers.append(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                    stride=stride, padding=kernel_size // 2)
                )
        # batch normalization, if wanted
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        # activation layer, if wanted
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        # put together the convolutional block as a sequence of the layers
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        '''
        Forward propagation

        :param input: input images, a tensor of size (N, in_channels, w, h)
        :return: output images, a tensor of size (N, out_channels, w, h)
        '''
        output = self.conv_block(input) #(N, out_channels, w, h)
        return output

class SubPixelConvolutionalBlock(nn.Module):
    """
    A subpixel convolutional block, comprising convolutional, pixel-shuffle, and PReLU activation layers.
    """
    def __init__(self, args, scaling_factor=2):
        """
        :param kernel_size: kernel size of the convolution
        :param n_channels: number of input and output channels
        :param scaling_factor: factor to scale input images by (along both dimensions)
        """
        super(SubPixelConvolutionalBlock, self).__init__()
        # convolutional layer that increases the number of channels by scaling factor^2, followed by pixel shuffle and PReLU
        self.conv = nn.Conv2d(in_channels=args.n_channels, out_channels=args.n_channels * (scaling_factor ** 2),
                            kernel_size=args.small_kernel_size, padding=args.small_kernel_size // 2)
        # These additional channels are shuffled to form additional pixels, upscaling each dimension by the scaling factor
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.prelu = nn.PReLU()

    def forward(self, input_):
        """
        Forward propagation.
        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: scaled output images, a tensor of size (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.conv(input_)  # (N, n_channels * scaling factor^2, w, h)
        output = self.pixel_shuffle(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.prelu(output)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return output


class ResidualBlock(nn.Module):
    """
    A residual block, comprising two convolutional blocks with a residual connection across them.
    """
    def __init__(self, args):
        """
        :param kernel_size: kernel size
        :param n_channels: number of input and output channels (same because the input must be added to the output)
        """
        super(ResidualBlock, self).__init__()

        # first convolutional block
        self.conv_block1 = ConvolutionalBlock(in_channels=args.n_channels, out_channels=args.n_channels, 
                kernel_size=args.small_kernel_size, batch_norm=True, activation='PReLu')

        # second convolutional block
        self.conv_block2 = ConvolutionalBlock(in_channels=args.n_channels, out_channels=args.n_channels,
                kernel_size=args.small_kernel_size, batch_norm=True, activation=None)

    def forward(self, input_):
        """
        Forward propagation
        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: output images, a tensor of size (N, n_channels, w, h)
        """
        residual = input_ # (N, n_channels, w, h)
        output = self.conv_block1(input_) # (N, n_channels, w, h)
        output = self.conv_block2(output) # (N, n_channels, w, h)
        output = output + residual # (N, n_channels, w, h)

        return output
    

class SRResNet(nn.Module):
    '''
    The SRResNet, as defined in the paper.
    credits: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution/blob/master/models.py
    '''
    def __init__(self, args):
        """
        :param large_kernel_size: kernel size of the first and last convolutions which transform the inputs and outputs
        :param small_kernel_size: kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
        :param n_channels: number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
        :param n_blocks: number of residual blocks
        :param scaling_factor: factor to scale input images by (along both dimensions) in the subpixel convolutional block
        """
        self.args = args
        super(SRResNet, self).__init__()

        # Scaling factor must be 2, 4 or 8
        scaling_factor = int(args.scaling_factor)
        assert scaling_factor in {2, 4, 8}, "The scaling factor must be 2, 4, or 8!"

        # First convolutional block
        self.conv_block1 = ConvolutionalBlock(in_channels=args.input_channels, out_channels=args.n_channels, 
                kernel_size=args.large_kernel_size,
                batch_norm=False, activation='PReLu')

        # Sequence of residual blocks
        self.residual_blocks = nn.Sequential(
                *[ResidualBlock(args) for i in range(args.n_blocks)]
                )

        # Another convolutional block
        self.conv_block2 = ConvolutionalBlock(in_channels=args.n_channels, out_channels=args.n_channels,
                kernel_size=args.small_kernel_size, batch_norm=True, activation=None)

        # Upscaling: by sub-pixel convolution, each such block upscaling by a factor of 2
        n_subpixel_convolutional_blocks = int(math.log2(args.scaling_factor))
        print(f'times subpixel: {n_subpixel_convolutional_blocks}')
        self.subpixel_convolutional_blocks = nn.Sequential(
                *[SubPixelConvolutionalBlock(args, scaling_factor=2) for i in range(n_subpixel_convolutional_blocks)]
                )

        # Last convolutional block
        self.conv_block3 = ConvolutionalBlock(in_channels=args.n_channels, out_channels=1,
                kernel_size=args.large_kernel_size, batch_norm=False, activation='Tanh')

        # Final sigmoid layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, lr_imgs):
        """
        Forward propagation

        :param lr_imgs: low-resolution input images, a tensor of size (N, 3, w, h)
        :return: super-resolution output images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        """
        output = self.conv_block1(lr_imgs) # (N, input_channels, w, h)
        residual = output # (N, n_channels, w, h)
        output = self.residual_blocks(output) # (N, n_channels, w, h)
        output =self.conv_block2(output) # (N, n_channels, w, h)
        output = output + residual
        output = self.subpixel_convolutional_blocks(output) # (N, n_channels, w * scaling factor, h * scaling factor)
        sr_imgs = self.conv_block3(output) # (N, 1, w * scaling factor, h * scaling factor)
        sr_imgs = self.sigmoid(sr_imgs)
        #sr_imgs = sr_imgs.round() # reduce to image containing only 1 and 0
        return sr_imgs

    def get_device(self):
        """Return gpu if available, else cpu"""
        #if torch.cuda.is_available():
        #    print('GPU available')
        #    return 'cuda:0'
        #else:
        #    print('running on CPU')
        #    return 'cpu'
        return 'cpu'

