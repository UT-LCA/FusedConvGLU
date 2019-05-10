#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single
import convtbcglu

manual_init = (0.10, 0.83, 0.23, 0.42, 0.80, 0.34, 0.53, 0.23, 0.43, 0.35,
               0.42, 0.30, 0.23, 0.45, 0.62, 0.45, 0.53, 0.28, 0.63, 0.24,
               0.22, 0.01, 0.42, 0.62, 0.45, 0.19, 0.08, 0.23, 0.84, 0.11,
               0.42, 0.30, 0.23, 0.01, 0.53, 0.32, 0.53, 0.23, 0.63, 0.24,
               0.22, 0.19, 0.08, 0.63, 0.45, 0.07, 0.84, 0.45, 0.07, 0.28,
               0.22, 0.01, 0.08, 0.23, 0.84, 0.42, 0.62, 0.45, 0.32, 0.11,
               0.42, 0.30, 0.42, 0.84, 0.42, 0.19, 0.11, 0.08, 0.42, 0.24,
               0.22, 0.19, 0.23, 0.23, 0.42, 0.01, 0.42, 0.42, 0.11, 0.28,
               0.28, 0.90, 0.34, 0.82, 0.07, 0.32, 0.34, 0.82, 0.32, 0.32,
               0.07, 0.32, 0.34, 0.82, 0.32, 0.32, 0.42, 0.01, 0.84, 0.45)

class ConvGLUTest(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, device):
        super(ConvGLUTest, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.padding = _single(padding)
        #self.conv = nn.Conv_TBC()

        self.weight = torch.nn.Parameter(torch.Tensor(
            self.kernel_size[0], in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

        cnt = 0
        for i in range(self.kernel_size[0]):
            for j in range(in_channels):
                for k in range(out_channels):
                    self.weight[i][j][k] = manual_init[cnt]
                    cnt = cnt + 1
        for i in range(out_channels):
            self.bias[i] = manual_init[cnt]
            cnt = cnt + 1
        #torch.nn.init.uniform(self.weight)
        #torch.nn.init.uniform(self.bias)
        #print("weight")
        #print(self.weight.size())
        #print(self.weight)
        #print("bias")
        #print(self.bias.size())
        #print(self.bias)
        cuda_device = torch.device(device)
        self.weight = torch.nn.Parameter(self.weight.cuda(cuda_device))
        self.bias = torch.nn.Parameter(self.bias.cuda(cuda_device))
        #print(self.weight)
        #print(self.bias)

    def forward(self, input):
        #x = torch.conv_tbc(input.contiguous(), self.weight,
        #                   self.bias, self.padding[0])

        #print(x.size())
        #print(x)

        #return F.glu(x, dim=0)
        # TODO: our target
        return convtbcglu.forward(input.contiguous(), self.weight,
                                  self.bias, self.padding[0])

def main(argv):

    parser = argparse.ArgumentParser(description="Test convglu layer")
    parser.add_argument("input", metavar="INPUT", help="The file having input")
    parser.add_argument("--in-channels", dest="in_channels", default=2,
                        type=int,
                        help="The number of input channels of the test layer")
    parser.add_argument("--out-channels", dest="out_channels", default=1,
                        type=int,
                        help="The number of output channels of the test layer")
    parser.add_argument("--kernel-sizes", dest="kernel_sizes", default=[3],
                        nargs="+", type=int,
                        help="The convolution filter sizes")
    parser.add_argument("--padding", dest="padding", default=[0],
                        nargs="+", type=int, help="Padding the input")
    parser.add_argument("--device", dest="device", default="cuda:0",
                        help="The cuda device to run the experiment")
    parser.add_argument("-o", dest="foname", default="test",
                        help="Iutput file base name")
    arg_list = parser.parse_args(argv[1:])

    batch_len = 0
    tensor2D = list()
    fi = open(arg_list.input)
    for line in fi.readlines():
        tensor1D = [float(element) for element in line.strip().split('\t')]
        if 0 == batch_len: # not set before
            batch_len = len(tensor1D)
        if (len(tensor1D) < batch_len): # need padding
            tensor1D = tensor1D + [0] * (batch_len - len(tensor1D))
        elif (len(tensor1D) > batch_len): # need slicing
            tensor1D = tensor1D[0:batch_len]
        tensor2D.append(tensor1D)
    fi.close()

    num_batch = len(tensor2D) // arg_list.in_channels
    tensor2D = np.array(tensor2D)
    tensor2D = tensor2D.transpose()
    tensor3D = tensor2D.reshape(batch_len, -1, arg_list.in_channels)
    #tensor3D = list()
    #for idx in range(arg_list.in_channels, len(tensor2D), arg_list.in_channels):
    #    tensor3D.append(tensor2D[:,idx-2:idx])
        
    
    #print("tensor3D in")
    #print(tensor3D)
    x = torch.tensor(tensor3D, dtype=torch.float)

    cuda_device = torch.device(arg_list.device)
    x = x.cuda(cuda_device) # return a copy on GPU memory

    # build a single layer to perform convolution and GLU
    single_layer = ConvGLUTest(arg_list.in_channels, arg_list.out_channels,
                               arg_list.kernel_sizes, arg_list.padding[0],
                               arg_list.device);

    y = single_layer.forward(x)
    y = y.cpu() # copy to CPU memory

    #print("tensor3D out")
    #print(y)
    assert(len(y.size()) == 3) # 3 dimentional tensor

    fo = open(arg_list.foname + ".tsv", 'w')
    for batch in range(y.size()[1]):
        for ch in range(y.size()[2]):
            for idx in range(y.size()[0]):
                fo.write("{0:.4f}".format(y[idx][batch][ch].item()))
                fo.write("\t")
            fo.write("\n")
    fo.close()

if "__main__" == __name__:
    main(sys.argv)
