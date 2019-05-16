#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single
from torch.autograd import Variable
import convtbcglu_cuda

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

class ConvGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, padding):
        outputs = convtbcglu_cuda.forward(input, weights, bias, padding)
        #x = torch.conv_tbc(input.contiguous(), weights, bias, padding)
        variables = (outputs, input, weights, bias)
        ctx.save_for_backward(*variables)

        return outputs

    @staticmethod
    def backward(ctx, doutput):
        doutput, input, weights, bias = ctx.saved_tensors
        outputs = convtbcglu_cuda.backward(
            doutput, input, weights, bias, 0)
        d_input, d_weights, d_bias = outputs
        return d_input, d_weights, d_bias, None

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
        x = torch.conv_tbc(input.contiguous(), self.weight,
                           self.bias, self.padding[0])

        #print(x.size())
        #print(x)

        #return F.glu(x, dim=0), x
        # TODO: our target
        return ConvGLUFunction.apply(input, self.weight,
                                     self.bias, self.padding[0]), x

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
    parser.add_argument("--padding", dest="padding", default=0,
                        type=int, help="Padding the input on Time dimension")
    parser.add_argument("--device", dest="device", default="cuda:0",
                        help="The cuda device to run the experiment")
    parser.add_argument("--loss", dest="loss", type=str,
                        help="The file having loss")
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
    x = Variable(x, requires_grad=True)

    # build a single layer to perform convolution and GLU
    single_layer = ConvGLUTest(arg_list.in_channels, arg_list.out_channels,
                               arg_list.kernel_sizes, arg_list.padding,
                               arg_list.device);

    y, convout = single_layer.forward(x)
    #y = y.cpu() # copy to CPU memory
    convout = Variable(convout, requires_grad=True)
    y2 = F.glu(convout, dim=0)

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

    if (None != arg_list.loss):
        fi = open(arg_list.loss)
        tensor2D = list()
        for line in fi.readlines():
            tensor1D = [float(element) for element in line.strip().split('\t')]
            if (len(tensor1D) < y.size(0)): # need padding
                tensor1D = tensor1D + [0] * (y.size(0) - len(tensor1D))
            elif (len(tensor1D) > y.size(0)): # need slicing
                tensor1D = tensor1D[0:y.size(0)]
            tensor2D.append(tensor1D)
        fi.close()
        tensor3D = np.array(tensor2D).reshape(y.size(0),
                                              arg_list.out_channels, -1)
        dy = torch.tensor(tensor3D, dtype=torch.float, device=cuda_device)
        dy.transpose_(1, 2) # make it TBC layout
        #print(dy)
        #y = y.cuda(cuda_device)
        y.backward(dy)
        #print(x.grad)
        #print(single_layer.weight.grad)
        #print(single_layer.bias.grad)
        dx = x.grad;
        dw = single_layer.weight.grad
        db = single_layer.bias.grad
        y2.backward(dy)
        #print(convout)
        #print(convout.grad)
        dconvout = convout.grad
        fo = open(arg_list.foname + "_grads.tsv", "w")
        fo.write("dx(T={},B={},C={})\n".format(dx.size(0),
                                               dx.size(1),dx.size(2)))
        for batch in range(dx.size(1)):
            for ch in range(dx.size(2)):
                for idx in range(dx.size(0)):
                    fo.write("{0:.4f}\t".format(dx[idx][batch][ch].item()))
                fo.write("\n")
        fo.write("dw(K={},I={},O={})\n".format(dw.size(0),
                                               dw.size(1),dw.size(2)))
        for out_ch in range(dw.size(2)):
            for in_ch in range(dw.size(1)):
                for k in range(dw.size(0)):
                    fo.write("{0:.4f}\t".format(dw[k][in_ch][out_ch].item()))
                fo.write("\n")
        fo.write("db(O={})\n".format(db.size(0)))
        for out_ch in range(db.size(0)):
            fo.write("{0:.4f}\n".format(db[out_ch].item()))
        fo.write("dconvout(T={},B={},C={})\n".format(dconvout.size(0),
                                                     dconvout.size(1),
                                                     dconvout.size(2)))
        for bat in range(dconvout.size(1)):
            for ch in range(dconvout.size(2)):
                for idx in range(dconvout.size(0)):
                    fo.write("{0:.4f}\t".format(dconvout[idx][bat][ch].item()))
                fo.write("\n")
        fo.close()

if "__main__" == __name__:
    main(sys.argv)
