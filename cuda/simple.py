#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single
import convtbcglu

def main(argv):
    convtbcglu.forward()

if "__main__" == __name__:
    main(sys.argv)
