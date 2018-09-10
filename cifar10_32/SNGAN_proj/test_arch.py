from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.realpath(__file__ + ('/..' * 3)))
print(f'Running from package root directory {sys.path[0]}')

from common.discriminators import Discriminator_SN
from common.generators import Generator_SN
import torch
import torch.optim as optim

generator = Generator_SN().train()

disc = Discriminator_SN(optim.Adam, 0.1, (0.1, 0.1)).train()


z_ = torch.randn(10, 128)

im_ = torch.randn(10, 3, 32, 32)

out_g = generator(z_)
out_d = disc(im_)
