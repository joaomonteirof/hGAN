from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.realpath(__file__ + ('/..' * 2)))
print(f'Running from package root directory {sys.path[0]}')

from common.generators import Generator
from common.discriminators import *
import torch
import torch.optim as optim

generator = Generator(128, [1024, 512, 256, 128, 64, 32], 3).train()

disc = Discriminator(3, [32, 64, 128, 256, 512, 1024], 1, optim.Adam, 0.1, (0.1, 0.1)).train()

z = torch.rand(10, 128, 1, 1)

im = generator(z)
print(im.size())
d=disc(im)
print(d.size())
