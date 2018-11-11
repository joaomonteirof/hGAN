from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.realpath(__file__ + ('/..' * 3)))
print(f'Running from package root directory {sys.path[0]}')

from common.discriminators import *
from common.generators import *
import torch
import torch.nn.functional as F
import torch.optim as optim
from common.flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number

generator = Generator(100, [1024, 512, 256, 128], 3)

optimizer = optim.Adam(generator.parameters(), lr=0.1, betas=(0.1, 0.1))

disc = Discriminator_noproj(3, [128, 256, 512, 1024], 1, optim.Adam, 0.1, (0.1, 0.1))

generator = add_flops_counting_methods(generator)
generator.train().start_flops_count()

disc = add_flops_counting_methods(disc)
disc.train().start_flops_count()
x = torch.randn(64, 3, 64, 64)
z_ = torch.randn(64, 100).view(-1, 100, 1, 1)
y_real_ = torch.ones(x.size(0))
y_fake_ = torch.zeros(x.size(0))

out_d = generator.forward(z_).detach()

d_real = disc.forward(x).squeeze()
d_fake = disc.forward(out_d).squeeze()
loss_disc = F.binary_cross_entropy(d_real, y_real_) + F.binary_cross_entropy(d_fake, y_fake_)
disc.optimizer.zero_grad()
loss_disc.backward()
disc.optimizer.step()

out = generator.forward(z_)

loss_G = F.binary_cross_entropy(disc.forward(out).squeeze(), y_real_)

loss_G.backward()
optimizer.step()

flops_G = generator.compute_average_flops_cost()
flops_D = disc.compute_average_flops_cost()

print('Flops G:  {}'.format(flops_G))
print('Flops D:  {}'.format(flops_D))
print('Total: {}'.format(flops_G+flops_D))
