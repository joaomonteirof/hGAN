from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.realpath(__file__ + ('/..' * 3)))
print(f'Running from package root directory {sys.path[0]}')

import argparse
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from common.generators import Generator_toy
import numpy as np

from common.utils import *
from common.toy_data import ToyData

if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing GANs under max hyper volume training')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--toy-length', type=int, default=2500, metavar='N', help='number of samples to  (default: 10000)')
	parser.add_argument('--toy-dataset', choices=['8gaussians', '25gaussians'], default='8gaussians')
	args = parser.parse_args()

	if args.cp_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	generator = Generator_toy(512)

	toy_data = ToyData(args.toy_dataset, args.toy_length)
	centers = toy_data.get_centers()
	cov = toy_data.get_cov()

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	generator.load_state_dict(ckpt['model_state'])

	#fixed_noise = Variable(ckpt['fixed_noise'])

	fixed_noise = Variable(torch.randn(2500, 2).view(-1, 2))

	generator.eval()

	x = generator.forward(fixed_noise)

	fd_, q_samples, cov_modes = metrics_toy_data(x.data.numpy(), centers, cov, args.toy_dataset)

	print('Epoch:', args.cp_path.split('checkpoint_')[-1].split('ep')[0])
	print('FD:', fd_)
	print('High quality samples:', (q_samples/x.size(0))*100)
	print('Covered modes:', cov_modes)

	save_name = args.cp_path[-3]+'.png'
	save_samples_toy_data(x, centers, save_name, toy_dataset = args.toy_dataset)
