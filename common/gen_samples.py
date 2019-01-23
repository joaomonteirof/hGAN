from __future__ import print_function

import os
import sys
import numpy as np

from generators import Generator
import argparse
import matplotlib.pyplot as plt
import torch.utils.data
from utils import save_samples_no_grid, denorm

if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing GANs under max hyper volume training')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--nsamples', type=int, default=10000, metavar='N', help='number of samples to generate (default: 10000)')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path to output samples')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.cp_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	model = Generator(100, [1024, 512, 256, 128], 3)

	ckpt = torch.load(args.cp_path, map_location=lambda storage, loc: storage)
	model.load_state_dict(ckpt['model_state'])

	if args.cuda:
		model = model.cuda()

	print('Cuda Mode is: {}'.format(args.cuda))

	save_samples_no_grid(model, args.nsamples, args.cuda, args.out_path)
