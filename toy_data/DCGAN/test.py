from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.realpath(__file__ + ('/..' * 3)))
print(f'Running from package root directory {sys.path[0]}')

import argparse
from common.generators import Generator_toy
from common.utils import *
import matplotlib.pyplot as plt
import os
import torch.utils.data
import numpy as np
from torch.autograd import Variable
from scipy.stats import chi2

if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing GANs under max hyper volume training')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data .hdf')
	parser.add_argument('--n-samples', type=int, default=2500, metavar='N', help='number of samples to  (default: 10000)')
	parser.add_argument('--toy-dataset', choices=['8gaussians', '25gaussians'], default='8gaussians')
	parser.add_argument('--no-plots', action='store_true', default=False, help='Disables plot of train/test losses')
	args = parser.parse_args()

	if args.cp_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	generator = Generator_toy(512)

	ckpt = torch.load(args.cp_path, map_location=lambda storage, loc: storage)
	generator.load_state_dict(ckpt['model_state'])

	history = ckpt['history']

	print('Min FD:', np.min(history['FD']))
	print('Epoch with min FD:', np.argmin(history['FD']))

	if not args.no_plots:
		plot_learningcurves(history, 'gen_loss')
		plot_learningcurves(history, 'disc_loss')
		# plot_learningcurves(history, 'gen_loss_minibatch')
		# plot_learningcurves(history, 'disc_loss_minibatch')
		plot_learningcurves(history, 'FD')
	# plot_learningcurves(history, 'quality_samples')
	# plot_learningcurves(history, 'quality_modes')

	save_samples_toy_data_gen(generator=generator, cp_name=args.cp_path.split('/')[-1].split('.')[0], save_name=args.cp_path.split('/')[-2].split('.')[0], n_samples=args.n_samples, toy_dataset=args.toy_dataset)
