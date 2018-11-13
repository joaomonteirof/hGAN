from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.realpath(__file__ + ('/..' * 2)))
print(f'Running from package root directory {sys.path[0]}')

import argparse

from common.generators import Generator
import matplotlib.pyplot as plt
import torch.utils.data

from common.utils import test_model, save_samples


def denorm(unorm):
	norm = (unorm + 1) / 2
	return norm.clamp(0, 1)


def plot_learningcurves(history, *keys):
	for key in keys:
		plt.plot(history[key])

	plt.show()


if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing GANs under max hyper volume training')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--n-tests', type=int, default=4, metavar='N', help='number of samples to  (default: 64)')
	parser.add_argument('--no-plots', action='store_true', default=False, help='Disables plot of train/test losses')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.cp_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	model = Generator(128, [1024, 512, 256, 128, 64, 32], 3)

	ckpt = torch.load(args.cp_path, map_location=lambda storage, loc: storage)
	model.load_state_dict(ckpt['model_state'])

	if args.cuda:
		model = model.cuda()

	print('Cuda Mode is: {}'.format(args.cuda))

	history = ckpt['history']

	if not args.no_plots:
		plot_learningcurves(history, 'gen_loss')
		plot_learningcurves(history, 'disc_loss')
		plot_learningcurves(history, 'gen_loss_minibatch')
		plot_learningcurves(history, 'disc_loss_minibatch')

	test_model(model=model, n_tests=args.n_tests, cuda_mode=args.cuda, SNGAN=True)
	save_samples(prefix='cats', generator=model, cp_name=args.cp_path.split('/')[-1].split('.')[0], cuda_mode=args.cuda, enhance = False, im_size = 256, fig_size=(4, 8), SNGAN=True)
