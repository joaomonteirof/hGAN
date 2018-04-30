from __future__ import print_function

import argparse

import matplotlib.pyplot as plt
import model as model_
import torch.utils.data

from common.utils import save_samples


def plot_learningcurves(history, *keys):
	for key in keys:
		plt.plot(history[key])

	plt.show()


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

	generator = model_.Generator_toy(512)

	ckpt = torch.load(args.cp_path, map_location=lambda storage, loc: storage)
	generator.load_state_dict(ckpt['model_state'])

	history = ckpt['history']

	if not args.no_plots:
		plot_learningcurves(history, 'gen_loss')
		plot_learningcurves(history, 'disc_loss')
		plot_learningcurves(history, 'gen_loss_minibatch')
		plot_learningcurves(history, 'disc_loss_minibatch')
		plot_learningcurves(history, 'FD')

	save_samples(generator=generator, cp_name=args.cp_path.split('/')[-1].split('.')[0], save_name=args.cp_path.split('/')[-2].split('.')[0], n_samples=args.n_samples, toy_dataset=args.toy_dataset)
