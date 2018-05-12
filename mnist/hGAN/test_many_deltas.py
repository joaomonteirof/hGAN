from __future__ import print_function

import argparse

import os
import sys
import glob

sys.path.insert(0, os.path.realpath(__file__ + ('/..' * 3)))
print(f'Running from package root directory {sys.path[0]}')

import matplotlib.pyplot as plt
import torch.utils.data

from common.generators import Generator_mnist
from common.utils import *

if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing GANs under max hyper volume training')
	parser.add_argument('--cp-folder', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--n-tests', type=int, default=4, metavar='N', help='number of samples to generate (default: 4)')
	parser.add_argument('--no-plots', action='store_true', default=False, help='Disables plot of train/test losses')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.cp_folder is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	files_list = glob.glob(args.cp_folder + 'G_hyper_*_50ep_*.pt')
	files_list.sort()

	best_fid = []
	best_epoch = []

	for file_id in files_list:

		model = Generator_mnist()

		ckpt = torch.load(file_id, map_location=lambda storage, loc: storage)
		model.load_state_dict(ckpt['model_state'])

		if args.cuda:
			model = model.cuda()

		history = ckpt['history']

		min_fid = np.min(history['FID-c'])
		min_epoch = np.argmin(history['FID-c'])

		print('Min FID:', min_fid)
		print('Epoch with min FID:', min_epoch)

		best_fid.append(min_fid)
		best_epoch.append(min_epoch)

		test_model(model=model, n_tests=args.n_tests, cuda_mode=args.cuda)
		save_samples(generator=model, cp_name=file_id.split('/')[-1].split('.')[0], prefix='mnist', fig_size=(10, 10), nc=1, im_size=28, cuda_mode=args.cuda)

	fid_mean = np.mean(best_fid)
	fid_std = np.std(best_fid)

	epoch_mean = np.mean(best_epoch)
	epoch_std = np.std(best_epoch)

	print('Average Min fid:', fid_mean)
	print('STD Min fid:', fid_std)

	print('Average Min fid epoch:', epoch_mean)
	print('STD Min fid epoch:', epoch_std)
