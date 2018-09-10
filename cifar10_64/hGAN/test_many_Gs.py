from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.realpath(__file__ + ('/..' * 3)))
print(f'Running from package root directory {sys.path[0]}')

import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from scipy.stats import sem

if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing GANs under max hyper volume training')
	parser.add_argument('--models-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	args = parser.parse_args()

	if args.models_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --models-path to indicate the path!')

	models_list = glob.glob(args.models_path + 'G*.pt')

	sdd = []
	fid = []

	for model_ in models_list:
		ckpt = torch.load(model_, map_location=lambda storage, loc: storage)
		history = ckpt['history']
		sdd.append(history['steepest_dir_norm'])
		fid.append(history['FID-c'])

	sdd = np.asarray(sdd)
	fid = np.asarray(fid)

	min_sdd = sdd.min(1)
	min_fid = fid.min(1)

	print('min sdd: {:0.4f} +- {:0.4f}'.format(min_sdd.min(), min_sdd.std()))
	print('min fid: {:0.4f} +- {:0.4f}'.format(min_fid.min(), min_fid.std()))

	plt.figure(1)
	plt.errorbar(np.arange(sdd.shape[1]), sdd.mean(0), yerr=sem(sdd, axis=0))
	plt.title('Steepest descent direction norm')

	plt.figure(2)
	plt.errorbar(np.arange(fid.shape[1]), fid.mean(0), yerr=sem(fid, axis=0))
	plt.title('Frechet inception distance')

	plt.show()
