from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.realpath(__file__ + ('/..' * 2)))
print(f'Running from package root directory {sys.path[0]}')

import argparse

from common.generators import Generator
import matplotlib.pyplot as plt
import glob
import torch

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
	parser.add_argument('--models-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	args = parser.parse_args()

	if args.models_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --models-path to indicate the path!')

	model = Generator(100, [1024, 512, 256, 128], 3)

	files_list = glob.glob(args.models_path + 'G_*.pt')
	files_list.sort()

	for file_ in files_list:

		ckpt = torch.load(file_, map_location=lambda storage, loc: storage)
		model.load_state_dict(ckpt['model_state'])

		save_samples(prefix='CELEBA_hGAN_VaryingD', generator=model, cp_name=file_.split('/')[-1].split('.')[0], cuda_mode=False, fig_size=(2,14))
