from __future__ import print_function

import argparse

import os
import sys
import glob

sys.path.insert(0, os.path.realpath(__file__ + ('/..' * 3)))
print(f'Running from package root directory {sys.path[0]}')

import matplotlib.pyplot as plt
import torch.utils.data

import pandas as pd
import seaborn as sns

from common.generators import Generator_mnist
from common.utils import *
from common.models_fid import *
from common.metrics import compute_fid
from scipy.interpolate import spline

def plot_FID_alldisc(fid1, fid2, fid3, hyper):		

	plt.plot(fid1, 'blue', label = '8 discriminators')
	plt.plot(fid2, 'green', label = '16 discriminators')
	plt.plot(fid3, 'fuchsia', label = '24 discriminators')

	plt.xlabel('Epochs')
	plt.ylabel('FID')
	plt.ylim((0, 30))


	plt.legend()
	
	save_fn = 'Cifar10_alldisc_'+ save_str + '.png'
	plt.savefig(save_fn)

	plt.show()

	plt.close()

if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing GANs under max hyper volume training')
	parser.add_argument('--cp-folder', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	
	args = parser.parse_args()

	if args.cp_folder is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-folder to indicate the path!')

	folders = glob.glob(args.cp_folder + '*/')

	labels_dict = {'hyper': 'HV', 'gman': 'GMAN', 'vanilla': 'AVG', 'mgd': 'MGD'}
	markers_dict = {'hyper': ':', 'gman': '--', 'vanilla': '-.', 'mgd': '-'}
	color = iter([plt.get_cmap('nipy_spectral')(1. * i/6) for i in range(6)])

	for dir_ in folders:

		file_ = glob.glob(dir_ + '/G_*_8_100ep.pt')	

		if file_ != []:
			ckpt = torch.load(file_[0], map_location = lambda storage, loc: storage)
			history = ckpt['history']
			steep_dir = history['steepest_dir_norm']
			label_ = file_[0].split('/')[-1].split('_')[1]
			steep_dir_ = [i * 8 for i in steep_dir]
			x = range(len(steep_dir_))
			x_smooth1 = np.linspace(0, len(steep_dir_)-1, 25)
			y_smooth1 = spline(x, steep_dir_, x_smooth1)
			x_smooth2 = np.linspace(0, len(steep_dir_)-1, 1000)
			y_smooth2 = spline(x, steep_dir_, x_smooth2)
			c = next(color)
			plt.plot(x_smooth1, y_smooth1, color = c, label = labels_dict[label_], linestyle = markers_dict[label_])
			plt.plot(x_smooth2, y_smooth2, alpha = 0.2, color = c, linestyle = markers_dict[label_])


			#plt.plot(steep_dir_, label = labels_dict[label_], linestyle = markers_dict[label_])


	plt.xlabel('Epochs', fontsize = 15)
	plt.ylabel('Update direction norm - MNIST', fontsize = 15)
	plt.tick_params(labelsize = 15)
	plt.ylim(0, 2.4)
	plt.legend()
	plt.grid(True, alpha = 0.3, linestyle = '--')
	plt.savefig('steep_mnist.pdf')
	plt.show()
	plt.close()

