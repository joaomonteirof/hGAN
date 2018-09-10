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
from matplotlib.pyplot import cm 

from pylab import rcParams
rcParams['figure.figsize'] = 15, 10

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
	parser.add_argument('--to-plot', type=str, default=None, metavar='Path', help='Thing to plot: FID-c or steepest_dir_norm')
	
	args = parser.parse_args()

	if args.cp_folder is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-folder to indicate the path!')

	folders = glob.glob(args.cp_folder + '*/')

	models_dict = {'hyper8': 'HV-8', 'hyper16': 'HV-16', 'hyper24': 'HV-24', 'vanilla8': 'AVG-8', 'vanilla16': 'AVG-16', 'vanilla24': 'AVG-24', 'gman8': 'GMAN-8', 'gman16': 'GMAN-16', 'gman24': 'GMAN-24'}
	markers_dict = {'HV': '-', 'GMAN': '--', 'AVG': '-.'}

	#color = iter([plt.get_cmap('nipy_spectral')(1. * i/6) for i in range(6)])

	color = iter(['darkgreen', 'mediumseagreen', 'darkorchid', 'magenta', 'midnightblue', 'deepskyblue'])


	#color = iter(cm.rainbow(np.linspace(0, 1, 6)))

	files_list = glob.glob(args.cp_folder + 'G_*.pt')
	files_list.sort()

	for file_id in files_list:

		file_name = file_id.split('/')[-1].split('_')[1]

		if (file_name != 'DCGAN') & (file_name != 'WGANGP'): 
			ckpt = torch.load(file_id, map_location = lambda storage, loc: storage)
			history = ckpt['history']
			to_plot = history[args.to_plot]
			n_disc = int(models_dict[file_name].split('-')[-1])
			label_ = models_dict[file_name].split('-')[0]
			print(n_disc)

			if (n_disc == 8) | (n_disc == 24):

				if (args.to_plot == 'FID-c'):
					x = range(len(to_plot))
					x_smooth1 = np.linspace(0, len(to_plot) - 1, 25)
					y_smooth1 = spline(x, to_plot, x_smooth1)
					x_smooth2 = np.linspace(0, len(to_plot) - 1, 1000)
					y_smooth2 = spline(x, to_plot, x_smooth2)
					c = next(color)
					plt.plot(x_smooth1, y_smooth1, color = c, label = models_dict[file_name], linestyle = markers_dict[label_])
					plt.plot(x_smooth2, y_smooth2, alpha = 0.2, color = c, linestyle = markers_dict[label_])
				else:
					steep_dir_ = [n_disc*i for i in to_plot]	
					x = range(len(steep_dir_))
					x_smooth1 = np.linspace(0, len(steep_dir_) - 1, 25)
					y_smooth1 = spline(x, steep_dir_, x_smooth1)
					x_smooth2 = np.linspace(0, len(steep_dir_) - 1, 1000)
					y_smooth2 = spline(x, steep_dir_, x_smooth2)
					c = next(color)
					plt.plot(x_smooth1, y_smooth1, color = c, label = models_dict[file_name], linestyle = markers_dict[label_])
					plt.plot(x_smooth2, y_smooth2, alpha = 0.2, color = c, linestyle = markers_dict[label_])

	plt.xlabel('Epochs', fontsize = 25)

	if (args.to_plot == 'FID-c'):
		plt.ylabel('FID - CIFAR-10', fontsize = 25)
		plt.ylim(0, 15)
	else:
		plt.ylabel('Update direction norm - CIFAR-10', fontsize = 25)
		plt.ylim(0, 15)
	
	plt.legend(fontsize = 19)
	plt.tick_params(labelsize = 25)
	plt.grid(True, alpha = 0.3, linestyle = '--')
	to_save = args.to_plot + 'best_cifar' + '.pdf'
	plt.savefig(to_save)
	plt.show()
	plt.close()

