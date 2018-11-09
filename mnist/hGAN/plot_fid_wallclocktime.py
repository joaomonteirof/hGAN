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


	labels_dict = {'hyper': 'HV', 'gman': 'GMAN', 'vanilla': 'AVG', 'mgd': 'MGD'}
	markers_dict = {'hyper': '-', 'gman': '-', 'vanilla': '-', 'mgd': '-'}
	color = iter([plt.get_cmap('nipy_spectral')(1. * i/6) for i in range(4)])


	file_ = glob.glob(args.cp_folder +'G_*_8_100ep*.pt')

	print(file_)

	for fil in file_:
		ckpt = torch.load(fil, map_location = lambda storage, loc: storage)
		history = ckpt['history']
		fid = history['FID-c']
		label_ = fil.split('/')[-1].split('_')[1]

		print(label_)

		x_gman = np.linspace(0, 320.95, 100)
		x_avg = np.linspace(0, 320.95, 100)
		x_hyper = np.linspace(0, 320.95, 100)
		x_mgd = np.linspace(0, 1995, 100)

		curr_min = 1000000000000
		min_fid = []
		for f in fid:

			if f < curr_min: 
				curr_min = f

			min_fid.append(curr_min) 

		c = next(color)
		if label_ == 'gman':
			plt.subplot(412)
			plt.plot(x_gman, min_fid, color = c, label = labels_dict[label_], linestyle = markers_dict[label_])
			plt.scatter(x_gman[15], min_fid[15])
			#plt.xlabel('Wallclock time', fontsize = 15)
			plt.ylabel('Min. FID', fontsize = 10)
			plt.ylim(0, 50)
			plt.xlim(0, 330)
			plt.legend()
			plt.grid(True, alpha = 0.3, linestyle = '--')

		elif label_ == 'vanilla':
			plt.subplot(411)
			plt.plot(x_avg, min_fid, color = c, label = labels_dict[label_], linestyle = markers_dict[label_])
			plt.scatter(x_avg[19], min_fid[19])
			#plt.xlabel('Wallclock time', fontsize = 15)
			plt.ylabel('Min. FID', fontsize = 10)
			plt.ylim(0, 50)
			plt.xlim(0, 330)
			plt.legend()
			plt.grid(True, alpha = 0.3, linestyle = '--')

		elif label_ == 'hyper':
			plt.subplot(413)
			plt.plot(x_hyper, min_fid, color = c, label = labels_dict[label_], linestyle = markers_dict[label_])
			plt.scatter(x_hyper[80], min_fid[80])
			#plt.xlabel('Wallclock time', fontsize = 15)
			plt.ylabel('Min. FID', fontsize = 10)
			plt.ylim(0, 50)
			plt.xlim(0, 330)
			plt.legend()
			plt.grid(True, alpha = 0.3, linestyle = '--')
	
		elif label_ == 'mgd':
			plt.subplot(414)
			plt.plot(x_mgd, min_fid, color = c, label = labels_dict[label_], linestyle = markers_dict[label_])
			plt.scatter(x_mgd[93], min_fid[93])
			plt.xlabel('Wall-clock time (minutes)', fontsize = 10)
			plt.ylabel('Min. FID', fontsize = 10)
			plt.ylim(0, 50)
			plt.xlim(0, 2100)
			plt.legend()
			plt.grid(True, alpha = 0.3, linestyle = '--')
		

	
	plt.subplots_adjust(hspace=0.5)


	#plt.tick_params(labelsize = 15)
	#plt.ylim(0, 100)
	#plt.legend()
	#plt.grid(True, alpha = 0.3, linestyle = '--')
	plt.savefig('fid_wallclock.pdf')
	plt.show()
	plt.close()

