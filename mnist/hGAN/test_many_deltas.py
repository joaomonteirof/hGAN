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

if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing GANs under max hyper volume training')
	parser.add_argument('--cp-folder', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--ntests', type=int, default=5, metavar='N', help='number of samples to generate (default: 4)')
	parser.add_argument('--nsamples', type=int, default=15000, metavar='Path', help='number of samples per replication')
	parser.add_argument('--no-plots', action='store_true', default=False, help='Disables plot of train/test losses')
	parser.add_argument('--fid-model-path', type=str, default=None, metavar='Path', help='Path to fid model')
	parser.add_argument('--data-stat-path', type=str, default='../test_data_statistics.p', metavar='Path', help='Path to file containing test data statistics')
	parser.add_argument('--model-mnist', choices=['cnn', 'mlp'], default='cnn', help='model for FID computation on Cifar. (Default=cnn)')
	parser.add_argument('--batch-size', type=int, default=512, metavar='Path', help='batch size')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.model_mnist == 'cnn':
		fid_model = cnn().eval()

	elif args.model_mnist == 'mlp':
		fid_model = mlp().eval()

	mod_state = torch.load(args.fid_model_path, map_location = lambda storage, loc: storage)
	fid_model.load_state_dict(mod_state['model_state'])

	if args.cuda:
		fid_model = fid_model.cuda()

	disc_list = [8, 16, 24]

	fid_dict = {}

	pfile = open(args.data_stat_path, 'rb')
	statistics = pickle.load(pfile)
	pfile.close()

	m, C = statistics['m'], statistics['C']

	for disc in disc_list:

		if args.cp_folder is None:
			raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

		cp_folder_disc = args.cp_folder + str(disc) + '/'

		print(cp_folder_disc)

		files_list = glob.glob(cp_folder_disc + 'G_hyper_*_50ep_*.pt')
		files_list.sort()

		fid = []

		for file_id in files_list:

			generator = Generator_mnist().eval()
			gen_state = torch.load(file_id, map_location=lambda storage, loc: storage)
			generator.load_state_dict(gen_state['model_state'])

			if args.cuda:
				generator = generator.cuda()
				fid_model = fid_model.cuda()

			for i in range(args.ntests):
				fid.append(compute_fid(generator, fid_model, args.batch_size, args.nsamples, m, C, args.cuda, inception = False, mnist = True))

		fid_dict[disc] = fid

	df = pd.DataFrame(fid_dict)
	df.head()
	box = sns.boxplot(data = df, palette = "Set3", width = 0.4, linewidth = 1.0, showfliers = False)
	box.set_xlabel('Number of discriminators', fontsize = 15)
	box.set_ylabel('FID', fontsize = 15)
	plt.grid(True, alpha = 0.3, linestyle = '--')	
	plt.savefig('FID_hyper_manyslacks.pdf')
	plt.show()
