from __future__ import print_function

import argparse

import os
import sys

sys.path.insert(0, os.path.realpath(__file__ + ('/..' * 3)))
print(f'Running from package root directory {sys.path[0]}')

import matplotlib.pyplot as plt
import torch.utils.data

import pandas as pd
import seaborn as sns
import pickle

from common.generators import Generator
from common.utils import *
from common.models_fid import *
from common.metrics import compute_fid, compute_fid_real_data
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import PIL.Image as Image

from pylab import rcParams
rcParams['figure.figsize'] = 15, 10

if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing GANs under max hyper volume training')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--ntests', type=int, default=15, metavar='N', help='number of samples to generate (default: 4)')
	parser.add_argument('--nsamples', type=int, default=10000, metavar='Path', help='number of samples per replication')
	parser.add_argument('--fid-model-path', type=str, default=None, metavar='Path', help='Path to fid model')
	parser.add_argument('--data-stat-path', type=str, default='../test_data_statistics.p', metavar='Path', help='Path to file containing test data statistics')
	parser.add_argument('--data-path', type=str, default='../data/', metavar='Path', help='Path to data')
	parser.add_argument('--out-file', type=str, default='./boxplot_data_new.p', metavar='Path', help='file for dumping boxplot data')
	parser.add_argument('--in-file', type=str, default='./boxplot_data.p', metavar='Path', help='file for dumping boxplot data')
	parser.add_argument('--sub-key', type=str, default=None, metavar='Path', help='key for substitution')
	parser.add_argument('--model-cifar', choices=['resnet', 'vgg', 'inception'], default='resnet', help='model for FID computation on Cifar. (Default=Resnet)')
	parser.add_argument('--batch-size', type=int, default=512, metavar='Path', help='batch size')
	parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.model_cifar == 'resnet':
		fid_model = ResNet18().eval()
	elif args.model_cifar == 'vgg':
		fid_model = VGG().eval()
	elif args.model_cifar == 'inception':
		fid_model = inception_v3(pretrained=True, transform_input=False).eval()

	if args.cuda:
		fid_model = fid_model.cuda()

	mod_state = torch.load(args.fid_model_path, map_location = lambda storage, loc: storage)
	fid_model.load_state_dict(mod_state['model_state'])

	models_dict = {'hyper8': 'HV-8', 'hyper16': 'HV-16', 'hyper24': 'HV-24', 'vanilla8': 'AVG-8', 'vanilla16': 'AVG-16', 'vanilla24': 'AVG-24', 'gman8': 'GMAN-8', 'gman16': 'GMAN-16', 'gman24': 'GMAN-24', 'DCGAN': 'DCGAN', 'WGANGP': 'WGAN-GP'}

	pfile = open(args.data_stat_path, 'rb')
	statistics = pickle.load(pfile)
	pfile.close()

	pfile = open(args.in_file, 'rb')
	fid_dict = pickle.load(pfile)
	pfile.close()

	
	m, C = statistics['m'], statistics['C']

	if args.cp_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	if args.sub_key is None:
		raise ValueError('There is no key to substitute. Use arg --sub-key to indicate the key!')

	print(args.sub_key, args.cp_path)		

	generator = Generator(100, [1024, 512, 256, 128], 3).eval()
	gen_state = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	generator.load_state_dict(gen_state['model_state'])

	if args.cuda:
		generator = generator.cuda()
	
	fid = []

	for i in range(args.ntests):
		fid.append(compute_fid(generator, fid_model, args.batch_size, args.nsamples, m, C, args.cuda, inception = True if args.model_cifar == 'inception' else False, mnist = False))

	fid_dict[args.sub_key] = fid
	

	df = pd.DataFrame(fid_dict)
	df.head()
	order_plot = ['DCGAN', 'WGAN-GP', 'AVG-8', 'GMAN-8', 'HV-8', 'AVG-16', 'GMAN-16', 'HV-16', 'AVG-24', 'GMAN-24', 'HV-24']
	box = sns.boxplot(data = df, palette = "Set3", width = 0.2, linewidth = 1.0, showfliers = False, order = order_plot)
	box.set_xlabel('Model', fontsize = 20)
	box.set_ylabel('FID - CIFAR-10', fontsize = 20)	
	box.set_yscale('log')
	box.tick_params(labelsize = 15)
	plt.grid(True, alpha = 0.3, linestyle = '--')
	#plt.axhline(fid_dict['random'], color='r', linestyle = 'dashed', linewidth = 1)
	#plt.axhline(fid_dict['real'], color='b', linestyle='dashed', linewidth=1)
	plt.axhline(89.24368468, color='r', linestyle = 'dashed', linewidth = 1)
	#plt.axhline(0.035210, color='b', linestyle='dashed', linewidth=1)
	plt.axvline(1.5, color = 'grey', alpha = 0.5, linestyle = 'dashed', linewidth = 1)
	plt.axvline(4.5, color = 'grey', alpha = 0.5, linestyle = 'dashed', linewidth = 1)
	plt.axvline(7.5, color = 'grey', alpha = 0.5, linestyle = 'dashed', linewidth = 1)
	plt.savefig('FID_best_models.pdf')
	plt.show()
	
	pfile = open(args.out_file, "wb")
	pickle.dump(fid_dict, pfile)
	pfile.close()
	
