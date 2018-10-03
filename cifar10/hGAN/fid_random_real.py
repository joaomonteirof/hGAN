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
	parser.add_argument('--ntests', type=int, default=10, metavar='N', help='number of samples to generate (default: 4)')
	parser.add_argument('--nsamples', type=int, default=10000, metavar='Path', help='number of samples per replication')
	parser.add_argument('--fid-model-path', type=str, default=None, metavar='Path', help='Path to fid model')
	parser.add_argument('--data-stat-path', type=str, default='../test_data_statistics.p', metavar='Path', help='Path to file containing test data statistics')
	parser.add_argument('--data-path', type=str, default='../data/', metavar='Path', help='Path to data')
	parser.add_argument('--out-file', type=str, default='./boxplot_data.p', metavar='Path', help='file for dumping boxplot data')
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


	try:

		pfile = open(args.data_stat_path, 'rb')
		statistics = pickle.load(pfile)
		pfile.close()

		m, C = statistics['m'], statistics['C']

	except:

		transform = transforms.Compose([transforms.Resize((64, 64), interpolation=Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		testset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)
		test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=args.workers)

		save_testdata_statistics(fid_model, test_loader, downsample_=True, cuda_mode=args.cuda)

		pfile = open(args.data_stat_path, 'rb')
		statistics = pickle.load(pfile)
		pfile.close()

		m, C = statistics['m'], statistics['C']
	
	'''
	# Random generator
	random_generator = Generator(100, [1024, 512, 256, 128], 3).eval()
	
	if args.cuda:
		random_generator = random_generator.cuda()
	
	fid_random = []
	for i in range(args.ntests):
		fid_random.append(compute_fid(random_generator, fid_model, args.batch_size, args.nsamples, m, C, args.cuda, inception = True if args.model_cifar == 'inception' else False, mnist = False))

	print(np.mean(fid_random))
   '''

	# Real data
	transform = transforms.Compose([transforms.Resize((64, 64), interpolation=Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	trainset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
	fid_real = compute_fid_real_data(train_loader, fid_model, m, C, args.cuda, inception = True if args.model_cifar == 'inception' else False, mnist = False)
	
	print(np.mean(fid_real))


	
