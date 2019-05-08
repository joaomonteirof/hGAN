from __future__ import print_function
import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.utils.data
from scipy.stats import sem
import scipy.linalg as sla
from generators import *
from models_fid import *
from metrics import compute_fid
from utils import save_testdata_statistics
from inception import InceptionV3
import torchvision.transforms as transforms
import PIL.Image as Image
import torchvision.datasets as datasets

import pickle

if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='FID computation')
	parser.add_argument('--model-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--data-stat-path', type=str, default='../test_data_statistics.p', metavar='Path', help='Path to file containing test data statistics')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data if data statistics are not provided')
	parser.add_argument('--fid-model-path', type=str, default=None, metavar='Path', help='Path to fid model')
	parser.add_argument('--model-cifar', choices=['resnet', 'vgg', 'inception'], default='resnet', help='model for FID computation on Cifar. (Default=Resnet)')
	parser.add_argument('--model-mnist', choices=['cnn', 'mlp'], default='resnet', help='model for FID computation on Cifar. (Default=cnn)')
	parser.add_argument('--batch-size', type=int, default=512, metavar='Path', help='batch size')
	parser.add_argument('--nsamples', type=int, default=10000, metavar='Path', help='number of samples per replication')
	parser.add_argument('--ntests', type=int, default=3, metavar='Path', help='number of replications')
	parser.add_argument('--dataset', choices=['cifar10', 'mnist', 'celeba'], default='cifar10', help='cifar10, mnist, or celeba')
	parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--sngan', action='store_true', default=False, help='Enables computing FID for SNGAN')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False
	args.bsize = min(args.batch_size, args.nsamples)

	if args.model_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --model-path to indicate the path!')

	if args.dataset == 'cifar10':
		if args.model_cifar=='resnet':
			fid_model = ResNet18().eval()
			mod_state = torch.load(args.fid_model_path, map_location=lambda storage, loc: storage)
			fid_model.load_state_dict(mod_state['model_state'])
		elif args.model_cifar=='vgg':
			fid_model = VGG().eval()
			mod_state = torch.load(args.fid_model_path, map_location=lambda storage, loc: storage)
			fid_model.load_state_dict(mod_state['model_state'])
		elif args.model_cifar=='inception':
			fid_model = InceptionV3([3])

		if args.sngan:
			generator = Generator_SN()
		else:
			generator = Generator(100, [1024, 512, 256, 128], 3).eval()

		gen_state = torch.load(args.model_path, map_location=lambda storage, loc: storage)
		generator.load_state_dict(gen_state['model_state'])

	elif args.dataset == 'mnist':
		if args.model_mnist=='cnn':
			fid_model = cnn().eval()
		elif args.model_mnist=='mlp':
			fid_model = mlp().eval()

		mod_state = torch.load(args.fid_model_path, map_location=lambda storage, loc: storage)
		fid_model.load_state_dict(mod_state['model_state'])

		generator = Generator_mnist().eval()
		gen_state = torch.load(args.model_path, map_location=lambda storage, loc: storage)
		generator.load_state_dict(gen_state['model_state'])

	elif args.dataset == 'celeba':
			fid_model = InceptionV3([3])

			generator = Generator(100, [1024, 512, 256, 128], 3).eval()

			gen_state = torch.load(args.model_path, map_location=lambda storage, loc: storage)
			generator.load_state_dict(gen_state['model_state'])


	try:

		pfile = open(args.data_stat_path, 'rb')
		statistics = pickle.load(pfile)
		pfile.close()

		m, C = statistics['m'], statistics['C']

	except:

		if args.dataset == 'cifar10':
			transform = transforms.Compose([transforms.Resize((64, 64), interpolation=Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			testset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)
			test_loader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=args.workers)

		elif args.dataset == 'mnist':
			transform = transforms.Compose([transforms.Resize((28, 28), interpolation=Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			testset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform)
			test_loader = torch.utils.data.DataLoader(trainset, batch_size=1000, num_workers=args.workers)

		elif args.dataset == 'celeba':
			transform = transforms.Compose([transforms.Resize((64, 64), interpolation=Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			testset = datasets.ImageFolder(args.data_path, transform=transform)
			test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

		save_testdata_statistics(fid_model, test_loader, downsample_=True, cuda_mode=args.cuda)

		pfile = open(args.data_stat_path, 'rb')
		statistics = pickle.load(pfile)
		pfile.close()

		m, C = statistics['m'], statistics['C']

	if args.cuda:
		generator = generator.cuda()
		fid_model = fid_model.cuda()

	fid = []

	for i in range(args.ntests):

		fid.append(compute_fid(generator, fid_model, args.bsize, args.nsamples, m, C, args.cuda, inception = True if args.model_cifar == 'inception' else False, mnist = True if args.dataset == 'mnist' else False, SNGAN=args.sngan))

		print(fid[-1])
	fid = np.asarray(fid)
	print(fid)

	print('min sdd: {:0.4f} +- {:0.4f}'.format(fid.mean(), fid.std()))
