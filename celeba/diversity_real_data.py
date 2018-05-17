from __future__ import print_function

import argparse

import os
import sys
import glob

sys.path.insert(0, os.path.realpath(__file__ + ('/..' * 2)))
print(f'Running from package root directory {sys.path[0]}')

import matplotlib.pyplot as plt

import pickle

from common.utils import *
from common.metrics import compute_diversity_mssim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import PIL.Image as Image

if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing GANs under max hyper volume training')
	parser.add_argument('--ntests', type=int, default=5, metavar='N', help='number of samples to generate (default: 4)')
	parser.add_argument('--nsamples', type=int, default=1000, metavar='Path', help='number of samples per replication')
	parser.add_argument('--data-path', type=str, default='../data/', metavar='Path', help='Path to data')
	parser.add_argument('--out-file', type=str, default='./celeba_real_diversity.p', metavar='Path', help='file for dumping boxplot data')
	args = parser.parse_args()


	transform = transforms.Compose([transforms.Resize((64, 64), interpolation=Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	celebA_data = datasets.ImageFolder(args.data_path, transform=transform)
	train_loader = iter(torch.utils.data.DataLoader(celebA_data, batch_size=args.nsamples, shuffle=True))

	mssim = {'mssim':[]}

	for i in range(args.ntests):
		samples = next(train_loader)[0]
		mssim['mssim'].append(compute_diversity_mssim(samples, real = True, mnist = False))

	print(mssim)

	pfile = open(args.out_file, "wb")
	pickle.dump(mssim, pfile)
	pfile.close()
