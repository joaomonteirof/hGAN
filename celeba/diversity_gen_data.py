from __future__ import print_function

import argparse

import os
import sys
import glob

sys.path.insert(0, os.path.realpath(__file__ + ('/..' * 2)))
print(f'Running from package root directory {sys.path[0]}')

import matplotlib.pyplot as plt

import pickle

from common.generators import Generator
from common.utils import *
from common.metrics import compute_diversity_mssim, get_gen_samples
import torchvision.transforms as transforms

if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing GANs under max hyper volume training')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--ntests', type=int, default=5, metavar='N', help='number of samples to generate (default: 4)')
	parser.add_argument('--nsamples', type=int, default=1000, metavar='Path', help='number of samples per replication')
	parser.add_argument('--data-stat-path', type=str, default='./celeba_real_diversity.p', metavar='Path', help='Path to file containing real data statistics')
	parser.add_argument('--data-path', type=str, default='../data/', metavar='Path', help='Path to data')
	parser.add_argument('--out-file', type=str, default='./gen_diversity.p', metavar='Path', help='file for dumping boxplot data')
	parser.add_argument('--batch-size', type=int, default=512, metavar='Path', help='batch size')
	parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	pfile = open('celeba_real_diversity.p', 'rb')
	real_dict = pickle.load(pfile)
	pfile.close()
	real_mean = np.mean(real_dict['mssim'])
	print(real_mean)

	if args.cp_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	generator = Generator(100, [1024, 512, 256, 128], 3).eval()
	gen_state = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	generator.load_state_dict(gen_state['model_state'])

	mssim = {'mssim':[]}

	for i in range(args.ntests):
		samples = get_gen_samples(generator, batch_size=args.batch_size, nsamples=args.nsamples, cuda=args.cuda, mnist=False)
		curr_mssim = compute_diversity_mssim(samples, real = False, mnist=False) - real_mean
		mssim['mssim'].append(curr_mssim)
		print(curr_mssim)

	pfile = open(args.out_file, "wb")
	pickle.dump(mssim, pfile)
	pfile.close()
