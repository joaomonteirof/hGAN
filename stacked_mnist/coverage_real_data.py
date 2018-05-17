from __future__ import print_function

import argparse

import os
import sys

sys.path.insert(0, os.path.realpath(__file__ + ('/..' * 2)))
print(f'Running from package root directory {sys.path[0]}')

import matplotlib.pyplot as plt
import torch.utils.data

from common.models_fid import cnn
from common.utils import *
from common.metrics import *
from data_load import Loader

if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing GANs under max hyper volume training')
	parser.add_argument('--classifier-path', type=str, default=None, metavar='Path', help='Path to pretrained classifier on MNIST')
	parser.add_argument('--data-path', type=str, default='./test.hdf', metavar='Path', help='Path to hdf file containing stacked MNIST. Can be generated with gen_data.py')
	parser.add_argument('--out-file', type=str, default='./test_data_coverage.p', metavar='Path', help='files for dumping coverage data')
	parser.add_argument('--batch-size', type=int, default=512, metavar='Path', help='batch size')
	parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.classifier_path is None:
		raise ValueError('There is no classifier path. Use arg --classifier-path to indicate the path!')
	
	classifier = cnn().eval()
	classifier_state = torch.load(args.classifier_path, map_location=lambda storage, loc: storage)
	classifier.load_state_dict(classifier_state['model_state'])

	if args.cuda:
		classifier = classifier.cuda()

	print('Cuda Mode is: {}'.format(args.cuda))

	testset = Loader(args.data_path)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, num_workers=args.workers)

	freqs = compute_freqs_real_data(test_loader, classifier, cuda=args.cuda)
	freqs/=freqs.sum()

	pfile = open(args.out_file, "wb")
	pickle.dump(freqs, pfile)
	pfile.close()
