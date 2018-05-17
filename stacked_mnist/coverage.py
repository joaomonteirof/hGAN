from __future__ import print_function

import argparse

import os
import sys

sys.path.insert(0, os.path.realpath(__file__ + ('/..' * 3)))
print(f'Running from package root directory {sys.path[0]}')

import matplotlib.pyplot as plt
import torch.utils.data

from common.generators import Generator_stacked_mnist
from common.models_fid import cnn
from common.utils import *

if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing GANs under max hyper volume training')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--classifier-path', type=str, default=None, metavar='Path', help='Path to pretrained classifier on MNIST')
	parser.add_argument('--n-tests', type=int, default=4, metavar='N', help='number of samples to generate (default: 4)')
	parser.add_argument('--no-plots', action='store_true', default=False, help='Disables plot of train/test losses')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.cp_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')
	if args.classifier_path is None:
		raise ValueError('There is no classifier path. Use arg --classifier-path to indicate the path!')

	model = Generator_mnist()
	ckpt = torch.load(args.cp_path, map_location=lambda storage, loc: storage)
	model.load_state_dict(ckpt['model_state'])
	
	classifier = cnn().eval()
	ckpt = torch.load(args.classifier_path, map_location=lambda storage, loc: storage)
	classifier.load_state_dict(ckpt['model_state'])

	if args.cuda:
		model = model.cuda()
		classifier = classifier.cuda()

	print('Cuda Mode is: {}'.format(args.cuda))
