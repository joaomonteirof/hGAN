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
	parser.add_argument('--cp-folder', type=str, default=None, metavar='Path', help='Checkpoints/models path')
	parser.add_argument('--classifier-path', type=str, default=None, metavar='Path', help='Path to pretrained classifier on MNIST')
	parser.add_argument('--data-statistics-path', type=str, default=None, metavar='Path', help='Path to precomputed test data statistics fo KL div computation')
	parser.add_argument('--n-tests', type=int, default=4, metavar='N', help='number of replications (default: 4)')
	parser.add_argument('--n-samples', type=int, default=1000, metavar='N', help='number of samples for each  (default: 4)')
	parser.add_argument('--batch-size', type=int, default=512, metavar='Path', help='batch size')
	parser.add_argument('--no-plots', action='store_true', default=False, help='Disables plot of train/test losses')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.cp_folder is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-folder to indicate the path!')
	if args.classifier_path is None:
		raise ValueError('There is no classifier path. Use arg --classifier-path to indicate the path!')
	
	classifier = cnn().eval()
	classifier_state = torch.load(args.classifier_path, map_location=lambda storage, loc: storage)
	classifier.load_state_dict(classifier_state['model_state'])

	if args.cuda:
		classifier = classifier.cuda()

	print('Cuda Mode is: {}'.format(args.cuda))

	files_list = glob.glob(args.cp_folder + 'G_*.pt')
	files_list.sort()

	coverage, KL = [], []

	for file_id in files_list:

		file_name = file_id.split('/')[-1].split('_')[1]

		print(file_name)		

		key = models_dict[file_name]

		generator = Generator_stacked_mnist().eval()
		gen_state = torch.load(file_id, map_location = lambda storage, loc: storage)
		generator.load_state_dict(gen_state['model_state'])

		if args.cuda:
			generator = generator.cuda()

		
