from __future__ import print_function
import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data
from scipy.stats import sem
import scipy.linalg as sla
from generators import *
from models_fid import *
from metrics import compute_fid

import pickle

if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='FID computation')
	parser.add_argument('--model-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--data-stat-path', type=str, default='../test_data_statistics.p', metavar='Path', help='Path to file containing test data statistics')
	parser.add_argument('--fid-model-path', type=str, default=None, metavar='Path', help='Path to fid model')
	parser.add_argument('--batch-size', type=int, default=512, metavar='Path', help='batch size')
	parser.add_argument('--nsamples', type=int, default=1000, metavar='Path', help='number of samples per replication')
	parser.add_argument('--ntests', type=int, default=3, metavar='Path', help='number of replications')
	parser.add_argument('--dataset', choices=['cifar10', 'mnist'], default='cifar10', help='cifar10 or mnist')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False
	args.bsize = min(args.batch_size, args.nsamples)

	if args.model_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --model-path to indicate the path!')

	pfile = open(args.data_stat_path, 'rb')
	statistics = pickle.load(pfile)
	pfile.close()

	m, C = statistics['m'], statistics['C']

	if args.dataset == 'cifar10':
		fid_model = ResNet18().eval()
		mod_state = torch.load(args.fid_model_path, map_location=lambda storage, loc: storage)
		fid_model.load_state_dict(mod_state['model_state'])

		generator = Generator(100, [1024, 512, 256, 128], 3).eval()
		gen_state = torch.load(args.model_path, map_location=lambda storage, loc: storage)
		generator.load_state_dict(gen_state['model_state'])

	elif args.dataset == 'mnist':
		fid_model = cnn().eval()
		mod_state = torch.load(args.fid_model_path, map_location=lambda storage, loc: storage)
		fid_model.load_state_dict(mod_state['model_state'])

		generator = Generator_mnist().eval()
		gen_state = torch.load(args.model_path, map_location=lambda storage, loc: storage)
		generator.load_state_dict(gen_state['model_state'])

	if args.cuda:
		generator = generator.cuda()
		fid_model = fid_model.cuda()

	fid = []

	for i in range(args.ntests):
		fid.append(compute_fid(generator, fid_model, args.bsize, args.nsamples, m, C, args.cuda))

	fid = np.asarray(fid)

	print('min sdd: {:0.4f} +- {:0.4f}'.format(fid.mean(), fid.std()))
