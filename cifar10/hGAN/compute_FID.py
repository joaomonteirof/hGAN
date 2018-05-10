from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.realpath(__file__ + ('/..' * 3)))
print(f'Running from package root directory {sys.path[0]}')

import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data
from scipy.stats import sem
from common.generators import Generator
from common.resnet import ResNet18
from common.discriminators import *
import pickle

def compute_fid(model, fid_model_, batch_size, nsamples, m_data, C_data, cuda):

	model.eval()
	fid_model_.eval()

	if nsamples % batch_size == 0: n_batches = nsamples//batch_size else: n_batches = nsamples//batch_size + 1

	logits = []

	for i in range(n_batches):

		z_ = torch.randn(min(batch_size, nsamples - batch_size*i), 100).view(-1, 100, 1, 1)
		if cuda:
			z_ = z_.cuda()

		z_ = Variable(z_)

		x_gen = model.forward(z_)

		logits.append( fid_model_.forward(x_gen).cpu().data.numpy() )

	logits = np.asarray(logits)
	m_gen = logits.mean(0)
	C_gen = np.cov(logits, rowvar=False)

	fid = ((m_data - m_gen) ** 2).sum() + np.matrix.trace(C_data + C_gen - 2 * sla.sqrtm(np.matmul(C_data, C_gen)))

	return fid

if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='FID computation')
	parser.add_argument('--model-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--data-stat-path', type=str, default='../test_data_statistics.p', metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--fid-model-path', type=str, default=None, metavar='Path', help='Path to fid model')
	parser.add_argument('--batch-size', type=int, default=512, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--nsamples', type=int, default=1000, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--ntests', type=int, default=3, metavar='Path', help='Checkpoint/model path')
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

	fid_model = ResNet18().eval()
	mod_state = torch.load(args.fid_model_path, map_location=lambda storage, loc: storage)
	fid_model.load_state_dict(mod_state['model_state'])

	generator = Generator(100, [1024, 512, 256, 128], 3).eval()
	gen_state = torch.load(args.model_path, map_location=lambda storage, loc: storage)
	generator.load_state_dict(gen_state['model_state'])

	if args.cuda:
		generator = generator.cuda()
		fid_model = fid_model.cuda()

	fid = []

	for i in range(args.ntests):
		fid.append(compute_fid(generator, fid_model, args.nsamples, m, C, args.cuda))

	fid = np.asarray(fid)

	min_sdd = sdd.min(1)

	print('min sdd: {:0.4f} +- {:0.4f}'.format(min_sdd.min(), min_sdd.std()))
