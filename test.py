from __future__ import print_function
import argparse
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data
from torch.autograd import Variable
import torchvision
from PIL import ImageFilter
import matplotlib.pyplot as plt
import model as model_
import numpy as np


def denorm(unorm):

	print(unorm.min(), unorm.max())

	norm = (unorm + 1) / 2

	print(norm.min(), norm.max())

	return norm.clamp(0, 1)

def test_model(model, n_tests, cuda_mode):

	model.eval()

	to_pil = transforms.ToPILImage()
	to_tensor = transforms.ToTensor()

	z_ = torch.randn(x.size(0), 100).view(-1, 100, 1, 1)

	if self.cuda_mode:
		z_ = z_.cuda()

	z_ = Variable(z_)
	out = self.model.forward(z_)

	for i in range(out.size(0)):
		sample = denorm(out[i])
		sample = to_pil(sample.cpu())
		sample.save('sample_{}.png'.format(i+1))

def plot_learningcurves(history, *keys):

	for key in keys:
		plt.plot(history[key])
	
	plt.show()


if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing GANs under max hyper volume training')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data .hdf')
	parser.add_argument('--n-tests', type=int, default=4, metavar='N', help='number of samples to  (default: 64)')
	parser.add_argument('--no-plots', action='store_true', default=False, help='Disables plot of train/test losses')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.cp_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	model = model_.Generator()

	quality_model = model_.model_nriqa()


	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	model.load_state_dict(ckpt['model_state'])

	if args.cuda:
		quality_model = quality_model.cuda()
		model = model.cuda()

	print('Cuda Mode is: {}'.format(args.cuda))

	history = ckpt['history']

	if not args.no_plots:

		plot_learningcurves(history, 'gen_loss')
		plot_learningcurves(history, 'disc_loss')

	test_model(model=model, n_tests=args.n_tests, cuda_mode=args.cuda)
