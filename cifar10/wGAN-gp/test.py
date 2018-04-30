from __future__ import print_function
import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3
from PIL import ImageFilter
import matplotlib.pyplot as plt
import model as model_
import numpy as np
from scipy.stats import entropy
import os

from numpy.lib.stride_tricks import as_strided

from common.metrics import inception_score


def denorm(unorm):

	norm = (unorm + 1) / 2

	return norm.clamp(0, 1)

def test_model(model, n_tests, cuda_mode):

	model.eval()

	to_pil = transforms.ToPILImage()
	to_tensor = transforms.ToTensor()

	z_ = torch.randn(n_tests, 100).view(-1, 100, 1, 1)

	if cuda_mode:
		z_ = z_.cuda()

	z_ = Variable(z_)
	out = model.forward(z_)

	for i in range(out.size(0)):
		sample = denorm(out[i].data)
		sample = to_pil(sample.cpu())
		sample.save('sample_{}.png'.format(i+1))

def save_samples(generator, cp_name, cuda_mode, save_dir='./', fig_size=(5, 5)):
	generator.eval()

	n_tests = fig_size[0]*fig_size[1]

	noise = torch.randn(n_tests, 100).view(-1, 100, 1, 1)

	if cuda_mode:
		noise = noise.cuda()

	noise = Variable(noise, volatile=True)
	gen_image = generator(noise)
	gen_image = denorm(gen_image)

	generator.train()

	n_rows = np.sqrt(noise.size()[0]).astype(np.int32)
	n_cols = np.sqrt(noise.size()[0]).astype(np.int32)
	fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
	for ax, img in zip(axes.flatten(), gen_image):
		ax.axis('off')
		ax.set_adjustable('box-forced')
		# Scale to 0-255
		img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
		# ax.imshow(img.cpu().data.view(image_size, image_size, 3).numpy(), cmap=None, aspect='equal')
		ax.imshow(img, cmap=None, aspect='equal')
	plt.subplots_adjust(wspace=0, hspace=0)
	title = 'Samples'
	fig.text(0.5, 0.04, title, ha='center')

	# save figure

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	save_fn = save_dir + 'Cifar10_wGAN-gp_'+ cp_name + '.png'
	plt.savefig(save_fn)

	plt.close()

def plot_learningcurves(history, *keys):

	for key in keys:
		plt.plot(history[key])

	plt.show()

if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing GANs under max hyper volume training')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--n-tests', type=int, default=4, metavar='N', help='number of samples to generate (default: 4)')
	parser.add_argument('--no-plots', action='store_true', default=False, help='Disables plot of train/test losses')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--inception', action='store_true', default=False, help='Enables computation of the inception score over the test set of cifar10')
	parser.add_argument('--n-inception', type=int, default=1024, metavar='N', help='number of samples to calculate inception score (default: 1024)')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.cp_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	model = model_.Generator(100, [1024, 512, 256, 128], 3)

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	model.load_state_dict(ckpt['model_state'])

	if args.cuda:
		model = model.cuda()

	print('Cuda Mode is: {}'.format(args.cuda))

	history = ckpt['history']

	if not args.no_plots:

		plot_learningcurves(history, 'gen_loss')
		plot_learningcurves(history, 'disc_loss')
		plot_learningcurves(history, 'gen_loss_minibatch')
		plot_learningcurves(history, 'disc_loss_minibatch')
		plot_learningcurves(history, 'FID-c')

	test_model(model=model, n_tests=args.n_tests, cuda_mode=args.cuda)
	save_samples(generator=model, cp_name=args.cp_path.split('/')[-1].split('.')[0], cuda_mode=args.cuda)

	if args.inception:
		print( inception_score(model, N=args.n_inception, cuda=args.cuda, resize=True, splits=10) )
