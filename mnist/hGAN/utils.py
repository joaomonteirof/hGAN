from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.realpath(__file__ + ('/..' * 3)))
print(f'Running from package root directory {sys.path[0]}')

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from scipy.stats import chi2
from torch.autograd import Variable
from torchvision.transforms import transforms


def save_samples(generator: torch.nn.Module, cp_name: str, cuda_mode: bool, prefix: str, save_dir='./', fig_size=(5, 5), nc=3, im_size=64):
	generator.eval()

	n_tests = fig_size[0] * fig_size[1]

	noise = torch.randn(n_tests, 100)

	if cuda_mode:
		noise = noise.cuda()

	noise = Variable(noise, volatile=True)
	gen_image = generator(noise).view(noise.size(0), nc, im_size, im_size)
	gen_image = denorm(gen_image)

	generator.train()
	n_rows = np.sqrt(noise.size()[0]).astype(np.int32)
	n_cols = np.sqrt(noise.size()[0]).astype(np.int32)
	fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
	for ax, img in zip(axes.flatten(), gen_image):
		ax.axis('off')
		ax.set_adjustable('box-forced')
		# Scale to 0-255
		img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8).squeeze()
		# img = (img).cpu().data.numpy().transpose(1, 2, 0).squeeze()

		# ax.imshow(img.cpu().data.view(image_size, image_size, 3).numpy(), cmap=None, aspect='equal')
		ax.imshow(img, cmap="gray", aspect='equal')
	# ax.imshow(img)
	plt.subplots_adjust(wspace=0, hspace=0)
	title = 'Samples'
	fig.text(0.5, 0.04, title, ha='center')

	# save figure

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	save_fn = save_dir + prefix + '_' + cp_name + '.png'
	plt.savefig(save_fn)

	plt.close()


def plot_ellipse(semimaj=1, semimin=1, phi=0, x_cent=0, y_cent=0, theta_num=1000, ax=None, plot_kwargs=None, cov=None,
				 mass_level=0.68):
	# Get Ellipse Properties from cov matrix
	eig_vec, eig_val, u = np.linalg.svd(cov)
	# Make sure 0th eigenvector has positive x-coordinate
	if eig_vec[0][0] < 0:
		eig_vec[0] *= -1
	semimaj = np.sqrt(eig_val[0])
	semimin = np.sqrt(eig_val[1])
	distances = np.linspace(0, 20, 20001)
	chi2_cdf = chi2.cdf(distances, df=2)
	multiplier = np.sqrt(
		distances[np.where(np.abs(chi2_cdf - mass_level) == np.abs(chi2_cdf - mass_level).min())[0][0]])
	semimaj *= multiplier
	semimin *= multiplier
	phi = np.arccos(np.dot(eig_vec[0], np.array([1, 0])))
	if eig_vec[0][1] < 0 and phi > 0:
		phi *= -1

	# Generate data for ellipse structure
	theta = np.linspace(0, 2 * np.pi, theta_num)
	r = 1 / np.sqrt((np.cos(theta)) ** 2 + (np.sin(theta)) ** 2)
	x = r * np.cos(theta)
	y = r * np.sin(theta)
	data = np.array([x, y])
	S = np.array([[semimaj, 0], [0, semimin]])
	R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
	T = np.dot(R, S)
	data = np.dot(T, data)
	data[0] += x_cent
	data[1] += y_cent

	return data


def denorm(unorm):
	norm = (unorm + 1) / 2

	return norm.clamp(0, 1)


def plot_learningcurves(history, *keys):
	for key in keys:
		plt.plot(history[key])

	plt.show()


def test_model(model, n_tests, cuda_mode, nc=3, im_size=64):
	model.eval()

	to_pil = transforms.ToPILImage()
	to_tensor = transforms.ToTensor()

	z_ = torch.randn(n_tests, 100).view(-1, 100, 1, 1)

	if cuda_mode:
		z_ = z_.cuda()

	z_ = Variable(z_)
	out = model.forward(z_)

	for i in range(out.size(0)):
		sample = denorm(out[i].data.view(nc, im_size, im_size))

		sample = to_pil(sample.cpu())
		sample.save('sample_{}.png'.format(i + 1))
