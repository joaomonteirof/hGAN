from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from scipy.stats import chi2
from torch.autograd import Variable
from torchvision.transforms import transforms


def save_samples(generator, cp_name, save_name, n_samples, toy_dataset, save_dir='./'):
	generator.eval()

	noise = torch.randn(n_samples, 2).view(-1, 2)

	noise = Variable(noise, volatile=True)
	samples = generator(noise)

	if (toy_dataset == '8gaussians'):
		scale = 2.0 / 1.414
		centers = [
			(1, 0),
			(-1, 0),
			(0, 1),
			(0, -1),
			(1. / np.sqrt(2), 1. / np.sqrt(2)),
			(1. / np.sqrt(2), -1. / np.sqrt(2)),
			(-1. / np.sqrt(2), 1. / np.sqrt(2)),
			(-1. / np.sqrt(2), -1. / np.sqrt(2))
		]

		centers = [(scale * x, scale * y) for x, y in centers]
		centers = np.asarray(centers)
		cov_all = np.array([(0.02, 0), (0, 0.02)])

	elif (toy_dataset == '25gaussians'):
		range_ = np.arange(-2, 3)
		centers = np.transpose(np.meshgrid(range_, range_, indexing='ij'), (1, 2, 0)).reshape(-1, 2)
		scale = 1. / 2.828
		cov_all = np.array([(0.05, 0), (0, 0.05)])

	plt.scatter(samples[:, 0], samples[:, 1], c='red', marker='o', alpha=0.1)
	plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', alpha=1)

	for k in range(centers.shape[0]):
		ellipse_data = plot_ellipse(x_cent=centers[k, 0], y_cent=centers[k, 1], cov=cov_all, mass_level=0.7)
		plt.plot(ellipse_data[0], ellipse_data[1], c='black', alpha=0.2)

	# save figure

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	save_fn = save_dir + 'toy_data_' + save_name + '_' + cp_name + '.png'
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
		sample.save('sample_{}.png'.format(i + 1))
