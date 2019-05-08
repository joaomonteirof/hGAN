from __future__ import print_function

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from scipy.stats import chi2
import scipy.linalg as sla
from torchvision.transforms import transforms

from PIL import ImageEnhance

def compute_KL(p, q):
	return np.sum(np.where(p!=0, p*np.log(p/(q+1e-10)), 0))

def save_testdata_statistics(model, data_loader, cuda_mode, downsample_ = True):

	if cuda_mode:
		model = model.cuda()

	for batch in data_loader:

		x, y = batch

		if cuda_mode:
			x = x.cuda()

		out = model.forward(x, downsample_).detach().cpu().numpy()

		try:
			logits = np.concatenate([logits, out], 0)
		except NameError:
			logits = out

	if cuda_mode:
		model = model.cpu()

	m = logits.mean(0)
	C = np.cov(logits, rowvar=False)

	pfile = open('../test_data_statistics.p', "wb")
	pickle.dump({'m': m, 'C': C}, pfile)
	pfile.close()


def save_samples(generator: torch.nn.Module, cp_name: str, cuda_mode: bool, prefix: str, save_dir='./', nc=3, im_size=64, fig_size=(5, 5), enhance=True, SNGAN=False):
	generator.eval()

	n_tests = fig_size[0] * fig_size[1]

	to_pil = transforms.ToPILImage()
	to_tensor = transforms.ToTensor()

	if SNGAN:
		noise = torch.randn(n_tests, 128).view(-1, 128, 1, 1)
	else:
		noise = torch.randn(n_tests, 100).view(-1, 100, 1, 1)

	if cuda_mode:
		noise = noise.cuda()

	gen_image = generator(noise).view(-1, nc, im_size, im_size)
	gen_image = denorm(gen_image)

	#n_rows = np.sqrt(noise.size()[0]).astype(np.int32)
	#n_cols = np.sqrt(noise.size()[0]).astype(np.int32)
	n_cols, n_rows = fig_size
	fig, axes = plt.subplots(n_cols, n_rows, figsize=(n_rows, n_cols))
	for ax, img in zip(axes.flatten(), gen_image):
		ax.axis('off')
		ax.set_adjustable('box-forced')

		img = img.cpu().data

		if enhance:
			img_E = ImageEnhance.Sharpness( to_pil(img) ).enhance(1.)
			img = to_tensor(img_E)

		# Scale to 0-255
		img = (((img - img.min()) * 255) / (img.max() - img.min())).numpy().transpose(1, 2, 0).astype(np.uint8).squeeze()
		# ax.imshow(img.cpu().detach().view(image_size, image_size, 3).numpy(), cmap=None, aspect='equal')

		if nc == 1:
			ax.imshow(img, cmap="gray", aspect='equal')
		else:
			ax.imshow(img, cmap=None, aspect='equal')	

	plt.subplots_adjust(wspace=0, hspace=0)
	#title = 'Samples'
	#fig.text(0.5, 0.04, title, ha='center')

	# save figure

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	save_fn = save_dir + prefix + '_' + cp_name + '.pdf'
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


def test_model(model, n_tests, cuda_mode, enhance=True, SNGAN=False):
	model.eval()

	to_pil = transforms.ToPILImage()
	to_tensor = transforms.ToTensor()

	if SNGAN:
		z_ = torch.randn(n_tests, 128).view(-1, 128, 1, 1)
	else:
		z_ = torch.randn(n_tests, 100).view(-1, 100, 1, 1)

	if cuda_mode:
		z_ = z_.cuda()

	out = model.forward(z_)

	for i in range(out.size(0)):
		sample = denorm(out[i].data)

		if len(sample.size())<3:
			sample = sample.view(1, 28, 28)

		if enhance:
			sample = ImageEnhance.Sharpness( to_pil(sample.cpu()) ).enhance(1.2)
		else:
			sample = to_pil(sample.cpu())

		sample.save('sample_{}.pdf'.format(i + 1))

def plot_toy_data(x, centers, toy_dataset):
	
	if toy_dataset == '8gaussians':
		cov_all = np.array([(0.02**2, 0), (0, 0.02**2)])
		scale = 1.414

	elif toy_dataset == '25gaussians':
		cov_all = np.array([(0.05**2, 0), (0, 0.05**2)])
		scale = 2.828
		
	samples = scale*x

	plt.scatter(samples[:, 0], samples[:, 1], c = 'red', marker = 'o', alpha = 0.1)
	plt.scatter(centers[:, 0], centers[:, 1], c = 'black', marker = 'x', alpha = 1)

	for k in range(centers.shape[0]):
		ellipse_data = plot_ellipse(x_cent = centers[k, 0], y_cent = centers[k, 1], cov = cov_all, mass_level = 0.9545)
		plt.plot(ellipse_data[0], ellipse_data[1], c = 'black', alpha = 0.2)

	plt.show()

def save_samples_toy_data(x, centers, save_name, toy_dataset):
	
	if toy_dataset == '8gaussians':
		cov_all = np.array([(0.02**2, 0), (0, 0.02**2)])

		scale = 1.414

	elif toy_dataset == '25gaussians':
		cov_all = np.array([(0.05**2, 0), (0, 0.05**2)])

		scale = 2.828
		
	samples = scale*x

	plt.scatter(samples[:, 0], samples[:, 1], c = 'red', marker = 'o', alpha = 0.1)
	plt.scatter(centers[:, 0], centers[:, 1], c = 'black', marker = 'x', alpha = 1)


	for k in range(centers.shape[0]):

		ellipse_data = plot_ellipse(x_cent = centers[k, 0], y_cent = centers[k, 1], cov = cov_all, mass_level = 0.9973)
		plt.plot(ellipse_data[0], ellipse_data[1], c = 'black', alpha = 1)

	# save figure

	save_name = save_name + '.pdf'
	plt.savefig(save_name)

	plt.close()

def save_samples_toy_data_gen(generator, cp_name, save_name, n_samples, toy_dataset, save_dir='./'):
	generator.eval()

	noise = torch.randn(n_samples, 2).view(-1, 2)

	samples = generator(noise)

	if toy_dataset == '8gaussians':
		scale_cent = 2.
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

		centers = [(scale_cent * x, scale_cent * y) for x, y in centers]
		centers = np.asarray(centers)
		cov_all = np.array([(0.02 ** 2, 0), (0, 0.02 ** 2)])

		scale = 1.414

	elif toy_dataset == '25gaussians':
		range_ = np.arange(-2, 3)
		centers = 2 * np.transpose(np.meshgrid(range_, range_, indexing='ij'), (1, 2, 0)).reshape(-1, 2)
		cov_all = np.array([(0.05 ** 2, 0), (0, 0.05 ** 2)])

		scale = 2.828

	samples = scale * samples

	plt.scatter(samples[:, 0], samples[:, 1], c='red', marker='o', alpha=0.1)
	plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', alpha=1)

	for k in range(centers.shape[0]):
		ellipse_data = plot_ellipse(x_cent=centers[k, 0], y_cent=centers[k, 1], cov=cov_all, mass_level=0.9973)
		plt.plot(ellipse_data[0], ellipse_data[1], c='black', alpha=1)

	# save figure

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	save_fn = save_dir + 'toy_data_' + save_name + '_' + cp_name + '.pdf'
	plt.savefig(save_fn)

	plt.close()

def calculate_dist(x_, y_):

	dist_matrix = np.zeros([x_.shape[0], y_.shape[0]])

	for i in range(x_.shape[0]):
		for j in range(y_.shape[0]):

			dist_matrix[i, j] = np.sqrt((x_[i, 0] - y_[j, 0])**2 + (x_[i, 1] - y_[j, 1])**2)

	return dist_matrix

def metrics_toy_data(x, centers, cov, toy_dataset, slack = 3.0):

	if toy_dataset == '8gaussians':
		distances = calculate_dist(1.414*x, centers)
	
	elif toy_dataset == '25gaussians':
		distances = calculate_dist(2.828*x, centers)
		
	closest_center = np.argmin(distances, 1)

	n_gaussians = centers.shape[0]

	fd = 0
	quality_samples = 0
	quality_modes = 0

	for cent in range(n_gaussians):

		center_samples = x[np.where(closest_center == cent)]

		center_distances = distances[np.where(closest_center == cent)]

		sigma = cov[0, 0]

		quality_samples_center = np.sum(center_distances[:, cent] <= slack*np.sqrt(sigma))
		quality_samples += quality_samples_center

		if quality_samples_center > 0:
			quality_modes += 1

		if center_samples.shape[0] > 3:

			m = np.mean(center_samples, 0)
			C = np.cov(center_samples, rowvar = False)

			fd += ((centers[cent] - m)**2).sum() + np.matrix.trace(C + cov - 2*sla.sqrtm( np.matmul(C, cov)))


	fd_all = fd / len(np.unique(closest_center))

	return fd_all, quality_samples, quality_modes

class ResnetBlock(nn.Module):
	def __init__(self, fin, fout, fhidden=None, is_bias=True):
		super().__init__()
		# Attributes
		self.is_bias = is_bias
		self.learned_shortcut = (fin != fout)
		self.fin = fin
		self.fout = fout
		if fhidden is None:
			self.fhidden = min(fin, fout)
		else:
			self.fhidden = fhidden

		# Submodules
		self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
		self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
		if self.learned_shortcut:
			self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)


	def forward(self, x):
		x_s = self._shortcut(x)
		dx = self.conv_0(F.leaky_relu(x, 2e-1))
		dx = self.conv_1(F.leaky_relu(dx, 2e-1))
		out = x_s + 0.1*dx

		return out

	def _shortcut(self, x):
		if self.learned_shortcut:
			x_s = self.conv_s(x)
		else:
			x_s = x
		return x_s
