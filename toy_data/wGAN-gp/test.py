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
import os
from scipy.stats import chi2


def plot_ellipse(semimaj=1, semimin=1, phi=0, x_cent=0, y_cent=0, theta_num=1e3, ax=None, plot_kwargs=None, cov=None, mass_level=0.68):
	# Get Ellipse Properties from cov matrix
	eig_vec, eig_val, u = np.linalg.svd(cov)
	# Make sure 0th eigenvector has positive x-coordinate
	if eig_vec[0][0] < 0:
		eig_vec[0] *= -1
	semimaj = np.sqrt(eig_val[0])
	semimin = np.sqrt(eig_val[1])
	distances = np.linspace(0,20,20001)
	chi2_cdf = chi2.cdf(distances,df=2)
	multiplier = np.sqrt(distances[np.where(np.abs(chi2_cdf-mass_level)==np.abs(chi2_cdf-mass_level).min())[0][0]])
	semimaj *= multiplier
	semimin *= multiplier
	phi = np.arccos(np.dot(eig_vec[0],np.array([1,0])))
	if eig_vec[0][1] < 0 and phi > 0:
		phi *= -1

	# Generate data for ellipse structure
	theta = np.linspace(0, 2*np.pi, theta_num)
	r = 1 / np.sqrt((np.cos(theta))**2 + (np.sin(theta))**2)
	x = r*np.cos(theta)
	y = r*np.sin(theta)
	data = np.array([x,y])
	S = np.array([[semimaj, 0], [0, semimin]])
	R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
	T = np.dot(R,S)
	data = np.dot(T, data)
	data[0] += x_cent
	data[1] += y_cent

	return data

def save_samples(generator, cp_name, save_name, n_samples, toy_dataset, save_dir = './'):
	
	generator.eval()


	noise = torch.randn(n_samples, 2).view(-1, 2)

	noise = Variable(noise, volatile = True)
	samples = generator(noise)

	if (toy_dataset == '8gaussians'):
		scale = 2.0/1.414 
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
		centers = np.transpose(np.meshgrid(range_, range_, indexing = 'ij'), (1, 2, 0)).reshape(-1, 2)
		scale = 1./2.828
		cov_all = np.array([(0.05, 0), (0, 0.05)])

	plt.scatter(samples[:, 0], samples[:, 1], c = 'red', marker = 'o', alpha = 0.1)
	plt.scatter(centers[:, 0], centers[:, 1], c = 'black', marker = 'x', alpha = 1)


	for k in range(centers.shape[0]):

		ellipse_data = plot_ellipse(x_cent = centers[k, 0], y_cent = centers[k, 1], cov = cov_all, mass_level = 0.7)
		plt.plot(ellipse_data[0], ellipse_data[1], c = 'black', alpha = 0.2)

	# save figure

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	save_fn = save_dir + 'toy_data_' + save_name + '_' + cp_name + '.png'
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
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data .hdf')
	parser.add_argument('--n-samples', type=int, default=2500, metavar='N', help='number of samples to  (default: 10000)')
	parser.add_argument('--toy-dataset', choices=['8gaussians', '25gaussians'], default='8gaussians')
	parser.add_argument('--no-plots', action='store_true', default=False, help='Disables plot of train/test losses')
	args = parser.parse_args()

	if args.cp_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	generator = model_.Generator_toy(512)

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	generator.load_state_dict(ckpt['model_state'])


	history = ckpt['history']

	if not args.no_plots:

		plot_learningcurves(history, 'gen_loss')
		plot_learningcurves(history, 'disc_loss')
		plot_learningcurves(history, 'gen_loss_minibatch')
		plot_learningcurves(history, 'disc_loss_minibatch')
		plot_learningcurves(history, 'FD')

	save_samples(generator = generator, cp_name = args.cp_path.split('/')[-1].split('.')[0], save_name = args.cp_path.split('/')[-2].split('.')[0], n_samples = args.n_samples, toy_dataset = args.toy_dataset)
