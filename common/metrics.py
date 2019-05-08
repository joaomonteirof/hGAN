from __future__ import print_function

import numpy as np
import torch
import torch.utils.data
from numpy.lib.stride_tricks import as_strided
from scipy.stats import entropy
from torch import nn
from torch.nn import functional as F
from torchvision.models.inception import inception_v3
import scipy.linalg as sla
from skimage.measure import compare_ssim
import itertools

def compute_freqs(model, classifier, batch_size, nsamples, cuda):

	model.eval()
	classifier.eval()

	counter = np.zeros(1000)

	if nsamples % batch_size == 0: 
		n_batches = nsamples//batch_size
	else: n_batches = nsamples//batch_size + 1

	with torch.no_grad():

		for i in range(n_batches):

			z_ = torch.randn(min(batch_size, nsamples - batch_size*i), 100).view(-1, 100, 1, 1)
			if cuda:
				z_ = z_.cuda()

			x_gen = model.forward(z_)

			x_gen_1, x_gen_2, x_gen_3 = x_gen[:, 0, :, :].unsqueeze(1), x_gen[:, 1, :, :].unsqueeze(1), x_gen[:, 2, :, :].unsqueeze(1)

			logits_1, logits_2, logits_3 = classifier.forward(x_gen_1), classifier.forward(x_gen_2), classifier.forward(x_gen_3)

			pred_1, pred_2, pred_3 = logits_1.detach().max(1)[1], logits_2.detach().max(1)[1], logits_3.detach().max(1)[1]

			for j in range(z_.size(0)):
				counter[100*pred_1[j]+10*pred_2[j]+pred_3[j]]+=1.

	return np.count_nonzero(counter), counter

def compute_freqs_real_data(data_loader, classifier, cuda):

	classifier.eval()

	counter = np.zeros(1000)

	with torch.no_grad():

		for batch in data_loader:

			if cuda:
				batch = batch.cuda()

			x_gen_1, x_gen_2, x_gen_3 = batch[:, 0, :, :].unsqueeze(1), batch[:, 1, :, :].unsqueeze(1), batch[:, 2, :, :].unsqueeze(1)

			logits_1, logits_2, logits_3 = classifier.forward(x_gen_1), classifier.forward(x_gen_2), classifier.forward(x_gen_3)

			pred_1, pred_2, pred_3 = logits_1.detach().max(1)[1], logits_2.detach().max(1)[1], logits_3.detach().max(1)[1]

			for j in range(batch.size(0)):
				counter[100*pred_1[j]+10*pred_2[j]+pred_3[j]]+=1.
	return counter

def compute_diversity_mssim(samples, real = True, mnist=True):
	l = list(range(len(samples)))
	pairs = itertools.combinations(l, 2)
	
	mssim = []

	for pair in pairs:

		if real:

			im1 = samples[pair[0]].cpu().numpy()
			im2 = samples[pair[1]].cpu().numpy()
		else:
			im1 = samples[pair[0]]
			im2 = samples[pair[1]]
			
		if mnist:
			im1, im2 = im1.squeeze(), im2.squeeze()
			mssim.append( compare_ssim(im1, im2) )
		else:
			im1, im2 = np.rollaxis(im1, 0, 3), np.rollaxis(im2, 0, 3)
			mssim.append(compare_ssim(im1, im2, multichannel = True))

	return np.mean(mssim)

def get_gen_samples(model, batch_size, nsamples, cuda, mnist=True):

	model.eval()

	if nsamples % batch_size == 0: 
		n_batches = nsamples//batch_size
	else: n_batches = nsamples//batch_size + 1

	out_samples = None

	with torch.no_grad():

		for i in range(n_batches):

			z_ = torch.randn(min(batch_size, nsamples - batch_size*i), 100).view(-1, 100, 1, 1)
			if cuda:
				z_ = z_.cuda()
				model = model.cuda()

			if mnist:
				x_gen = model.forward(z_).view(z_.size(0), 1, 28, 28)
			else:
				x_gen = model.forward(z_)

			if out_samples is not None:
				out_samples = np.concatenate( [out_samples, x_gen.cpu().detach().numpy()], axis=0)
			else:
				out_samples = x_gen.cpu().detach().numpy()

	return out_samples

def compute_fid(model, fid_model_, batch_size, nsamples, m_data, C_data, cuda, inception=False, mnist=True, SNGAN = False):

	model.eval()
	fid_model_.eval()

	if nsamples % batch_size == 0: 
		n_batches = nsamples//batch_size
	else: n_batches = nsamples//batch_size + 1

	logits = None

	with torch.no_grad():

		for i in range(n_batches):

			if SNGAN:
				z_ = torch.randn(min(batch_size, nsamples - batch_size*i), 128)
				downsample_=False
			else:
				z_ = torch.randn(min(batch_size, nsamples - batch_size*i), 100).view(-1, 100, 1, 1)
				downsample_=True
			if cuda:
				z_ = z_.cuda()

			if mnist:
				x_gen = model.forward(z_).view(z_.size(0), 1, 28, 28)
			else:
				x_gen = model.forward(z_)

			if inception:
				new_logits = fid_model_(x_gen)[0].view(z_.size(0), -1).detach().cpu().numpy()
			else:
				new_logits = fid_model_.forward(x_gen, downsample_).cpu().detach().numpy()

			if logits is not None:
				logits = np.concatenate( [logits, fid_model_.forward(x_gen, downsample_).cpu().detach().numpy()], axis=0 )
			else:
				logits = fid_model_.forward(x_gen, downsample_).cpu().detach().numpy()

	logits = np.asarray(logits)

	m_gen = logits.mean(0)
	C_gen = np.cov(logits, rowvar=False)

	fid = ((m_data - m_gen) ** 2).sum() + np.matrix.trace(C_data + C_gen - 2 * sla.sqrtm(np.matmul(C_data, C_gen)))

	return fid

def compute_fid_real_data(data_loader, fid_model_, m_data, C_data, cuda, inception=False, mnist=True):

	fid_model_.eval()

	if inception:
		up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

	logits = None

	with torch.no_grad():

		for batch in data_loader:

			x_gen, _ = batch

			if cuda:
				x_gen = x_gen.cuda()

			if mnist:
				x_gen = x_gen.view(x_gen.size(0), 1, 28, 28)

			if inception:
				x_gen = up(x_gen)
				new_logits = F.softmax(fid_model_.forward(x_gen), dim=1).detach().cpu().numpy()
			else:
				new_logits = fid_model_.forward(x_gen).cpu().detach().numpy()

			if logits is not None:
				logits = np.concatenate( [logits, new_logits], axis=0 )
			else:
				logits = new_logits

	logits = np.asarray(logits)

	m_gen = logits.mean(0)
	C_gen = np.cov(logits, rowvar=False)

	fid = ((m_data - m_gen) ** 2).sum() + np.matrix.trace(C_data + C_gen - 2 * sla.sqrtm(np.matmul(C_data, C_gen)))

	return fid

def inception_score(model, N=1000, cuda=True, batch_size=32, resize=False, splits=1, SNGAN = False):
	"""
	adapted from: https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
	Computes the inception score of images generated by model
	model -- Pretrained Generator
	N -- Number of samples to test
	cuda -- whether or not to run on GPU
	batch_size -- batch size for feeding into Inception v3
	splits -- number of splits
	"""

	assert batch_size > 0
	assert N > batch_size

	# Set up dtype
	if cuda:
		dtype = torch.cuda.FloatTensor
	else:
		if torch.cuda.is_available():
			print("WARNING: You have a CUDA device, so you should probably set cuda=True")
		dtype = torch.FloatTensor

	# Load inception model
	inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
	inception_model.eval()
	up = nn.functional.interpolate

	def get_pred(N_s):

		if SNGAN:
			z_ = torch.randn(N_s, 128)
		else:	
			z_ = torch.randn(N_s, 100).view(-1, 100, 1, 1)

		if cuda:
			z_ = z_.cuda()

		with torch.no_grad():

			x = model.forward(z_)

			if resize:
				x = up(x, size=(299, 299), mode='bilinear', align_corners=True).type(dtype)
			x = inception_model(x)
			return F.softmax(x, dim=1).detach().cpu().numpy()

	indexes = strided_app(np.arange(N), batch_size, batch_size)

	N = indexes[-1][-1] + 1

	# Get predictions
	preds = np.zeros((N, 1000))

	for i, idx in enumerate(indexes, 0):
		batch_size_i = idx.shape[0]

		preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batch_size_i)

	# Now compute the mean kl-div
	split_scores = []

	for k in range(splits):
		part = preds[k * (N // splits): (k + 1) * (N // splits), :]
		py = np.mean(part, axis=0)
		scores = []
		for i in range(part.shape[0]):
			pyx = part[i, :]
			scores.append(entropy(pyx, py))
		split_scores.append(np.exp(np.mean(scores)))

	return np.mean(split_scores), np.std(split_scores)


def strided_app(a, L, S):
	nrows = ((len(a) - L) // S) + 1
	n = a.strides[0]
	return as_strided(a, shape=(nrows, L), strides=(S * n, n))
