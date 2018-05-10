import os

import numpy as np
import scipy.linalg as sla
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm


class TrainLoop(object):

	def __init__(self, generator, disc, optimizer, toy_dataset, centers, cov, train_loader, checkpoint_path=None, checkpoint_epoch=None):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch_fmt_gen = os.path.join(self.checkpoint_path, 'checkpoint_{}ep.pt')
		self.save_epoch_fmt_disc = os.path.join(self.checkpoint_path, 'D_checkpoint_{}ep.pt')
		self.model = generator
		self.disc = disc
		self.optimizer = optimizer
		self.train_loader = train_loader
		self.history = {'gen_loss': [], 'gen_loss_minibatch': [], 'disc_loss': [], 'disc_loss_minibatch': [], 'FD': [], 'quality_samples': [], 'quality_modes': []}
		self.total_iters = 0
		self.cur_epoch = 0
		self.toy_dataset = toy_dataset
		self.centers = centers
		self.cov = cov

		if checkpoint_epoch is not None:
			self.load_checkpoint(checkpoint_epoch)
		else:
			self.fixed_noise = torch.randn(10000, 2).view(-1, 2)

	def train(self, n_epochs=1, save_every=1):

		try:
			best_fd = np.min(self.history['FD'])
		except ValueError:
			best_fd = np.inf

		while self.cur_epoch < n_epochs:
			print('Epoch {}/{}'.format(self.cur_epoch + 1, n_epochs))
			# self.scheduler.step()
			train_iter = tqdm(enumerate(self.train_loader))
			gen_loss = 0.0
			disc_loss = 0.0
			for t, batch in train_iter:
				new_gen_loss, new_disc_loss = self.train_step(batch)
				gen_loss += new_gen_loss
				disc_loss += new_disc_loss
				self.total_iters += 1
				self.history['gen_loss_minibatch'].append(new_gen_loss)
				self.history['disc_loss_minibatch'].append(new_disc_loss)

			fd_, q_samples_, q_modes_ = self.valid()

			self.history['gen_loss'].append(gen_loss / (t + 1))
			self.history['disc_loss'].append(disc_loss / (t + 1))
			self.history['FD'].append(fd_)
			self.history['quality_samples'].append(q_samples_)
			self.history['quality_modes'].append(q_modes_)

			self.cur_epoch += 1

			if self.history['FD'][-1] < best_fd:
				best_fd = self.history['FD'][-1]
				self.checkpointing()
			elif self.cur_epoch % save_every == 0:
				self.checkpointing()

		# saving final common
		print('Saving final model...')
		self.checkpointing()

	def train_step(self, batch):

		## Train each D

		x = batch
		x = x['data']
		z_ = torch.randn(x.size(0), 2).view(-1, 2)
		y_real_ = torch.ones(x.size(0))
		y_fake_ = torch.zeros(x.size(0))

		x = Variable(x)
		z_ = Variable(z_)
		y_real_ = Variable(y_real_)
		y_fake_ = Variable(y_fake_)

		out_d = self.model.forward(z_).detach()

		d_real = self.disc.forward(x).squeeze()
		d_fake = self.disc.forward(out_d).squeeze()
		loss_disc = F.binary_cross_entropy(d_real, y_real_) + F.binary_cross_entropy(d_fake, y_fake_)
		self.disc.optimizer.zero_grad()
		loss_disc.backward()
		self.disc.optimizer.step()

		## Train G

		self.model.train()

		z_ = torch.randn(x.size(0), 2).view(-1, 2)

		z_ = Variable(z_)
		out = self.model.forward(z_)

		loss_G = F.binary_cross_entropy(self.disc.forward(out).squeeze(), y_real_)

		self.optimizer.zero_grad()
		loss_G.backward()
		self.optimizer.step()

		return loss_G.data[0], loss_disc.data[0]

	def valid(self):

		self.model.eval()

		z_ = Variable(self.fixed_noise)

		x_gen = self.model.forward(z_).cpu().data.numpy()

		fd, q_samples, q_modes = self.metrics(x_gen, self.centers, self.cov)

		return fd, q_samples, q_modes

	def calculate_dist(self, x_, y_):

		dist_matrix = np.zeros([x_.shape[0], y_.shape[0]])

		for i in range(x_.shape[0]):
			for j in range(y_.shape[0]):
				dist_matrix[i, j] = np.sqrt((x_[i, 0] - y_[j, 0]) ** 2 + (x_[i, 1] - y_[j, 1]) ** 2)

		return dist_matrix

	def metrics(self, x, centers, cov, slack=3.0):

		if self.toy_dataset == '8gaussians':
			distances = self.calculate_dist(1.414 * x, self.centers)

		elif self.toy_dataset == '25gaussians':
			distances = self.calculate_dist(2.828 * x, self.centers)

		closest_center = np.argmin(distances, 1)

		n_gaussians = centers.shape[0]

		fd = 0
		quality_samples = 0
		quality_modes = 0
		fd_modes = 0

		for cent in range(n_gaussians):

			center_samples = x[np.where(closest_center == cent)]

			center_distances = distances[np.where(closest_center == cent)]

			sigma = cov[0, 0]

			quality_samples_center = np.sum(center_distances[:, cent] <= slack * np.sqrt(sigma))
			quality_samples += quality_samples_center

			if quality_samples_center > 0:
				quality_modes += 1

			if center_samples.shape[0] > 3:
				fd_modes += 1
				m = np.mean(center_samples, 0)
				C = np.cov(center_samples, rowvar=False)

				fd += ((centers[cent] - m) ** 2).sum() + np.matrix.trace(C + cov - 2 * sla.sqrtm(np.matmul(C, cov)))

		if (fd_modes > 0):
			fd_all = fd / fd_modes
		else:
			fd_all = float("inf")

		return fd_all, quality_samples, quality_modes

	def checkpointing(self):

		# Checkpointing
		print('Checkpointing...')
		ckpt = {'model_state': self.model.state_dict(),
				'optimizer_state': self.optimizer.state_dict(),
				'history': self.history,
				'total_iters': self.total_iters,
				'fixed_noise': self.fixed_noise,
				'cur_epoch': self.cur_epoch}
		torch.save(ckpt, self.save_epoch_fmt_gen.format(self.cur_epoch))

		ckpt = {'model_state': self.disc.state_dict(),
				'optimizer_state': self.disc.optimizer.state_dict()}
		torch.save(ckpt, self.save_epoch_fmt_disc.format(self.cur_epoch))

	def load_checkpoint(self, epoch):

		ckpt = self.save_epoch_fmt_gen.format(epoch)

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt)
			# Load model state
			self.model.load_state_dict(ckpt['model_state'])
			# Load optimizer state
			self.optimizer.load_state_dict(ckpt['optimizer_state'])
			# Load history
			self.history = ckpt['history']
			self.total_iters = ckpt['total_iters']
			self.cur_epoch = ckpt['cur_epoch']
			self.fixed_noise = ckpt['fixed_noise']

			ckpt = torch.load(self.save_epoch_fmt_disc.format(epoch))
			self.disc.load_state_dict(ckpt['model_state'])
			self.disc.optimizer.load_state_dict(ckpt['optimizer_state'])
		else:
			print('No checkpoint found at: {}'.format(ckpt))

	def print_grad_norms(self):
		norm = 0.0
		for params in list(self.model.parameters()):
			norm += params.grad.norm(2).data[0]
		print('Sum of grads norms: {}'.format(norm))

	def check_nans(self):
		for params in list(self.model.parameters()):
			if np.any(np.isnan(params.data.cpu().numpy())):
				print('params NANs!!!!!')
			if np.any(np.isnan(params.grad.data.cpu().numpy())):
				print('grads NANs!!!!!!')
