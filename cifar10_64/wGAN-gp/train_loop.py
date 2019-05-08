import os
import pickle

import numpy as np
import scipy.linalg as sla
import torch
from tqdm import tqdm


class TrainLoop(object):

	def __init__(self, generator, fid_model, disc, optimizer, train_loader, lambda_grad, its_disc, checkpoint_path=None, checkpoint_epoch=None, cuda=True):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch_fmt_gen = os.path.join(self.checkpoint_path, 'G_WGANGP_{}ep.pt')
		self.save_epoch_fmt_disc = os.path.join(self.checkpoint_path, 'D_WGANGP_{}ep.pt')
		self.cuda_mode = cuda
		self.model = generator
		self.fid_model = fid_model
		self.disc = disc
		self.optimizer = optimizer
		self.train_loader = train_loader
		self.history = {'gen_loss': [], 'gen_loss_minibatch': [], 'disc_loss': [], 'disc_loss_minibatch': [], 'FID-c': []}
		self.total_iters = 0
		self.cur_epoch = 0
		self.lambda_grad = lambda_grad
		self.its_disc = its_disc

		pfile = open('../test_data_statistics.p', 'rb')
		statistics = pickle.load(pfile)
		pfile.close()

		self.m, self.C = statistics['m'], statistics['C']

		if checkpoint_epoch is not None:
			self.load_checkpoint(checkpoint_epoch)
		else:
			self.fixed_noise = torch.randn(1000, 100).view(-1, 100, 1, 1)

	def train(self, n_epochs=1, save_every=1):

		try:
			best_fid = np.min(self.history['FID-c'])
		except ValueError:
			best_fid = np.inf

		while self.cur_epoch < n_epochs:
			print('Epoch {}/{}'.format(self.cur_epoch + 1, n_epochs))
			# self.scheduler.step()
			train_iter = tqdm(enumerate(self.train_loader))
			gen_loss = 0.0
			disc_loss = 0.0
			for t, batch in train_iter:
				new_gen_loss, new_disc_loss = self.train_step(batch, t)
				gen_loss += new_gen_loss
				disc_loss += new_disc_loss
				self.total_iters += 1
				self.history['gen_loss_minibatch'].append(new_gen_loss)
				self.history['disc_loss_minibatch'].append(new_disc_loss)

			fid_c = self.valid()

			self.history['gen_loss'].append(gen_loss / (t + 1))
			self.history['disc_loss'].append(disc_loss / (t + 1))
			self.history['FID-c'].append(fid_c)

			print('Best FID so far is: {}'.format(np.min(self.history['FID-c'])))

			self.cur_epoch += 1

			if self.history['FID-c'][-1] < best_fid:
				best_fid = self.history['FID-c'][-1]
				self.checkpointing()
			elif self.cur_epoch % save_every == 0:
				self.checkpointing()

		# saving final common
		print('Saving final model...')
		self.checkpointing()

	def train_step(self, batch, curr_it):

		## Train each D

		x, _ = batch

		#for i in range(self.its_disc):

		z_ = torch.randn(x.size(0), 100).view(-1, 100, 1, 1)

		if self.cuda_mode:
			z_ = z_.cuda()
			x = x.cuda()

		out_d = self.model.forward(z_).detach()

		loss_d = 0

		self.disc.optimizer.zero_grad()
		d_real = self.disc.forward(x).squeeze().mean()
		d_fake = self.disc.forward(out_d).squeeze().mean()
		loss_disc = d_fake - d_real + self.calc_gradient_penalty(x, out_d)
		loss_disc.backward()
		self.disc.optimizer.step()

		## Train G
		loss_G = torch.zeros(1)
		if (curr_it % self.its_disc == 0):

			self.model.train()

			z_ = torch.randn(x.size(0), 100).view(-1, 100, 1, 1)

			if self.cuda_mode:
				z_ = z_.cuda()

			out = self.model.forward(z_)

			loss_G = -self.disc.forward(out).mean()

			self.optimizer.zero_grad()
			loss_G.backward()
			self.optimizer.step()

		return loss_G.item(), loss_disc.item()

	def valid(self):

		self.model.eval()

		if self.cuda_mode:
			z_ = self.fixed_noise.cuda()
		else:
			z_ = self.fixed_noise

		with torch.no_grad():

			x_gen = self.model.forward(z_)

			logits = self.fid_model.forward(x_gen.cpu()).detach().numpy()

		m = logits.mean(0)
		C = np.cov(logits, rowvar=False)

		fid = ((self.m - m) ** 2).sum() + np.matrix.trace(C + self.C - 2 * sla.sqrtm(np.matmul(C, self.C)))

		return fid

	def calc_gradient_penalty(self, real_data, fake_data):
		shape = [real_data.size(0)] + [1] * (real_data.dim() - 1)
		alpha = torch.rand(shape)

		if self.cuda_mode:
			alpha = alpha.cuda()

		interpolates = alpha * real_data + ((1 - alpha) * fake_data)

		interpolates.requires_grad_()

		disc_interpolates = self.disc.forward(interpolates)

		grad_outs = torch.ones(disc_interpolates.size())

		if self.cuda_mode:
			grad_outs = grad_outs.cuda()

		gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=grad_outs, create_graph=True)[0].view(interpolates.size(0), -1)

		gradient_penalty = ((gradients.norm(p=2, dim=1) - 1) ** 2).mean() * self.lambda_grad

		return gradient_penalty

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
			norm += params.grad.norm(2).item()
		print('Sum of grads norms: {}'.format(norm))

	def check_nans(self):
		for params in list(self.model.parameters()):
			if np.any(np.isnan(params.detach().cpu().numpy())):
				print('params NANs!!!!!')
			if np.any(np.isnan(params.grad.detach().cpu().numpy())):
				print('grads NANs!!!!!!')
