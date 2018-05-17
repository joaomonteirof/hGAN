import os
import pickle

import numpy as np
import scipy.linalg as sla
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from torch.autograd import Variable
from tqdm import tqdm

from common.MGD_utils import *


class TrainLoop(object):

	def __init__(self, generator, disc_list, optimizer, train_loader, alpha=0.8, nadir_slack=1.1, train_mode='vanilla', checkpoint_path=None, checkpoint_epoch=None, cuda=True, job_id=None):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		if job_id:
			self.save_epoch_fmt_gen = os.path.join(self.checkpoint_path, 'G_' + train_mode + '_' + str(len(disc_list)) + '_{}ep_' + job_id + '.pt')
			self.save_epoch_fmt_disc = os.path.join(self.checkpoint_path, 'D_{}_' + train_mode + '_' + job_id + '.pt')
		else:
			self.save_epoch_fmt_gen = os.path.join(self.checkpoint_path, 'G_' + train_mode + '_' + str(len(disc_list)) + '_{}ep.pt')
			self.save_epoch_fmt_disc = os.path.join(self.checkpoint_path, 'D_{}_' + train_mode + '_.pt')

		self.cuda_mode = cuda
		self.model = generator
		self.disc_list = disc_list
		self.optimizer = optimizer
		self.train_loader = train_loader
		self.history = {'gen_loss': [], 'gen_loss_minibatch': [], 'disc_loss': [], 'disc_loss_minibatch': [], 'steepest_dir_norm': []}
		self.total_iters = 0
		self.cur_epoch = 0
		self.alpha = alpha
		self.nadir_slack = nadir_slack
		self.train_mode = train_mode
		self.constraints = make_constraints(len(disc_list))
		self.proba = np.random.rand(len(disc_list))
		self.proba /= np.sum(self.proba)
		self.Q = np.zeros(len(self.disc_list))

		if checkpoint_epoch is not None:
			self.load_checkpoint(checkpoint_epoch)
		else:
			self.fixed_noise = torch.randn(1000, 100).view(-1, 100, 1, 1)

	def train(self, n_epochs=1, save_every=1):

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

			st_dir_norm = self.valid()

			self.history['gen_loss'].append(gen_loss / (t + 1))
			self.history['disc_loss'].append(disc_loss / (t + 1))
			self.history['steepest_dir_norm'].append(st_dir_norm)

			self.cur_epoch += 1

			if self.cur_epoch % save_every == 0:
				self.checkpointing()

		# saving final models
		print('Saving final model...')
		self.checkpointing()

	def train_step(self, batch):

		## Train each D

		x = batch
		z_ = torch.randn(x.size(0), 100).view(-1, 100, 1, 1)
		y_real_ = torch.ones(x.size(0))
		y_fake_ = torch.zeros(x.size(0))

		if self.cuda_mode:
			x = x.cuda()
			z_ = z_.cuda()
			y_real_ = y_real_.cuda()
			y_fake_ = y_fake_.cuda()

		x = Variable(x)
		z_ = Variable(z_)
		y_real_ = Variable(y_real_)
		y_fake_ = Variable(y_fake_)

		out_d = self.model.forward(z_).detach()

		loss_d = 0

		for disc in self.disc_list:
			d_real = disc.forward(x).squeeze()
			d_fake = disc.forward(out_d).squeeze()
			loss_disc = F.binary_cross_entropy(d_real, y_real_) + F.binary_cross_entropy(d_fake, y_fake_)
			disc.optimizer.zero_grad()
			loss_disc.backward()
			disc.optimizer.step()

			loss_d += loss_disc.data[0]

		loss_d /= len(self.disc_list)

		## Train G

		self.model.train()

		z_ = torch.randn(x.size(0), 100).view(-1, 100, 1, 1)

		if self.cuda_mode:
			z_ = z_.cuda()

		z_ = Variable(z_)
		out = self.model.forward(z_)

		loss_G = 0

		if self.train_mode == 'hyper':

			losses_list_float = []
			losses_list_var = []
			prob_list = []

			for disc in self.disc_list:
				losses_list_var.append(F.binary_cross_entropy(disc.forward(out).squeeze(), y_real_))
				losses_list_float.append(losses_list_var[-1].data[0])

			self.update_nadir_point(losses_list_float)

			coefs_sum = 0.0

			for i, loss in enumerate(losses_list_var):
				loss_G -= torch.log(self.nadir - loss)
				prob_list.append(1 / (self.nadir - losses_list_float[i]))
				coefs_sum += prob_list[-1]

			self.proba = np.asarray(prob_list) / coefs_sum

		elif self.train_mode == 'gman':

			losses_list_float = []
			losses_list_var = []

			for disc in self.disc_list:
				losses_list_var.append(F.binary_cross_entropy(disc.forward(out).squeeze(), y_real_))
				losses_list_float.append(losses_list_var[-1].data[0])

			losses = Variable(torch.FloatTensor(losses_list_float))
			self.proba = torch.nn.functional.softmax(self.alpha * losses, dim=0).data.cpu().numpy()

			acm = 0.0
			for loss_weight in zip(losses_list_var, self.proba):
				loss_G += loss_weight[0] * float(loss_weight[1])
			loss_G

		elif self.train_mode == 'gman_grad':

			grads_list = []
			losses_list = []

			for disc in self.disc_list:
				loss = F.binary_cross_entropy(disc.forward(self.model.forward(z_)).squeeze(), y_real_)
				grads_list.append(self.get_gen_grads_norm(loss))

			grads = Variable(torch.FloatTensor(grads_list))
			self.proba = torch.nn.functional.softmax(self.alpha * grads, dim=0).data.cpu().numpy()

			self.model.zero_grad()

			out = self.model.forward(z_)

			for disc in self.disc_list:
				losses_list.append(F.binary_cross_entropy(disc.forward(out).squeeze(), y_real_))

			for loss_weight in zip(losses_list, self.proba):
				loss_G += loss_weight[0] * float(loss_weight[1])

		elif self.train_mode == 'mgd':

			grads_list = []
			losses_list = []

			for disc in self.disc_list:
				loss = F.binary_cross_entropy(disc.forward(self.model.forward(z_)).squeeze(), y_real_)
				grads_list.append(self.get_gen_grads(loss).cpu().data.numpy())

			grads_list = np.asarray(grads_list).T

			# Steepest descent direction calc
			result = minimize(steep_direct_cost, self.proba, args=grads_list, jac=steep_direc_cost_deriv, constraints=self.constraints, method='SLSQP', options={'disp': False})

			self.proba = result.x

			self.model.zero_grad()

			out = self.model.forward(z_)

			for disc in self.disc_list:
				losses_list.append(F.binary_cross_entropy(disc.forward(out).squeeze(), y_real_))

			for loss_weight in zip(losses_list, self.proba):
				loss_G += loss_weight[0] * float(loss_weight[1])

		elif self.train_mode == 'loss_delta':

			z_probs = torch.randn(x.size(0), 100).view(-1, 100, 1, 1)

			if self.cuda_mode:
				z_probs = z_probs.cuda()

			z_probs = Variable(z_probs)

			out_probs = self.model.forward(z_probs)

			outs_before = []
			losses_list = []

			for i, disc in enumerate(self.disc_list):
				disc_out = disc.forward(out_probs).squeeze()
				losses_list.append(float(self.proba[i]) * F.binary_cross_entropy(disc_out, y_real_))
				outs_before.append(disc_out.data.mean())

			for loss_ in losses_list:
				loss_G += loss_

		elif self.train_mode == 'vanilla':
			for disc in self.disc_list:
				loss_G += F.binary_cross_entropy(disc.forward(out).squeeze(), y_real_)
			self.proba = np.ones(len(self.disc_list)) * 1 / len(self.disc_list)

		self.optimizer.zero_grad()
		loss_G.backward()
		self.optimizer.step()

		if self.train_mode == 'loss_delta':

			out_probs = self.model.forward(z_probs)

			outs_after = []

			for i, disc in enumerate(self.disc_list):
				disc_out = disc.forward(out_probs).squeeze()
				outs_after.append(disc_out.mean())

			self.update_prob(outs_before, outs_after)

		return loss_G.data[0], loss_d

	def valid(self):

		self.model.eval()
		steepest_dir_norm = self.compute_steepest_direction_norm()

		return steepest_dir_norm

	def checkpointing(self):

		# Checkpointing
		print('Checkpointing...')
		ckpt = {'model_state': self.model.state_dict(),
				'optimizer_state': self.optimizer.state_dict(),
				'history': self.history,
				'total_iters': self.total_iters,
				'fixed_noise': self.fixed_noise,
				'proba': self.proba,
				'Q': self.Q,
				'cur_epoch': self.cur_epoch}
		torch.save(ckpt, self.save_epoch_fmt_gen.format(self.cur_epoch))

		for i, disc in enumerate(self.disc_list):
			ckpt = {'model_state': disc.state_dict(),
					'optimizer_state': disc.optimizer.state_dict()}
			torch.save(ckpt, self.save_epoch_fmt_disc.format(i + 1))

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
			self.proba = ckpt['proba']
			self.Q = ckpt['Q']

			for i, disc in enumerate(self.disc_list):
				ckpt = torch.load(self.save_epoch_fmt_disc.format(i + 1))
				disc.load_state_dict(ckpt['model_state'])
				disc.optimizer.load_state_dict(ckpt['optimizer_state'])

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

	def define_nadir_point(self):
		disc_outs = []

		z_ = torch.randn(20, 100).view(-1, 100, 1, 1)
		y_real_ = torch.ones(z_.size(0))

		if self.cuda_mode:
			z_ = z_.cuda()
			y_real_ = y_real_.cuda()

		z_ = Variable(z_)
		y_real_ = Variable(y_real_)
		out = self.model.forward(z_)

		for disc in self.disc_list:
			d_out = disc.forward(out).squeeze()
			disc_outs.append(F.binary_cross_entropy(d_out, y_real_).data[0])

		self.nadir = float(np.max(disc_outs) + self.nadir_slack)

	def update_nadir_point(self, losses_list):
		self.nadir = float(np.max(losses_list) * self.nadir_slack + 1e-8)

	def update_prob(self, before, after):

		reward = [el[1] - el[0] for el in zip(before, after)]

		for i in range(len(self.Q)):
			self.Q[i] = self.alpha * reward[i] + (1 - self.alpha) * self.Q[i]

		self.proba = torch.nn.functional.softmax(15 * Variable(torch.FloatTensor(self.Q)), dim=0).data.cpu().numpy()

	def compute_steepest_direction_norm(self):
		self.model.train()

		z_ = torch.randn(128, 100).view(-1, 100, 1, 1)

		y_real_ = torch.ones(128)

		if self.cuda_mode:
			z_ = z_.cuda()
			y_real_ = y_real_.cuda()

		z_ = Variable(z_, requires_grad=False)
		y_real_ = Variable(y_real_)

		grads_list = []

		for disc in self.disc_list:
			self.model.zero_grad()
			out = self.model.forward(z_)
			loss = F.binary_cross_entropy(disc.forward(out).squeeze(), y_real_)
			grads_list.append(self.get_gen_grads(loss))

		grad_sum = 0.0

		for weight_grad in zip(self.proba, grads_list):
			try:
				grad_sum += float(weight_grad[0]) * weight_grad[1]
			except TypeError:
				grad_sum = float(weight_grad[0]) * weight_grad[1]

		return float(grad_sum.norm(2).data[0])

	def get_gen_grads(self, loss_):
		grads = torch.autograd.grad(outputs=loss_, inputs=self.model.parameters())
		self.model.zero_grad()
		for params_grads in grads:

			try:
				grads_ = torch.cat([grads_, params_grads.view(-1)], 0)
			except:
				grads_ = params_grads.view(-1)

		return grads_

	def get_gen_grads_norm(self, loss_):
		norm = 0.0
		self.model.zero_grad()
		grads = torch.autograd.grad(outputs=loss_, inputs=self.model.parameters())
		for params_grads in grads:
			norm += params_grads.norm(2).data[0] ** 2
		return np.sqrt(norm)
