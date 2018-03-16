import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

import os
from tqdm import tqdm


class TrainLoop(object):

	def __init__(self, generator, disc_list, optimizer, train_loader, checkpoint_path=None, checkpoint_epoch=None, nadir_slack=None, cuda=True):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch_fmt_gen = os.path.join(self.checkpoint_path, 'checkpoint_{}ep.pt')
		self.save_epoch_fmt_disc = os.path.join(self.checkpoint_path, 'D_{}_checkpoint_{}ep.pt')
		self.cuda_mode = cuda
		self.model = generator
		self.disc_list = disc_list
		self.optimizer = optimizer
		self.train_loader = train_loader
		self.history = {'gen_loss': [], 'gen_loss_minibatch': [], 'disc_loss': [], 'disc_loss_minibatch': []}
		self.total_iters = 0
		self.cur_epoch = 0

		if checkpoint_epoch is not None:
			self.load_checkpoint(checkpoint_epoch)

			if nadir_slack:
				self.hyper_mode = True
				self.nadir_slack = nadir_slack
			else:
				self.hyper_mode = False
				self.nadir = 0.0

		else:
			if nadir_slack:
				#self.define_nadir_point(nadir_slack)
				self.nadir_slack = nadir_slack
				self.hyper_mode = True
			else:
				self.hyper_mode = False
				self.nadir = 0.0

	def train(self, n_epochs=1, save_every=1):

		while (self.cur_epoch < n_epochs):
			print('Epoch {}/{}'.format(self.cur_epoch+1, n_epochs))
			#self.scheduler.step()
			train_iter = tqdm(enumerate(self.train_loader))
			gen_loss=0.0
			disc_loss=0.0
			for t, batch in train_iter:
				new_gen_loss, new_disc_loss = self.train_step(batch)
				gen_loss+=new_gen_loss
				disc_loss+=new_disc_loss
				self.total_iters += 1
				self.history['gen_loss_minibatch'].append(new_gen_loss)
				self.history['disc_loss_minibatch'].append(new_disc_loss)

			self.history['gen_loss'].append(gen_loss/(t+1))
			self.history['disc_loss'].append(disc_loss/(t+1))

			self.cur_epoch += 1

			if self.cur_epoch % save_every == 0:
				self.checkpointing()

		# saving final models
		print('Saving final model...')
		self.checkpointing()

	def train_step(self, batch):

		## Train each D

		x, _ = batch
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

		z_ = torch.randn(x.size(0), 100).view(-1, 100, 1, 1)

		if self.cuda_mode:
			z_ = z_.cuda()

		z_ = Variable(z_)
		out = self.model.forward(z_)

		loss_G = 0

		if self.hyper_mode:

			losses_list_float = []
			losses_list_var = []

			for disc in self.disc_list:
				losses_list_var.append( F.binary_cross_entropy( disc.forward(out).squeeze(), y_real_) )
				losses_list_float.append( losses_list_var[-1].data[0] )

			self.update_nadir_point(losses_list_float)

			for loss in self.losses_list_var:
				loss_G -= torch.log( self.nadir - loss )

		else:
			for disc in self.disc_list:
				loss_G += F.binary_cross_entropy(disc.forward(out).squeeze(), y_real_)

		self.optimizer.zero_grad()
		loss_G.backward()
		self.optimizer.step()

		return loss_G.data[0] / len(self.disc_list), loss_d

	def checkpointing(self):

		# Checkpointing
		print('Checkpointing...')
		ckpt = {'model_state': self.model.state_dict(),
		'optimizer_state': self.optimizer.state_dict(),
		'history': self.history,
		'total_iters': self.total_iters,
		'nadir_point': self.nadir,
		'cur_epoch': self.cur_epoch}
		torch.save(ckpt, self.save_epoch_fmt_gen.format(self.cur_epoch))

		for i, disc in enumerate(self.disc_list):
			ckpt = {'model_state': disc.state_dict(),
			'optimizer_state': disc.optimizer.state_dict()}
			torch.save(ckpt, self.save_epoch_fmt_disc.format(i+1, self.cur_epoch))

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
			self.nadir = ckpt['nadir']

			for i, disc in enumerate(self.disc_list):
				ckpt = torch.load(self.save_epoch_fmt_disc.format(i+1, epoch))
				disc.load_state_dict(ckpt['model_state'])
				disc.optimizer.load_state_dict(ckpt['optimizer_state'])

		else:
			print('No checkpoint found at: {}'.format(ckpt))

	def print_grad_norms(self):
		norm = 0.0
		for params in list(self.model.parameters()):
			norm+=params.grad.norm(2).data[0]
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
			disc_outs.append( F.binary_cross_entropy(d_out, y_real_).data[0] )

		self.nadir = float(np.max(disc_outs) + self.nadir_slack)

	def update_nadir_point(self, losses_list):
		self.nadir = float(np.max(losses_list) + self.nadir_slack)
