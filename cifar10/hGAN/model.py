import torch
import torch.nn as nn

class Generator(torch.nn.Module):
	def __init__(self, input_dim, num_filters, output_dim):
		super(Generator, self).__init__()

		# Hidden layers
		self.hidden_layer = torch.nn.Sequential()
		for i in range(len(num_filters)):
			# Deconvolutional layer
			if i == 0:
				deconv = nn.ConvTranspose2d(input_dim, num_filters[i], kernel_size=4, stride=1, padding=0)
			else:
				deconv = nn.ConvTranspose2d(num_filters[i - 1], num_filters[i], kernel_size=4, stride=2, padding=1)

			deconv_name = 'deconv' + str(i + 1)
			self.hidden_layer.add_module(deconv_name, deconv)

			# Initializer
			nn.init.normal(deconv.weight, mean=0.0, std=0.02)
			nn.init.constant(deconv.bias, 0.0)

			# Batch normalization
			bn_name = 'bn' + str(i + 1)
			self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

			# Activation
			act_name = 'act' + str(i + 1)
			self.hidden_layer.add_module(act_name, torch.nn.ReLU())

		# Output layer
		self.output_layer = torch.nn.Sequential()
		# Deconvolutional layer
		out = torch.nn.ConvTranspose2d(num_filters[i], output_dim, kernel_size=4, stride=2, padding=1)
		self.output_layer.add_module('out', out)
		# Initializer
		nn.init.normal(out.weight, mean=0.0, std=0.02)
		nn.init.constant(out.bias, 0.0)
		# Activation
		self.output_layer.add_module('act', torch.nn.Tanh())

	def forward(self, x):
		h = self.hidden_layer(x)
		out = self.output_layer(h)
		return out

# Discriminator model
class Discriminator(torch.nn.Module):
	def __init__(self, input_dim, num_filters, output_dim, optimizer, lr, betas, batch_norm=False):
		super(Discriminator, self).__init__()

		self.projection = nn.utils.weight_norm(nn.Conv2d(input_dim, 1, kernel_size=8, stride=2, padding=3, bias=False), name="weight")
		self.projection.weight_g.data.fill_(1)

		# Hidden layers
		self.hidden_layer = torch.nn.Sequential()
		for i in range(len(num_filters)):
			# Convolutional layer
			if i == 0:
				conv = nn.Conv2d(1, num_filters[i], kernel_size=4, stride=2, padding=1)
			else:
				conv = nn.Conv2d(num_filters[i - 1], num_filters[i], kernel_size=4, stride=2, padding=1)

			conv_name = 'conv' + str(i + 1)
			self.hidden_layer.add_module(conv_name, conv)

			# Initializer
			nn.init.normal(conv.weight, mean=0.0, std=0.02)
			nn.init.constant(conv.bias, 0.0)

			# Batch normalization
			if i != 0 and batch_norm:
				bn_name = 'bn' + str(i + 1)
				self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

			# Activation
			act_name = 'act' + str(i + 1)
			self.hidden_layer.add_module(act_name, torch.nn.LeakyReLU(0.2))

		# Output layer
		self.output_layer = torch.nn.Sequential()
		# Convolutional layer
		out = nn.Conv2d(num_filters[i], output_dim, kernel_size=4, stride=1, padding=1)
		self.output_layer.add_module('out', out)
		# Initializer
		nn.init.normal(out.weight, mean=0.0, std=0.02)
		nn.init.constant(out.bias, 0.0)
		# Activation
		self.output_layer.add_module('act', nn.Sigmoid())

		self.optimizer = optimizer(list(self.hidden_layer.parameters()) + list(self.output_layer.parameters()), lr=lr, betas=betas)

	def forward(self, x):
		p_x = self.projection(x)
		h = self.hidden_layer(p_x)
		out = self.output_layer(h)
		return out


## vanilla discriminator with kernel size=4
class Discriminator_vanilla(nn.Module):
	def __init__(self, ndf, nc, optimizer, lr, betas):
		super(Discriminator_vanilla, self).__init__()
		self.main = nn.Sequential(
			# input is (nc) x 64 x 64
			nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 8 x 8
			nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*8) x 4 x 4
			nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
			nn.Sigmoid())

		self.optimizer = optimizer(self.parameters(), lr=lr, betas=betas)

	def forward(self, x):
		return self.main(x)


## discriminator with kernel size = 8
class Discriminator_f8(nn.Module):
	def __init__(self, ndf, nc, optimizer, lr, betas):
		super(Discriminator_f8, self).__init__()
		self.main = nn.Sequential(

			# input is (nc) x 64 x 64
			nn.Conv2d(nc, ndf, 8, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 30 x 30

			nn.Conv2d(ndf, ndf * 2, 8, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 13 x 13

			nn.Conv2d(ndf * 2, ndf * 4, 8, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),

			nn.LeakyReLU(0.2, inplace=True),

			# state size. (ndf*4) x 4 x 4

			nn.Conv2d(ndf * 4, 1, 4, 2, 0, bias=False),
			nn.Sigmoid())

		self.optimizer = optimizer(self.parameters(), lr=lr, betas=betas)

	def forward(self, x):
		return self.main(x)


## discriminator with kernel size = 16
class Discriminator_f16(nn.Module):
	def __init__(self, ndf, nc, optimizer, lr, betas):
		super(Discriminator_f16, self).__init__()
		self.main = nn.Sequential(

			# input is (nc) x 64 x 64
			nn.Conv2d(nc, ndf, 16, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 26 x 26

			nn.Conv2d(ndf, ndf * 2, 16, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 7 x 7

			nn.Conv2d(ndf * 2, 1, 7, 2, 0, bias=False),
			nn.Sigmoid())

		self.optimizer = optimizer(self.parameters(), lr=lr, betas=betas)

	def forward(self, x):
		return self.main(x)


## discrminator with 1 layer of kernel size=4, remaining part= dense
class Discriminator_dense(nn.Module):
	def __init__(self, ndf, nc, optimizer, lr, betas):
		super(Discriminator_dense, self).__init__()
		self.main = nn.Sequential(
			# input is (nc) x 64 x 64
			nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True))
		# state size. (ndf) x 32 x 32 )
		self.linear = nn.Sequential(nn.Linear(ndf * 32 * 32, 1), nn.Sigmoid())

		self.optimizer = optimizer(self.parameters(), lr=lr, betas=betas)

	def forward(self, x):
		x = self.main(x)
		return self.linear(x.view(x.size(0), -1))
