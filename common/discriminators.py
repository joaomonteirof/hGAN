import torch
import torch.nn as nn
import torch.nn.functional as F
from common.utils import ResnetBlock
from common.spectralnorm import SpectralNorm

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
			nn.init.normal_(conv.weight, mean=0.0, std=0.02)
			nn.init.constant_(conv.bias, 0.0)

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
		nn.init.normal_(out.weight, mean=0.0, std=0.02)
		nn.init.constant_(out.bias, 0.0)
		# Activation
		self.output_layer.add_module('act', nn.Sigmoid())

		self.optimizer = optimizer(list(self.hidden_layer.parameters()) + list(self.output_layer.parameters()), lr=lr, betas=betas)

	def forward(self, x):

		x = self.projection(x)
		h = self.hidden_layer(x)
		out = self.output_layer(h)
		return out

class Discriminator_noproj(torch.nn.Module):
	def __init__(self, input_dim, num_filters, output_dim, optimizer, lr, betas, batch_norm=False):
		super(Discriminator_noproj, self).__init__()

		# Hidden layers
		self.hidden_layer = torch.nn.Sequential()
		for i in range(len(num_filters)):
			# Convolutional layer
			if i == 0:
				conv = nn.Conv2d(input_dim, num_filters[i], kernel_size=4, stride=2, padding=1)
			else:
				conv = nn.Conv2d(num_filters[i-1], num_filters[i], kernel_size=4, stride=2, padding=1)

			conv_name = 'conv' + str(i + 1)
			self.hidden_layer.add_module(conv_name, conv)

			# Initializer
			nn.init.normal_(conv.weight, mean=0.0, std=0.02)
			nn.init.constant_(conv.bias, 0.0)

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
		out = nn.Conv2d(num_filters[i], output_dim, kernel_size=4, stride=1, padding=0)
		self.output_layer.add_module('out', out)
		# Initializer
		nn.init.normal_(out.weight, mean=0.0, std=0.02)
		nn.init.constant_(out.bias, 0.0)
		# Activation
		self.output_layer.add_module('act', nn.Sigmoid())

		self.optimizer = optimizer(list(self.hidden_layer.parameters()) + list(self.output_layer.parameters()), lr=lr, betas=betas)

	def forward(self, x):
		h = self.hidden_layer(x)
		out = self.output_layer(h)
		return out

class Discriminator_SN_noproj(nn.Module):
	def __init__(self, optimizer, lr, betas, optimizer_name='adam'):
		super().__init__()

		m_g = 4
		ch = 512

		self.layer1 = self.make_layer(3, ch//8)
		self.layer2 = self.make_layer(ch//8, ch//4)
		self.layer3 = self.make_layer(ch//4, ch//2)
		self.layer4 = SpectralNorm(nn.Conv2d(ch//2, ch, 3, 1, 1) )
		self.linear = SpectralNorm(nn.Linear(ch*m_g*m_g, 1) )

		self.optimizer = optimizer(self.parameters(), lr=lr, betas=betas)

		if optimizer_name == 'adam':
			self.optimizer = optimizer(list(self.layer1.parameters()) + list(self.layer2.parameters()) + list(self.layer3.parameters()) + list(self.layer4.parameters()) + list(self.linear.parameters()), lr=lr, betas=betas)
		elif optimizer_name == 'amsgrad':
			self.optimizer = optimizer(list(self.layer1.parameters()) + list(self.layer2.parameters()) + list(self.layer3.parameters()) + list(self.layer4.parameters()) + list(self.linear.parameters()), lr=lr, betas=betas, amsgrad = True)
		elif optimizer_name == 'rmsprop':
			self.optimizer = optimizer(list(self.layer1.parameters()) + list(self.layer2.parameters()) + list(self.layer3.parameters()) + list(self.layer4.parameters()) + list(self.linear.parameters()), lr=lr, alpha = betas[0])

	def make_layer(self, in_plane, out_plane):
		return nn.Sequential( SpectralNorm( nn.Conv2d(in_plane, out_plane, 3, 1, 1) ),
			nn.LeakyReLU(0.1),
			SpectralNorm(nn.Conv2d(out_plane, out_plane, 4, 2, 1) ),
			nn.LeakyReLU(0.1) )

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = out.view(out.size(0), -1)
		out = self.linear(out)

		return torch.sigmoid(out.squeeze())

class Discriminator_SN(nn.Module):
	def __init__(self, optimizer, lr, betas, optimizer_name='adam'):
		super().__init__()

		m_g = 4
		ch = 512

		self.projection = nn.utils.weight_norm(nn.Conv2d(3, 1, kernel_size=4, stride=1, padding=2, bias=False), name="weight")
		self.projection.weight_g.data.fill_(1)

		self.layer1 = self.make_layer(1, ch//8)
		self.layer2 = self.make_layer(ch//8, ch//4)
		self.layer3 = self.make_layer(ch//4, ch//2)
		self.layer4 = SpectralNorm(nn.Conv2d(ch//2, ch, 3, 1, 1) )
		self.linear = SpectralNorm(nn.Linear(ch*m_g*m_g, 1, 1) )

		if optimizer_name == 'adam':
			self.optimizer = optimizer(list(self.layer1.parameters()) + list(self.layer2.parameters()) + list(self.layer3.parameters()) + list(self.layer4.parameters()) + list(self.linear.parameters()), lr=lr, betas=betas)
		elif optimizer_name == 'amsgrad':
			self.optimizer = optimizer(list(self.layer1.parameters()) + list(self.layer2.parameters()) + list(self.layer3.parameters()) + list(self.layer4.parameters()) + list(self.linear.parameters()), lr=lr, betas=betas, amsgrad = True)
		elif optimizer_name == 'rmsprop':
			self.optimizer = optimizer(list(self.layer1.parameters()) + list(self.layer2.parameters()) + list(self.layer3.parameters()) + list(self.layer4.parameters()) + list(self.linear.parameters()), lr=lr, alpha = betas[0])


	def make_layer(self, in_plane, out_plane):
		return nn.Sequential( SpectralNorm( nn.Conv2d(in_plane, out_plane, 3, 1, 1) ),
			nn.LeakyReLU(0.1),
			SpectralNorm(nn.Conv2d(out_plane, out_plane, 4, 2, 1) ),
			nn.LeakyReLU(0.1) )

	def forward(self, x):

		p_x = self.projection(x)
		out = self.layer1(p_x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = out.view(out.size(0), -1)
		out = self.linear(out)

		return torch.sigmoid(out.squeeze())

class Discriminator_cifar32(nn.Module):
	def __init__(self, optimizer, optimizer_name, lr, betas):
		super().__init__()

		m_g = 4
		ch = 512

		self.projection = nn.utils.weight_norm(nn.Conv2d(3, 1, kernel_size=4, stride=1, padding=2, bias=False), name="weight")
		self.projection.weight_g.data.fill_(1)

		self.layer1 = self.make_layer(1, ch//8)
		self.layer2 = self.make_layer(ch//8, ch//4)
		self.layer3 = self.make_layer(ch//4, ch//2)
		self.layer4 = nn.Sequential( nn.Conv2d(ch//2, ch, 3, 1, 1), nn.LeakyReLU(0.2) )
		self.linear = nn.Linear(ch*m_g*m_g, 1, 1)

		if optimizer_name == 'adam':
			self.optimizer = optimizer(list(self.layer1.parameters()) + list(self.layer2.parameters()) + list(self.layer3.parameters()) + list(self.layer4.parameters()) + list(self.linear.parameters()), lr=lr, betas=betas)
		elif optimizer_name == 'amsgrad':
			self.optimizer = optimizer(list(self.layer1.parameters()) + list(self.layer2.parameters()) + list(self.layer3.parameters()) + list(self.layer4.parameters()) + list(self.linear.parameters()), lr=lr, betas=betas, amsgrad = True)
		elif optimizer_name == 'rmsprop':
			self.optimizer = optimizer(list(self.layer1.parameters()) + list(self.layer2.parameters()) + list(self.layer3.parameters()) + list(self.layer4.parameters()) + list(self.linear.parameters()), lr=lr, alpha = betas[0])


	def make_layer(self, in_plane, out_plane):
		return nn.Sequential( nn.Conv2d(in_plane, out_plane, 3, 1, 1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(out_plane, out_plane, 4, 2, 1),
			nn.LeakyReLU(0.2) )

	def forward(self, x):

		p_x = self.projection(x)
		out = self.layer1(p_x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = out.view(out.size(0), -1)
		out = self.linear(out)

		return torch.sigmoid(out.squeeze())

class Discriminator_stacked_mnist(torch.nn.Module):
	def __init__(self, optimizer, optimizer_name, lr, betas, batch_norm=False):
		super(Discriminator_stacked_mnist, self).__init__()

		self.projection = nn.utils.weight_norm(nn.Conv2d(3, 3, kernel_size=8, stride=2, padding=3, bias=False), name="weight")
		self.projection.weight_g.data.fill_(1)

		# Hidden layers
		self.hidden_layer = torch.nn.Sequential()
		num_filters = [64, 128, 256]
		for i in range(3):
			# Convolutional layer
			if i == 0:
				conv = nn.Conv2d(3, num_filters[i], kernel_size=4, stride=2, padding=2)
			elif i == 2:
				conv = nn.Conv2d(num_filters[i - 1], num_filters[i], kernel_size=4, stride=2, padding=1)
			else:
				conv = nn.Conv2d(num_filters[i - 1], num_filters[i], kernel_size=4, stride=2, padding=2)

			conv_name = 'conv' + str(i + 1)
			self.hidden_layer.add_module(conv_name, conv)

			# Initializer
			nn.init.normal_(conv.weight, mean=0.0, std=0.02)
			nn.init.constant_(conv.bias, 0.0)

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
		out = nn.Conv2d(num_filters[i], 1, kernel_size=4, stride=1, padding=1)
		self.output_layer.add_module('out', out)
		# Initializer
		nn.init.normal_(out.weight, mean=0.0, std=0.02)
		nn.init.constant_(out.bias, 0.0)
		# Activation
		self.output_layer.add_module('act', nn.Sigmoid())

		if optimizer_name == 'adam':
			self.optimizer = optimizer(list(self.hidden_layer.parameters()) + list(self.output_layer.parameters()), lr=lr, betas=betas)
		elif optimizer_name == 'amsgrad':
			self.optimizer = optimizer(list(self.hidden_layer.parameters()) + list(self.output_layer.parameters()), lr=lr, betas=betas, amsgrad = True)
		elif optimizer_name == 'rmsprop':
			self.optimizer = optimizer(list(self.hidden_layer.parameters()) + list(self.output_layer.parameters()), lr=lr, alpha = betas[0])

	def forward(self, x):
		p_x = self.projection(x)
		h = self.hidden_layer(p_x)
		out = self.output_layer(h)
		return out

class Discriminator_wgan_noproj(torch.nn.Module):
	def __init__(self, input_dim, num_filters, output_dim, optimizer, lr, betas, batch_norm=False):
		super(Discriminator_wgan_noproj, self).__init__()

		# Hidden layers
		self.hidden_layer = torch.nn.Sequential()
		for i in range(len(num_filters)):
			# Convolutional layer
			if i == 0:
				conv = nn.Conv2d(input_dim, num_filters[i], kernel_size=4, stride=2, padding=1)
			else:
				conv = nn.Conv2d(num_filters[i-1], num_filters[i], kernel_size=4, stride=2, padding=1)

			conv_name = 'conv' + str(i + 1)
			self.hidden_layer.add_module(conv_name, conv)

			# Initializer
			nn.init.normal_(conv.weight, mean=0.0, std=0.02)
			nn.init.constant_(conv.bias, 0.0)

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
		out = nn.Conv2d(num_filters[i], output_dim, kernel_size=4, stride=1, padding=0)
		self.output_layer.add_module('out', out)
		# Initializer
		nn.init.normal_(out.weight, mean=0.0, std=0.02)
		nn.init.constant_(out.bias, 0.0)

		self.optimizer = optimizer(list(self.hidden_layer.parameters()) + list(self.output_layer.parameters()), lr=lr, betas=betas)

	def forward(self, x):
		h = self.hidden_layer(x)
		out = self.output_layer(h)
		return out

class Discriminator_wgan(torch.nn.Module):
	def __init__(self, input_dim, num_filters, output_dim, optimizer, lr, betas, batch_norm=False):
		super(Discriminator_wgan, self).__init__()

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
			nn.init.normal_(conv.weight, mean=0.0, std=0.02)
			nn.init.constant_(conv.bias, 0.0)

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
		nn.init.normal_(out.weight, mean=0.0, std=0.02)
		nn.init.constant_(out.bias, 0.0)

		self.optimizer = optimizer(list(self.hidden_layer.parameters()) + list(self.output_layer.parameters()), lr=lr, betas=betas)

	def forward(self, x):

		x = self.projection(x)
		h = self.hidden_layer(x)
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


class Discriminator_f6(nn.Module):
	def __init__(self, ndf, nc, optimizer, lr, betas):
		super(Discriminator_f6, self).__init__()
		self.main = nn.Sequential(
			# input is (nc) x 64 x 64
			nn.Conv2d(nc, ndf, 6, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 31 x 31

			nn.Conv2d(ndf, ndf * 2, 6, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 14 x 14

			nn.Conv2d(ndf * 2, ndf * 4, 6, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),

			## state size. (ndf*4) x 4 x 4
			nn.Conv2d(ndf * 4, 1, 6, 2, 0, bias=False),
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


## discriminator with kernel size = 4 and stride = 3
class Discriminator_f4s3(nn.Module):
	def __init__(self, ndf, nc, optimizer, lr, betas):
		super(Discriminator_f4s3, self).__init__()
		self.main = nn.Sequential(
			# input is (nc) x 64 x 64
			nn.Conv2d(nc, ndf, 4, 3, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 21 x 21
			nn.Conv2d(ndf, ndf * 2, 4, 3, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 7 x 7
			nn.Conv2d(ndf * 2, ndf * 4, 4, 3, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 2 x 2
			nn.Conv2d(ndf * 4, 1, 4, 3, 1, bias=False),
			nn.Sigmoid())

		self.optimizer = optimizer(self.parameters(), lr=lr, betas=betas)

	def forward(self, x):
		return self.main(x)


class Discriminator_f4_dense(nn.Module):
	def __init__(self, ndf, nc, optimizer, lr, betas):
		super(Discriminator_f4_dense, self).__init__()
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
			nn.LeakyReLU(0.2, inplace=True))

		self.linear = nn.Sequential(
			nn.Linear(self.ndf * 8 * 4 * 4, 1),
			nn.Sigmoid())

		self.optimizer = optimizer(self.parameters(), lr=lr, betas=betas)

	def forward(self, x):
		output = self.main(x).view(x.size(0), -1)
		output = self.linear(output)

		return output.view(-1, 1).squeeze(1)


class Discriminator_f6_dense(nn.Module):
	def __init__(self, ndf, nc, optimizer, lr, betas):
		super(Discriminator_f6_dense, self).__init__()

		self.main = nn.Sequential(
			# input is (nc) x 64 x 64
			nn.Conv2d(nc, ndf, 6, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 31 x 31

			nn.Conv2d(ndf, ndf * 2, 6, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 14 x 14

			nn.Conv2d(ndf * 2, ndf * 4, 6, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True))

		self.linear = nn.Sequential(
			nn.Linear(self.ndf * 4 * 6 * 6, 1),
			nn.Sigmoid())

	def forward(self, x):
		output = self.main(input).view(x.size(0), -1)
		output = self.linear(output)

		return output.view(-1, 1).squeeze(1)


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


class Discriminator_toy(torch.nn.Module):
	def __init__(self, hidden_dim, optimizer, lr, betas):
		super(Discriminator_toy, self).__init__()

		self.projection = nn.utils.weight_norm(nn.Linear(2, 2, bias=False), name="weight")
		self.projection.weight_g.data.fill_(1)

		self.all_layers = nn.Sequential(
			nn.Linear(2, hidden_dim),
			nn.ReLU(True),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(True),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(True),
			nn.Linear(hidden_dim, 1),
			nn.Sigmoid()
		)

		self.optimizer = optimizer(list(self.all_layers.parameters()), lr=lr, betas=betas)

	def forward(self, x):
		# p_x = self.projection(x)
		out = self.all_layers(x)
		return out


class Discriminator_toy_wgan(torch.nn.Module):
	def __init__(self, hidden_dim, optimizer, lr, betas):
		super(Discriminator_toy_wgan, self).__init__()

		self.projection = nn.utils.weight_norm(nn.Linear(2, 2, bias=False), name="weight")
		self.projection.weight_g.data.fill_(1)

		self.all_layers = nn.Sequential(
			nn.Linear(2, hidden_dim),
			nn.ReLU(True),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(True),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(True),
			nn.Linear(hidden_dim, 1)
		)

		self.optimizer = optimizer(list(self.all_layers.parameters()), lr=lr, betas=betas)

	def forward(self, x):
		# p_x = self.projection(x)
		out = self.all_layers(x)
		return out

class Discriminator_mnist(nn.Module):
	def __init__(self, optimizer, lr, betas):
		super().__init__()

		self.projection = nn.utils.weight_norm(nn.Linear(784, 512), name="weight")
		self.projection.weight_g.data.fill_(1)

		self.hidden_layer = nn.Sequential(
			nn.Linear(512, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),
			nn.Linear(1024, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),
			nn.Linear(256, 1),
			nn.Sigmoid()
		)

		self.optimizer = optimizer(self.hidden_layer.parameters(), lr=lr, betas=betas)

	def forward(self, x):
		p_x = self.projection(x.view(x.size(0), 784))
		out = self.hidden_layer(p_x)
		return out

class Discriminator_mnist_noproj(nn.Module):
	def __init__(self, optimizer, lr, betas):
		super().__init__()

		self.hidden_layer = nn.Sequential(
			nn.Linear(784, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),
			nn.Linear(1024, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),
			nn.Linear(256, 1),
			nn.Sigmoid()
		)

		self.optimizer = optimizer(self.hidden_layer.parameters(), lr=lr, betas=betas)

	def forward(self, x):
		out = self.hidden_layer(x.view(x.size(0), 784))
		return out

class Discriminator_mnist_wgan(nn.Module):
	def __init__(self, optimizer, lr, betas):
		super().__init__()

		self.hidden_layer = nn.Sequential(
			nn.Linear(784, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),
			nn.Linear(1024, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),
			nn.Linear(256, 1)
		)

		self.optimizer = optimizer(self.hidden_layer.parameters(), lr=lr, betas=betas)

	def forward(self, x):
		out = self.hidden_layer(x.view(x.size(0), 784))
		return out

class Discriminator_res(nn.Module):
	def __init__(self, optimizer, lr, alpha, z_dim=128, size=256, nfilter=64):
		super().__init__()

		self.projection = nn.utils.weight_norm(nn.Conv2d(3, 1, kernel_size=8, stride=2, padding=3, bias=False), name="weight")
		self.projection.weight_g.data.fill_(1)

		s0 = self.s0 = size // 32
		nf = self.nf = nfilter

		# Submodules
		self.conv_img = nn.Conv2d(1, 1*nf, 3, padding=1)

		self.resnet_0_0 = ResnetBlock(1*nf, 1*nf)
		self.resnet_0_1 = ResnetBlock(1*nf, 2*nf)

		self.resnet_1_0 = ResnetBlock(2*nf, 2*nf)
		self.resnet_1_1 = ResnetBlock(2*nf, 4*nf)

		self.resnet_2_0 = ResnetBlock(4*nf, 4*nf)
		self.resnet_2_1 = ResnetBlock(4*nf, 8*nf)

		self.resnet_3_0 = ResnetBlock(8*nf, 8*nf)
		self.resnet_3_1 = ResnetBlock(8*nf, 16*nf)

		self.resnet_4_0 = ResnetBlock(16*nf, 16*nf)
		self.resnet_4_1 = ResnetBlock(16*nf, 16*nf)

		self.resnet_5_0 = ResnetBlock(16*nf, 16*nf)
		self.resnet_5_1 = ResnetBlock(16*nf, 16*nf)

		self.fc = nn.Linear(16*nf*s0*s0, 1)

		self.optimizer = optimizer(list(self.conv_img.parameters()) + list(self.resnet_0_0.parameters()) + list(self.resnet_0_1.parameters()) + list(self.resnet_1_0.parameters()) + list(self.resnet_1_1.parameters()) + list(self.resnet_2_0.parameters()) + list(self.resnet_2_1.parameters()) + list(self.resnet_3_0.parameters()) + list(self.resnet_3_1.parameters()) + list(self.resnet_4_0.parameters()) + list(self.resnet_4_1.parameters()) + list(self.resnet_5_0.parameters()) + list(self.resnet_5_1.parameters()) + list(self.fc.parameters()), lr=lr, alpha=alpha)


	def forward(self, x):
		batch_size = x.size(0)

		p_x = self.projection(x)

		out = self.conv_img(p_x)

		out = self.resnet_0_0(out)
		out = self.resnet_0_1(out)

		out = F.avg_pool2d(out, 3, stride=2, padding=1)
		out = self.resnet_1_0(out)
		out = self.resnet_1_1(out)

		out = F.avg_pool2d(out, 3, stride=2, padding=1)
		out = self.resnet_2_0(out)
		out = self.resnet_2_1(out)

		out = F.avg_pool2d(out, 3, stride=2, padding=1)
		out = self.resnet_3_0(out)
		out = self.resnet_3_1(out)

		out = F.avg_pool2d(out, 3, stride=2, padding=1)
		out = self.resnet_4_0(out)
		out = self.resnet_4_1(out)

		#out = F.avg_pool2d(out, 3, stride=2, padding=1)
		out = self.resnet_5_0(out)
		out = self.resnet_5_1(out)

		out = out.view(batch_size, 16*self.nf*self.s0*self.s0)
		out = self.fc(F.leaky_relu(out, 2e-1))

		return nn.Sigmoid()(out)
